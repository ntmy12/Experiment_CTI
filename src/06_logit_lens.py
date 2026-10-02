"""
Experiment 2: Logit Lens Analysis at m = 1 Across Transformer Layers.

Research Question:
    At position t-1 (one token before object mention), at which transformer
    layer does the divergence between hallucinated and real objects first emerge,
    and at which layer is it strongest?

Method:
    For each matched pair (halluc, real) from Experiment 1's labels.jsonl:
    - Run a single teacher-forcing forward pass with output_hidden_states=True.
    - At position t-1, extract hidden state h^(l) for every layer l in 1..L.
    - Project h^(l) through the model's final LayerNorm and lm_head to obtain
      pseudo-logits z^(l) (the "Logit Lens" projection).
    - Compute the normalized object favorability score S(o, layer=l) from z^(l).
    - Record S for both the hallucinated and real object in each pair.

Inputs (read-only, produced by Experiment 1):
    data/labels.jsonl
    data/synonyms.txt

Outputs (written to results/exp2_logitlens/):
    layer_records.csv    -- Per-pair, per-layer S scores
    layer_summary.csv    -- Mean S_halluc, S_real, Delta per layer with stats
    funnel.json          -- Pair counts and filtering info

Produces:
- results/exp2_logitlens/layer_records.csv
- results/exp2_logitlens/layer_summary.csv
- results/exp2_logitlens/funnel.json
- results/exp2_logitlens/figures/s_vs_layer.png
- results/exp2_logitlens/figures/delta_vs_layer.png
- results/exp2_logitlens/REPORT.md
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.common import (
    load_synonyms,
    match_object_pairs,
    paired_bootstrap_delta,
    cohens_dz,
    compute_wilcoxon_p,
    holm_bonferroni,
    bootstrap_auroc,
    resolve_image_path,
)

PROMPT = "USER: <image>\nPlease describe this image in detail. ASSISTANT:"


# ---------------------------------------------------------------------------
# Section 1: Candidate selection (identical filter logic to Experiment 1)
# ---------------------------------------------------------------------------

def select_candidates(
    labels_file: str,
    K: int = 10,
    caliper: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Reads labels.jsonl (Experiment 1 output) and builds 1:1 matched pairs.
    Applies the same filters as 03_lag_curve.py:
      - first mention only
      - tok_idx >= K
      - word_start_valid
      - caliper matching on rel_pos

    Returns:
        matched_h   -- list of matched hallucinated candidate dicts
        matched_r   -- list of matched real candidate dicts
        funnel_data -- dict with count statistics
    """
    h_candidates: List[Dict] = []
    r_candidates: List[Dict] = []

    total_captions = 0
    total_mentions = 0
    first_mentions = 0
    halluc_all = 0
    real_all = 0

    with open(labels_file, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            total_captions += 1
            image_id = record["image_id"]
            file_name = record["file_name"]
            gen_ids = record["gen_ids"]
            G = len(gen_ids)

            for m in record["mentions"]:
                total_mentions += 1
                if not m["first"]:
                    continue
                first_mentions += 1

                is_halluc = m["halluc"]
                if is_halluc:
                    halluc_all += 1
                else:
                    real_all += 1

                tok_idx = m["tok_idx"]
                if tok_idx < K:
                    continue
                if not m.get("word_start_valid", True):
                    continue

                cand = {
                    "image_id": image_id,
                    "file_name": file_name,
                    "canon": m["canon"],
                    "word": m["word"],
                    "t": tok_idx,
                    "tok_id": m["tok_id"],
                    "G": G,
                    "rel_pos": m["rel_pos"],
                    "halluc": is_halluc,
                    "gen_ids": gen_ids,
                }

                if is_halluc:
                    h_candidates.append(cand)
                else:
                    r_candidates.append(cand)

    matched_h, matched_r, dropped = match_object_pairs(
        h_candidates, r_candidates, caliper=caliper, seed=seed
    )

    funnel_data = {
        "captions": total_captions,
        "mentions": total_mentions,
        "first_mentions": first_mentions,
        "halluc_all": halluc_all,
        "real_all": real_all,
        "matched_pairs": len(matched_h),
        "unmatched_halluc_dropped": dropped,
        "caliper": caliper,
        "K": K,
        "seed": seed,
    }

    return matched_h, matched_r, funnel_data


# ---------------------------------------------------------------------------
# Section 2: Logit Lens score computation at a single sequence position
# ---------------------------------------------------------------------------

def compute_logit_lens_s(
    hidden_states: Tuple[torch.Tensor, ...],
    norm_module: torch.nn.Module,
    lm_head_module: torch.nn.Module,
    pos: int,
    o_tok_id: int,
    v_union_tensor: torch.Tensor,
    o_in_union_idx: int,
    n_layers: int,
) -> List[Dict[str, Any]]:
    """
    For every transformer layer l in 1..n_layers, project the hidden state at
    sequence position `pos` through LayerNorm + lm_head (Logit Lens), then
    compute the normalized object favorability score S.

    Args:
        hidden_states     -- tuple of (n_layers+1) tensors, each (1, L, D).
                             hidden_states[0] = embedding; hidden_states[l] = after layer l.
        norm_module       -- model.model.norm (final LayerNorm of the LLM).
        lm_head_module    -- model.lm_head (Linear D -> V).
        pos               -- position index in the full sequence (0-indexed).
        o_tok_id          -- vocabulary ID of object token o.
        v_union_tensor    -- LongTensor of vocabulary IDs for V_obj union {o}.
        o_in_union_idx    -- index of o within v_union_tensor.
        n_layers          -- number of transformer layers (e.g. 32).

    Returns:
        List of dicts, one per layer, with keys: layer, s, logp, rank, conf, ent.
    """
    results = []

    for l in range(1, n_layers + 1):
        # h^(l) at position pos: shape (D,)
        h_l = hidden_states[l][0, pos, :].float()

        # Apply final LayerNorm and project to vocabulary
        with torch.no_grad():
            h_normed = norm_module(h_l.unsqueeze(0)).squeeze(0)  # (D,)
            z_l = lm_head_module(h_normed)                        # (V,)

        log_probs = F.log_softmax(z_l, dim=-1)
        probs = log_probs.exp()

        # log P(o | z_l)
        logp = float(log_probs[o_tok_id].item())

        # rank of o
        rank = int((z_l > z_l[o_tok_id]).sum().item()) + 1

        # Normalized score S = log P(o) - logsumexp_{v in V_union} log P(v)
        sub_log_probs = log_probs[v_union_tensor]
        s_score = float(
            (sub_log_probs[o_in_union_idx] - torch.logsumexp(sub_log_probs, dim=-1)).item()
        )

        # Top-1 confidence and Shannon entropy
        conf = float(probs.max().item())
        ent = float(-torch.sum(probs * log_probs).item())

        results.append({
            "layer": l,
            "s": s_score,
            "logp": logp,
            "rank": rank,
            "conf": conf,
            "ent": ent,
        })

    return results


# ---------------------------------------------------------------------------
# Section 3: Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    # ------------------------------------------------------------------
    # 3.1 Load synonyms and build vocabulary set
    # ------------------------------------------------------------------
    print("Loading synonyms...")
    syn2canon, canon_list = load_synonyms(args.synonyms_file)
    vocab_words = set(syn2canon.keys())


    # ------------------------------------------------------------------
    # 3.2 Select matched pairs (reuse Experiment 1 labels)
    # ------------------------------------------------------------------
    print("Selecting matched pairs from labels.jsonl...")
    matched_h, matched_r, funnel_data = select_candidates(
        labels_file=args.labels_file,
        K=args.K,
        caliper=args.caliper,
        seed=args.seed,
    )

    with open(os.path.join(args.output_dir, "funnel.json"), "w", encoding="utf-8") as f:
        json.dump(funnel_data, f, indent=2)

    n_pairs = len(matched_h)
    print(f"Matched pairs: {n_pairs}")
    if n_pairs == 0:
        print("No matched pairs found. Exiting.")
        return

    # ------------------------------------------------------------------
    # 3.3 Load model and processor
    # ------------------------------------------------------------------
    print(f"Loading model '{args.model_name}' with output_hidden_states support...")
    from PIL import Image
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Number of transformer layers in the LLM backbone
    n_layers = model.language_model.config.num_hidden_layers
    print(f"Model loaded. Transformer layers: {n_layers}")

    # Extract LayerNorm and lm_head for Logit Lens projection
    # For LLaVA-1.5, the LLM backbone is model.language_model (LlamaModel)
    try:
        norm_module = model.language_model.model.norm
        lm_head_module = model.language_model.lm_head
    except AttributeError:
        # Fallback for alternative attribute paths
        norm_module = model.model.norm
        lm_head_module = model.lm_head

    norm_module.eval()
    lm_head_module.eval()

    # Move norm and lm_head to a fixed device for projection
    proj_device = next(lm_head_module.parameters()).device

    # ------------------------------------------------------------------
    # 3.4 Build vocabulary token ID set for S computation
    # ------------------------------------------------------------------
    print("Building COCO vocabulary token ID set...")
    tokenizer = processor.tokenizer
    v_obj_ids: Set[int] = set()
    for word in vocab_words:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if ids:
            v_obj_ids.add(ids[0])

    # ------------------------------------------------------------------
    # 3.5 Group matched pairs by image to minimize redundant forward passes
    # ------------------------------------------------------------------
    # Each image may contribute multiple matched pairs; we run one forward
    # pass per image and process all pairs from that image together.
    from collections import defaultdict
    image_to_objs: Dict[str, List[Dict]] = defaultdict(list)

    for h_obj, r_obj in zip(matched_h, matched_r):
        pair_id = h_obj["pair_id"]
        image_id_h = h_obj["image_id"]
        image_id_r = r_obj["image_id"]

        image_to_objs[image_id_h].append({
            "pair_id": pair_id,
            "tok_id": h_obj["tok_id"],
            "t": h_obj["t"],
            "G": h_obj["G"],
            "gen_ids": h_obj["gen_ids"],
            "file_name": h_obj["file_name"],
            "canon": h_obj["canon"],
            "group": "halluc",
        })
        image_to_objs[image_id_r].append({
            "pair_id": pair_id,
            "tok_id": r_obj["tok_id"],
            "t": r_obj["t"],
            "G": r_obj["G"],
            "gen_ids": r_obj["gen_ids"],
            "file_name": r_obj["file_name"],
            "canon": r_obj["canon"],
            "group": "real",
        })

    # ------------------------------------------------------------------
    # 3.6 Run Logit Lens forward passes
    # ------------------------------------------------------------------
    layer_records: List[Dict] = []
    target_device = next(model.parameters()).device

    for image_id, obj_list in tqdm(
        image_to_objs.items(),
        desc="Logit Lens forward passes",
        unit="image",
    ):
        # Use file_name from first object in this image's list
        file_name = obj_list[0]["file_name"]
        gen_ids = obj_list[0]["gen_ids"]
        G = obj_list[0]["G"]

        # Resolve image path
        img_path = resolve_image_path(args.image_dir, file_name)
        if img_path is None:
            print(f"Warning: image not found: {file_name}. Skipping.")
            continue

        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=PROMPT, images=image, return_tensors="pt")
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        gen_tensor = torch.tensor([gen_ids], device=inputs["input_ids"].device)
        ids = torch.cat([inputs["input_ids"], gen_tensor], dim=1)

        # Forward pass requesting ALL hidden states
        with torch.no_grad():
            outputs = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                pixel_values=inputs["pixel_values"],
                output_hidden_states=True,
            )

        # hidden_states: tuple of (n_layers+1) tensors, each (1, L_total, D)
        hidden_states = outputs.hidden_states

        L_total = ids.shape[1]
        L_prompt = L_total - G   # number of prompt tokens

        # Process each object's Logit Lens at position t-1 (m=1)
        for obj in obj_list:
            t = obj["t"]
            o_tok_id = obj["tok_id"]
            if o_tok_id is None:
                continue

            # Build V_obj union {o} for this object
            v_union = list(v_obj_ids | {o_tok_id})
            v_union_tensor = torch.tensor(
                v_union, dtype=torch.long, device=proj_device
            )
            o_in_union_idx = v_union.index(o_tok_id)

            # Position t-1 in full sequence (0-indexed)
            # gen token y_i is at position L_prompt + i in the full sequence.
            # t is the index within generated tokens, so:
            # position of y_{t-1} = L_prompt + (t - 1)
            pos_t_minus_1 = L_prompt + (t - 1)

            if pos_t_minus_1 < 0 or pos_t_minus_1 >= L_total:
                continue

            # Move hidden states slice to projection device for computation
            layer_scores = compute_logit_lens_s(
                hidden_states=hidden_states,
                norm_module=norm_module,
                lm_head_module=lm_head_module,
                pos=pos_t_minus_1,
                o_tok_id=o_tok_id,
                v_union_tensor=v_union_tensor,
                o_in_union_idx=o_in_union_idx,
                n_layers=n_layers,
            )

            for rec in layer_scores:
                layer_records.append({
                    "pair_id": obj["pair_id"],
                    "image_id": image_id,
                    "canon": obj["canon"],
                    "group": obj["group"],
                    "t": t,
                    "G": G,
                    "layer": rec["layer"],
                    "s": rec["s"],
                    "logp": rec["logp"],
                    "rank": rec["rank"],
                    "conf": rec["conf"],
                    "ent": rec["ent"],
                })

        # Free memory
        del outputs, hidden_states, ids
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3.7 Save layer_records.csv
    # ------------------------------------------------------------------
    records_df = pd.DataFrame(layer_records)
    records_path = os.path.join(args.output_dir, "layer_records.csv")
    records_df.to_csv(records_path, index=False)
    print(f"Saved {len(records_df)} layer records to {records_path}")

    # ------------------------------------------------------------------
    # 3.8 Compute per-layer summary statistics
    # ------------------------------------------------------------------
    print("Computing per-layer summary statistics...")
    summary_rows = []

    halluc_df = records_df[records_df["group"] == "halluc"].copy()
    real_df = records_df[records_df["group"] == "real"].copy()

    # Merge on pair_id and layer to get matched pairs for statistics
    merged = pd.merge(
        halluc_df[["pair_id", "layer", "s"]].rename(columns={"s": "s_h"}),
        real_df[["pair_id", "layer", "s"]].rename(columns={"s": "s_r"}),
        on=["pair_id", "layer"],
    )

    layers = sorted(merged["layer"].unique())
    for layer in layers:
        sub = merged[merged["layer"] == layer]
        s_h = sub["s_h"].values
        s_r = sub["s_r"].values
        delta = s_h - s_r

        mean_s_h = float(np.mean(s_h))
        mean_s_r = float(np.mean(s_r))
        mean_delta = float(np.mean(delta))

        ci_lo, ci_hi = paired_bootstrap_delta(s_h, s_r, B=2000, seed=args.seed)
        dz = cohens_dz(delta)
        wilcoxon_p = compute_wilcoxon_p(s_h, s_r)
        frac_h_higher = float(np.mean(s_h > s_r))
        auroc = bootstrap_auroc(s_r, s_h, B=500, seed=args.seed)

        summary_rows.append({
            "layer": layer,
            "n_pairs": len(sub),
            "mean_S_halluc": round(mean_s_h, 4),
            "mean_S_real": round(mean_s_r, 4),
            "mean_delta": round(mean_delta, 4),
            "delta_ci_lo": round(ci_lo, 4),
            "delta_ci_hi": round(ci_hi, 4),
            "dz": round(dz, 4),
            "wilcoxon_p": wilcoxon_p,
            "frac_halluc_higher": round(frac_h_higher, 4),
            "auroc_S": round(auroc[0], 4),
            "median_rank_halluc": float(
                halluc_df[halluc_df["layer"] == layer]["rank"].median()
            ),
            "median_rank_real": float(
                real_df[real_df["layer"] == layer]["rank"].median()
            ),
        })

    # Apply Holm-Bonferroni correction across all layers
    raw_p_values = [row["wilcoxon_p"] for row in summary_rows]
    holm_p_values = holm_bonferroni(raw_p_values)
    for row, hp in zip(summary_rows, holm_p_values):
        row["holm_p"] = hp

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "layer_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved layer summary to {summary_path}")

    # ------------------------------------------------------------------
    # 3.9 Visualization
    # ------------------------------------------------------------------
    _generate_figures(summary_df, args.output_dir, n_layers)

    # ------------------------------------------------------------------
    # 3.10 Generate REPORT.md
    # ------------------------------------------------------------------
    _generate_report(summary_df, funnel_data, args.output_dir, n_layers)

    print("Experiment 2 (Logit Lens) complete.")


# ---------------------------------------------------------------------------
# Section 4: Visualization
# ---------------------------------------------------------------------------

def _generate_figures(summary_df: pd.DataFrame, output_dir: str, n_layers: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping figures.")
        return

    fig_dir = os.path.join(output_dir, "figures")
    layers = summary_df["layer"].values

    # Figure 1: Mean S vs. Layer
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, summary_df["mean_S_halluc"].values, color="#d62728",
            linewidth=1.8, label="Hallucinated Object")
    ax.plot(layers, summary_df["mean_S_real"].values, color="#1f77b4",
            linewidth=1.8, label="Real Object")
    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel("Mean Normalized Score S(o, layer, m=1)", fontsize=12)
    ax.set_title("Logit Lens: Object Favorability Score at Position t-1 Across Layers", fontsize=13)
    ax.set_xticks(range(1, n_layers + 1, 2))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "s_vs_layer.png"), dpi=300)
    plt.close()

    # Figure 2: Delta vs. Layer with CI and Holm significance markers
    fig, ax = plt.subplots(figsize=(10, 5))
    delta = summary_df["mean_delta"].values
    ci_lo = summary_df["delta_ci_lo"].values
    ci_hi = summary_df["delta_ci_hi"].values
    holm_p = summary_df["holm_p"].values

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(layers, ci_lo, ci_hi, alpha=0.2, color="#ff7f0e")
    ax.plot(layers, delta, color="#ff7f0e", linewidth=1.8,
            label="Delta = S(halluc) - S(real)")

    # Mark layers with Holm-significant results
    sig_layers = [l for l, p in zip(layers, holm_p) if p < 0.05]
    sig_delta = [d for l, d, p in zip(layers, delta, holm_p) if p < 0.05]
    if sig_layers:
        ax.scatter(sig_layers, sig_delta, color="#2ca02c", zorder=5,
                   s=60, label="Holm p < 0.05")

    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel("Mean Delta S (halluc - real)", fontsize=12)
    ax.set_title("Logit Lens: Paired Difference Delta at Position t-1 Across Layers", fontsize=13)
    ax.set_xticks(range(1, n_layers + 1, 2))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "delta_vs_layer.png"), dpi=300)
    plt.close()

    # Figure 3: Median Rank vs. Layer
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, summary_df["median_rank_halluc"].values, color="#d62728",
            linewidth=1.8, label="Hallucinated Object")
    ax.plot(layers, summary_df["median_rank_real"].values, color="#1f77b4",
            linewidth=1.8, label="Real Object")
    ax.invert_yaxis()
    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel("Median Vocabulary Rank (lower = better)", fontsize=12)
    ax.set_title("Logit Lens: Median Rank at Position t-1 Across Layers", fontsize=13)
    ax.set_xticks(range(1, n_layers + 1, 2))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "rank_vs_layer.png"), dpi=300)
    plt.close()

    print("Figures saved.")


# ---------------------------------------------------------------------------
# Section 5: Report generation
# ---------------------------------------------------------------------------

def _generate_report(
    summary_df: pd.DataFrame,
    funnel_data: Dict,
    output_dir: str,
    n_layers: int,
) -> None:
    sig_df = summary_df[summary_df["holm_p"] < 0.05].sort_values("mean_delta")
    if len(sig_df) > 0:
        peak_layer = int(summary_df.loc[summary_df["mean_delta"].idxmin(), "layer"])
        peak_delta = float(summary_df["mean_delta"].min())
        first_sig_layer = int(sig_df["layer"].min())
        classification = "LAYER-SPECIFIC SIGNAL"
    else:
        peak_layer = -1
        peak_delta = 0.0
        first_sig_layer = -1
        classification = "NO SIGNIFICANT LAYER"

    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# EXPERIMENT 2 REPORT: Logit Lens Analysis at Position t-1",
        "",
        f"*Generated on: {now}*",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        f"- **Research Question:** At which transformer layer (1..{n_layers}) does divergence",
        "  between hallucinated and real objects first emerge at position t-1 (m=1)?",
        f"- **Matched Pairs:** {funnel_data['matched_pairs']}",
        f"- **Primary Finding:** {classification}",
        f"- **First Significant Layer:** {first_sig_layer}",
        f"- **Peak Divergence Layer:** {peak_layer} (Delta = {peak_delta:.4f})",
        "",
        "---",
        "",
        "## 2. Experimental Setup",
        "- **Analysis Position:** t-1 (one token before object mention, i.e. m=1).",
        "- **Projection Method:** Logit Lens (hidden state -> LayerNorm -> lm_head).",
        "- **Labels Source:** data/labels.jsonl (Experiment 1, read-only).",
        f"- **Matching Caliper:** <= {funnel_data['caliper']}",
        f"- **Minimum t:** >= {funnel_data['K']}",
        f"- **Random Seed:** {funnel_data['seed']}",
        "",
        "---",
        "",
        "## 3. Data Funnel",
        "| Funnel Stage | Count |",
        "|---|---|",
        f"| Total generated captions | {funnel_data['captions']} |",
        f"| First mentions extracted | {funnel_data['first_mentions']} |",
        f"| Hallucinated mentions | {funnel_data['halluc_all']} |",
        f"| Real mentions | {funnel_data['real_all']} |",
        f"| **Matched 1:1 pairs** | **{funnel_data['matched_pairs']}** |",
        f"| Dropped unmatched hallucinated | {funnel_data['unmatched_halluc_dropped']} |",
        "",
        "---",
        "",
        "## 4. Layer-by-Layer Results (Holm-significant layers only)",
        "",
    ]

    if len(sig_df) > 0:
        lines.append(
            "| Layer | Mean S(halluc) | Mean S(real) | Delta | 95% CI | Cohen dz | Holm p |"
        )
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for _, row in sig_df.iterrows():
            lines.append(
                f"| **{int(row['layer'])}** "
                f"| {row['mean_S_halluc']:.4f} "
                f"| {row['mean_S_real']:.4f} "
                f"| **{row['mean_delta']:.4f}** "
                f"| [{row['delta_ci_lo']:.4f}, {row['delta_ci_hi']:.4f}] "
                f"| {row['dz']:.4f} "
                f"| **{row['holm_p']:.4e}** |"
            )
    else:
        lines.append("No layers reached Holm-adjusted significance (p < 0.05).")

    lines += [
        "",
        "---",
        "",
        "## 5. Interpretation",
        f"- The Logit Lens analysis at position t-1 across {n_layers} transformer layers",
        "  reveals at which depth of computation the model's preference for real vs.",
        "  hallucinated objects begins to diverge.",
        "- Layers with strongly negative Delta and Holm p < 0.05 represent the",
        "  transformer depth at which visual grounding information dominates the",
        "  token prediction distribution.",
        "- The transition from non-significant to significant layers identifies the",
        "  computational boundary between early language pattern matching and",
        "  vision-conditioned object selection.",
        "",
        "---",
        "",
        "## 6. Figures",
        "- `figures/s_vs_layer.png`: Mean S for hallucinated vs. real objects per layer.",
        "- `figures/delta_vs_layer.png`: Paired Delta with 95% Bootstrap CI per layer.",
        "- `figures/rank_vs_layer.png`: Median vocabulary rank per layer.",
        "",
        "---",
        "",
        "## 7. Limitations",
        "1. Analysis is fixed at position t-1 (m=1). Extension to m=0 and m=2 is straightforward.",
        "2. Logit Lens projection assumes the final LayerNorm and lm_head generalize to",
        "   intermediate representations, which may not hold for early layers.",
        "3. Single model (LLaVA-1.5-7B) and dataset (COCO val2014).",
    ]

    report_path = os.path.join(output_dir, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report saved to {report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 2: Logit Lens analysis at position t-1 across transformer layers."
    )
    parser.add_argument(
        "--labels_file", type=str, default="data/labels.jsonl",
        help="Path to labels.jsonl produced by Experiment 1 (read-only)."
    )
    parser.add_argument(
        "--synonyms_file", type=str, default="data/synonyms.txt",
        help="Path to synonyms.txt."
    )
    parser.add_argument(
        "--image_dir", type=str, default="data/val2014",
        help="Directory containing COCO val2014 images."
    )
    parser.add_argument(
        "--model_name", type=str, default="llava-hf/llava-1.5-7b-hf",
        help="HuggingFace model identifier."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/exp2_logitlens",
        help="Output directory for Experiment 2 results."
    )
    parser.add_argument(
        "--K", type=int, default=10,
        help="Minimum token index for candidate objects (default: 10)."
    )
    parser.add_argument(
        "--caliper", type=float, default=0.1,
        help="Matching caliper on relative position (default: 0.1)."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for matching and bootstrap (default: 0)."
    )
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
