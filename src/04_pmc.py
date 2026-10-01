"""
Step 4: Analysis B - Reproduction of Preceding Minimum Confidence (PMC) from TruthPrInt.
Conforms to EXPERIMENT1_SPEC.md (Section 5, 6.6).

Features:
- Evaluates all first-mention objects
- Sentence start identification based on '.' token pieces
- n_prec = t - sent_start
- PMC = min_{i in [sent_start, t-1]} conf_i
- argmin_dist = t - argmin_i conf_i
- Length control via n_prec bins: 1-3, 4-6, 7-10, >=11
- Comparison on matched pairs
- Outputs:
  results/exp1/pmc_records.csv
  results/exp1/pmc_summary.csv
"""

import argparse
import collections
import json
import os
import sys
from typing import Dict, List, Set, Tuple, Any

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm


PROMPT = "USER: <image>\nPlease describe this image in detail. ASSISTANT:"


def find_sent_start(pieces: List[str], t: int) -> int:
    """
    Finds sent_start according to Section 6.6:
    Token immediately following the last token whose piece ends with '.' before position t.
    If no preceding period, sent_start = 0.
    """
    sent_start = 0
    for i in range(t - 1, -1, -1):
        p = pieces[i]
        if p.endswith(".") or p.rstrip().endswith("."):
            sent_start = i + 1
            break
    return sent_start


def get_nprec_bin(n: int) -> str:
    if 1 <= n <= 3:
        return "1-3"
    elif 4 <= n <= 6:
        return "4-6"
    elif 7 <= n <= 10:
        return "7-10"
    else:
        return ">=11"


def main():
    parser = argparse.ArgumentParser(description="Reproduce TruthPrInt PMC metric (Analysis B)")
    parser.add_argument("--image_dir", type=str, default="data/val2014")
    parser.add_argument("--labels_file", type=str, default="data/labels.jsonl")
    parser.add_argument("--captions_file", type=str, default="data/captions.jsonl")
    parser.add_argument("--lag_records_file", type=str, default="results/exp1/lag_records.csv")
    parser.add_argument("--output_dir", type=str, default="results/exp1")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load_8bit", action="store_true", help="Load model in 8-bit to fit 16GB GPUs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load pieces mapping from captions.jsonl
    print(f"Loading pieces mapping from {args.captions_file}...")
    pieces_by_image = {}
    with open(args.captions_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pieces_by_image[rec["image_id"]] = rec["pieces"]

    # 2. Collect all first-mention objects
    print(f"Loading first-mention objects from {args.labels_file}...")
    objects_by_image = collections.defaultdict(list)
    total_objects = 0

    with open(args.labels_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            img_id = rec["image_id"]
            file_name = rec["file_name"]
            gen_ids = rec["gen_ids"]
            pieces = pieces_by_image.get(img_id, [])

            for m in rec["mentions"]:
                if not m["first"]:
                    continue
                t = m["tok_idx"]
                sent_start = find_sent_start(pieces, t)
                n_prec = t - sent_start
                if n_prec < 1:
                    continue

                objects_by_image[img_id].append({
                    "image_id": img_id,
                    "file_name": file_name,
                    "canon": m["canon"],
                    "halluc": m["halluc"],
                    "t": t,
                    "sent_start": sent_start,
                    "n_prec": n_prec,
                    "gen_ids": gen_ids
                })
                total_objects += 1

    print(f"Found {total_objects} first-mention objects across {len(objects_by_image)} images.")

    if total_objects == 0:
        print("No objects for PMC analysis. Exiting.")
        return

    # 3. Load model
    print(f"Loading model '{args.model_id}' on {args.device}...")
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_id)

    model_kwargs = {"torch_dtype": torch.float16 if args.device == "cuda" else torch.float32}
    if args.load_8bit:
        model_kwargs["load_in_8bit"] = True

    device_map = "auto" if args.device == "cuda" else None
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map=device_map,
        **model_kwargs
    )
    if device_map is None:
        model.to(args.device)
    model.eval()

    target_device = model.device if hasattr(model, "device") else args.device

    pmc_records = []

    # 4. Forward teacher-forcing per image and compute PMC
    for img_id, obj_list in tqdm(objects_by_image.items(), desc="Computing PMC", dynamic_ncols=True, unit="img"):
        file_name = obj_list[0]["file_name"]
        gen_ids = obj_list[0]["gen_ids"]
        img_path = os.path.join(args.image_dir, file_name)

        if not os.path.exists(img_path):
            continue

        raw_img = Image.open(img_path).convert("RGB")
        inputs = processor(images=raw_img, text=PROMPT, return_tensors="pt")
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        gen_tensor = torch.tensor([gen_ids], device=inputs["input_ids"].device)
        ids = torch.cat([inputs["input_ids"], gen_tensor], dim=1)

        with torch.no_grad():
            logits = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                pixel_values=inputs["pixel_values"]
            ).logits[0]

        G = len(gen_ids)
        Lout = logits.shape[0]
        # pred[i] = logits used to predict gen_ids[i]
        pred = logits[Lout - G - 1 : Lout - 1].float()

        # Compute max confidence for every step
        probs = F.softmax(pred, dim=-1)
        conf_seq = torch.max(probs, dim=-1).values.cpu().numpy()

        for obj in obj_list:
            t = obj["t"]
            sent_start = obj["sent_start"]
            window_confs = conf_seq[sent_start:t]

            pmc = float(np.min(window_confs))
            rel_argmin = int(np.argmin(window_confs))
            argmin_tok = sent_start + rel_argmin
            argmin_dist = int(t - argmin_tok)

            pmc_records.append({
                "image_id": img_id,
                "canon": obj["canon"],
                "halluc": obj["halluc"],
                "t": t,
                "n_prec": obj["n_prec"],
                "pmc": round(pmc, 4),
                "argmin_dist": argmin_dist
            })

    pmc_df = pd.DataFrame(pmc_records)
    pmc_records_path = os.path.join(args.output_dir, "pmc_records.csv")
    pmc_df.to_csv(pmc_records_path, index=False)
    print(f"PMC records saved to {pmc_records_path}")

    # 5. Summarize PMC analysis
    pmc_df["n_prec_bin"] = pmc_df["n_prec"].apply(get_nprec_bin)

    summary_list = []

    # Overall
    h_pmc = pmc_df[pmc_df["halluc"] == True]["pmc"]
    r_pmc = pmc_df[pmc_df["halluc"] == False]["pmc"]
    summary_list.append({
        "subset": "overall",
        "n_halluc": len(h_pmc),
        "n_real": len(r_pmc),
        "mean_pmc_halluc": round(float(h_pmc.mean()), 4) if len(h_pmc) > 0 else 0.0,
        "std_pmc_halluc": round(float(h_pmc.std()), 4) if len(h_pmc) > 0 else 0.0,
        "mean_pmc_real": round(float(r_pmc.mean()), 4) if len(r_pmc) > 0 else 0.0,
        "std_pmc_real": round(float(r_pmc.std()), 4) if len(r_pmc) > 0 else 0.0,
        "mean_argmin_dist_halluc": round(float(pmc_df[pmc_df["halluc"] == True]["argmin_dist"].mean()), 2) if len(h_pmc) > 0 else 0.0,
        "mean_argmin_dist_real": round(float(pmc_df[pmc_df["halluc"] == False]["argmin_dist"].mean()), 2) if len(r_pmc) > 0 else 0.0,
    })

    # By n_prec bins
    for b in ["1-3", "4-6", "7-10", ">=11"]:
        sub = pmc_df[pmc_df["n_prec_bin"] == b]
        sub_h = sub[sub["halluc"] == True]["pmc"]
        sub_r = sub[sub["halluc"] == False]["pmc"]
        summary_list.append({
            "subset": f"n_prec_{b}",
            "n_halluc": len(sub_h),
            "n_real": len(sub_r),
            "mean_pmc_halluc": round(float(sub_h.mean()), 4) if len(sub_h) > 0 else 0.0,
            "std_pmc_halluc": round(float(sub_h.std()), 4) if len(sub_h) > 0 else 0.0,
            "mean_pmc_real": round(float(sub_r.mean()), 4) if len(sub_r) > 0 else 0.0,
            "std_pmc_real": round(float(sub_r.std()), 4) if len(sub_r) > 0 else 0.0,
            "mean_argmin_dist_halluc": round(float(sub[sub["halluc"] == True]["argmin_dist"].mean()), 2) if len(sub_h) > 0 else 0.0,
            "mean_argmin_dist_real": round(float(sub[sub["halluc"] == False]["argmin_dist"].mean()), 2) if len(sub_r) > 0 else 0.0,
        })

    # On matched pairs if lag_records.csv exists
    if os.path.exists(args.lag_records_file):
        try:
            lag_df = pd.read_csv(args.lag_records_file)
            matched_keys = set(zip(lag_df["image_id"], lag_df["t"]))
            sub_matched = pmc_df[pmc_df.apply(lambda r: (r["image_id"], r["t"]) in matched_keys, axis=1)]
            m_h = sub_matched[sub_matched["halluc"] == True]["pmc"]
            m_r = sub_matched[sub_matched["halluc"] == False]["pmc"]
            summary_list.append({
                "subset": "matched_pairs",
                "n_halluc": len(m_h),
                "n_real": len(m_r),
                "mean_pmc_halluc": round(float(m_h.mean()), 4) if len(m_h) > 0 else 0.0,
                "std_pmc_halluc": round(float(m_h.std()), 4) if len(m_h) > 0 else 0.0,
                "mean_pmc_real": round(float(m_r.mean()), 4) if len(m_r) > 0 else 0.0,
                "std_pmc_real": round(float(m_r.std()), 4) if len(m_r) > 0 else 0.0,
                "mean_argmin_dist_halluc": round(float(sub_matched[sub_matched["halluc"] == True]["argmin_dist"].mean()), 2) if len(m_h) > 0 else 0.0,
                "mean_argmin_dist_real": round(float(sub_matched[sub_matched["halluc"] == False]["argmin_dist"].mean()), 2) if len(m_r) > 0 else 0.0,
            })
        except Exception as e:
            print(f"Warning: Could not compute matched pairs PMC: {e}")

    summary_df = pd.DataFrame(summary_list)
    summary_path = os.path.join(args.output_dir, "pmc_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"PMC summary saved to {summary_path}")
    print("\nPMC Summary Table:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
