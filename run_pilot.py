"""
End-to-end execution script for Pilot Experiment H2.1 on VLM (COCO val2014, 20 images).
Runs caption generation, teacher-forcing evaluation, prefix perturbations,
robustness checks, statistical analyses, and report generation with live logging and checkpointing.
"""

import os
import sys
import time
import json
import random
import logging
import argparse
import subprocess
import traceback
from typing import Dict, List, Set, Tuple, Optional, Any

# =============================================================================
# 0. Ensure dependencies and print system status
# =============================================================================
def ensure_dependencies():
    """Installs only missing packages at startup."""
    packages = {
        "transformers": "transformers>=4.40.0",
        "accelerate": "accelerate>=0.28.0",
        "pycocotools": "pycocotools",
        "tqdm": "tqdm>=4.65.0",
        "scipy": "scipy>=1.10.0",
        "statsmodels": "statsmodels",
        "matplotlib": "matplotlib>=3.7.0",
        "pandas": "pandas",
        "PIL": "pillow",
    }
    missing = []
    import importlib.util
    for mod_name, pkg_req in packages.items():
        if importlib.util.find_spec(mod_name) is None:
            missing.append(pkg_req)

    if missing:
        print(f"Installing missing dependencies: {missing}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


ensure_dependencies()

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from config import Config
from data import COCODataset
from objects import ObjectVocabulary
from model import VLMRunner
from compute import (
    jsd_divergence,
    compute_restricted_distribution,
    compute_entropy,
    format_top3_objects,
    is_candidate_perturbation_site,
    select_alternative_token,
    run_placebo_test,
)
from analysis import run_full_analysis, generate_markdown_report
from viz import (
    plot_fig1_caption_strips,
    plot_fig2_vt_vs_position,
    plot_fig3_st_flip_vs_position,
    plot_fig4_st_vs_vt,
    plot_fig5_mediation,
    plot_fig6_case_studies,
    plot_fig7_distance,
    generate_html_report,
)


# =============================================================================
# 1. Logging setup
# =============================================================================
def setup_logging(log_file: str):
    """Sets up Python logging to stdout and file in append mode."""
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Clear existing handlers
    root_logger.handlers = []

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # File handler with flush
    class FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    fh = FlushingFileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)


logger = logging.getLogger("pilot_h21")


# =============================================================================
# 2. Main Pilot Pipeline
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Pilot Experiment for Hypothesis H2.1 on VLM")
    parser.add_argument("--n_images", type=int, default=20, help="Number of sampled images (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf", help="Model name")
    parser.add_argument("--model_path", type=str, default=None, help="Local model directory override")
    parser.add_argument("--val2014_dir", type=str, default=None, help="COCO val2014 images directory")
    parser.add_argument("--instances_json", type=str, default=None, help="COCO instances_val2014.json path")
    parser.add_argument("--results_dir", type=str, default=None, help="Output results directory")
    parser.add_argument("--no_image_mode", type=str, default="remove", choices=["remove", "black"], help="Primary no-image mode")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from progress.jsonl if exists")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        n_images=args.n_images,
        model_name=args.model_name,
        model_path=args.model_path,
        no_image_mode=args.no_image_mode,
    )
    if args.val2014_dir:
        cfg.val2014_dir = args.val2014_dir
    if args.instances_json:
        cfg.instances_json = args.instances_json
    if args.results_dir:
        cfg.output_dir = args.results_dir

    cfg.resolve_paths()
    setup_logging(cfg.log_path)

    # Set seeds for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("STARTING PILOT EXPERIMENT FOR HYPOTHESIS H2.1 (VLM)")
    logger.info("=" * 70)
    logger.info(f"Python: {sys.version.split()[0]} | Torch: {torch.__version__}")
    try:
        import transformers, accelerate
        logger.info(f"Transformers: {transformers.__version__} | Accelerate: {accelerate.__version__}")
    except Exception:
        pass

    logger.info(f"CUDA Available: {torch.cuda.is_available()} | Devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    logger.info("Configuration dump:")
    for k, v in vars(cfg).items():
        logger.info(f"  {k}: {v}")

    # =========================================================================
    # Step 0: Initialize Dataset and Sample Images
    # =========================================================================
    logger.info(f"Resolving COCO dataset paths...")
    logger.info(f"  val2014_dir: {cfg.val2014_dir}")
    logger.info(f"  instances_json: {cfg.instances_json}")

    coco = COCODataset(val2014_dir=cfg.val2014_dir, instances_json=cfg.instances_json, seed=cfg.seed)
    sampled_image_ids = coco.sample_images(n_images=cfg.n_images, save_path=cfg.image_ids_path)

    # =========================================================================
    # Step 1 & 2: Load Model and Build Object Vocabulary
    # =========================================================================
    load_start = time.time()
    runner = VLMRunner(config=cfg)
    logger.info(f"VLM Runner loaded in {time.time() - load_start:.2f}s")

    obj_vocab = ObjectVocabulary(
        tokenizer=runner.tokenizer,
        synonyms_file=cfg.synonyms_file,
        dropped_words_path=cfg.dropped_words_path,
    )
    special_ids = set(runner.tokenizer.all_special_ids)

    # Checkpointing / Resume detection
    done_image_ids = set()
    if os.path.isfile(cfg.progress_jsonl_path):
        with open(cfg.progress_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        done_image_ids.add(record.get("image_id"))
                    except Exception:
                        pass
        if done_image_ids:
            logger.info(f"RESUME: {len(done_image_ids)} images already processed according to progress.jsonl")

    # CSV headers
    csv_columns = [
        "image_id", "t", "T", "rel_pos", "word", "category",
        "s", "k", "orig_token", "alt_token", "p_alt",
        "V_t", "S_t", "flip", "delta_p", "salience", "in_gt",
        "confidence", "entropy", "top3_before", "top3_after"
    ]

    if not os.path.isfile(cfg.records_path):
        with open(cfg.records_path, "w", encoding="utf-8") as f:
            f.write(",".join(csv_columns) + "\n")
    if not os.path.isfile(cfg.records_black_path):
        with open(cfg.records_black_path, "w", encoding="utf-8") as f:
            f.write(",".join(csv_columns) + "\n")

    # Storage for captions and analysis data
    captions_data = {}
    if os.path.isfile(cfg.captions_path):
        try:
            with open(cfg.captions_path, "r", encoding="utf-8") as f:
                captions_data = json.load(f)
        except Exception:
            captions_data = {}

    all_records = []
    all_records_black = []
    # If resuming, load existing rows
    if os.path.isfile(cfg.records_path):
        try:
            existing_df = pd.read_csv(cfg.records_path)
            all_records = existing_df.to_dict(orient="records")
        except Exception:
            all_records = []

    total_objects_found = 0
    total_perturbations_done = 0
    total_skipped = 0

    # =========================================================================
    # Step 3: Main Iteration Over Images
    # =========================================================================
    logger.info(f"Processing {len(sampled_image_ids)} images...")

    with logging_redirect_tqdm():
        outer_pbar = tqdm(
            sampled_image_ids,
            desc="Ảnh",
            total=len(sampled_image_ids),
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
            mininterval=0.5,
        )

        for img_idx, image_id in enumerate(outer_pbar):
            if image_id in done_image_ids:
                outer_pbar.set_postfix({
                    "id": image_id,
                    "status": "resumed",
                    "objs": total_objects_found,
                    "records": len(all_records),
                })
                continue

            img_start_time = time.time()
            img_records = []
            img_records_black = []

            try:
                # 1. Load image
                pil_img = coco.load_image(image_id)

                # 2. Greedy caption generation
                prompt_ids, caption_text, T, gen_ids = runner.generate_caption(pil_img, stream=False)

                captions_data[str(image_id)] = {
                    "image_id": image_id,
                    "caption": caption_text,
                    "T": T,
                    "gen_ids": gen_ids[0].tolist(),
                }

                # 3. Full teacher-forcing pass with image
                inputs_img = runner.processor(text=cfg.prompt, images=pil_img, return_tensors="pt")
                pixel_values = inputs_img["pixel_values"].to(runner.device)

                logits_img_all, match_rate = runner.get_object_logits(
                    prompt_ids=prompt_ids,
                    gen_ids=gen_ids,
                    pixel_values=pixel_values,
                )

                if match_rate < 0.95:
                    logger.warning(
                        f"Image {image_id}: Greedy sanity check match rate is {match_rate:.2%} (< 95%)"
                    )

                # 4. No-image modes (remove and black)
                logits_noimg_remove = runner.get_no_image_logits(gen_ids=gen_ids, mode="remove")
                logits_noimg_black = runner.get_no_image_logits(
                    gen_ids=gen_ids,
                    mode="black",
                    image_size=pil_img.size,
                )

                # 5. Identify object positions
                obj_positions = []
                for t in range(T):
                    tok_id = int(gen_ids[0, t].item())
                    if obj_vocab.is_object_token(tok_id):
                        w = obj_vocab.get_word(tok_id)
                        cat = obj_vocab.get_category(tok_id)
                        obj_positions.append((t, tok_id, w, cat))

                total_objects_found += len(obj_positions)
                img_skipped = 0
                img_pert_done = 0

                # Middle bar: Object tokens of the current image
                middle_pbar = tqdm(
                    obj_positions,
                    desc=f"Object (img {image_id})",
                    file=sys.stdout,
                    dynamic_ncols=True,
                    leave=False,
                    mininterval=0.5,
                )

                for (t, obj_tok_id, obj_word, obj_cat) in middle_pbar:
                    rel_pos = float(t / T) if T > 0 else 0.0

                    # Restricted distributions at position t
                    p_img = compute_restricted_distribution(logits_img_all[t], obj_vocab.obj_ids)
                    p_remove = compute_restricted_distribution(logits_noimg_remove[t], obj_vocab.obj_ids)
                    p_black = compute_restricted_distribution(logits_noimg_black[t], obj_vocab.obj_ids)

                    # V_t in both modes
                    v_t_remove = jsd_divergence(p_img, p_remove)
                    v_t_black = jsd_divergence(p_img, p_black)

                    # Covariates at position t
                    salience, in_gt = coco.get_salience_and_in_gt(image_id, obj_cat)
                    confidence = float(np.max(p_img))
                    entropy = compute_entropy(p_img)
                    top3_before_str = format_top3_objects(p_img, obj_vocab)

                    # Check object mass warning
                    if confidence < 0.01:
                        logger.warning(f"Image {image_id}, t={t} ({obj_word}): obj_mass confidence < 0.01 ({confidence:.4f})")

                    # Candidate perturbation sites s in [t-8, t-1]
                    s_min = max(0, t - cfg.window_size)
                    s_max = t - 1
                    candidate_sites = []

                    for s in range(s_min, s_max + 1):
                        cand_tok_id = int(gen_ids[0, s].item())
                        cand_tok_str = runner.tokenizer.convert_ids_to_tokens(cand_tok_id) or ""
                        if is_candidate_perturbation_site(cand_tok_id, cand_tok_str, obj_vocab, special_ids):
                            candidate_sites.append(s)

                    if not candidate_sites:
                        logger.info(f"Image {image_id}, t={t}: No valid candidate sites in window [{s_min}, {s_max}], skipping.")
                        img_skipped += 1
                        total_skipped += 1
                        continue

                    # Randomly pick up to 3 sites (seed 42 deterministic per object)
                    rng = random.Random(cfg.seed + image_id * 1000 + t)
                    chosen_sites = rng.sample(candidate_sites, min(len(candidate_sites), cfg.max_sites_per_object))

                    # Inner bar: Perturbation sites
                    inner_pbar = tqdm(
                        chosen_sites,
                        desc="Perturb",
                        file=sys.stdout,
                        dynamic_ncols=True,
                        leave=False,
                        mininterval=0.5,
                    )

                    for s in inner_pbar:
                        orig_tok_id = int(gen_ids[0, s].item())
                        orig_tok_str = runner.tokenizer.decode([orig_tok_id]).strip()

                        # Model's logits at position s (logits predicting y_s)
                        logits_at_s = logits_img_all[s]

                        # Select alternative token
                        alt_tok_id, p_alt, p_orig_full = select_alternative_token(
                            full_logits_at_s=logits_at_s,
                            orig_tok_id=orig_tok_id,
                            obj_vocab=obj_vocab,
                            tokenizer=runner.tokenizer,
                            special_ids=special_ids,
                            top_n_search=cfg.top_k_candidates * 10,
                        )

                        if alt_tok_id is None:
                            logger.info(f"Image {image_id}, t={t}, s={s}: No alternative token found, skipping.")
                            img_skipped += 1
                            total_skipped += 1
                            continue

                        alt_tok_str = runner.tokenizer.decode([alt_tok_id]).strip()
                        k = t - s

                        # Build perturbed prefix up to t
                        pert_prefix = gen_ids[:, :t].clone()
                        pert_prefix[0, s] = alt_tok_id

                        # Forward pass for perturbed prefix
                        logits_pert_at_t = runner.get_perturbed_target_logits(
                            prompt_ids=prompt_ids,
                            gen_ids_perturbed_prefix=pert_prefix,
                            pixel_values=pixel_values,
                        )

                        p_pert = compute_restricted_distribution(logits_pert_at_t, obj_vocab.obj_ids)
                        s_t = jsd_divergence(p_img, p_pert)

                        # Flip check
                        flip = 1 if (np.argmax(p_img) != np.argmax(p_pert)) else 0

                        # Delta p of the ground-truth object token y_t
                        try:
                            obj_idx_in_vocab = obj_vocab.obj_ids.index(obj_tok_id)
                            delta_p = float(p_img[obj_idx_in_vocab] - p_pert[obj_idx_in_vocab])
                        except ValueError:
                            delta_p = 0.0

                        top3_after_str = format_top3_objects(p_pert, obj_vocab)

                        # PLACEBO TEST for first perturbation site of this object
                        if s == chosen_sites[0]:
                            s_placebo, flip_placebo = run_placebo_test(
                                runner=runner,
                                prompt_ids=prompt_ids,
                                gen_ids=gen_ids,
                                pixel_values=pixel_values,
                                t=t,
                                s=s,
                                p_orig_restricted=p_img,
                                obj_vocab=obj_vocab,
                            )
                            if s_placebo > 1e-4:
                                logger.warning(
                                    f"PLACEBO WARNING: Image {image_id}, t={t}: S_t_placebo={s_placebo:.6f} > 1e-4"
                                )
                            assert flip_placebo == 0, f"Placebo flip={flip_placebo} != 0 at image {image_id}, t={t}"

                        # Record for primary mode (remove)
                        rec = {
                            "image_id": image_id,
                            "t": t,
                            "T": T,
                            "rel_pos": rel_pos,
                            "word": obj_word,
                            "category": obj_cat,
                            "s": s,
                            "k": k,
                            "orig_token": orig_tok_str,
                            "alt_token": alt_tok_str,
                            "p_alt": p_alt,
                            "V_t": v_t_remove,
                            "S_t": s_t,
                            "flip": flip,
                            "delta_p": delta_p,
                            "salience": salience,
                            "in_gt": in_gt,
                            "confidence": confidence,
                            "entropy": entropy,
                            "top3_before": top3_before_str,
                            "top3_after": top3_after_str,
                        }
                        img_records.append(rec)
                        all_records.append(rec)

                        # Record for robustness mode (black)
                        rec_black = rec.copy()
                        rec_black["V_t"] = v_t_black
                        img_records_black.append(rec_black)
                        all_records_black.append(rec_black)

                        img_pert_done += 1
                        total_perturbations_done += 1

                # Checkpointing after each image: append to records.csv
                if img_records:
                    df_append = pd.DataFrame(img_records)[csv_columns]
                    df_append.to_csv(cfg.records_path, mode="a", header=False, index=False)

                    df_black_append = pd.DataFrame(img_records_black)[csv_columns]
                    df_black_append.to_csv(cfg.records_black_path, mode="a", header=False, index=False)

                # Save progress.jsonl
                progress_line = {
                    "image_id": image_id,
                    "T": T,
                    "n_objects": len(obj_positions),
                    "n_perturbations": img_pert_done,
                    "n_skipped": img_skipped,
                    "match_rate": match_rate,
                    "time_sec": round(time.time() - img_start_time, 2),
                }
                with open(cfg.progress_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(progress_line) + "\n")

                # Memory hygiene
                peak_mems = runner.get_peak_memory_mb()
                runner.clean_memory()

                # Image log
                caption_trunc = (caption_text[:300] + "...") if len(caption_text) > 300 else caption_text
                logger.info(
                    f"Image {image_id} finished in {time.time() - img_start_time:.2f}s | "
                    f"T={T}, Objs={len(obj_positions)}, Pert={img_pert_done}, Skipped={img_skipped}, "
                    f"Match={match_rate:.2%}, PeakMem={peak_mems}"
                )
                logger.info(f"  Caption: \"{caption_trunc}\"")

                # Running summary every 5 images
                if (img_idx + 1) % 5 == 0:
                    current_df = pd.DataFrame(all_records)
                    if len(current_df) > 0:
                        m_vt = current_df["V_t"].mean()
                        m_st = current_df["S_t"].mean()
                        fl_rate = current_df["flip"].mean()
                        logger.info(
                            f"--- RUNNING SUMMARY (Image {img_idx+1}/{len(sampled_image_ids)}) --- "
                            f"Records={len(current_df)}, Mean V_t={m_vt:.4f}, Mean S_t={m_st:.4f}, Flip rate={fl_rate:.2%}"
                        )

            except Exception as e:
                logger.error(f"Error processing image {image_id}: {e}\n{traceback.format_exc()}")
                continue

            outer_pbar.set_postfix({
                "id": image_id,
                "objs": total_objects_found,
                "records": len(all_records),
            })

    # Save captions.json
    with open(cfg.captions_path, "w", encoding="utf-8") as f:
        json.dump(captions_data, f, indent=2)
    logger.info(f"Saved generated captions to {cfg.captions_path}")

    # =========================================================================
    # Step 4: Statistical Analysis (A0 through A5)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("RUNNING STATISTICAL ANALYSES (A0 - A5, 2000 cluster-bootstraps)...")
    logger.info("=" * 70)

    df_full = pd.DataFrame(all_records)
    df_black_full = pd.DataFrame(all_records_black)

    if len(df_full) == 0:
        logger.error("No perturbation records generated! Aborting analysis.")
        return

    analysis_res = run_full_analysis(
        df=df_full,
        df_black=df_black_full,
        n_resamples=cfg.cluster_bootstrap_resamples,
        seed=cfg.seed,
    )

    # Save analysis.json
    with open(cfg.analysis_json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_res, f, indent=2)
    logger.info(f"Saved analysis results to {cfg.analysis_json_path}")

    # Save analysis.md
    generate_markdown_report(analysis_res, cfg.analysis_md_path)

    # =========================================================================
    # Step 5: Visualizations (fig1 - fig7) and report.html
    # =========================================================================
    logger.info("=" * 70)
    logger.info("GENERATING FIGURES AND STANDALONE REPORT.HTML...")
    logger.info("=" * 70)

    figs_dir = os.path.join(cfg.output_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    fig_paths = {}
    # fig1
    try:
        fig1_paths = plot_fig1_caption_strips(df_full, captions_data, coco, figs_dir, top_n_images=3)
        fig_paths["fig1"] = fig1_paths
    except Exception as e:
        logger.warning(f"Error plotting fig1: {e}")

    # fig2
    fig2_path = os.path.join(figs_dir, "fig2_Vt_vs_position.png")
    plot_fig2_vt_vs_position(df_full, analysis_res, fig2_path)
    fig_paths["fig2"] = fig2_path

    # fig3
    fig3_path = os.path.join(figs_dir, "fig3_St_flip_vs_position.png")
    plot_fig3_st_flip_vs_position(df_full, analysis_res, fig3_path)
    fig_paths["fig3"] = fig3_path

    # fig4
    fig4_path = os.path.join(figs_dir, "fig4_St_vs_Vt.png")
    plot_fig4_st_vs_vt(df_full, fig4_path)
    fig_paths["fig4"] = fig4_path

    # fig5
    fig5_path = os.path.join(figs_dir, "fig5_mediation.png")
    plot_fig5_mediation(analysis_res, fig5_path)
    fig_paths["fig5"] = fig5_path

    # fig6
    fig6_paths = plot_fig6_case_studies(df_full, figs_dir)
    fig_paths["fig6"] = fig6_paths

    # fig7
    fig7_path = os.path.join(figs_dir, "fig7_distance.png")
    plot_fig7_distance(df_full, analysis_res, fig7_path)
    fig_paths["fig7"] = fig7_path

    # Generate standalone report.html
    generate_html_report(analysis_res, fig_paths, cfg.report_html_path)

    # =========================================================================
    # Step 6: Final Summary Block
    # =========================================================================
    total_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info("PILOT EXPERIMENT H2.1 EXECUTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total processed images: {len(sampled_image_ids)}")
    logger.info(f"Total object tokens found: {total_objects_found}")
    logger.info(f"Total perturbation records: {len(df_full)}")
    logger.info(f"Total skipped sites: {total_skipped}")
    logger.info(f"Total execution time: {total_time:.2f}s ({total_time/60.0:.2f} min)")
    logger.info("Generated output files:")
    logger.info(f"  - records.csv:          {cfg.records_path}")
    logger.info(f"  - records_v_black.csv:  {cfg.records_black_path}")
    logger.info(f"  - captions.json:        {cfg.captions_path}")
    logger.info(f"  - image_ids.json:       {cfg.image_ids_path}")
    logger.info(f"  - analysis.json:        {cfg.analysis_json_path}")
    logger.info(f"  - analysis.md:          {cfg.analysis_md_path}")
    logger.info(f"  - Figures directory:    {figs_dir}")
    logger.info(f"  - report.html:          {cfg.report_html_path}")
    logger.info(f"  - run.log:              {cfg.log_path}")
    logger.info(f"  - progress.jsonl:       {cfg.progress_jsonl_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
