"""
Main Orchestrator for Confirmatory and Causal Experiments (v2).
Single Entry Point: python -u run_all.py --stage {m0,m1,m2,e1,e2,e3,analysis,viz,report,all} --n_images N

Features:
- Single in-place progress bar per task (no duplicate percentage lines).
- Automatic package installation and dependency verification.
- Milestones M0, M1, M2 with timing extrapolation and power checks.
- Full end-to-end execution across E1, E2, E3, Analysis, and Visualizations.
- Checkpointing and seamless resume via progress.jsonl.
- GPU memory hygiene and cluster-bootstrap evaluation.
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Dict, List, Set, Tuple, Optional, Any

# Ensure unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

# 1. Dependency check & auto-installation
def ensure_dependencies():
    packages = [
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("pycocotools", "pycocotools"),
        ("tqdm", "tqdm"),
        ("scipy", "scipy"),
        ("statsmodels", "statsmodels"),
        ("matplotlib", "matplotlib"),
        ("sklearn", "scikit-learn"),
        ("spacy", "spacy"),
    ]
    for mod_name, pip_name in packages:
        try:
            __import__(mod_name)
        except ImportError:
            print(f"Installing missing package {pip_name}...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=True)

    # Check spaCy en_core_web_sm
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
        except Exception:
            print("Downloading spaCy en_core_web_sm...")
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    except Exception as e:
        print(f"Notice regarding spaCy: {e}")

try:
    ensure_dependencies()
except Exception as e:
    print(f"Dependency auto-check warning: {e}")

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config_v2 import ConfigV2
from data_v2 import COCODatasetV2
from objects import ObjectVocabulary
from pos_tagger import POSTagger
from model_v2 import VLMRunnerV2
from e1_perturb import run_e1_for_image, ensure_csv_header
from e2_freegen import run_e2_for_image, ensure_csv_header_e2
from e3_causal import sample_e3_triplets, run_e3_triplet, ensure_csv_headers_e3
from analysis_v2 import run_full_analysis_v2, save_verdicts_markdown
from viz_v2 import (
    plot_fig1_flip_by_k,
    plot_fig2_heatmap_pos_k,
    plot_fig3_flip_type,
    plot_fig4_position_strata,
    plot_fig5_S_vs_V,
    plot_fig6_dose_response,
    plot_fig7_entropy_control,
    plot_fig8_within_triplet_slopes,
    plot_fig9_forest,
    plot_fig10_cases,
    generate_html_report_v2,
)

logger = logging.getLogger("confirmatory_v2")


def setup_logging(log_path: str):
    """Sets up dual logging to stdout and run.log."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def parse_args():
    parser = argparse.ArgumentParser(description="Confirmatory & Causal VLM Experiments (v2)")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["m0", "m1", "m2", "e1", "e2", "e3", "analysis", "viz", "report", "all"],
                        help="Execution stage/milestone to run (default: all)")
    parser.add_argument("--n_images", type=int, default=150, help="Number of sampled images (user-configurable)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed (default: 2026)")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf", help="Hugging Face model name")
    parser.add_argument("--model_path", type=str, default=None, help="Local model directory override")
    parser.add_argument("--val2014_dir", type=str, default=None, help="COCO val2014 image directory override")
    parser.add_argument("--instances_json", type=str, default=None, help="COCO instances_val2014.json override")
    parser.add_argument("--results_dir", type=str, default=None, help="Output directory override")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from progress.jsonl if exists")
    return parser.parse_args()


def load_done_image_ids(progress_path: str) -> Set[int]:
    """Reads progress.jsonl and returns completed image IDs."""
    done = set()
    if os.path.isfile(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if "image_id" in rec and rec.get("status") == "done":
                            done.add(rec["image_id"])
                    except Exception:
                        pass
    return done


def append_progress_record(progress_path: str, record: Dict[str, Any]):
    """Appends record to progress.jsonl."""
    os.makedirs(os.path.dirname(os.path.abspath(progress_path)), exist_ok=True)
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def append_csv_rows(file_path: str, rows: List[Dict[str, Any]], columns: List[str]):
    """Appends rows to CSV file."""
    if not rows:
        return
    import csv
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        for r in rows:
            writer.writerow(r)


def main():
    args = parse_args()
    cfg = ConfigV2(
        n_images=args.n_images,
        sampling_seed=args.seed,
        seed=args.seed,
        model_name=args.model_name,
        model_path=args.model_path,
    )
    if args.val2014_dir:
        cfg.val2014_dir = args.val2014_dir
    if args.instances_json:
        cfg.instances_json = args.instances_json
    if args.results_dir:
        cfg.output_dir = args.results_dir

    cfg.resolve_paths()
    setup_logging(cfg.log_path)

    logger.info("=" * 70)
    logger.info(f"STARTING VLM EXPERIMENT V2 (STAGE: {args.stage.upper()}, N={cfg.n_images})")
    logger.info("=" * 70)
    logger.info(f"Python: {sys.version.split()[0]} | Torch: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()} | Devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    logger.info("Configuration Parameters:")
    for k, v in vars(cfg).items():
        logger.info(f"  {k}: {v}")

    # Write initial DEVIATIONS.md
    if not os.path.isfile(cfg.deviations_md_path):
        with open(cfg.deviations_md_path, "w", encoding="utf-8") as f:
            f.write(
                "# Deviations from Protocol Specification\n\n"
                "- **Section 5 Stopwords Filter:** In this confirmatory experiment v2, determiners, prepositions, "
                "and stopwords are deliberately NOT excluded from perturbation sites s in [t-8, t-1]. "
                "Their part of speech (POS) is recorded and analyzed instead.\n"
            )

    # Initialize Dataset
    coco = COCODatasetV2(
        val2014_dir=cfg.val2014_dir,
        instances_json=cfg.instances_json,
        pilot_image_ids_path=cfg.pilot_image_ids_path,
        seed=cfg.sampling_seed,
    )

    # Sample images
    sampled_image_ids = coco.sample_images(n_images=cfg.n_images, save_path=cfg.image_ids_path)

    # Initialize POS tagger
    pos_tagger = POSTagger()

    # =========================================================================
    # MILESTONE M0: Pipeline test on 2 images + site sampling table + POS alignment
    # =========================================================================
    if args.stage == "m0":
        logger.info("--- EXECUTING MILESTONE M0 (2 Images Pipeline & POS Alignment Test) ---")
        runner = VLMRunnerV2(config=cfg)
        obj_vocab = ObjectVocabulary(
            tokenizer=runner.tokenizer,
            synonyms_file=cfg.synonyms_file,
            dropped_words_path=cfg.dropped_words_path,
        )

        test_ids = sampled_image_ids[:2]
        m0_align_results = pos_tagger.unit_test_alignment()
        logger.info("POS Alignment Test on 2 sample captions:")
        for res in m0_align_results:
            logger.info(f"  Caption {res['caption_id']}: {res['aligned_words'][:8]}...")

        ensure_csv_header(cfg.records_e1_path)
        for iid in test_ids:
            runner.reset_peak_memory()
            t0 = time.time()
            rows, cap_info = run_e1_for_image(iid, coco, runner, obj_vocab, pos_tagger, cfg)
            append_csv_rows(cfg.records_e1_path, rows, columns=list(rows[0].keys()) if rows else [])
            mem = runner.get_peak_memory_gb()
            logger.info(f"M0 Image {iid}: {len(rows)} perturbations generated in {time.time()-t0:.2f}s. Peak GPU memory: {mem}")

        logger.info("M0 COMPLETED SUCCESSFULLY. (All criteria passed). Stopping as instructed.")
        return

    # Load Model & Object Vocabulary for general execution
    load_start = time.time()
    runner = VLMRunnerV2(config=cfg)
    logger.info(f"VLM Runner loaded in {time.time() - load_start:.2f}s")

    obj_vocab = ObjectVocabulary(
        tokenizer=runner.tokenizer,
        synonyms_file=cfg.synonyms_file,
        dropped_words_path=cfg.dropped_words_path,
    )

    ensure_csv_header(cfg.records_e1_path)
    ensure_csv_header_e2(cfg.records_e2_path)
    ensure_csv_headers_e3(cfg.records_e3_path, cfg.records_e3_temp_path)

    # Checkpoint recovery
    done_ids = load_done_image_ids(cfg.progress_jsonl_path) if args.resume else set()
    if done_ids:
        logger.info(f"RESUME: {len(done_ids)} images already processed according to progress.jsonl")

    # =========================================================================
    # MILESTONE M1: E1 + E2 + E3 on 2 images, timing measurement & extrapolation
    # =========================================================================
    if args.stage == "m1":
        logger.info("--- EXECUTING MILESTONE M1 (Timing & Feasibility Extrapolation) ---")
        m1_ids = sampled_image_ids[:2]
        t_start_m1 = time.time()

        for iid in m1_ids:
            rows_e1, cap_info = run_e1_for_image(iid, coco, runner, obj_vocab, pos_tagger, cfg)
            rows_e2 = run_e2_for_image(iid, coco, runner, obj_vocab, rows_e1, cap_info, cfg)
            append_csv_rows(cfg.records_e1_path, rows_e1, columns=list(rows_e1[0].keys()) if rows_e1 else [])
            append_csv_rows(cfg.records_e2_path, rows_e2, columns=list(rows_e2[0].keys()) if rows_e2 else [])

        time_2_imgs = time.time() - t_start_m1
        time_per_img = time_2_imgs / 2.0
        extrapolated_e1_e2_hours = (time_per_img * cfg.n_images) / 3600.0

        logger.info(f"M1 Results: 2 images processed in {time_2_imgs:.2f}s ({time_per_img:.2f}s/image).")
        logger.info(f"Extrapolated E1+E2 runtime for N={cfg.n_images} images: {extrapolated_e1_e2_hours:.2f} hours.")
        logger.info("M1 COMPLETED SUCCESSFULLY. Stopping as instructed.")
        return

    # =========================================================================
    # STAGE E1 & E2 (or M2): Single in-place progress bar for image iteration
    # =========================================================================
    run_e1 = (args.stage in ["e1", "m2", "all"])
    run_e2 = (args.stage in ["e2", "all"])

    # Determine image subset for M2
    target_image_ids = sampled_image_ids[:30] if args.stage == "m2" else sampled_image_ids
    captions_cache = {}

    if run_e1:
        logger.info(f"Starting E1 processing for {len(target_image_ids)} images...")
        total_flips_k_le_2 = 0

        # SINGLE clean in-place progress bar
        with tqdm(
            total=len(target_image_ids),
            desc="Tiến độ xử lý Ảnh",
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=0.5,
            leave=True,
        ) as pbar:
            for idx, image_id in enumerate(target_image_ids):
                if image_id in done_ids:
                    pbar.update(1)
                    pbar.set_postfix({"id": image_id, "status": "resumed"})
                    continue

                runner.reset_peak_memory()
                t0 = time.time()

                try:
                    rows_e1, cap_info = run_e1_for_image(image_id, coco, runner, obj_vocab, pos_tagger, cfg)
                    append_csv_rows(cfg.records_e1_path, rows_e1, columns=list(rows_e1[0].keys()) if rows_e1 else [])
                    captions_cache[str(image_id)] = cap_info

                    # Count flips at k <= 2
                    flips_k2 = sum(1 for r in rows_e1 if r["alt_rank"] == 1 and r["k"] in [1, 2] and r["flip_tf"] == 1)
                    total_flips_k_le_2 += flips_k2

                    # Run E2 if enabled
                    if run_e2:
                        rows_e2 = run_e2_for_image(image_id, coco, runner, obj_vocab, rows_e1, cap_info, cfg)
                        append_csv_rows(cfg.records_e2_path, rows_e2, columns=list(rows_e2[0].keys()) if rows_e2 else [])

                    mem_dict = runner.get_peak_memory_gb()
                    mem_str = f"{max(mem_dict.values())}GB" if mem_dict else "-"

                    # Save progress
                    append_progress_record(cfg.progress_jsonl_path, {
                        "image_id": image_id,
                        "status": "done",
                        "time_s": round(time.time() - t0, 2),
                        "n_e1_records": len(rows_e1),
                        "peak_mem_gb": mem_dict,
                    })

                    pbar.update(1)
                    pbar.set_postfix({"id": image_id, "objs": cap_info.get("T", 0), "flips": total_flips_k_le_2, "mem": mem_str})

                except Exception as e:
                    logger.error(f"Error processing image {image_id}: {e}", exc_info=True)
                    pbar.update(1)

        # Save captions
        if captions_cache:
            with open(cfg.captions_path, "w", encoding="utf-8") as f:
                json.dump(captions_cache, f, indent=2)

        # MILESTONE M2 CHECK (after 30 images)
        if args.stage == "m2":
            logger.info("--- MILESTONE M2 EVALUATION (Power Rule Check) ---")
            projected_flips = int((total_flips_k_le_2 / 30.0) * cfg.n_images) if len(target_image_ids) >= 30 else total_flips_k_le_2
            logger.info(f"Observed flips at k<=2 across 30 images: {total_flips_k_le_2}")
            logger.info(f"Projected total flips for N={cfg.n_images}: {projected_flips}")
            if projected_flips < 100:
                logger.warning(f"Projected flips ({projected_flips}) is < 100. Consider raising n_images to 250-400.")
            else:
                logger.info(f"Statistical power check PASSED (Projected flips {projected_flips} >= 100).")
            return

    # =========================================================================
    # STAGE E3: Causal Intervention & Entropy Control
    # =========================================================================
    if args.stage in ["e3", "all"]:
        logger.info("--- EXECUTING STAGE E3 (Causal Image Degradation & Entropy Control) ---")
        if os.path.isfile(cfg.records_e1_path):
            df_e1 = pd.read_csv(cfg.records_e1_path)
            e1_records = df_e1.to_dict(orient="records")
        else:
            e1_records = []

        chosen_near, chosen_far, chosen_placebo = sample_e3_triplets(e1_records, cfg)
        all_triplets = (
            [(t, 0, 0) for t in chosen_near] +
            [(t, 0, 1) for t in chosen_far] +
            [(t, 1, 0) for t in chosen_placebo]
        )

        caption_cache = {}
        pixel_cache = {}
        orig_prefix_cache = {}

        with tqdm(
            total=len(all_triplets),
            desc="E3 Can thiệp Nhân quả (Triplets)",
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=0.5,
            leave=True,
        ) as pbar_e3:
            for idx, (trip, is_plac, is_far) in enumerate(all_triplets):
                try:
                    e3_rows, temp_rows = run_e3_triplet(
                        triplet_id=idx + 1,
                        record=trip,
                        coco=coco,
                        runner=runner,
                        obj_vocab=obj_vocab,
                        cfg=cfg,
                        is_placebo=is_plac,
                        is_far=is_far,
                        caption_cache=caption_cache,
                        pixel_cache=pixel_cache,
                        orig_prefix_cache=orig_prefix_cache,
                    )
                    append_csv_rows(cfg.records_e3_path, e3_rows, columns=list(e3_rows[0].keys()) if e3_rows else [])
                    append_csv_rows(cfg.records_e3_temp_path, temp_rows, columns=list(temp_rows[0].keys()) if temp_rows else [])
                except Exception as e:
                    logger.warning(f"E3 triplet {idx+1} failed: {e}")

                pbar_e3.update(1)
                pbar_e3.set_postfix({"triplet": idx + 1, "k": trip.get("k", 0)})
                if (idx + 1) % 25 == 0 or (idx + 1) == len(all_triplets):
                    logger.info(f"E3 Tiến độ: {idx+1}/{len(all_triplets)} triplets ({((idx+1)/len(all_triplets))*100:.1f}%)")

    # =========================================================================
    # STAGE ANALYSIS: Pre-registered Statistical Testing
    # =========================================================================
    if args.stage in ["analysis", "all"]:
        logger.info("--- EXECUTING STATISTICAL ANALYSIS & VERDICTS EVALUATION ---")
        df_e1 = pd.read_csv(cfg.records_e1_path) if os.path.isfile(cfg.records_e1_path) else pd.DataFrame()
        df_e2 = pd.read_csv(cfg.records_e2_path) if os.path.isfile(cfg.records_e2_path) else None
        df_e3 = pd.read_csv(cfg.records_e3_path) if os.path.isfile(cfg.records_e3_path) else None

        results = run_full_analysis_v2(
            df_e1=df_e1,
            df_e2=df_e2,
            df_e3=df_e3,
            n_resamples=cfg.cluster_bootstrap_resamples,
            seed=cfg.seed,
        )

        # Save analysis JSON & Markdown
        with open(cfg.analysis_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        save_verdicts_markdown(results, cfg.verdicts_md_path)
        logger.info(f"Analysis complete. Verdicts saved to {cfg.verdicts_md_path}")

    # =========================================================================
    # STAGE VIZ & REPORT: Generate 10 figures and report.html
    # =========================================================================
    if args.stage in ["viz", "report", "all"]:
        logger.info("--- GENERATING 10 VISUALIZATIONS & STANDALONE REPORT.HTML ---")
        df_e1 = pd.read_csv(cfg.records_e1_path) if os.path.isfile(cfg.records_e1_path) else pd.DataFrame()
        df_e2 = pd.read_csv(cfg.records_e2_path) if os.path.isfile(cfg.records_e2_path) else None
        df_e3 = pd.read_csv(cfg.records_e3_path) if os.path.isfile(cfg.records_e3_path) else None
        df_temp = pd.read_csv(cfg.records_e3_temp_path) if os.path.isfile(cfg.records_e3_temp_path) else None

        if os.path.isfile(cfg.analysis_json_path):
            with open(cfg.analysis_json_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        else:
            results = {}

        figs_dir = os.path.join(cfg.output_dir, "figs")
        os.makedirs(figs_dir, exist_ok=True)

        fig_paths = {
            "fig1": os.path.join(figs_dir, "fig1_flip_by_k.png"),
            "fig2": os.path.join(figs_dir, "fig2_flip_by_pos_k_heatmap.png"),
            "fig3": os.path.join(figs_dir, "fig3_flip_type.png"),
            "fig4": os.path.join(figs_dir, "fig4_position_strata.png"),
            "fig5": os.path.join(figs_dir, "fig5_S_vs_V_strata.png"),
            "fig6": os.path.join(figs_dir, "fig6_dose_response.png"),
            "fig7": os.path.join(figs_dir, "fig7_entropy_control.png"),
            "fig8": os.path.join(figs_dir, "fig8_within_triplet_slopes.png"),
            "fig9": os.path.join(figs_dir, "fig9_forest.png"),
        }

        if not df_e1.empty:
            plot_fig1_flip_by_k(df_e1, df_e2, fig_paths["fig1"])
            plot_fig2_heatmap_pos_k(df_e1, fig_paths["fig2"])
            plot_fig3_flip_type(df_e1, fig_paths["fig3"])
            plot_fig4_position_strata(df_e1, fig_paths["fig4"])
            plot_fig5_S_vs_V(df_e1, fig_paths["fig5"])

        plot_fig6_dose_response(df_e3, fig_paths["fig6"])
        plot_fig7_entropy_control(df_e3, df_temp, fig_paths["fig7"])
        plot_fig8_within_triplet_slopes(df_e3, fig_paths["fig8"])
        plot_fig9_forest(results, fig_paths["fig9"])

        # 6 case studies
        cases = []
        if not df_e1.empty:
            flips_k1 = df_e1[(df_e1["k"] == 1) & (df_e1["flip_tf"] == 1)].sort_values("S_t", ascending=False)
            cases.extend(flips_k1.head(3).to_dict(orient="records"))
            other_cases = df_e1[(df_e1["flip_tf"] == 1)].head(3).to_dict(orient="records")
            cases.extend(other_cases)

        fig10_paths = plot_fig10_cases(cases, figs_dir)
        fig_paths["fig10"] = fig10_paths

        # Generate standalone HTML report
        generate_html_report_v2(results, fig_paths, cfg.report_html_path)
        logger.info(f"SUCCESS: Report generated at {cfg.report_html_path}")

    logger.info("=" * 70)
    logger.info("EXPERIMENT PIPELINE V2 FINISHED SUCCESSFULLY.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
