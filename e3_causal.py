"""
E3 Module: Causal Intervention on Image Content + Entropy Control.
Blends original image with black image (lambda in [1.0, 0.75, 0.5, 0.25, 0.0]).
Implements temperature-scaling entropy control curve S_ref(H) and excess sensitivity.
"""

import os
import csv
import random
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from PIL import Image

import torch
import numpy as np

from config_v2 import ConfigV2
from data_v2 import COCODatasetV2
from objects import ObjectVocabulary
from model_v2 import VLMRunnerV2, blend_image_raw
from compute_v2 import (
    jsd_divergence,
    compute_restricted_distribution,
    compute_entropy,
    compute_margin,
)

logger = logging.getLogger("confirmatory_v2")

E3_COLUMNS = [
    "triplet_id", "image_id", "t", "s", "k", "word", "category",
    "lambda_val", "V_lambda", "S_lambda", "H_lambda", "margin_lambda",
    "dlogp_lambda", "flip_lambda", "S_ref_lambda", "excess_lambda",
    "out_of_range", "is_placebo", "is_far"
]

E3_TEMP_COLUMNS = [
    "triplet_id", "image_id", "t", "s", "k",
    "tau", "S_tau", "H_tau", "margin_tau"
]


def ensure_csv_headers_e3(e3_path: str, e3_temp_path: str):
    """Ensures CSV headers exist for E3 files."""
    if not os.path.isfile(e3_path):
        os.makedirs(os.path.dirname(os.path.abspath(e3_path)), exist_ok=True)
        with open(e3_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(E3_COLUMNS)

    if not os.path.isfile(e3_temp_path):
        os.makedirs(os.path.dirname(os.path.abspath(e3_temp_path)), exist_ok=True)
        with open(e3_temp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(E3_TEMP_COLUMNS)


def sample_e3_triplets(
    all_e1_records: List[Dict[str, Any]],
    cfg: ConfigV2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Samples triplets from E1 rows with low_mass=0, alt_rank=1, is_placebo=0:
    - up to e3_n_near (400) triplets with k in {1,2}
    - up to e3_n_far (100) triplets with k >= 4 (negative control)
    - 50 placebo triplets for sanity check.
    """
    valid_near = [r for r in all_e1_records if r.get("low_mass", 0) == 0 and r.get("alt_rank", 0) == 1 and r.get("is_placebo", 0) == 0 and r.get("k", 0) in {1, 2}]
    valid_far = [r for r in all_e1_records if r.get("low_mass", 0) == 0 and r.get("alt_rank", 0) == 1 and r.get("is_placebo", 0) == 0 and r.get("k", 0) >= 4]

    rng = random.Random(cfg.seed)
    chosen_near = rng.sample(valid_near, min(len(valid_near), cfg.e3_n_near)) if valid_near else []
    chosen_far = rng.sample(valid_far, min(len(valid_far), cfg.e3_n_far)) if valid_far else []

    # 50 placebo triplets chosen from near
    chosen_placebo = rng.sample(chosen_near, min(len(chosen_near), 50)) if chosen_near else []

    logger.info(
        f"Sampled E3 triplets: {len(chosen_near)} near (k in {{1,2}}), "
        f"{len(chosen_far)} far (k >= 4), {len(chosen_placebo)} placebo."
    )
    return chosen_near, chosen_far, chosen_placebo


def run_e3_triplet(
    triplet_id: int,
    record: Dict[str, Any],
    coco: COCODatasetV2,
    runner: VLMRunnerV2,
    obj_vocab: ObjectVocabulary,
    cfg: ConfigV2,
    is_placebo: int = 0,
    is_far: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Runs E3 causal degradation and temperature scaling on a single triplet:
    1. Evaluates all lambdas [1.0, 0.75, 0.5, 0.25, 0.0] on raw blended images.
    2. Computes S_ref curve via temperature scaling tau in [1, 1.25, 1.5, 2, 3, 4] at lambda=1.
    3. Interpolates excess sensitivity.
    """
    image_id = record["image_id"]
    t = record["t"]
    s = record["s"]
    k = record["k"]
    word = record["word"]
    category = record["category"]

    orig_img = coco.load_image(image_id)

    # 1. Reconstruct prompt and prefix
    inputs_text = runner.processor(text=cfg.prompt, images=orig_img, return_tensors="pt")
    prompt_ids = inputs_text["input_ids"].to(runner.device)

    # Load caption gen_ids from cache or captions_data
    cache_path = os.path.join(cfg.cache_dir, f"{image_id}.npz")
    # Prefix up to t
    orig_tok_str = record["orig_token"]
    alt_tok_str = orig_tok_str if (is_placebo == 1) else record["alt_token"]

    alt_tok_id = runner.tokenizer.convert_tokens_to_ids("\u2581" + alt_tok_str)
    if alt_tok_id == runner.tokenizer.unk_token_id or alt_tok_id is None:
        alt_tok_id = runner.tokenizer.convert_tokens_to_ids(" " + alt_tok_str)
    if alt_tok_id == runner.tokenizer.unk_token_id or alt_tok_id is None:
        enc = runner.tokenizer.encode(" " + alt_tok_str, add_special_tokens=False)
        alt_tok_id = enc[0] if enc else 0

    # Reconstruct prefix up to t
    # From cache or running greedy caption
    prompt_ids, _, _, gen_ids = runner.generate_caption(orig_img)

    orig_prefix_up_to_t = gen_ids[:, :t].clone()
    pert_prefix_up_to_t = gen_ids[:, :t].clone()
    pert_prefix_up_to_t[0, s] = alt_tok_id

    # Store distributions across lambdas
    lambda_p_orig: Dict[float, np.ndarray] = {}
    lambda_p_pert: Dict[float, np.ndarray] = {}
    lambda_logits_orig: Dict[float, torch.Tensor] = {}
    lambda_logits_pert: Dict[float, torch.Tensor] = {}

    for l_val in cfg.lambdas:
        blended_img = blend_image_raw(orig_img, l_val)
        inputs_blend = runner.processor(text=cfg.prompt, images=blended_img, return_tensors="pt")
        pix_vals = inputs_blend["pixel_values"].to(runner.device)

        # Forward pass on original prefix
        logits_o = runner.get_perturbed_target_logits(
            prompt_ids=prompt_ids,
            gen_ids_perturbed_prefix=orig_prefix_up_to_t,
            pixel_values=pix_vals,
        )
        # Forward pass on perturbed prefix
        logits_p = runner.get_perturbed_target_logits(
            prompt_ids=prompt_ids,
            gen_ids_perturbed_prefix=pert_prefix_up_to_t,
            pixel_values=pix_vals,
        )

        p_o = compute_restricted_distribution(logits_o, obj_vocab.obj_ids)
        p_p = compute_restricted_distribution(logits_p, obj_vocab.obj_ids)

        lambda_p_orig[l_val] = p_o
        lambda_p_pert[l_val] = p_p
        lambda_logits_orig[l_val] = logits_o
        lambda_logits_pert[l_val] = logits_p

    # p_0 is distribution at lambda=0.0 (pure black image)
    p_0 = lambda_p_orig[0.0]
    p_1_orig = lambda_p_orig[1.0]
    o_star_idx = int(np.argmax(p_1_orig))  # o* fixed as argmax at lambda=1 on original prefix

    # --- Temperature Control at lambda=1.0 ---
    temp_rows: List[Dict[str, Any]] = []
    tau_curve: List[Tuple[float, float]] = []  # (H, S)

    logits_o_1 = lambda_logits_orig[1.0].to(torch.float32)[obj_vocab.obj_ids]
    logits_p_1 = lambda_logits_pert[1.0].to(torch.float32)[obj_vocab.obj_ids]

    for tau in cfg.temperatures:
        p_tau_o = torch.softmax(logits_o_1 / tau, dim=-1).cpu().numpy().astype(np.float64)
        p_tau_p = torch.softmax(logits_p_1 / tau, dim=-1).cpu().numpy().astype(np.float64)

        s_tau = jsd_divergence(p_tau_o, p_tau_p)
        h_tau = compute_entropy(p_tau_o)
        m_tau = compute_margin(p_tau_o)

        temp_rows.append({
            "triplet_id": triplet_id, "image_id": image_id, "t": t, "s": s, "k": k,
            "tau": tau, "S_tau": s_tau, "H_tau": h_tau, "margin_tau": m_tau,
        })
        tau_curve.append((h_tau, s_tau))

    # Sort tau_curve by H for interpolation
    tau_curve.sort(key=lambda x: x[0])
    h_vals = [pt[0] for pt in tau_curve]
    s_vals = [pt[1] for pt in tau_curve]

    # --- Compute E3 metrics for each lambda ---
    e3_rows: List[Dict[str, Any]] = []

    for l_val in cfg.lambdas:
        p_o = lambda_p_orig[l_val]
        p_p = lambda_p_pert[l_val]

        v_l = jsd_divergence(p_o, p_0)
        s_l = jsd_divergence(p_o, p_p)
        h_l = compute_entropy(p_o)
        margin_l = compute_margin(p_o)

        # dlogp(lambda) with o* fixed from lambda=1
        p_o_star = max(float(p_o[o_star_idx]), 1e-12)
        p_p_star = max(float(p_p[o_star_idx]), 1e-12)
        dlogp_l = float(np.log(p_o_star) - np.log(p_p_star))

        flip_l = 1 if (np.argmax(p_o) != np.argmax(p_p)) else 0

        # Interpolate S_ref at H(lambda)
        out_of_range = 0
        if h_l < h_vals[0]:
            s_ref = s_vals[0]
            out_of_range = 1
        elif h_l > h_vals[-1]:
            s_ref = s_vals[-1]
            out_of_range = 1
        else:
            s_ref = float(np.interp(h_l, h_vals, s_vals))

        excess_l = s_l - s_ref

        e3_rows.append({
            "triplet_id": triplet_id, "image_id": image_id, "t": t, "s": s, "k": k,
            "word": word, "category": category, "lambda_val": l_val,
            "V_lambda": v_l, "S_lambda": s_l, "H_lambda": h_l, "margin_lambda": margin_l,
            "dlogp_lambda": dlogp_l, "flip_lambda": flip_l,
            "S_ref_lambda": s_ref, "excess_lambda": excess_l,
            "out_of_range": out_of_range, "is_placebo": is_placebo, "is_far": is_far,
        })

    return e3_rows, temp_rows
