"""
E1 Module: Confirmatory Stratified Teacher-Forced Prefix Perturbations.
Stratifies sites by distance k in {1,2,3} plus up to 2 in [4,8].
Tests alternatives at alt_rank=1 and alt_rank=2. Runs placebo control.
"""

import os
import csv
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from PIL import Image

import torch
import numpy as np

from config_v2 import ConfigV2
from data_v2 import COCODatasetV2
from objects import ObjectVocabulary
from pos_tagger import POSTagger
from model_v2 import VLMRunnerV2
from compute_v2 import (
    jsd_divergence,
    compute_restricted_distribution,
    compute_obj_mass,
    compute_entropy,
    compute_margin,
    format_top3_objects,
    find_candidate_perturbation_sites,
    select_stratified_sites,
    select_alternative_tokens_ranked,
    compute_sentence_and_word_distance,
)

logger = logging.getLogger("confirmatory_v2")

E1_COLUMNS = [
    "image_id", "t", "T", "rel_pos", "word", "category",
    "s", "k", "k_words", "same_sentence", "orig_token", "alt_token",
    "alt_rank", "p_alt", "pos_s", "V_black", "V_remove", "S_t",
    "flip_tf", "flip_type", "dlogp", "entropy", "margin", "confidence",
    "obj_mass", "salience", "in_gt", "is_repeat", "n_prev_objects",
    "top3_before", "top3_after", "is_placebo", "low_mass"
]


def ensure_csv_header(file_path: str):
    """Writes header if file does not exist."""
    if not os.path.isfile(file_path):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(E1_COLUMNS)


def run_e1_for_image(
    image_id: int,
    coco: COCODatasetV2,
    runner: VLMRunnerV2,
    obj_vocab: ObjectVocabulary,
    pos_tagger: POSTagger,
    cfg: ConfigV2,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes E1 pipeline for a single image:
    1. Generates caption.
    2. Identifies object positions.
    3. Caches base logits (image, black, remove) to results_v2/cache/<image_id>.npz.
    4. Stratifies sites k in {1,2,3} + up to 2 in [4,8].
    5. Runs ranked perturbations (alt_rank=1, alt_rank=2) and placebo.
    Returns:
        rows: List of record dicts.
        caption_info: Dict containing caption text, T, gen_ids.
    """
    pil_img = coco.load_image(image_id)
    special_ids = set(runner.tokenizer.all_special_ids)

    # 1. Greedy caption generation
    prompt_ids, caption_text, T, gen_ids = runner.generate_caption(pil_img)
    caption_info = {
        "image_id": image_id,
        "caption": caption_text,
        "T": T,
        "gen_ids": gen_ids[0].tolist(),
    }

    if T == 0:
        logger.warning(f"Image {image_id}: Generated empty caption.")
        return [], caption_info

    # 2. Get full logits with image and verify sanity check
    inputs_img = runner.processor(text=cfg.prompt, images=pil_img, return_tensors="pt")
    pixel_values = inputs_img["pixel_values"].to(runner.device)

    logits_img_all, match_rate = runner.get_logits(prompt_ids, gen_ids, pixel_values)
    if match_rate < 0.95:
        logger.warning(f"Image {image_id}: Sanity check match rate is {match_rate:.2%} (< 95%)")

    # 3. Compute baseline no-image logits (black and remove)
    logits_black_all = runner.get_no_image_logits(gen_ids, mode="black", image_size=pil_img.size)
    logits_remove_all = runner.get_no_image_logits(gen_ids, mode="remove")

    # 4. Identify object positions
    obj_positions = []
    for t in range(T):
        tok_id = int(gen_ids[0, t].item())
        if tok_id in obj_vocab.obj_ids_set:
            w = obj_vocab.get_word(tok_id)
            cat = obj_vocab.get_category(tok_id)
            obj_positions.append((t, tok_id, w, cat))

    obj_positions_set = {t for (t, _, _, _) in obj_positions}

    # 5. Cache restricted logits for this image
    cache_path = os.path.join(cfg.cache_dir, f"{image_id}.npz")
    cache_data = {}
    for (t, tok_id, _, _) in obj_positions:
        cache_data[f"p_img_{t}"] = compute_restricted_distribution(logits_img_all[t], obj_vocab.obj_ids)
        cache_data[f"p_black_{t}"] = compute_restricted_distribution(logits_black_all[t], obj_vocab.obj_ids)
        cache_data[f"p_remove_{t}"] = compute_restricted_distribution(logits_remove_all[t], obj_vocab.obj_ids)
    np.savez_compressed(cache_path, **cache_data)

    if not obj_positions:
        logger.info(f"Image {image_id}: No object tokens identified in caption.")
        return [], caption_info

    # 6. POS tagging for tokens in caption
    token_ids_list = gen_ids[0].tolist()
    token_pos_map = pos_tagger.tag_caption_tokens(caption_text, token_ids_list, runner.tokenizer)

    # 7. Iterate over objects and perturbation sites
    rows: List[Dict[str, Any]] = []
    seen_categories: List[str] = []

    for obj_idx, (t, obj_tok_id, obj_word, obj_cat) in enumerate(obj_positions):
        rel_pos = float(t / T) if T > 0 else 0.0

        p_img = cache_data[f"p_img_{t}"]
        p_black = cache_data[f"p_black_{t}"]
        p_remove = cache_data[f"p_remove_{t}"]

        v_black = jsd_divergence(p_img, p_black)
        v_remove = jsd_divergence(p_img, p_remove)

        entropy_val = compute_entropy(p_img)
        margin_val = compute_margin(p_img)
        confidence_val = float(np.max(p_img))
        obj_mass_val = compute_obj_mass(logits_img_all[t], obj_vocab.obj_ids)
        low_mass_flag = 1 if obj_mass_val < 0.01 else 0

        salience_val, in_gt_val = coco.get_salience_and_in_gt(image_id, obj_cat)

        is_repeat = 1 if (obj_cat in seen_categories) else 0
        n_prev_objects = len(seen_categories)
        seen_categories.append(obj_cat)

        top3_before_str = format_top3_objects(p_img, obj_vocab)
        orig_argmax_idx = int(np.argmax(p_img))

        # Find eligible candidate sites in [t-8, t-1]
        candidate_sites = find_candidate_perturbation_sites(
            t=t,
            gen_ids=gen_ids,
            obj_positions_set=obj_positions_set,
            obj_vocab=obj_vocab,
            special_ids=special_ids,
            tokenizer=runner.tokenizer,
        )

        if not candidate_sites:
            continue

        # Stratified site selection: all k in {1,2,3} + up to 2 in [4,8]
        chosen_sites = select_stratified_sites(
            candidate_sites=candidate_sites,
            t=t,
            image_id=image_id,
            seed_base=cfg.seed,
        )

        for site_idx, s in enumerate(chosen_sites):
            k = t - s
            k_words, same_sentence = compute_sentence_and_word_distance(s, t, gen_ids, runner.tokenizer)
            orig_tok_id = int(gen_ids[0, s].item())
            orig_tok_str = runner.tokenizer.decode([orig_tok_id]).strip()
            pos_s = token_pos_map.get(s, "UNK")

            # --- PLACEBO TEST (Mandatory on the first site of each object) ---
            if site_idx == 0:
                pert_prefix_placebo = gen_ids[:, :t].clone()
                # Replacing token by itself: y_s -> y_s
                logits_placebo = runner.get_perturbed_target_logits(
                    prompt_ids=prompt_ids,
                    gen_ids_perturbed_prefix=pert_prefix_placebo,
                    pixel_values=pixel_values,
                )
                p_placebo = compute_restricted_distribution(logits_placebo, obj_vocab.obj_ids)
                s_placebo = jsd_divergence(p_img, p_placebo)
                flip_placebo = 1 if (np.argmax(p_img) != np.argmax(p_placebo)) else 0

                if s_placebo >= 1e-6 or flip_placebo != 0:
                    logger.warning(
                        f"Image {image_id}, t={t}: PLACEBO failure! S_t={s_placebo:.6f}, flip={flip_placebo}"
                    )

                rows.append({
                    "image_id": image_id, "t": t, "T": T, "rel_pos": rel_pos,
                    "word": obj_word, "category": obj_cat, "s": s, "k": k,
                    "k_words": k_words, "same_sentence": same_sentence,
                    "orig_token": orig_tok_str, "alt_token": orig_tok_str,
                    "alt_rank": 0, "p_alt": 1.0, "pos_s": pos_s,
                    "V_black": v_black, "V_remove": v_remove, "S_t": s_placebo,
                    "flip_tf": flip_placebo, "flip_type": "none", "dlogp": 0.0,
                    "entropy": entropy_val, "margin": margin_val, "confidence": confidence_val,
                    "obj_mass": obj_mass_val, "salience": salience_val, "in_gt": in_gt_val,
                    "is_repeat": is_repeat, "n_prev_objects": n_prev_objects,
                    "top3_before": top3_before_str, "top3_after": top3_before_str,
                    "is_placebo": 1, "low_mass": low_mass_flag,
                })

            # --- REAL PERTURBATIONS: alt_rank=1 and alt_rank=2 ---
            alt_dict = select_alternative_tokens_ranked(
                full_logits_at_s=logits_img_all[s],
                orig_tok_id=orig_tok_id,
                obj_vocab=obj_vocab,
                tokenizer=runner.tokenizer,
                special_ids=special_ids,
                min_prob_threshold=0.01,
            )

            for rank in [1, 2]:
                if rank not in alt_dict:
                    continue

                alt_info = alt_dict[rank]
                alt_tok_id = alt_info["token_id"]
                alt_tok_str = alt_info["token_str"]
                p_alt = alt_info["p_alt"]

                # Build perturbed prefix up to t
                pert_prefix = gen_ids[:, :t].clone()
                pert_prefix[0, s] = alt_tok_id

                logits_pert = runner.get_perturbed_target_logits(
                    prompt_ids=prompt_ids,
                    gen_ids_perturbed_prefix=pert_prefix,
                    pixel_values=pixel_values,
                )
                p_pert = compute_restricted_distribution(logits_pert, obj_vocab.obj_ids)
                s_t = jsd_divergence(p_img, p_pert)

                pert_argmax_idx = int(np.argmax(p_pert))
                flip_tf = 1 if (orig_argmax_idx != pert_argmax_idx) else 0

                # Determine flip type
                if flip_tf == 1:
                    orig_pred_cat = obj_vocab.get_category(obj_vocab.obj_ids[orig_argmax_idx])
                    pert_pred_cat = obj_vocab.get_category(obj_vocab.obj_ids[pert_argmax_idx])
                    flip_type = "same_category" if (orig_pred_cat == pert_pred_cat) else "cross_category"
                else:
                    flip_type = "none"

                # dlogp = log p_orig(o*) - log p_pert(o*)
                p_orig_star = max(float(p_img[orig_argmax_idx]), 1e-12)
                p_pert_star = max(float(p_pert[orig_argmax_idx]), 1e-12)
                dlogp_val = float(np.log(p_orig_star) - np.log(p_pert_star))

                top3_after_str = format_top3_objects(p_pert, obj_vocab)

                rows.append({
                    "image_id": image_id, "t": t, "T": T, "rel_pos": rel_pos,
                    "word": obj_word, "category": obj_cat, "s": s, "k": k,
                    "k_words": k_words, "same_sentence": same_sentence,
                    "orig_token": orig_tok_str, "alt_token": alt_tok_str,
                    "alt_rank": rank, "p_alt": p_alt, "pos_s": pos_s,
                    "V_black": v_black, "V_remove": v_remove, "S_t": s_t,
                    "flip_tf": flip_tf, "flip_type": flip_type, "dlogp": dlogp_val,
                    "entropy": entropy_val, "margin": margin_val, "confidence": confidence_val,
                    "obj_mass": obj_mass_val, "salience": salience_val, "in_gt": in_gt_val,
                    "is_repeat": is_repeat, "n_prev_objects": n_prev_objects,
                    "top3_before": top3_before_str, "top3_after": top3_after_str,
                    "is_placebo": 0, "low_mass": low_mass_flag,
                })

    return rows, caption_info
