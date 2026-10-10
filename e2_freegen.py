"""
E2 Module: Free-Generation Check ("Generate until the next object").
Tests whether objects actually change when the model continues generating on its own.
Evaluates alt_rank=1 candidates and placebo-free control runs.
"""

import os
import csv
import random
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from PIL import Image

import torch

from config_v2 import ConfigV2
from data_v2 import COCODatasetV2
from objects import ObjectVocabulary
from model_v2 import VLMRunnerV2

logger = logging.getLogger("confirmatory_v2")

E2_COLUMNS = [
    "image_id", "t", "s", "alt_rank", "k", "is_placebo",
    "orig_word", "orig_cat", "alt_token",
    "first_obj_word_free", "first_obj_cat_free",
    "flip_free_word", "flip_free_cat", "no_obj_free",
    "delta_len", "num_gen_tokens"
]


def ensure_csv_header_e2(file_path: str):
    """Writes header if file does not exist."""
    if not os.path.isfile(file_path):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(E2_COLUMNS)


def run_e2_for_image(
    image_id: int,
    coco: COCODatasetV2,
    runner: VLMRunnerV2,
    obj_vocab: ObjectVocabulary,
    e1_records_for_image: List[Dict[str, Any]],
    caption_info: Dict[str, Any],
    cfg: ConfigV2,
) -> List[Dict[str, Any]]:
    """
    Selects up to e2_max_per_image perturbations with alt_rank=1 and placebo runs,
    executes free generation until next object, and records outcome.
    """
    if not e1_records_for_image:
        return []

    pil_img = coco.load_image(image_id)
    inputs_img = runner.processor(text=cfg.prompt, images=pil_img, return_tensors="pt")
    prompt_ids = inputs_img["input_ids"].to(runner.device)
    pixel_values = inputs_img["pixel_values"].to(runner.device)

    gen_ids_list = caption_info["gen_ids"]
    gen_ids = torch.tensor([gen_ids_list], dtype=torch.long, device=runner.device)

    # 1. Filter eligible candidates: alt_rank == 1 and not low_mass
    near_candidates = [r for r in e1_records_for_image if r["alt_rank"] == 1 and r["k"] in {1, 2} and r["low_mass"] == 0]
    far_candidates = [r for r in e1_records_for_image if r["alt_rank"] == 1 and r["k"] >= 4 and r["low_mass"] == 0]
    placebo_candidates = [r for r in e1_records_for_image if r["is_placebo"] == 1 and r["low_mass"] == 0]

    # Select candidates per image according to rules:
    # All with k in {1,2} first (random order), then up to 3 with k >= 4, plus placebo up to 2 objects.
    rng = random.Random(cfg.seed + image_id)
    rng.shuffle(near_candidates)

    selected_far = rng.sample(far_candidates, min(len(far_candidates), 3)) if far_candidates else []
    selected_placebo = rng.sample(placebo_candidates, min(len(placebo_candidates), 2)) if placebo_candidates else []

    selected_runs = (near_candidates + selected_far)[:cfg.e2_max_per_image] + selected_placebo

    rows: List[Dict[str, Any]] = []

    for cand in selected_runs:
        t = cand["t"]
        s = cand["s"]
        k = cand["k"]
        orig_word = cand["word"]
        orig_cat = cand["category"]
        is_placebo = cand["is_placebo"]

        if is_placebo == 1:
            alt_tok_id = int(gen_ids[0, s].item())
            alt_tok_str = cand["orig_token"]
            alt_rank = 0
        else:
            alt_tok_str = cand["alt_token"]
            alt_rank = cand["alt_rank"]
            # Look up token id of alt_token
            alt_tok_id = runner.tokenizer.convert_tokens_to_ids("\u2581" + alt_tok_str)
            if alt_tok_id == runner.tokenizer.unk_token_id or alt_tok_id is None:
                alt_tok_id = runner.tokenizer.convert_tokens_to_ids(" " + alt_tok_str)
            if alt_tok_id == runner.tokenizer.unk_token_id or alt_tok_id is None:
                encoded = runner.tokenizer.encode(" " + alt_tok_str, add_special_tokens=False)
                alt_tok_id = encoded[0] if encoded else int(gen_ids[0, s].item())

        gen_ids_up_to_s = gen_ids[:, :s]

        # Free generation with custom stopping criteria
        new_tokens = runner.free_generate_until_object(
            prompt_ids=prompt_ids,
            gen_ids_up_to_s=gen_ids_up_to_s,
            alt_token_id=alt_tok_id,
            pixel_values=pixel_values,
            obj_ids_set=obj_vocab.obj_ids_set,
            max_new_tokens=cfg.e2_max_new_tokens,
        )

        num_gen = len(new_tokens)

        # Find first emitted object in generated tokens
        first_obj_id = None
        for tok in new_tokens:
            if tok in obj_vocab.obj_ids_set:
                first_obj_id = tok
                break

        if first_obj_id is not None:
            first_obj_word = obj_vocab.get_word(first_obj_id)
            first_obj_cat = obj_vocab.get_category(first_obj_id)
            no_obj_free = 0
            flip_free_word = 1 if (first_obj_word != orig_word) else 0
            flip_free_cat = 1 if (first_obj_cat != orig_cat) else 0
        else:
            first_obj_word = None
            first_obj_cat = None
            no_obj_free = 1
            flip_free_word = 1
            flip_free_cat = 1

        # delta_len = number of tokens generated before the object minus (t - s - 1)
        expected_tokens_before_obj = t - s - 1
        # Tokens generated before the object
        if first_obj_id is not None:
            tokens_before = new_tokens.index(first_obj_id)
        else:
            tokens_before = num_gen
        delta_len = tokens_before - expected_tokens_before_obj

        # PLACEBO SANITY CHECK
        if is_placebo == 1:
            if flip_free_word == 1:
                logger.warning(
                    f"Image {image_id}, t={t}: Placebo-free produced word flip: {orig_word} -> {first_obj_word}"
                )

        rows.append({
            "image_id": image_id, "t": t, "s": s, "alt_rank": alt_rank, "k": k,
            "is_placebo": is_placebo, "orig_word": orig_word, "orig_cat": orig_cat,
            "alt_token": alt_tok_str, "first_obj_word_free": first_obj_word,
            "first_obj_cat_free": first_obj_cat, "flip_free_word": flip_free_word,
            "flip_free_cat": flip_free_cat, "no_obj_free": no_obj_free,
            "delta_len": delta_len, "num_gen_tokens": num_gen,
        })

    return rows
