"""
Mathematical and perturbation computation utilities for Confirmatory and Causal Experiments (v2).
Implements exact metrics: JSD, Shannon entropy, margin, obj_mass, dlogp,
stratified site selection (k in {1,2,3} + up to 2 in [4,8]), and alternative token ranking.
"""

import math
import random
import logging
from typing import Dict, List, Set, Tuple, Optional, Any

import torch
import numpy as np

from objects import ObjectVocabulary

logger = logging.getLogger("confirmatory_v2")

PUNCTUATION_CHARS = set(".,!?:;\"'()[]{}-/\\\n\t`~@#$%^&*+=<>")
SENTENCE_FINAL_PUNCT = {".", "!", "?"}


def jsd_divergence(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Computes Jensen-Shannon DIVERGENCE (base 2), strictly in range [0, 1].
    JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m), where m = 0.5 * (p + q).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum > 0:
        p = p / p_sum
    if q_sum > 0:
        q = q / q_sum

    m = 0.5 * (p + q)

    kl_pm = np.sum(np.where(p > eps, p * np.log2((p + eps) / (m + eps)), 0.0))
    kl_qm = np.sum(np.where(q > eps, q * np.log2((q + eps) / (m + eps)), 0.0))

    jsd = 0.5 * (kl_pm + kl_qm)
    return float(np.clip(jsd, 0.0, 1.0))


def compute_restricted_distribution(
    logits: torch.Tensor,
    obj_ids: List[int],
) -> np.ndarray:
    """
    Computes restricted softmax distribution p(·) over K object token IDs.
    Casts logits to float32 BEFORE softmax. Returns 1D numpy float64 array summing to 1.
    """
    logits_float = logits.to(torch.float32)
    logits_obj = logits_float[obj_ids]
    probs_obj = torch.softmax(logits_obj, dim=-1).cpu().numpy()
    return probs_obj.astype(np.float64)


def compute_obj_mass(
    full_logits: torch.Tensor,
    obj_ids: List[int],
) -> float:
    """Computes total full-vocabulary probability mass on obj_ids at that position."""
    logits_float = full_logits.to(torch.float32)
    full_probs = torch.softmax(logits_float, dim=-1)
    mass = full_probs[obj_ids].sum().item()
    return float(mass)


def compute_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """Computes Shannon entropy H(p) in base 2 (bits): -sum p * log2(p)."""
    p = np.asarray(p, dtype=np.float64)
    ent = -np.sum(np.where(p > eps, p * np.log2(p + eps), 0.0))
    return float(ent)


def compute_margin(p: np.ndarray, eps: float = 1e-12) -> float:
    """Computes log margin(p) = log p(top1) - log p(top2) in natural log or bits (base 2)."""
    sorted_p = np.sort(p)[::-1]
    top1 = max(float(sorted_p[0]), eps)
    top2 = max(float(sorted_p[1]) if len(sorted_p) > 1 else eps, eps)
    return float(np.log(top1) - np.log(top2))


def format_top3_objects(p_obj: np.ndarray, obj_vocab: ObjectVocabulary) -> str:
    """Formats top-3 object words and probabilities as a string (e.g. 'dog:0.75,cat:0.15,horse:0.05')."""
    top_indices = np.argsort(p_obj)[::-1][:3]
    parts = []
    for idx in top_indices:
        tok_id = obj_vocab.obj_ids[idx]
        w = obj_vocab.get_word(tok_id) or str(tok_id)
        prob = p_obj[idx]
        parts.append(f"{w}:{prob:.4f}")
    return ",".join(parts)


def is_word_start_token(tok_str: str) -> bool:
    """Checks if token represents the start of a word."""
    return tok_str.startswith("\u2581") or tok_str.startswith(" ")


def is_punctuation_or_special(tok_str: str, tok_id: int, special_ids: Set[int]) -> bool:
    """Checks if token is punctuation, whitespace, newline, or special token."""
    if tok_id in special_ids:
        return True
    clean = tok_str.replace("\u2581", "").replace(" ", "").strip()
    if not clean:
        return True
    if all(c in PUNCTUATION_CHARS for c in clean):
        return True
    return False


def find_candidate_perturbation_sites(
    t: int,
    gen_ids: torch.Tensor,
    obj_positions_set: Set[int],
    obj_vocab: ObjectVocabulary,
    special_ids: Set[int],
    tokenizer: Any,
) -> List[int]:
    """
    Finds candidate sites s in [max(0, t-8), t-1] such that:
    - y_s is a word-start token;
    - not punctuation/newline, not special/EOS, not in obj_ids;
    - NO object token lies strictly between s and t (so that t is the first object after s);
    - NOTE: Determiners/prepositions/stopwords are NOT excluded (recorded in DEVIATIONS.md).
    """
    s_min = max(0, t - 8)
    s_max = t - 1
    candidates = []

    for s in range(s_min, s_max + 1):
        # 1. No object token strictly between s and t
        has_object_between = any((idx in obj_positions_set) for idx in range(s + 1, t))
        if has_object_between:
            continue

        cand_tok_id = int(gen_ids[0, s].item())
        if cand_tok_id in obj_vocab.obj_ids_set:
            continue

        cand_tok_str = tokenizer.convert_ids_to_tokens(cand_tok_id) or ""
        if not is_word_start_token(cand_tok_str):
            continue

        if is_punctuation_or_special(cand_tok_str, cand_tok_id, special_ids):
            continue

        candidates.append(s)

    return candidates


def select_stratified_sites(
    candidate_sites: List[int],
    t: int,
    image_id: int,
    seed_base: int = 2026,
) -> List[int]:
    """
    Site selection per object (seed = 2026 + image_id*1000 + t):
    - Take ALL eligible sites with k in {1, 2, 3} (at most 3 exist);
    - Plus up to 2 random sites with k in [4, 8] (negative control).
    """
    near_sites = [s for s in candidate_sites if (t - s) in {1, 2, 3}]
    far_sites = [s for s in candidate_sites if (t - s) >= 4]

    rng = random.Random(seed_base + image_id * 1000 + t)
    selected_far = rng.sample(far_sites, min(len(far_sites), 2)) if far_sites else []

    chosen = sorted(near_sites + selected_far)
    return chosen


def select_alternative_tokens_ranked(
    full_logits_at_s: torch.Tensor,
    orig_tok_id: int,
    obj_vocab: ObjectVocabulary,
    tokenizer: Any,
    special_ids: Set[int],
    top_n_search: int = 100,
    min_prob_threshold: float = 0.01,
) -> Dict[int, Dict[str, Any]]:
    """
    Finds TWO alternatives per site from the model's full-vocabulary distribution at position s:
    alt_rank=1 (best remaining) and alt_rank=2 (second best remaining).
    Excludes: original token, tokens in obj_ids, EOS, punctuation, non-word-start tokens.
    Requires probability >= 0.01, else that rank is skipped.
    Returns: Dict[rank, {"token_id": int, "p_alt": float, "token_str": str, "p_orig": float}]
    """
    logits_float = full_logits_at_s.to(torch.float32)
    full_probs = torch.softmax(logits_float, dim=-1).cpu().numpy()
    p_orig = float(full_probs[orig_tok_id])

    # Top candidates descending
    top_candidate_ids = np.argsort(full_probs)[::-1][:top_n_search]

    valid_alternatives = []
    for cand_id in top_candidate_ids:
        cand_id = int(cand_id)
        if cand_id == orig_tok_id:
            continue
        if cand_id in obj_vocab.obj_ids_set:
            continue
        if cand_id in special_ids:
            continue

        cand_str = tokenizer.convert_ids_to_tokens(cand_id) or ""
        if not is_word_start_token(cand_str):
            continue
        if is_punctuation_or_special(cand_str, cand_id, special_ids):
            continue

        p_cand = float(full_probs[cand_id])
        if p_cand < min_prob_threshold:
            # Since sorted descending, subsequent probabilities are also below threshold
            break

        cand_decoded = tokenizer.decode([cand_id]).strip()
        valid_alternatives.append({
            "token_id": cand_id,
            "p_alt": p_cand,
            "token_str": cand_decoded,
            "p_orig": p_orig,
        })
        if len(valid_alternatives) >= 2:
            break

    result = {}
    if len(valid_alternatives) >= 1:
        result[1] = valid_alternatives[0]
    if len(valid_alternatives) >= 2:
        result[2] = valid_alternatives[1]

    return result


def compute_sentence_and_word_distance(
    s: int,
    t: int,
    gen_ids: torch.Tensor,
    tokenizer: Any,
) -> Tuple[int, int]:
    """
    Computes:
    - k_words: number of words between s and t.
    - same_sentence: 1 if no sentence-final punctuation token (. ! ?) between s and t.
    """
    same_sentence = 1
    word_count = 0

    for idx in range(s + 1, t):
        tok_id = int(gen_ids[0, idx].item())
        tok_str = tokenizer.convert_ids_to_tokens(tok_id) or ""
        clean = tok_str.replace("\u2581", "").replace(" ", "").strip()

        if any(p in clean for p in SENTENCE_FINAL_PUNCT):
            same_sentence = 0

        if is_word_start_token(tok_str) and clean:
            word_count += 1

    return word_count, same_sentence
