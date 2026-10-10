"""
Core computation functions for Pilot Experiment H2.1:
- Restricted object distribution softmax and Jensen-Shannon Divergence (base 2).
- Visual contribution V_t.
- Prefix perturbation generation, sensitivity S_t, and flip detection.
- Placebo assertion and covariate computation.
"""

import math
import random
import logging
from typing import Dict, List, Set, Tuple, Optional, Any

import torch
import numpy as np

from objects import ObjectVocabulary

logger = logging.getLogger("pilot_h21")

STOPWORDS = {
    "a", "an", "the", "of", "and", "in", "on", "is", "are", "with",
    "at", "by", "for", "from", "to", "into", "it", "its", "that", "this",
    "was", "were", "be", "been", "has", "have", "had", "as", "or", "but"
}

PUNCTUATION_CHARS = set(".,!?:;\"'()[]{}-/\\\n\t`~@#$%^&*+=<>")


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

    # Renormalize if needed
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum > 0:
        p = p / p_sum
    if q_sum > 0:
        q = q / q_sum

    m = 0.5 * (p + q)

    # KL(p || m) with base 2
    # Where p > 0: p * log2(p / m)
    kl_pm = np.sum(np.where(p > eps, p * np.log2((p + eps) / (m + eps)), 0.0))
    kl_qm = np.sum(np.where(q > eps, q * np.log2((q + eps) / (m + eps)), 0.0))

    jsd = 0.5 * (kl_pm + kl_qm)
    return float(np.clip(jsd, 0.0, 1.0))


def compute_restricted_distribution(
    logits: torch.Tensor,
    obj_ids: List[int],
) -> np.ndarray:
    """
    Computes restricted softmax distribution p(·) over K object token IDs, cast to float32.
    Returns 1D numpy array of size K summing to 1.
    """
    logits_obj = logits[obj_ids].to(torch.float32)
    probs_obj = torch.softmax(logits_obj, dim=-1).cpu().numpy()
    return probs_obj.astype(np.float64)


def compute_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """Computes Shannon entropy in base 2: -sum p * log2(p)."""
    p = np.asarray(p, dtype=np.float64)
    ent = -np.sum(np.where(p > eps, p * np.log2(p + eps), 0.0))
    return float(ent)


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


def is_candidate_perturbation_site(
    tok_id: int,
    tok_str: str,
    obj_vocab: ObjectVocabulary,
    special_ids: Set[int],
) -> bool:
    """
    Checks if token at site s is a valid candidate for perturbation:
    Excludes:
    - tokens in obj_ids
    - punctuation/newline tokens
    - special tokens
    - stopwords
    - non-word-start tokens (tokens not starting with ' ' / '\u2581')
    """
    if tok_id in obj_vocab.obj_ids_set:
        return False
    if tok_id in special_ids:
        return False

    # Check word start prefix
    if not (tok_str.startswith("\u2581") or tok_str.startswith(" ")):
        return False

    clean_text = tok_str.replace("\u2581", "").strip().lower()
    if not clean_text:
        return False

    # Check punctuation
    if all(c in PUNCTUATION_CHARS for c in clean_text):
        return False

    # Check stopwords
    if clean_text in STOPWORDS:
        return False

    return True


def select_alternative_token(
    full_logits_at_s: torch.Tensor,
    orig_tok_id: int,
    obj_vocab: ObjectVocabulary,
    tokenizer: Any,
    special_ids: Set[int],
    top_n_search: int = 50,
) -> Tuple[Optional[int], float, float]:
    """
    Finds the rank-1 alternative token among the model's top plausible full-vocab candidates.
    Excludes: original token, tokens in obj_ids, EOS/special tokens, punctuation, non-word-start tokens.
    Returns: (alt_token_id, p_alt, p_orig)
    """
    probs_full = torch.softmax(full_logits_at_s.to(torch.float32), dim=-1).cpu().numpy()
    p_orig = float(probs_full[orig_tok_id])

    # Get top candidate indices
    top_indices = np.argsort(probs_full)[::-1][:top_n_search]

    for cand_id in top_indices:
        cand_id = int(cand_id)
        if cand_id == orig_tok_id:
            continue
        if cand_id in obj_vocab.obj_ids_set or cand_id in special_ids:
            continue

        cand_str = tokenizer.convert_ids_to_tokens(cand_id)
        if not cand_str:
            continue

        # Must be word start
        if not (cand_str.startswith("\u2581") or cand_str.startswith(" ")):
            continue

        clean_text = cand_str.replace("\u2581", "").strip().lower()
        if not clean_text:
            continue
        if all(c in PUNCTUATION_CHARS for c in clean_text):
            continue

        # Found the top plausible alternative
        p_alt = float(probs_full[cand_id])
        return cand_id, p_alt, p_orig

    return None, 0.0, p_orig


def run_placebo_test(
    runner: Any,
    prompt_ids: torch.Tensor,
    gen_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    t: int,
    s: int,
    p_orig_restricted: np.ndarray,
    obj_vocab: ObjectVocabulary,
) -> Tuple[float, int]:
    """
    Placebo test: replaces y_s by itself.
    Asserts S_t < 1e-6 and flip = 0.
    """
    perturbed_prefix = gen_ids[:, :t].clone()
    # Replace y_s by y_s (itself)
    perturbed_prefix[0, s] = gen_ids[0, s]

    logits_placebo = runner.get_perturbed_target_logits(prompt_ids, perturbed_prefix, pixel_values)
    p_placebo_restricted = compute_restricted_distribution(logits_placebo, obj_vocab.obj_ids)

    s_t_placebo = jsd_divergence(p_orig_restricted, p_placebo_restricted)
    flip_placebo = 1 if (np.argmax(p_orig_restricted) != np.argmax(p_placebo_restricted)) else 0

    return s_t_placebo, flip_placebo
