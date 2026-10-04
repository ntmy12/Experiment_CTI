"""
Shared utilities, metric formulations, and statistical functions for Experiment H1.
Provides:
  - Vocabulary token set extraction
  - Candidate replacement generation and split-half assignment (A/B)
  - Split-half concentration and stability metrics (Mbar, HPR, logHPR, rho, rho_z)
  - Permutation null distribution for logHPR
  - Statistical testing (paired Wilcoxon, 1-sample Wilcoxon, Holm-Bonferroni, Cohen's d_z, Bootstrap CI)
"""

import math
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy import stats
except ImportError:
    stats = None

# SentencePiece word piece regex: begins with space or \u2581, followed by alphabetic characters
WORD_PIECE_REGEX = re.compile(r"^[\s\u2581][A-Za-z]+$")

# Functional word lexicon for fallback mode (--pos_mode lexicon)
FUNCTION_WORDS = {
    # Articles & Determiners
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "every", "each",
    # Prepositions
    "in", "on", "at", "by", "with", "from", "to", "into", "onto", "upon", "near", "beside",
    "under", "over", "above", "below", "behind", "in front of", "across", "through",
    # Conjunctions
    "and", "or", "but", "while", "as",
    # Pronouns
    "it", "its", "they", "their", "there",
    # Be auxiliary forms
    "is", "are", "was", "were", "be", "being", "been"
}


# ===========================================================================
# 1. Vocabulary Extraction (Identical to Experiment 1)
# ===========================================================================

def get_v_obj_tokens(tokenizer: Any, syn2canon: Dict[str, str]) -> Set[int]:
    """
    Builds V_obj token set according to Experiment 1 (Section 4.3 & 6.4):
    First token id of every word in synonyms.txt.
    Checks whether tokenizer prepends leading space.
    """
    words = list(syn2canon.keys())
    sample_words = words[:20]

    # Check whether tokenizer prepends leading space
    test_enc = tokenizer.encode(sample_words[0], add_special_tokens=False)
    test_piece = tokenizer.convert_ids_to_tokens(test_enc[0])
    has_leading_space_token = test_piece.startswith(" ") or test_piece.startswith("\u2581")

    v_obj_set = set()
    sample_log = []

    for w in words:
        cand_str = " " + w if not has_leading_space_token else w
        enc = tokenizer.encode(cand_str, add_special_tokens=False)
        if enc:
            tok_id = enc[0]
            v_obj_set.add(tok_id)
            if len(sample_log) < 10:
                sample_log.append((w, tok_id, tokenizer.convert_ids_to_tokens(tok_id)))

    print("\nTokenizer V_obj verification (10 samples):")
    for w, tid, piece in sample_log:
        print(f"  Word: '{w}' -> Token ID: {tid}, Piece: '{piece}'")
    print(f"Total V_obj tokens: {len(v_obj_set)}\n")
    assert len(v_obj_set) > 50, f"Error: V_obj set too small ({len(v_obj_set)}). Tokenizer issue!"

    return v_obj_set


# ===========================================================================
# 2. Linguistic Filtering & Candidate Replacement Generation
# ===========================================================================

def clean_piece(piece: str) -> str:
    """Strips leading SentencePiece space markers (\u2581 or whitespace)."""
    return piece.lstrip(" \u2581")


def is_complete_word_piece(piece: str, next_piece: Optional[str] = None) -> bool:
    """
    Checks if a token piece represents a complete single word piece:
    1. Piece matches regex ^[ \\u2581][A-Za-z]+$
    2. If next_piece is provided, next_piece starts with space/punctuation
       (i.e. this word is not split across multiple subwords).
    """
    if not WORD_PIECE_REGEX.match(piece):
        return False

    if next_piece is not None:
        # Next token should start with a space marker or punctuation
        starts_new_word = (
            next_piece.startswith(" ") or
            next_piece.startswith("\u2581") or
            (len(next_piece) > 0 and not next_piece[0].isalnum())
        )
        if not starts_new_word:
            return False

    return True


def check_a_an_agreement(cand_word: str, next_word: Optional[str]) -> bool:
    """
    Phonetic/orthographic agreement check for 'a' vs 'an':
    - 'an' must precede vowel-initial words (a, e, i, o, u)
    - 'a' must precede consonant-initial words
    """
    cand_lower = cand_word.lower()
    if cand_lower not in {"a", "an"}:
        return True

    if not next_word:
        return True

    next_clean = clean_piece(next_word).lower()
    if not next_clean:
        return True

    first_char = next_clean[0]
    is_vowel = first_char in "aeiou"

    if cand_lower == "an":
        return is_vowel
    else:  # "a"
        return not is_vowel


def get_token_upos(sentence_text: str, target_word_clean: str, word_char_index: int, nlp: Any) -> Optional[str]:
    """
    Runs spaCy pipeline on sentence and retrieves UPOS for target word.
    """
    if nlp is None:
        return None
    doc = nlp(sentence_text)
    for token in doc:
        # Match by character offset or surface text
        if token.idx <= word_char_index < token.idx + len(token.text):
            return token.pos_
    # Fallback: match by text
    for token in doc:
        if token.text.lower() == target_word_clean.lower():
            return token.pos_
    return None


def filter_and_split_candidates(
    top50_token_ids: List[int],
    top50_token_probs: List[float],
    tokenizer: Any,
    y_j_id: int,
    sentence_text: str,
    target_char_idx: int,
    next_token_piece: Optional[str] = None,
    nlp: Any = None,
    pos_mode: str = "spacy",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filters replacement candidates and splits into balanced split-halves A and B.
    Conditions:
      1. Exclude original token y_j.
      2. Piece matches regex ^[ \\u2581][A-Za-z]+$.
      3. Same UPOS tag as y_j in sentence context (or same function-word lexicon in fallback).
      4. Phonetic agreement for 'a' / 'an'.
      5. Case-insensitively distinct from y_j.
      6. Select top <= 6 candidates, alternating assignment:
         rank 1, 3, 5 -> Half A
         rank 2, 4, 6 -> Half B
    """
    y_j_piece = tokenizer.convert_ids_to_tokens(y_j_id)
    y_j_word = clean_piece(y_j_piece).lower()

    # Determine original UPOS
    orig_upos = None
    if pos_mode == "spacy" and nlp is not None:
        orig_upos = get_token_upos(sentence_text, y_j_word, target_char_idx, nlp)

    valid_candidates: List[Dict[str, Any]] = []

    for rank, (cand_id, cand_p) in enumerate(zip(top50_token_ids, top50_token_probs), start=1):
        if cand_id == y_j_id:
            continue

        cand_piece = tokenizer.convert_ids_to_tokens(cand_id)
        if not WORD_PIECE_REGEX.match(cand_piece):
            continue

        cand_word = clean_piece(cand_piece).lower()
        if cand_word == y_j_word:
            continue

        # Check a / an agreement
        if not check_a_an_agreement(cand_word, next_token_piece):
            continue

        # Check POS matching
        if pos_mode == "spacy" and nlp is not None and orig_upos is not None:
            # Substitute candidate into sentence and check if UPOS matches
            mutated_sentence = (
                sentence_text[:target_char_idx] +
                clean_piece(cand_piece) +
                sentence_text[target_char_idx + len(y_j_word):]
            )
            cand_upos = get_token_upos(mutated_sentence, cand_word, target_char_idx, nlp)
            if cand_upos != orig_upos:
                continue
        elif pos_mode == "lexicon":
            # Lexicon fallback mode
            if y_j_word not in FUNCTION_WORDS or cand_word not in FUNCTION_WORDS:
                continue

        valid_candidates.append({
            "cand_id": cand_id,
            "cand_piece": cand_piece,
            "cand_word": cand_word,
            "cand_rank": rank,
            "cand_p": float(cand_p),
        })

        if len(valid_candidates) >= 6:
            break

    # Require at least 4 valid candidates (min 2 per half)
    if len(valid_candidates) < 4:
        return [], []

    # Alternating split by rank
    half_A: List[Dict[str, Any]] = []
    half_B: List[Dict[str, Any]] = []

    for idx, cand in enumerate(valid_candidates):
        if idx % 2 == 0:
            cand["half"] = "A"
            half_A.append(cand)
        else:
            cand["half"] = "B"
            half_B.append(cand)

    return half_A, half_B


# ===========================================================================
# 3. Object-level Metric Computation (Mbar, HPR, rho)
# ===========================================================================

def _rank_data(a: List[float]) -> List[float]:
    """Computes fractional ranks for a list of values."""
    n = len(a)
    sorted_indices = sorted(range(n), key=lambda i: a[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[sorted_indices[j + 1]] == a[sorted_indices[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_indices[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman_corr(x: List[float], y: List[float]) -> float:
    """Computes Spearman rank correlation using scipy.stats if available, or pure Python."""
    if stats is not None:
        try:
            res = stats.spearmanr(x, y)
            val = res.statistic if hasattr(res, "statistic") else res.correlation
            return 0.0 if math.isnan(val) else float(val)
        except Exception:
            pass
    # Pure Python fallback
    rx = _rank_data(x)
    ry = _rank_data(y)
    n = len(rx)
    if n <= 1:
        return 0.0
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    var_x = sum((xi - mean_x) ** 2 for xi in rx)
    var_y = sum((yi - mean_y) ** 2 for yi in ry)
    if var_x < 1e-15 or var_y < 1e-15:
        return 0.0
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(rx, ry))
    return cov / math.sqrt(var_x * var_y)


def compute_object_metrics(
    effects_by_pos: Dict[int, Dict[str, List[float]]],
    valid_window: Set[int],
) -> Optional[Dict[str, Any]]:
    """
    Computes Mbar, HPR, logHPR, rho, rho_z, top1_agree for an object across valid window W.

    Args:
        effects_by_pos: mapping from position j -> {"A": [e_j_c, ...], "B": [e_j_c, ...]}
        valid_window: positions j in W = {j : d in [2, 10]}

    Returns:
        dict of metrics or None if fewer than 5 eligible positions.
    """
    eligible_j = [
        j for j in sorted(effects_by_pos.keys())
        if j in valid_window and len(effects_by_pos[j].get("A", [])) >= 2 and len(effects_by_pos[j].get("B", [])) >= 2
    ]

    if len(eligible_j) < 5:
        return None

    n = len(eligible_j)
    e_A_arr = [float(sum(effects_by_pos[j]["A"]) / len(effects_by_pos[j]["A"])) for j in eligible_j]
    e_B_arr = [float(sum(effects_by_pos[j]["B"]) / len(effects_by_pos[j]["B"])) for j in eligible_j]
    e_mean_arr = [(a + b) / 2.0 for a, b in zip(e_A_arr, e_B_arr)]

    abs_e_A = [abs(x) for x in e_A_arr]
    abs_e_B = [abs(x) for x in e_B_arr]
    abs_e_mean = [abs(x) for x in e_mean_arr]

    # 1. Mean Magnitude Mbar
    mbar = float(sum(abs_e_mean) / n)

    # 2. Held-Out Peak Ratio (HPR)
    mean_abs_B = float(sum(abs_e_B) / n)
    mean_abs_A = float(sum(abs_e_A) / n)

    if mean_abs_B < 1e-12 or mean_abs_A < 1e-12:
        return None

    j_star_A_idx = int(max(range(n), key=lambda i: abs_e_A[i]))
    j_star_B_idx = int(max(range(n), key=lambda i: abs_e_B[i]))

    rA = float(abs_e_B[j_star_A_idx] / mean_abs_B)
    rB = float(abs_e_A[j_star_B_idx] / mean_abs_A)

    hpr = float((rA + rB) / 2.0)
    log_hpr = float(math.log(max(hpr, 1e-6)))

    # 3. Spearman Profile Correlation rho & Fisher z
    if all(x == abs_e_A[0] for x in abs_e_A) or all(x == abs_e_B[0] for x in abs_e_B):
        rho = 0.0
    else:
        rho = _spearman_corr(abs_e_A, abs_e_B)

    clamped_rho = max(-0.999, min(0.999, rho))
    rho_z = float(0.5 * math.log((1.0 + clamped_rho) / (1.0 - clamped_rho)))

    # 4. Top-1 Agreement
    top1_agree = 1 if (j_star_A_idx == j_star_B_idx) else 0

    # 5. Peak position argmax
    peak_idx = int(max(range(n), key=lambda i: abs_e_mean[i]))
    peak_j = eligible_j[peak_idx]

    return {
        "n_eligible": n,
        "eligible_j": eligible_j,
        "Mbar": mbar,
        "HPR": hpr,
        "logHPR": log_hpr,
        "rho": rho,
        "rho_z": rho_z,
        "top1_agree": top1_agree,
        "peak_j": peak_j,
        "abs_e_A": np.array(abs_e_A) if np is not None else abs_e_A,
        "abs_e_B": np.array(abs_e_B) if np is not None else abs_e_B,
        "abs_e_mean": np.array(abs_e_mean) if np is not None else abs_e_mean,
    }


# ===========================================================================
# 4. Permutation Null & Statistical Testing
# ===========================================================================

def permutation_null_logHPR(
    abs_e_A_list: List[Any],
    abs_e_B_list: List[Any],
    n_perm: int = 2000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Computes permutation null distribution of mean(logHPR) by permuting positions of B.
    Returns: (actual_mean_logHPR, null_mean, p_value_perm)
    """
    if len(abs_e_A_list) == 0:
        return 0.0, 0.0, 1.0

    actual_log_hprs = []
    cleaned_pairs = []
    for abs_A, abs_B in zip(abs_e_A_list, abs_e_B_list):
        a_list = list(abs_A)
        b_list = list(abs_B)
        nA = len(a_list)
        nB = len(b_list)
        if nA == 0 or nB == 0:
            continue
        mean_B = sum(b_list) / nB
        mean_A = sum(a_list) / nA
        if mean_B < 1e-12 or mean_A < 1e-12:
            continue
        j_A = max(range(nA), key=lambda i: a_list[i])
        j_B = max(range(nB), key=lambda i: b_list[i])
        rA = b_list[j_A] / mean_B
        rB = a_list[j_B] / mean_A
        actual_log_hprs.append(math.log(max((rA + rB) / 2.0, 1e-6)))
        cleaned_pairs.append((a_list, b_list, mean_A, mean_B))

    if not actual_log_hprs:
        return 0.0, 0.0, 1.0

    actual_mean = sum(actual_log_hprs) / len(actual_log_hprs)

    rng = random.Random(seed)
    null_means = []

    for b in range(n_perm):
        perm_log_hprs = []
        for a_list, b_list, mean_A, mean_B in cleaned_pairs:
            perm_B = list(b_list)
            rng.shuffle(perm_B)
            j_A = max(range(len(a_list)), key=lambda i: a_list[i])
            j_B = max(range(len(perm_B)), key=lambda i: perm_B[i])
            rA = perm_B[j_A] / mean_B
            rB = a_list[j_B] / mean_A
            perm_log_hprs.append(math.log(max((rA + rB) / 2.0, 1e-6)))
        null_means.append(sum(perm_log_hprs) / len(perm_log_hprs))

    p_val_perm = sum(1 for m in null_means if m >= actual_mean) / n_perm
    null_mean = sum(null_means) / n_perm
    return actual_mean, null_mean, p_val_perm


def paired_bootstrap_delta(
    deltas: Any,
    n_boot: int = 5000,
    seed: int = 0
) -> Tuple[float, float, float]:
    """
    Computes mean delta and 95% paired bootstrap CI.
    Returns: (mean_delta, ci_lo, ci_hi)
    """
    d_list = [float(x) for x in deltas]
    n = len(d_list)
    if n == 0:
        return 0.0, 0.0, 0.0

    mean_delta = sum(d_list) / n
    if n == 1:
        return mean_delta, mean_delta, mean_delta

    rng = random.Random(seed)
    boot_means = []

    for b in range(n_boot):
        sample = rng.choices(d_list, k=n)
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    lo_idx = int(round(0.025 * (n_boot - 1)))
    hi_idx = int(round(0.975 * (n_boot - 1)))
    ci_lo = float(boot_means[lo_idx])
    ci_hi = float(boot_means[hi_idx])
    return mean_delta, ci_lo, ci_hi


def cohens_dz(deltas: Any) -> float:
    """Computes Cohen's d_z for paired differences: mean(delta) / sd(delta)."""
    d_list = [float(x) for x in deltas]
    n = len(d_list)
    if n <= 1:
        return 0.0
    mean_val = sum(d_list) / n
    var = sum((x - mean_val) ** 2 for x in d_list) / (n - 1)
    sd = math.sqrt(var)
    if sd < 1e-12:
        return 0.0
    return float(mean_val / sd)


def compute_wilcoxon_paired_p(deltas: Any) -> float:
    """Computes two-sided Wilcoxon signed-rank test p-value for paired differences."""
    d_list = [float(x) for x in deltas]
    if len(d_list) == 0 or all(x == 0 for x in d_list):
        return 1.0
    if stats is not None:
        try:
            res = stats.wilcoxon(d_list, alternative="two-sided")
            return float(res.pvalue)
        except Exception:
            return 1.0
    # Approximate normal fallback when scipy is not available
    non_zero = [x for x in d_list if x != 0]
    n = len(non_zero)
    if n == 0:
        return 1.0
    ranks = _rank_data([abs(x) for x in non_zero])
    w_pos = sum(r for x, r in zip(non_zero, ranks) if x > 0)
    w_neg = sum(r for x, r in zip(non_zero, ranks) if x < 0)
    w = min(w_pos, w_neg)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w < 1e-12:
        return 1.0
    z = (w - mean_w) / math.sqrt(var_w)
    p_val = math.erfc(abs(z) / math.sqrt(2.0))
    return float(p_val)


def compute_wilcoxon_onesample_p(values: Any, popmean: float = 0.0) -> float:
    """Computes two-sided Wilcoxon signed-rank test p-value for one sample vs popmean."""
    diffs = [float(x) - popmean for x in values]
    return compute_wilcoxon_paired_p(diffs)


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """
    Applies Holm-Bonferroni step-down correction on a list of p-values.
    Returns adjusted p-values in original order.
    """
    m = len(p_values)
    if m == 0:
        return []

    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adj_p = [0.0] * m

    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank
        val = min(1.0, multiplier * p_val)
        running_max = max(running_max, val)
        adj_p[orig_idx] = running_max

    return adj_p
