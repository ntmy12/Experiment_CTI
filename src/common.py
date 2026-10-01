"""
Core utilities for Experiment 1:
- Tokenizer pieces-to-text reconstruction and character-to-token span mapping
- Mention extraction with CHAIR-style lemmatization
- Ground truth COCO categories extraction
- Object pair matching algorithm
- Statistical testing: paired bootstrap, Wilcoxon signed-rank, Holm correction, Cohen's d_z, AUROC
"""

import json
import math
import os
import random
import re
from typing import Dict, List, Tuple, Set, Optional, Any

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy import stats
except ImportError:
    stats = None

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


_CACHED_IMG_DIR: Optional[str] = None


def resolve_image_path(image_dir: str, file_name: str) -> Optional[str]:
    """
    Finds the full absolute path of an image file.
    Robust against nested folders (e.g. data/val2014/val2014)
    and searches /kaggle/input if necessary, caching the discovered directory.
    """
    global _CACHED_IMG_DIR

    # 1. Check cached directory if available
    if _CACHED_IMG_DIR is not None:
        cand = os.path.join(_CACHED_IMG_DIR, file_name)
        if os.path.exists(cand):
            return cand

    # 2. Check direct path: image_dir / file_name
    cand = os.path.join(image_dir, file_name)
    if os.path.exists(cand):
        _CACHED_IMG_DIR = os.path.dirname(os.path.abspath(cand))
        return cand

    # 3. Check nested subdirectory: image_dir / val2014 / file_name
    cand_nested = os.path.join(image_dir, "val2014", file_name)
    if os.path.exists(cand_nested):
        _CACHED_IMG_DIR = os.path.dirname(os.path.abspath(cand_nested))
        return cand_nested

    # 4. Search in /kaggle/input if running on Kaggle
    if os.path.exists("/kaggle/input"):
        import glob
        matches = glob.glob(f"/kaggle/input/**/{file_name}", recursive=True)
        if matches:
            _CACHED_IMG_DIR = os.path.dirname(os.path.abspath(matches[0]))
            return matches[0]

    return None


def load_synonyms(path: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Loads synonyms mapping from file.
    Supports comma-separated, colon-separated, or space-separated formats.
    First word/term on each line is considered the canonical COCO category.
    Returns:
        syn2canon: Dict[str, str] mapping synonym word/phrase to canonical name
        canon_list: List[str] list of canonical category names
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Synonyms file not found at: {path}")

    syn2canon = {}
    canon_list = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Handle colon separator: "dog: dog, puppy, pup"
            if ":" in line:
                canon_part, syns_part = line.split(":", 1)
                canon = canon_part.strip().lower()
                items = [canon] + [s.strip().lower() for s in syns_part.split(",") if s.strip()]
            elif "," in line:
                items = [s.strip().lower() for s in line.split(",") if s.strip()]
                canon = items[0]
            else:
                items = [s.strip().lower() for s in line.split() if s.strip()]
                canon = items[0]

            if canon not in canon_list:
                canon_list.append(canon)

            for item in items:
                # setdefault: if word appears in multiple lines, keep first category
                syn2canon.setdefault(item, canon)

    return syn2canon, canon_list


def pieces_to_text(pieces: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Converts SentencePiece / LLaMA tokenizer pieces to reconstructed text
    and computes character spans (start, end) for each token piece.

    Rules:
    - ' ' (or '\\u2581') is replaced with a single space ' '.
    - Byte token '<0x0A>' is replaced with newline '\\n'.
    - Other byte tokens '<0xNN>' are converted to chr(int(NN, 16)) if printable, else space.
    - Other tokens kept as-is.

    Returns:
        text: Reconstructed full string
        spans: List of (char_start, char_end) for each piece in pieces
    """
    spans = []
    text_parts = []
    curr_idx = 0

    for piece in pieces:
        # 1. Handle byte token <0xNN>
        if piece == "<0x0A>":
            piece_str = "\n"
        elif re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece):
            hex_val = int(piece[3:5], 16)
            try:
                char = chr(hex_val)
                # Keep printable ASCII/newline, otherwise map to space
                piece_str = char if (char.isprintable() or char in "\t\n\r") else " "
            except Exception:
                piece_str = " "
        else:
            # Replace SentencePiece prefix ' ' / '\u2581' with space
            piece_str = piece.replace("\u2581", " ").replace(" ", " ")

        start = curr_idx
        end = curr_idx + len(piece_str)
        spans.append((start, end))
        text_parts.append(piece_str)
        curr_idx = end

    text = "".join(text_parts)
    return text, spans


def char_to_token(spans: List[Tuple[int, int]], char_idx: int) -> int:
    """
    Finds which token index contains char_idx (i.e. start <= char_idx < end).
    If char_idx matches the end of the last span or beyond, returns last token index.
    """
    for i, (start, end) in enumerate(spans):
        if start <= char_idx < end:
            return i
    # Fallback to the closest span
    if spans and char_idx >= spans[-1][1]:
        return len(spans) - 1
    return 0


def lemmatize_word(word: str) -> List[str]:
    """
    Generates lemmatized candidate roots according to Section 6.2:
    - Original root
    - -ies -> -y
    - -es -> -
    - -s -> -
    """
    candidates = [word]
    if word.endswith("ies") and len(word) > 3:
        candidates.append(word[:-3] + "y")
    if word.endswith("es") and len(word) > 2:
        candidates.append(word[:-2])
    if word.endswith("s") and len(word) > 1:
        candidates.append(word[:-1])
    return candidates


def find_mentions(text: str, syn2canon: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Extracts object mentions from text using syn2canon.
    Rules:
    - Words extracted with [A-Za-z]+, lowercased.
    - Priority: 2-word collocations first (separated by exactly one space), then 1-word.
    - Lemmatization applied to the last word.
    - Non-overlapping mentions.
    Returns:
        List of dicts: {"canon": canon, "word": word_in_text, "char_start": start, "char_end": end}
    """
    # Find all words with character spans
    words_info = []
    for m in re.finditer(r"[A-Za-z]+", text):
        words_info.append({
            "word": m.group().lower(),
            "start": m.start(),
            "end": m.end()
        })

    mentions = []
    used_word_indices: Set[int] = set()

    # Pass 1: Two-word collocations (adjacent words separated by exactly one space)
    for i in range(len(words_info) - 1):
        w1_info = words_info[i]
        w2_info = words_info[i + 1]

        # Check if adjacent with single space in between
        if w2_info["start"] == w1_info["end"] + 1 and text[w1_info["end"]] == " ":
            w1 = w1_info["word"]
            w2 = w2_info["word"]

            # Try lemmatizing w2
            for w2_cand in lemmatize_word(w2):
                two_word = f"{w1} {w2_cand}"
                if two_word in syn2canon:
                    canon = syn2canon[two_word]
                    raw_phrase = text[w1_info["start"]: w2_info["end"]]
                    mentions.append({
                        "canon": canon,
                        "word": raw_phrase,
                        "char_start": w1_info["start"],
                        "char_end": w2_info["end"],
                        "_idx": (i, i + 1)
                    })
                    used_word_indices.add(i)
                    used_word_indices.add(i + 1)
                    break

    # Pass 2: Single-word mentions
    for i, w_info in enumerate(words_info):
        if i in used_word_indices:
            continue

        w = w_info["word"]
        for w_cand in lemmatize_word(w):
            if w_cand in syn2canon:
                canon = syn2canon[w_cand]
                raw_word = text[w_info["start"]: w_info["end"]]
                mentions.append({
                    "canon": canon,
                    "word": raw_word,
                    "char_start": w_info["start"],
                    "char_end": w_info["end"],
                    "_idx": (i,)
                })
                used_word_indices.add(i)
                break

    # Sort mentions by char_start
    mentions.sort(key=lambda m: m["char_start"])

    # Clean internal keys
    for m in mentions:
        m.pop("_idx", None)

    return mentions


def check_word_start(spans: List[Tuple[int, int]], tok_idx: int, char_start: int) -> bool:
    """
    Checks if token at tok_idx starts at the word boundary according to Section 6.2:
    spans[tok_idx].start must equal char_start - 1 (preceded by a space in token)
    or char_start (start of text, or following a newline/special token).
    """
    if tok_idx < 0 or tok_idx >= len(spans):
        return False
    tok_start = spans[tok_idx][0]
    return (tok_start == char_start - 1) or (tok_start == char_start)


def build_coco_gt(
    image_id: int,
    instances_by_image: Dict[int, List[str]],
    captions_by_image: Dict[int, List[str]],
    syn2canon: Dict[str, str]
) -> Set[str]:
    """
    Computes ground truth objects GT(image) according to Section 4.4:
    GT(image) = {category in instances_val2014 via syn2canon} U
                {objects in 5 human captions extracted using find_mentions}
    """
    gt_objects = set()

    # 1. Instances annotations
    if image_id in instances_by_image:
        for cat_name in instances_by_image[image_id]:
            canon = syn2canon.get(cat_name.lower())
            if canon:
                gt_objects.add(canon)

    # 2. Human reference captions
    if image_id in captions_by_image:
        for cap in captions_by_image[image_id]:
            cap_mentions = find_mentions(cap, syn2canon)
            for m in cap_mentions:
                gt_objects.add(m["canon"])

    return gt_objects


def match_object_pairs(
    halluc_objects: List[Dict[str, Any]],
    real_objects: List[Dict[str, Any]],
    caliper: float = 0.1,
    seed: int = 0
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Matches 1:1 control pair between hallucinated and real objects according to Section 4.5:
    - Real object r not used yet (without replacement)
    - |rel_pos(r) - rel_pos(h)| <= caliper
    - Smallest difference; ties broken randomly using seed
    - Hallucinated objects order shuffled with seed
    Returns:
        matched_h: List of matched hallucinated objects (with pair_id added)
        matched_r: List of matched real objects (with pair_id added)
        dropped_h_count: Number of hallucinated objects dropped due to no match
    """
    rng = random.Random(seed)

    # Copy to avoid mutating original lists
    h_pool = list(halluc_objects)
    rng.shuffle(h_pool)

    r_pool = list(real_objects)
    available_r_indices = set(range(len(r_pool)))

    matched_h = []
    matched_r = []
    pair_id = 0

    for h in h_pool:
        h_pos = h["rel_pos"]
        candidates = []

        for r_idx in available_r_indices:
            r = r_pool[r_idx]
            diff = abs(r["rel_pos"] - h_pos)
            if diff <= caliper:
                candidates.append((diff, r_idx))

        if not candidates:
            continue

        # Sort by smallest difference; break ties randomly
        rng.shuffle(candidates)
        candidates.sort(key=lambda x: x[0])

        best_r_idx = candidates[0][1]
        available_r_indices.remove(best_r_idx)

        h_matched = dict(h)
        r_matched = dict(r_pool[best_r_idx])

        h_matched["pair_id"] = pair_id
        r_matched["pair_id"] = pair_id

        matched_h.append(h_matched)
        matched_r.append(r_matched)
        pair_id += 1

    dropped_h_count = len(h_pool) - len(matched_h)
    return matched_h, matched_r, dropped_h_count


# ==========================================
# Statistical Functions
# ==========================================

def paired_bootstrap_delta(
    deltas: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0
) -> Tuple[float, float, float]:
    """
    Computes mean delta and 95% paired bootstrap CI according to Section 6.5.
    Returns: (mean_delta, ci_lo, ci_hi)
    """
    if len(deltas) == 0:
        return 0.0, 0.0, 0.0

    mean_delta = float(np.mean(deltas))
    if len(deltas) == 1:
        return mean_delta, mean_delta, mean_delta

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)

    n = len(deltas)
    for b in range(n_boot):
        sample = rng.choice(deltas, size=n, replace=True)
        boot_means[b] = np.mean(sample)

    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    return mean_delta, ci_lo, ci_hi


def cohens_dz(deltas: np.ndarray) -> float:
    """
    Computes Cohen's d_z for paired differences: mean(delta) / sd(delta)
    """
    if len(deltas) <= 1:
        return 0.0
    sd = np.std(deltas, ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(deltas) / sd)


def compute_wilcoxon_p(deltas: np.ndarray) -> float:
    """
    Computes two-sided Wilcoxon signed-rank test p-value.
    Falls back to 1.0 if all differences are zero.
    """
    if len(deltas) == 0 or np.all(deltas == 0):
        return 1.0
    try:
        res = stats.wilcoxon(deltas, alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        return 1.0


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """
    Applies Holm-Bonferroni step-down correction on a list of p-values.
    Returns adjusted p-values in original order.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values with original indices
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adj_p = [0.0] * m

    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank
        val = min(1.0, multiplier * p_val)
        running_max = max(running_max, val)
        adj_p[orig_idx] = running_max

    return adj_p


def bootstrap_auroc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0
) -> Tuple[float, float, float]:
    """
    Computes AUROC and 95% bootstrap CI.
    Returns: (auroc, ci_lo, ci_hi)
    """
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.5

    try:
        base_auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        return 0.5, 0.5, 0.5

    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_aucs = []

    for _ in range(n_boot):
        indices = rng.choice(n, size=n, replace=True)
        sub_true = y_true[indices]
        if len(np.unique(sub_true)) < 2:
            continue
        try:
            boot_aucs.append(roc_auc_score(sub_true, y_scores[indices]))
        except Exception:
            pass

    if len(boot_aucs) < 50:
        return base_auc, base_auc, base_auc

    ci_lo = float(np.percentile(boot_aucs, 2.5))
    ci_hi = float(np.percentile(boot_aucs, 97.5))
    return base_auc, ci_lo, ci_hi
