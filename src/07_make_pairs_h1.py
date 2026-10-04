"""
Experiment H1 - Step 1: Pair Construction (07_make_pairs_h1.py)

Builds 1:1 matched pairs for Experiment H1:
1. Primary Matching (samecat):
   - First mention of canonical category (first=True)
   - tok_idx >= K (default K=10)
   - word_start_valid=True
   - 1:1 without replacement
   - canon(r) == canon(h)
   - |rel_pos(r) - rel_pos(h)| <= caliper (default caliper=0.10)
   - image_id(r) != image_id(h)
   - Smallest relative position difference, tie-break random with seed

2. Secondary Matching (pos_only, Sensitivity S1):
   - Same filters but without category constraint (reusing Exp 1 matching logic)

Outputs:
  results/h1/pairs_h1.csv
  results/h1/funnel_pairs.json
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.common import match_object_pairs


def load_candidates_from_labels(
    labels_file: str,
    K: int = 10
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """
    Extracts first-mention object candidates from labels.jsonl meeting K and word_start criteria.
    """
    h_candidates: List[Dict[str, Any]] = []
    r_candidates: List[Dict[str, Any]] = []

    total_captions = 0
    total_mentions = 0
    first_mentions = 0
    halluc_all = 0
    real_all = 0
    dropped_tok_idx = 0
    dropped_not_word_start = 0

    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total_captions += 1
            image_id = record["image_id"]
            file_name = record["file_name"]
            gen_ids = record["gen_ids"]
            G = len(gen_ids)

            for m in record.get("mentions", []):
                total_mentions += 1
                if not m.get("first", False):
                    continue
                first_mentions += 1

                is_halluc = m.get("halluc", False)
                if is_halluc:
                    halluc_all += 1
                else:
                    real_all += 1

                tok_idx = m.get("tok_idx", 0)
                if tok_idx < K:
                    dropped_tok_idx += 1
                    continue

                if not m.get("word_start_valid", True):
                    dropped_not_word_start += 1
                    continue

                cand = {
                    "image_id": image_id,
                    "file_name": file_name,
                    "canon": m["canon"],
                    "word": m.get("word", ""),
                    "tok_idx": tok_idx,
                    "tok_id": m.get("tok_id"),
                    "t": tok_idx,
                    "G": G,
                    "rel_pos": m.get("rel_pos", round(tok_idx / G, 4) if G > 0 else 0.0),
                    "halluc": is_halluc,
                    "gen_ids": gen_ids,
                }

                if is_halluc:
                    h_candidates.append(cand)
                else:
                    r_candidates.append(cand)

    funnel_counts = {
        "total_captions": total_captions,
        "total_mentions": total_mentions,
        "first_mentions": first_mentions,
        "halluc_first_mentions": halluc_all,
        "real_first_mentions": real_all,
        "dropped_tok_idx_lt_K": dropped_tok_idx,
        "dropped_not_word_start": dropped_not_word_start,
        "h_candidates": len(h_candidates),
        "r_candidates": len(r_candidates),
    }

    return h_candidates, r_candidates, funnel_counts


def match_object_pairs_samecat(
    h_candidates: List[Dict[str, Any]],
    r_candidates: List[Dict[str, Any]],
    caliper: float = 0.10,
    seed: int = 0
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Matches hallucinated objects 1:1 with real objects satisfying:
      - canon(r) == canon(h)
      - image_id(r) != image_id(h)
      - |rel_pos(r) - rel_pos(h)| <= caliper
      - smallest distance, tie-break random with seed
      - without replacement
    """
    rng = random.Random(seed)

    h_pool = list(h_candidates)
    rng.shuffle(h_pool)

    # Group real candidates by category
    r_by_canon: Dict[str, List[int]] = defaultdict(list)
    for idx, r in enumerate(r_candidates):
        r_by_canon[r["canon"]].append(idx)

    available_r_indices = set(range(len(r_candidates)))

    matched_h = []
    matched_r = []
    pair_id = 0

    for h in h_pool:
        h_pos = h["rel_pos"]
        h_img = h["image_id"]
        canon = h["canon"]

        candidate_r_indices = r_by_canon.get(canon, [])
        valid_candidates = []

        for r_idx in candidate_r_indices:
            if r_idx not in available_r_indices:
                continue
            r = r_candidates[r_idx]
            if r["image_id"] == h_img:
                continue
            diff = abs(r["rel_pos"] - h_pos)
            if diff <= caliper:
                valid_candidates.append((diff, r_idx))

        if not valid_candidates:
            continue

        rng.shuffle(valid_candidates)
        valid_candidates.sort(key=lambda x: x[0])

        best_r_idx = valid_candidates[0][1]
        available_r_indices.remove(best_r_idx)

        h_item = dict(h)
        r_item = dict(r_candidates[best_r_idx])

        h_item["pair_id"] = pair_id
        r_item["pair_id"] = pair_id

        matched_h.append(h_item)
        matched_r.append(r_item)
        pair_id += 1

    dropped_h_count = len(h_pool) - len(matched_h)
    return matched_h, matched_r, dropped_h_count


def build_and_save_pairs(
    labels_file: str,
    output_dir: str,
    split_name: str = "confirm",
    caliper: float = 0.10,
    K: int = 10,
    seed: int = 0
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading candidates from {labels_file} (K={K})...")
    h_cands, r_cands, funnel = load_candidates_from_labels(labels_file, K=K)
    print(f"Candidates: {len(h_cands)} hallucinated, {len(r_cands)} real.")

    # 1. Primary matching (same category + position)
    print(f"Running primary matching (samecat, caliper={caliper}, seed={seed})...")
    h_samecat, r_samecat, dropped_samecat = match_object_pairs_samecat(
        h_cands, r_cands, caliper=caliper, seed=seed
    )
    print(f"Primary (samecat) matched pairs: {len(h_samecat)} (dropped: {dropped_samecat})")

    if len(h_samecat) < 200:
        print(f"[WARNING] Fewer than 200 same-category matched pairs found ({len(h_samecat)}). Low statistical power.")

    # 2. Secondary matching (pos_only, Sensitivity S1)
    print(f"Running secondary matching (pos_only, caliper={caliper}, seed={seed})...")
    # Add image_id != check in pos_only
    h_pool_pos = list(h_cands)
    random.Random(seed).shuffle(h_pool_pos)
    r_pool_pos = list(r_cands)
    avail_r = set(range(len(r_pool_pos)))
    rng_pos = random.Random(seed)

    h_posonly = []
    r_posonly = []
    pair_id_pos = 0

    for h in h_pool_pos:
        valid_pos = []
        for r_idx in avail_r:
            r = r_pool_pos[r_idx]
            if r["image_id"] == h["image_id"]:
                continue
            diff = abs(r["rel_pos"] - h["rel_pos"])
            if diff <= caliper:
                valid_pos.append((diff, r_idx))
        if not valid_pos:
            continue
        rng_pos.shuffle(valid_pos)
        valid_pos.sort(key=lambda x: x[0])
        best_r = valid_pos[0][1]
        avail_r.remove(best_r)

        h_item = dict(h)
        r_item = dict(r_pool_pos[best_r])
        h_item["pair_id"] = pair_id_pos
        r_item["pair_id"] = pair_id_pos
        h_posonly.append(h_item)
        r_posonly.append(r_item)
        pair_id_pos += 1

    print(f"Secondary (pos_only) matched pairs: {len(h_posonly)}")

    # Construct unified dataframe
    rows = []
    for h, r in zip(h_samecat, r_samecat):
        rows.append({
            "pair_id": f"samecat_{h['pair_id']}",
            "split": split_name,
            "match_type": "samecat",
            "h_image_id": h["image_id"],
            "h_canon": h["canon"],
            "h_t": h["t"],
            "h_G": h["G"],
            "h_rel_pos": round(h["rel_pos"], 4),
            "r_image_id": r["image_id"],
            "r_canon": r["canon"],
            "r_t": r["t"],
            "r_G": r["G"],
            "r_rel_pos": round(r["rel_pos"], 4),
        })

    for h, r in zip(h_posonly, r_posonly):
        rows.append({
            "pair_id": f"posonly_{h['pair_id']}",
            "split": split_name,
            "match_type": "pos_only",
            "h_image_id": h["image_id"],
            "h_canon": h["canon"],
            "h_t": h["t"],
            "h_G": h["G"],
            "h_rel_pos": round(h["rel_pos"], 4),
            "r_image_id": r["image_id"],
            "r_canon": r["canon"],
            "r_t": r["t"],
            "r_G": r["G"],
            "r_rel_pos": round(r["rel_pos"], 4),
        })

    pairs_csv = os.path.join(output_dir, "pairs_h1.csv")
    if pd is not None:
        pairs_df = pd.DataFrame(rows)
        pairs_df.to_csv(pairs_csv, index=False)
    else:
        import csv
        if rows:
            with open(pairs_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
    print(f"Saved {len(rows)} total pair records to {pairs_csv}")

    funnel["matched_pairs_samecat"] = len(h_samecat)
    funnel["matched_pairs_posonly"] = len(h_posonly)
    funnel["caliper"] = caliper
    funnel["K"] = K
    funnel["seed"] = seed

    funnel_json = os.path.join(output_dir, "funnel_pairs.json")
    with open(funnel_json, "w", encoding="utf-8") as f:
        json.dump(funnel, f, indent=2)
    print(f"Saved pair funnel to {funnel_json}")


def main():
    parser = argparse.ArgumentParser(description="Experiment H1: Pair Construction")
    parser.add_argument("--labels_file", type=str, default="data/labels.jsonl", help="Path to labels.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/h1", help="Output directory")
    parser.add_argument("--split", type=str, default="confirm", help="Split name (confirm or dev)")
    parser.add_argument("--caliper", type=float, default=0.10, help="Matching caliper on rel_pos (default: 0.10)")
    parser.add_argument("--K", type=int, default=10, help="Minimum token index (default: 10)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    args = parser.parse_args()
    build_and_save_pairs(
        labels_file=args.labels_file,
        output_dir=args.output_dir,
        split_name=args.split,
        caliper=args.caliper,
        K=args.K,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
