"""
Step 2: CHAIR Labeling and Object Extraction.
Conforms to EXPERIMENT1_SPEC.md (Section 4.4, 5, 6.2).

Computes:
- Ground truth objects per image (instances_val2014 + 5 reference human captions)
- Mentions extraction via pieces_to_text + find_mentions
- Token alignment and word_start validation
- Hallucination status (canon not in GT)
- First mention flag per canonical object
- CHAIR_S and CHAIR_I summary metrics
- Writes data/labels.jsonl
"""

import argparse
import collections
import json
import os
import sys
from typing import Dict, List, Set, Tuple, Any

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tqdm import tqdm

from src.common import (
    load_synonyms,
    pieces_to_text,
    find_mentions,
    char_to_token,
    check_word_start,
    build_coco_gt,
)


def load_instances_mapping(instances_json: str) -> Dict[int, List[str]]:
    """
    Loads category names for each image from instances_val2014.json.
    """
    if not os.path.exists(instances_json):
        raise FileNotFoundError(f"Instances file not found: {instances_json}")

    print(f"Loading instances annotations from {instances_json}...")
    with open(instances_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    image_to_cats = collections.defaultdict(list)

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_name = cat_id_to_name.get(ann["category_id"])
        if cat_name:
            image_to_cats[img_id].append(cat_name)

    return image_to_cats


def load_captions_mapping(captions_json: str) -> Dict[int, List[str]]:
    """
    Loads human reference captions for each image from captions_val2014.json.
    """
    if not os.path.exists(captions_json):
        raise FileNotFoundError(f"Captions annotations file not found: {captions_json}")

    print(f"Loading reference captions from {captions_json}...")
    with open(captions_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_to_captions = collections.defaultdict(list)
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        image_to_captions[img_id].append(ann["caption"])

    return image_to_captions


def main():
    parser = argparse.ArgumentParser(description="Label captions with CHAIR and extract object mentions")
    parser.add_argument("--captions_file", type=str, default="data/captions.jsonl")
    parser.add_argument("--instances_json", type=str, default="data/instances_val2014.json")
    parser.add_argument("--captions_json", type=str, default="data/captions_val2014.json")
    parser.add_argument("--synonyms_file", type=str, default="data/synonyms.txt")
    parser.add_argument("--output_file", type=str, default="data/labels.jsonl")
    args = parser.parse_args()

    # 1. Load synonyms
    print(f"Loading synonyms from {args.synonyms_file}...")
    syn2canon, canon_list = load_synonyms(args.synonyms_file)
    print(f"Loaded {len(canon_list)} canonical categories and {len(syn2canon)} synonym terms.")

    # 2. Load GT annotations
    instances_by_image = load_instances_mapping(args.instances_json)
    captions_by_image = load_captions_mapping(args.captions_json)

    # 3. Process generated captions
    if not os.path.exists(args.captions_file):
        raise FileNotFoundError(f"Captions file not found: {args.captions_file}")

    print(f"Reading generated captions from {args.captions_file}...")
    with open(args.captions_file, "r", encoding="utf-8") as f:
        caption_lines = [line.strip() for line in f if line.strip()]

    total_images_with_objs = 0
    images_with_halluc = 0
    total_mentions_count = 0
    total_halluc_count = 0
    dropped_not_word_start = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for line in tqdm(caption_lines, desc="Labeling CHAIR", dynamic_ncols=True, unit="cap"):
            record = json.loads(line)
            image_id = record["image_id"]
            file_name = record["file_name"]
            gen_ids = record["gen_ids"]
            pieces = record["pieces"]
            G = len(gen_ids)

            # Build GT objects
            gt_objects = build_coco_gt(image_id, instances_by_image, captions_by_image, syn2canon)

            # Reconstruct text and token spans
            text, spans = pieces_to_text(pieces)
            raw_mentions = find_mentions(text, syn2canon)

            processed_mentions = []
            seen_canons = set()
            img_has_obj = False
            img_has_halluc = False

            for m in raw_mentions:
                canon = m["canon"]
                char_start = m["char_start"]
                tok_idx = char_to_token(spans, char_start)
                tok_id = gen_ids[tok_idx] if tok_idx < G else None

                # Check word start condition
                is_word_start = check_word_start(spans, tok_idx, char_start)
                if not is_word_start:
                    dropped_not_word_start += 1

                is_halluc = (canon not in gt_objects)
                is_first = (canon not in seen_canons)
                seen_canons.add(canon)

                rel_pos = round(tok_idx / G, 4) if G > 0 else 0.0

                mention_entry = {
                    "canon": canon,
                    "word": m["word"],
                    "tok_idx": tok_idx,
                    "tok_id": tok_id,
                    "halluc": is_halluc,
                    "first": is_first,
                    "rel_pos": rel_pos,
                    "word_start_valid": is_word_start
                }
                processed_mentions.append(mention_entry)

                # CHAIR accumulation (on all mentions)
                img_has_obj = True
                total_mentions_count += 1
                if is_halluc:
                    img_has_halluc = True
                    total_halluc_count += 1

            if img_has_obj:
                total_images_with_objs += 1
            if img_has_halluc:
                images_with_halluc += 1

            output_record = {
                "image_id": image_id,
                "file_name": file_name,
                "gen_ids": gen_ids,
                "mentions": processed_mentions
            }
            out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    # 4. Report CHAIR scores
    chair_s = (images_with_halluc / total_images_with_objs * 100.0) if total_images_with_objs > 0 else 0.0
    chair_i = (total_halluc_count / total_mentions_count * 100.0) if total_mentions_count > 0 else 0.0

    print("\n" + "=" * 50)
    print("CHAIR EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total evaluated captions: {len(caption_lines)}")
    print(f"Captions with >= 1 object: {total_images_with_objs}")
    print(f"Captions with hallucination: {images_with_halluc}")
    print(f"CHAIR_S: {chair_s:.2f}% (Paper benchmark ~19.6%)")
    print(f"Total object mentions: {total_mentions_count}")
    print(f"Total hallucinated mentions: {total_halluc_count}")
    print(f"CHAIR_I: {chair_i:.2f}% (Paper benchmark ~6.0%)")
    print(f"Dropped not word-start: {dropped_not_word_start}")
    print(f"Labels written to: {args.output_file}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
