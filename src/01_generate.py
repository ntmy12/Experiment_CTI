"""
Step 1: Greedy Caption Generation with LLaVA-1.5-7B on COCO val2014.
Conforms to EXPERIMENT1_SPEC.md (Section 6.1).

Features:
- Fixed prompt: "USER: <image>\\nPlease describe this image in detail. ASSISTANT:"
- Greedy decoding (do_sample=False), max_new_tokens=512
- Non-overlapping splits: dev [0, 500), confirm [500, 2500), smoke [0, 3)
- Fully resumable: skips already processed images in captions.jsonl
- Per-line flush for durability
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Set, Tuple, Any, Optional

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PIL import Image
import torch
from tqdm import tqdm


PROMPT = "USER: <image>\nPlease describe this image in detail. ASSISTANT:"


def get_image_list(instances_json: str, seed: int = 0) -> List[Dict[str, Any]]:
    """
    Loads images from COCO instances JSON, sorts by id, and shuffles with fixed seed.
    """
    if not os.path.exists(instances_json):
        raise FileNotFoundError(f"Instances file not found: {instances_json}")

    with open(instances_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = sorted(data["images"], key=lambda x: x["id"])
    rng = random.Random(seed)
    rng.shuffle(images)
    return images


def parse_split_range(split: str, offset: int = None, n_images: int = None) -> Tuple[int, int]:
    """
    Determines start and end indices for splits according to Section 6.1.
    """
    if offset is not None and n_images is not None:
        return offset, offset + n_images

    if split == "dev":
        start = offset if offset is not None else 0
        count = n_images if n_images is not None else 500
        return start, start + count
    elif split == "confirm":
        start = offset if offset is not None else 500
        count = n_images if n_images is not None else 2000
        return start, start + count
    elif split == "smoke":
        start = offset if offset is not None else 0
        count = n_images if n_images is not None else 3
        return start, start + count
    else:
        start = offset if offset is not None else 0
        count = n_images if n_images is not None else 500
        return start, start + count


def load_existing_image_ids(output_file: str) -> Set[int]:
    """
    Reads existing captions.jsonl to support resumable execution.
    """
    done = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    done.add(obj["image_id"])
                except Exception:
                    pass
    return done


def main():
    parser = argparse.ArgumentParser(description="Generate captions with LLaVA-1.5-7B")
    parser.add_argument("--image_dir", type=str, default="data/val2014", help="Path to val2014 images")
    parser.add_argument("--instances_json", type=str, default="data/instances_val2014.json", help="Path to instances_val2014.json")
    parser.add_argument("--output_file", type=str, default="data/captions.jsonl", help="Output jsonl path")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "confirm", "smoke", "custom"])
    parser.add_argument("--offset", type=int, default=None, help="Start offset in shuffled list")
    parser.add_argument("--n_images", type=int, default=None, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling images")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--load_8bit", action="store_true", help="Load model in 8-bit to fit 16GB GPUs")
    parser.add_argument("--load_4bit", action="store_true", help="Load model in 4-bit for low memory")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    # 1. Image split selection
    all_images = get_image_list(args.instances_json, seed=args.seed)
    start_idx, end_idx = parse_split_range(args.split, args.offset, args.n_images)
    selected_images = all_images[start_idx:end_idx]

    print(f"Total available images: {len(all_images)}")
    print(f"Selected split '{args.split}': indices [{start_idx}, {end_idx}) ({len(selected_images)} images)")

    # 2. Check already completed images
    done_ids = load_existing_image_ids(args.output_file)
    print(f"Found {len(done_ids)} existing generated captions in {args.output_file}")

    todo_images = [img for img in selected_images if img["id"] not in done_ids]
    print(f"Images remaining to process: {len(todo_images)}")

    if not todo_images:
        print("All target images already generated. Exiting.")
        return

    # 3. Load model and processor
    print(f"Loading processor and model '{args.model_id}' on {args.device}...")
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_id)

    model_kwargs = {"torch_dtype": torch.float16 if args.device == "cuda" else torch.float32}
    if args.load_8bit:
        model_kwargs["load_in_8bit"] = True
    elif args.load_4bit:
        model_kwargs["load_in_4bit"] = True

    device_map = "auto" if args.device == "cuda" else None
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map=device_map,
        **model_kwargs
    )
    if device_map is None:
        model.to(args.device)
    model.eval()

    target_device = model.device if hasattr(model, "device") else args.device

    tokenizer = processor.tokenizer
    eos_id = tokenizer.eos_token_id

    # 4. Generate loop
    with open(args.output_file, "a", encoding="utf-8") as out_f:
        for img_info in tqdm(todo_images, desc=f"Generating {args.split}", dynamic_ncols=True, unit="img"):
            img_id = img_info["id"]
            file_name = img_info["file_name"]
            img_path = os.path.join(args.image_dir, file_name)

            if not os.path.exists(img_path):
                print(f"Warning: Image file not found: {img_path}. Skipping.")
                continue

            try:
                raw_image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error reading image {img_path}: {e}. Skipping.")
                continue

            # Process input
            inputs = processor(images=raw_image, text=PROMPT, return_tensors="pt")
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )

            # Slice only generated tokens
            gen_ids = outputs[0][prompt_len:].tolist()

            # Remove trailing EOS token if present
            if gen_ids and gen_ids[-1] == eos_id:
                gen_ids = gen_ids[:-1]

            # Convert token ids to token pieces
            pieces = tokenizer.convert_ids_to_tokens(gen_ids)

            record = {
                "image_id": img_id,
                "file_name": file_name,
                "gen_ids": gen_ids,
                "pieces": pieces
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"Caption generation finished. Saved to {args.output_file}")


if __name__ == "__main__":
    main()
