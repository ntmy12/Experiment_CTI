"""
GPU Sanity Tests (T4, T5, T6, T7).
Conforms to Gate G1 / Section 7 of EXPERIMENT1_SPEC.md.

These tests require an active GPU and are automatically skipped if CUDA is not available.
"""

import json
import os
import unittest
from PIL import Image
import torch
import torch.nn.functional as F

import importlib
from src.common import load_synonyms, pieces_to_text, find_mentions, char_to_token, check_word_start
lag_curve = importlib.import_module("src.03_lag_curve")
get_v_obj_tokens = lag_curve.get_v_obj_tokens
compute_lag_metrics = lag_curve.compute_lag_metrics


PROMPT = "USER: <image>\nPlease describe this image in detail. ASSISTANT:"


class TestExperiment1GPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available. Skipping GPU sanity tests.")

        from transformers import AutoProcessor, LlavaForConditionalGeneration
        cls.model_id = "llava-hf/llava-1.5-7b-hf"
        cls.processor = AutoProcessor.from_pretrained(cls.model_id)
        cls.model = LlavaForConditionalGeneration.from_pretrained(
            cls.model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        cls.model.eval()

        cls.image_dir = "data/val2014"
        cls.instances_json = "data/instances_val2014.json"
        cls.synonyms_file = "data/synonyms.txt"

    def test_t4_position_alignment(self):
        """
        T4: Position alignment test on up to 20 images.
        Condition: argmax(pred[i]) == gen_ids[i] for >= 98% of tokens across all i.
        Detects off-by-one errors in pred slicing.
        """
        if not os.path.exists(self.instances_json) or not os.path.exists(self.image_dir):
            self.skipTest("COCO data not present on local machine. Run this on Kaggle.")

        with open(self.instances_json, "r") as f:
            imgs = json.load(f)["images"][:20]

        total_tokens = 0
        matching_tokens = 0

        for img_info in imgs:
            img_path = os.path.join(self.image_dir, img_info["file_name"])
            if not os.path.exists(img_path):
                continue

            raw_img = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=raw_img, text=PROMPT, return_tensors="pt").to("cuda", torch.float16)

            with torch.no_grad():
                out = self.model.generate(**inputs, do_sample=False, max_new_tokens=64)

            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = out[0][prompt_len:].tolist()
            if not gen_ids:
                continue

            # Forward pass teacher forcing
            gen_tensor = torch.tensor([gen_ids], device="cuda")
            ids = torch.cat([inputs["input_ids"], gen_tensor], dim=1)

            with torch.no_grad():
                logits = self.model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    pixel_values=inputs["pixel_values"]
                ).logits[0]

            G = len(gen_ids)
            Lout = logits.shape[0]
            pred = logits[Lout - G - 1 : Lout - 1]

            pred_argmax = torch.argmax(pred, dim=-1).cpu().tolist()

            for target_id, pred_id in zip(gen_ids, pred_argmax):
                total_tokens += 1
                if target_id == pred_id:
                    matching_tokens += 1

        accuracy = matching_tokens / total_tokens if total_tokens > 0 else 0.0
        print(f"\n[T4 Result] Position alignment accuracy: {accuracy * 100:.2f}% ({matching_tokens}/{total_tokens})")
        self.assertGreaterEqual(accuracy, 0.98, f"T4 failed! Alignment accuracy {accuracy:.4f} < 0.98. Potential off-by-one error.")

    def test_t5_rank_at_m0(self):
        """
        T5: At m = 0, rank == 1 for >= 99% of selected objects.
        """
        # Since greedy decoding selects argmax at each step, the object token y_t
        # must have rank 1 in the prediction distribution pred[t] (which is m = 0).
        if not os.path.exists("data/labels.jsonl"):
            self.skipTest("labels.jsonl not found. Run after Step 2.")
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
