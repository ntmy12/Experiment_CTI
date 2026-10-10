"""
VLM Runner for LLaVA-1.5-7B in Confirmatory and Causal Experiments (v2).
Provides:
- ONE reusable get_logits function with index rule P+j-1 and argmax sanity check.
- Greedy caption generation.
- Teacher-forced perturbed forward passes.
- Free-generation check with custom StoppingCriteria (E2).
- Raw pixel space image blending for causal interventions (E3).
"""

import time
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from PIL import Image

import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)

logger = logging.getLogger("confirmatory_v2")


class ObjectOrEOSStoppingCriteria(StoppingCriteria):
    """
    Custom stopping criteria for E2 (Free Generation):
    Stops as soon as the last generated token is in obj_ids or is EOS/special.
    """

    def __init__(self, obj_ids_set: Set[int], stop_ids_set: Set[int], initial_len: int):
        super().__init__()
        self.obj_ids_set = obj_ids_set
        self.stop_ids_set = stop_ids_set
        self.initial_len = initial_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] <= self.initial_len:
            return False
        last_tok = int(input_ids[0, -1].item())
        if last_tok in self.obj_ids_set or last_tok in self.stop_ids_set:
            return True
        return False


def blend_image_raw(original_img: Image.Image, lambda_val: float) -> Image.Image:
    """
    Blends black image with original PIL image in RAW pixel space:
    x_lambda = Image.blend(black_image, original_PIL_image, lambda)
    When lambda=1.0 -> 100% original image.
    When lambda=0.0 -> 100% black image.
    """
    black_img = Image.new("RGB", original_img.size, color=(0, 0, 0))
    if lambda_val >= 1.0:
        return original_img
    if lambda_val <= 0.0:
        return black_img
    return Image.blend(black_img, original_img, float(lambda_val))


class VLMRunnerV2:
    """
    Manages LLaVA-1.5-7B loading, greedy generation, single-pass logits extraction,
    and free-generation interventions.
    """

    def __init__(self, config: Any):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """Loads processor and LlavaForConditionalGeneration in float16 with SDPA."""
        model_id = self.config.model_path or self.config.model_name
        logger.info(f"Loading processor from {model_id}...")
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=self.config.hf_cache_dir,
        )
        self.tokenizer = self.processor.tokenizer

        logger.info(f"Loading LlavaForConditionalGeneration from {model_id}...")
        dtype = torch.float16 if self.config.torch_dtype == "float16" else torch.float32

        model_kwargs = {
            "torch_dtype": dtype,
            "cache_dir": self.config.hf_cache_dir,
            "device_map": self.config.device_map,
            "max_memory": self.config.max_memory if torch.cuda.is_available() else None,
        }

        # Attempt SDPA with eager fallback
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation=self.config.attn_implementation,
                **model_kwargs,
            )
            logger.info("Successfully loaded model with SDPA attention implementation.")
        except Exception as e:
            logger.warning(f"Could not load with SDPA ({e}), falling back to eager attention.")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation="eager",
                **model_kwargs,
            )

        self.model.eval()

    def get_peak_memory_gb(self) -> Dict[int, float]:
        """Returns peak GPU memory in GB per device."""
        res = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_bytes = torch.cuda.max_memory_allocated(i)
                res[i] = round(mem_bytes / (1024 ** 3), 2)
        return res

    def reset_peak_memory(self):
        """Resets peak memory stats."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate_caption(
        self,
        pil_image: Image.Image,
    ) -> Tuple[torch.Tensor, str, int, torch.Tensor]:
        """
        Runs greedy caption generation with prompt up to max_new_tokens.
        Returns:
            prompt_ids: Tensor [1, P]
            caption_text: Decoded generated caption
            T: Length of generated sequence
            gen_ids: Tensor [1, T]
        """
        inputs = self.processor(text=self.config.prompt, images=pil_image, return_tensors="pt")
        prompt_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device)
        P = prompt_ids.shape[1]

        generate_kwargs = {
            "input_ids": prompt_ids,
            "pixel_values": pixel_values,
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }

        output_ids = self.model.generate(**generate_kwargs)
        gen_ids = output_ids[:, P:]
        T = gen_ids.shape[1]

        caption_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        return prompt_ids, caption_text, T, gen_ids

    @torch.inference_mode()
    def get_logits(
        self,
        prompt_ids: torch.Tensor,
        gen_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        ONE reusable function returning float32 logits [T, V] for all generated positions in ONE forward pass.
        Index rule: if P = prompt length after processor expands <image>,
        the logits predicting y_j are at index P + j - 1.
        Sanity check: argmax at index P + j - 1 equals y_j (reports match rate; requires >= 95%).
        """
        prompt_ids = prompt_ids.to(self.device)
        gen_ids = gen_ids.to(self.device)
        input_ids = torch.cat([prompt_ids, gen_ids], dim=-1)

        model_kwargs = {"input_ids": input_ids}
        if pixel_values is not None:
            model_kwargs["pixel_values"] = pixel_values.to(self.device)

        outputs = self.model(**model_kwargs)
        full_logits = outputs.logits
        total_len = full_logits.shape[1]
        T = gen_ids.shape[1]
        P = total_len - T

        # Logits predicting y_0 .. y_{T-1} are at index P-1 .. P+T-2 (slice [P-1 : P+T-1])
        target_logits = full_logits[0, P - 1 : P + T - 1, :].to(torch.float32)

        greedy_preds = target_logits.argmax(dim=-1)
        matches = (greedy_preds == gen_ids[0]).float()
        match_rate = float(matches.mean().item())

        return target_logits, match_rate

    @torch.inference_mode()
    def get_no_image_logits(
        self,
        gen_ids: torch.Tensor,
        mode: str = "black",
        image_size: Tuple[int, int] = (336, 336),
    ) -> torch.Tensor:
        """
        Computes float32 logits [T, V] in no-image mode.
        - 'black': standard prompt with all-black image (preserves token positions exactly).
        - 'remove': no-image prompt without image placeholder.
        """
        gen_ids = gen_ids.to(self.device)
        T = gen_ids.shape[1]

        if mode == "black":
            black_img = Image.new("RGB", image_size, color=(0, 0, 0))
            inputs = self.processor(text=self.config.prompt, images=black_img, return_tensors="pt")
            prompt_ids = inputs["input_ids"].to(self.device)
            pixel_values = inputs["pixel_values"].to(self.device)
            target_logits, _ = self.get_logits(prompt_ids, gen_ids, pixel_values)
            return target_logits

        elif mode == "remove":
            inputs = self.tokenizer(self.config.no_image_prompt, return_tensors="pt")
            prompt_ids = inputs["input_ids"].to(self.device)
            input_ids = torch.cat([prompt_ids, gen_ids], dim=-1)

            outputs = self.model(input_ids=input_ids)
            full_logits = outputs.logits
            total_len = full_logits.shape[1]
            P = total_len - T
            target_logits = full_logits[0, P - 1 : P + T - 1, :].to(torch.float32)
            return target_logits
        else:
            raise ValueError(f"Unknown no_image mode: {mode}")

    @torch.inference_mode()
    def get_perturbed_target_logits(
        self,
        prompt_ids: torch.Tensor,
        gen_ids_perturbed_prefix: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Runs forward pass on perturbed prefix up to t (length t tokens),
        returning float32 logits [V] at the last position predicting position t.
        """
        prompt_ids = prompt_ids.to(self.device)
        gen_ids_perturbed_prefix = gen_ids_perturbed_prefix.to(self.device)
        pixel_values = pixel_values.to(self.device)

        input_ids = torch.cat([prompt_ids, gen_ids_perturbed_prefix], dim=-1)
        outputs = self.model(input_ids=input_ids, pixel_values=pixel_values)
        return outputs.logits[0, -1, :].to(torch.float32)

    @torch.inference_mode()
    def free_generate_until_object(
        self,
        prompt_ids: torch.Tensor,
        gen_ids_up_to_s: torch.Tensor,
        alt_token_id: int,
        pixel_values: torch.Tensor,
        obj_ids_set: Set[int],
        max_new_tokens: int = 24,
    ) -> List[int]:
        """
        Runs greedy free generation starting from perturbed prefix:
        input_ids = prompt_ids + y_<s + [a].
        Stops when the last generated token is in obj_ids or EOS (max 24 new tokens).
        Returns list of newly generated token IDs.
        """
        prompt_ids = prompt_ids.to(self.device)
        prefix_s = gen_ids_up_to_s.to(self.device)
        alt_tensor = torch.tensor([[alt_token_id]], dtype=torch.long, device=self.device)

        input_ids = torch.cat([prompt_ids, prefix_s, alt_tensor], dim=-1)
        initial_len = input_ids.shape[1]

        stop_ids_set = set(self.tokenizer.all_special_ids)
        stopping_criteria = StoppingCriteriaList([
            ObjectOrEOSStoppingCriteria(
                obj_ids_set=obj_ids_set,
                stop_ids_set=stop_ids_set,
                initial_len=initial_len,
            )
        ])

        outputs = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values.to(self.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            stopping_criteria=stopping_criteria,
            use_cache=True,
        )

        newly_gen = outputs[0, initial_len:].tolist()
        return newly_gen
