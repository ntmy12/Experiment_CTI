"""
Model loading, greedy generation, and reusable teacher-forcing forward passes for LLaVA-1.5.
Supports 2xT4 GPU sharding (device_map='auto', max_memory), SDPA with eager fallback,
clean memory hygiene, and fp16 -> fp32 casting before softmax.
"""

import gc
import logging
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image

import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TextIteratorStreamer,
)

from config import Config

logger = logging.getLogger("pilot_h21")


class VLMRunner:
    """
    Manages LLaVA model loading, generation, and teacher-forced logit extraction.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = None
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Loads model and processor according to config and hardware."""
        model_id = self.config.model_path or self.config.model_name
        logger.info(f"Loading VLM model and processor from {model_id}...")

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=self.config.hf_cache_dir,
        )
        self.tokenizer = self.processor.tokenizer

        # Determine precision and device placement
        is_cuda = torch.cuda.is_available()
        num_gpus = torch.cuda.device_count() if is_cuda else 0

        logger.info(f"CUDA available: {is_cuda}, Visible GPUs: {num_gpus}")
        if is_cuda:
            for i in range(num_gpus):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        dtype = torch.float16 if is_cuda else torch.float32

        # Device mapping and memory sharding
        if is_cuda and num_gpus > 1:
            logger.info(f"Multi-GPU detected ({num_gpus} GPUs). Using device_map='auto' with max_memory.")
            device_map = self.config.device_map
            max_memory = self.config.max_memory
        elif is_cuda:
            device_map = "auto"
            max_memory = None
        else:
            device_map = "cpu"
            max_memory = None

        attn_impl = self.config.attn_implementation
        try:
            logger.info(f"Attempting to load model with attn_implementation='{attn_impl}'...")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
                device_map=device_map,
                max_memory=max_memory,
                cache_dir=self.config.hf_cache_dir,
            )
        except Exception as e:
            logger.warning(f"Failed to load with attn_implementation='{attn_impl}': {e}. Falling back to 'eager'...")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                attn_implementation="eager",
                device_map=device_map,
                max_memory=max_memory,
                cache_dir=self.config.hf_cache_dir,
            )

        self.model.eval()
        self.device = self.model.device
        logger.info(f"Model loaded successfully. Base device: {self.device}")

    def clean_memory(self):
        """Cleans PyTorch memory cache and runs garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_peak_memory_mb(self) -> Dict[int, float]:
        """Returns peak GPU memory in MB for each CUDA device."""
        peak_mem = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                peak_mem[i] = torch.cuda.max_memory_allocated(i) / (1024.0 * 1024.0)
        return peak_mem

    @torch.inference_mode()
    def generate_caption(
        self,
        image: Image.Image,
        stream: bool = False,
    ) -> Tuple[torch.Tensor, str, int, torch.Tensor]:
        """
        Generates caption greedily for the given image.
        Returns:
            prompt_ids: [1, P_raw] input_ids of the prompt
            caption_text: decoded generated string
            T: number of generated tokens
            gen_ids: [1, T] tensor of generated token ids
        """
        inputs = self.processor(text=self.config.prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_ids = inputs["input_ids"]
        raw_prompt_len = prompt_ids.shape[1]

        streamer = None
        if stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {
            **inputs,
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "streamer": streamer,
        }

        output_ids = self.model.generate(**gen_kwargs)
        # Extract generated tokens (excluding prompt)
        gen_ids = output_ids[:, raw_prompt_len:]
        T = gen_ids.shape[1]

        caption_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        return prompt_ids, caption_text, T, gen_ids

    @torch.inference_mode()
    def get_object_logits(
        self,
        prompt_ids: torch.Tensor,
        gen_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        One reusable teacher-forcing forward pass returning logits [T, V] for all generated positions.
        Index rule:
        If P = length of prompt after processor expands <image> placeholder into image tokens,
        the logits that predict y_j are at index P + j - 1.
        Returns:
            target_logits: [T, V] cast to float32
            match_rate: proportion of greedy tokens where argmax(full_vocab_logits) == y_j
        """
        prompt_ids = prompt_ids.to(self.device)
        gen_ids = gen_ids.to(self.device)
        input_ids = torch.cat([prompt_ids, gen_ids], dim=-1)

        model_kwargs = {"input_ids": input_ids}
        if pixel_values is not None:
            model_kwargs["pixel_values"] = pixel_values.to(self.device)

        outputs = self.model(**model_kwargs)
        # Full vocab logits [1, total_len, V]
        full_logits = outputs.logits
        total_len = full_logits.shape[1]
        T = gen_ids.shape[1]
        P = total_len - T

        # Logits predicting y_0 .. y_{T-1} are at index P-1 .. P+T-2
        # Slice [P-1 : P+T-1] has length T
        target_logits = full_logits[0, P - 1 : P + T - 1, :].to(torch.float32)

        # Sanity check: argmax at index P + j - 1 should equal y_j for greedy captions
        greedy_preds = target_logits.argmax(dim=-1)
        matches = (greedy_preds == gen_ids[0]).float()
        match_rate = matches.mean().item()

        return target_logits, match_rate

    @torch.inference_mode()
    def get_no_image_logits(
        self,
        gen_ids: torch.Tensor,
        mode: str = "remove",
        image_size: Tuple[int, int] = (336, 336),
    ) -> torch.Tensor:
        """
        Computes logits [T, V] in no-image mode.
        - "remove": prompt 'USER: \\nDescribe this image in detail. ASSISTANT:' with NO image tokens/pixels.
        - "black": standard prompt with an all-black image.
        """
        gen_ids = gen_ids.to(self.device)
        T = gen_ids.shape[1]

        if mode == "remove":
            inputs = self.tokenizer(self.config.no_image_prompt, return_tensors="pt")
            prompt_ids = inputs["input_ids"].to(self.device)
            input_ids = torch.cat([prompt_ids, gen_ids], dim=-1)

            outputs = self.model(input_ids=input_ids)
            full_logits = outputs.logits
            total_len = full_logits.shape[1]
            P = total_len - T
            target_logits = full_logits[0, P - 1 : P + T - 1, :].to(torch.float32)
            return target_logits

        elif mode == "black":
            black_img = Image.new("RGB", image_size, color=(0, 0, 0))
            inputs = self.processor(text=self.config.prompt, images=black_img, return_tensors="pt")
            prompt_ids = inputs["input_ids"].to(self.device)
            pixel_values = inputs["pixel_values"].to(self.device)
            input_ids = torch.cat([prompt_ids, gen_ids], dim=-1)

            outputs = self.model(input_ids=input_ids, pixel_values=pixel_values)
            full_logits = outputs.logits
            total_len = full_logits.shape[1]
            P = total_len - T
            target_logits = full_logits[0, P - 1 : P + T - 1, :].to(torch.float32)
            return target_logits
        else:
            raise ValueError(f"Unknown no_image_mode: {mode}")

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
        # The last position in logits predicts the token at index t
        return outputs.logits[0, -1, :].to(torch.float32)
