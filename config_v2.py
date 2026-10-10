"""
Configuration module for Confirmatory and Causal Experiments (v2) on VLM.
Defines all paths, model hyperparameters, perturbation settings, and auto-discovery fallbacks.
"""

import os
import glob
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("confirmatory_v2")


def auto_discover_coco_paths(
    preferred_img_dir: Optional[str] = None,
    preferred_ann_path: Optional[str] = None,
    search_roots: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Finds valid COCO val2014 image directory and instances_val2014.json file.
    If preferred paths are valid, uses them. Otherwise searches recursively in search_roots.
    """
    val_img = preferred_img_dir if (preferred_img_dir and os.path.isdir(preferred_img_dir)) else None
    val_ann = preferred_ann_path if (preferred_ann_path and os.path.isfile(preferred_ann_path)) else None

    if val_img and val_ann:
        return val_img, val_ann

    if search_roots is None:
        search_roots = ["/kaggle/input", "./data", "../data", os.path.expanduser("~/data"), "."]

    candidate_dirs = []
    candidate_anns = []

    for root in search_roots:
        if not os.path.exists(root):
            continue
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                for d in dirnames:
                    if d.lower() == "val2014":
                        candidate_dirs.append(os.path.join(dirpath, d))
                for f in filenames:
                    if f.lower() == "instances_val2014.json":
                        candidate_anns.append(os.path.join(dirpath, f))
        except Exception as e:
            logger.warning(f"Error scanning {root} during COCO path discovery: {e}")

    logger.info("COCO auto-discovery candidates found:")
    logger.info(f"  val2014 directories ({len(candidate_dirs)}): {candidate_dirs}")
    logger.info(f"  instances_val2014.json files ({len(candidate_anns)}): {candidate_anns}")

    selected_dir = val_img or (candidate_dirs[0] if candidate_dirs else None)
    selected_ann = val_ann or (candidate_anns[0] if candidate_anns else None)

    return selected_dir, selected_ann


def find_pilot_image_ids_file(preferred_path: Optional[str] = None) -> str:
    """Finds pilot image_ids.json to exclude from v2 sampling."""
    candidates = [
        preferred_path,
        "data/pilot_image_ids.json",
        "results/pilot_h21_results/image_ids.json",
        "results/image_ids.json",
        "/kaggle/working/Experiment_CTI/data/pilot_image_ids.json",
        "/kaggle/working/results/image_ids.json",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)

    # Search in working directory or repo
    for root in [".", "/kaggle/working", "/kaggle/input"]:
        if os.path.exists(root):
            matches = glob.glob(os.path.join(root, "**/pilot_image_ids.json"), recursive=True)
            if matches:
                return os.path.abspath(matches[0])
            matches2 = glob.glob(os.path.join(root, "**/image_ids.json"), recursive=True)
            if matches2:
                return os.path.abspath(matches2[0])

    raise FileNotFoundError(
        "Pilot image_ids.json not found! Cannot ensure disjoint sampling. "
        "Provide --pilot_image_ids or ensure data/pilot_image_ids.json exists."
    )


@dataclass
class ConfigV2:
    # Sampling & Scope
    n_images: int = 150
    sampling_seed: int = 2026
    seed: int = 2026

    # Model settings
    model_name: str = "llava-hf/llava-1.5-7b-hf"
    model_path: Optional[str] = None
    hf_cache_dir: str = "/kaggle/working/hf_cache" if os.path.exists("/kaggle") else "./hf_cache"
    prompt: str = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
    no_image_prompt: str = "USER: \nDescribe this image in detail. ASSISTANT:"
    max_new_tokens: int = 200
    do_sample: bool = False
    torch_dtype: str = "float16"
    attn_implementation: str = "sdpa"
    device_map: str = "auto"
    max_memory: Dict[int, str] = field(default_factory=lambda: {0: "13GiB", 1: "13GiB"})

    # COCO Paths
    val2014_dir: str = "/kaggle/input/datasets/biminhco/val2014"
    instances_json: str = "/kaggle/input/datasets/zhenhoblngjia/annotations-trainval2014/annotations/instances_val2014.json"
    pilot_image_ids_path: Optional[str] = "data/pilot_image_ids.json"
    synonyms_file: str = "data/synonyms.txt"

    # Output directory
    output_dir: str = "/kaggle/working/results_v2" if os.path.exists("/kaggle") else "./results_v2"

    # E1 & E2 parameters
    e2_max_per_image: int = 12
    e2_max_new_tokens: int = 24

    # E3 parameters
    e3_n_near: int = 400
    e3_n_far: int = 100
    lambdas: List[float] = field(default_factory=lambda: [1.0, 0.75, 0.5, 0.25, 0.0])
    temperatures: List[float] = field(default_factory=lambda: [1.0, 1.25, 1.5, 2.0, 3.0, 4.0])

    # Analysis settings
    cluster_bootstrap_resamples: int = 2000

    def resolve_paths(self):
        """Resolves COCO paths and pilot exclusion list."""
        if not os.path.exists("/kaggle") and self.output_dir.startswith("/kaggle"):
            self.output_dir = "./results_v2"
        if not os.path.exists("/kaggle") and self.hf_cache_dir.startswith("/kaggle"):
            self.hf_cache_dir = "./hf_cache"

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "figs"), exist_ok=True)
        os.makedirs(self.hf_cache_dir, exist_ok=True)

        found_img, found_ann = auto_discover_coco_paths(
            preferred_img_dir=self.val2014_dir,
            preferred_ann_path=self.instances_json,
        )
        if found_img:
            self.val2014_dir = found_img
        if found_ann:
            self.instances_json = found_ann

        self.pilot_image_ids_path = find_pilot_image_ids_file(self.pilot_image_ids_path)

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.output_dir, "cache")

    @property
    def captions_path(self) -> str:
        return os.path.join(self.output_dir, "captions.json")

    @property
    def image_ids_path(self) -> str:
        return os.path.join(self.output_dir, "image_ids.json")

    @property
    def dropped_words_path(self) -> str:
        return os.path.join(self.output_dir, "dropped_object_words.txt")

    @property
    def records_e1_path(self) -> str:
        return os.path.join(self.output_dir, "records_e1.csv")

    @property
    def records_e2_path(self) -> str:
        return os.path.join(self.output_dir, "records_e2.csv")

    @property
    def records_e3_path(self) -> str:
        return os.path.join(self.output_dir, "records_e3.csv")

    @property
    def records_e3_temp_path(self) -> str:
        return os.path.join(self.output_dir, "records_e3_temp.csv")

    @property
    def analysis_json_path(self) -> str:
        return os.path.join(self.output_dir, "analysis.json")

    @property
    def analysis_md_path(self) -> str:
        return os.path.join(self.output_dir, "analysis.md")

    @property
    def verdicts_md_path(self) -> str:
        return os.path.join(self.output_dir, "verdicts.md")

    @property
    def deviations_md_path(self) -> str:
        return os.path.join(self.output_dir, "DEVIATIONS.md")

    @property
    def report_html_path(self) -> str:
        return os.path.join(self.output_dir, "report.html")

    @property
    def log_path(self) -> str:
        return os.path.join(self.output_dir, "run.log")

    @property
    def progress_jsonl_path(self) -> str:
        return os.path.join(self.output_dir, "progress.jsonl")
