"""
Configuration for Pilot Experiment H2.1 on VLM (COCO val2014, 20 images).
Defines all paths, model hyperparameters, perturbation settings, and auto-discovery fallbacks.
"""

import os
import glob
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("pilot_h21")


def auto_discover_coco_paths(
    preferred_img_dir: Optional[str] = None,
    preferred_ann_path: Optional[str] = None,
    search_roots: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Finds valid COCO val2014 image directory and instances_val2014.json file.
    If preferred paths are valid, uses them. Otherwise searches recursively in search_roots.
    Prints every candidate found and returns the first valid pair.
    """
    # 1. Check preferred paths if explicitly provided and exist
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


def ensure_coco_annotations(ann_path: Optional[str], target_dir: str = "data") -> str:
    """
    Ensures instances_val2014.json exists.
    If the file is not found locally or in /kaggle/input, downloads and extracts
    the official COCO 2014 validation annotations.
    """
    if ann_path and os.path.isfile(ann_path):
        return os.path.abspath(ann_path)

    local_target = os.path.join(target_dir, "instances_val2014.json")
    if os.path.isfile(local_target):
        return os.path.abspath(local_target)

    # Search again in case it exists in subdirectories
    for root in ["/kaggle/input", "./data", "../data", "."]:
        if os.path.exists(root):
            for dirpath, _, filenames in os.walk(root):
                for f in filenames:
                    if f.lower() == "instances_val2014.json":
                        return os.path.abspath(os.path.join(dirpath, f))

    logger.info(
        "instances_val2014.json not found in input directories. "
        "Downloading official COCO 2014 validation annotations (~241 MB)..."
    )
    import urllib.request
    import zipfile

    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "annotations_trainval2014.zip")
    url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if "instances_val2014.json" in member:
                with zf.open(member) as src, open(local_target, "wb") as dst:
                    dst.write(src.read())
                break

    if os.path.isfile(zip_path):
        try:
            os.remove(zip_path)
        except Exception:
            pass

    logger.info(f"Successfully downloaded and extracted annotations to {local_target}")
    return os.path.abspath(local_target)


@dataclass
class Config:
    # Pilot scope
    seed: int = 42
    n_images: int = 20

    # Model settings
    model_name: str = "llava-hf/llava-1.5-7b-hf"
    model_path: Optional[str] = None  # Local override path
    hf_cache_dir: str = "/kaggle/working/hf_cache" if os.path.exists("/kaggle") else "./hf_cache"
    prompt: str = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
    no_image_prompt: str = "USER: \nDescribe this image in detail. ASSISTANT:"
    max_new_tokens: int = 200
    do_sample: bool = False  # Greedy decoding
    torch_dtype: str = "float16"
    attn_implementation: str = "sdpa"  # sdpa with eager fallback
    device_map: str = "auto"
    max_memory: Dict[int, str] = field(default_factory=lambda: {0: "13GiB", 1: "13GiB"})

    # No-image mode for V_t: "remove" (default) or "black"
    no_image_mode: str = "remove"

    # COCO Paths
    val2014_dir: str = "/kaggle/input/coco-2014-dataset-for-yolov3/coco2014/val2014"
    instances_json: str = "/kaggle/input/coco-2014-dataset-for-yolov3/coco2014/annotations/instances_val2014.json"
    synonyms_file: str = "data/synonyms.txt"

    # Output paths
    output_dir: str = "/kaggle/working/results" if os.path.exists("/kaggle") else "./results"

    # Perturbation parameters
    window_size: int = 8  # s in [t-8, t-1]
    max_sites_per_object: int = 3
    top_k_candidates: int = 5  # model's own top-5 full-vocab alternatives

    # Analysis settings
    cluster_bootstrap_resamples: int = 2000

    def resolve_paths(self):
        """Resolves COCO paths via auto-discovery fallback if default paths don't exist."""
        # Adjust output dir and hf cache dir if not on kaggle
        if not os.path.exists("/kaggle") and self.output_dir.startswith("/kaggle"):
            self.output_dir = "./results"
        if not os.path.exists("/kaggle") and self.hf_cache_dir.startswith("/kaggle"):
            self.hf_cache_dir = "./hf_cache"

        os.makedirs(self.output_dir, exist_ok=True)
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

        # If instances_json still doesn't exist on disk, auto-download
        if not os.path.isfile(self.instances_json):
            self.instances_json = ensure_coco_annotations(self.instances_json)

    @property
    def captions_path(self) -> str:
        return os.path.join(self.output_dir, "captions.json")

    @property
    def image_ids_path(self) -> str:
        return os.path.join(self.output_dir, "image_ids.json")

    @property
    def records_path(self) -> str:
        return os.path.join(self.output_dir, "records.csv")

    @property
    def records_black_path(self) -> str:
        return os.path.join(self.output_dir, "records_v_black.csv")

    @property
    def dropped_words_path(self) -> str:
        return os.path.join(self.output_dir, "dropped_object_words.txt")

    @property
    def analysis_json_path(self) -> str:
        return os.path.join(self.output_dir, "analysis.json")

    @property
    def analysis_md_path(self) -> str:
        return os.path.join(self.output_dir, "analysis.md")

    @property
    def report_html_path(self) -> str:
        return os.path.join(self.output_dir, "report.html")

    @property
    def log_path(self) -> str:
        return os.path.join(self.output_dir, "run.log")

    @property
    def progress_jsonl_path(self) -> str:
        return os.path.join(self.output_dir, "progress.jsonl")
