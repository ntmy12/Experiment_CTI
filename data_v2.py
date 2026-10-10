"""
Dataset loader and COCO sampling for Confirmatory and Causal Experiments (v2).
Samples N images with >= 2 distinct categories (sampling seed 2026),
strictly excluding the pilot image IDs.
"""

import os
import json
import random
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from PIL import Image

logger = logging.getLogger("confirmatory_v2")


class COCODatasetV2:
    """
    Manages COCO val2014 images, instance annotations, and disjoint sampling against pilot.
    """

    def __init__(
        self,
        val2014_dir: str,
        instances_json: str,
        pilot_image_ids_path: Optional[str] = None,
        seed: int = 2026,
    ):
        self.val2014_dir = val2014_dir
        self.instances_json = instances_json
        self.seed = seed

        self.pilot_ids: Set[int] = set()
        if pilot_image_ids_path and os.path.isfile(pilot_image_ids_path):
            with open(pilot_image_ids_path, "r", encoding="utf-8") as f:
                self.pilot_ids = set(json.load(f))
            logger.info(f"Loaded {len(self.pilot_ids)} pilot image IDs to EXCLUDE: {sorted(list(self.pilot_ids))[:5]}...")

        self.images_info: Dict[int, Dict[str, Any]] = {}
        self.cat_id_to_name: Dict[int, str] = {}
        self.name_to_cat_id: Dict[str, int] = {}
        self.image_to_annotations: Dict[int, List[Dict[str, Any]]] = {}
        self.image_to_categories: Dict[int, Set[str]] = {}
        self._cached_img_dir: Optional[str] = None

        self._load_annotations()

    @property
    def images_to_categories(self) -> Dict[int, Set[str]]:
        return self.image_to_categories

    def _load_annotations(self):
        """Loads instances_val2014.json and indexes categories and annotations."""
        if not os.path.isfile(self.instances_json):
            raise FileNotFoundError(
                f"instances_val2014.json not found at {self.instances_json}."
            )

        logger.info(f"Loading COCO annotations from {self.instances_json}...")
        with open(self.instances_json, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        for cat in coco_data.get("categories", []):
            cid = cat["id"]
            cname = cat["name"].lower().strip()
            self.cat_id_to_name[cid] = cname
            self.name_to_cat_id[cname] = cid

        for img in coco_data.get("images", []):
            iid = img["id"]
            self.images_info[iid] = img
            self.image_to_annotations[iid] = []
            self.image_to_categories[iid] = set()

        for ann in coco_data.get("annotations", []):
            iid = ann["image_id"]
            cid = ann["category_id"]
            cname = self.cat_id_to_name.get(cid, "")
            if iid in self.images_info:
                self.image_to_annotations[iid].append(ann)
                if cname:
                    self.image_to_categories[iid].add(cname)

        logger.info(
            f"Loaded {len(self.images_info)} images, {len(self.cat_id_to_name)} categories for COCO."
        )

    def _resolve_image_candidate(self, file_name: str) -> Optional[str]:
        """Fast image resolution with cached directory."""
        if self._cached_img_dir:
            cand = os.path.join(self._cached_img_dir, file_name)
            if os.path.isfile(cand):
                return cand

        # 1. Direct path
        cand = os.path.join(self.val2014_dir, file_name)
        if os.path.isfile(cand):
            self._cached_img_dir = os.path.dirname(os.path.abspath(cand))
            return cand

        # 2. Nested val2014
        cand_nested = os.path.join(self.val2014_dir, "val2014", file_name)
        if os.path.isfile(cand_nested):
            self._cached_img_dir = os.path.dirname(os.path.abspath(cand_nested))
            return cand_nested

        # 3. Subdirectories in val2014_dir
        if os.path.isdir(self.val2014_dir):
            import glob
            matches = glob.glob(os.path.join(self.val2014_dir, "**", file_name), recursive=True)
            if matches:
                self._cached_img_dir = os.path.dirname(os.path.abspath(matches[0]))
                return matches[0]

        # 4. Search in /kaggle/input if on Kaggle
        if os.path.exists("/kaggle/input"):
            import glob
            matches = glob.glob(f"/kaggle/input/**/{file_name}", recursive=True)
            if matches:
                self._cached_img_dir = os.path.dirname(os.path.abspath(matches[0]))
                return matches[0]

        return None

    def get_image_path(self, image_id: int) -> Optional[str]:
        """Returns absolute path to image file."""
        img_info = self.images_info.get(image_id)
        if img_info and "file_name" in img_info:
            cand = self._resolve_image_candidate(img_info["file_name"])
            if cand:
                return cand

        formatted_name = f"COCO_val2014_{image_id:012d}.jpg"
        return self._resolve_image_candidate(formatted_name)

    def load_image(self, image_id: int) -> Image.Image:
        """Loads and returns PIL RGB Image."""
        path = self.get_image_path(image_id)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Image {image_id} not found on disk at {self.val2014_dir}")
        return Image.open(path).convert("RGB")

    def sample_images(
        self,
        n_images: int = 150,
        sampling_seed: int = 2026,
        save_path: Optional[str] = None,
    ) -> List[int]:
        """
        Samples N images with sampling_seed=2026, having >= 2 distinct categories,
        existing on disk, and strictly excluding the pilot image IDs.
        """
        eligible_ids = []
        for iid, cats in self.image_to_categories.items():
            if len(cats) >= 2:
                # Must not be in pilot IDs
                if iid in self.pilot_ids:
                    continue
                # Must exist on disk
                img_path = self.get_image_path(iid)
                if img_path and os.path.isfile(img_path):
                    eligible_ids.append(iid)

        eligible_ids.sort()
        logger.info(
            f"Found {len(eligible_ids)} candidate images with >= 2 distinct categories "
            f"existing on disk (excluding {len(self.pilot_ids)} pilot IDs)."
        )

        if len(eligible_ids) < n_images:
            raise ValueError(
                f"Requested {n_images} images but only {len(eligible_ids)} valid images found in {self.val2014_dir}."
            )

        rng = random.Random(sampling_seed)
        sampled_ids = rng.sample(eligible_ids, n_images)
        sampled_ids.sort()

        # Strict check that selected images are completely disjoint from pilot
        assert set(sampled_ids).isdisjoint(self.pilot_ids), "Sampled IDs must be completely disjoint from pilot IDs!"

        logger.info(f"Sampled {len(sampled_ids)} new images with seed {sampling_seed}. (Disjoint check passed).")

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(sampled_ids, f, indent=2)
            logger.info(f"Saved sampled image IDs to {save_path}")

        return sampled_ids

    def get_salience_and_in_gt(self, image_id: int, category_name: str) -> Tuple[float, int]:
        """Computes bbox_area / image_area salience and in_gt indicator."""
        img_info = self.images_info.get(image_id, {})
        img_w = img_info.get("width", 0)
        img_h = img_info.get("height", 0)
        img_area = float(img_w * img_h) if (img_w and img_h) else 0.0

        anns = self.image_to_annotations.get(image_id, [])
        target_cat = category_name.lower().strip()

        matching_areas = []
        for ann in anns:
            cid = ann["category_id"]
            cname = self.cat_id_to_name.get(cid, "")
            if cname == target_cat:
                bbox = ann.get("bbox", [])
                if len(bbox) >= 4 and img_area > 0:
                    box_area = float(bbox[2] * bbox[3])
                    matching_areas.append(box_area / img_area)
                elif "area" in ann and img_area > 0:
                    matching_areas.append(float(ann["area"]) / img_area)

        if matching_areas:
            return float(max(matching_areas)), 1
        return 0.0, 0
