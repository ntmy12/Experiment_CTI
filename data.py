"""
Dataset loader and COCO sampling for Pilot Experiment H2.1.
Samples 20 images with >= 2 distinct annotated categories (seed 42),
computes ground-truth instance annotations and salience covariates.
"""

import os
import json
import random
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from PIL import Image

logger = logging.getLogger("pilot_h21")


class COCODataset:
    """
    Manages COCO val2014 images and instances annotations.
    """

    def __init__(self, val2014_dir: str, instances_json: str, seed: int = 42):
        self.val2014_dir = val2014_dir
        self.instances_json = instances_json
        self.seed = seed

        self.images_info: Dict[int, Dict[str, Any]] = {}
        self.cat_id_to_name: Dict[int, str] = {}
        self.name_to_cat_id: Dict[str, int] = {}
        self.image_to_annotations: Dict[int, List[Dict[str, Any]]] = {}
        self.image_to_categories: Dict[int, Set[str]] = {}

        self._load_annotations()

    def _load_annotations(self):
        """Loads instances_val2014.json and indexes annotations by image_id."""
        if not os.path.isfile(self.instances_json):
            from config import ensure_coco_annotations
            self.instances_json = ensure_coco_annotations(self.instances_json)

        if not os.path.isfile(self.instances_json):
            raise FileNotFoundError(
                f"instances_val2014.json not found at {self.instances_json}. "
                "Ensure COCO annotations exist or specify correct path."
            )

        logger.info(f"Loading COCO annotations from {self.instances_json}...")
        with open(self.instances_json, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # Index categories
        for cat in coco_data.get("categories", []):
            cid = cat["id"]
            cname = cat["name"].lower().strip()
            self.cat_id_to_name[cid] = cname
            self.name_to_cat_id[cname] = cid

        # Index images
        for img in coco_data.get("images", []):
            iid = img["id"]
            self.images_info[iid] = img
            self.image_to_annotations[iid] = []
            self.image_to_categories[iid] = set()

        # Index annotations
        for ann in coco_data.get("annotations", []):
            iid = ann["image_id"]
            cid = ann["category_id"]
            cname = self.cat_id_to_name.get(cid, "")
            if iid in self.images_info:
                self.image_to_annotations[iid].append(ann)
                if cname:
                    self.image_to_categories[iid].add(cname)

        logger.info(
            f"Loaded {len(self.images_info)} images, {len(self.cat_id_to_name)} categories, "
            f"and annotations for COCO."
        )

    def sample_images(self, n_images: int = 20, save_path: Optional[str] = None) -> List[int]:
        """
        Samples n_images with seed 42 among images that have >= 2 distinct annotated categories,
        and whose image file exists on disk.
        Saves sampled image IDs to results/image_ids.json.
        """
        # Find eligible images
        eligible_ids = []
        for iid, cats in self.images_to_categories.items():
            if len(cats) >= 2:
                # Check if image file actually exists on disk
                img_path = self.get_image_path(iid)
                if img_path and os.path.isfile(img_path):
                    eligible_ids.append(iid)

        eligible_ids.sort()
        logger.info(f"Found {len(eligible_ids)} images with >= 2 distinct categories existing on disk.")

        if len(eligible_ids) < n_images:
            raise ValueError(
                f"Requested {n_images} images but only {len(eligible_ids)} valid images found in {self.val2014_dir}."
            )

        rng = random.Random(self.seed)
        sampled_ids = rng.sample(eligible_ids, n_images)
        sampled_ids.sort()

        logger.info(f"Sampled {len(sampled_ids)} images with seed {self.seed}: {sampled_ids}")

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(sampled_ids, f, indent=2)
            logger.info(f"Saved sampled image IDs to {save_path}")

        return sampled_ids

    def get_image_path(self, image_id: int) -> Optional[str]:
        """Returns the absolute path to the image file, trying standard COCO naming."""
        img_info = self.images_info.get(image_id)
        if img_info and "file_name" in img_info:
            cand = os.path.join(self.val2014_dir, img_info["file_name"])
            if os.path.isfile(cand):
                return cand

        # Default COCO val2014 format: COCO_val2014_000000000000.jpg
        formatted_name = f"COCO_val2014_{image_id:012d}.jpg"
        cand = os.path.join(self.val2014_dir, formatted_name)
        if os.path.isfile(cand):
            return cand

        return None

    def load_image(self, image_id: int) -> Image.Image:
        """Loads and returns PIL RGB Image."""
        path = self.get_image_path(image_id)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Image {image_id} not found at {path}")
        return Image.open(path).convert("RGB")

    def get_salience_and_in_gt(self, image_id: int, category_name: str) -> Tuple[float, int]:
        """
        Computes:
        - salience: max over GT instances of the matched COCO category of (bbox area / image area);
                    0 if the category is not annotated in the image.
        - in_gt: 1 if the category is annotated in the image, else 0.
        """
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
                bbox = ann.get("bbox", [])  # [x, y, w, h]
                if len(bbox) >= 4 and img_area > 0:
                    box_area = float(bbox[2] * bbox[3])
                    matching_areas.append(box_area / img_area)
                elif "area" in ann and img_area > 0:
                    matching_areas.append(float(ann["area"]) / img_area)

        if matching_areas:
            return float(max(matching_areas)), 1
        return 0.0, 0
