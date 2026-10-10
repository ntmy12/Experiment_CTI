"""
Object vocabulary management for Pilot Experiment H2.1.
Maps words to COCO categories, filters to single-token words under the model's tokenizer,
and manages restricted object distribution indices.
"""

import os
import logging
from typing import Dict, List, Set, Tuple, Optional

logger = logging.getLogger("pilot_h21")

# Default mapping for 80 COCO categories and common synonyms/plurals
DEFAULT_COCO_SYNONYMS = {
    # Person
    "person": "person", "people": "person", "man": "person", "men": "person",
    "woman": "person", "women": "person", "boy": "person", "boys": "person",
    "girl": "person", "girls": "person", "child": "person", "children": "person",
    "kid": "person", "kids": "person", "baby": "person", "babies": "person",
    "guy": "person", "guys": "person", "human": "person", "humans": "person",
    "player": "person", "players": "person", "rider": "person", "riders": "person",
    # Vehicles
    "bicycle": "bicycle", "bicycles": "bicycle", "bike": "bicycle", "bikes": "bicycle",
    "car": "car", "cars": "car", "automobile": "car", "vehicle": "car", "vehicles": "car",
    "motorcycle": "motorcycle", "motorcycles": "motorcycle", "scooter": "motorcycle",
    "airplane": "airplane", "airplanes": "airplane", "plane": "airplane", "planes": "airplane", "jet": "airplane",
    "bus": "bus", "buses": "bus",
    "train": "train", "trains": "train", "subway": "train",
    "truck": "truck", "trucks": "truck", "van": "truck", "vans": "truck",
    "boat": "boat", "boats": "boat", "ship": "boat", "ships": "boat", "canoe": "boat", "kayak": "boat",
    # Outdoor
    "traffic light": "traffic light", "fire hydrant": "fire hydrant", "stop sign": "stop sign",
    "parking meter": "parking meter", "bench": "bench", "benches": "bench",
    # Animals
    "bird": "bird", "birds": "bird", "pigeon": "bird", "duck": "bird", "goose": "bird", "seagull": "bird",
    "cat": "cat", "cats": "cat", "kitten": "cat", "kittens": "cat",
    "dog": "dog", "dogs": "dog", "puppy": "dog", "puppies": "dog",
    "horse": "horse", "horses": "horse", "pony": "horse",
    "sheep": "sheep", "lamb": "sheep", "cow": "cow", "cows": "cow", "cattle": "cow", "bull": "cow",
    "elephant": "elephant", "elephants": "elephant",
    "bear": "bear", "bears": "bear", "zebra": "zebra", "zebras": "zebra",
    "giraffe": "giraffe", "giraffes": "giraffe",
    # Accessories
    "backpack": "backpack", "backpacks": "backpack", "bag": "handbag", "bags": "handbag",
    "umbrella": "umbrella", "umbrellas": "umbrella",
    "handbag": "handbag", "handbags": "handbag", "purse": "handbag",
    "tie": "tie", "ties": "tie", "suitcase": "suitcase", "suitcases": "suitcase", "luggage": "suitcase",
    # Sports
    "frisbee": "frisbee", "skis": "skis", "ski": "skis", "snowboard": "snowboard",
    "ball": "sports ball", "balls": "sports ball", "kite": "kite", "kites": "kite",
    "baseball bat": "baseball bat", "baseball glove": "baseball glove",
    "skateboard": "skateboard", "skateboards": "skateboard",
    "surfboard": "surfboard", "surfboards": "surfboard",
    "tennis racket": "tennis racket", "racket": "tennis racket",
    # Kitchen & Food
    "bottle": "bottle", "bottles": "bottle", "wine glass": "wine glass",
    "cup": "cup", "cups": "cup", "mug": "cup", "mugs": "cup",
    "fork": "fork", "forks": "fork", "knife": "knife", "knives": "knife",
    "spoon": "spoon", "spoons": "spoon", "bowl": "bowl", "bowls": "bowl",
    "banana": "banana", "bananas": "banana", "apple": "apple", "apples": "apple",
    "sandwich": "sandwich", "sandwiches": "sandwich", "burger": "sandwich", "burgers": "sandwich",
    "orange": "orange", "oranges": "orange", "broccoli": "broccoli",
    "carrot": "carrot", "carrots": "carrot", "hot dog": "hot dog",
    "pizza": "pizza", "pizzas": "pizza", "donut": "donut", "donuts": "donut",
    "cake": "cake", "cakes": "cake",
    # Furniture
    "chair": "chair", "chairs": "chair", "seat": "chair", "seats": "chair", "stool": "chair",
    "couch": "couch", "couches": "couch", "sofa": "couch", "sofas": "couch",
    "plant": "potted plant", "plants": "potted plant", "potted plant": "potted plant",
    "bed": "bed", "beds": "bed", "table": "dining table", "tables": "dining table",
    "desk": "dining table", "dining table": "dining table",
    "toilet": "toilet", "toilets": "toilet",
    # Electronics & Appliances
    "tv": "tv", "television": "tv", "monitor": "tv", "screen": "tv",
    "laptop": "laptop", "laptops": "laptop", "computer": "laptop",
    "mouse": "mouse", "remote": "remote", "remotes": "remote",
    "keyboard": "keyboard", "keyboards": "keyboard",
    "phone": "cell phone", "phones": "cell phone", "cell phone": "cell phone", "cellphone": "cell phone",
    "microwave": "microwave", "oven": "oven", "toaster": "toaster",
    "sink": "sink", "sinks": "sink", "refrigerator": "refrigerator", "fridge": "refrigerator",
    # Indoor items
    "book": "book", "books": "book", "clock": "clock", "clocks": "clock", "watch": "clock",
    "vase": "vase", "vases": "vase", "scissors": "scissors",
    "teddy bear": "teddy bear", "teddy": "teddy bear",
    "hair drier": "hair drier", "toothbrush": "toothbrush"
}


def load_synonyms_from_file(path: str) -> Dict[str, str]:
    """Loads synonyms dictionary (word -> coco_category) from synonyms.txt."""
    if not os.path.exists(path):
        logger.warning(f"Synonyms file not found at {path}, using default dictionary.")
        return DEFAULT_COCO_SYNONYMS.copy()

    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                canon_part, syns_part = line.split(":", 1)
                canon = canon_part.strip().lower()
                words = [canon] + [s.strip().lower() for s in syns_part.split(",") if s.strip()]
            elif "," in line:
                words = [s.strip().lower() for s in line.split(",") if s.strip()]
                canon = words[0] if words else ""
            else:
                words = [s.strip().lower() for s in line.split() if s.strip()]
                canon = words[0] if words else ""

            if canon and words:
                for w in words:
                    mapping[w] = canon

    return mapping


class ObjectVocabulary:
    """
    Manages the restricted object vocabulary:
    - K single-token token ids
    - Fast lookups for obj_ids, id -> word, id -> category, word -> category
    - Logs dropped multi-token/unknown words to results/dropped_object_words.txt
    """

    def __init__(self, tokenizer, synonyms_file: Optional[str] = None, dropped_words_path: Optional[str] = None):
        self.tokenizer = tokenizer
        if synonyms_file and os.path.exists(synonyms_file):
            raw_synonyms = load_synonyms_from_file(synonyms_file)
        else:
            raw_synonyms = DEFAULT_COCO_SYNONYMS.copy()

        self.word_to_category: Dict[str, str] = {}
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.id_to_category: Dict[int, str] = {}
        self.obj_ids: List[int] = []
        self.obj_ids_set: Set[int] = set()
        self.dropped_words: List[str] = []

        self._build_vocab(raw_synonyms)

        if dropped_words_path:
            os.makedirs(os.path.dirname(os.path.abspath(dropped_words_path)), exist_ok=True)
            with open(dropped_words_path, "w", encoding="utf-8") as f:
                for dw in sorted(self.dropped_words):
                    f.write(f"{dw}\n")
            logger.info(f"Saved {len(self.dropped_words)} dropped object words to {dropped_words_path}")

    def _build_vocab(self, synonyms: Dict[str, str]):
        """
        Filters words that map to a single existing token with word-initial space.
        In SentencePiece / LLaMA tokenizer: ' word' (represented as '\u2581' + word).
        """
        spiece_prefix = "\u2581"
        unk_id = getattr(self.tokenizer, "unk_token_id", None)

        for word, category in sorted(synonyms.items()):
            w_clean = word.strip().lower()
            if not w_clean:
                continue

            # We test tokenization with leading space / sentencepiece underscore
            # 1. Direct convert_tokens_to_ids(" " + word)
            cand_token = spiece_prefix + w_clean
            tok_id = self.tokenizer.convert_tokens_to_ids(cand_token)

            # If that returns unk or not found, try tokenizer.encode(" " + w_clean)
            if tok_id is None or tok_id == unk_id:
                encoded = self.tokenizer.encode(" " + w_clean, add_special_tokens=False)
                if len(encoded) == 1 and encoded[0] != unk_id:
                    tok_id = encoded[0]
                else:
                    self.dropped_words.append(f"{w_clean} (encoded_len={len(encoded)})")
                    continue

            # Verify it maps back to single token
            if tok_id == unk_id:
                self.dropped_words.append(f"{w_clean} (unk)")
                continue

            # If token already exists under another word (e.g. synonym), keep mapping
            self.word_to_category[w_clean] = category
            self.word_to_id[w_clean] = tok_id

            if tok_id not in self.id_to_word:
                self.id_to_word[tok_id] = w_clean
                self.id_to_category[tok_id] = category
                self.obj_ids.append(tok_id)

        self.obj_ids.sort()
        self.obj_ids_set = set(self.obj_ids)
        logger.info(f"Built object vocabulary: {len(self.obj_ids)} unique single-token object IDs (K={len(self.obj_ids)}).")
        logger.info(f"Dropped {len(self.dropped_words)} multi-token / unk words.")

    def is_object_token(self, token_id: int) -> bool:
        return token_id in self.obj_ids_set

    def get_word(self, token_id: int) -> str:
        return self.id_to_word.get(token_id, "")

    def get_category(self, token_id: int) -> str:
        return self.id_to_category.get(token_id, "")
