"""
POS tagging module with character-offset token alignment.
Uses spaCy (en_core_web_sm) with fallbacks to NLTK and UNK tagging.
Supports Milestone M0 alignment verification.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger("confirmatory_v2")

_SPACY_NLP = None
_NLTK_AVAILABLE = False


def get_spacy_nlp():
    """Initializes spaCy en_core_web_sm model with fallback handling."""
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP

    try:
        import spacy
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
            logger.info("Successfully loaded spaCy 'en_core_web_sm'.")
            return _SPACY_NLP
        except Exception:
            logger.info("Downloading spaCy 'en_core_web_sm' model...")
            try:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                _SPACY_NLP = spacy.load("en_core_web_sm")
                return _SPACY_NLP
            except Exception as e2:
                logger.warning(f"Could not download spaCy model: {e2}")
    except ImportError:
        logger.warning("spaCy is not installed.")

    return None


class POSTagger:
    """
    Manages part-of-speech tagging and token-level character-span alignment.
    """

    def __init__(self):
        self.nlp = get_spacy_nlp()
        self.backend = "spacy" if self.nlp is not None else "fallback"
        if self.backend == "fallback":
            try:
                import nltk
                nltk.download("averaged_perceptron_tagger", quiet=True)
                nltk.download("punkt", quiet=True)
                self.backend = "nltk"
                logger.info("Using NLTK POS tagger fallback.")
            except Exception:
                self.backend = "unk"
                logger.warning("Neither spaCy nor NLTK available. Using POS='UNK' fallback.")

    def tag_caption_tokens(
        self,
        caption_text: str,
        token_ids: List[int],
        tokenizer: Any,
    ) -> Dict[int, str]:
        """
        Aligns generated tokens to words and assigns a coarse POS tag for every token.
        Returns: Dict[int, str] mapping token index j -> coarse POS tag.
        """
        token_pos_map: Dict[int, str] = {}
        if not token_ids:
            return token_pos_map

        if self.backend == "spacy" and self.nlp is not None:
            doc = self.nlp(caption_text)
            char_to_pos: Dict[int, str] = {}
            for token in doc:
                for c in range(token.idx, token.idx + len(token.text)):
                    char_to_pos[c] = token.pos_

            # Reconstruct token character spans
            current_char = 0
            for j, tid in enumerate(token_ids):
                tok_str = tokenizer.convert_ids_to_tokens(tid) or ""
                # Strip sentencepiece space
                clean_str = tok_str.replace("\u2581", " ").replace(" ", " ")
                clean_strip = clean_str.strip()

                if not clean_strip:
                    token_pos_map[j] = "PUNCT"
                    continue

                # Locate occurrence in caption_text starting from current_char
                pos_in_text = caption_text.find(clean_strip, current_char)
                if pos_in_text != -1:
                    matched_pos = char_to_pos.get(pos_in_text, "UNK")
                    token_pos_map[j] = matched_pos
                    current_char = pos_in_text + len(clean_strip)
                else:
                    # Fallback lookup around current_char
                    matched_pos = char_to_pos.get(current_char, "UNK")
                    token_pos_map[j] = matched_pos

        elif self.backend == "nltk":
            import nltk
            tokens = nltk.word_tokenize(caption_text)
            tagged = nltk.pos_tag(tokens)
            tag_dict = {w.lower(): pos for w, pos in tagged}
            for j, tid in enumerate(token_ids):
                word = tokenizer.decode([tid]).strip().lower()
                token_pos_map[j] = tag_dict.get(word, "UNK")
        else:
            for j in range(len(token_ids)):
                token_pos_map[j] = "UNK"

        return token_pos_map

    def unit_test_alignment(self, sample_captions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Runs unit test on 2 captions and returns alignment data for Milestone M0.
        """
        if sample_captions is None:
            sample_captions = [
                "A young woman is playing tennis on an outdoor court with two yellow balls.",
                "There is a wooden dining table in the kitchen with two chairs and an apple."
            ]

        results = []
        for i, text in enumerate(sample_captions):
            if self.nlp is not None:
                doc = self.nlp(text)
                words = [(t.text, t.pos_) for t in doc]
            else:
                words = [(w, "UNK") for w in text.split()]
            results.append({"caption_id": i + 1, "text": text, "aligned_words": words})

        return results
