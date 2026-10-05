"""
Gate A: Unit Tests for Experiment H1 (tests/test_h1_units.py)

Offline tests runnable with `python3 -m unittest tests/test_h1_units.py` without GPU.
Covers:
  - U1: Candidate filtering & alternating rank A/B splitting
  - U2: Metric computation on synthetic effect matrices (noise, single dominant position, uniform)
  - U3: Permutation null and bootstrap tests on synthetic distributions
  - U4: Same-category 1:1 matching logic verification
  - U5: Text processing and phonetic a/an agreement heuristics
"""

import math
import random
import unittest
from typing import Any, Dict, List

import importlib

make_pairs_mod = importlib.import_module("src.07_make_pairs_h1")
match_object_pairs_samecat = make_pairs_mod.match_object_pairs_samecat
from src.h1_common import (
    check_a_an_agreement,
    clean_piece,
    cohens_dz,
    compute_object_metrics,
    filter_and_split_candidates,
    holm_bonferroni,
    is_complete_word_piece,
    paired_bootstrap_delta,
    permutation_null_logHPR,
)


class MockTokenizer:
    """Mock tokenizer providing convert_ids_to_tokens and basic lookup for U1."""
    def __init__(self, id_to_piece: Dict[int, str]):
        self.id_to_piece = id_to_piece
        self.piece_to_id = {v: k for k, v in id_to_piece.items()}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.id_to_piece.get(token_id, "<unk>")


class TestExperimentH1Units(unittest.TestCase):
    # =======================================================================
    # U1: Candidate Filtering & Alternating A/B Splitting
    # =======================================================================
    def test_u1_candidate_filtering_and_alternating_split(self):
        r"""
        U1: Validates that candidate screening:
          1. Excludes original token.
          2. Requires SentencePiece word piece regex (^[\s\u2581][A-Za-z]+$).
          3. Enforces phonetic a/an agreement.
          4. Assigns rank 1, 3, 5 -> A and rank 2, 4, 6 -> B.
          5. Rejects candidates when fewer than 4 valid tokens are available.
        """
        # Vocabulary: piece 0: ' the', 1: ' an', 2: ' a', 3: ' some', 4: ' this', 5: ' that', 6: ' every', 7: '123'
        vocab = {
            0: " the",
            1: " an",
            2: " a",
            3: " some",
            4: " this",
            5: " that",
            6: " every",
            7: " 123",      # non-alpha
            8: "badpiece",  # missing leading space marker
        }
        tokenizer = MockTokenizer(vocab)

        # Top 50 token IDs and probs
        cand_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        cand_probs = [0.40, 0.20, 0.15, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01]

        # Case 1: Next piece is consonant-initial (' dog')
        half_A, half_B = filter_and_split_candidates(
            top50_token_ids=cand_ids,
            top50_token_probs=cand_probs,
            tokenizer=tokenizer,
            y_j_id=0,
            sentence_text="There is the dog.",
            target_char_idx=9,
            next_token_piece=" dog",
            nlp=None,
            pos_mode="lexicon",
        )

        self.assertGreaterEqual(len(half_A), 2)
        self.assertGreaterEqual(len(half_B), 2)

        # Check alternating assignment: half A gets alternating indices 0, 2, 4; half B gets 1, 3
        for cand in half_A:
            self.assertEqual(cand["half"], "A")

        for cand in half_B:
            self.assertEqual(cand["half"], "B")

        # Original token id 0 (' the') must NOT be present
        self.assertNotIn(0, [c["cand_id"] for c in half_A + half_B])
        # ' an' (id 1) must NOT be present before ' dog'
        self.assertNotIn(1, [c["cand_id"] for c in half_A + half_B])
        # Non-alphabetic and ill-formed tokens must NOT be present
        self.assertNotIn(7, [c["cand_id"] for c in half_A + half_B])
        self.assertNotIn(8, [c["cand_id"] for c in half_A + half_B])

        # Case 2: Insufficient valid candidates (< 4) should return empty lists
        short_ids = [0, 7, 8]
        short_probs = [0.8, 0.1, 0.1]
        hA_empty, hB_empty = filter_and_split_candidates(
            top50_token_ids=short_ids,
            top50_token_probs=short_probs,
            tokenizer=tokenizer,
            y_j_id=0,
            sentence_text="There is the dog.",
            target_char_idx=9,
            next_token_piece=" dog",
            nlp=None,
            pos_mode="lexicon",
        )
        self.assertEqual(len(hA_empty), 0)
        self.assertEqual(len(hB_empty), 0)

    # =======================================================================
    # U2: Math Sanity on Synthetic Effect Matrices
    # =======================================================================
    def test_u2_metric_computation_sanity(self):
        """
        U2: Validates mathematical behavior of Mbar, HPR, logHPR, rho, rho_z, top1_agree:
          (a) Pure noise -> logHPR close to 0, rho close to 0
          (b) Dominant single position -> HPR > 2.0, top1_agree == 1
          (c) Uniform effect across all positions -> HPR close to 1.0
        """
        valid_window = set(range(2, 11))  # 9 positions: j in [2, 10]

        # (a) Uniform signal: e_j = 1.0 for all j in both halves
        uniform_effects = {
            j: {"A": [1.0, 1.0], "B": [1.0, 1.0]} for j in valid_window
        }
        res_uniform = compute_object_metrics(uniform_effects, valid_window)
        self.assertIsNotNone(res_uniform)
        self.assertAlmostEqual(res_uniform["Mbar"], 1.0, places=4)
        self.assertAlmostEqual(res_uniform["HPR"], 1.0, places=4)
        self.assertAlmostEqual(res_uniform["logHPR"], 0.0, places=4)
        self.assertAlmostEqual(res_uniform["rho"], 0.0, places=4)

        # (b) Single dominant position: j = 5 has huge effect (10.0), other positions have 1.0
        dominant_effects = {
            j: {"A": [1.0, 1.0], "B": [1.0, 1.0]} for j in valid_window
        }
        dominant_effects[5] = {"A": [10.0, 10.0], "B": [10.0, 10.0]}
        res_dom = compute_object_metrics(dominant_effects, valid_window)
        self.assertIsNotNone(res_dom)
        # Expected HPR: mean of 8 ones and 1 ten = 18 / 9 = 2.0. Peak at 5 is 10.0.
        # Ratio = 10.0 / 2.0 = 5.0
        self.assertGreater(res_dom["HPR"], 2.0)
        self.assertAlmostEqual(res_dom["HPR"], 5.0, places=3)
        self.assertEqual(res_dom["top1_agree"], 1)
        self.assertEqual(res_dom["peak_j"], 5)
        self.assertGreater(res_dom["rho"], 0.8)

        # (c) Insufficient positions (< 5) returns None
        small_window = {2, 3, 4}
        small_effects = {j: {"A": [1.0, 1.0], "B": [1.0, 1.0]} for j in small_window}
        res_small = compute_object_metrics(small_effects, small_window)
        self.assertIsNone(res_small)

    # =======================================================================
    # U3: Permutation Null and Paired Bootstrap Tests
    # =======================================================================
    def test_u3_statistical_tests_on_synthetic_data(self):
        """
        U3: Validates permutation null and paired bootstrap CI on known distributions:
          - Bootstrap CI contains known true population mean.
          - Cohen's d_z equals expected value.
          - Holm-Bonferroni maintains correct step-down adjusted ordering.
        """
        rng = random.Random(42)

        # 1. Bootstrap CI of known mean
        true_mean = 1.5
        synthetic_deltas = [true_mean + rng.gauss(0.0, 0.5) for _ in range(500)]
        mean_d, ci_lo, ci_hi = paired_bootstrap_delta(synthetic_deltas, n_boot=2000, seed=42)

        self.assertLess(ci_lo, true_mean)
        self.assertGreater(ci_hi, true_mean)
        self.assertAlmostEqual(mean_d, true_mean, delta=0.1)

        # 2. Cohen's d_z
        dz = cohens_dz(synthetic_deltas)
        # mean ~ 1.5, sd ~ 0.5 -> dz ~ 3.0
        self.assertGreater(dz, 2.5)

        # 3. Holm-Bonferroni correction
        p_raw = [0.005, 0.02, 0.04]
        p_adj = holm_bonferroni(p_raw)
        # m = 3:
        # rank 1: min(1, 0.005 * 3) = 0.015
        # rank 2: min(1, 0.02 * 2) = 0.04
        # rank 3: min(1, 0.04 * 1) = 0.04
        self.assertAlmostEqual(p_adj[0], 0.015, places=5)
        self.assertAlmostEqual(p_adj[1], 0.04, places=5)
        self.assertAlmostEqual(p_adj[2], 0.04, places=5)
        self.assertTrue(p_adj[0] <= p_adj[1] <= p_adj[2])

        # 4. Permutation null on independent noise profiles
        # When abs_A and abs_B are uncorrelated noise, permutation p-value is distributed uniformly
        abs_A_list = [[1.0 + rng.random() for _ in range(9)] for _ in range(50)]
        abs_B_list = [[1.0 + rng.random() for _ in range(9)] for _ in range(50)]
        act_mean, null_mean, p_perm = permutation_null_logHPR(abs_A_list, abs_B_list, n_perm=500, seed=42)
        self.assertGreater(p_perm, 0.01)  # Noise should not produce extreme permutation p-value

    # =======================================================================
    # U4: Same-Category 1:1 Matching Logic Verification
    # =======================================================================
    def test_u4_same_category_matching_logic(self):
        """
        U4: Validates same-category 1:1 matching:
          - category constraint: canon(r) == canon(h)
          - image constraint: image_id(r) != image_id(h)
          - position caliper: |rel_pos(r) - rel_pos(h)| <= caliper
          - greedy pairing without replacement (no real object reused)
        """
        h_pool = [
            {"image_id": "img_01", "canon": "chair", "rel_pos": 0.50, "t": 10, "G": 20},
            {"image_id": "img_02", "canon": "dog", "rel_pos": 0.30, "t": 6, "G": 20},
            {"image_id": "img_03", "canon": "chair", "rel_pos": 0.52, "t": 11, "G": 20},
        ]

        r_pool = [
            {"image_id": "img_04", "canon": "chair", "rel_pos": 0.53, "t": 11, "G": 20},  # Exactly ONE valid chair in r_pool
            {"image_id": "img_05", "canon": "dog", "rel_pos": 0.31, "t": 6, "G": 20},    # Valid for dog!
            {"image_id": "img_06", "canon": "cat", "rel_pos": 0.30, "t": 6, "G": 20},    # Category mismatch -> invalid!
            {"image_id": "img_07", "canon": "chair", "rel_pos": 0.85, "t": 17, "G": 20},  # Caliper exceeded (|0.85 - 0.52| = 0.33 > 0.10)
        ]

        matched_h, matched_r, dropped = match_object_pairs_samecat(h_pool, r_pool, caliper=0.10, seed=0)

        # Expect 2 matched pairs:
        # 1 chair pair: h matched with r from img_04
        # 1 dog pair: h from img_02 matched with r from img_05
        # One chair remains dropped due to no remaining chair candidate within caliper
        self.assertEqual(len(matched_h), 2)
        self.assertEqual(len(matched_r), 2)
        self.assertEqual(dropped, 1)

        # Check pair invariants
        used_r_images = set()
        for h, r in zip(matched_h, matched_r):
            self.assertEqual(h["canon"], r["canon"])
            self.assertNotEqual(h["image_id"], r["image_id"])
            self.assertLessEqual(abs(h["rel_pos"] - r["rel_pos"]), 0.10)
            self.assertNotIn(r["image_id"], used_r_images)
            used_r_images.add(r["image_id"])

    # =======================================================================
    # U5: Text Processing & Phonetic a/an Agreement Heuristics
    # =======================================================================
    def test_u5_text_processing_and_a_an_agreement(self):
        """
        U5: Validates clean_piece, is_complete_word_piece, and check_a_an_agreement.
        """
        # 1. clean_piece
        self.assertEqual(clean_piece(" chair"), "chair")
        self.assertEqual(clean_piece("\u2581chair"), "chair")
        self.assertEqual(clean_piece("chair"), "chair")

        # 2. is_complete_word_piece
        # Valid word pieces
        self.assertTrue(is_complete_word_piece(" chair", next_piece=" on"))
        self.assertTrue(is_complete_word_piece("\u2581dog", next_piece="."))
        self.assertTrue(is_complete_word_piece(" a", next_piece=" table"))

        # Invalid: missing leading space
        self.assertFalse(is_complete_word_piece("chair"))
        # Invalid: contains numbers / punctuation
        self.assertFalse(is_complete_word_piece(" 123"))
        self.assertFalse(is_complete_word_piece(" dog!"))
        # Invalid: word split across subwords (next piece does NOT start with space/punctuation)
        self.assertFalse(is_complete_word_piece(" sub", next_piece="word"))

        # 3. check_a_an_agreement
        # 'a' before consonants -> True
        self.assertTrue(check_a_an_agreement("a", " dog"))
        self.assertTrue(check_a_an_agreement("a", " table"))
        # 'a' before vowels -> False
        self.assertFalse(check_a_an_agreement("a", " apple"))
        self.assertFalse(check_a_an_agreement("a", " elephant"))
        self.assertFalse(check_a_an_agreement("a", " orange"))

        # 'an' before vowels -> True
        self.assertTrue(check_a_an_agreement("an", " apple"))
        self.assertTrue(check_a_an_agreement("an", " orange"))
        # 'an' before consonants -> False
        self.assertFalse(check_a_an_agreement("an", " dog"))
        self.assertFalse(check_a_an_agreement("an", " car"))

        # Non-article words -> always True
        self.assertTrue(check_a_an_agreement("the", " dog"))
        self.assertTrue(check_a_an_agreement("some", " apple"))

    # =======================================================================
    # U6: KV-Cache Cropping & Batch Repeating (DynamicCache + Legacy Tuple)
    # =======================================================================
    def test_u6_kv_cache_crop_and_repeat(self):
        """
        U6: Validates that crop_or_repeat_kv_cache:
          1. Correctly crops sequence dimension to length P.
          2. Correctly repeats batch dimension from 1 to B (e.g. B=16).
          3. Handles DynamicCache with .layers, with .key_cache, and legacy tuple.
        """
        run_h1_mod = importlib.import_module("src.08_run_h1")
        crop_or_repeat_kv_cache = run_h1_mod.crop_or_repeat_kv_cache

        class MockTensor:
            def __init__(self, shape):
                self.shape = shape

            def clone(self):
                return MockTensor(self.shape)

            def repeat_interleave(self, repeats, dim=0):
                new_shape = list(self.shape)
                new_shape[dim] *= repeats
                return MockTensor(tuple(new_shape))

            def __getitem__(self, item):
                new_shape = list(self.shape)
                if isinstance(item, tuple):
                    for arg in item:
                        if isinstance(arg, slice) and arg.stop is not None:
                            new_shape[-2] = min(new_shape[-2], arg.stop)
                return MockTensor(tuple(new_shape))

        # Case 1: Legacy Tuple of (key, value)
        k = MockTensor((1, 32, 600, 128))
        v = MockTensor((1, 32, 600, 128))
        legacy_cache = ((k, v), (k, v))

        out_legacy = crop_or_repeat_kv_cache(legacy_cache, P=588, batch_size=16)
        self.assertEqual(len(out_legacy), 2)
        self.assertEqual(out_legacy[0][0].shape, (16, 32, 588, 128))
        self.assertEqual(out_legacy[0][1].shape, (16, 32, 588, 128))

        # Case 2: Modern DynamicCache with .layers
        class MockLayer:
            def __init__(self, k, v):
                self.keys = k
                self.values = v

        class MockDynamicCacheWithLayers:
            def __init__(self, layers):
                self.layers = layers

            def get_seq_length(self):
                return 600

            def crop(self, num):
                pass

            def batch_repeat_interleave(self, repeats):
                for layer in self.layers:
                    layer.keys = layer.keys.repeat_interleave(repeats, dim=0)
                    layer.values = layer.values.repeat_interleave(repeats, dim=0)

        mock_layers = [MockLayer(MockTensor((1, 32, 600, 128)), MockTensor((1, 32, 600, 128))) for _ in range(2)]
        dyn_cache = MockDynamicCacheWithLayers(mock_layers)

        out_dyn = crop_or_repeat_kv_cache(dyn_cache, P=588, batch_size=16)
        self.assertEqual(out_dyn.layers[0].keys.shape, (16, 32, 588, 128))
        self.assertEqual(out_dyn.layers[0].values.shape, (16, 32, 588, 128))


if __name__ == "__main__":
    unittest.main()
