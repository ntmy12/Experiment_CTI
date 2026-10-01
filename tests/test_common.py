"""
Unit tests (T1, T2, T3) runnable without GPU.
These satisfy Gate G0 requirements in EXPERIMENT1_SPEC.md.
"""

import math
import random
import unittest
try:
    import numpy as np
except ImportError:
    np = None

from src.common import (
    pieces_to_text,
    find_mentions,
    char_to_token,
    check_word_start,
    match_object_pairs,
    paired_bootstrap_delta,
    holm_bonferroni,
    bootstrap_auroc,
    cohens_dz,
)


class TestExperiment1Core(unittest.TestCase):
    def test_t1_token_alignment_and_mention_extraction(self):
        """
        T1: pieces_to_text + find_mentions + char_to_token on standard benchmark string:
        [' The',' image',' shows',' a',' man',' and',' two',' dog','s',' near',' a',' teddy',' bear','<0x0A>',' Mugs','.']
        Expected:
        - man -> tok 4
        - dogs -> tok 7 (' dog')
        - teddy bear -> tok 11 (' teddy')
        - mugs -> cup, tok 14 (' Mugs')
        - word-start condition valid for all 4 mentions
        """
        pieces = [
            " The", " image", " shows", " a", " man", " and", " two",
            " dog", "s", " near", " a", " teddy", " bear", "<0x0A>", " Mugs", "."
        ]

        syn2canon = {
            "man": "person",
            "dog": "dog",
            "teddy bear": "teddy bear",
            "teddy": "teddy bear",
            "bear": "bear",
            "cup": "cup",
            "mug": "cup",
        }

        text, spans = pieces_to_text(pieces)

        # Verify newline was correctly translated
        self.assertIn("\n", text)

        mentions = find_mentions(text, syn2canon)
        self.assertEqual(len(mentions), 4, f"Expected 4 mentions, found {len(mentions)}: {mentions}")

        # Map each mention to token and check word start
        aligned = []
        for m in mentions:
            tok_idx = char_to_token(spans, m["char_start"])
            is_valid = check_word_start(spans, tok_idx, m["char_start"])
            aligned.append({
                "word": m["word"].lower(),
                "canon": m["canon"],
                "tok_idx": tok_idx,
                "piece": pieces[tok_idx],
                "valid": is_valid
            })

        # 1. Check man -> tok 4
        self.assertEqual(aligned[0]["canon"], "person")
        self.assertEqual(aligned[0]["tok_idx"], 4)
        self.assertTrue(aligned[0]["valid"])

        # 2. Check dogs -> tok 7 (piece ' dog')
        self.assertEqual(aligned[1]["canon"], "dog")
        self.assertEqual(aligned[1]["tok_idx"], 7)
        self.assertTrue(aligned[1]["valid"])

        # 3. Check teddy bear -> tok 11 (piece ' teddy')
        self.assertEqual(aligned[2]["canon"], "teddy bear")
        self.assertEqual(aligned[2]["tok_idx"], 11)
        self.assertTrue(aligned[2]["valid"])

        # 4. Check mugs -> tok 14 (piece ' Mugs', canon 'cup')
        self.assertEqual(aligned[3]["canon"], "cup")
        self.assertEqual(aligned[3]["tok_idx"], 14)
        self.assertTrue(aligned[3]["valid"])

    def test_t2_object_matching_and_caliper(self):
        """
        T2: 1:1 control matching on synthetic data.
        Verifies:
        - |rel_pos(r) - rel_pos(h)| <= caliper
        - No reuse of real objects (1:1 matching without replacement)
        - Deterministic results across runs with same seed
        """
        # Create synthetic pool
        halluc_pool = [
            {"image_id": 1, "canon": "dog", "t": 15, "rel_pos": 0.20},
            {"image_id": 2, "canon": "cat", "t": 25, "rel_pos": 0.50},
            {"image_id": 3, "canon": "car", "t": 35, "rel_pos": 0.85},
            {"image_id": 4, "canon": "bus", "t": 12, "rel_pos": 0.15},
        ]

        real_pool = [
            {"image_id": 10, "canon": "chair", "t": 14, "rel_pos": 0.18},
            {"image_id": 11, "canon": "table", "t": 24, "rel_pos": 0.52},
            {"image_id": 12, "canon": "couch", "t": 26, "rel_pos": 0.55},
            {"image_id": 13, "canon": "bed", "t": 40, "rel_pos": 0.99}, # beyond 0.85 by 0.14 (> 0.1)
            {"image_id": 14, "canon": "bench", "t": 15, "rel_pos": 0.16}, # partner for bus/dog
        ]

        matched_h, matched_r, dropped_cnt = match_object_pairs(halluc_pool, real_pool, caliper=0.1, seed=0)

        # Check pair counts
        self.assertEqual(len(matched_h), len(matched_r))
        self.assertEqual(len(matched_h), 3, "Expected 3 matches, 1 dropped due to caliper")
        self.assertEqual(dropped_cnt, 1)

        # Check caliper and 1:1 matching
        used_r_ids = set()
        for h, r in zip(matched_h, matched_r):
            self.assertEqual(h["pair_id"], r["pair_id"])
            diff = abs(h["rel_pos"] - r["rel_pos"])
            self.assertLessEqual(diff, 0.1 + 1e-6)
            self.assertNotIn(r["image_id"], used_r_ids)
            used_r_ids.add(r["image_id"])

        # Check determinism with same seed
        matched_h2, matched_r2, _ = match_object_pairs(halluc_pool, real_pool, caliper=0.1, seed=0)
        self.assertEqual([h["pair_id"] for h in matched_h], [h["pair_id"] for h in matched_h2])
        self.assertEqual([h["canon"] for h in matched_h], [h["canon"] for h in matched_h2])

    def test_holm_bonferroni(self):
        """Verify Holm-Bonferroni step-down correction logic."""
        raw_p = [0.001, 0.01, 0.03, 0.05, 0.20]
        adj_p = holm_bonferroni(raw_p)
        self.assertEqual(len(adj_p), len(raw_p))
        for p_adj, p_raw in zip(adj_p, raw_p):
            self.assertGreaterEqual(p_adj, p_raw)
            self.assertLessEqual(p_adj, 1.0)
        # Check monotonicity on sorted
        self.assertTrue(all(adj_p[i] <= adj_p[i+1] for i in range(len(adj_p)-1)))

    def test_t3_statistical_functions(self):
        """
        T3: Statistical testing with known synthetic data.
        Verifies:
        - Bootstrap CI contains known true mean delta
        - AUROC equals 1.0 for perfectly separable data
        """
        if np is None:
            self.skipTest("numpy/scipy not installed on host machine. Will run on Kaggle/environment with requirements.txt.")
        np.random.seed(42)
        true_mean = 2.5
        samples = np.random.normal(loc=true_mean, scale=1.0, size=500)
        mean_delta, ci_lo, ci_hi = paired_bootstrap_delta(samples, n_boot=1000, seed=42)

        self.assertLess(ci_lo, true_mean)
        self.assertGreater(ci_hi, true_mean)
        self.assertAlmostEqual(mean_delta, np.mean(samples), places=5)

        # 2. Holm-Bonferroni correction
        raw_p = [0.001, 0.01, 0.03, 0.05, 0.20]
        adj_p = holm_bonferroni(raw_p)
        self.assertEqual(len(adj_p), len(raw_p))
        for p_adj, p_raw in zip(adj_p, raw_p):
            self.assertGreaterEqual(p_adj, p_raw)
            self.assertLessEqual(p_adj, 1.0)
        # Check monotonicity on sorted
        self.assertTrue(all(adj_p[i] <= adj_p[i+1] for i in range(len(adj_p)-1)))

        # 3. AUROC for perfectly separable data
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        auc, ci_lo_auc, ci_hi_auc = bootstrap_auroc(y_true, y_scores, n_boot=500, seed=42)
        self.assertAlmostEqual(auc, 1.0, places=5)
        self.assertAlmostEqual(ci_lo_auc, 1.0, places=5)
        self.assertAlmostEqual(ci_hi_auc, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
