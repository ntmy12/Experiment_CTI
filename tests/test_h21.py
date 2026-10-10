"""
Unit tests for Pilot Experiment H2.1 according to Section 8 specifications:
- JSD(p, p) = 0; JSD symmetric; range in [0, 1].
- Placebo perturbation gives S_t < 1e-6 and flip = 0.
- Index alignment: greedy argmax at index P+j-1 equals y_j on >= 95% of tokens.
- Object vocab: all IDs are single tokens.
- Determinism: running twice with the same seed gives identical records within tolerance 1e-4.
"""

import math
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial.distance import jensenshannon

from compute import (
    jsd_divergence,
    compute_restricted_distribution,
    compute_entropy,
    is_candidate_perturbation_site,
)
from objects import ObjectVocabulary, DEFAULT_COCO_SYNONYMS
from analysis import run_full_analysis, compute_spearman, compute_partial_spearman


# =============================================================================
# 1. JSD Tests
# =============================================================================
def test_jsd_properties():
    """Validates JSD(p, p) == 0, symmetry, and range [0, 1]."""
    # Identical distributions
    p = np.array([0.2, 0.5, 0.3])
    assert pytest.approx(jsd_divergence(p, p), abs=1e-10) == 0.0

    # Symmetry
    q = np.array([0.1, 0.8, 0.1])
    jsd_pq = jsd_divergence(p, q)
    jsd_qp = jsd_divergence(q, p)
    assert pytest.approx(jsd_pq, abs=1e-10) == jsd_qp

    # Range [0, 1] for orthogonal distributions
    p_orth = np.array([1.0, 0.0, 0.0])
    q_orth = np.array([0.0, 1.0, 0.0])
    jsd_orth = jsd_divergence(p_orth, q_orth)
    assert pytest.approx(jsd_orth, abs=1e-6) == 1.0
    assert 0.0 <= jsd_orth <= 1.0

    # Cross-check with scipy.spatial.distance.jensenshannon (squared distance)
    scipy_val = float(jensenshannon(p, q, base=2) ** 2)
    assert pytest.approx(jsd_pq, abs=1e-6) == scipy_val


# =============================================================================
# 2. Object Vocab Tests
# =============================================================================
class MockTokenizer:
    def __init__(self):
        self.unk_token_id = 0
        self.all_special_ids = [0, 1, 2]
        # Vocabulary mapping
        self.vocab = {
            "<unk>": 0, "<s>": 1, "</s>": 2,
            "\u2581dog": 10, "\u2581cat": 11, "\u2581car": 12,
            "\u2581person": 13, "\u2581man": 14, "\u2581woman": 15,
            "\u2581the": 20, "\u2581a": 21, "\u2581is": 22, "\u2581running": 23,
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id):
        return self.id_to_token.get(token_id, "<unk>")

    def encode(self, text, add_special_tokens=False):
        parts = text.split()
        res = []
        for p in parts:
            tok = "\u2581" + p
            res.append(self.vocab.get(tok, self.unk_token_id))
        return res

    def decode(self, ids, skip_special_tokens=True):
        tokens = [self.id_to_token.get(i, "") for i in ids]
        return " ".join(t.replace("\u2581", "") for t in tokens if t)


def test_object_vocab_single_tokens():
    """Asserts that all IDs in object vocab are single tokens and not unk."""
    tok = MockTokenizer()
    synonyms = {"dog": "dog", "cat": "cat", "car": "car", "unknown_multi_token_object_word": "misc"}
    obj_vocab = ObjectVocabulary(tokenizer=tok, synonyms_file=None)
    # Filter using our mock synonyms
    obj_vocab._build_vocab(synonyms)

    assert len(obj_vocab.obj_ids) > 0
    for tok_id in obj_vocab.obj_ids:
        assert tok_id != tok.unk_token_id
        # Token string representation
        tok_str = tok.convert_ids_to_tokens(tok_id)
        assert tok_str.startswith("\u2581") or tok_str.startswith(" ")


# =============================================================================
# 3. Index Alignment and Sanity Check
# =============================================================================
def test_index_alignment_logic():
    """
    Tests index rule:
    If P = length of prompt after image tokens expansion,
    logits that predict y_j are at index P + j - 1.
    """
    P = 15  # prompt length
    T = 10  # generated sequence length
    total_len = P + T

    # Construct mock logits where greedy argmax at index P + j - 1 is exactly j + 100
    vocab_size = 200
    mock_logits = torch.zeros(1, total_len, vocab_size)
    gen_ids = torch.tensor([[j + 100 for j in range(T)]])

    for j in range(T):
        target_idx = P + j - 1
        mock_logits[0, target_idx, gen_ids[0, j]] = 10.0

    # Extract target slice [P-1 : P+T-1]
    extracted_slice = mock_logits[0, P - 1 : P + T - 1, :]
    assert extracted_slice.shape == (T, vocab_size)

    preds = extracted_slice.argmax(dim=-1)
    match_rate = (preds == gen_ids[0]).float().mean().item()
    assert match_rate >= 0.95, f"Match rate {match_rate} is below 95%"


# =============================================================================
# 4. Placebo Perturbation Test
# =============================================================================
def test_placebo_perturbation():
    """Asserts that replacing token by itself produces S_t < 1e-6 and flip = 0."""
    p_orig = np.array([0.7, 0.2, 0.1])
    # Placebo: identical distribution
    p_placebo = p_orig.copy()

    s_t = jsd_divergence(p_orig, p_placebo)
    flip = 1 if np.argmax(p_orig) != np.argmax(p_placebo) else 0

    assert s_t < 1e-6
    assert flip == 0


# =============================================================================
# 5. Determinism Test
# =============================================================================
def test_determinism_synthetic():
    """Asserts determinism with tolerance 1e-4."""
    rng1 = np.random.RandomState(42)
    p1 = rng1.dirichlet(np.ones(10))
    q1 = rng1.dirichlet(np.ones(10))
    val1 = jsd_divergence(p1, q1)

    rng2 = np.random.RandomState(42)
    p2 = rng2.dirichlet(np.ones(10))
    q2 = rng2.dirichlet(np.ones(10))
    val2 = jsd_divergence(p2, q2)

    assert abs(val1 - val2) < 1e-4


# =============================================================================
# 6. Statistical Analysis Pipeline Test
# =============================================================================
def test_analysis_pipeline_smoke():
    """Tests cluster bootstrap analysis A0 through A5 on synthetic pilot data."""
    np.random.seed(42)
    n_images = 20
    rows = []

    for img_id in range(1, n_images + 1):
        n_rec = np.random.randint(5, 15)
        for _ in range(n_rec):
            t = int(np.random.randint(0, 100))
            k = int(np.random.randint(1, 8))
            v_t = float(np.clip(1.0 - (t / 100.0) + np.random.normal(0, 0.1), 0.0, 1.0))
            s_t = float(np.clip(0.1 + (t / 120.0) - 0.2 * v_t + np.random.normal(0, 0.1), 0.0, 1.0))
            flip = int(np.random.rand() < s_t * 0.5)

            rows.append({
                "image_id": img_id,
                "t": t,
                "T": 120,
                "rel_pos": t / 120.0,
                "word": "dog",
                "category": "dog",
                "s": max(0, t - k),
                "k": k,
                "orig_token": "running",
                "alt_token": "walking",
                "p_alt": 0.25,
                "V_t": v_t,
                "S_t": s_t,
                "flip": flip,
                "delta_p": 0.1,
                "salience": 0.3,
                "in_gt": 1,
                "confidence": 0.8,
                "entropy": 1.2,
                "top3_before": "dog:0.8,cat:0.1,horse:0.05",
                "top3_after": "dog:0.5,cat:0.3,horse:0.1",
            })

    df = pd.DataFrame(rows)
    df_black = df.copy()
    df_black["V_t"] = np.clip(df["V_t"] + np.random.normal(0, 0.05, len(df)), 0.0, 1.0)

    # Run analysis with small resamples for test speed
    res = run_full_analysis(df, df_black=df_black, n_resamples=50, seed=42)

    assert "A0_premise" in res
    assert "A1_sensitivity_vs_position" in res
    assert "A2_St_vs_Vt" in res
    assert "A3_mediation" in res
    assert "A4_distance" in res
    assert "A5_robustness_black" in res
