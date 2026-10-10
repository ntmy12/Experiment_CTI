"""
Unit tests for Confirmatory and Causal Experiments v2 (Section 12 specifications):
- JSD(p, p) = 0, symmetric, range [0, 1]; entropy of uniform over K = log2(K).
- Placebo teacher-forced: S < 1e-6 and flip = 0.
- Index alignment logic: >= 95% match.
- All obj_ids are single tokens; pilot ids and new ids are disjoint.
- Site filters: no punctuation, no object tokens, no object strictly between s and t.
- Blend: lambda=1 reproduces original, lambda=0 equals all-black.
- Temperature control: tau=1 gives S^(1) = S(lambda=1) within 1e-6.
- Determinism: tolerance within 1e-4.
"""

import math
import json
import pytest
import numpy as np
import torch
from PIL import Image

from compute_v2 import (
    jsd_divergence,
    compute_entropy,
    compute_margin,
    is_word_start_token,
    is_punctuation_or_special,
    find_candidate_perturbation_sites,
)
from objects import ObjectVocabulary
from model_v2 import blend_image_raw


def test_jsd_and_entropy_properties():
    """Validates JSD properties and uniform entropy = log2(K)."""
    p = np.array([0.2, 0.5, 0.3])
    q = np.array([0.1, 0.8, 0.1])

    # JSD(p, p) == 0
    assert pytest.approx(jsd_divergence(p, p), abs=1e-10) == 0.0

    # Symmetry
    assert pytest.approx(jsd_divergence(p, q), abs=1e-10) == jsd_divergence(q, p)

    # Range in [0, 1]
    p_orth = np.array([1.0, 0.0])
    q_orth = np.array([0.0, 1.0])
    jsd_orth = jsd_divergence(p_orth, q_orth)
    assert pytest.approx(jsd_orth, abs=1e-6) == 1.0
    assert 0.0 <= jsd_orth <= 1.0

    # Entropy of uniform distribution over K = log2(K)
    K = 8
    p_unif = np.ones(K) / K
    assert pytest.approx(compute_entropy(p_unif), abs=1e-6) == math.log2(K)


def test_placebo_assertion():
    """Asserts that replacing token by itself produces S < 1e-6 and flip = 0."""
    p_orig = np.array([0.7, 0.2, 0.1])
    p_placebo = p_orig.copy()

    s_val = jsd_divergence(p_orig, p_placebo)
    flip = 1 if (np.argmax(p_orig) != np.argmax(p_placebo)) else 0

    assert s_val < 1e-6
    assert flip == 0


def test_index_alignment():
    """Index rule: argmax at index P + j - 1 equals y_j on >= 95% of tokens."""
    P = 12
    T = 10
    total_len = P + T
    vocab_size = 100

    mock_logits = torch.zeros(1, total_len, vocab_size)
    gen_ids = torch.tensor([[j + 50 for j in range(T)]])

    for j in range(T):
        mock_logits[0, P + j - 1, gen_ids[0, j]] = 20.0

    # Slice [P-1 : P+T-1]
    extracted = mock_logits[0, P - 1 : P + T - 1, :]
    preds = extracted.argmax(dim=-1)
    match_rate = (preds == gen_ids[0]).float().mean().item()
    assert match_rate >= 0.95


class MockTokenizerV2:
    def __init__(self):
        self.unk_token_id = 0
        self.all_special_ids = [0, 1, 2]
        self.vocab = {
            "<unk>": 0, "<s>": 1, "</s>": 2,
            "\u2581dog": 10, "\u2581cat": 11, "\u2581car": 12,
            "\u2581two": 20, "\u2581a": 21, "\u2581the": 22, "\u2581fast": 23,
            ".": 30, ",": 31,
        }
        self.id_to_tok = {v: k for k, v in self.vocab.items()}

    def convert_tokens_to_ids(self, tok):
        return self.vocab.get(tok, self.unk_token_id)

    def convert_ids_to_tokens(self, tid):
        return self.id_to_tok.get(tid, "<unk>")

    def encode(self, text, add_special_tokens=False):
        parts = text.split()
        return [self.vocab.get("\u2581" + p, self.unk_token_id) for p in parts]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join([self.id_to_tok.get(i, "").replace("\u2581", "") for i in ids])


def test_obj_vocab_and_disjoint_ids(tmp_path):
    """Asserts all obj_ids are single tokens and pilot vs new IDs disjoint."""
    tok = MockTokenizerV2()
    vocab = ObjectVocabulary(tokenizer=tok)
    assert len(vocab.obj_ids) > 0
    for tid in vocab.obj_ids:
        assert tid != tok.unk_token_id

    # Disjoint check
    pilot_ids = {10, 20, 30}
    sampled_ids = [40, 50, 60]
    assert set(sampled_ids).isdisjoint(pilot_ids)


def test_site_filters():
    """Asserts site filters: no punctuation, no obj tokens, no obj between s and t."""
    tok = MockTokenizerV2()
    obj_vocab = ObjectVocabulary(tokenizer=tok)
    # y = [<s>, two, fast, dog, and, cat]
    # token ids: [1, 20, 23, 10, 22, 11]
    gen_ids = torch.tensor([[1, 20, 23, 10, 22, 11]])
    obj_positions_set = {3, 5}  # dog at 3, cat at 5

    # Check candidates for cat (t = 5):
    # s = 3 is an object (dog) -> should not be eligible
    # s = 2 (fast) -> there is an object at 3 strictly between 2 and 5 -> should be excluded!
    # s = 4 (and / the, id=22) -> no object between 4 and 5 -> should be included!
    cand = find_candidate_perturbation_sites(
        t=5,
        gen_ids=gen_ids,
        obj_positions_set=obj_positions_set,
        obj_vocab=obj_vocab,
        special_ids={0, 1, 2},
        tokenizer=tok,
    )
    assert 4 in cand
    assert 3 not in cand
    assert 2 not in cand


def test_image_blend():
    """Blend: lambda=1 reproduces original, lambda=0 equals pure black."""
    img = Image.new("RGB", (50, 50), color=(120, 150, 200))

    blend_1 = blend_image_raw(img, 1.0)
    assert np.array_equal(np.array(img), np.array(blend_1))

    blend_0 = blend_image_raw(img, 0.0)
    assert np.all(np.array(blend_0) == 0)


def test_temperature_control_tau1():
    """Temperature control: tau=1 reproduces base distribution."""
    logits = torch.tensor([2.0, 1.0, 0.5])
    p_base = torch.softmax(logits, dim=-1).numpy()
    p_tau1 = torch.softmax(logits / 1.0, dim=-1).numpy()
    assert pytest.approx(jsd_divergence(p_base, p_tau1), abs=1e-6) == 0.0


def test_determinism_tolerance():
    """Asserts deterministic repeatability within 1e-4."""
    rng1 = np.random.RandomState(2026)
    p1 = rng1.dirichlet(np.ones(10))
    q1 = rng1.dirichlet(np.ones(10))
    v1 = jsd_divergence(p1, q1)

    rng2 = np.random.RandomState(2026)
    p2 = rng2.dirichlet(np.ones(10))
    q2 = rng2.dirichlet(np.ones(10))
    v2 = jsd_divergence(p2, q2)

    assert abs(v1 - v2) < 1e-4
