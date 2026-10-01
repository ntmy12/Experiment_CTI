# CHANGELOG: Experiment 1 - "How Many Steps Earlier Does an Object Appear?"

This document tracks all system architecture implementations, design decisions, and revisions conforming to the requirements outlined in `EXPERIMENT1_SPEC.md`.

---

## [1.0.0] - 2026-10-01

### 1. Core Architecture and Pipeline Implementation
- **`src/common.py`**:
  - Implemented `pieces_to_text(pieces)`: accurately reconstructs SentencePiece/LLaMA token pieces into raw text strings, resolving whitespace prefixes (` ` / `\u2581`), byte newline tokens (`<0x0A>`), and arbitrary byte representations (`<0xNN>`). Returns exact character span boundaries `(char_start, char_end)` for each token piece.
  - Implemented `find_mentions(text, syn2canon)`: extracts object entities via `[A-Za-z]+`, prioritizing 2-word collocations prior to unigrams, applying regular lemmatization (`-ies -> -y`, `-es -> -`, `-s -> -`).
  - Implemented `check_word_start(spans, tok_idx, char_start)`: enforces valid token word-boundary conditions (`spans[tok_idx].start == char_start - 1` or `char_start`). Non-conforming mentions are filtered and logged to the data funnel (`dropped_not_word_start`).
  - Implemented `build_coco_gt`: merges category annotations from `instances_val2014` with entities extracted from the 5 human reference captions.
  - Implemented 1:1 control matching (`match_object_pairs`) on relative sentence position (`rel_pos`) with matching caliper $\le 0.1$, random tie-breaking, and sampling without replacement.
  - Implemented rigorous statistical methods: Paired Bootstrap Confidence Intervals ($B=2000$), Cohen's $d_z$, two-sided Wilcoxon signed-rank testing, Holm-Bonferroni step-down correction over $m = 0..K$, and Bootstrap AUROC.
- **`src/01_generate.py`**:
  - Added CLI parameters `--offset` and `--n_images` to enforce non-overlapping cohorts: development (`[0, 500)`), confirmation (`[500, 2500)`), and smoke (`[0, 3)`).
  - Integrated resumable execution: detects processed `image_id` entries in `captions.jsonl` and flushes buffers per record.
  - Supported `--load_8bit` and `--load_4bit` options to facilitate execution within 16 GB GPU constraints (P100 / T4).
- **`src/02_label_chair.py`**:
  - Implemented CHAIR evaluation, word-start boundary validation, and primary mention flagging (`first=True`).
  - Computed macro-level metrics `CHAIR_S` and `CHAIR_I`.
  - Outputted `data/labels.jsonl` matching Section 5 schema.
- **`src/03_lag_curve.py`**:
  - Batched evaluation by `image_id` ensuring a single teacher-forcing forward pass per image.
  - Validated tokenizer prefix conventions during vocabulary subset ($V_{obj}$) construction.
  - Recorded comprehensive step-level metrics: `logp`, `rank`, `S`, `conf`, `ent`, and `logp_actual`.
  - Recorded attrition metrics in `funnel.json`.
  - Conducted paired statistical analysis (`summary.csv`), matching stability evaluation across seeds 0..19, and sensitivity analyses (S1 caliper 0.05, S2 unmatched, S3 category fixed-effects).
- **`src/04_pmc.py`**:
  - Implemented Question B: reproduction of TruthPrInt Preceding Minimum Confidence (PMC).
  - Controlled for preceding token window length using $n_{prec}$ bins (1-3, 4-6, 7-10, >=11) and matched pairs.
  - Exported `pmc_records.csv` and `pmc_summary.csv`.
- **`src/05_visualize_report.py`**:
  - Automated generation of 5 publication-standard figures (300 DPI): `s_vs_m.png`, `delta_vs_m.png`, `auroc_vs_m.png`, `pmc_by_group.png`, `pmc_by_nprec.png`.
  - Produced `run_manifest.json` recording environment telemetry, hardware identifiers, random seed, and SHA-256 data checksums.
  - Automated generation of `REPORT.md` following the formal 10-section academic specification.
- **`tests/test_common.py`**:
  - Test T1: standard SentencePiece benchmark string validation, verifying token indices for `man`, `dogs`, `teddy bear`, and `mugs`.
  - Test T2: 1:1 control matching algorithm, caliper constraint, non-replacement, and seed determinism.
  - Test T3: paired bootstrap confidence intervals, Holm-Bonferroni correction, and AUROC calculation.
- **`tests/test_gpu.py`**:
  - Test T4: position alignment verification (`argmax(pred[i]) == gen_ids[i]` $\ge 98\%$) to prevent index off-by-one errors.
  - Test T5: verification that at $m = 0$, `rank == 1` for $\ge 99\%$ of selected objects.
- **`data/synonyms.txt`**:
  - Full synonym mappings for all 80 COCO categories conforming to CHAIR evaluation literature.
- **`notebooks/run_kaggle.ipynb`**:
  - Self-contained, modular Kaggle execution notebook structured across Gate milestones G0 through G5 with visual progress tracking.
