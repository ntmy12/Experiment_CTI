# EXPERIMENT 1 REPORT: "How Many Steps Earlier Does an Object Appear?"

*Generated on: 2026-10-01 16:32:40*

---

## 1. Executive Summary
- **Configuration:** LLaVA-1.5-7B greedy caption generation on COCO val2014; output distribution measured across lags $m = 0..10$.
- **Sample Size:** Matched 1:1 control pairs by relative caption position: **71** pairs.
- **Primary Finding:** **Final Step Only**.
- **Interpretation:** Significant at m = 0 (and possibly m = 1), but absent at m >= 2. Consistent with the immediate preceding hypothesis.

---

## 2. Experimental Setup
- **Model:** `llava-hf/llava-1.5-7b-hf` (greedy decoding, `max_new_tokens=512`).
- **Standard Prompt:** `USER: <image>\nPlease describe this image in detail. ASSISTANT:`.
- **Examined Lags:** $K = 10$.
- **Matching Caliper:** $\le 0.1$.
- **Random Seed:** 0.

---

## 3. Data Funnel
| Funnel Stage | Count |
|---|---|
| Total generated captions | 100 |
| Total extracted mentions | 745 |
| First mentions (`first=True`) | 351 |
| Hallucinated mentions | 98 |
| Real mentions | 253 |
| Hallucinated satisfying $t \ge K$ | 95 |
| Real satisfying $t \ge K$ | 166 |
| Hallucinated with valid word start | 95 |
| Real with valid word start | 166 |
| **Matched 1:1 pairs (`matched_pairs`)** | **71** |
| Dropped unmatched hallucinated mentions | 24 |

---

## 4. Quality Assurance & Sanity Checks
- **Position Alignment Check (T4):** Verified `argmax(pred[i]) == gen_ids[i]` exceeds 98% threshold (precludes index off-by-one errors).
- **Prediction Step Rank Check (T5):** At $m = 0$, verified `rank == 1` for >= 99% of selected objects.
- **Global CHAIR Evaluation:** Benchmarked against reference literature (CHAIR_S ~ 19.6%, CHAIR_I ~ 6.0%).

---

## 5. Question A Results: Object Favorability S across Lag m

### Statistical Summary (`summary.csv`)
|   m |   n_pairs |   mean_S_halluc |   mean_S_real |   mean_delta |   delta_ci_lo |   delta_ci_hi |      dz |   wilcoxon_p |    holm_p |   frac_halluc_higher |   auroc_S |   auroc_ci_lo |   auroc_ci_hi |   top10_halluc |   top10_real |   median_rank_halluc |   median_rank_real |
|----:|----------:|----------------:|--------------:|-------------:|--------------:|--------------:|--------:|-------------:|----------:|---------------------:|----------:|--------------:|--------------:|---------------:|-------------:|---------------------:|-------------------:|
|   0 |        71 |         -0.7347 |       -0.5257 |      -0.209  |       -0.3581 |       -0.0642 | -0.3403 |   0.00283352 | 0.0311687 |               0.3803 |    0.3734 |        0.2848 |        0.4721 |         1      |       1      |                    1 |                  1 |
|   1 |        71 |         -3.0071 |       -2.1742 |      -0.8329 |       -1.5556 |       -0.0626 | -0.2603 |   0.0102593  | 0.102593  |               0.3239 |    0.3432 |        0.2549 |        0.4397 |         0.2113 |       0.4366 |                   24 |                 13 |
|   2 |        71 |         -6.7676 |       -6.0921 |      -0.6755 |       -1.7382 |        0.3368 | -0.1484 |   0.21373    | 1         |               0.4507 |    0.4289 |        0.3297 |        0.5212 |         0.0845 |       0.0704 |                  252 |                124 |
|   3 |        71 |         -7.0257 |       -5.7191 |      -1.3066 |       -2.6115 |       -0.0787 | -0.2381 |   0.0556502  | 0.445201  |               0.4225 |    0.3837 |        0.2916 |        0.4797 |         0.0282 |       0.1408 |                  240 |                247 |
|   4 |        71 |         -6.5019 |       -7.1227 |       0.6207 |       -0.5277 |        1.7551 |  0.1245 |   0.233337   | 1         |               0.5915 |    0.538  |        0.4376 |        0.6332 |         0.1972 |       0.0563 |                  351 |                270 |
|   5 |        71 |         -7.154  |       -6.7307 |      -0.4233 |       -1.6254 |        0.8101 | -0.0779 |   0.513627   | 1         |               0.4648 |    0.471  |        0.3762 |        0.5739 |         0.0986 |       0.0704 |                  319 |                326 |
|   6 |        71 |         -8.3746 |       -6.9648 |      -1.4098 |       -2.8028 |        0.0136 | -0.2364 |   0.048065   | 0.432585  |               0.4085 |    0.4075 |        0.316  |        0.502  |         0.1127 |       0.0423 |                  439 |                251 |
|   7 |        71 |         -8.023  |       -7.1829 |      -0.8401 |       -1.9303 |        0.2221 | -0.1766 |   0.160376   | 1         |               0.4225 |    0.4481 |        0.3562 |        0.5417 |         0.1127 |       0.0423 |                  396 |                170 |
|   8 |        71 |         -8.3897 |       -8.6551 |       0.2654 |       -1.1308 |        1.6813 |  0.0431 |   0.713836   | 1         |               0.5634 |    0.5062 |        0.4094 |        0.6022 |         0.0563 |       0.0704 |                  338 |                447 |
|   9 |        71 |         -7.7223 |       -8.6759 |       0.9536 |       -0.3339 |        2.3098 |  0.1652 |   0.178138   | 1         |               0.5352 |    0.5453 |        0.4515 |        0.6391 |         0.0986 |       0.0563 |                  334 |                513 |
|  10 |        71 |         -8.3879 |       -7.8225 |      -0.5654 |       -2.0874 |        0.9202 | -0.0916 |   0.562783   | 1         |               0.4789 |    0.4902 |        0.3964 |        0.5897 |         0.0141 |       0.0423 |                  343 |                427 |

### Graphical Visualizations
- `figures/s_vs_m.png`: Mean normalized score $S$ for hallucinated versus real objects across lag $m$.
- `figures/delta_vs_m.png`: Paired difference $\Delta(m) = S(h) - S(r)$ with 95% Bootstrap CI.
- `figures/auroc_vs_m.png`: Discriminative capacity (AUROC) of $S$ separating hallucination across $m$.

---

## 6. Question B Results: Preceding Minimum Confidence (PMC)

### PMC Statistical Summary (`pmc_summary.csv`)
| subset      |   n_halluc |   n_real |   mean_pmc_halluc |   std_pmc_halluc |   mean_pmc_real |   std_pmc_real |   mean_argmin_dist_halluc |   mean_argmin_dist_real |
|:------------|-----------:|---------:|------------------:|-----------------:|----------------:|---------------:|--------------------------:|------------------------:|
| overall     |         98 |      253 |            0.2402 |           0.0797 |          0.2857 |         0.1034 |                      6.1  |                    4.71 |
| n_prec_1-3  |         15 |       22 |            0.3186 |           0.0805 |          0.2865 |         0.0703 |                      1.27 |                    1.27 |
| n_prec_4-6  |         18 |       72 |            0.2595 |           0.0606 |          0.3597 |         0.1156 |                      4.11 |                    2.85 |
| n_prec_7-10 |         15 |       59 |            0.2503 |           0.1005 |          0.2877 |         0.0767 |                      4.8  |                    4.36 |
| n_prec_>=11 |         50 |      100 |            0.2067 |           0.0581 |          0.2311 |         0.0784 |                      8.66 |                    7.02 |

- Comparison with TruthPrInt (Table 7): Benchmark comparison examining whether hallucinated mentions display distinct PMC dynamics.

---

## 7. Question C & Sensitivity Analyses (Exploratory)
- Sensitivity evaluations including S1 (caliper 0.05), S2 (unmatched cohort), and S3 (category fixed-effect centering) are recorded in `results/exp1/`.

---

## 8. Interpretation & Scope
- The empirical findings represent observational correlations in token output distributions produced by LLaVA-1.5-7B.
- **Causal Non-Interference:** These results must strictly not be interpreted as causal claims regarding hallucination generation.

---

## 9. Methodological Limitations
1. Simplified CHAIR dictionary extraction may introduce minor classification noise.
2. Analysis is evaluated on a single architecture (LLaVA-1.5-7B) and dataset (COCO 2014 val).
3. Computation utilizes half-precision (float16) teacher-forcing representations.

---

## 10. Proposed Next Steps
- Probing internal representation hidden states across intermediate transformer layers prior to output projection.
- Implementing controlled activation intervention and steering experiments at early positions ($m \ge 2$).
