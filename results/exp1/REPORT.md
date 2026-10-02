# EXPERIMENT 1 REPORT: "How Many Steps Earlier Does an Object Appear?"

*Generated on: 2026-10-01 17:32:25*

---

## 1. Executive Summary
- **Configuration:** LLaVA-1.5-7B greedy caption generation on COCO val2014; output distribution measured across lags $m = 0..10$.
- **Sample Size:** Matched 1:1 control pairs by relative caption position: **343** pairs.
- **Primary Finding:** **Mixed / Discontinuous**.
- **Interpretation:** Isolated statistical significance observed at m = [np.int64(3)].

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
| Total generated captions | 500 |
| Total extracted mentions | 3836 |
| First mentions (`first=True`) | 1709 |
| Hallucinated mentions | 443 |
| Real mentions | 1266 |
| Hallucinated satisfying $t \ge K$ | 435 |
| Real satisfying $t \ge K$ | 822 |
| Hallucinated with valid word start | 435 |
| Real with valid word start | 822 |
| **Matched 1:1 pairs (`matched_pairs`)** | **343** |
| Dropped unmatched hallucinated mentions | 92 |

---

## 4. Quality Assurance & Sanity Checks
- **Position Alignment Check (T4):** Verified `argmax(pred[i]) == gen_ids[i]` exceeds 98% threshold (precludes index off-by-one errors).
- **Prediction Step Rank Check (T5):** At $m = 0$, verified `rank == 1` for >= 99% of selected objects.
- **Global CHAIR Evaluation:** Benchmarked against reference literature (CHAIR_S ~ 19.6%, CHAIR_I ~ 6.0%).

---

## 5. Question A Results: Object Favorability S across Lag m

### Statistical Summary (`summary.csv`)
|   m |   n_pairs |   mean_S_halluc |   mean_S_real |   mean_delta |   delta_ci_lo |   delta_ci_hi |      dz |   wilcoxon_p |      holm_p |   frac_halluc_higher |   auroc_S |   auroc_ci_lo |   auroc_ci_hi |   top10_halluc |   top10_real |   median_rank_halluc |   median_rank_real |
|----:|----------:|----------------:|--------------:|-------------:|--------------:|--------------:|--------:|-------------:|------------:|---------------------:|----------:|--------------:|--------------:|---------------:|-------------:|---------------------:|-------------------:|
|   0 |       343 |         -0.7426 |       -0.5073 |      -0.2353 |       -0.299  |       -0.171  | -0.3725 |  1.33931e-11 | 1.47324e-10 |               0.3149 |    0.3597 |        0.3191 |        0.4009 |         1      |       1      |                    1 |                  1 |
|   1 |       343 |         -3.0655 |       -2.3445 |      -0.7209 |       -1.0939 |       -0.3586 | -0.2016 |  3.8405e-05  | 0.00038405  |               0.3673 |    0.3819 |        0.3355 |        0.423  |         0.2857 |       0.4023 |                   21 |                 15 |
|   2 |       343 |         -6.5756 |       -6.1221 |      -0.4534 |       -0.9432 |        0.042  | -0.0975 |  0.0594489   | 0.475591    |               0.4519 |    0.4535 |        0.41   |        0.4954 |         0.0671 |       0.0933 |                  204 |                155 |
|   3 |       343 |         -6.7744 |       -5.9245 |      -0.8499 |       -1.4359 |       -0.2658 | -0.1526 |  0.00207165  | 0.0186448   |               0.4198 |    0.4341 |        0.3922 |        0.4766 |         0.0729 |       0.1195 |                  269 |                244 |
|   4 |       343 |         -7.0341 |       -6.7863 |      -0.2478 |       -0.7958 |        0.2784 | -0.0484 |  0.306069    | 1           |               0.4665 |    0.4868 |        0.4432 |        0.5288 |         0.1079 |       0.0991 |                  292 |                254 |
|   5 |       343 |         -7.2052 |       -7.3748 |       0.1697 |       -0.404  |        0.7732 |  0.0307 |  0.832368    | 1           |               0.484  |    0.5034 |        0.4604 |        0.5461 |         0.0671 |       0.07   |                  312 |                339 |
|   6 |       343 |         -7.5937 |       -7.9532 |       0.3595 |       -0.2317 |        0.9311 |  0.0646 |  0.287926    | 1           |               0.5335 |    0.5199 |        0.4754 |        0.5612 |         0.0729 |       0.0554 |                  345 |                397 |
|   7 |       343 |         -7.8349 |       -8.3567 |       0.5218 |       -0.0666 |        1.1092 |  0.0907 |  0.201386    | 1           |               0.516  |    0.5219 |        0.4781 |        0.566  |         0.0437 |       0.0466 |                  355 |                400 |
|   8 |       343 |         -8.3202 |       -8.0236 |      -0.2966 |       -0.9135 |        0.3314 | -0.0484 |  0.565931    | 1           |               0.5073 |    0.4867 |        0.4419 |        0.5304 |         0.0641 |       0.0583 |                  340 |                407 |
|   9 |       343 |         -8.0726 |       -8.0691 |      -0.0035 |       -0.6854 |        0.6896 | -0.0006 |  0.853226    | 1           |               0.481  |    0.4882 |        0.4445 |        0.534  |         0.0437 |       0.07   |                  467 |                476 |
|  10 |       343 |         -7.9583 |       -8.1113 |       0.1531 |       -0.4701 |        0.7687 |  0.0266 |  0.885562    | 1           |               0.516  |    0.5119 |        0.4667 |        0.5552 |         0.0554 |       0.0612 |                  462 |                437 |

### Graphical Visualizations
- `figures/s_vs_m.png`: Mean normalized score $S$ for hallucinated versus real objects across lag $m$.
- `figures/delta_vs_m.png`: Paired difference $\Delta(m) = S(h) - S(r)$ with 95% Bootstrap CI.
- `figures/auroc_vs_m.png`: Discriminative capacity (AUROC) of $S$ separating hallucination across $m$.

---

## 6. Question B Results: Preceding Minimum Confidence (PMC)

### PMC Statistical Summary (`pmc_summary.csv`)
| subset        |   n_halluc |   n_real |   mean_pmc_halluc |   std_pmc_halluc |   mean_pmc_real |   std_pmc_real |   mean_argmin_dist_halluc |   mean_argmin_dist_real |
|:--------------|-----------:|---------:|------------------:|-----------------:|----------------:|---------------:|--------------------------:|------------------------:|
| overall       |        443 |     1265 |            0.24   |           0.0773 |          0.2966 |         0.1124 |                      5.95 |                    4.73 |
| n_prec_1-3    |         72 |      118 |            0.2955 |           0.0725 |          0.3258 |         0.1224 |                      1.1  |                    1.33 |
| n_prec_4-6    |         77 |      400 |            0.2738 |           0.0703 |          0.3635 |         0.1143 |                      3.71 |                    2.55 |
| n_prec_7-10   |         75 |      268 |            0.2464 |           0.0805 |          0.2957 |         0.0972 |                      5.36 |                    4.42 |
| n_prec_>=11   |        219 |      479 |            0.2077 |           0.0633 |          0.2341 |         0.0754 |                      8.53 |                    7.56 |
| matched_pairs |        343 |      343 |            0.2397 |           0.0782 |          0.2602 |         0.0966 |                      6.04 |                    5.4  |

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
