# EXPERIMENT 1 REPORT: "How Many Steps Earlier Does an Object Appear?"

*Generated on: 2026-10-01 22:17:01*

---

## 1. Executive Summary
- **Configuration:** LLaVA-1.5-7B greedy caption generation on COCO val2014; output distribution measured across lags $m = 0..10$.
- **Sample Size:** Matched 1:1 control pairs by relative caption position: **1430** pairs.
- **Primary Finding:** **Early Signal**.
- **Interpretation:** Holm-adjusted p < 0.05 across at least two consecutive m values in m >= 2. Signal emerges earlier than 1 step.

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
| Total generated captions | 2000 |
| Total extracted mentions | 15480 |
| First mentions (`first=True`) | 6948 |
| Hallucinated mentions | 1851 |
| Real mentions | 5097 |
| Hallucinated satisfying $t \ge K$ | 1814 |
| Real satisfying $t \ge K$ | 3359 |
| Hallucinated with valid word start | 1814 |
| Real with valid word start | 3359 |
| **Matched 1:1 pairs (`matched_pairs`)** | **1430** |
| Dropped unmatched hallucinated mentions | 384 |

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
|   0 |      1430 |         -0.7705 |       -0.4943 |      -0.2762 |       -0.3077 |       -0.2439 | -0.4504 |  4.85831e-54 | 5.34415e-53 |               0.3343 |    0.3358 |        0.3166 |        0.3559 |         1      |       1      |                  1   |                1   |
|   1 |      1430 |         -3.1336 |       -2.3087 |      -0.8249 |       -1.0101 |       -0.6328 | -0.2329 |  3.5439e-23  | 3.5439e-22  |               0.3818 |    0.3765 |        0.3566 |        0.3972 |         0.2664 |       0.4231 |                 22   |               13   |
|   2 |      1430 |         -6.6366 |       -6.1801 |      -0.4565 |       -0.7217 |       -0.1974 | -0.0931 |  0.000150603 | 0.00135543  |               0.4469 |    0.4523 |        0.4318 |        0.4727 |         0.0741 |       0.0965 |                219   |              152   |
|   3 |      1430 |         -6.3587 |       -5.905  |      -0.4536 |       -0.7494 |       -0.1669 | -0.0829 |  0.000687254 | 0.00481078  |               0.4524 |    0.4652 |        0.4447 |        0.4863 |         0.0888 |       0.1091 |                213   |              193.5 |
|   4 |      1430 |         -7.1384 |       -6.8941 |      -0.2442 |       -0.537  |        0.021  | -0.0457 |  0.0707947   | 0.280513    |               0.4678 |    0.4772 |        0.4565 |        0.4982 |         0.0958 |       0.0916 |                301   |              326   |
|   5 |      1430 |         -7.296  |       -7.0771 |      -0.2189 |       -0.4951 |        0.0402 | -0.0411 |  0.0701282   | 0.280513    |               0.4867 |    0.4801 |        0.4576 |        0.5016 |         0.0706 |       0.0762 |                346   |              388.5 |
|   6 |      1430 |         -7.9231 |       -7.6452 |      -0.2779 |       -0.5617 |        0.0151 | -0.0496 |  0.00880229  | 0.0440115   |               0.4706 |    0.4706 |        0.449  |        0.4931 |         0.0671 |       0.0755 |                417.5 |              410   |
|   7 |      1430 |         -8.076  |       -7.9077 |      -0.1683 |       -0.4874 |        0.1317 | -0.0286 |  0.18385     | 0.280513    |               0.4804 |    0.4823 |        0.4609 |        0.5026 |         0.0559 |       0.0643 |                432.5 |              445.5 |
|   8 |      1430 |         -8.147  |       -7.8941 |      -0.2529 |       -0.5743 |        0.0581 | -0.0404 |  0.0959436   | 0.280513    |               0.4783 |    0.4814 |        0.4608 |        0.5022 |         0.0664 |       0.0818 |                411   |              371   |
|   9 |      1430 |         -8.0986 |       -7.6296 |      -0.469  |       -0.7855 |       -0.139  | -0.0766 |  0.00149884  | 0.00899306  |               0.4706 |    0.4671 |        0.4462 |        0.4882 |         0.065  |       0.0678 |                441.5 |              396.5 |
|  10 |      1430 |         -8.5209 |       -7.9526 |      -0.5683 |       -0.909  |       -0.2491 | -0.0922 |  0.000196541 | 0.00157233  |               0.458  |    0.4559 |        0.4349 |        0.476  |         0.0469 |       0.0517 |                569   |              452   |

### Graphical Visualizations
- `figures/s_vs_m.png`: Mean normalized score $S$ for hallucinated versus real objects across lag $m$.
- `figures/delta_vs_m.png`: Paired difference $\Delta(m) = S(h) - S(r)$ with 95% Bootstrap CI.
- `figures/auroc_vs_m.png`: Discriminative capacity (AUROC) of $S$ separating hallucination across $m$.

---

## 6. Question B Results: Preceding Minimum Confidence (PMC)

### PMC Statistical Summary (`pmc_summary.csv`)
| subset        |   n_halluc |   n_real |   mean_pmc_halluc |   std_pmc_halluc |   mean_pmc_real |   std_pmc_real |   mean_argmin_dist_halluc |   mean_argmin_dist_real |
|:--------------|-----------:|---------:|------------------:|-----------------:|----------------:|---------------:|--------------------------:|------------------------:|
| overall       |       1851 |     5094 |            0.247  |           0.0892 |          0.2975 |         0.1106 |                      6.04 |                    4.82 |
| n_prec_1-3    |        330 |      459 |            0.3143 |           0.1104 |          0.3064 |         0.1136 |                      1.13 |                    1.34 |
| n_prec_4-6    |        323 |     1716 |            0.2692 |           0.0819 |          0.3644 |         0.1156 |                      3.74 |                    2.5  |
| n_prec_7-10   |        281 |     1063 |            0.2565 |           0.0861 |          0.2884 |         0.0933 |                      5.51 |                    4.79 |
| n_prec_>=11   |        917 |     1856 |            0.2119 |           0.0632 |          0.2387 |         0.0734 |                      8.79 |                    7.85 |
| matched_pairs |          0 |        0 |            0      |           0      |          0      |         0      |                      0    |                    0    |

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
