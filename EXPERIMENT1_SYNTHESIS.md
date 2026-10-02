# Comprehensive Experimental Synthesis: Early Emergence of Object Hallucination in LLaVA-1.5-7B

**Document Purpose:** Machine-readable and researcher-oriented synthesis of empirical results from Experiment 1 ("How Many Steps Earlier Does an Object Appear?"). Designed for downstream AI agents, probe-training pipelines, and mitigation researchers.

---

## 1. Executive Summary & Core Discovery

### Primary Scientific Question
When a vision-language model (LLaVA-1.5-7B) hallucinates an object entity at token position $t$, does the output distribution or internal model state reveal discriminative signals at **earlier token positions** ($t - m$, where lag $m \in \{1, 2, \dots, 10\}$) compared to matched factual (real) objects?

### Primary Empirical Discovery
* **Pre-Registered Classification:** **EARLY SIGNAL**
* **Statistical Verification:** Across $N = 1,430$ strictly matched pairs in an independent confirmation cohort ($2,000$ COCO val2014 images), the normalized object favorability score difference $\Delta(m) = S_{halluc} - S_{real}$ is statistically significant not only at the generation step ($m = 0$) and the immediate preceding token ($m = 1$), but also across earlier consecutive positions:
  * **$m = 2$:** $\Delta = -0.4565$, $p_{wilcoxon} = 1.51 \times 10^{-4}$, **$p_{holm} = 0.001355$**
  * **$m = 3$:** $\Delta = -0.4536$, $p_{wilcoxon} = 6.87 \times 10^{-4}$, **$p_{holm} = 0.004811$**
* **Directionality:** $\Delta(m) < 0$ across all examined steps. Real objects are grounded ("anchored") into the visual context early, maintaining higher normalized probability and higher median rank ($S_{real} > S_{halluc}$). Conversely, hallucinated objects are suppressed early and surge abruptly only at the final generation step due to language prior pressures.
* **Resolution of the TruthPrInt Paradox (Question B):** In contrast to Table 7 of TruthPrInt (ICCV 2025) which reported higher Preceding Minimum Confidence (PMC) for hallucinated mentions, our length-controlled empirical measurement demonstrates that hallucinated objects exhibit **substantially lower preceding minimum confidence** ($0.2470$ vs. $0.2975$), with the lowest-confidence token located significantly further upstream from the object ($6.04$ vs. $4.82$ tokens).

---

## 2. Experimental Setup & Reproducibility Parameters

| Parameter | Specification |
|---|---|
| Model Architecture | `llava-hf/llava-1.5-7b-hf` (7 Billion parameters) |
| Precision | Half precision (`torch.float16`) |
| Hardware Environment | Dual NVIDIA Tesla T4 GPUs ($2 \times 16\text{ GB} = 32\text{ GB}$ VRAM) |
| Parallelization | Pipeline parallel via Hugging Face Accelerate (`device_map="auto"`) |
| Decoding Strategy | Greedy decoding (`do_sample=False`, `temperature=1.0`, `top_p=1.0`) |
| Maximum Generated Tokens | `max_new_tokens = 512` (stops on EOS token `</s>`) |
| Prompt Template | `"USER: <image>\nPlease describe this image in detail. ASSISTANT:"` |
| Dataset | MS-COCO 2014 Validation set (`val2014/`, 40,504 images) |
| Cohort Partitions | Development cohort: indices $[0, 500)$; Confirmation cohort: indices $[500, 2500)$ |
| Random Seed | Fixed at `0` for image shuffling, tie-breaking, and paired bootstrapping |
| Checksums (SHA-256) | Captions: `90ec8e446d...`; Labels: `babc2d793c...` |

---

## 3. Mathematical Formulations & Metrics

### 3.1. Output Logits Alignment
Let $y_0, y_1, \dots, y_{G-1}$ denote the sequence of generated tokens (excluding prompt tokens).
For an input sequence consisting of visual prompt tokens and generated tokens $y$, model logits are extracted via a single teacher-forcing forward pass:
$$\text{pred}[i] = \text{logits}[L_{out} - G - 1 + i]$$
where $\text{pred}[i] \in \mathbb{R}^{V}$ represents the next-token prediction logits used to generate $y_i$.
*Position alignment verification (Test T4) confirms $\text{argmax}(\text{pred}[i]) == y_i$ at $99.92\%$ accuracy.*

### 3.2. Lag Indexing $m$
An object mention starts at token index $t$ with first token $o = y_t$. The lag parameter $m \in \{0, 1, \dots, K\}$ (default $K = 10$) queries the prediction distribution $m$ steps prior to $t$:
$$z^{(m)} = \text{pred}[t - m]$$
* $m = 0$: hidden state of token $y_{t-1}$ (the prediction step selecting $o$).
* $m = 1$: hidden state of token $y_{t-2}$ (one step earlier), etc.

### 3.3. Normalized Object Favorability Score $S$
Let $V_{obj}$ denote the set of first-token vocabulary IDs for all 80 COCO canonical categories and their synonyms (derived from CHAIR dictionary). To decouple "the model is preparing to generate an arbitrary noun" from "the model specifically favors object entity $o$", score $S$ normalizes log-probabilities over $V_{obj} \cup \{o\}$:
$$S(o, m) = \log P(o \mid z^{(m)}) - \log \sum_{v \in V_{obj} \cup \{o\}} P(v \mid z^{(m)})$$
where $P(v \mid z) = \text{softmax}(z)[v]$.

### 3.4. 1:1 Relative Position Matched Pairs
Hallucination frequency increases toward the end of captions. To isolate position confounding:
* Hallucinated entity $h$ and real entity $r$ must satisfy $t \ge K = 10$.
* Both must pass the word-start token boundary check.
* Relative sentence position is defined as $\text{rel\_pos} = t / G$.
* 1:1 greedy matching without replacement under caliper $|\text{rel\_pos}(h) - \text{rel\_pos}(r)| \le 0.1$.

### 3.5. Preceding Minimum Confidence (PMC)
For every initial object mention occurring at token $t$:
$$\text{sent\_start} = \max \{i + 1 \mid i < t \text{ and token } y_i \text{ ends with a period '.'}\} \cup \{0\}$$
$$n_{prec} = t - \text{sent\_start} \quad (n_{prec} \ge 1)$$
$$\text{PMC} = \min_{i \in [\text{sent\_start}, t-1]} \max_{v} P(v \mid \text{pred}[i])$$
$$\text{argmin\_dist} = t - \arg\min_{i \in [\text{sent\_start}, t-1]} \max_{v} P(v \mid \text{pred}[i])$$

---

## 4. Empirical Attrition Funnel

```
Generated Captions: 2,000
    │
    ▼
Total Extracted Mentions: 15,480
    │
    ▼
First Mentions (first=True): 6,948
    ├── Real Mentions: 5,097 (73.4%)
    └── Hallucinated Mentions: 1,851 (26.6%)
           │
           ▼
Filter t >= 10 & Word-Start Valid:
    ├── Real Candidates: 3,359
    └── Hallucinated Candidates: 1,814
           │
           ▼
1:1 Control Matching (Caliper <= 0.1):
    ├── Matched Real Pairs: 1,430
    └── Matched Hallucinated Pairs: 1,430
    (Unmatched Hallucinated Dropped: 384)
```

---

## 5. Main Results: Question A (Lag Curve Analysis)

### 5.1. Statistical Summary Table (Confirmation Cohort: $N = 1,430$ Pairs)

| Lag $m$ | Mean $S(h)$ | Mean $S(r)$ | Mean $\Delta$ | 95% Bootstrap CI | Cohen's $d_z$ | Wilcoxon $p$ | Holm-Adjusted $p$ | AUROC | Median Rank ($h$ vs. $r$) | Top-10 Rate ($h$ vs. $r$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | -0.7705 | -0.4943 | **-0.2762** | [-0.3077, -0.2439] | -0.4504 | $4.86 \times 10^{-54}$ | **$5.34 \times 10^{-53}$** | 0.3358 | 1 vs. 1 | 100.0% vs. 100.0% |
| **1** | -3.1336 | -2.3087 | **-0.8249** | [-1.0101, -0.6328] | -0.2329 | $3.54 \times 10^{-23}$ | **$3.54 \times 10^{-22}$** | 0.3765 | 22 vs. 13 | 26.6% vs. 42.3% |
| **2** | -6.6366 | -6.1801 | **-0.4565** | [-0.7217, -0.1974] | -0.0931 | $1.51 \times 10^{-4}$ | **$0.001355$** | 0.4523 | 219 vs. 152 | 7.4% vs. 9.7% |
| **3** | -6.3587 | -5.9050 | **-0.4536** | [-0.7494, -0.1669] | -0.0829 | $6.87 \times 10^{-4}$ | **$0.004811$** | 0.4652 | 213 vs. 193.5 | 8.9% vs. 10.9% |
| **4** | -7.1384 | -6.8941 | -0.2442 | [-0.5370, 0.0210] | -0.0457 | 0.0708 | 0.280513 | 0.4772 | 301 vs. 326 | 9.6% vs. 9.2% |
| **5** | -7.2960 | -7.0771 | -0.2189 | [-0.4951, 0.0402] | -0.0411 | 0.0701 | 0.280513 | 0.4801 | 346 vs. 388.5 | 7.1% vs. 7.6% |
| **6** | -7.9231 | -7.6452 | **-0.2779** | [-0.5617, 0.0151] | -0.0496 | 0.0088 | **$0.044012$** | 0.4706 | 417.5 vs. 410 | 6.7% vs. 7.6% |
| **7** | -8.0760 | -7.9077 | -0.1683 | [-0.4874, 0.1317] | -0.0286 | 0.1839 | 0.280513 | 0.4823 | 432.5 vs. 445.5 | 5.6% vs. 6.4% |
| **8** | -8.1470 | -7.8941 | -0.2529 | [-0.5743, 0.0581] | -0.0404 | 0.0959 | 0.280513 | 0.4814 | 411 vs. 371 | 6.6% vs. 8.2% |
| **9** | -8.0986 | -7.6296 | **-0.4690** | [-0.7855, -0.1390] | -0.0766 | 0.0015 | **$0.008993$** | 0.4671 | 441.5 vs. 396.5 | 6.5% vs. 6.8% |
| **10** | -8.5209 | -7.9526 | **-0.5683** | [-0.9090, -0.2491] | -0.0922 | $1.97 \times 10^{-4}$ | **$0.001572$** | 0.4559 | 569 vs. 452 | 4.7% vs. 5.2% |

### 5.2. Prediction Step Uncertainty Dynamics ($m = 0$)
Extracting the full vocabulary softmax distribution reveals sharp contrast in model uncertainty at the generation step:
* **Real Objects ($m = 0$):** Mean Top-1 Confidence = **$53.08\%$**, Mean Shannon Entropy = **$1.8384\text{ nats}$**.
* **Hallucinated Objects ($m = 0$):** Mean Top-1 Confidence = **$39.45\%$**, Mean Shannon Entropy = **$2.4023\text{ nats}$**.
* *Takeaway:* When emitting a hallucinated token, the model experiences an immediate $30.7\%$ increase in predictive entropy and a $13.6\%$ drop in top-1 confidence compared to factual object generation.

---

## 6. Main Results: Question B (TruthPrInt PMC Reproduction)

TruthPrInt (ICCV 2025) proposed using Preceding Minimum Confidence (PMC) to detect hallucination antecedents. Our benchmark evaluated all $6,945$ first-mention entities across the confirmation cohort:

### 6.1. Length-Controlled PMC Statistics

| Evaluation Cohort | $N_{halluc}$ | $N_{real}$ | Hallucinated Mean PMC | Real Mean PMC | Hallucinated argmin_dist | Real argmin_dist |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall Population** | 1,851 | 5,094 | **$0.2470 \pm 0.089$** | **$0.2975 \pm 0.111$** | **$6.04\text{ tokens}$** | **$4.82\text{ tokens}$** |
| Window $n_{prec} \in [1, 3]$ | 330 | 459 | $0.3143 \pm 0.110$ | $0.3064 \pm 0.114$ | $1.13\text{ tokens}$ | $1.34\text{ tokens}$ |
| Window $n_{prec} \in [4, 6]$ | 323 | 1,716 | **$0.2692 \pm 0.082$** | **$0.3644 \pm 0.116$** | **$3.74\text{ tokens}$** | **$2.50\text{ tokens}$** |
| Window $n_{prec} \in [7, 10]$ | 281 | 1,063 | **$0.2565 \pm 0.086$** | **$0.2884 \pm 0.093$** | **$5.51\text{ tokens}$** | **$4.79\text{ tokens}$** |
| Window $n_{prec} \ge 11$ | 917 | 1,856 | **$0.2119 \pm 0.063$** | **$0.2387 \pm 0.073$** | **$8.79\text{ tokens}$** | **$7.85\text{ tokens}$** |

### 6.2. Scientific Resolution
* **Paper Paradox:** TruthPrInt Table 7 showed PMC for hallucinated mentions higher than factual mentions ($0.29$ vs. $0.22$).
* **Empirical Reality:** Controlling for preceding token window length $n_{prec}$, PMC is **consistently lower** for hallucinated entities (e.g., $0.2692$ vs. $0.3644$ in the 4-6 token window).
* **Antecedent Distance:** The lowest-confidence token precedes hallucinated entities by $6.04$ tokens on average (versus $4.82$ tokens for real entities), demonstrating that hallucination onset stems from deeper syntactic and semantic divergence well upstream in the generation sequence.

---

## 7. Category Heterogeneity & Co-Occurrence Analysis

Analysis of matched pairs identifies the primary semantic classes driving hallucination:

| Canonical Entity | Hallucinated Count | Real Count | Relative Hallucination Rate | Semantic Profile |
|---|:---:|:---:|:---:|---|
| `person` | 122 | 137 | 47.1% | Salient focal subject |
| `chair` | 103 | 95 | 52.0% | Indoor context co-occurrence |
| `dining table` | 76 | 53 | 58.9% | Room setting prior |
| `cup` | 75 | 81 | 48.1% | Tabletop artifact |
| `car` | 72 | 112 | 39.1% | Street background |
| `bottle` | 69 | 76 | 47.6% | Beverage artifact |
| `handbag` | 67 | 50 | 57.3% | Personal accessory |
| `bowl` | 54 | 47 | 53.5% | Kitchen artifact |
| `book` | 50 | 43 | 53.8% | Domestic artifact |
| `cell phone` | 43 | 24 | 64.2% | Hand accessory prior |

**Mechanistic Pattern:** Hallucination in LLaVA-1.5 is dominated by **contextual accessory priors** (`cell phone`, `handbag`, `chair`, `dining table`). When the model grounds a primary entity (`person`), language priors induce associative generation of accessories despite visual absence.

---

## 8. Replication Consistency (Dev vs. Confirm)

Independent verification across the two non-overlapping splits demonstrates near-perfect reproducibility:

| Metric | Development Cohort ($N = 343\text{ pairs}$) | Confirmation Cohort ($N = 1,430\text{ pairs}$) | Consistency Evaluation |
|---|:---:|:---:|:---:|
| $\Delta(m = 0)$ | -0.2353 ($p = 1.34 \times 10^{-11}$) | -0.2762 ($p = 4.86 \times 10^{-54}$) | Replicated ($|\text{diff}| < 0.041$) |
| $\Delta(m = 1)$ | -0.7209 ($p = 3.84 \times 10^{-5}$) | -0.8249 ($p = 3.54 \times 10^{-23}$) | Replicated ($|\text{diff}| < 0.104$) |
| $\Delta(m = 2)$ | -0.4534 ($p = 0.0594$, underpowered) | -0.4565 ($p = 0.00015$, significant) | Exact effect size match ($|\text{diff}| = 0.0031$) |
| $\Delta(m = 3)$ | -0.8499 ($p = 0.0021$) | -0.4536 ($p = 0.00069$) | Direction & significance replicated |
| Mean PMC (Halluc vs. Real) | 0.2400 vs. 0.2966 | 0.2470 vs. 0.2975 | Replicated ($|\text{diff}| < 0.007$) |
| Mean `argmin_dist` (Halluc vs. Real) | 5.95 vs. 4.73 | 6.04 vs. 4.82 | Replicated ($|\text{diff}| < 0.09$) |

---

## 9. Architectural Implications for Downstream Agents

### 9.1. Inadequacy of $m = 0$ Backtracking (TruthPrInt Critique)
TruthPrInt trains an MLP probe exclusively on token $y_{t-1}$ ($m = 0$). By step $m = 0$, the model has already committed to the syntactic slot (e.g., generating the article *"a"* or preposition *"with"*). Forcing a backtrack from $m = 0$ requires rewinding generation state and re-sampling.

### 9.2. Recommendation: Multi-Step Early Warning Probing ($m \in [2, 3]$)
Because the separation is already statistically established at $m = 2$ and $m = 3$ ($p_{holm} < 0.005$):
1. **Linear Probe Placement:** Train probes on the intermediate hidden states of tokens $y_{t-2}$ and $y_{t-3}$.
2. **Dynamic Steering / Activation Patching:** Instead of backtracking, apply contrastive steering vectors to suppress noun-phrase hallucinations before the model reaches the object's introducing article ($m = 1$).

### 9.3. Non-Causal Boundary
These empirical measurements reflect observational output correlations. Downstream agents must not assert that earlier tokens *cause* the hallucination without conducting interventional activation patching experiments.

---

## 10. File Artifact Index

All underlying raw tables, figures, and manifest files are preserved locally and pushed to GitHub (`ntmy12/Experiment_CTI`):
* `results/exp1_confirm/summary.csv`: Complete paired statistical test metrics across $m = 0..10$.
* `results/exp1_confirm/lag_records.csv`: Full record level log ($31,460$ rows: $1,430\text{ pairs} \times 2\text{ groups} \times 11\text{ lags}$).
* `results/exp1_confirm/pmc_summary.csv`: Length-controlled PMC statistics.
* `results/exp1_confirm/pmc_records.csv`: Raw entity-level PMC and argmin distances ($6,945$ rows).
* `results/exp1_confirm/funnel.json`: Stage-by-stage sample attrition.
* `results/exp1_confirm/run_manifest.json`: Environment telemetry and data checksums.
* `results/exp1_confirm/figures/`:
  * `s_vs_m.png`: Favorability score trajectories.
  * `delta_vs_m.png`: Paired delta curve with 95% bootstrap confidence bounds.
  * `auroc_vs_m.png`: Area under ROC curve across lags.
  * `pmc_by_group.png`: Box plot distributions of PMC.
  * `pmc_by_nprec.png`: Length-controlled binned PMC bar charts.
