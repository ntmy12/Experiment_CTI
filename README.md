# Experiment 1: "How Many Steps Earlier Does an Object Appear?"

> **Research Objective:** Measure the probability and normalized favorability score $S$ of an object at positions *earlier* ($m = 0..K$ steps) than its actual generation step on the vision-language model **LLaVA-1.5-7B** (greedy captioning on COCO val2014), comparing **hallucinated** versus **real** objects matched 1:1 by relative sentence position. Includes a secondary analysis reproducing the **Preceding Minimum Confidence (PMC)** metric from TruthPrInt (ICCV 2025).

---

## Directory Structure

```text
.
├── EXPERIMENT1_SPEC.md              # Technical specification document
├── README.md                        # Project documentation and usage guide
├── CHANGELOG.md                     # Implementation history and design decisions
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Ignored files (data caches, results, checkpoints)
├── data/
│   ├── synonyms.txt                 # Synonyms mapping for 80 COCO categories (CHAIR standard)
│   └── README.md                    # Instructions for COCO 2014 val dataset setup
├── src/
│   ├── __init__.py
│   ├── common.py                    # Token alignment, mention extraction, and statistical testing
│   ├── 01_generate.py               # Step 1: LLaVA-1.5-7B greedy caption generation
│   ├── 02_label_chair.py            # Step 2: CHAIR annotation, word-start validation, CHAIR_S / CHAIR_I
│   ├── 03_lag_curve.py              # Step 3: 1:1 control matching, teacher-forcing forward pass, Question A
│   ├── 04_pmc.py                    # Step 4: Question B (TruthPrInt PMC) and length control analysis
│   └── 05_visualize_report.py       # Step 5: Figure generation, run_manifest.json, and REPORT.md
├── tests/
│   ├── __init__.py
│   ├── test_common.py               # Unit tests T1, T2, T3 (CPU executable)
│   └── test_gpu.py                  # Sanity checks T4, T5, T6, T7 (GPU alignment verification)
└── notebooks/
    └── run_kaggle.ipynb             # Interactive Kaggle notebook structured by Gate milestones
```

---

## Execution Guide on Kaggle

The notebook [`notebooks/run_kaggle.ipynb`](notebooks/run_kaggle.ipynb) is optimized for execution on Kaggle GPU instances (P100 or T4 x 2):

1. **Push Codebase to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "feat: complete experiment 1 pipeline and kaggle notebook"
   git remote add origin https://github.com/ntmy12/Experiment_CTI.git
   git push -u origin main
   ```
2. **Open Kaggle:** Create a new notebook at [kaggle.com](https://www.kaggle.com/) (select **GPU P100** or **GPU T4 x 2** under *Settings* -> *Accelerator*).
3. **Import Notebook:** Select *File* -> *Import Notebook* and upload `notebooks/run_kaggle.ipynb`.
4. **Repository Link:** Verify `GITHUB_REPO_URL` in Cell 2 matches your repository.
5. **Sequential Execution:** Execute cells sequentially following Gate milestones G0 through G5.
6. **Export Artifacts:** The final cell compresses the `results/` directory into `experiment1_results.zip` for download.

---

## Command-Line Interface (CLI) Execution

### 1. Environment Installation
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
Place the following files in the `data/` directory:
- `instances_val2014.json`
- `captions_val2014.json`
- `val2014/` (image directory)

### 3. Step-by-Step Pipeline

#### Milestone 0: Unit Testing (Gate G0)
```bash
python3 -m unittest discover -s tests -p "test_common.py" -v
```

#### Step 1: Caption Generation with LLaVA-1.5-7B
```bash
# Smoke test (3 images):
python3 src/01_generate.py --split smoke --n_images 3 --output_file data/smoke_captions.jsonl

# Development cohort (500 images, range [0, 500)):
python3 src/01_generate.py --split dev --offset 0 --n_images 500 --output_file data/captions.jsonl
```
*(Add `--load_8bit` when running on GPUs with 16 GB VRAM).*

#### Step 2: CHAIR Annotation and Mention Extraction
```bash
python3 src/02_label_chair.py \
  --captions_file data/captions.jsonl \
  --output_file data/labels.jsonl
```

#### Step 3: Teacher-Forcing Forward Pass and Lag Curve (Question A)
```bash
python3 src/03_lag_curve.py \
  --labels_file data/labels.jsonl \
  --output_dir results/exp1
```

#### Step 4: Preceding Minimum Confidence Reproduction (Question B)
```bash
python3 src/04_pmc.py \
  --labels_file data/labels.jsonl \
  --captions_file data/captions.jsonl \
  --output_dir results/exp1
```

#### Step 5: Visualization, Manifest, and Automated Report
```bash
python3 src/05_visualize_report.py \
  --captions_file data/captions.jsonl \
  --labels_file data/labels.jsonl \
  --output_dir results/exp1
```

---

## Output Artifacts Specification (`results/exp1/`)

| File | Description |
|---|---|
| `lag_records.csv` | Granular per-object metrics across lag $m=0..10$ (`pair_id, image_id, canon, group, t, G, rel_pos, m, logp, rank, S, conf, ent, logp_actual`) |
| `summary.csv` | Primary statistical summary across $m$ (mean $S$, paired $\Delta$, 95% Bootstrap CI, Wilcoxon p, Holm-adjusted p, AUROC, top-10 rates, median ranks) |
| `pmc_records.csv` | Preceding Minimum Confidence and `argmin_dist` values across first mentions |
| `pmc_summary.csv` | Aggregated PMC summary overall, partitioned by $n_{prec}$ bins, and on matched pairs |
| `funnel.json` | Attrition metrics across successive filtering stages |
| `run_manifest.json` | Hardware telemetry, GPU specifications, random seed, commit hash, and SHA-256 checksums |
| `REPORT.md` | Formal 10-section empirical report populated with verified metrics |
| `figures/*.png` | Publication-ready visual charts (`s_vs_m.png`, `delta_vs_m.png`, `auroc_vs_m.png`, `pmc_by_group.png`, `pmc_by_nprec.png`) |

---

## Scientific Evaluation Protocol

Empirical findings are classified into one of four pre-registered hypotheses:
1. **Early Signal:** Holm-adjusted p < 0.05 across at least two consecutive lag steps in $m \ge 2$.
2. **Final Step Only:** Significance observed at $m = 0$ (and possibly $m = 1$), but absent at $m \ge 2$.
3. **No Signal:** No lag index achieves statistical significance.
4. **Reverse Direction:** Statistically significant difference where hallucinated objects exhibit higher $S$ scores than real objects.

*Non-Causal Principle:* Reported outcomes reflect observational correlations in output token distributions and do not constitute causal claims regarding generation mechanics.
