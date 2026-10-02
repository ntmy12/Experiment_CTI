"""
GATE G6: Experiment 2 -- Logit Lens Analysis at Position t-1
=============================================================

Copy paste từng cell dưới đây vào Kaggle notebook theo thứ tự.
Tất cả output ghi vào results/exp2_logitlens/ -- KHONG SUA bat ky file nao cua Experiment 1.
"""

# ============================================================
# CELL G6-A: Kiem tra dieu kien tien quyet
# ============================================================

import os

REPO = "/kaggle/working/Experiment_CTI"
labels_path = os.path.join(REPO, "data", "labels.jsonl")

if not os.path.exists(labels_path):
    raise FileNotFoundError(
        f"labels.jsonl not found at {labels_path}.\n"
        "Gate G3 or G5 from Experiment 1 must complete before running Gate G6."
    )

n_lines = sum(1 for _ in open(labels_path))
print(f"labels.jsonl found: {n_lines} caption records available.")

synonyms_path = os.path.join(REPO, "data", "synonyms.txt")
assert os.path.exists(synonyms_path), f"synonyms.txt not found at {synonyms_path}"
print("synonyms.txt found.")

script_path = os.path.join(REPO, "src", "06_logit_lens.py")
assert os.path.exists(script_path), f"06_logit_lens.py not found at {script_path}"
print("06_logit_lens.py found.")

print("All prerequisite checks passed. Proceeding to Logit Lens analysis.")


# ============================================================
# CELL G6-B: Chay Experiment 2 -- Logit Lens
# ============================================================

import subprocess, sys, os

REPO = "/kaggle/working/Experiment_CTI"

result = subprocess.run(
    [
        sys.executable, "src/06_logit_lens.py",
        "--labels_file",   "data/labels.jsonl",
        "--synonyms_file", "data/synonyms.txt",
        "--image_dir",     "data/val2014",
        "--model_name",    "llava-hf/llava-1.5-7b-hf",
        "--output_dir",    "results/exp2_logitlens",
        "--K",             "10",
        "--caliper",       "0.1",
        "--seed",          "0",
    ],
    cwd=REPO,
    text=True,
)

if result.returncode != 0:
    raise RuntimeError(f"Experiment 2 failed with exit code {result.returncode}.")
print("Experiment 2 (Logit Lens) completed successfully.")


# ============================================================
# CELL G6-C: Ket qua -- Tim layer phan hoa som nhat va manh nhat
# ============================================================

import pandas as pd
import os

REPO = "/kaggle/working/Experiment_CTI"
summary_path = os.path.join(REPO, "results", "exp2_logitlens", "layer_summary.csv")
df = pd.read_csv(summary_path)

sig_df = df[df["holm_p"] < 0.05].sort_values("layer")
peak_row = df.loc[df["mean_delta"].idxmin()]

print("=" * 60)
print("EXPERIMENT 2 -- LOGIT LENS RESULTS")
print("=" * 60)
print(f"Total layers analyzed  : {len(df)}")
print(f"Layers Holm p < 0.05   : {sorted(sig_df['layer'].tolist())}")

if len(sig_df) > 0:
    first_sig = int(sig_df["layer"].min())
    print(f"First significant layer: {first_sig}")
    print(f"  Delta = {float(sig_df[sig_df['layer']==first_sig]['mean_delta'].values[0]):.4f}")
    print(f"  Holm p = {float(sig_df[sig_df['layer']==first_sig]['holm_p'].values[0]):.4e}")
else:
    print("No layers reached Holm p < 0.05.")

print(f"\nPeak divergence layer  : {int(peak_row['layer'])}")
print(f"  Delta = {float(peak_row['mean_delta']):.4f}")
print(f"  Holm p = {float(peak_row['holm_p']):.4e}")
print(f"  Median rank (halluc vs real): "
      f"{float(peak_row['median_rank_halluc']):.0f} vs "
      f"{float(peak_row['median_rank_real']):.0f}")

print("\n--- Full layer summary ---")
cols = ["layer", "mean_S_halluc", "mean_S_real", "mean_delta",
        "delta_ci_lo", "delta_ci_hi", "dz", "holm_p",
        "median_rank_halluc", "median_rank_real"]
print(df[cols].to_string(index=False))


# ============================================================
# CELL G6-D: Hien thi bieu do
# ============================================================

from IPython.display import Image as IPyImage, display
import os

REPO = "/kaggle/working/Experiment_CTI"
fig_dir = os.path.join(REPO, "results", "exp2_logitlens", "figures")

for fig_name in ["s_vs_layer.png", "delta_vs_layer.png", "rank_vs_layer.png"]:
    fig_path = os.path.join(fig_dir, fig_name)
    if os.path.exists(fig_path):
        print(f"--- {fig_name} ---")
        display(IPyImage(filename=fig_path))
    else:
        print(f"Figure not found: {fig_name}")


# ============================================================
# CELL G6-E: Dong goi ket qua de tai ve
# ============================================================

import subprocess, os

REPO = "/kaggle/working/Experiment_CTI"
archive_path = "/kaggle/working/exp2_logitlens_results.tar.gz"

subprocess.run(
    ["tar", "-czf", archive_path,
     "-C", os.path.join(REPO, "results"),
     "exp2_logitlens"],
    check=True
)

size_mb = os.path.getsize(archive_path) / 1024 / 1024
print(f"Archive: {archive_path} ({size_mb:.1f} MB)")
print("Download from Kaggle Output panel.")
