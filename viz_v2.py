"""
Visualization and Standalone HTML Report Generation for Confirmatory Experiments (v2).
Generates fig1 to fig10 meeting Section 10 specs, with base64 embedded standalone report.html.
"""

import os
import base64
import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logger = logging.getLogger("confirmatory_v2")

CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False


def encode_image_base64(path: str) -> str:
    """Encodes image file to base64 string."""
    if not os.path.isfile(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def plot_fig1_flip_by_k(df_e1: pd.DataFrame, df_e2: Optional[pd.DataFrame], output_path: str):
    """fig1_flip_by_k.png: flip rate by k=1..8, teacher-forced vs free-generation side by side."""
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    k_vals = np.arange(1, 9)
    width = 0.35

    rates_tf = []
    for k in k_vals:
        sub = df_e1[(df_e1["k"] == k) & (df_e1["is_placebo"] == 0) & (df_e1["low_mass"] == 0)]
        rates_tf.append(sub["flip_tf"].mean() if not sub.empty else 0.0)

    rates_free = []
    for k in k_vals:
        if df_e2 is not None and not df_e2.empty:
            sub2 = df_e2[(df_e2["k"] == k) & (df_e2["is_placebo"] == 0)]
            rates_free.append(sub2["flip_free_word"].mean() if not sub2.empty else 0.0)
        else:
            rates_free.append(0.0)

    rects1 = ax.bar(k_vals - width / 2, rates_tf, width, label="Teacher-forced (E1)", color=CB_PALETTE[0], alpha=0.85)
    rects2 = ax.bar(k_vals + width / 2, rates_free, width, label="Free-generation (E2)", color=CB_PALETTE[1], alpha=0.85)

    ax.set_title("Tỉ lệ object đổi theo khoảng cách k", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Khoảng cách token k = t - s", fontsize=12)
    ax.set_ylabel("Tỉ lệ đổi object (Flip rate)", fontsize=12)
    ax.set_xticks(k_vals)
    ax.set_ylim(0, max(max(rates_tf + [0.1]), max(rates_free + [0.1])) * 1.25)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(fontsize=11)

    for r in rects1:
        h = r.get_height()
        if h > 0:
            ax.text(r.get_x() + r.get_width() / 2, h + 0.01, f"{h:.1%}", ha="center", va="bottom", fontsize=9)
    for r in rects2:
        h = r.get_height()
        if h > 0:
            ax.text(r.get_x() + r.get_width() / 2, h + 0.01, f"{h:.1%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_fig2_heatmap_pos_k(df_e1: pd.DataFrame, output_path: str):
    """fig2_flip_by_pos_k_heatmap.png: heatmap flip rate by POS x k (cells with n<10 masked)."""
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    df_clean = df_e1[(df_e1["is_placebo"] == 0) & (df_e1["low_mass"] == 0)].copy()

    df_clean["k_group"] = df_clean["k"].apply(lambda k: str(k) if k in [1, 2, 3] else "4+")
    pos_tags = sorted([p for p in df_clean["pos_s"].unique() if p != "UNK"])

    cols = ["1", "2", "3", "4+"]
    matrix = np.full((len(pos_tags), len(cols)), np.nan)
    n_matrix = np.zeros((len(pos_tags), len(cols)), dtype=int)

    for i, p in enumerate(pos_tags):
        for j, c in enumerate(cols):
            sub = df_clean[(df_clean["pos_s"] == p) & (df_clean["k_group"] == c)]
            n_matrix[i, j] = len(sub)
            if len(sub) >= 10:
                matrix[i, j] = sub["flip_tf"].mean()

    cax = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1.0, aspect="auto")
    fig.colorbar(cax, ax=ax, label="Tỉ lệ Flip")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"k={c}" for c in cols], fontsize=11)
    ax.set_yticks(range(len(pos_tags)))
    ax.set_yticklabels(pos_tags, fontsize=11)
    ax.set_title("Tỉ lệ Flip theo Từ loại (POS) và Khoảng cách k [EXPLORATORY]", fontsize=13, fontweight="bold", pad=12)

    for i in range(len(pos_tags)):
        for j in range(len(cols)):
            n = n_matrix[i, j]
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, f"n={n}\n(<10)", ha="center", va="center", color="gray", fontsize=8.5)
            else:
                ax.text(j, i, f"{val:.1%}\nn={n}", ha="center", va="center", color="black", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_fig3_flip_type(df_e1: pd.DataFrame, output_path: str):
    """fig3_flip_type.png: shares of same_category vs cross_category flips by k."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    flips = df_e1[(df_e1["flip_tf"] == 1) & (df_e1["is_placebo"] == 0)].copy()

    k_vals = range(1, 5)
    same_shares, cross_shares = [], []

    for k in k_vals:
        sub = flips[flips["k"] == k] if k < 4 else flips[flips["k"] >= 4]
        if len(sub) > 0:
            same = (sub["flip_type"] == "same_category").mean()
            cross = (sub["flip_type"] == "cross_category").mean()
        else:
            same, cross = 0.0, 0.0
        same_shares.append(same)
        cross_shares.append(cross)

    labels = ["k=1", "k=2", "k=3", "k>=4"]
    ax.bar(labels, same_shares, label="Cùng category COCO (same_category)", color=CB_PALETTE[2], alpha=0.85)
    ax.bar(labels, cross_shares, bottom=same_shares, label="Khác category COCO (cross_category)", color=CB_PALETTE[1], alpha=0.85)

    ax.set_title("Cơ cấu Loại Đổi Từ (Flip Type) theo Khoảng cách k", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Tỉ trọng", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_fig4_position_strata(df_e1: pd.DataFrame, output_path: str):
    """fig4_position_strata.png: three panels (k=1, k=2, k=3): S_t and flip rate by position bin."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white", sharey=True)
    df_clean = df_e1[(df_e1["is_placebo"] == 0) & (df_e1["low_mass"] == 0)].copy()

    bins = [0, 20, 40, 60, 80, 200]
    bin_labels = ["[0,20)", "[20,40)", "[40,60)", "[60,80)", "[80+]"]
    df_clean["bin"] = pd.cut(df_clean["t"], bins=bins, labels=bin_labels, right=False)

    for idx, k_val in enumerate([1, 2, 3]):
        ax = axes[idx]
        sub = df_clean[df_clean["k"] == k_val]
        grouped = sub.groupby("bin", observed=False).agg({"S_t": "mean", "flip_tf": "mean", "t": "count"})

        x = np.arange(len(bin_labels))
        ax.plot(x, grouped["S_t"], marker="o", color=CB_PALETTE[0], label="Mức nhạy S_t", linewidth=2.2)
        ax.plot(x, grouped["flip_tf"], marker="s", color=CB_PALETTE[1], label="Tỉ lệ Flip", linewidth=2.2, linestyle="--")

        for i, count in enumerate(grouped["t"]):
            ax.text(x[i], -0.05, f"n={count}", ha="center", fontsize=8.5, color="gray")

        ax.set_title(f"Phân tầng k = {k_val}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=10)
        ax.set_xlabel("Vị trí token t", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.set_ylabel("Giá trị", fontsize=11)
            ax.legend(fontsize=10)

    fig.suptitle("Độ nhạy S_t và Tỉ lệ Flip theo Vị trí t (Phân tầng theo k)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_fig5_S_vs_V(df_e1: pd.DataFrame, output_path: str):
    """fig5_S_vs_V_strata.png: S_t (log scale) vs V_black within each k stratum."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    df_clean = df_e1[(df_e1["is_placebo"] == 0) & (df_e1["low_mass"] == 0)].copy()

    for idx, k_val in enumerate([1, 2, 3]):
        ax = axes[idx]
        sub = df_clean[df_clean["k"] == k_val]
        ax.scatter(sub["V_black"], np.log10(sub["S_t"] + 1e-6), alpha=0.45, color=CB_PALETTE[idx], s=35)

        ax.set_title(f"k = {k_val}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Đóng góp thị giác V_black", fontsize=11)
        if idx == 0:
            ax.set_ylabel("log10(S_t + 1e-6)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Quan hệ S_t và Đóng góp thị giác V_black theo phân tầng k", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_fig6_dose_response(df_e3: Optional[pd.DataFrame], output_path: str):
    """fig6_dose_response.png: V(lambda), H(lambda), S(lambda) versus lambda."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    if df_e3 is None or df_e3.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "Chưa có dữ liệu E3", ha="center", va="center")
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    sub = df_e3[(df_e3["is_placebo"] == 0) & (df_e3["is_far"] == 0)]
    grp = sub.groupby("lambda_val").mean(numeric_only=True)

    axes[0].plot(grp.index, grp["V_lambda"], marker="o", color=CB_PALETTE[0], linewidth=2.5)
    axes[0].set_title("V(lambda) theo lambda", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Tỉ lệ giữ ảnh lambda")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(grp.index, grp["H_lambda"], marker="s", color=CB_PALETTE[1], linewidth=2.5)
    axes[1].set_title("Entropy H(lambda) theo lambda", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Tỉ lệ giữ ảnh lambda")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(grp.index, grp["S_lambda"], marker="^", color=CB_PALETTE[2], linewidth=2.5)
    axes[2].set_title("Độ nhạy S(lambda) theo lambda", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Tỉ lệ giữ ảnh lambda")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_fig7_entropy_control(df_e3: Optional[pd.DataFrame], df_temp: Optional[pd.DataFrame], output_path: str):
    """fig7_entropy_control.png: S vs Entropy (Temperature control vs Image degradation)."""
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="white")
    if df_e3 is None or df_e3.empty:
        ax.text(0.5, 0.5, "Chưa có dữ liệu E3", ha="center", va="center")
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    sub_e3 = df_e3[(df_e3["is_placebo"] == 0) & (df_e3["is_far"] == 0)]
    grp_e3 = sub_e3.groupby("lambda_val").agg({"H_lambda": "mean", "S_lambda": "mean"}).sort_values("H_lambda")

    ax.plot(grp_e3["H_lambda"], grp_e3["S_lambda"], marker="o", color=CB_PALETTE[1], linewidth=2.5, label="Giảm nội dung ảnh (S(lambda))")

    if df_temp is not None and not df_temp.empty:
        grp_t = df_temp.groupby("tau").agg({"H_tau": "mean", "S_tau": "mean"}).sort_values("H_tau")
        ax.plot(grp_t["H_tau"], grp_t["S_tau"], marker="s", color=CB_PALETTE[0], linestyle="--", linewidth=2.5, label="Kiểm soát nhiệt độ (S^(tau))")

    ax.set_title("So sánh Hiệu ứng Giảm Ảnh và Hiệu ứng Entropy Nhiệt độ", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Entropy H (bits)", fontsize=11)
    ax.set_ylabel("Độ nhạy S", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_fig8_within_triplet_slopes(df_e3: Optional[pd.DataFrame], output_path: str):
    """fig8_within_triplet_slopes.png: histogram of per-triplet slopes of y vs V(lambda)."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    if df_e3 is None or df_e3.empty:
        ax.text(0.5, 0.5, "Chưa có dữ liệu E3", ha="center", va="center")
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    sub = df_e3[(df_e3["is_placebo"] == 0) & (df_e3["is_far"] == 0)].copy()
    sub["y"] = np.log10(sub["S_lambda"] + 1e-6)

    slopes = []
    for tid, grp in sub.groupby("triplet_id"):
        if len(grp) >= 3 and grp["V_lambda"].std() > 1e-6:
            slope, _, _, _, _ = stats.linregress(grp["V_lambda"], grp["y"])
            if not np.isnan(slope):
                slopes.append(slope)

    ax.hist(slopes, bins=25, color=CB_PALETTE[0], edgecolor="black", alpha=0.75)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.8, label="Đường 0")
    ax.set_title("Phân phối Hệ số Góc (Slope) của y theo V(lambda) trong từng bộ ba", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hệ số d(y)/d(V)", fontsize=11)
    ax.set_ylabel("Số lượng bộ ba", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_fig9_forest(results: Dict[str, Any], output_path: str):
    """fig9_forest.png: forest plot of standardised coefficients in HC/HD."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    names, ests, lows, highs = [], [], [], []

    hc = results.get("HC", {})
    if "estimate" in hc:
        names.append("t (vị trí - HC)")
        ests.append(hc["estimate"])
        lows.append(hc["ci_lower"])
        highs.append(hc["ci_upper"])

    hd = results.get("HD", {})
    if "estimate" in hd:
        names.append("V_black (đóng góp ảnh - HD)")
        ests.append(hd["estimate"])
        lows.append(hd["ci_lower"])
        highs.append(hd["ci_upper"])

    if not names:
        ax.text(0.5, 0.5, "Chưa có dữ liệu hồi quy", ha="center", va="center")
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    y_pos = np.arange(len(names))
    ax.errorbar(ests, y_pos, xerr=[np.array(ests) - np.array(lows), np.array(highs) - np.array(ests)], fmt="o", color=CB_PALETTE[1], elinewidth=2.2, capsize=5)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_title("Biểu đồ Forest Plot: Hệ số Hồi quy Chuẩn hóa (95% CI)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Hệ số chuẩn hóa", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_fig10_cases(cases: List[Dict[str, Any]], output_dir: str) -> List[str]:
    """fig10_case_{i}.png: 6 cases chosen automatically."""
    saved_paths = []
    for i, c in enumerate(cases[:6]):
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
        orig_w = c.get("word", "")
        alt_w = c.get("alt_token", "")
        k = c.get("k", 0)
        s_t = c.get("S_t", 0.0)

        text = (
            f"Case {i+1} (Ảnh ID: {c.get('image_id')}, k={k})\n\n"
            f"Đối tượng gốc: '{orig_w}' | Can thiệp thay thế: '{alt_w}'\n"
            f"Mức nhạy S_t = {s_t:.4f} | Flip = {c.get('flip_tf', 0)}\n\n"
            f"Top 3 trước: {c.get('top3_before', '')}\n"
            f"Top 3 sau:   {c.get('top3_after', '')}"
        )
        ax.text(0.05, 0.5, text, fontsize=11, va="center", family="monospace")
        ax.axis("off")
        p = os.path.join(output_dir, f"fig10_case_{i+1}.png")
        plt.tight_layout()
        plt.savefig(p, dpi=200)
        plt.close()
        saved_paths.append(p)
    return saved_paths


def generate_html_report_v2(
    results: Dict[str, Any],
    fig_paths: Dict[str, str],
    output_path: str,
):
    """
    Generates ONE self-contained HTML report (images base64-embedded, max-width 1100px, font 18px)
    in exact order:
    1. Question
    2. Pre-registered hypotheses
    3. Definitions
    4. Verdict table
    5. Figures with one-sentence caption stating what would SUPPORT or REFUTE
    6. Limitations box.
    """
    logger.info("Generating standalone report.html with base64 embedded figures...")

    b64_figs = {}
    for k, p in fig_paths.items():
        if isinstance(p, list):
            b64_figs[k] = [encode_image_base64(item) for item in p]
        else:
            b64_figs[k] = encode_image_base64(p)

    verdicts_table_html = """
    <table class="verdict-table">
      <tr><th>Hypothesis</th><th>Estimate</th><th>95% CI</th><th>Pre-specified criterion</th><th>Verdict</th></tr>
    """
    for hyp, key in [("HA1 (tf: k=1 vs k>=4)", "HA1"), ("HA2 (free: k=1 vs placebo)", "HA2"),
                     ("HB (Distance k)", "HB"), ("HC (Position t in near)", "HC"),
                     ("HD (Visual contribution V)", "HD"), ("HE (Causal image blend)", "HE")]:
        h_data = results.get(key, {})
        est = f"{h_data.get('estimate', 0):.4f}" if "estimate" in h_data else "-"
        ci = f"[{h_data.get('ci_lower', 0):.4f}, {h_data.get('ci_upper', 0):.4f}]" if "ci_lower" in h_data else "-"
        crit = h_data.get("criterion", "")
        verd = h_data.get("verdict", "INCONCLUSIVE")
        badge_cls = "badge-supported" if verd == "SUPPORTED" else ("badge-contrary" if verd == "CONTRARY" else "badge-neutral")
        verdicts_table_html += f"<tr><td><b>{hyp}</b></td><td><code>{est}</code></td><td><code>{ci}</code></td><td>{crit}</td><td><span class='badge {badge_cls}'>{verd}</span></td></tr>"
    verdicts_table_html += "</table>"

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Báo Cáo Nghiên Cứu Confirmatory & Causal Experiments v2</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 18px; line-height: 1.6; color: #222; background: #f8f9fa; margin: 0; padding: 24px; }}
    .container {{ max-width: 1100px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
    h1 {{ font-size: 30px; color: #111; border-bottom: 3px solid #0072B2; padding-bottom: 12px; }}
    h2 {{ font-size: 24px; color: #0072B2; margin-top: 36px; border-bottom: 1px solid #e0e0e0; padding-bottom: 8px; }}
    .verdict-table {{ width: 100%; border-collapse: collapse; margin: 24px 0; font-size: 16px; }}
    .verdict-table th, .verdict-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
    .verdict-table th {{ background: #f1f3f5; font-weight: bold; }}
    .badge {{ padding: 4px 8px; border-radius: 6px; font-weight: bold; font-size: 14px; text-transform: uppercase; }}
    .badge-supported {{ background: #d4edda; color: #155724; }}
    .badge-contrary {{ background: #f8d7da; color: #721c24; }}
    .badge-neutral {{ background: #e2e3e5; color: #383d41; }}
    .fig-box {{ margin: 32px 0; text-align: center; background: #fafafa; padding: 20px; border-radius: 8px; border: 1px solid #eee; }}
    .fig-box img {{ max-width: 100%; height: auto; border-radius: 6px; }}
    .fig-caption {{ font-size: 16px; color: #555; margin-top: 12px; text-align: left; font-style: italic; }}
    .limitations-box {{ background: #fff3cd; border-left: 6px solid #ffc107; padding: 20px; margin-top: 40px; border-radius: 6px; font-size: 16px; }}
  </style>
</head>
<body>
<div class="container">
  <h1>Nghiên Cứu Xác Nhận & Can Thiệp Nhân Quả (v2): Độ Nhạy Tiền Tố trong VLM</h1>

  <h2>1. Câu hỏi Nghiên cứu (Research Question)</h2>
  <p>Chúng tôi nghiên cứu cơ chế <b>Lựa chọn Đối tượng (Object Selection)</b> trong mô hình LLaVA-1.5-7B khi sinh mô tả ảnh: Liệu việc thay đổi token tiền tố có làm thay đổi từ chỉ đối tượng được phát ra ở các vị trí tiếp theo, và sự thay đổi này phụ thuộc như thế nào vào khoảng cách <code>k</code>, vị trí <code>t</code> và mức độ đóng góp thị giác <code>V_t</code>?</p>

  <h2>2. Giả thuyết Đăng ký Trước (Pre-registered Hypotheses)</h2>
  <ul>
    <li><b>HA (Existence):</b> Tỉ lệ đổi từ tại k=1 lớn hơn k>=4, và tỉ lệ đổi từ khi sinh tự do lớn hơn mức placebo.</li>
    <li><b>HB (Distance):</b> Mức nhạy S_t và tỉ lệ đổi từ giảm dần theo khoảng cách k.</li>
    <li><b>HC (Position):</b> Tại cự ly gần k in {{1, 2}}, S_t tăng theo vị trí t khi đã hiệu chỉnh các biến phụ.</li>
    <li><b>HD (Visual Contribution):</b> Tại cự ly gần, V_t có liên hệ nghịch với S_t sau khi kiểm soát entropy.</li>
    <li><b>HE (Causal Intervention):</b> Giảm nội dung ảnh làm tăng S vượt qua mức tăng gây ra thuần túy bởi entropy.</li>
  </ul>

  <h2>3. Định nghĩa Chuẩn hóa (Definitions)</h2>
  <p><b>V_black(t):</b> JSD(p_img, p_black). <b>S_t:</b> JSD(p_img, p_pert). <b>k = t - s:</b> khoảng cách token. Toàn bộ khoảng tin cậy tính bằng <b>Cluster Bootstrap theo ảnh (2.000 lượt resamples)</b>.</p>

  <h2>4. Bảng Phán Quyết Kết Quả (Verdict Table)</h2>
  {verdicts_table_html}

  <h2>5. Biểu đồ Thực nghiệm & Diễn giải Tiêu chí (Figures)</h2>

  <div class="fig-box">
    <img src="data:image/png;base64,{b64_figs.get('fig1', '')}" alt="Fig 1">
    <div class="fig-caption"><b>Hình 1 (HA, HB):</b> Tỉ lệ object đổi theo khoảng cách k (Teacher-forced vs Sinh tự do). <i>Ủng hộ nếu k=1 vượt trội rõ rệt so với k>=4; Bác bỏ nếu đi ngang qua mọi k.</i></div>
  </div>

  <div class="fig-box">
    <img src="data:image/png;base64,{b64_figs.get('fig2', '')}" alt="Fig 2">
    <div class="fig-caption"><b>Hình 2 (HF - Khám phá):</b> Heatmap tỉ lệ Flip theo Từ loại (POS) và k. <i>Cho thấy loại từ tiền tố nào nhạy cảm nhất.</i></div>
  </div>

  <div class="fig-box">
    <img src="data:image/png;base64,{b64_figs.get('fig3', '')}" alt="Fig 3">
    <div class="fig-caption"><b>Hình 3:</b> Cơ cấu loại đổi từ (Cùng nhóm category COCO vs Khác nhóm category).</div>
  </div>

  <div class="fig-box">
    <img src="data:image/png;base64,{b64_figs.get('fig4', '')}" alt="Fig 4">
    <div class="fig-caption"><b>Hình 4 (HC):</b> Mức nhạy S_t và Flip theo Vị trí t (Phân tầng theo k=1, 2, 3). <i>Ủng hộ HC nếu đường dốc lên; Bác bỏ nếu đi ngang hoặc cong hình chuông.</i></div>
  </div>

  <div class="fig-box">
    <img src="data:image/png;base64,{b64_figs.get('fig5', '')}" alt="Fig 5">
    <div class="fig-caption"><b>Hình 5 (HD):</b> Mức nhạy S_t theo Đóng góp thị giác V_black theo phân tầng k. <i>Ủng hộ HD nếu hệ số hồi quy đa biến âm rõ rệt.</i></div>
  </div>

  <div class="fig-box">
    <img src="data:image/png;base64,{b64_figs.get('fig6', '')}" alt="Fig 6">
    <div class="fig-caption"><b>Hình 6 (HE):</b> Phản ứng liều lượng V(lambda), H(lambda), S(lambda) khi làm mờ ảnh.</div>
  </div>

  <div class="fig-box">
    <img src="data:image/png;base64,{b64_figs.get('fig7', '')}" alt="Fig 7">
    <div class="fig-caption"><b>Hình 7 (HE - Key):</b> So sánh tác động của làm mờ ảnh (S(lambda)) so với làm phẳng entropy thuần túy (S^(tau)). <i>Ủng hộ HE nếu đường làm mờ ảnh nằm cao hơn rõ rệt so với đường nhiệt độ.</i></div>
  </div>

  <div class="fig-box">
    <img src="data:image/png;base64,{b64_figs.get('fig8', '')}" alt="Fig 8">
    <div class="fig-caption"><b>Hình 8:</b> Phân phối hệ số góc d(y)/d(V) trong từng bộ ba can thiệp.</div>
  </div>

  <div class="fig-box">
    <img src="data:image/png;base64,{b64_figs.get('fig9', '')}" alt="Fig 9">
    <div class="fig-caption"><b>Hình 9:</b> Forest plot các hệ số hồi quy chuẩn hóa (95% CI).</div>
  </div>

  <h2>6. Hộp Hạn chế Phương pháp Luận (Limitations Box)</h2>
  <div class="limitations-box">
    <b>Các giới hạn đã được kiểm soát của nghiên cứu:</b>
    <ul>
      <li>Nghiên cứu trên một kiến trúc mô hình đơn lẻ (LLaVA-1.5-7B).</li>
      <li>Can thiệp teacher-forced mang tính chất cục bộ trên chuỗi tiền tố.</li>
      <li>Tập từ vựng đối tượng chỉ gồm các từ đơn token thuộc bảng phân loại COCO.</li>
      <li>Trộn với ảnh đen không phải là phép loại bỏ ảnh hoàn hảo và đồng thời làm thay đổi entropy (do đó cần phép kiểm soát nhiệt độ).</li>
      <li>V_t là độ lệch phân phối JSD, không phải trọng số attention nội tại của mô hình.</li>
      <li>Dữ liệu nhãn ground-truth COCO chỉ được dùng làm biến phụ kiểm soát, không phản ánh nhãn đánh giá đúng/sai.</li>
    </ul>
  </div>
</div>
</body>
</html>
    """

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Saved standalone report.html to {output_path}")
