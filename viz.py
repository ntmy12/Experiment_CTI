"""
Visualization and meeting presentation report generator for Pilot Experiment H2.1:
Generates fig1 through fig7 with matplotlib (DejaVu Sans, 200 dpi, colour-blind safe palette),
Vietnamese titles and labels, Latin math symbols (V_t, S_t),
and embeds all figures into a self-contained report.html.
"""

import os
import base64
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from PIL import Image
from scipy import stats

logger = logging.getLogger("pilot_h21")

# Configure Matplotlib styling
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11

CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]


def encode_image_base64(image_path: str) -> str:
    """Encodes an image file to base64 string for standalone HTML embedding."""
    if not os.path.isfile(image_path):
        return ""
    with open(image_path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    mime = "image/png" if ext == "png" else "image/jpeg"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def plot_fig1_caption_strips(
    df: pd.DataFrame,
    captions_data: Dict[str, Any],
    coco_dataset: Any,
    output_dir: str,
    top_n_images: int = 3,
) -> List[str]:
    """
    fig1_caption_strip_{k}.png:
    Left: COCO image.
    Right: Generated caption laid out with word tokens coloured by V_t,
    object words outlined, with mean S_t printed underneath.
    """
    saved_paths = []
    # Pick top images with most object tokens
    obj_counts = df.groupby("image_id")["t"].nunique().sort_values(ascending=False)
    selected_img_ids = obj_counts.head(top_n_images).index.tolist()

    for idx, img_id in enumerate(selected_img_ids):
        img_records = df[df["image_id"] == img_id]
        cap_info = captions_data.get(str(img_id), {})
        caption_text = cap_info.get("caption", "")

        fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.3]})
        fig.patch.set_facecolor("white")

        # Left panel: Image
        try:
            pil_img = coco_dataset.load_image(img_id)
            ax_img.imshow(pil_img)
            ax_img.set_title(f"Ảnh COCO (ID: {img_id})", fontsize=14, fontweight="bold", pad=10)
        except Exception as e:
            ax_img.text(0.5, 0.5, f"Ảnh {img_id}\n(Không tải được: {e})", ha="center", va="center")
        ax_img.axis("off")

        # Right panel: Text with V_t highlight
        ax_text.set_xlim(0, 10)
        ax_text.set_ylim(0, 10)
        ax_text.axis("off")
        ax_text.set_title("Đóng góp của ảnh ($V_t$) theo từng token", fontsize=14, fontweight="bold", pad=10)

        # Map position t to V_t and mean S_t
        t_to_vt = img_records.groupby("t")["V_t"].first().to_dict()
        t_to_st = img_records.groupby("t")["S_t"].mean().to_dict()
        t_to_word = img_records.groupby("t")["word"].first().to_dict()

        # Lay out words
        words = caption_text.split()
        cmap = matplotlib.colormaps["viridis"]
        norm = plt.Normalize(vmin=0.0, vmax=1.0)

        x_start, y_start = 0.5, 9.0
        line_height = 1.1
        cur_x = x_start
        cur_y = y_start

        # Approximate token-to-word matching
        word_idx = 0
        for w in words:
            # Check if this word matches an object in this image
            is_obj = False
            w_vt = 0.0
            w_st = 0.0
            for t_pos, obj_w in t_to_word.items():
                if obj_w and obj_w.lower() in w.lower():
                    is_obj = True
                    w_vt = t_to_vt.get(t_pos, 0.0)
                    w_st = t_to_st.get(t_pos, 0.0)
                    break

            word_len = len(w) * 0.28 + 0.3
            if cur_x + word_len > 9.5:
                cur_x = x_start
                cur_y -= line_height

            bg_color = cmap(norm(w_vt)) if is_obj else "#F0F0F0"
            border_color = "#D55E00" if is_obj else "none"
            border_width = 1.8 if is_obj else 0.0

            # Draw background box
            box = patches.FancyBboxPatch(
                (cur_x, cur_y - 0.25),
                word_len - 0.1,
                0.7,
                boxstyle="round,pad=0.1",
                facecolor=bg_color,
                edgecolor=border_color,
                linewidth=border_width,
            )
            ax_text.add_patch(box)

            # Draw word text
            text_color = "white" if (is_obj and w_vt > 0.4) else "black"
            ax_text.text(cur_x + 0.1, cur_y, w, fontsize=11, color=text_color, va="center")

            # If object, annotate S_t below
            if is_obj:
                ax_text.text(
                    cur_x + 0.1,
                    cur_y - 0.45,
                    f"$S_t$: {w_st:.2f}",
                    fontsize=8.5,
                    color="#D55E00",
                    fontweight="bold",
                )

            cur_x += word_len
            word_idx += 1

        out_path = os.path.join(output_dir, f"fig1_caption_strip_{idx+1}.png")
        plt.subplots_adjust(bottom=0.18, top=0.92, left=0.08, right=0.95)
        # Add colorbar for V_t
        cbar_ax = fig.add_axes([0.60, 0.06, 0.30, 0.035])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(r"Đóng góp của ảnh ($V_t \in [0, 1]$)", fontsize=10)

        plt.savefig(out_path, dpi=200)
        plt.close()
        saved_paths.append(out_path)

    return saved_paths


def plot_fig2_vt_vs_position(df: pd.DataFrame, analysis_res: Dict[str, Any], output_path: str):
    """
    fig2_Vt_vs_position.png:
    Scatter (alpha 0.4) + binned mean with bootstrap CI band.
    Message: ảnh đóng góp ít dần theo vị trí (tiền đề).
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor("white")

    # Scatter
    ax.scatter(df["t"], df["V_t"], alpha=0.35, color=CB_PALETTE[0], edgecolors="none", s=35, label="Điểm quan sát ($V_t$)")

    # Binned means and CIs
    binned_data = analysis_res.get("A0_premise", {}).get("binned_Vt", {})
    bin_centers = []
    bin_means = []
    ci_lowers = []
    ci_uppers = []

    # Map bin labels to approximate midpoint on x-axis
    midpoints = {"[0,20)": 10, "[20,40)": 30, "[40,60)": 50, "[60,80)": 70, "[80+]": 90, "[60+]": 75, "[40+]": 65}

    for b_label, vals in binned_data.items():
        mid = midpoints.get(b_label, 50)
        bin_centers.append(mid)
        bin_means.append(vals["mean"])
        ci_lowers.append(vals["ci_lower"])
        ci_uppers.append(vals["ci_upper"])

    if bin_centers:
        sorted_indices = np.argsort(bin_centers)
        bin_centers = np.array(bin_centers)[sorted_indices]
        bin_means = np.array(bin_means)[sorted_indices]
        ci_lowers = np.array(ci_lowers)[sorted_indices]
        ci_uppers = np.array(ci_uppers)[sorted_indices]

        ax.plot(bin_centers, bin_means, color="#D55E00", marker="o", linewidth=2.5, markersize=8, label="Trung bình theo bin")
        ax.fill_between(bin_centers, ci_lowers, ci_uppers, color="#D55E00", alpha=0.2, label="95% CI (Cluster Bootstrap)")

    sp = analysis_res.get("A0_premise", {}).get("spearman_Vt_t", {})
    sp_est = sp.get("estimate", 0.0)
    sp_l = sp.get("ci_lower", 0.0)
    sp_u = sp.get("ci_upper", 0.0)

    ax.set_title("Tiền đề P0: Đóng góp của ảnh ($V_t$) theo vị trí sinh ($t$)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Vị trí token trong câu sinh ($t$, 0-based)", fontsize=12)
    ax.set_ylabel("Đóng góp của ảnh ($V_t = JSD$)", fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.text(
        0.04, 0.12,
        f"Spearman($V_t$, $t$) = {sp_est:.3f} [{sp_l:.3f}, {sp_u:.3f}]\nexploratory (n=20 images)",
        transform=ax.transAxes,
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#CCCCCC"),
    )
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_fig3_st_flip_vs_position(df: pd.DataFrame, analysis_res: Dict[str, Any], output_path: str):
    """
    fig3_St_flip_vs_position.png:
    Two panels: (a) S_t vs t scatter + binned mean±CI; (b) flip rate per bin (bar + CI, n per bin printed on top).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("white")

    # Panel (a): S_t vs t
    ax1.scatter(df["t"], df["S_t"], alpha=0.35, color=CB_PALETTE[0], edgecolors="none", s=30, label="Quan sát ($S_t$)")

    binned_data = analysis_res.get("A1_sensitivity_vs_position", {}).get("binned_St_and_flip", {})
    midpoints = {"[0,20)": 10, "[20,40)": 30, "[40,60)": 50, "[60,80)": 70, "[80+]": 90, "[60+]": 75, "[40+]": 65}

    bin_centers, bin_st, st_low, st_up = [], [], [], []
    for b_label, vals in binned_data.items():
        mid = midpoints.get(b_label, 50)
        bin_centers.append(mid)
        bin_st.append(vals["mean_St"])
        st_low.append(vals["st_ci_lower"])
        st_up.append(vals["st_ci_upper"])

    if bin_centers:
        s_idx = np.argsort(bin_centers)
        bc = np.array(bin_centers)[s_idx]
        bs = np.array(bin_st)[s_idx]
        sl = np.array(st_low)[s_idx]
        su = np.array(st_up)[s_idx]
        ax1.plot(bc, bs, color="#009E73", marker="s", linewidth=2.5, markersize=7, label="Trung bình $S_t$")
        ax1.fill_between(bc, sl, su, color="#009E73", alpha=0.2, label="95% CI")

    sp1 = analysis_res.get("A1_sensitivity_vs_position", {}).get("spearman_St_t", {})
    ax1.set_title("(a) Độ nhạy tiền tố ($S_t$) theo vị trí ($t$)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Vị trí token trong câu sinh ($t$)", fontsize=12)
    ax1.set_ylabel("Độ nhạy tiền tố ($S_t = JSD$)", fontsize=12)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.text(
        0.04, 0.85,
        f"Spearman($S_t$, $t$) = {sp1.get('estimate', 0.0):.3f}\n[{sp1.get('ci_lower', 0.0):.3f}, {sp1.get('ci_upper', 0.0):.3f}]",
        transform=ax1.transAxes,
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor="#CCCCCC"),
    )
    ax1.legend(loc="upper right")

    # Panel (b): Flip rate per bin
    labels = list(binned_data.keys())
    flip_rates = [binned_data[l]["flip_rate"] for l in labels]
    fl_low = [binned_data[l]["flip_ci_lower"] for l in labels]
    fl_up = [binned_data[l]["flip_ci_upper"] for l in labels]
    ns = [binned_data[l]["n"] for l in labels]

    yerr_lower = [max(0.0, rate - low) for rate, low in zip(flip_rates, fl_low)]
    yerr_upper = [max(0.0, up - rate) for rate, up in zip(flip_rates, fl_up)]
    yerr = [yerr_lower, yerr_upper]

    x_pos = np.arange(len(labels))
    bars = ax2.bar(x_pos, flip_rates, yerr=yerr, capsize=5, color=CB_PALETTE[1], alpha=0.85, edgecolor="#333333")

    for i, (b, n_count) in enumerate(zip(bars, ns)):
        ax2.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.04, f"n={n_count}", ha="center", va="bottom", fontsize=10)

    ax2.set_title("(b) Tỷ lệ đổi nhãn (Flip Rate) theo bin vị trí", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Khoảng vị trí ($t$)", fontsize=12)
    ax2.set_ylabel("Tỷ lệ Flip (argmax thay đổi)", fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0.0, max(0.5, max(fl_up) + 0.15 if fl_up else 0.5))
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_fig4_st_vs_vt(df: pd.DataFrame, output_path: str):
    """
    fig4_St_vs_Vt.png (KEY FIGURE):
    Scatter of S_t vs V_t, points coloured by position bin, one fitted line PER bin.
    Message: cùng vị trí, V_t thấp thì S_t cao?
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")

    if "pos_bin" not in df.columns:
        from analysis import assign_position_bins
        df = assign_position_bins(df)

    bin_labels = sorted(df["pos_bin"].unique().dropna().tolist())
    cmap = matplotlib.colormaps["tab10"]

    for idx, b_label in enumerate(bin_labels):
        sub = df[df["pos_bin"] == b_label]
        if len(sub) == 0:
            continue
        c = cmap(idx % 10)
        ax.scatter(sub["V_t"], sub["S_t"], color=c, alpha=0.45, s=35, label=f"Bin {b_label} (n={len(sub)})")

        # Fit line if enough points
        if len(sub) >= 5 and sub["V_t"].nunique() > 1:
            slope, intercept, r_val, _, _ = stats.linregress(sub["V_t"], sub["S_t"])
            x_vals = np.linspace(sub["V_t"].min(), sub["V_t"].max(), 50)
            y_vals = np.clip(slope * x_vals + intercept, 0.0, 1.0)
            ax.plot(x_vals, y_vals, color=c, linewidth=2.2, linestyle="-")

    ax.set_title("Kiểm định P2: Độ nhạy tiền tố ($S_t$) so với Đóng góp ảnh ($V_t$)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Đóng góp của ảnh ($V_t$)", fontsize=12)
    ax.set_ylabel("Độ nhạy tiền tố ($S_t$)", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

    ax.text(
        0.04, 0.08,
        "Đường hồi quy riêng cho từng bin vị trí\nexploratory (n=20 images)",
        transform=ax.transAxes,
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor="#CCCCCC"),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_fig5_mediation(analysis_res: Dict[str, Any], output_path: str):
    """
    fig5_mediation.png:
    Bar chart of the coefficient of t in M1 vs M2 (with CI) and the proportion mediated.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.patch.set_facecolor("white")

    ols_res = analysis_res.get("A3_mediation", {}).get("OLS", {})
    m1_bt = ols_res.get("M1_b_t", {})
    m2_bt = ols_res.get("M2_b_t", {})
    m2_bvt = ols_res.get("M2_b_Vt", {})
    prop_med = ols_res.get("proportion_mediated", {})

    # Left: Regression coefficients
    models = [
        "M1: Không có $V_t$\n" + r"($S_t \sim t+k+...$)",
        "M2: Có kiểm soát $V_t$\n" + r"($S_t \sim t+V_t+...$)",
    ]
    vals = [m1_bt.get("estimate", 0.0), m2_bt.get("estimate", 0.0)]
    lows = [m1_bt.get("ci_lower", 0.0), m2_bt.get("ci_lower", 0.0)]
    ups = [m1_bt.get("ci_upper", 0.0), m2_bt.get("ci_upper", 0.0)]

    yerr = [[v - l for v, l in zip(vals, lows)], [u - v for v, u in zip(vals, ups)]]

    bars1 = ax1.bar(models, vals, yerr=yerr, capsize=6, color=[CB_PALETTE[0], CB_PALETTE[2]], width=0.45)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_title("Hệ số chuẩn hóa của vị trí $t$ ($b_t$)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Hệ số chuẩn hóa $b_t$", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    for b in bars1:
        h = b.get_height()
        va_pos = "bottom" if h >= 0 else "top"
        ax1.text(b.get_x() + b.get_width() / 2.0, h + (0.01 if h >= 0 else -0.02), f"{h:.3f}", ha="center", va=va_pos, fontweight="bold")

    # Right: Proportion mediated
    pm_val = prop_med.get("estimate", 0.0)
    pm_l = prop_med.get("ci_lower", 0.0)
    pm_u = prop_med.get("ci_upper", 0.0)

    pm_err = [[pm_val - pm_l], [pm_u - pm_val]]
    bar2 = ax2.bar(["Tỷ lệ trung gian\n(Proportion Mediated)"], [pm_val], yerr=pm_err, capsize=6, color=CB_PALETTE[3], width=0.4)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("Độ suy giảm của hiệu ứng vị trí qua $V_t$", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Tỷ lệ $(b_t(M1) - b_t(M2)) / b_t(M1)$", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    h2 = bar2[0].get_height()
    va_pos2 = "bottom" if h2 >= 0 else "top"
    ax2.text(bar2[0].get_x() + bar2[0].get_width() / 2.0, h2 + (0.02 if h2 >= 0 else -0.04), f"{pm_val:.2%}", ha="center", va=va_pos2, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_fig6_case_studies(df: pd.DataFrame, output_dir: str) -> List[str]:
    """
    fig6_case_{i}.png (4 cases: 2 with largest S_t, 2 with S_t ~ 0):
    Shows original prefix vs perturbed prefix (changed token highlighted),
    and two side-by-side horizontal bar charts of top-5 object probabilities before/after.
    """
    saved_cases = []
    if len(df) == 0:
        return saved_cases

    # Sort records by S_t
    sorted_df = df.sort_values("S_t", ascending=False)
    high_cases = sorted_df.head(2)
    low_cases = sorted_df.tail(2)
    selected_cases = pd.concat([high_cases, low_cases], ignore_index=True)

    for i, row in selected_cases.iterrows():
        fig = plt.figure(figsize=(12, 5.5))
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(2, 2, height_ratios=[0.45, 1])

        ax_text = fig.add_subplot(gs[0, :])
        ax_p1 = fig.add_subplot(gs[1, 0])
        ax_p2 = fig.add_subplot(gs[1, 1])

        # Header case info
        s_t = row["S_t"]
        flip = row["flip"]
        t = row["t"]
        s = row["s"]
        orig_tok = row["orig_token"]
        alt_tok = row["alt_token"]
        obj_word = row["word"]

        ax_text.axis("off")
        title_text = (
            f"Trường hợp {i+1}: {('Độ nhạy cực cao (S_t lớn)' if s_t > 0.3 else 'Độ nhạy gần 0 (Miễn nhiễm)')} "
            f"| Object: '{obj_word}' (t={t}) | Can thiệp tại s={s}: '{orig_tok}' -> '{alt_tok}'"
        )
        ax_text.text(0.0, 0.75, title_text, fontsize=12, fontweight="bold", color="#111111")
        delta_p_val = row["delta_p"]
        ax_text.text(
            0.0, 0.25,
            f"Kết quả: $S_t$ = {s_t:.4f} | Flip = {flip} | " + r"$\Delta p$" + f" = {delta_p_val:.4f} | $V_t$ = {row['V_t']:.4f}",
            fontsize=11, color="#D55E00" if s_t > 0.3 else "#0072B2", fontweight="bold"
        )

        # Parse top3_before and top3_after
        def parse_top_objs(s_str):
            items = []
            if pd.isna(s_str) or not str(s_str).strip():
                return items
            for part in str(s_str).split(","):
                if ":" in part:
                    w, p = part.split(":", 1)
                    try:
                        items.append((w.strip(), float(p)))
                    except ValueError:
                        pass
            return items

        top_before = parse_top_objs(row.get("top3_before", ""))
        top_after = parse_top_objs(row.get("top3_after", ""))

        # Bar charts
        if top_before:
            w_b, p_b = zip(*top_before)
            y_pos = np.arange(len(w_b))
            ax_p1.barh(y_pos, p_b, color=CB_PALETTE[0], alpha=0.85)
            ax_p1.set_yticks(y_pos)
            ax_p1.set_yticklabels(w_b, fontsize=11)
            ax_p1.invert_yaxis()
            ax_p1.set_xlabel("Xác suất phân phối $p_{img}$ (Gốc)", fontsize=11)
            ax_p1.set_xlim(0, 1.05)
            for y_i, v in zip(y_pos, p_b):
                ax_p1.text(v + 0.02, y_i, f"{v:.2f}", va="center", fontsize=9.5)

        if top_after:
            w_a, p_a = zip(*top_after)
            y_pos2 = np.arange(len(w_a))
            ax_p2.barh(y_pos2, p_a, color=CB_PALETTE[1], alpha=0.85)
            ax_p2.set_yticks(y_pos2)
            ax_p2.set_yticklabels(w_a, fontsize=11)
            ax_p2.invert_yaxis()
            ax_p2.set_xlabel("Xác suất phân phối $p_{pert}$ (Can thiệp)", fontsize=11)
            ax_p2.set_xlim(0, 1.05)
            for y_i, v in zip(y_pos2, p_a):
                ax_p2.text(v + 0.02, y_i, f"{v:.2f}", va="center", fontsize=9.5)

        ax_p1.grid(True, linestyle="--", alpha=0.5, axis="x")
        ax_p2.grid(True, linestyle="--", alpha=0.5, axis="x")

        out_path = os.path.join(output_dir, f"fig6_case_{i+1}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        saved_cases.append(out_path)

    return saved_cases


def plot_fig7_distance(df: pd.DataFrame, analysis_res: Dict[str, Any], output_path: str):
    """
    fig7_distance.png:
    S_t vs k (distance between perturbation site and object).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    # Scatter
    ax.scatter(df["k"], df["S_t"], alpha=0.35, color=CB_PALETTE[0], s=35, label="Quan sát")

    # Binned mean by k
    k_data = analysis_res.get("A4_distance", {}).get("by_distance_k", {})
    k_vals, means, lows, ups = [], [], [], []
    for k_str, val in k_data.items():
        k_vals.append(int(k_str))
        means.append(val["mean_St"])
        lows.append(val["st_ci_lower"])
        ups.append(val["st_ci_upper"])

    if k_vals:
        s_idx = np.argsort(k_vals)
        k_vals = np.array(k_vals)[s_idx]
        means = np.array(means)[s_idx]
        lows = np.array(lows)[s_idx]
        ups = np.array(ups)[s_idx]

        ax.plot(k_vals, means, color="#D55E00", marker="o", linewidth=2.5, label="Trung bình theo khoảng cách $k$")
        ax.fill_between(k_vals, lows, ups, color="#D55E00", alpha=0.2, label="95% CI")

    sp_k = analysis_res.get("A4_distance", {}).get("spearman_St_k", {})
    ax.set_title("Kiểm soát Khoảng cách can thiệp: Độ nhạy ($S_t$) theo $k = t - s$", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Khoảng cách token ($k = t - s$)", fontsize=12)
    ax.set_ylabel("Độ nhạy tiền tố ($S_t$)", fontsize=12)
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.text(
        0.04, 0.85,
        f"Spearman($S_t$, $k$) = {sp_k.get('estimate', 0.0):.3f}\n[{sp_k.get('ci_lower', 0.0):.3f}, {sp_k.get('ci_upper', 0.0):.3f}]",
        transform=ax.transAxes,
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor="#CCCCCC"),
    )
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_html_report(
    analysis_res: Dict[str, Any],
    fig_paths: Dict[str, Any],
    output_path: str,
):
    """
    Generates ONE self-contained HTML report with base64 embedded images,
    hypothesis text, definitions table, fig1-fig7 with interpretation guides,
    and a limitations box. Optimized for 1100px projector presentation.
    """
    logger.info(f"Generating standalone report.html with base64 embedded figures...")

    # Embed figures to base64
    b64_figs = {}
    for key, path in fig_paths.items():
        if isinstance(path, list):
            b64_figs[key] = [encode_image_base64(p) for p in path]
        else:
            b64_figs[key] = encode_image_base64(path)

    html_template = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Báo Cáo Thực Nghiệm Pilot H2.1: Phân Tích Độ Nhạy Tiền Tố Trên VLM</title>
  <style>
    :root {{
      --primary: #1e3a8a;
      --secondary: #0d9488;
      --accent: #d97706;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --highlight: #fef3c7;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "DejaVu Sans", sans-serif;
      line-height: 1.6;
      background-color: var(--bg);
      color: var(--text);
      font-size: 18px;
      margin: 0;
      padding: 24px;
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
    }}
    header {{
      background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
      color: white;
      padding: 32px;
      border-radius: 12px;
      margin-bottom: 28px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 10px 0;
      font-weight: 700;
    }}
    .badge {{
      display: inline-block;
      background: #fbbf24;
      color: #78350f;
      font-weight: 600;
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 14px;
      text-transform: uppercase;
      margin-bottom: 12px;
    }}
    .section-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 28px;
      margin-bottom: 28px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }}
    h2 {{
      color: var(--primary);
      font-size: 22px;
      border-bottom: 2px solid var(--border);
      padding-bottom: 8px;
      margin-top: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 18px 0;
      font-size: 16px;
    }}
    th, td {{
      padding: 12px 14px;
      border: 1px solid var(--border);
      text-align: left;
    }}
    th {{
      background-color: #f1f5f9;
      font-weight: 600;
    }}
    tr:nth-child(even) {{
      background-color: #f8fafc;
    }}
    .figure-block {{
      margin: 24px 0;
      text-align: center;
    }}
    .figure-block img {{
      max-width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid var(--border);
      box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }}
    .caption-box {{
      background: #f1f5f9;
      border-left: 4px solid var(--primary);
      padding: 14px 18px;
      margin-top: 12px;
      border-radius: 0 8px 8px 0;
      text-align: left;
      font-size: 16px;
    }}
    .interpretation {{
      margin-top: 10px;
      font-size: 15px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .supp-box {{
      background: #ecfdf5;
      border: 1px solid #a7f3d0;
      padding: 10px 14px;
      border-radius: 6px;
      color: #065f46;
    }}
    .ref-box {{
      background: #fff1f2;
      border: 1px solid #fecdd3;
      padding: 10px 14px;
      border-radius: 6px;
      color: #9f1239;
    }}
    .limitations-box {{
      background: #fffbeb;
      border: 1px solid #fef3c7;
      border-left: 6px solid #d97706;
      border-radius: 8px;
      padding: 20px;
      margin: 28px 0;
    }}
    .limitations-box h3 {{
      margin-top: 0;
      color: #92400e;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <span class="badge">Exploratory (n=20 images)</span>
      <h1>Báo Cáo Kết Quả Thực Nghiệm Pilot Giả Thuyết H2.1</h1>
      <p style="margin:0; opacity: 0.9;">Phân tích tương tác giữa Đóng góp của Ảnh (V_t), Vị trí câu (t) và Độ nhạy can thiệp tiền tố (S_t) trên mô hình Vision-Language (LLaVA-1.5-7B)</p>
    </header>

    <!-- SECTION 0: HYPOTHESIS -->
    <div class="section-card">
      <h2>0. Giả Thuyết Nghiên Cứu (Hypothesis H2.1)</h2>
      <ul>
        <li><strong>(P0 - Tiền đề / Reproduction):</strong> Đóng góp của ảnh (<strong>V_t</strong>) giảm dần khi vị trí sinh (<strong>t</strong>) tiến xa hơn trong chuỗi.</li>
        <li><strong>(P1):</strong> Xác suất mà một OBJECT thay đổi khi một token TRƯỚC ĐÓ bị thay đổi tăng dần theo vị trí xuất hiện của object trong chuỗi (vị trí càng trễ, càng nhạy với tiền tố).</li>
        <li><strong>(P2):</strong> Tại cùng một vị trí trong câu, các object có đóng góp ảnh thấp (<strong>V_t</strong> thấp) sẽ dễ bị đổi (<strong>S_t</strong> cao hơn, flip nhiều hơn).</li>
      </ul>
      <p><em>Mục đích: Pilot trên 20 ảnh COCO val2014 nhằm xác nhận tính khả thi của pipeline và quan sát chiều hướng hiệu ứng (direction of effects). Không quy nạp ý nghĩa thống kê tổng quát.</em></p>
    </div>

    <!-- SECTION 1: DEFINITIONS TABLE -->
    <div class="section-card">
      <h2>1. Bảng Định Nghĩa Toán Học & Ký Hiệu</h2>
      <table>
        <thead>
          <tr>
            <th>Ký hiệu</th>
            <th>Tên gọi</th>
            <th>Định nghĩa toán học / Triển khai</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>y_{{<t}}</strong></td>
            <td>Prefix tiền tố</td>
            <td>Các token mô hình tự sinh trước bước t (không tính prompt và token ảnh).</td>
          </tr>
          <tr>
            <td><strong>Object token (t)</strong></td>
            <td>Token vật thể</td>
            <td>Token y_t là token đầu tiên của một từ trong từ điển COCO 80 categories (K token ids đơn).</td>
          </tr>
          <tr>
            <td><strong>t</strong> & <strong>rel_pos</strong></td>
            <td>Vị trí token</td>
            <td>Chỉ số 0-based trong chuỗi sinh; rel_pos = t / T với T là tổng độ dài caption.</td>
          </tr>
          <tr>
            <td><strong>p(·)</strong></td>
            <td>Phân phối giới hạn</td>
            <td>Softmax trên logits của DUY NHẤT K token ID trong từ điển vật thể, chuẩn hóa tổng = 1 (Float32).</td>
          </tr>
          <tr>
            <td><strong>V_t</strong></td>
            <td>Đóng góp của ảnh</td>
            <td>JSD( p(· | image, y_{{<t}}), p(· | no image, y_{{<t}}) ), đo bằng teacher-forcing trên cùng prefix.</td>
          </tr>
          <tr>
            <td><strong>Can thiệp (s)</strong></td>
            <td>Perturbation</td>
            <td>Thay thế duy nhất một token y_s (s < t, cửa sổ [t-8, t-1]) bằng token thay thế hợp lý của mô hình, giữ nguyên ảnh.</td>
          </tr>
          <tr>
            <td><strong>S_t</strong></td>
            <td>Độ nhạy tiền tố</td>
            <td>JSD( p(· | image, y_{{<t}}), p(· | image, y'_{{<t}}) ) trên phân phối giới hạn object.</td>
          </tr>
          <tr>
            <td><strong>flip</strong></td>
            <td>Đổi nhãn argmax</td>
            <td>= 1 nếu argmax over restricted object vocab bị thay đổi sau can thiệp, ngược lại = 0.</td>
          </tr>
          <tr>
            <td><strong>JSD</strong></td>
            <td>Jensen-Shannon Div</td>
            <td>Divergence (bình phương khoảng cách JS), cơ số 2, giá trị chuẩn hóa nghiêm ngặt trong [0, 1].</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- SECTION 2: FIGURES & TESTS -->
    <div class="section-card">
      <h2>2. Kết Quả Thực Nghiệm & Trực Quan Hóa (Figures)</h2>

      <!-- FIG 1 -->
      <div class="figure-block">
        <h3>Hình 1: Dải Token Caption & Mức Đóng Góp Ảnh (V_t)</h3>
        {''.join([f'<img src="{b64}" alt="Fig 1 Strip" style="margin-bottom:16px;">' for b64 in b64_figs.get("fig1", [])])}
        <div class="caption-box">
          <strong>Nội dung kiểm định:</strong> Minh họa trực quan caption được sinh, nền mỗi từ thể hiện mức độ phụ thuộc vào ảnh (V_t), các từ chỉ object được viền khung kèm độ nhạy tiền tố trung bình S_t dưới mỗi từ.
        </div>
      </div>

      <!-- FIG 2 -->
      <div class="figure-block">
        <h3>Hình 2: Tiền Đề (P0) - Đóng Góp Của Ảnh (V_t) Theo Vị Trí (t)</h3>
        <img src="{b64_figs.get('fig2', '')}" alt="Fig 2 Vt vs Position">
        <div class="caption-box">
          <strong>Nội dung kiểm định:</strong> Kiểm tra xem sự chú ý và đóng góp thông tin từ ảnh có giảm dần theo bước sinh hay không.
          <div class="interpretation">
            <div class="supp-box"><strong>Ủng hộ P0:</strong> Đường trung bình V_t đi xuống rõ rệt, Spearman(V_t, t) âm.</div>
            <div class="ref-box"><strong>Bác bỏ P0:</strong> Đường V_t đi ngang hoặc tăng dần theo vị trí.</div>
          </div>
        </div>
      </div>

      <!-- FIG 3 -->
      <div class="figure-block">
        <h3>Hình 3: Giả Thuyết P1 - Độ Nhạy Tiền Tố (S_t) và Tỷ Lệ Flip Theo Vị Trí</h3>
        <img src="{b64_figs.get('fig3', '')}" alt="Fig 3 St and Flip vs Position">
        <div class="caption-box">
          <strong>Nội dung kiểm định:</strong> Kiểm tra xem việc can thiệp token trước đó có làm object ở cuối câu dễ bị lung lay hơn so với đầu câu hay không.
          <div class="interpretation">
            <div class="supp-box"><strong>Ủng hộ P1:</strong> S_t trung bình và tỷ lệ Flip tăng dần theo các bin vị trí t, Spearman(S_t, t) > 0.</div>
            <div class="ref-box"><strong>Bác bỏ P1:</strong> S_t và tỷ lệ Flip không đổi hoặc giảm khi t tăng.</div>
          </div>
        </div>
      </div>

      <!-- FIG 4 -->
      <div class="figure-block">
        <h3>Hình 4: Trọng Tâm P2 - Tương Quan Giữa S_t và V_t Cùng Vị Trí (Key Figure)</h3>
        <img src="{b64_figs.get('fig4', '')}" alt="Fig 4 St vs Vt">
        <div class="caption-box">
          <strong>Nội dung kiểm định:</strong> Cùng một khoảng vị trí, các object nhận ít tín hiệu từ ảnh (V_t thấp) có nhạy cảm hơn với tiền tố (S_t cao hơn) hay không?
          <div class="interpretation">
            <div class="supp-box"><strong>Ủng hộ P2:</strong> Các đường hồi quy trong từng bin có hệ số góc âm; Partial Spearman(S_t, V_t | t) âm.</div>
            <div class="ref-box"><strong>Bác bỏ P2:</strong> Hệ số góc dương hoặc gần bằng 0 (V_t không bảo vệ object khỏi bị lung lay tiền tố).</div>
          </div>
        </div>
      </div>

      <!-- FIG 5 -->
      <div class="figure-block">
        <h3>Hình 5: Phân Tích Trung Gian (Mediation Analysis M1 vs M2)</h3>
        <img src="{b64_figs.get('fig5', '')}" alt="Fig 5 Mediation">
        <div class="caption-box">
          <strong>Nội dung kiểm định:</strong> Liệu tác động của vị trí t lên S_t có được trung gian hóa (mediated) bởi sự suy giảm của V_t hay không?
          <div class="interpretation">
            <div class="supp-box"><strong>Có trung gian:</strong> Hệ số b_t(M2) co lại đáng kể so với b_t(M1), tỷ lệ trung gian > 20% rõ ràng.</div>
            <div class="ref-box"><strong>Không trung gian:</strong> Hệ số b_t hầu như không đổi giữa M1 và M2 (tác động qua kênh ngôn ngữ/cú pháp độc lập).</div>
          </div>
        </div>
      </div>

      <!-- FIG 6 -->
      <div class="figure-block">
        <h3>Hình 6: Phân Tích Tình Huống Cụ Thể (Case Studies)</h3>
        {''.join([f'<img src="{b64}" alt="Fig 6 Case" style="margin-bottom:16px;">' for b64 in b64_figs.get("fig6", [])])}
        <div class="caption-box">
          <strong>Nội dung kiểm định:</strong> So sánh trực tiếp phân phối xác suất top-5 object trước và sau can thiệp cho 2 ca S_t cao nhất và 2 ca S_t xấp xỉ 0.
        </div>
      </div>

      <!-- FIG 7 -->
      <div class="figure-block">
        <h3>Hình 7: Kiểm Soát Khoảng Cách Can Thiệp (k = t - s)</h3>
        <img src="{b64_figs.get('fig7', '')}" alt="Fig 7 Distance">
        <div class="caption-box">
          <strong>Nội dung kiểm định:</strong> Kiểm tra ảnh hưởng của khoảng cách can thiệp k = t - s đến mức độ nhạy cảm S_t của token vật thể.
        </div>
      </div>

    </div>

    <!-- SECTION 3: LIMITATIONS -->
    <div class="limitations-box">
      <h3>Giới Hạn Của Thí Nghiệm Pilot (Limitations)</h3>
      <ul>
        <li><strong>Cỡ mẫu nhỏ (n = 20 ảnh):</strong> Mọi ước lượng chỉ phản ánh chiều hướng ban đầu (exploratory), không đại diện tổng thể cho phân phối COCO.</li>
        <li><strong>Một mô hình duy nhất:</strong> Chỉ thử nghiệm trên LLaVA-1.5-7B (kiến trúc CLIP ViT-L/14 + Vicuna-7B); chưa kiểm chứng trên Qwen2-VL, InternVL hoặc InstructBLIP.</li>
        <li><strong>Can thiệp cục bộ (Local Teacher-Forced):</strong> Can thiệp thay đổi 1 token đơn lẻ trong cửa sổ 8 bước gần nhất, chưa khảo sát can thiệp đa token hoặc can thiệp toàn bộ mệnh đề.</li>
        <li><strong>Phương thức thay thế token:</strong> Sử dụng top plausible alternative của chính mô hình; chưa thử nghiệm thay thế ngẫu nhiên hoặc đối kháng (adversarial).</li>
        <li><strong>Từ vựng giới hạn:</strong> Chỉ xét các từ vựng object đơn token (single-token words) để đảm bảo tính chặt chẽ của phân phối xác suất giới hạn.</li>
      </ul>
    </div>

  </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    logger.info(f"Successfully saved standalone HTML report to {output_path}")
