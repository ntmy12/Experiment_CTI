"""
Statistical analysis module for Pilot Experiment H2.1:
- Cluster bootstrap over images (2000 resamples) for all metrics and CIs.
- A0 premise: Spearman(V_t, t) and binned V_t.
- A1: Spearman(S_t, t), flip rate and mean S_t by position bin.
- A2: within-bin Spearman(S_t, V_t) and partial Spearman(S_t, V_t | t).
- A3 regression: standardized M1 and M2 OLS with cluster-robust SE and proportion mediated; logistic version for flip.
- A4 distance control: S_t and flip vs k.
- A5 robustness: A2 repeated with black-image V_t.
Outputs results/analysis.json and results/analysis.md.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

logger = logging.getLogger("pilot_h21")


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    stat_fn: Any,
    cluster_col: str = "image_id",
    n_resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Computes point estimate and 95% cluster-bootstrap confidence interval over clusters (images).
    Returns (point_estimate, ci_lower, ci_upper).
    """
    point_est = stat_fn(df)
    unique_clusters = df[cluster_col].unique()
    n_clusters = len(unique_clusters)

    if n_clusters < 2:
        return float(point_est), float(point_est), float(point_est)

    # Group dataframe by cluster for fast resampling
    cluster_groups = {c: group for c, group in df.groupby(cluster_col)}

    rng = np.random.RandomState(seed)
    boot_estimates = []

    for _ in range(n_resamples):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        boot_df = pd.concat([cluster_groups[c] for c in sampled_clusters], ignore_index=True)
        try:
            est = stat_fn(boot_df)
            if not np.isnan(est) and not np.isinf(est):
                boot_estimates.append(est)
        except Exception:
            continue

    if len(boot_estimates) < 10:
        return float(point_est), float(point_est), float(point_est)

    boot_estimates = np.array(boot_estimates)
    ci_lower = float(np.percentile(boot_estimates, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_estimates, 100 * (1.0 - alpha / 2.0)))

    return float(point_est), ci_lower, ci_upper


def compute_spearman(df: pd.DataFrame, col_x: str, col_y: str) -> float:
    """Computes Spearman rank correlation between col_x and col_y."""
    if len(df) < 3 or df[col_x].nunique() < 2 or df[col_y].nunique() < 2:
        return 0.0
    corr, _ = stats.spearmanr(df[col_x], df[col_y])
    return float(corr) if not np.isnan(corr) else 0.0


def compute_partial_spearman(df: pd.DataFrame, col_x: str, col_y: str, col_z: str) -> float:
    """
    Computes partial Spearman correlation r(X, Y | Z):
    Pearson correlation of ranks controlling for rank of Z.
    """
    if len(df) < 5 or df[col_x].nunique() < 2 or df[col_y].nunique() < 2 or df[col_z].nunique() < 2:
        return 0.0

    rx = df[col_x].rank()
    ry = df[col_y].rank()
    rz = df[col_z].rank()

    r_xy = stats.pearsonr(rx, ry)[0]
    r_xz = stats.pearsonr(rx, rz)[0]
    r_yz = stats.pearsonr(ry, rz)[0]

    denom = np.sqrt(max(0.0, 1.0 - r_xz**2) * max(0.0, 1.0 - r_yz**2))
    if denom < 1e-8:
        return 0.0
    r_xyz = (r_xy - r_xz * r_yz) / denom
    return float(np.clip(r_xyz, -1.0, 1.0))


def assign_position_bins(df: pd.DataFrame, min_rows_per_bin: int = 10) -> pd.DataFrame:
    """
    Assigns position bins: [0,20), [20,40), [40,60), [60,80), [80+].
    Merges adjacent bins with < 10 rows.
    """
    df = df.copy()
    bins = [0, 20, 40, 60, 80, 1000]
    labels = ["[0,20)", "[20,40)", "[40,60)", "[60,80)", "[80+]"]

    df["pos_bin"] = pd.cut(df["t"], bins=bins, right=False, labels=labels)

    # Check row counts and merge if needed
    bin_counts = df["pos_bin"].value_counts().to_dict()

    # Sequential merging for small bins from right to left
    merged_labels = list(labels)
    # If [80+] < min_rows, merge with [60,80)
    if bin_counts.get("[80+]", 0) < min_rows_per_bin:
        df["pos_bin"] = df["pos_bin"].replace({"[80+]": "[60+]", "[60,80)": "[60+]"})

    # Recalculate counts
    bin_counts = df["pos_bin"].value_counts().to_dict()
    if bin_counts.get("[60+]", 0) < min_rows_per_bin and "[60+]" in df["pos_bin"].values:
        df["pos_bin"] = df["pos_bin"].replace({"[60+]": "[40+]", "[40,60)": "[40+]"})

    return df


def run_full_analysis(
    df: pd.DataFrame,
    df_black: Optional[pd.DataFrame] = None,
    n_resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes analyses A0 through A5 with 2000 cluster-bootstrap resamples over images.
    Returns analysis dictionary with all point estimates and 95% CIs.
    """
    results: Dict[str, Any] = {
        "metadata": {
            "n_images": int(df["image_id"].nunique()),
            "n_records": len(df),
            "label": "exploratory (n=20 images)",
            "n_bootstrap": n_resamples,
        }
    }

    # Assign position bins with merging
    df = assign_position_bins(df)
    if df_black is not None and len(df_black) > 0:
        df_black = assign_position_bins(df_black)

    # =========================================================================
    # A0 premise: Spearman(V_t, t) and binned mean of V_t
    # =========================================================================
    v_t_spearman, v_t_ci_l, v_t_ci_u = cluster_bootstrap_ci(
        df,
        lambda d: compute_spearman(d, "V_t", "t"),
        n_resamples=n_resamples,
        seed=seed,
    )
    binned_vt = {}
    for b_label, group in df.groupby("pos_bin", observed=True):
        m, ci_l, ci_u = cluster_bootstrap_ci(
            group,
            lambda d: float(d["V_t"].mean()),
            n_resamples=n_resamples,
            seed=seed,
        )
        binned_vt[str(b_label)] = {"mean": m, "ci_lower": ci_l, "ci_upper": ci_u, "n": len(group)}

    results["A0_premise"] = {
        "spearman_Vt_t": {"estimate": v_t_spearman, "ci_lower": v_t_ci_l, "ci_upper": v_t_ci_u},
        "binned_Vt": binned_vt,
    }

    # =========================================================================
    # A1: Spearman(S_t, t); flip rate and mean S_t by position bin
    # =========================================================================
    s_t_spearman, s_t_ci_l, s_t_ci_u = cluster_bootstrap_ci(
        df,
        lambda d: compute_spearman(d, "S_t", "t"),
        n_resamples=n_resamples,
        seed=seed + 1,
    )
    binned_st_flip = {}
    for b_label, group in df.groupby("pos_bin", observed=True):
        m_st, st_l, st_u = cluster_bootstrap_ci(
            group,
            lambda d: float(d["S_t"].mean()),
            n_resamples=n_resamples,
            seed=seed,
        )
        m_flip, fl_l, fl_u = cluster_bootstrap_ci(
            group,
            lambda d: float(d["flip"].mean()),
            n_resamples=n_resamples,
            seed=seed,
        )
        binned_st_flip[str(b_label)] = {
            "mean_St": m_st, "st_ci_lower": st_l, "st_ci_upper": st_u,
            "flip_rate": m_flip, "flip_ci_lower": fl_l, "flip_ci_upper": fl_u,
            "n": len(group),
        }

    results["A1_sensitivity_vs_position"] = {
        "spearman_St_t": {"estimate": s_t_spearman, "ci_lower": s_t_ci_l, "ci_upper": s_t_ci_u},
        "binned_St_and_flip": binned_st_flip,
    }

    # =========================================================================
    # A2: within each position bin Spearman(S_t, V_t) + overall partial Spearman(S_t, V_t | t)
    # =========================================================================
    within_bin_spearman = {}
    for b_label, group in df.groupby("pos_bin", observed=True):
        if len(group) >= 5:
            sp, sp_l, sp_u = cluster_bootstrap_ci(
                group,
                lambda d: compute_spearman(d, "S_t", "V_t"),
                n_resamples=n_resamples,
                seed=seed,
            )
            within_bin_spearman[str(b_label)] = {"spearman": sp, "ci_lower": sp_l, "ci_upper": sp_u, "n": len(group)}

    partial_sp, psp_l, psp_u = cluster_bootstrap_ci(
        df,
        lambda d: compute_partial_spearman(d, "S_t", "V_t", "t"),
        n_resamples=n_resamples,
        seed=seed + 2,
    )

    results["A2_St_vs_Vt"] = {
        "within_bin_spearman": within_bin_spearman,
        "partial_spearman_St_Vt_given_t": {"estimate": partial_sp, "ci_lower": psp_l, "ci_upper": psp_u},
    }

    # =========================================================================
    # A3 regression: Standardized M1 and M2 OLS + logistic regression
    # =========================================================================
    df_std = df.copy()
    pred_cols = ["t", "k", "salience", "confidence", "V_t"]
    for col in pred_cols:
        col_std = df_std[col].std()
        df_std[f"{col}_z"] = (df_std[col] - df_std[col].mean()) / (col_std if col_std > 0 else 1.0)

    # Build dynamic formulas excluding constant covariates
    covars = []
    for c in ["k", "salience", "confidence"]:
        if df_std[f"{c}_z"].std() > 1e-6:
            covars.append(f"{c}_z")
    covar_str = (" + " + " + ".join(covars)) if covars else ""

    formula_m1 = f"S_t ~ t_z{covar_str}"
    formula_m2 = f"S_t ~ t_z + V_t_z{covar_str}"

    # Point estimates for OLS
    try:
        m1_fit = smf.ols(formula_m1, data=df_std).fit(
            cov_type="cluster", cov_kwds={"groups": df_std["image_id"]}
        )
        b_t_m1 = float(m1_fit.params.get("t_z", 0.0))
    except Exception as e:
        logger.warning(f"M1 OLS fit failed: {e}")
        b_t_m1 = 0.0

    try:
        m2_fit = smf.ols(formula_m2, data=df_std).fit(
            cov_type="cluster", cov_kwds={"groups": df_std["image_id"]}
        )
        b_t_m2 = float(m2_fit.params.get("t_z", 0.0))
        b_vt_m2 = float(m2_fit.params.get("V_t_z", 0.0))
    except Exception as e:
        logger.warning(f"M2 OLS fit failed: {e}")
        b_t_m2 = 0.0
        b_vt_m2 = 0.0

    prop_med = (b_t_m1 - b_t_m2) / b_t_m1 if abs(b_t_m1) > 1e-8 else 0.0

    # Cluster-bootstrap CIs for regression
    def calc_m1_bt(d):
        dz = d.copy()
        valid_covs = []
        for c in ["t", "k", "salience", "confidence"]:
            s = dz[c].std()
            if s > 1e-6:
                dz[f"{c}_z"] = (dz[c] - dz[c].mean()) / s
                if c != "t":
                    valid_covs.append(f"{c}_z")
            else:
                dz[f"{c}_z"] = 0.0
        f_str = "S_t ~ t_z" + (" + " + " + ".join(valid_covs) if valid_covs else "")
        fit = smf.ols(f_str, data=dz).fit()
        return float(fit.params.get("t_z", 0.0))

    def calc_m2_bt(d):
        dz = d.copy()
        valid_covs = []
        for c in ["t", "k", "salience", "confidence", "V_t"]:
            s = dz[c].std()
            if s > 1e-6:
                dz[f"{c}_z"] = (dz[c] - dz[c].mean()) / s
                if c not in ["t", "V_t"]:
                    valid_covs.append(f"{c}_z")
            else:
                dz[f"{c}_z"] = 0.0
        f_str = "S_t ~ t_z + V_t_z" + (" + " + " + ".join(valid_covs) if valid_covs else "")
        fit = smf.ols(f_str, data=dz).fit()
        return float(fit.params.get("t_z", 0.0))

    def calc_m2_bvt(d):
        dz = d.copy()
        valid_covs = []
        for c in ["t", "k", "salience", "confidence", "V_t"]:
            s = dz[c].std()
            if s > 1e-6:
                dz[f"{c}_z"] = (dz[c] - dz[c].mean()) / s
                if c not in ["t", "V_t"]:
                    valid_covs.append(f"{c}_z")
            else:
                dz[f"{c}_z"] = 0.0
        f_str = "S_t ~ t_z + V_t_z" + (" + " + " + ".join(valid_covs) if valid_covs else "")
        fit = smf.ols(f_str, data=dz).fit()
        return float(fit.params.get("V_t_z", 0.0))

    def calc_prop_med(d):
        b1 = calc_m1_bt(d)
        b2 = calc_m2_bt(d)
        return (b1 - b2) / b1 if abs(b1) > 1e-8 else 0.0

    b1_est, b1_l, b1_u = cluster_bootstrap_ci(df, calc_m1_bt, n_resamples=n_resamples, seed=seed + 3)
    b2_est, b2_l, b2_u = cluster_bootstrap_ci(df, calc_m2_bt, n_resamples=n_resamples, seed=seed + 4)
    bvt_est, bvt_l, bvt_u = cluster_bootstrap_ci(df, calc_m2_bvt, n_resamples=n_resamples, seed=seed + 5)
    pm_est, pm_l, pm_u = cluster_bootstrap_ci(df, calc_prop_med, n_resamples=n_resamples, seed=seed + 6)

    # Logistic version for flip
    formula_log_m1 = f"flip ~ t_z{covar_str}"
    formula_log_m2 = f"flip ~ t_z + V_t_z{covar_str}"

    try:
        m1_logit = smf.logit(formula_log_m1, data=df_std).fit(
            cov_type="cluster", cov_kwds={"groups": df_std["image_id"]}, disp=False
        )
        b_t_logit_m1 = float(m1_logit.params.get("t_z", 0.0))
    except Exception:
        # Fallback to linear probability model
        m1_logit = smf.ols(formula_log_m1, data=df_std).fit()
        b_t_logit_m1 = float(m1_logit.params.get("t_z", 0.0))

    try:
        m2_logit = smf.logit(formula_log_m2, data=df_std).fit(
            cov_type="cluster", cov_kwds={"groups": df_std["image_id"]}, disp=False
        )
        b_t_logit_m2 = float(m2_logit.params.get("t_z", 0.0))
        b_vt_logit_m2 = float(m2_logit.params.get("V_t_z", 0.0))
    except Exception:
        m2_logit = smf.ols(formula_log_m2, data=df_std).fit()
        b_t_logit_m2 = float(m2_logit.params.get("t_z", 0.0))
        b_vt_logit_m2 = float(m2_logit.params.get("V_t_z", 0.0))

    results["A3_mediation"] = {
        "OLS": {
            "M1_b_t": {"estimate": b1_est, "ci_lower": b1_l, "ci_upper": b1_u},
            "M2_b_t": {"estimate": b2_est, "ci_lower": b2_l, "ci_upper": b2_u},
            "M2_b_Vt": {"estimate": bvt_est, "ci_lower": bvt_l, "ci_upper": bvt_u},
            "proportion_mediated": {"estimate": pm_est, "ci_lower": pm_l, "ci_upper": pm_u},
        },
        "Logistic": {
            "M1_b_t": b_t_logit_m1,
            "M2_b_t": b_t_logit_m2,
            "M2_b_Vt": b_vt_logit_m2,
        },
    }

    # =========================================================================
    # A4: Control for distance k
    # =========================================================================
    spearman_k, sk_l, sk_u = cluster_bootstrap_ci(
        df,
        lambda d: compute_spearman(d, "S_t", "k"),
        n_resamples=n_resamples,
        seed=seed + 7,
    )
    binned_k = {}
    for k_val, group in df.groupby("k"):
        m_st, st_l, st_u = cluster_bootstrap_ci(
            group,
            lambda d: float(d["S_t"].mean()),
            n_resamples=n_resamples,
            seed=seed,
        )
        m_flip, fl_l, fl_u = cluster_bootstrap_ci(
            group,
            lambda d: float(d["flip"].mean()),
            n_resamples=n_resamples,
            seed=seed,
        )
        binned_k[str(k_val)] = {
            "mean_St": m_st, "st_ci_lower": st_l, "st_ci_upper": st_u,
            "flip_rate": m_flip, "flip_ci_lower": fl_l, "flip_ci_upper": fl_u,
            "n": len(group),
        }

    results["A4_distance"] = {
        "spearman_St_k": {"estimate": spearman_k, "ci_lower": sk_l, "ci_upper": sk_u},
        "by_distance_k": binned_k,
    }

    # =========================================================================
    # A5 robustness: repeat A2 with black-image mode V_t
    # =========================================================================
    if df_black is not None and len(df_black) > 0:
        within_bin_black = {}
        for b_label, group in df_black.groupby("pos_bin", observed=True):
            if len(group) >= 5:
                sp, sp_l, sp_u = cluster_bootstrap_ci(
                    group,
                    lambda d: compute_spearman(d, "S_t", "V_t"),
                    n_resamples=n_resamples,
                    seed=seed,
                )
                within_bin_black[str(b_label)] = {"spearman": sp, "ci_lower": sp_l, "ci_upper": sp_u, "n": len(group)}

        psp_black, psp_bl_l, psp_bl_u = cluster_bootstrap_ci(
            df_black,
            lambda d: compute_partial_spearman(d, "S_t", "V_t", "t"),
            n_resamples=n_resamples,
            seed=seed + 8,
        )
        results["A5_robustness_black"] = {
            "within_bin_spearman": within_bin_black,
            "partial_spearman_St_Vt_given_t": {"estimate": psp_black, "ci_lower": psp_bl_l, "ci_upper": psp_bl_u},
        }

    return results


def generate_markdown_report(analysis_results: Dict[str, Any], output_path: str):
    """Writes human-readable markdown report summarizing all statistical analyses."""
    md = []
    md.append("# Kết quả Phân Tích Thực Nghiệm Pilot H2.1 (VLM)")
    md.append("> **Ghi chú quan trọng:** Tất cả kết quả dưới đây đều mang tính **exploratory (n=20 images)**. Không khẳng định ý nghĩa thống kê suy diễn tổng quát.\n")

    meta = analysis_results.get("metadata", {})
    md.append(f"- **Số lượng ảnh phân tích:** {meta.get('n_images', 20)}")
    md.append(f"- **Tổng số bản ghi perturbation:** {meta.get('n_records', 0)}")
    md.append(f"- **Phương pháp khoảng tin cậy:** Cluster bootstrap theo ảnh ({meta.get('n_bootstrap', 2000)} resamples, 95% CI)\n")

    # A0
    a0 = analysis_results.get("A0_premise", {})
    sp0 = a0.get("spearman_Vt_t", {})
    md.append("## A0. Tiền đề: Đóng góp của ảnh (V_t) theo vị trí (t)")
    md.append(f"- **Spearman(V_t, t):** `{sp0.get('estimate', 0.0):.4f}` [95% CI: `{sp0.get('ci_lower', 0.0):.4f}`, `{sp0.get('ci_upper', 0.0):.4f}`]")
    md.append("- **Giá trị trung bình V_t theo khoảng vị trí:**")
    md.append("| Khoảng vị trí (t) | Trung bình V_t | 95% CI | Số bản ghi (n) |")
    md.append("| :--- | :--- | :--- | :--- |")
    for b, stats_dict in a0.get("binned_Vt", {}).items():
        md.append(f"| {b} | {stats_dict['mean']:.4f} | [{stats_dict['ci_lower']:.4f}, {stats_dict['ci_upper']:.4f}] | {stats_dict['n']} |")
    md.append("")

    # A1
    a1 = analysis_results.get("A1_sensitivity_vs_position", {})
    sp1 = a1.get("spearman_St_t", {})
    md.append("## A1. Độ nhạy tiền tố (S_t) và Tỷ lệ Flip theo vị trí (t)")
    md.append(f"- **Spearman(S_t, t):** `{sp1.get('estimate', 0.0):.4f}` [95% CI: `{sp1.get('ci_lower', 0.0):.4f}`, `{sp1.get('ci_upper', 0.0):.4f}`]")
    md.append("- **Phân bố theo khoảng vị trí:**")
    md.append("| Khoảng vị trí (t) | Trung bình S_t | 95% CI S_t | Tỷ lệ Flip | 95% CI Flip | n |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for b, stats_dict in a1.get("binned_St_and_flip", {}).items():
        md.append(
            f"| {b} | {stats_dict['mean_St']:.4f} | [{stats_dict['st_ci_lower']:.4f}, {stats_dict['st_ci_upper']:.4f}] | "
            f"{stats_dict['flip_rate']:.4f} | [{stats_dict['flip_ci_lower']:.4f}, {stats_dict['flip_ci_upper']:.4f}] | {stats_dict['n']} |"
        )
    md.append("")

    # A2
    a2 = analysis_results.get("A2_St_vs_Vt", {})
    psp = a2.get("partial_spearman_St_Vt_given_t", {})
    md.append("## A2. Mối quan hệ giữa S_t và V_t (Kiểm định P2)")
    md.append(f"- **Partial Spearman(S_t, V_t | t):** `{psp.get('estimate', 0.0):.4f}` [95% CI: `{psp.get('ci_lower', 0.0):.4f}`, `{psp.get('ci_upper', 0.0):.4f}`]")
    md.append("- **Spearman(S_t, V_t) trong từng khoảng vị trí:**")
    md.append("| Khoảng vị trí | Spearman(S_t, V_t) | 95% CI | n |")
    md.append("| :--- | :--- | :--- | :--- |")
    for b, stats_dict in a2.get("within_bin_spearman", {}).items():
        md.append(f"| {b} | {stats_dict['spearman']:.4f} | [{stats_dict['ci_lower']:.4f}, {stats_dict['ci_upper']:.4f}] | {stats_dict['n']} |")
    md.append("")

    # A3
    a3 = analysis_results.get("A3_mediation", {}).get("OLS", {})
    md.append("## A3. Hồi quy Tuyến tính và Phân tích Trung gian (Mediation)")
    md.append("- **Mô hình M1:** `S_t ~ t + k + salience + confidence`")
    md.append(f"  - Hệ số chuẩn hóa $b_t(M1)$: `{a3.get('M1_b_t', {}).get('estimate', 0.0):.4f}` [95% CI: `{a3.get('M1_b_t', {}).get('ci_lower', 0.0):.4f}`, `{a3.get('M1_b_t', {}).get('ci_upper', 0.0):.4f}`]")
    md.append("- **Mô hình M2:** `S_t ~ t + V_t + k + salience + confidence`")
    md.append(f"  - Hệ số chuẩn hóa $b_t(M2)$: `{a3.get('M2_b_t', {}).get('estimate', 0.0):.4f}` [95% CI: `{a3.get('M2_b_t', {}).get('ci_lower', 0.0):.4f}`, `{a3.get('M2_b_t', {}).get('ci_upper', 0.0):.4f}`]")
    md.append(f"  - Hệ số chuẩn hóa $b_{{V_t}}(M2)$: `{a3.get('M2_b_Vt', {}).get('estimate', 0.0):.4f}` [95% CI: `{a3.get('M2_b_Vt', {}).get('ci_lower', 0.0):.4f}`, `{a3.get('M2_b_Vt', {}).get('ci_upper', 0.0):.4f}`]")
    md.append(f"- **Tỷ lệ trung gian (Proportion Mediated):** `{a3.get('proportion_mediated', {}).get('estimate', 0.0):.4f}` [95% CI: `{a3.get('proportion_mediated', {}).get('ci_lower', 0.0):.4f}`, `{a3.get('proportion_mediated', {}).get('ci_upper', 0.0):.4f}`]")
    md.append("")

    # A4
    a4 = analysis_results.get("A4_distance", {})
    spk = a4.get("spearman_St_k", {})
    md.append("## A4. Kiểm soát Khoảng cách can thiệp (k = t - s)")
    md.append(f"- **Spearman(S_t, k):** `{spk.get('estimate', 0.0):.4f}` [95% CI: `{spk.get('ci_lower', 0.0):.4f}`, `{spk.get('ci_upper', 0.0):.4f}`]")
    md.append("")

    # A5
    if "A5_robustness_black" in analysis_results:
        a5 = analysis_results["A5_robustness_black"]
        pspb = a5.get("partial_spearman_St_Vt_given_t", {})
        md.append("## A5. Kiểm định Độ vững với Black-image mode")
        md.append(f"- **Partial Spearman(S_t, V_t(black) | t):** `{pspb.get('estimate', 0.0):.4f}` [95% CI: `{pspb.get('ci_lower', 0.0):.4f}`, `{pspb.get('ci_upper', 0.0):.4f}`]")
        md.append("")

    # Interpretation
    md.append("## 7. Hướng dẫn Diễn giải Kết quả (Interpretation Guide)")
    md.append("- **P1 (S_t và Flip tăng theo vị trí):** Được ủng hộ nếu Spearman(S_t, t) > 0 và flip rate tăng dần theo các bin vị trí. Bị bác bỏ nếu đi ngang hoặc giảm.")
    md.append("- **P2 (Cùng vị trí, V_t thấp thì S_t cao):** Được ủng hộ nếu correlation giữa S_t và V_t trong từng bin âm, và hệ số b_{V_t}(M2) mang giá trị âm rõ rệt.")
    md.append("- **Hiệu ứng Trung gian (Mediation):** Được gợi ý nếu hệ số của t thu nhỏ đáng kể từ M1 sang M2. Nếu không đổi, V_t không đóng vai trò trung gian và vị trí tác động qua kênh khác (tự tương quan ngôn ngữ, độ tin cậy, cú pháp).")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    logger.info(f"Saved analysis markdown report to {output_path}")
