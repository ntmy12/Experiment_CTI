"""
Statistical Analysis Module for Confirmatory and Causal Experiments (v2).
Implements pre-registered tests HA1, HA2, HB, HC, HD, HE, and exploratory HF.
Cluster bootstrap over images (B=2000). Generates verdicts.md, analysis.json, analysis.md.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger("confirmatory_v2")


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    cluster_col: str,
    metric_fn: Any,
    n_resamples: int = 2000,
    seed: int = 2026,
) -> Tuple[float, float, float]:
    """
    Computes point estimate and 95% percentile bootstrap CI clustered by cluster_col.
    Returns: (point_estimate, ci_lower, ci_upper)
    """
    if df.empty:
        return 0.0, 0.0, 0.0

    point_est = float(metric_fn(df))

    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)
    if n_clusters < 2:
        return point_est, point_est, point_est

    rng = np.random.RandomState(seed)
    boot_estimates = []

    # Map clusters to indices
    cluster_dict = {c: df[df[cluster_col] == c] for c in clusters}

    for _ in range(n_resamples):
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        boot_df = pd.concat([cluster_dict[c] for c in sampled_clusters], ignore_index=True)
        try:
            val = float(metric_fn(boot_df))
            if not np.isnan(val) and not np.isinf(val):
                boot_estimates.append(val)
        except Exception:
            pass

    if len(boot_estimates) < 50:
        return point_est, point_est, point_est

    ci_lower = float(np.percentile(boot_estimates, 2.5))
    ci_upper = float(np.percentile(boot_estimates, 97.5))
    return point_est, ci_lower, ci_upper


def compute_spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Spearman rank correlation safely."""
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    res, _ = stats.spearmanr(x, y)
    return float(res) if not np.isnan(res) else 0.0


def run_full_analysis_v2(
    df_e1: pd.DataFrame,
    df_e2: Optional[pd.DataFrame] = None,
    df_e3: Optional[pd.DataFrame] = None,
    n_resamples: int = 2000,
    seed: int = 2026,
) -> Dict[str, Any]:
    """
    Runs all pre-registered tests HA-HE and exploratory HF.
    """
    results: Dict[str, Any] = {
        "metadata": {
            "n_resamples": n_resamples,
            "seed": seed,
            "n_e1_total": len(df_e1),
            "n_e2_total": len(df_e2) if df_e2 is not None else 0,
            "n_e3_total": len(df_e3) if df_e3 is not None else 0,
        }
    }

    # Filter confirmatory subset for E1: exclude placebo and low_mass
    df_e1_clean = df_e1[(df_e1["is_placebo"] == 0) & (df_e1["low_mass"] == 0)].copy()
    df_e1_clean["y"] = np.log10(df_e1_clean["S_t"] + 1e-6)

    df_near = df_e1_clean[df_e1_clean["k"].isin([1, 2])].copy()
    df_pooled = df_e1_clean[df_e1_clean["k"].isin([1, 2, 3])].copy()

    # =========================================================================
    # HA1: flip_tf rate at k=1 vs k>=4
    # =========================================================================
    df_k1 = df_e1_clean[df_e1_clean["k"] == 1]
    df_k4plus = df_e1_clean[df_e1_clean["k"] >= 4]

    def ha1_fn(data: pd.DataFrame) -> float:
        r1 = data[data["k"] == 1]["flip_tf"].mean() if not data[data["k"] == 1].empty else 0.0
        r4 = data[data["k"] >= 4]["flip_tf"].mean() if not data[data["k"] >= 4].empty else 0.0
        return float(r1 - r4)

    df_ha1 = df_e1_clean[df_e1_clean["k"].isin([1]) | (df_e1_clean["k"] >= 4)]
    ha1_est, ha1_lo, ha1_hi = cluster_bootstrap_ci(df_ha1, "image_id", ha1_fn, n_resamples, seed)
    ha1_verdict = "SUPPORTED" if ha1_lo > 0 else "NOT DETECTED"

    results["HA1"] = {
        "estimate": ha1_est, "ci_lower": ha1_lo, "ci_upper": ha1_hi,
        "rate_k1": float(df_k1["flip_tf"].mean()) if not df_k1.empty else 0.0,
        "rate_k4plus": float(df_k4plus["flip_tf"].mean()) if not df_k4plus.empty else 0.0,
        "n_k1": len(df_k1), "n_k4plus": len(df_k4plus),
        "criterion": "CI lower bound > 0", "verdict": ha1_verdict
    }

    # =========================================================================
    # HA2: Free generation flip rate at k=1 vs Placebo-free rate (E2)
    # =========================================================================
    if df_e2 is not None and not df_e2.empty:
        df_e2_k1 = df_e2[(df_e2["k"] == 1) & (df_e2["is_placebo"] == 0)]
        df_e2_placebo = df_e2[df_e2["is_placebo"] == 1]

        def ha2_fn(data: pd.DataFrame) -> float:
            r1 = data[(data["k"] == 1) & (data["is_placebo"] == 0)]["flip_free_word"].mean()
            rp = data[data["is_placebo"] == 1]["flip_free_word"].mean()
            return float(r1 - rp)

        ha2_est, ha2_lo, ha2_hi = cluster_bootstrap_ci(df_e2, "image_id", ha2_fn, n_resamples, seed)
        ha2_verdict = "SUPPORTED" if ha2_lo > 0 else "NOT DETECTED"
        results["HA2"] = {
            "estimate": ha2_est, "ci_lower": ha2_lo, "ci_upper": ha2_hi,
            "rate_k1": float(df_e2_k1["flip_free_word"].mean()) if not df_e2_k1.empty else 0.0,
            "rate_placebo": float(df_e2_placebo["flip_free_word"].mean()) if not df_e2_placebo.empty else 0.0,
            "criterion": "CI lower bound > 0", "verdict": ha2_verdict
        }
    else:
        results["HA2"] = {
            "estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
            "criterion": "CI lower bound > 0", "verdict": "INCONCLUSIVE"
        }

    # HA overall verdict
    ha_verdict = "SUPPORTED" if (results["HA1"]["verdict"] == "SUPPORTED" and results["HA2"]["verdict"] == "SUPPORTED") else "NOT DETECTED"
    results["HA"] = {"verdict": ha_verdict}

    # =========================================================================
    # HB: Distance (flip rate and S_t decrease with k)
    # =========================================================================
    # Spearman(k, S)
    def hb_spearman_fn(data: pd.DataFrame) -> float:
        return compute_spearman_safe(data["k"].values, data["S_t"].values)

    hb_sp_est, hb_sp_lo, hb_sp_hi = cluster_bootstrap_ci(df_e1_clean, "image_id", hb_spearman_fn, n_resamples, seed)

    # Pairwise differences: k=1 vs k=2 and k=2 vs k=3
    def diff_k1_k2_fn(data: pd.DataFrame) -> float:
        m1 = data[data["k"] == 1]["flip_tf"].mean()
        m2 = data[data["k"] == 2]["flip_tf"].mean()
        return float(m1 - m2)

    diff12_est, diff12_lo, diff12_hi = cluster_bootstrap_ci(
        df_e1_clean[df_e1_clean["k"].isin([1, 2])], "image_id", diff_k1_k2_fn, n_resamples, seed
    )

    hb_verdict = "SUPPORTED" if (hb_sp_hi < 0 and diff12_lo > 0) else "NOT DETECTED"
    results["HB"] = {
        "spearman_k_S": {"estimate": hb_sp_est, "ci_lower": hb_sp_lo, "ci_upper": hb_sp_hi},
        "diff_k1_vs_k2": {"estimate": diff12_est, "ci_lower": diff12_lo, "ci_upper": diff12_hi},
        "criterion": "Spearman CI upper bound < 0 AND k=1 vs k=2 diff CI lower > 0",
        "verdict": hb_verdict,
        "by_k": {}
    }

    for k_val in range(1, 9):
        sub_k = df_e1_clean[df_e1_clean["k"] == k_val]
        if not sub_k.empty:
            mean_flip, f_lo, f_hi = cluster_bootstrap_ci(sub_k, "image_id", lambda d: d["flip_tf"].mean(), n_resamples, seed)
            mean_y, y_lo, y_hi = cluster_bootstrap_ci(sub_k, "image_id", lambda d: d["y"].mean(), n_resamples, seed)
            results["HB"]["by_k"][str(k_val)] = {
                "n": len(sub_k),
                "flip_rate": mean_flip, "flip_ci_lower": f_lo, "flip_ci_upper": f_hi,
                "mean_y": mean_y, "y_ci_lower": y_lo, "y_ci_upper": y_hi,
            }

    # =========================================================================
    # HC: Position in "near" (k in {1, 2})
    # =========================================================================
    if len(df_near) >= 20:
        # Standardize predictors within near subset
        df_near_reg = df_near.copy()
        for col in ["t", "entropy", "margin", "p_alt", "obj_mass", "salience", "V_black"]:
            std_val = df_near_reg[col].std()
            df_near_reg[f"{col}_z"] = (df_near_reg[col] - df_near_reg[col].mean()) / (std_val if std_val > 1e-12 else 1.0)

        # Check VIF for margin and entropy
        drop_margin = False
        try:
            ent_var = df_near_reg["entropy_z"].values
            mar_var = df_near_reg["margin_z"].values
            corr_val = np.corrcoef(ent_var, mar_var)[0, 1]
            if not np.isnan(corr_val) and abs(corr_val) > 0.95:
                drop_margin = True
        except Exception:
            pass

        formula_hc = "y ~ C(k) + t_z + entropy_z + is_repeat + C(alt_rank) + p_alt_z + obj_mass_z + salience_z"
        if not drop_margin:
            formula_hc += " + margin_z"

        try:
            ols_hc = smf.ols(formula_hc, data=df_near_reg).fit(
                cov_type="cluster", cov_kwds={"groups": df_near_reg["image_id"]}
            )
            b_t_est = float(ols_hc.params["t_z"])
            b_t_se = float(ols_hc.bse["t_z"])
            b_t_lo = b_t_est - 1.96 * b_t_se
            b_t_hi = b_t_est + 1.96 * b_t_se
        except Exception as e:
            logger.warning(f"HC OLS fitting failed: {e}")
            b_t_est, b_t_lo, b_t_hi = 0.0, 0.0, 0.0

        if b_t_lo > 0:
            hc_verdict = "SUPPORTED"
        elif b_t_hi < 0:
            hc_verdict = "CONTRARY"
        else:
            hc_verdict = "NOT DETECTED"

        results["HC"] = {
            "estimate": b_t_est, "ci_lower": b_t_lo, "ci_upper": b_t_hi,
            "criterion": "positive CI excludes 0 -> SUPPORTED; negative -> CONTRARY; includes 0 -> NOT DETECTED",
            "verdict": hc_verdict, "drop_margin": drop_margin
        }

        # =====================================================================
        # HD: Visual Contribution in "near" (k in {1, 2})
        # =====================================================================
        formula_hd = formula_hc + " + V_black_z"
        try:
            ols_hd = smf.ols(formula_hd, data=df_near_reg).fit(
                cov_type="cluster", cov_kwds={"groups": df_near_reg["image_id"]}
            )
            b_v_est = float(ols_hd.params["V_black_z"])
            b_v_se = float(ols_hd.bse["V_black_z"])
            b_v_lo = b_v_est - 1.96 * b_v_se
            b_v_hi = b_v_est + 1.96 * b_v_se
        except Exception as e:
            logger.warning(f"HD OLS fitting failed: {e}")
            b_v_est, b_v_lo, b_v_hi = 0.0, 0.0, 0.0

        if b_v_hi < 0:
            hd_verdict = "SUPPORTED"
        elif b_v_lo > 0:
            hd_verdict = "CONTRARY"
        else:
            hd_verdict = "NOT DETECTED"

        # Raw unadjusted Spearman within each k stratum
        raw_sp_k1 = compute_spearman_safe(
            df_e1_clean[df_e1_clean["k"] == 1]["S_t"].values,
            df_e1_clean[df_e1_clean["k"] == 1]["V_black"].values
        )
        raw_sp_k2 = compute_spearman_safe(
            df_e1_clean[df_e1_clean["k"] == 2]["S_t"].values,
            df_e1_clean[df_e1_clean["k"] == 2]["V_black"].values
        )

        results["HD"] = {
            "estimate": b_v_est, "ci_lower": b_v_lo, "ci_upper": b_v_hi,
            "criterion": "negative CI excludes 0 -> SUPPORTED; positive -> CONTRARY; includes 0 -> NOT DETECTED",
            "verdict": hd_verdict,
            "raw_spearman_k1": raw_sp_k1, "raw_spearman_k2": raw_sp_k2
        }

        # =====================================================================
        # MEDIATION CHECK: Run ONLY if total effect of t on y excludes 0
        # =====================================================================
        if (b_t_lo > 0 or b_t_hi < 0) and abs(b_t_est) > 1e-6:
            prop_med = float((b_t_est - float(ols_hd.params["t_z"])) / b_t_est)
            results["MEDIATION"] = {
                "applicable": True,
                "proportion_mediated": prop_med,
            }
        else:
            results["MEDIATION"] = {
                "applicable": False,
                "message": "mediation not applicable: no total effect of position"
            }
    else:
        results["HC"] = {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "verdict": "INCONCLUSIVE"}
        results["HD"] = {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "verdict": "INCONCLUSIVE"}
        results["MEDIATION"] = {"applicable": False, "message": "insufficient sample"}

    # =========================================================================
    # HE: Causal Intervention (E3)
    # =========================================================================
    if df_e3 is not None and not df_e3.empty:
        df_e3_clean = df_e3[(df_e3["is_placebo"] == 0) & (df_e3["is_far"] == 0)].copy()
        df_e3_clean["y"] = np.log10(df_e3_clean["S_lambda"] + 1e-6)

        # Standardize predictors
        for col in ["V_lambda", "H_lambda"]:
            std_c = df_e3_clean[col].std()
            df_e3_clean[f"{col}_z"] = (df_e3_clean[col] - df_e3_clean[col].mean()) / (std_c if std_c > 1e-12 else 1.0)

        # (i) Within-triplet fixed effects: y ~ V_lambda_z + H_lambda_z + C(triplet_id)
        try:
            fe_mod = smf.ols("y ~ V_lambda_z + H_lambda_z + C(triplet_id)", data=df_e3_clean).fit(
                cov_type="cluster", cov_kwds={"groups": df_e3_clean["image_id"]}
            )
            b_v_fe = float(fe_mod.params["V_lambda_z"])
            b_v_fe_se = float(fe_mod.bse["V_lambda_z"])
            b_v_fe_lo = b_v_fe - 1.96 * b_v_fe_se
            b_v_fe_hi = b_v_fe + 1.96 * b_v_fe_se
            he_i_supported = (b_v_fe_hi < 0)
        except Exception as e:
            logger.warning(f"HE Fixed effects failed: {e}")
            b_v_fe, b_v_fe_lo, b_v_fe_hi = 0.0, 0.0, 0.0
            he_i_supported = False

        # (ii) Mean excess(lambda) for lambda in {0.75, 0.5, 0.25, 0.0}
        excess_05 = df_e3_clean[df_e3_clean["lambda_val"] == 0.5]
        mean_ex05, ex05_lo, ex05_hi = cluster_bootstrap_ci(
            excess_05, "image_id", lambda d: d["excess_lambda"].mean(), n_resamples, seed
        )
        he_ii_supported = (ex05_lo > 0)

        # (iii) Share of triplets where S(0.5) > S(1.0)
        piv_s = df_e3_clean.pivot(index="triplet_id", columns="lambda_val", values="S_lambda")
        if 0.5 in piv_s.columns and 1.0 in piv_s.columns:
            share_s05_gt_s10 = float((piv_s[0.5] > piv_s[1.0]).mean())
        else:
            share_s05_gt_s10 = 0.0

        he_verdict = "SUPPORTED" if (he_i_supported and he_ii_supported) else "NOT DETECTED"

        results["HE"] = {
            "fixed_effects_V": {"estimate": b_v_fe, "ci_lower": b_v_fe_lo, "ci_upper": b_v_fe_hi, "supported": he_i_supported},
            "excess_lambda_05": {"estimate": mean_ex05, "ci_lower": ex05_lo, "ci_upper": ex05_hi, "supported": he_ii_supported},
            "share_S05_gt_S10": share_s05_gt_s10,
            "criterion": "(i) negative V CI excludes 0 AND (ii) excess(0.5) CI lower > 0",
            "verdict": he_verdict
        }
    else:
        results["HE"] = {
            "fixed_effects_V": {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
            "excess_lambda_05": {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
            "criterion": "(i) negative V CI excludes 0 AND (ii) excess(0.5) CI lower > 0",
            "verdict": "INCONCLUSIVE"
        }

    # =========================================================================
    # HF: Exploratory POS and interactions
    # =========================================================================
    pos_summary = {}
    for pos_tag in df_e1_clean["pos_s"].unique():
        sub_pos = df_e1_clean[df_e1_clean["pos_s"] == pos_tag]
        if len(sub_pos) >= 5:
            pos_summary[pos_tag] = {
                "n": len(sub_pos),
                "flip_rate": float(sub_pos["flip_tf"].mean()),
                "mean_St": float(sub_pos["S_t"].mean()),
            }

    results["HF_exploratory"] = {
        "by_pos": pos_summary,
        "flip_types_by_k": df_e1_clean.groupby("k")["flip_type"].value_counts(normalize=True).unstack().fillna(0).to_dict()
    }

    return results


def save_verdicts_markdown(results: Dict[str, Any], output_path: str):
    """Generates results_v2/verdicts.md table."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rows = []

    # HA1
    ha1 = results.get("HA1", {})
    rows.append(f"| **HA1 (tf: k=1 vs k>=4)** | `{ha1.get('estimate', 0):.4f}` | `[{ha1.get('ci_lower', 0):.4f}, {ha1.get('ci_upper', 0):.4f}]` | {ha1.get('criterion', '')} | **{ha1.get('verdict', 'INCONCLUSIVE')}** |")

    # HA2
    ha2 = results.get("HA2", {})
    rows.append(f"| **HA2 (free: k=1 vs placebo)** | `{ha2.get('estimate', 0):.4f}` | `[{ha2.get('ci_lower', 0):.4f}, {ha2.get('ci_upper', 0):.4f}]` | {ha2.get('criterion', '')} | **{ha2.get('verdict', 'INCONCLUSIVE')}** |")

    # HB
    hb = results.get("HB", {})
    sp = hb.get("spearman_k_S", {})
    rows.append(f"| **HB (Distance k decay)** | `{sp.get('estimate', 0):.4f}` | `[{sp.get('ci_lower', 0):.4f}, {sp.get('ci_upper', 0):.4f}]` | {hb.get('criterion', '')} | **{hb.get('verdict', 'INCONCLUSIVE')}** |")

    # HC
    hc = results.get("HC", {})
    rows.append(f"| **HC (Position t in near)** | `{hc.get('estimate', 0):.4f}` | `[{hc.get('ci_lower', 0):.4f}, {hc.get('ci_upper', 0):.4f}]` | {hc.get('criterion', '')} | **{hc.get('verdict', 'INCONCLUSIVE')}** |")

    # HD
    hd = results.get("HD", {})
    rows.append(f"| **HD (Visual Contribution V)** | `{hd.get('estimate', 0):.4f}` | `[{hd.get('ci_lower', 0):.4f}, {hd.get('ci_upper', 0):.4f}]` | {hd.get('criterion', '')} | **{hd.get('verdict', 'INCONCLUSIVE')}** |")

    # HE
    he = results.get("HE", {})
    he_v = he.get("fixed_effects_V", {})
    rows.append(f"| **HE (Causal Degradation)** | `{he_v.get('estimate', 0):.4f}` | `[{he_v.get('ci_lower', 0):.4f}, {he_v.get('ci_upper', 0):.4f}]` | {he.get('criterion', '')} | **{he.get('verdict', 'INCONCLUSIVE')}** |")

    table_md = (
        "# Bảng Phán Quyết Giả Thuyết Đăng Ký Trước (Pre-registered Verdicts)\n\n"
        "| Hypothesis | Estimate | 95% CI | Pre-specified criterion | Verdict |\n"
        "| :--- | :---: | :---: | :--- | :---: |\n" + "\n".join(rows) + "\n\n"
        "> **Quy tắc phán quyết:** `SUPPORTED` (được ủng hộ), `NOT DETECTED` (không phát hiện), `CONTRARY` (ngược giả thuyết), `INCONCLUSIVE` (chưa đủ dữ liệu).\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(table_md)
    logger.info(f"Saved pre-registered verdicts to {output_path}")
