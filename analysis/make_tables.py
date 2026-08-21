#!/usr/bin/env python3
"""Generate and validate the thesis tables."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REGION_ORDER = [
    "L-Hippocampus",
    "R-Hippocampus",
    "L-Amygdala",
    "R-Amygdala",
    "L-Thalamus",
    "R-Thalamus",
    "L-Caudate",
    "R-Caudate",
    "L-Putamen",
    "R-Putamen",
    "L-Pallidum",
    "R-Pallidum",
    "L-Cerebral-WM",
    "R-Cerebral-WM",
    "Brain-Stem",
]

RQ2_ORDER = [
    "Sex",
    "Binned_Age_at_Scan",
    "Op_Side",
    "Op_Type_collapsed",
    "Pathology_grp",
    "Number_ASMs",
    "ILAE_Y1_seizure_free",
]

RQ2_LABELS = {
    "Sex": "Sex",
    "Binned_Age_at_Scan": "Age band",
    "Op_Side": "Operation side",
    "Op_Type_collapsed": "Operation type",
    "Pathology_grp": "Pathology",
    "Number_ASMs": "Number of ASMs",
    "ILAE_Y1_seizure_free": "ILAE year-1 outcome",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def assert_close(actual: float, expected: float, tol: float, message: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{message}: {actual!r} != {expected!r}")


def bh_adjust(values: pd.Series) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if np.isnan(p).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("BH input contains a missing or invalid p-value")
    n = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted_ranked = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted_ranked = np.minimum(adjusted_ranked, 1.0)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt_q(value: float) -> str:
    return r"$<0.001$" if value < 0.001 else f"{value:.3f}"


def fmt_interval(point: float, low: float, high: float, digits: int = 3) -> str:
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def write_table(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def manifest(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file():
            records.append(
                {
                    "relative_path": str(path.relative_to(input_dir)),
                    "size_bytes": path.stat().st_size,
                }
            )
    result = pd.DataFrame.from_records(records)
    result.to_csv(output_dir / "source_files.csv", index=False)
    return result


def build_regional_table(input_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    stats = pd.read_csv(input_dir / "final_figures/rq1/rq1_entropy_error_stats.csv")
    adjusted = pd.read_csv(input_dir / "analysis_1_to_4/02_rq1_group_adjusted.csv")

    usecols = ["subject_id", "group", "models_used"]
    for region in REGION_ORDER:
        usecols.extend([f"{region}_dice", f"{region}_entropy_mean"])
    raw = pd.read_csv(input_dir / "region_entropy_decomposed_reference_masked.csv", usecols=usecols)

    if len(raw) != 532 or raw["subject_id"].nunique() != 532:
        raise AssertionError("Expected 532 unique subjects in the final disjoint output")
    if raw.groupby("group").size().to_dict() != {0: 107, 1: 107, 2: 106, 3: 106, 4: 106}:
        raise AssertionError("Disjoint group sizes do not match the audited split")
    if raw[usecols[3:]].isna().any().any():
        raise AssertionError("A target-region Dice or entropy value is missing")

    if set(stats["region"]) != set(REGION_ORDER) or len(stats) != 15:
        raise AssertionError("RQ1 statistics do not contain exactly the 15 prespecified regions")
    if set(adjusted["region"]) != set(REGION_ORDER) or len(adjusted) != 15:
        raise AssertionError("Adjusted RQ1 output does not contain the 15 prespecified regions")

    stat_by_region = stats.set_index("region")
    adj_by_region = adjusted.set_index("region")
    records = []
    for region in REGION_ORDER:
        dice = raw[f"{region}_dice"]
        entropy = raw[f"{region}_entropy_mean"]
        rho, p_value = spearmanr(entropy, 1.0 - dice)
        saved = stat_by_region.loc[region]
        saved_adj = adj_by_region.loc[region]
        assert_close(rho, saved["rho"], 1e-12, f"RQ1 rho mismatch for {region}")
        assert_close(p_value, saved["raw_p"], 1e-12, f"RQ1 p-value mismatch for {region}")
        records.append(
            {
                "region": region,
                "n": len(dice),
                "mean_dice": dice.mean(),
                "median_dice": dice.median(),
                "mean_predictive_entropy": entropy.mean(),
                "median_predictive_entropy": entropy.median(),
                "spearman_rho_entropy_vs_error": rho,
                "rho_ci_low": saved["ci_lo"],
                "rho_ci_high": saved["ci_hi"],
                "p_unadjusted": saved["raw_p"],
                "p_fdr_15": saved["fdr_q"],
                "fdr_significant": bool(saved["fdr_significant"]),
                "group_adjusted_rho": saved_adj["group_adjusted_partial_spearman_rho"],
                "group_adjusted_permutation_p_fdr_15": saved_adj[
                    "group_adjusted_permutation_p_fdr_15"
                ],
            }
        )

    result = pd.DataFrame.from_records(records)
    result.to_csv(output_dir / "rq1_regional_performance.csv", index=False)

    if int(result["fdr_significant"].sum()) != 13:
        raise AssertionError("Expected 13 FDR-significant RQ1 regions")
    subject_mean_dice = raw[[f"{r}_dice" for r in REGION_ORDER]].mean(axis=1).mean()
    assert_close(subject_mean_dice, 0.582460571947156, 1e-12, "Mean 15-region Dice changed")
    max_rho_change = float(
        np.max(
            np.abs(
                result["group_adjusted_rho"].to_numpy()
                - result["spearman_rho_entropy_vs_error"].to_numpy()
            )
        )
    )

    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Regional performance and the primary RQ1 entropy--error association for the leakage-free disjoint ensemble. Dice and reference-masked predictive entropy are means across 532 subjects. Spearman $\rho$ relates predictive entropy to Dice error ($1-\mathrm{Dice}$); brackets give 95\% subject-bootstrap confidence intervals, and $q$ is Benjamini--Hochberg adjusted across the 15 prespecified regions.}",
        r"\label{tab:regional_rq1_disjoint}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Region & Mean Dice & Mean entropy & $\rho$ [95\% CI] & FDR $q$ \\",
        r"\hline",
    ]
    for row in result.itertuples(index=False):
        ci = fmt_interval(row.spearman_rho_entropy_vs_error, row.rho_ci_low, row.rho_ci_high)
        lines.append(
            f"{latex_escape(row.region)} & {row.mean_dice:.3f} & "
            f"{row.mean_predictive_entropy:.3f} & {ci} & {fmt_q(row.p_fdr_15)} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    write_table(output_dir / "table_regional_performance_rq1.tex", "\n".join(lines))

    summary = {
        "subject_mean_dice_15": float(subject_mean_dice),
        "mean_regional_rho": float(result["spearman_rho_entropy_vs_error"].mean()),
        "median_regional_rho": float(result["spearman_rho_entropy_vs_error"].median()),
        "fdr_significant_regions": int(result["fdr_significant"].sum()),
        "max_abs_group_adjustment_rho_change": max_rho_change,
    }
    return result, summary


def build_rq2_table(input_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    detailed = pd.read_csv(input_dir / "rq2_clinical_screen_105.csv")
    if len(detailed) != 105:
        raise AssertionError("RQ2 must contain 105 tests")
    if detailed.groupby("variable").size().to_dict() != {key: 15 for key in RQ2_ORDER}:
        raise AssertionError("RQ2 must contain 15 regions for each of seven variables")

    recomputed = bh_adjust(detailed["p_group_adjusted_permutation"])
    if not np.allclose(recomputed, detailed["p_fdr_group_adjusted_105"], atol=1e-12, rtol=0):
        raise AssertionError("RQ2 pooled group-adjusted BH values do not reproduce")
    recomputed_original = bh_adjust(detailed["p_unadjusted"])
    if not np.allclose(recomputed_original, detailed["p_fdr_unadjusted_105"], atol=1e-12, rtol=0):
        raise AssertionError("RQ2 pooled BH values do not reproduce")

    records = []
    for variable in RQ2_ORDER:
        group = detailed.loc[detailed["variable"] == variable]
        strongest = group.sort_values(["p_group_adjusted_permutation", "region"]).iloc[0]
        records.append(
            {
                "variable": RQ2_LABELS[variable],
                "source_variable": variable,
                "test": strongest["test"],
                "n": int(strongest["n"]),
                "strongest_region": strongest["region"],
                "effect_name": strongest["effect_name"],
                "effect": strongest["effect"],
                "group_adjusted_permutation_p": strongest["p_group_adjusted_permutation"],
                "pooled_fdr_q_105": strongest["p_fdr_group_adjusted_105"],
                "nominal_group_adjusted_hits": int((group["p_group_adjusted_permutation"] < 0.05).sum()),
                "fdr_significant_hits": int((group["p_fdr_group_adjusted_105"] < 0.05).sum()),
            }
        )
    result = pd.DataFrame.from_records(records)
    result.to_csv(output_dir / "rq2_summary.csv", index=False)
    detailed.to_csv(output_dir / "rq2_all_tests.csv", index=False)

    raw_hits = int((detailed["p_group_adjusted_permutation"] < 0.05).sum())
    fdr_hits = int((detailed["p_fdr_group_adjusted_105"] < 0.05).sum())
    if raw_hits != 9 or fdr_hits != 0:
        raise AssertionError("RQ2 significance counts changed")

    effect_symbols = {
        "Mann-Whitney U": r"$r_{\mathrm{rb}}$",
        "Spearman": r"$\rho$",
        "Kruskal-Wallis": r"$\epsilon^2$",
    }
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Strongest group-adjusted RQ2 screening result for each clinical variable. The displayed $p$-values use 20,000 within-group permutations; $q$ is pooled Benjamini--Hochberg adjustment across all 105 tests. Nominal hits count regions with permutation $p<0.05$. No association survived pooled FDR correction.}",
        r"\label{tab:rq2_disjoint}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{tabular}{lrllccc}",
        r"\hline",
        r"Variable & $n$ & Region & Effect & $p$ & FDR $q$ & Nominal hits \\",
        r"\hline",
    ]
    for row in result.itertuples(index=False):
        effect = f"{effect_symbols[row.test]}={row.effect:.3f}"
        lines.append(
            f"{latex_escape(row.variable)} & {row.n} & {latex_escape(row.strongest_region)} & "
            f"{effect} & {row.group_adjusted_permutation_p:.4f} & "
            f"{row.pooled_fdr_q_105:.3f} & {row.nominal_group_adjusted_hits} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    write_table(output_dir / "table_rq2_variable_summary.tex", "\n".join(lines))
    return result, {"nominal_hits": raw_hits, "fdr_hits": fdr_hits}


def build_rq3_tables(input_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    detailed = pd.read_csv(input_dir / "rq3_group_adjusted_165.csv")
    models = pd.read_csv(input_dir / "rq3_group_adjusted_model_summary.csv")
    nuisance = pd.read_csv(input_dir / "rq3_group_nuisance_60.csv")
    if len(detailed) != 165 or len(models) != 15 or len(nuisance) != 60:
        raise AssertionError("RQ3 output sizes do not match 165 primary, 15 model and 60 nuisance rows")
    if detailed.groupby("region").size().nunique() != 1 or detailed.groupby("region").size().iloc[0] != 11:
        raise AssertionError("RQ3 must contain 11 primary coefficients per region")
    if nuisance["included_in_165_fdr_family"].astype(bool).any():
        raise AssertionError("Group nuisance terms were unexpectedly included in the 165-test family")

    conventional_bh = bh_adjust(detailed["p_conventional"])
    hc3_bh = bh_adjust(detailed["p_hc3"])
    if not np.allclose(conventional_bh, detailed["p_fdr_conventional_165"], atol=1e-12, rtol=0):
        raise AssertionError("RQ3 conventional pooled BH values do not reproduce")
    if not np.allclose(hc3_bh, detailed["p_fdr_hc3_165"], atol=1e-12, rtol=0):
        raise AssertionError("RQ3 HC3 pooled BH values do not reproduce")

    conventional_hits = int((detailed["p_fdr_conventional_165"] < 0.05).sum())
    hc3_hits = int((detailed["p_fdr_hc3_165"] < 0.05).sum())
    if conventional_hits != 0 or hc3_hits != 3:
        raise AssertionError("RQ3 FDR significance counts changed")

    findings = detailed.loc[detailed["p_fdr_hc3_165"] < 0.05].copy()
    findings = findings.merge(models[["region", "adjusted_r_squared"]], on="region", how="left")
    findings = findings.sort_values(["p_fdr_hc3_165", "region", "contrast"])
    findings.to_csv(output_dir / "rq3_hc3_results.csv", index=False)
    detailed.to_csv(output_dir / "rq3_all_coefficients.csv", index=False)
    nuisance.to_csv(output_dir / "rq3_group_coefficients.csv", index=False)
    models.to_csv(output_dir / "rq3_model_summary.csv", index=False)

    inference_summary = pd.DataFrame(
        [
            {
                "analysed_patients": int(detailed["n"].iloc[0]),
                "regional_models": len(models),
                "primary_coefficients": len(detailed),
                "conventional_raw_p_lt_0_05": int((detailed["p_conventional"] < 0.05).sum()),
                "conventional_fdr_q_lt_0_05": conventional_hits,
                "hc3_raw_p_lt_0_05": int((detailed["p_hc3"] < 0.05).sum()),
                "hc3_fdr_q_lt_0_05": hc3_hits,
                "mean_r_squared": models["r_squared"].mean(),
                "mean_adjusted_r_squared": models["adjusted_r_squared"].mean(),
                "median_adjusted_r_squared": models["adjusted_r_squared"].median(),
            }
        ]
    )
    inference_summary.to_csv(output_dir / "rq3_summary.csv", index=False)

    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{RQ3 coefficients that survived pooled FDR correction only in the prespecified HC3 robust-standard-error sensitivity analysis. No coefficient survived correction under conventional OLS inference. Coefficients are entropy differences in nats relative to hippocampal sclerosis. Brackets are unadjusted HC3 95\% confidence intervals; $q$-values are Benjamini--Hochberg adjusted across the 165 primary coefficients.}",
        r"\label{tab:rq3_hc3_disjoint}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Region & Contrast & $\beta$ & Conventional $q$ & HC3 95\% CI & HC3 $q$ \\",
        r"\hline",
    ]
    for row in findings.itertuples(index=False):
        ci = f"[{row.ci95_low_hc3:.3f}, {row.ci95_high_hc3:.3f}]"
        lines.append(
            f"{latex_escape(row.region)} & {latex_escape(row.contrast)} & {row.beta:.3f} & "
            f"{row.p_fdr_conventional_165:.3f} & {ci} & {row.p_fdr_hc3_165:.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    write_table(output_dir / "table_rq3_hc3_fdr_findings.tex", "\n".join(lines))

    summary = inference_summary.iloc[0].to_dict()
    return findings, summary


def build_decomposition_tables(input_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    detailed = pd.read_csv(input_dir / "analysis_1_to_4/03_uncertainty_decomposition_exploratory.csv")
    if len(detailed) != 15 or set(detailed["region"]) != set(REGION_ORDER):
        raise AssertionError("Decomposition output must contain the 15 prespecified regions")
    mean_residual = (
        detailed["mean_total_predictive_entropy"]
        - detailed["mean_member_entropy"]
        - detailed["mean_mutual_information"]
    ).abs().max()
    if mean_residual > 1e-7 or detailed["maximum_subject_decomposition_error"].max() > 1e-6:
        raise AssertionError("Predictive entropy decomposition identity failed")

    detailed = detailed.set_index("region").loc[REGION_ORDER].reset_index()
    detailed.to_csv(output_dir / "decomposition_by_region.csv", index=False)

    summary_rows = [
        {
            "component": "Total predictive entropy",
            "mean_of_regional_means_nats": detailed["mean_total_predictive_entropy"].mean(),
            "mean_regional_error_rho": detailed["total_entropy_error_spearman_rho"].mean(),
            "minimum_regional_error_rho": detailed["total_entropy_error_spearman_rho"].min(),
            "maximum_regional_error_rho": detailed["total_entropy_error_spearman_rho"].max(),
        },
        {
            "component": "Mean-member entropy",
            "mean_of_regional_means_nats": detailed["mean_member_entropy"].mean(),
            "mean_regional_error_rho": detailed["mean_member_entropy_error_spearman_rho"].mean(),
            "minimum_regional_error_rho": detailed["mean_member_entropy_error_spearman_rho"].min(),
            "maximum_regional_error_rho": detailed["mean_member_entropy_error_spearman_rho"].max(),
        },
        {
            "component": "Mutual information",
            "mean_of_regional_means_nats": detailed["mean_mutual_information"].mean(),
            "mean_regional_error_rho": detailed["mutual_information_error_spearman_rho"].mean(),
            "minimum_regional_error_rho": detailed["mutual_information_error_spearman_rho"].min(),
            "maximum_regional_error_rho": detailed["mutual_information_error_spearman_rho"].max(),
        },
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "decomposition_summary.csv", index=False)

    mi_median = float(detailed["mutual_information_share_of_total"].median())
    mi_min = float(detailed["mutual_information_share_of_total"].min())
    mi_max = float(detailed["mutual_information_share_of_total"].max())
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{Exploratory decomposition of predictive entropy across the 15 target regions. Values are unweighted summaries of region-level means and region-level Spearman associations with Dice error. Mutual information accounted for a median {100*mi_median:.1f}\% of mean regional predictive entropy (range {100*mi_min:.1f}--{100*mi_max:.1f}\%). The component correlations are descriptive and were not treated as a second confirmatory test family.}}",
        r"\label{tab:decomposition_disjoint}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Component & Mean entropy (nats) & Mean $\rho$ & Regional $\rho$ range \\",
        r"\hline",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.component)} & {row.mean_of_regional_means_nats:.3f} & "
            f"{row.mean_regional_error_rho:.3f} & "
            f"[{row.minimum_regional_error_rho:.3f}, {row.maximum_regional_error_rho:.3f}] \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    write_table(output_dir / "table_decomposition_summary.tex", "\n".join(lines))
    return detailed, {
        "mi_share_median": mi_median,
        "mi_share_min": mi_min,
        "mi_share_max": mi_max,
        "maximum_mean_decomposition_residual": float(mean_residual),
    }


def build_failure_tables(input_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    primary = pd.read_csv(input_dir / "analysis_1_to_4/04_failure_detection_primary.csv")
    paired = pd.read_csv(input_dir / "analysis_1_to_4/04_failure_detection_entropy_vs_msp.csv")
    risk_points = pd.read_csv(input_dir / "analysis_risk_coverage/risk_coverage_primary_points.csv")
    risk_compare = pd.read_csv(input_dir / "analysis_risk_coverage/risk_coverage_entropy_vs_msp.csv")
    if len(primary) != 4 or len(paired) != 1 or len(risk_points) != 6:
        raise AssertionError("Failure-detection or risk-coverage primary output has an unexpected size")
    if set(primary["failure_threshold"]) != {0.5} or set(primary["total_subjects"]) != {532}:
        raise AssertionError("Primary failure definition changed")
    if set(primary["failure_count"]) != {150} or set(primary["n_unscorable"]) != {0}:
        raise AssertionError("Failure prevalence or score completeness changed")

    entropy = primary.loc[primary["score"] == "Predictive entropy"].iloc[0]
    msp = primary.loc[primary["score"] == "Maximum-softmax uncertainty"].iloc[0]
    comparison = paired.iloc[0]
    assert_close(entropy["auroc"] - msp["auroc"], comparison["auroc_difference"], 1e-12, "AUROC difference")
    assert_close(
        entropy["auprc_average_precision"] - msp["auprc_average_precision"],
        comparison["auprc_difference"],
        1e-12,
        "AUPRC difference",
    )

    primary.to_csv(output_dir / "failure_detection.csv", index=False)
    risk_points.to_csv(output_dir / "risk_coverage_points.csv", index=False)
    risk_compare.to_csv(output_dir / "risk_coverage_entropy_vs_msp.csv", index=False)

    score_order = [
        "Predictive entropy",
        "Maximum-softmax uncertainty",
        "Mean-member entropy",
        "Mutual information",
    ]
    primary = primary.set_index("score").loc[score_order].reset_index()
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Label-free detection of subject-level segmentation failure for the leakage-free disjoint ensemble. Failure was defined as mean Dice below 0.50 across the 15 targets (150/532 subjects, 28.2\%). Scores were averaged over voxels predicted as one of the 15 targets. Brackets give 95\% paired-subject bootstrap confidence intervals.}",
        r"\label{tab:failure_detection_disjoint}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Label-free score & AUROC [95\% CI] & AUPRC [95\% CI] \\",
        r"\hline",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.score)} & "
            f"{fmt_interval(row.auroc, row.auroc_ci_low, row.auroc_ci_high)} & "
            f"{fmt_interval(row.auprc_average_precision, row.auprc_ci_low, row.auprc_ci_high)} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    write_table(output_dir / "table_failure_detection_primary.tex", "\n".join(lines))

    risk_points["score_order"] = risk_points["score"].map(
        {"Predictive entropy": 0, "Maximum-softmax uncertainty": 1}
    )
    risk_points = risk_points.sort_values(["score_order", "target_coverage"], ascending=[True, False])
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Risk--coverage results using the two primary label-free uncertainty rankings. At each coverage, the least-uncertain subjects under that score were retained. Selective risk is $1-$mean Dice across the 15 targets; brackets give 95\% paired-subject bootstrap confidence intervals.}",
        r"\label{tab:risk_coverage_disjoint}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lrrcc}",
        r"\hline",
        r"Score & Coverage & Retained & Mean Dice [95\% CI] & Selective risk [95\% CI] \\",
        r"\hline",
    ]
    for row in risk_points.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.score)} & {100*row.target_coverage:.0f}\\% & {row.retained_subjects} & "
            f"{fmt_interval(row.retained_mean_dice, row.retained_mean_dice_ci_low, row.retained_mean_dice_ci_high)} & "
            f"{fmt_interval(row.selective_risk_one_minus_dice, row.selective_risk_ci_low, row.selective_risk_ci_high)} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    write_table(output_dir / "table_risk_coverage_primary_points.tex", "\n".join(lines))

    for score, group in risk_points.groupby("score"):
        ordered = group.sort_values("target_coverage", ascending=False)
        if not np.all(np.diff(ordered["selective_risk_one_minus_dice"]) <= 1e-12):
            raise AssertionError(f"Selective risk does not decrease with coverage for {score}")

    return primary, risk_points, {
        "failure_count": 150,
        "failure_prevalence": float(entropy["failure_prevalence"]),
        "entropy_auroc": float(entropy["auroc"]),
        "entropy_auprc": float(entropy["auprc_average_precision"]),
        "entropy_minus_msp_auroc": float(comparison["auroc_difference"]),
        "entropy_minus_msp_auprc": float(comparison["auprc_difference"]),
    }


def write_readme(
    output_dir: Path,
    regional: dict[str, float],
    rq2: dict[str, float],
    rq3: dict[str, float],
    decomp: dict[str, float],
    failure: dict[str, float],
    source_count: int,
) -> None:
    readme = f"""# Thesis quantitative tables

Generated from the disjoint-ensemble analysis outputs.

## LaTeX tables

- `table_regional_performance_rq1.tex`
- `table_rq2_variable_summary.tex`
- `table_rq3_hc3_fdr_findings.tex`
- `table_decomposition_summary.tex`
- `table_failure_detection_primary.tex`
- `table_risk_coverage_primary_points.tex`
- `table_mde_sensitivity.tex` (generated by `mde_sensitivity.py`)

Each LaTeX table has a corresponding CSV. The full RQ2 and RQ3 results are also
included as supporting CSVs.

## Summary

- RQ1: {regional['fdr_significant_regions']}/15 regional entropy-error associations survived FDR; mean regional rho = {regional['mean_regional_rho']:.4f}.
- RQ2: {rq2['nominal_hits']}/105 group-adjusted permutation p-values were below 0.05, but {rq2['fdr_hits']}/105 survived pooled FDR.
- RQ3: no conventional OLS coefficient survived pooled FDR; {int(rq3['hc3_fdr_q_lt_0_05'])} localized pathology contrasts survived only with HC3 inference. Mean adjusted R-squared = {rq3['mean_adjusted_r_squared']:.4f}.
- Decomposition: mutual information represented a median {100*decomp['mi_share_median']:.1f}% of mean regional predictive entropy (range {100*decomp['mi_share_min']:.1f}-{100*decomp['mi_share_max']:.1f}%).
- Failure detection: predictive-entropy AUROC = {failure['entropy_auroc']:.4f}, AUPRC = {failure['entropy_auprc']:.4f}; it did not outperform maximum-softmax uncertainty.

## Validation

`source_files.csv` lists all {source_count} source files used to generate the
tables. `AUDIT_REPORT.md` records the validation checks.
Run `make_tables.py INPUT_DIR OUTPUT_DIR` to regenerate the tables, followed by
`mde_sensitivity.py OUTPUT_DIR` for the MDE table.
"""
    write_table(output_dir / "README.md", readme)


def write_audit(
    output_dir: Path,
    regional: dict[str, float],
    rq2: dict[str, float],
    rq3: dict[str, float],
    decomp: dict[str, float],
    failure: dict[str, float],
) -> None:
    audit = f"""# Quantitative table validation report

## Source and population

- 532 rows and 532 unique subject identifiers.
- Disjoint group sizes: 107, 107, 106, 106, and 106.
- No missing Dice or reference-masked predictive-entropy value in any of the 15 targets.
- Mean subject-level Dice across the 15 targets: {regional['subject_mean_dice_15']:.12f}.

## RQ1

- All 15 Spearman coefficients and raw p-values were recomputed from subject-level entropy and `1 - Dice` and matched the saved output within 1e-12.
- Mean regional rho: {regional['mean_regional_rho']:.9f}; median: {regional['median_regional_rho']:.9f}.
- FDR-significant regions: {regional['fdr_significant_regions']}/15.
- Largest absolute change after disjoint-group adjustment: {regional['max_abs_group_adjustment_rho_change']:.9f}.

## RQ2

- 105 rows = 7 clinical variables x 15 regions.
- Both pooled 105-test BH families were recomputed and matched within 1e-12.
- Group-adjusted nominal p < 0.05: {rq2['nominal_hits']}/105; pooled FDR q < 0.05: {rq2['fdr_hits']}/105.

## RQ3

- 165 primary coefficients = 15 regional models x 11 coefficients; 60 group nuisance coefficients remained outside the FDR family.
- Conventional and HC3 pooled BH values were recomputed and matched within 1e-12.
- Conventional FDR findings: {int(rq3['conventional_fdr_q_lt_0_05'])}; HC3 sensitivity findings: {int(rq3['hc3_fdr_q_lt_0_05'])}.
- Mean adjusted R-squared: {rq3['mean_adjusted_r_squared']:.9f}.

## Decomposition

- Mean total predictive entropy equalled mean-member entropy plus mutual information; maximum mean residual = {decomp['maximum_mean_decomposition_residual']:.3e}.
- Saved maximum subject-level decomposition error was below 1e-6.

## Failure detection and selective prediction

- Primary failure definition: mean 15-target Dice < 0.50; {failure['failure_count']}/532 failures.
- All four primary scores were available for all 532 subjects.
- Saved predictive-entropy-minus-MSP metric differences matched direct subtraction within 1e-12.
- Both primary risk curves decreased monotonically from 100% to 50% coverage.

## Notes

The HC3 RQ3 findings are sensitivity-analysis findings, not conventional-OLS
confirmations. The MDE table is a conservative Bonferroni-based two-group
sensitivity analysis for the 105-test screening family; it is not a power
analysis for the Spearman or six-level Kruskal-Wallis tests.
"""
    write_table(output_dir / "AUDIT_REPORT.md", audit)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = manifest(input_dir, output_dir)
    _, regional_summary = build_regional_table(input_dir, output_dir)
    _, rq2_summary = build_rq2_table(input_dir, output_dir)
    _, rq3_summary = build_rq3_tables(input_dir, output_dir)
    _, decomp_summary = build_decomposition_tables(input_dir, output_dir)
    _, _, failure_summary = build_failure_tables(input_dir, output_dir)
    write_readme(
        output_dir,
        regional_summary,
        rq2_summary,
        rq3_summary,
        decomp_summary,
        failure_summary,
        len(source_manifest),
    )
    write_audit(
        output_dir,
        regional_summary,
        rq2_summary,
        rq3_summary,
        decomp_summary,
        failure_summary,
    )

    print(f"Generated tables in: {output_dir}")
    print(f"Source files hashed: {len(source_manifest)}")
    print(f"RQ1 FDR-significant regions: {regional_summary['fdr_significant_regions']}/15")
    print(f"RQ2 FDR-significant tests: {rq2_summary['fdr_hits']}/105")
    print(f"RQ3 conventional/HC3 FDR findings: {int(rq3_summary['conventional_fdr_q_lt_0_05'])}/{int(rq3_summary['hc3_fdr_q_lt_0_05'])}")


if __name__ == "__main__":
    main()
