#!/usr/bin/env python3
"""Run the group, RQ1, decomposition and failure-detection analyses."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, rankdata, spearmanr
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


TARGETS = [
    "Brain-Stem",
    "L-Cerebral-WM",
    "R-Cerebral-WM",
    "L-Thalamus",
    "R-Thalamus",
    "L-Caudate",
    "R-Caudate",
    "L-Putamen",
    "R-Putamen",
    "L-Pallidum",
    "R-Pallidum",
    "L-Hippocampus",
    "R-Hippocampus",
    "L-Amygdala",
    "R-Amygdala",
]

COMPONENTS = [
    "entropy",
    "mean_member_entropy",
    "mutual_information",
]

PREDTGT_SCORES = {
    "Predictive entropy": "predtgt_entropy_mean",
    "Maximum-softmax uncertainty": "predtgt_msp_unc_mean",
    "Mean-member entropy": "predtgt_mean_member_entropy_mean",
    "Mutual information": "predtgt_mutual_information_mean",
}

PREDFG_SCORES = {
    "Predictive entropy": "predfg_entropy_mean",
    "Maximum-softmax uncertainty": "predfg_msp_unc_mean",
}


def bh_fdr(values: np.ndarray | pd.Series) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving input order."""
    p = np.asarray(values, dtype=float)
    if np.isnan(p).any():
        raise ValueError("BH-FDR input contains missing p-values")
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion."""
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def fisher_interval(rho: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Approximate Fisher-z 95% CI for a rank-correlation effect size."""
    if n <= 3 or not np.isfinite(rho):
        return np.nan, np.nan
    clipped = float(np.clip(rho, -0.999999, 0.999999))
    transformed = np.arctanh(clipped)
    half = z / math.sqrt(n - 3)
    return float(np.tanh(transformed - half)), float(np.tanh(transformed + half))


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    xx = np.asarray(x[mask], dtype=float)
    yy = np.asarray(y[mask], dtype=float)
    if len(xx) < 4 or np.unique(xx).size < 2 or np.unique(yy).size < 2:
        return np.nan, np.nan, len(xx)
    rho, p_value = spearmanr(xx, yy)
    return float(rho), float(p_value), len(xx)


def metric_pair(labels: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Return AUROC and average precision; higher score means more uncertain."""
    labels = np.asarray(labels, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(labels) == 0 or np.unique(labels).size < 2:
        return np.nan, np.nan
    return float(roc_auc_score(labels, score)), float(average_precision_score(labels, score))


def bootstrap_score_table(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Paired subject bootstrap for several complete-case uncertainty scores."""
    labels = np.asarray(labels, dtype=int)
    score_names = list(scores)
    matrix = np.column_stack([np.asarray(scores[name], dtype=float) for name in score_names])
    complete = np.isfinite(matrix).all(axis=1) & np.isfinite(labels)
    y = labels[complete]
    matrix = matrix[complete]
    has_two_classes = np.unique(y).size >= 2

    point_auc = []
    point_ap = []
    for column in range(matrix.shape[1]):
        auc, ap = metric_pair(y, matrix[:, column]) if has_two_classes else (np.nan, np.nan)
        point_auc.append(auc)
        point_ap.append(ap)

    boot_auc = np.full((n_bootstrap, matrix.shape[1]), np.nan, dtype=float)
    boot_ap = np.full((n_bootstrap, matrix.shape[1]), np.nan, dtype=float)

    if has_two_classes:
        for iteration in range(n_bootstrap):
            indices = rng.integers(0, len(y), size=len(y))
            sampled_y = y[indices]
            if np.unique(sampled_y).size < 2:
                continue
            sampled_scores = matrix[indices]
            for column in range(matrix.shape[1]):
                boot_auc[iteration, column], boot_ap[iteration, column] = metric_pair(
                    sampled_y,
                    sampled_scores[:, column],
                )

    rows = []
    for column, name in enumerate(score_names):
        auc_lo, auc_hi = percentile_interval(boot_auc[:, column])
        ap_lo, ap_hi = percentile_interval(boot_ap[:, column])
        rows.append(
            {
                "score": name,
                "n_scored": int(complete.sum()),
                "n_unscorable": int((~complete).sum()),
                "auroc": point_auc[column],
                "auroc_ci_low": auc_lo,
                "auroc_ci_high": auc_hi,
                "auprc_average_precision": point_ap[column],
                "auprc_ci_low": ap_lo,
                "auprc_ci_high": ap_hi,
                "valid_bootstrap_replicates": int(np.isfinite(boot_auc[:, column]).sum()),
            }
        )

    distributions = {
        name: np.column_stack([boot_auc[:, column], boot_ap[:, column]])
        for column, name in enumerate(score_names)
    }
    distributions["_complete_mask"] = complete
    return pd.DataFrame(rows), distributions


def parse_models_used(value: object) -> set[int]:
    text = str(value).strip()
    if not text:
        return set()
    separators = ["|", ",", ";", " "]
    for separator in separators[1:]:
        text = text.replace(separator, separators[0])
    return {int(token) for token in text.split("|") if token != ""}


def validate_input(data: pd.DataFrame, expected_subjects: int) -> dict[str, object]:
    """Validate the inference table."""
    required = ["subject_id", "group", "models_used", "predtgt_n_voxels", "predfg_n_voxels"]
    for region in TARGETS:
        required.extend(
            [
                f"{region}_dice",
                f"{region}_entropy_mean",
                f"{region}_mean_member_entropy_mean",
                f"{region}_mutual_information_mean",
                f"{region}_msp_unc_mean",
                f"{region}_pred_entropy_mean",
                f"{region}_pred_msp_unc_mean",
            ]
        )
    required.extend(PREDTGT_SCORES.values())
    required.extend(PREDFG_SCORES.values())
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise RuntimeError(f"Missing required columns ({len(missing)}): {missing}")

    if len(data) != expected_subjects:
        raise RuntimeError(f"Expected {expected_subjects} rows, found {len(data)}")
    if data["subject_id"].isna().any() or data["subject_id"].duplicated().any():
        raise RuntimeError("Subject IDs are missing or duplicated")

    group_numeric = pd.to_numeric(data["group"], errors="coerce")
    if group_numeric.isna().any():
        raise RuntimeError("Disjoint-group column contains non-numeric or missing values")
    groups = group_numeric.astype(int)
    if sorted(groups.unique()) != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Expected groups 0-4, found {sorted(groups.unique())}")

    expected_models = set(range(5))
    invalid_model_rows = []
    for row_index, (group, models) in enumerate(zip(groups, data["models_used"])):
        observed = parse_models_used(models)
        expected = expected_models - {int(group)}
        if observed != expected:
            invalid_model_rows.append((row_index, int(group), sorted(observed), sorted(expected)))
    if invalid_model_rows:
        raise RuntimeError(
            "Subject-specific model exclusion is inconsistent. First rows: "
            + repr(invalid_model_rows[:5])
        )

    numeric_columns = []
    for region in TARGETS:
        numeric_columns.extend(
            [
                f"{region}_dice",
                f"{region}_entropy_mean",
                f"{region}_mean_member_entropy_mean",
                f"{region}_mutual_information_mean",
                f"{region}_msp_unc_mean",
            ]
        )
    numeric = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    bad_numeric = numeric.isna().sum()
    if (bad_numeric > 0).any():
        raise RuntimeError(
            "Missing/non-numeric reference-masked target values:\n"
            + bad_numeric[bad_numeric > 0].to_string()
        )

    dice_columns = [f"{region}_dice" for region in TARGETS]
    dice = numeric[dice_columns].to_numpy(dtype=float)
    if np.any((dice < 0.0) | (dice > 1.0)):
        raise RuntimeError("Dice values outside [0, 1]")

    entropy_columns = [f"{region}_entropy_mean" for region in TARGETS]
    mme_columns = [f"{region}_mean_member_entropy_mean" for region in TARGETS]
    mi_columns = [f"{region}_mutual_information_mean" for region in TARGETS]
    entropy = numeric[entropy_columns].to_numpy(dtype=float)
    mme = numeric[mme_columns].to_numpy(dtype=float)
    mi = numeric[mi_columns].to_numpy(dtype=float)
    if np.min(entropy) < -1e-8 or np.min(mme) < -1e-8 or np.min(mi) < -1e-8:
        raise RuntimeError("Negative entropy/decomposition values beyond numerical tolerance")
    decomposition_error = np.abs(entropy - mme - mi)
    maximum_decomposition_error = float(np.max(decomposition_error))
    if maximum_decomposition_error > 1e-4:
        raise RuntimeError(
            "Stored regional decomposition violates H(mean p) = mean H(p_m) + MI; "
            f"maximum absolute error={maximum_decomposition_error:.6g}"
        )

    msp_columns = [f"{region}_msp_unc_mean" for region in TARGETS]
    msp = numeric[msp_columns].to_numpy(dtype=float)
    if np.any((msp < -1e-8) | (msp > 1.0 + 1e-8)):
        raise RuntimeError("Maximum-softmax uncertainty values outside [0, 1]")

    group_sizes = groups.value_counts().sort_index().to_dict()
    if max(group_sizes.values()) - min(group_sizes.values()) > 1:
        raise RuntimeError(f"Disjoint groups are unexpectedly uneven: {group_sizes}")
    return {
        "groups": groups.to_numpy(dtype=int),
        "group_sizes": {str(int(key)): int(value) for key, value in group_sizes.items()},
        "maximum_decomposition_error": maximum_decomposition_error,
    }


def group_diagnostic(data: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare results across the five subject groups."""
    subject = data[["subject_id", "group"]].copy()
    subject["target_mean_entropy_15"] = data[
        [f"{region}_entropy_mean" for region in TARGETS]
    ].mean(axis=1)
    subject["target_mean_dice_15"] = data[
        [f"{region}_dice" for region in TARGETS]
    ].mean(axis=1)
    subject.to_csv(output_dir / "01_group_diagnostic_subject_values.csv", index=False)

    summary_rows = []
    test_rows = []
    for metric in ["target_mean_entropy_15", "target_mean_dice_15"]:
        for group, values in subject.groupby("group")[metric]:
            values = values.to_numpy(dtype=float)
            summary_rows.append(
                {
                    "metric": metric,
                    "group": int(group),
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )

        samples = [
            group_values[metric].to_numpy(dtype=float)
            for _, group_values in subject.groupby("group")
        ]
        result = kruskal(*samples)
        n = len(subject)
        k = len(samples)
        epsilon_squared = max(0.0, (float(result.statistic) - k + 1.0) / (n - k))
        group_means = subject.groupby("group")[metric].mean()
        within_sd = subject.groupby("group")[metric].std().mean()
        test_rows.append(
            {
                "metric": metric,
                "n": n,
                "groups": k,
                "kruskal_h": float(result.statistic),
                "p_value": float(result.pvalue),
                "epsilon_squared": epsilon_squared,
                "group_mean_range": float(group_means.max() - group_means.min()),
                "mean_within_group_sd": float(within_sd),
                "range_to_within_sd_ratio": float((group_means.max() - group_means.min()) / within_sd),
            }
        )

    summary = pd.DataFrame(summary_rows)
    tests = pd.DataFrame(test_rows)
    summary.to_csv(output_dir / "01_group_diagnostic_summary.csv", index=False)
    tests.to_csv(output_dir / "01_group_diagnostic_omnibus.csv", index=False)

    region_rows = []
    for outcome, suffix in [("entropy", "entropy_mean"), ("dice", "dice")]:
        for region in TARGETS:
            column = f"{region}_{suffix}"
            samples = [
                group_values[column].to_numpy(dtype=float)
                for _, group_values in data.groupby("group")
            ]
            result = kruskal(*samples)
            n = sum(len(sample) for sample in samples)
            k = len(samples)
            group_means = data.groupby("group")[column].mean()
            region_rows.append(
                {
                    "outcome": outcome,
                    "region": region,
                    "n": n,
                    "kruskal_h": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "epsilon_squared": max(0.0, (float(result.statistic) - k + 1.0) / (n - k)),
                    "group_mean_min": float(group_means.min()),
                    "group_mean_max": float(group_means.max()),
                    "group_mean_range": float(group_means.max() - group_means.min()),
                }
            )
    region_tests = pd.DataFrame(region_rows)
    region_tests["p_fdr_within_outcome_15"] = np.nan
    for outcome in ["entropy", "dice"]:
        mask = region_tests["outcome"] == outcome
        region_tests.loc[mask, "p_fdr_within_outcome_15"] = bh_fdr(
            region_tests.loc[mask, "p_value"]
        )
    region_tests.to_csv(output_dir / "01_group_diagnostic_regionwise.csv", index=False)
    return subject, summary, tests


def centered_within_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    for group in np.unique(groups):
        indices = groups == group
        result[indices] -= result[indices].mean()
    return result


def partial_spearman_with_group(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
    batch_size: int = 1000,
) -> tuple[float, float]:
    """Partial Spearman by residualising global ranks on group fixed effects."""
    x_rank = rankdata(np.asarray(x, dtype=float), method="average")
    y_rank = rankdata(np.asarray(y, dtype=float), method="average")
    x_centered = centered_within_group(x_rank, groups)
    y_centered = centered_within_group(y_rank, groups)
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denominator <= 0:
        return np.nan, np.nan
    observed = float(np.dot(x_centered, y_centered) / denominator)

    group_indices = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    exceedances = 0
    completed = 0
    while completed < permutations:
        batch = min(batch_size, permutations - completed)
        permuted = np.empty((batch, len(x_centered)), dtype=float)
        for indices in group_indices:
            order = np.argsort(rng.random((batch, len(indices))), axis=1)
            permuted[:, indices] = x_centered[indices][order]
        permuted_rho = (permuted @ y_centered) / denominator
        exceedances += int(np.sum(np.abs(permuted_rho) >= abs(observed) - 1e-12))
        completed += batch
    p_value = (1.0 + exceedances) / (permutations + 1.0)
    return observed, float(p_value)


def rq1_group_sensitivity(
    data: pd.DataFrame,
    groups: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate group-adjusted regional correlations."""
    rows = []
    within_rows = []
    for region in TARGETS:
        entropy = pd.to_numeric(data[f"{region}_entropy_mean"], errors="raise").to_numpy(float)
        error = 1.0 - pd.to_numeric(data[f"{region}_dice"], errors="raise").to_numpy(float)
        unadjusted_rho, unadjusted_p, n = safe_spearman(entropy, error)
        adjusted_rho, adjusted_p = partial_spearman_with_group(
            entropy,
            error,
            groups,
            permutations,
            rng,
        )
        adjusted_ci_low, adjusted_ci_high = fisher_interval(adjusted_rho, n - 4)
        rows.append(
            {
                "region": region,
                "n": n,
                "unadjusted_spearman_rho": unadjusted_rho,
                "unadjusted_p": unadjusted_p,
                "group_adjusted_partial_spearman_rho": adjusted_rho,
                "group_adjusted_rho_ci_low_approx": adjusted_ci_low,
                "group_adjusted_rho_ci_high_approx": adjusted_ci_high,
                "group_adjusted_within_group_permutation_p": adjusted_p,
                "rho_change_adjusted_minus_unadjusted": adjusted_rho - unadjusted_rho,
                "permutations": permutations,
            }
        )

        for group in sorted(np.unique(groups)):
            mask = groups == group
            rho, p_value, group_n = safe_spearman(entropy[mask], error[mask])
            ci_low, ci_high = fisher_interval(rho, group_n)
            within_rows.append(
                {
                    "region": region,
                    "group": int(group),
                    "n": group_n,
                    "spearman_rho": rho,
                    "rho_ci_low_approx": ci_low,
                    "rho_ci_high_approx": ci_high,
                    "p_value_descriptive": p_value,
                }
            )

    results = pd.DataFrame(rows)
    results["unadjusted_p_fdr_15"] = bh_fdr(results["unadjusted_p"])
    results["group_adjusted_permutation_p_fdr_15"] = bh_fdr(
        results["group_adjusted_within_group_permutation_p"]
    )
    within = pd.DataFrame(within_rows)
    within_summary = within.groupby("region").agg(
        within_group_rho_min=("spearman_rho", "min"),
        within_group_rho_max=("spearman_rho", "max"),
        within_group_rho_mean=("spearman_rho", "mean"),
        groups_positive=("spearman_rho", lambda values: int((values > 0).sum())),
    ).reset_index()
    results = results.merge(within_summary, on="region", how="left", validate="one_to_one")
    results.to_csv(output_dir / "02_rq1_group_adjusted.csv", index=False)
    within.to_csv(output_dir / "02_rq1_within_group_correlations.csv", index=False)
    return results, within


def decomposition_analysis(
    data: pd.DataFrame,
    bootstrap_replicates: int,
    rng: np.random.Generator,
    output_dir: Path,
) -> pd.DataFrame:
    """Analyse the entropy decomposition."""
    total = data[[f"{region}_entropy_mean" for region in TARGETS]].to_numpy(float)
    mme = data[[f"{region}_mean_member_entropy_mean" for region in TARGETS]].to_numpy(float)
    mi = data[[f"{region}_mutual_information_mean" for region in TARGETS]].to_numpy(float)
    dice = data[[f"{region}_dice" for region in TARGETS]].to_numpy(float)
    error = 1.0 - dice

    boot_total = np.empty((bootstrap_replicates, len(TARGETS)), dtype=float)
    boot_mme = np.empty_like(boot_total)
    boot_mi = np.empty_like(boot_total)
    batch_size = 500
    completed = 0
    while completed < bootstrap_replicates:
        batch = min(batch_size, bootstrap_replicates - completed)
        indices = rng.integers(0, len(data), size=(batch, len(data)))
        target_slice = slice(completed, completed + batch)
        boot_total[target_slice] = total[indices].mean(axis=1)
        boot_mme[target_slice] = mme[indices].mean(axis=1)
        boot_mi[target_slice] = mi[indices].mean(axis=1)
        completed += batch
    boot_share = np.divide(
        boot_mi,
        boot_total,
        out=np.full_like(boot_mi, np.nan),
        where=boot_total > 0,
    )

    rows = []
    for column, region in enumerate(TARGETS):
        total_mean = float(total[:, column].mean())
        mme_mean = float(mme[:, column].mean())
        mi_mean = float(mi[:, column].mean())
        mi_share = mi_mean / total_mean if total_mean > 0 else np.nan
        total_ci = percentile_interval(boot_total[:, column])
        mme_ci = percentile_interval(boot_mme[:, column])
        mi_ci = percentile_interval(boot_mi[:, column])
        share_ci = percentile_interval(boot_share[:, column])

        correlation_values = {}
        for component_name, values in [
            ("total_entropy", total[:, column]),
            ("mean_member_entropy", mme[:, column]),
            ("mutual_information", mi[:, column]),
        ]:
            rho, _, n = safe_spearman(values, error[:, column])
            ci_low, ci_high = fisher_interval(rho, n)
            correlation_values[f"{component_name}_error_spearman_rho"] = rho
            correlation_values[f"{component_name}_error_rho_ci_low_approx"] = ci_low
            correlation_values[f"{component_name}_error_rho_ci_high_approx"] = ci_high

        rows.append(
            {
                "region": region,
                "n": len(data),
                "mean_total_predictive_entropy": total_mean,
                "mean_total_entropy_ci_low": total_ci[0],
                "mean_total_entropy_ci_high": total_ci[1],
                "mean_member_entropy": mme_mean,
                "mean_member_entropy_ci_low": mme_ci[0],
                "mean_member_entropy_ci_high": mme_ci[1],
                "mean_mutual_information": mi_mean,
                "mean_mutual_information_ci_low": mi_ci[0],
                "mean_mutual_information_ci_high": mi_ci[1],
                "mutual_information_share_of_total": mi_share,
                "mutual_information_share_ci_low": share_ci[0],
                "mutual_information_share_ci_high": share_ci[1],
                "maximum_subject_decomposition_error": float(
                    np.max(np.abs(total[:, column] - mme[:, column] - mi[:, column]))
                ),
                **correlation_values,
                "mean_bootstrap_replicates": bootstrap_replicates,
                "correlation_ci_method": "Fisher-z approximation",
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "03_uncertainty_decomposition_exploratory.csv", index=False)
    return results


def group_rank_score(score: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Within-group percentile ranks for a group-adjusted ranking sensitivity."""
    result = np.full(len(score), np.nan, dtype=float)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        values = score[indices]
        valid = np.isfinite(values)
        if valid.any():
            result[indices[valid]] = rankdata(values[valid], method="average") / valid.sum()
    return result


def failure_detection_analysis(
    data: pd.DataFrame,
    groups: np.ndarray,
    bootstrap_replicates: int,
    regional_bootstrap_replicates: int,
    rng: np.random.Generator,
    output_dir: Path,
) -> dict[str, object]:
    """Task 4: primary and sensitivity label-free failure detection."""
    dice_columns = [f"{region}_dice" for region in TARGETS]
    target_mean_dice = data[dice_columns].mean(axis=1).to_numpy(float)
    if not np.isfinite(target_mean_dice).all():
        raise RuntimeError("Subject-level target mean Dice contains missing values")

    primary_failure = (target_mean_dice < 0.50).astype(int)
    if np.unique(primary_failure).size < 2:
        raise RuntimeError(
            "Primary failure outcome has only one class at the prespecified Dice < 0.50 threshold"
        )
    failure_count = int(primary_failure.sum())
    prevalence = failure_count / len(primary_failure)
    prevalence_ci = wilson_interval(failure_count, len(primary_failure))

    primary_score_arrays = {
        name: pd.to_numeric(data[column], errors="coerce").to_numpy(float)
        for name, column in PREDTGT_SCORES.items()
    }
    missing_patterns = [np.isnan(values) for values in primary_score_arrays.values()]
    if not all(np.array_equal(missing_patterns[0], pattern) for pattern in missing_patterns[1:]):
        raise RuntimeError("Predicted-target score components have inconsistent missingness")
    empty_target = pd.to_numeric(data["predtgt_n_voxels"], errors="coerce").to_numpy(float) <= 0
    if not np.array_equal(empty_target, missing_patterns[0]):
        raise RuntimeError(
            "predtgt_n_voxels == 0 does not match missing predicted-target uncertainty scores"
        )

    primary_table, primary_boot = bootstrap_score_table(
        primary_failure,
        primary_score_arrays,
        bootstrap_replicates,
        rng,
    )
    primary_table.insert(0, "analysis", "Primary subject-level target score")
    primary_table.insert(1, "failure_threshold", 0.50)
    primary_table.insert(2, "failure_count", failure_count)
    primary_table.insert(3, "total_subjects", len(primary_failure))
    primary_table.insert(4, "failure_prevalence", prevalence)
    primary_table.insert(5, "failure_prevalence_ci_low_wilson", prevalence_ci[0])
    primary_table.insert(6, "failure_prevalence_ci_high_wilson", prevalence_ci[1])
    primary_table["unscorable_failures"] = int(np.sum(empty_target & (primary_failure == 1)))
    primary_table["unscorable_nonfailures"] = int(np.sum(empty_target & (primary_failure == 0)))
    primary_table.to_csv(output_dir / "04_failure_detection_primary.csv", index=False)

    entropy_boot = primary_boot["Predictive entropy"]
    msp_boot = primary_boot["Maximum-softmax uncertainty"]
    entropy_row = primary_table[primary_table["score"] == "Predictive entropy"].iloc[0]
    msp_row = primary_table[primary_table["score"] == "Maximum-softmax uncertainty"].iloc[0]
    delta_auc = entropy_boot[:, 0] - msp_boot[:, 0]
    delta_ap = entropy_boot[:, 1] - msp_boot[:, 1]
    delta_auc_ci = percentile_interval(delta_auc)
    delta_ap_ci = percentile_interval(delta_ap)
    comparison = pd.DataFrame(
        [
            {
                "comparison": "Predictive entropy minus maximum-softmax uncertainty",
                "n_common_scored": int(primary_boot["_complete_mask"].sum()),
                "auroc_difference": float(entropy_row["auroc"] - msp_row["auroc"]),
                "auroc_difference_ci_low": delta_auc_ci[0],
                "auroc_difference_ci_high": delta_auc_ci[1],
                "auprc_difference": float(
                    entropy_row["auprc_average_precision"] - msp_row["auprc_average_precision"]
                ),
                "auprc_difference_ci_low": delta_ap_ci[0],
                "auprc_difference_ci_high": delta_ap_ci[1],
                "bootstrap_replicates": bootstrap_replicates,
            }
        ]
    )
    comparison.to_csv(output_dir / "04_failure_detection_entropy_vs_msp.csv", index=False)

    threshold_rows = []
    for threshold in [0.40, 0.50, 0.60]:
        labels = (target_mean_dice < threshold).astype(int)
        count = int(labels.sum())
        ci = wilson_interval(count, len(labels))
        scores = {
            "Predictive entropy": primary_score_arrays["Predictive entropy"],
            "Maximum-softmax uncertainty": primary_score_arrays["Maximum-softmax uncertainty"],
        }
        table, _ = bootstrap_score_table(labels, scores, bootstrap_replicates, rng)
        for _, row in table.iterrows():
            threshold_rows.append(
                {
                    "failure_threshold": threshold,
                    "failure_count": count,
                    "total_subjects": len(labels),
                    "failure_prevalence": count / len(labels),
                    "failure_prevalence_ci_low_wilson": ci[0],
                    "failure_prevalence_ci_high_wilson": ci[1],
                    **row.to_dict(),
                }
            )
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(output_dir / "04_failure_detection_threshold_sensitivity.csv", index=False)

    predfg_arrays = {
        name: pd.to_numeric(data[column], errors="coerce").to_numpy(float)
        for name, column in PREDFG_SCORES.items()
    }
    predfg_table, _ = bootstrap_score_table(
        primary_failure,
        predfg_arrays,
        bootstrap_replicates,
        rng,
    )
    predfg_table.insert(0, "analysis", "All predicted foreground sensitivity")
    predfg_table.insert(1, "failure_threshold", 0.50)
    predfg_table.to_csv(output_dir / "04_failure_detection_all_foreground_sensitivity.csv", index=False)

    group_adjusted_arrays = {
        name: group_rank_score(values, groups)
        for name, values in {
            "Predictive entropy": primary_score_arrays["Predictive entropy"],
            "Maximum-softmax uncertainty": primary_score_arrays["Maximum-softmax uncertainty"],
        }.items()
    }
    group_adjusted_table, _ = bootstrap_score_table(
        primary_failure,
        group_adjusted_arrays,
        bootstrap_replicates,
        rng,
    )
    group_adjusted_table.insert(0, "analysis", "Within-group percentile-rank sensitivity")
    group_adjusted_table.insert(1, "failure_threshold", 0.50)
    group_adjusted_table.to_csv(output_dir / "04_failure_detection_group_sensitivity.csv", index=False)

    region_rows = []
    for region in TARGETS:
        region_dice = pd.to_numeric(data[f"{region}_dice"], errors="coerce").to_numpy(float)
        valid_outcome = np.isfinite(region_dice)
        labels = (region_dice[valid_outcome] < 0.50).astype(int)
        entropy_score = pd.to_numeric(
            data.loc[valid_outcome, f"{region}_pred_entropy_mean"], errors="coerce"
        ).to_numpy(float)
        msp_score = pd.to_numeric(
            data.loc[valid_outcome, f"{region}_pred_msp_unc_mean"], errors="coerce"
        ).to_numpy(float)
        table, _ = bootstrap_score_table(
            labels,
            {
                "Predictive entropy": entropy_score,
                "Maximum-softmax uncertainty": msp_score,
            },
            regional_bootstrap_replicates,
            rng,
        )
        for _, row in table.iterrows():
            score_values = entropy_score if row["score"] == "Predictive entropy" else msp_score
            region_rows.append(
                {
                    "region": region,
                    "failure_threshold": 0.50,
                    "n_with_dice": int(valid_outcome.sum()),
                    "failure_count": int(labels.sum()),
                    "failure_prevalence": float(labels.mean()),
                    "unscorable_failures": int(np.sum(np.isnan(score_values) & (labels == 1))),
                    "unscorable_nonfailures": int(np.sum(np.isnan(score_values) & (labels == 0))),
                    **row.to_dict(),
                }
            )
    region_table = pd.DataFrame(region_rows)
    region_table.to_csv(output_dir / "04_failure_detection_regionwise.csv", index=False)

    macro = region_table.groupby("score").agg(
        regions=("region", "nunique"),
        mean_auroc=("auroc", "mean"),
        median_auroc=("auroc", "median"),
        mean_auprc=("auprc_average_precision", "mean"),
        median_auprc=("auprc_average_precision", "median"),
        total_unscorable_failures=("unscorable_failures", "sum"),
    ).reset_index()
    macro.to_csv(output_dir / "04_failure_detection_regionwise_macro.csv", index=False)

    make_failure_detection_figure(
        primary_failure,
        primary_score_arrays,
        primary_table,
        prevalence,
        output_dir,
    )

    return {
        "target_mean_dice": target_mean_dice,
        "primary_failure": primary_failure,
        "primary_table": primary_table,
        "comparison": comparison,
        "threshold_table": threshold_table,
        "predfg_table": predfg_table,
        "group_adjusted_table": group_adjusted_table,
        "region_table": region_table,
        "macro": macro,
        "empty_target_count": int(empty_target.sum()),
    }


def make_failure_detection_figure(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    table: pd.DataFrame,
    prevalence: float,
    output_dir: Path,
) -> None:
    """Save the ROC and precision-recall figure."""
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "matplotlib-thesis-analysis-cache"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "Predictive entropy": "#0072B2",
        "Maximum-softmax uncertainty": "#D55E00",
        "Mean-member entropy": "#009E73",
        "Mutual information": "#CC79A7",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    for name, values in scores.items():
        values = np.asarray(values, dtype=float)
        valid = np.isfinite(values)
        y = labels[valid]
        score = values[valid]
        if np.unique(y).size < 2:
            continue
        row = table[table["score"] == name].iloc[0]
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        axes[0].plot(
            fpr,
            tpr,
            lw=2,
            color=colors[name],
            label=f"{name} (AUROC {row['auroc']:.2f})",
        )
        axes[1].plot(
            recall,
            precision,
            lw=2,
            color=colors[name],
            label=f"{name} (AUPRC {row['auprc_average_precision']:.2f})",
        )

    axes[0].plot([0, 1], [0, 1], "--", color="0.55", lw=1.2, label="Chance")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    axes[1].axhline(prevalence, ls="--", color="0.55", lw=1.2, label=f"Prevalence ({prevalence:.2f})")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.01)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Label-free detection of subjects with mean target Dice < 0.50", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "04_failure_detection_roc_pr.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "04_failure_detection_roc_pr.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(
    input_path: Path,
    output_dir: Path,
    audit: dict[str, object],
    group_tests: pd.DataFrame,
    rq1: pd.DataFrame,
    decomposition: pd.DataFrame,
    failure: dict[str, object],
    args: argparse.Namespace,
) -> str:
    group_entropy = group_tests[group_tests["metric"] == "target_mean_entropy_15"].iloc[0]
    group_dice = group_tests[group_tests["metric"] == "target_mean_dice_15"].iloc[0]

    primary = failure["primary_table"]
    comparison = failure["comparison"].iloc[0]
    macro = failure["macro"]
    target_mean_dice = failure["target_mean_dice"]
    primary_failure = failure["primary_failure"]

    entropy_row = primary[primary["score"] == "Predictive entropy"].iloc[0]
    msp_row = primary[primary["score"] == "Maximum-softmax uncertainty"].iloc[0]
    mme_row = primary[primary["score"] == "Mean-member entropy"].iloc[0]
    mi_row = primary[primary["score"] == "Mutual information"].iloc[0]

    lines = [
        "DISJOINT ENSEMBLE ANALYSES",
        "=" * 72,
        f"Input: {input_path}",
        f"Rows/subjects: {len(target_mean_dice)}",
        f"Disjoint-group sizes: {audit['group_sizes']}",
        f"Maximum regional decomposition identity error: {audit['maximum_decomposition_error']:.6g}",
        "",
        "1. ALL-15-TARGET GROUP DIAGNOSTIC",
        f"Mean entropy: H={group_entropy['kruskal_h']:.6f}, p={group_entropy['p_value']:.6g}, "
        f"epsilon^2={group_entropy['epsilon_squared']:.6f}, group-mean range={group_entropy['group_mean_range']:.6f}",
        f"Mean Dice: H={group_dice['kruskal_h']:.6f}, p={group_dice['p_value']:.6g}, "
        f"epsilon^2={group_dice['epsilon_squared']:.6f}, group-mean range={group_dice['group_mean_range']:.6f}",
        "",
        "2. RQ1 GROUP SENSITIVITY",
        f"Mean unadjusted regional rho: {rq1['unadjusted_spearman_rho'].mean():.6f}",
        f"Median unadjusted regional rho: {rq1['unadjusted_spearman_rho'].median():.6f}",
        f"Mean group-adjusted regional rho: {rq1['group_adjusted_partial_spearman_rho'].mean():.6f}",
        f"Median group-adjusted regional rho: {rq1['group_adjusted_partial_spearman_rho'].median():.6f}",
        f"Positive adjusted correlations: {(rq1['group_adjusted_partial_spearman_rho'] > 0).sum()}/15",
        f"FDR-significant adjusted correlations: {(rq1['group_adjusted_permutation_p_fdr_15'] < 0.05).sum()}/15",
        f"Largest absolute rho change after adjustment: {rq1['rho_change_adjusted_minus_unadjusted'].abs().max():.6f}",
        "",
        "Adjusted regional results:",
        rq1[
            [
                "region",
                "unadjusted_spearman_rho",
                "group_adjusted_partial_spearman_rho",
                "group_adjusted_within_group_permutation_p",
                "group_adjusted_permutation_p_fdr_15",
                "within_group_rho_min",
                "within_group_rho_max",
            ]
        ].to_string(index=False),
        "",
        "3. EXPLORATORY UNCERTAINTY DECOMPOSITION",
        f"Median regional MI share of total entropy: {decomposition['mutual_information_share_of_total'].median():.6f}",
        f"Regional MI-share range: {decomposition['mutual_information_share_of_total'].min():.6f} to "
        f"{decomposition['mutual_information_share_of_total'].max():.6f}",
        f"Mean regional mean-member-entropy/error rho: "
        f"{decomposition['mean_member_entropy_error_spearman_rho'].mean():.6f}",
        f"Mean regional MI/error rho: {decomposition['mutual_information_error_spearman_rho'].mean():.6f}",
        "",
        "Regional decomposition:",
        decomposition[
            [
                "region",
                "mean_total_predictive_entropy",
                "mean_member_entropy",
                "mean_mutual_information",
                "mutual_information_share_of_total",
                "mean_member_entropy_error_spearman_rho",
                "mutual_information_error_spearman_rho",
            ]
        ].to_string(index=False),
        "",
        "4. LABEL-FREE FAILURE DETECTION",
        f"Failure definition: subject mean Dice across 15 targets < 0.50",
        f"Mean target Dice: {np.mean(target_mean_dice):.6f}",
        f"Failures: {int(primary_failure.sum())}/{len(primary_failure)} "
        f"({np.mean(primary_failure):.6f})",
        f"Subjects with no predicted target voxels/unscorable primary score: {failure['empty_target_count']}",
        "",
        f"Predictive entropy: AUROC={entropy_row['auroc']:.6f} "
        f"[{entropy_row['auroc_ci_low']:.6f}, {entropy_row['auroc_ci_high']:.6f}], "
        f"AUPRC={entropy_row['auprc_average_precision']:.6f} "
        f"[{entropy_row['auprc_ci_low']:.6f}, {entropy_row['auprc_ci_high']:.6f}]",
        f"Maximum-softmax uncertainty: AUROC={msp_row['auroc']:.6f} "
        f"[{msp_row['auroc_ci_low']:.6f}, {msp_row['auroc_ci_high']:.6f}], "
        f"AUPRC={msp_row['auprc_average_precision']:.6f} "
        f"[{msp_row['auprc_ci_low']:.6f}, {msp_row['auprc_ci_high']:.6f}]",
        f"Mean-member entropy: AUROC={mme_row['auroc']:.6f}, "
        f"AUPRC={mme_row['auprc_average_precision']:.6f}",
        f"Mutual information: AUROC={mi_row['auroc']:.6f}, "
        f"AUPRC={mi_row['auprc_average_precision']:.6f}",
        f"Entropy - max-softmax AUROC difference: {comparison['auroc_difference']:.6f} "
        f"[{comparison['auroc_difference_ci_low']:.6f}, {comparison['auroc_difference_ci_high']:.6f}]",
        f"Entropy - max-softmax AUPRC difference: {comparison['auprc_difference']:.6f} "
        f"[{comparison['auprc_difference_ci_low']:.6f}, {comparison['auprc_difference_ci_high']:.6f}]",
        "",
        "Region-wise macro summary (no subject-region pooling):",
        macro.to_string(index=False),
        "",
        "RUN PARAMETERS",
        f"RQ1 within-group permutations per region: {args.permutations}",
        f"Subject-level/decomposition bootstrap replicates: {args.bootstrap}",
        f"Region-level bootstrap replicates: {args.regional_bootstrap}",
        f"Random seed: {args.seed}",
        f"Output directory: {output_dir}",
    ]
    summary = "\n".join(lines) + "\n"
    (output_dir / "analysis_1_to_4_summary.txt").write_text(summary, encoding="utf-8")
    return summary


def make_synthetic_data(subjects: int, seed: int) -> pd.DataFrame:
    """Generate data for the self-test."""
    if subjects % 5 != 0:
        raise ValueError("Synthetic subject count must be divisible by five")
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(5), subjects // 5)
    rng.shuffle(groups)
    rows: dict[str, object] = {
        "subject_id": [f"TEST_{index:04d}" for index in range(subjects)],
        "group": groups,
        "models_used": ["|".join(str(model) for model in range(5) if model != group) for group in groups],
    }
    latent_error = np.clip(rng.beta(2.5, 4.5, size=subjects) + 0.02 * (groups - 2), 0, 0.95)
    subject_dice = np.clip(1.0 - latent_error, 0.05, 0.98)
    total_subject_entropy = np.clip(0.18 + 0.75 * latent_error + rng.normal(0, 0.07, subjects), 0.01, None)
    mme_subject = total_subject_entropy * np.clip(rng.normal(0.82, 0.025, subjects), 0.70, 0.92)
    mi_subject = total_subject_entropy - mme_subject
    msp_subject = np.clip(total_subject_entropy / 3.0 + rng.normal(0, 0.02, subjects), 0.001, 0.99)

    for region_index, region in enumerate(TARGETS):
        jitter = rng.normal(0, 0.05, subjects)
        dice = np.clip(subject_dice + 0.03 * math.sin(region_index) + jitter, 0, 1)
        total = np.clip(total_subject_entropy + 0.02 * math.cos(region_index) + rng.normal(0, 0.02, subjects), 0.001, None)
        share = np.clip(0.15 + rng.normal(0, 0.015, subjects), 0.08, 0.25)
        mi = total * share
        mme = total - mi
        msp = np.clip(msp_subject + rng.normal(0, 0.01, subjects), 0.0001, 0.999)
        rows[f"{region}_dice"] = dice
        rows[f"{region}_entropy_mean"] = total
        rows[f"{region}_mean_member_entropy_mean"] = mme
        rows[f"{region}_mutual_information_mean"] = mi
        rows[f"{region}_msp_unc_mean"] = msp
        rows[f"{region}_pred_entropy_mean"] = total + rng.normal(0, 0.01, subjects)
        rows[f"{region}_pred_msp_unc_mean"] = np.clip(msp + rng.normal(0, 0.005, subjects), 0, 1)

    rows["predtgt_n_voxels"] = rng.integers(1000, 5000, subjects)
    rows["predfg_n_voxels"] = rng.integers(5000, 20000, subjects)
    rows["predtgt_entropy_mean"] = total_subject_entropy
    rows["predtgt_mean_member_entropy_mean"] = mme_subject
    rows["predtgt_mutual_information_mean"] = mi_subject
    rows["predtgt_msp_unc_mean"] = msp_subject
    rows["predfg_entropy_mean"] = total_subject_entropy + rng.normal(0, 0.03, subjects)
    rows["predfg_msp_unc_mean"] = np.clip(msp_subject + rng.normal(0, 0.015, subjects), 0, 1)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", nargs="?", type=Path)
    parser.add_argument("output_dir", nargs="?", type=Path)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--regional-bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--expected-subjects", type=int, default=532)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a fast synthetic smoke test instead of reading a CSV",
    )
    args = parser.parse_args()
    if not args.self_test and (args.input_csv is None or args.output_dir is None):
        parser.error("input_csv and output_dir are required unless --self-test is used")
    if args.permutations < 99 or args.bootstrap < 99 or args.regional_bootstrap < 99:
        parser.error("Permutation/bootstrap counts must each be at least 99")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        args.expected_subjects = 100
        args.permutations = min(args.permutations, 199)
        args.bootstrap = min(args.bootstrap, 199)
        args.regional_bootstrap = min(args.regional_bootstrap, 99)
        input_path = Path("SYNTHETIC_SELF_TEST")
        output_dir = args.output_dir or args.input_csv or Path("synthetic_analysis_1_to_4")
        data = make_synthetic_data(args.expected_subjects, args.seed)
    else:
        input_path = args.input_csv.resolve()
        output_dir = args.output_dir.resolve()
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        data = pd.read_csv(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    audit = validate_input(data, args.expected_subjects)
    groups = np.asarray(audit["groups"], dtype=int)

    _, _, group_tests = group_diagnostic(data, output_dir)
    rq1, _ = rq1_group_sensitivity(
        data,
        groups,
        args.permutations,
        rng,
        output_dir,
    )
    decomposition = decomposition_analysis(
        data,
        args.bootstrap,
        rng,
        output_dir,
    )
    failure = failure_detection_analysis(
        data,
        groups,
        args.bootstrap,
        args.regional_bootstrap,
        rng,
        output_dir,
    )

    metadata = {
        "input": str(input_path),
        "output_directory": str(output_dir),
        "subjects": len(data),
        "target_regions": TARGETS,
        "random_seed": args.seed,
        "rq1_permutations": args.permutations,
        "subject_bootstrap_replicates": args.bootstrap,
        "regional_bootstrap_replicates": args.regional_bootstrap,
        "primary_failure_threshold": 0.50,
        "failure_score": "predtgt_entropy_mean",
        "baseline_score": "predtgt_msp_unc_mean",
        "audit": {
            "group_sizes": audit["group_sizes"],
            "maximum_decomposition_error": audit["maximum_decomposition_error"],
        },
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = write_summary(
        input_path,
        output_dir,
        audit,
        group_tests,
        rq1,
        decomposition,
        failure,
        args,
    )
    print(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
