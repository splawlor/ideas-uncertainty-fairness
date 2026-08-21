#!/usr/bin/env python3
"""Run the subject-level risk--coverage analysis."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


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

SCORES = {
    "Predictive entropy": "predtgt_entropy_mean",
    "Maximum-softmax uncertainty": "predtgt_msp_unc_mean",
}

DEFAULT_COVERAGES = tuple(np.arange(1.00, 0.49, -0.05).round(2))
SUMMARY_COVERAGES = (1.00, 0.75, 0.50)


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha], axis=0)
    return np.asarray(low, dtype=float), np.asarray(high, dtype=float)


def coverage_counts(coverages: np.ndarray, subjects: int) -> np.ndarray:
    """Return retained counts, using ceil so actual coverage is never below target."""
    counts = np.ceil(coverages * subjects - 1e-12).astype(int)
    return np.clip(counts, 1, subjects)


def retained_mean_dice(
    subject_mean_dice: np.ndarray,
    uncertainty_score: np.ndarray,
    retained_counts: np.ndarray,
) -> np.ndarray:
    """Rank low-to-high uncertainty and evaluate prefix means at each coverage."""
    order = np.argsort(uncertainty_score, kind="mergesort")
    cumulative_dice = np.cumsum(subject_mean_dice[order], dtype=float)
    return cumulative_dice[retained_counts - 1] / retained_counts


def parse_coverages(text: str) -> np.ndarray:
    try:
        values = np.asarray([float(value.strip()) for value in text.split(",")], dtype=float)
    except ValueError as error:
        raise argparse.ArgumentTypeError("coverages must be comma-separated numbers") from error
    if values.ndim != 1 or len(values) < 2:
        raise argparse.ArgumentTypeError("at least two coverage levels are required")
    if not np.isfinite(values).all() or np.any((values < 0.50) | (values > 1.00)):
        raise argparse.ArgumentTypeError("coverage levels must lie from 0.50 through 1.00")
    if len(np.unique(values)) != len(values):
        raise argparse.ArgumentTypeError("coverage levels must be unique")
    if not np.any(np.isclose(values, 1.00)) or not np.any(np.isclose(values, 0.50)):
        raise argparse.ArgumentTypeError("coverage levels must include 1.00 and 0.50")
    return np.sort(values)[::-1]


def validate_input(data: pd.DataFrame, expected_subjects: int) -> dict[str, object]:
    dice_columns = [f"{region}_dice" for region in TARGETS]
    required = ["subject_id", *dice_columns, *SCORES.values()]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    if len(data) != expected_subjects:
        raise RuntimeError(f"Expected {expected_subjects} subjects, found {len(data)}")
    if data["subject_id"].isna().any():
        raise RuntimeError("subject_id contains missing values")
    if data["subject_id"].duplicated().any():
        duplicates = data.loc[data["subject_id"].duplicated(), "subject_id"].tolist()
        raise RuntimeError(f"subject_id contains duplicates: {duplicates[:5]}")

    numeric = data[[*dice_columns, *SCORES.values()]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise RuntimeError(f"Required numerical columns contain missing/non-numeric values: {bad}")
    values = numeric.to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError("Required numerical columns contain non-finite values")

    dice = numeric[dice_columns].to_numpy(float)
    if np.any((dice < 0.0) | (dice > 1.0)):
        raise RuntimeError("Dice values must lie in [0, 1]")
    entropy = numeric[SCORES["Predictive entropy"]].to_numpy(float)
    if np.any(entropy < 0.0):
        raise RuntimeError("Predictive entropy contains negative values")
    msp = numeric[SCORES["Maximum-softmax uncertainty"]].to_numpy(float)
    if np.any((msp < 0.0) | (msp > 1.0)):
        raise RuntimeError("Maximum-softmax uncertainty must lie in [0, 1]")

    if "predtgt_n_voxels" in data.columns:
        voxels = pd.to_numeric(data["predtgt_n_voxels"], errors="coerce").to_numpy(float)
        if not np.isfinite(voxels).all() or np.any(voxels <= 0):
            raise RuntimeError("All subjects must have at least one predicted target voxel")

    subject_mean_dice = dice.mean(axis=1)
    score_arrays = {
        label: numeric[column].to_numpy(float)
        for label, column in SCORES.items()
    }
    return {
        "dice_columns": dice_columns,
        "subject_mean_dice": subject_mean_dice,
        "scores": score_arrays,
        "mean_target_dice": float(subject_mean_dice.mean()),
    }


def bootstrap_curves(
    subject_mean_dice: np.ndarray,
    scores: dict[str, np.ndarray],
    retained_counts: np.ndarray,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Paired subject bootstrap; ranking is recomputed within every resample."""
    rng = np.random.default_rng(seed)
    score_names = list(scores)
    result = np.empty((replicates, len(score_names), len(retained_counts)), dtype=float)
    subjects = len(subject_mean_dice)

    for replicate in range(replicates):
        indices = rng.integers(0, subjects, size=subjects)
        sampled_dice = subject_mean_dice[indices]
        for score_index, score_name in enumerate(score_names):
            sampled_score = scores[score_name][indices]
            result[replicate, score_index] = retained_mean_dice(
                sampled_dice,
                sampled_score,
                retained_counts,
            )
    return result


def build_tables(
    subject_mean_dice: np.ndarray,
    scores: dict[str, np.ndarray],
    coverages: np.ndarray,
    retained_counts: np.ndarray,
    bootstrap_mean_dice: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_names = list(scores)
    actual_coverages = retained_counts / len(subject_mean_dice)
    point_mean_dice = np.vstack(
        [
            retained_mean_dice(subject_mean_dice, scores[name], retained_counts)
            for name in score_names
        ]
    )
    boot_risk = 1.0 - bootstrap_mean_dice

    dice_low, dice_high = percentile_interval(bootstrap_mean_dice)
    risk_low, risk_high = percentile_interval(boot_risk)

    curve_rows: list[dict[str, object]] = []
    for score_index, score_name in enumerate(score_names):
        for coverage_index, target_coverage in enumerate(coverages):
            mean_dice = float(point_mean_dice[score_index, coverage_index])
            curve_rows.append(
                {
                    "score": score_name,
                    "score_column": SCORES[score_name],
                    "target_coverage": float(target_coverage),
                    "actual_coverage": float(actual_coverages[coverage_index]),
                    "retained_subjects": int(retained_counts[coverage_index]),
                    "total_subjects": int(len(subject_mean_dice)),
                    "retained_mean_dice": mean_dice,
                    "retained_mean_dice_ci_low": float(dice_low[score_index, coverage_index]),
                    "retained_mean_dice_ci_high": float(dice_high[score_index, coverage_index]),
                    "selective_risk_one_minus_dice": 1.0 - mean_dice,
                    "selective_risk_ci_low": float(risk_low[score_index, coverage_index]),
                    "selective_risk_ci_high": float(risk_high[score_index, coverage_index]),
                    "dice_gain_vs_full_coverage": float(
                        mean_dice - point_mean_dice[score_index, 0]
                    ),
                }
            )
    curve = pd.DataFrame(curve_rows)

    entropy_index = score_names.index("Predictive entropy")
    msp_index = score_names.index("Maximum-softmax uncertainty")
    dice_difference = point_mean_dice[entropy_index] - point_mean_dice[msp_index]
    boot_dice_difference = (
        bootstrap_mean_dice[:, entropy_index, :]
        - bootstrap_mean_dice[:, msp_index, :]
    )
    boot_risk_difference = -boot_dice_difference
    dice_diff_low, dice_diff_high = percentile_interval(boot_dice_difference)
    risk_diff_low, risk_diff_high = percentile_interval(boot_risk_difference)

    comparison = pd.DataFrame(
        {
            "target_coverage": coverages,
            "actual_coverage": actual_coverages,
            "retained_subjects": retained_counts,
            "total_subjects": len(subject_mean_dice),
            "mean_dice_difference_entropy_minus_msp": dice_difference,
            "mean_dice_difference_ci_low": dice_diff_low,
            "mean_dice_difference_ci_high": dice_diff_high,
            "risk_difference_entropy_minus_msp": -dice_difference,
            "risk_difference_ci_low": risk_diff_low,
            "risk_difference_ci_high": risk_diff_high,
        }
    )

    summary_mask = curve["target_coverage"].apply(
        lambda value: any(np.isclose(value, target) for target in SUMMARY_COVERAGES)
    )
    summary = curve.loc[summary_mask].reset_index(drop=True)
    return curve, comparison, summary


def make_figure(curve: pd.DataFrame, output_dir: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "matplotlib-risk-coverage-cache"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "Predictive entropy": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
        "Maximum-softmax uncertainty": {
            "color": "#D55E00",
            "linestyle": "--",
            "marker": "s",
        },
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for score_name, style in styles.items():
        rows = curve[curve["score"] == score_name].sort_values("actual_coverage")
        x = rows["actual_coverage"].to_numpy(float) * 100.0
        risk = rows["selective_risk_one_minus_dice"].to_numpy(float)
        low = rows["selective_risk_ci_low"].to_numpy(float)
        high = rows["selective_risk_ci_high"].to_numpy(float)
        axis.plot(
            x,
            risk,
            linewidth=2.1,
            markersize=4.5,
            label=score_name,
            **style,
        )
        axis.fill_between(x, low, high, color=style["color"], alpha=0.14, linewidth=0)

    full_risk = float(
        curve.loc[np.isclose(curve["target_coverage"], 1.0), "selective_risk_one_minus_dice"].iloc[0]
    )
    axis.axhline(
        full_risk,
        color="#555555",
        linestyle=":",
        linewidth=1.2,
        label=f"No-referral risk ({full_risk:.3f})",
    )
    axis.set_xlabel("Coverage: subjects retained (%)")
    axis.set_ylabel("Selective risk (1 - mean 15-region Dice)")
    figure.suptitle(
        "Risk–coverage for label-free subject triage",
        x=0.105,
        y=0.985,
        ha="left",
        va="top",
        fontsize=12,
    )
    figure.text(
        0.105,
        0.935,
        "Leakage-free four-model ensemble; lower risk is better; shaded bands are 95% subject-bootstrap CIs",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#444444",
    )
    axis.set_xlim(49, 101)
    axis.set_xticks(np.arange(50, 101, 10))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=8.5, loc="best")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure.savefig(output_dir / "risk_coverage_curve.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "risk_coverage_curve.pdf", bbox_inches="tight")
    plt.close(figure)


def write_summary(
    input_path: Path,
    output_dir: Path,
    curve: pd.DataFrame,
    comparison: pd.DataFrame,
    bootstrap_replicates: int,
    seed: int,
) -> str:
    lines = [
        "DISJOINT-ENSEMBLE LABEL-FREE RISK–COVERAGE ANALYSIS",
        "=" * 72,
        f"Input: {input_path}",
        f"Subjects: {int(curve['total_subjects'].iloc[0])}",
        "Outcome: subject mean Dice across 15 prespecified target regions",
        "Selective risk: 1 - retained mean Dice",
        "Selection rule: retain the least-uncertain subjects separately for each score",
        f"Paired subject bootstrap replicates: {bootstrap_replicates}",
        f"Random seed: {seed}",
        "",
        "PRIMARY SUMMARY POINTS",
    ]
    for coverage in SUMMARY_COVERAGES:
        lines.append(f"Coverage target: {coverage:.0%}")
        selected = curve[np.isclose(curve["target_coverage"], coverage)]
        for _, row in selected.iterrows():
            lines.append(
                f"  {row['score']}: retained {int(row['retained_subjects'])}/"
                f"{int(row['total_subjects'])}; mean Dice={row['retained_mean_dice']:.6f} "
                f"[{row['retained_mean_dice_ci_low']:.6f}, {row['retained_mean_dice_ci_high']:.6f}]; "
                f"risk={row['selective_risk_one_minus_dice']:.6f} "
                f"[{row['selective_risk_ci_low']:.6f}, {row['selective_risk_ci_high']:.6f}]"
            )
        difference = comparison[np.isclose(comparison["target_coverage"], coverage)].iloc[0]
        lines.append(
            "  Entropy - maximum-softmax risk difference: "
            f"{difference['risk_difference_entropy_minus_msp']:.6f} "
            f"[{difference['risk_difference_ci_low']:.6f}, "
            f"{difference['risk_difference_ci_high']:.6f}]"
        )
        lines.append("")
    lines.extend(
        [
            "INTERPRETATION RULE",
            "A useful uncertainty ranking produces lower selective risk as coverage decreases.",
            "A negative entropy-minus-maximum-softmax risk difference favours predictive entropy.",
            "Confidence intervals are descriptive uncertainty intervals for the retained-cohort mean;",
            "they do not turn the coverage grid into a family of confirmatory hypothesis tests.",
            "",
            f"Output directory: {output_dir}",
        ]
    )
    summary = "\n".join(lines) + "\n"
    (output_dir / "risk_coverage_summary.txt").write_text(summary, encoding="utf-8")
    return summary


def make_synthetic_data(subjects: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latent_error = np.clip(rng.beta(2.7, 4.3, subjects), 0.01, 0.95)
    rows: dict[str, object] = {
        "subject_id": [f"TEST_{index:04d}" for index in range(subjects)],
    }
    regional_dice = []
    for region_index, region in enumerate(TARGETS):
        dice = np.clip(
            1.0 - latent_error + 0.025 * math.sin(region_index) + rng.normal(0, 0.045, subjects),
            0.0,
            1.0,
        )
        rows[f"{region}_dice"] = dice
        regional_dice.append(dice)
    mean_error = 1.0 - np.column_stack(regional_dice).mean(axis=1)
    rows["predtgt_entropy_mean"] = np.clip(
        0.15 + 0.85 * mean_error + rng.normal(0, 0.08, subjects),
        0.0,
        None,
    )
    rows["predtgt_msp_unc_mean"] = np.clip(
        0.05 + 0.45 * mean_error + rng.normal(0, 0.06, subjects),
        0.0,
        1.0,
    )
    rows["predtgt_n_voxels"] = rng.integers(1_000, 8_000, subjects)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", nargs="?", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--expected-subjects", type=int, default=532)
    parser.add_argument(
        "--coverages",
        type=parse_coverages,
        default=np.asarray(DEFAULT_COVERAGES, dtype=float),
        help="Comma-separated coverage levels; default: 1.00,0.95,...,0.50",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Use a deterministic synthetic dataset for an implementation smoke test",
    )
    args = parser.parse_args()
    if args.bootstrap < 99:
        parser.error("--bootstrap must be at least 99")
    if not args.self_test and args.input_csv is None:
        parser.error("input_csv is required unless --self-test is used")
    if not args.self_test and args.output_dir is None:
        parser.error("--output-dir is required unless --self-test is used")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        expected_subjects = min(args.expected_subjects, 200)
        data = make_synthetic_data(expected_subjects, args.seed)
        input_path = Path("SYNTHETIC_SELF_TEST")
        output_dir = (args.output_dir or Path("risk_coverage_self_test")).resolve()
    else:
        expected_subjects = args.expected_subjects
        input_path = args.input_csv.resolve()
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        data = pd.read_csv(input_path)
        output_dir = args.output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    audit = validate_input(data, expected_subjects)
    subject_mean_dice = np.asarray(audit["subject_mean_dice"], dtype=float)
    scores = {
        name: np.asarray(values, dtype=float)
        for name, values in dict(audit["scores"]).items()
    }
    coverages = np.asarray(args.coverages, dtype=float)
    retained_counts = coverage_counts(coverages, len(subject_mean_dice))

    bootstrap_mean_dice = bootstrap_curves(
        subject_mean_dice,
        scores,
        retained_counts,
        args.bootstrap,
        args.seed,
    )
    curve, comparison, summary_table = build_tables(
        subject_mean_dice,
        scores,
        coverages,
        retained_counts,
        bootstrap_mean_dice,
    )

    full_rows = curve[np.isclose(curve["target_coverage"], 1.0)]
    if len(full_rows) != len(SCORES):
        raise RuntimeError("Expected one full-coverage row per uncertainty score")
    if not np.allclose(full_rows["retained_mean_dice"], audit["mean_target_dice"], atol=1e-12):
        raise RuntimeError("Full-coverage Dice does not match the directly computed cohort mean")
    if not np.isclose(
        full_rows["selective_risk_one_minus_dice"].max(),
        full_rows["selective_risk_one_minus_dice"].min(),
        atol=1e-12,
    ):
        raise RuntimeError("The two scores disagree at full coverage")
    if not np.allclose(
        curve["selective_risk_one_minus_dice"],
        1.0 - curve["retained_mean_dice"],
        atol=1e-12,
    ):
        raise RuntimeError("Risk identity check failed")

    curve.to_csv(output_dir / "risk_coverage_curve.csv", index=False)
    comparison.to_csv(output_dir / "risk_coverage_entropy_vs_msp.csv", index=False)
    summary_table.to_csv(output_dir / "risk_coverage_primary_points.csv", index=False)
    pd.DataFrame(
        {
            "subject_id": data["subject_id"].astype(str),
            "subject_mean_dice_15": subject_mean_dice,
            **{SCORES[name]: scores[name] for name in SCORES},
        }
    ).to_csv(output_dir / "risk_coverage_subject_values.csv", index=False)
    make_figure(curve, output_dir)
    summary = write_summary(
        input_path,
        output_dir,
        curve,
        comparison,
        args.bootstrap,
        args.seed,
    )

    metadata = {
        "analysis": "label-free subject-level risk-coverage",
        "input": str(input_path),
        "script": str(Path(__file__).resolve()),
        "subjects": len(data),
        "target_regions": TARGETS,
        "outcome": "subject mean Dice across 15 target regions",
        "risk": "1 - retained mean Dice",
        "selection": "retain least-uncertain subjects separately for each score",
        "scores": SCORES,
        "target_coverages": coverages.tolist(),
        "retained_subjects": retained_counts.tolist(),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_unit": "subject; paired across uncertainty scores",
        "random_seed": args.seed,
        "confidence_interval": "95% percentile subject bootstrap",
        "full_coverage_mean_dice": float(audit["mean_target_dice"]),
        "validation": {
            "unique_subject_ids": True,
            "complete_required_values": True,
            "dice_in_unit_interval": True,
            "full_coverage_scores_identical": True,
            "risk_identity_verified": True,
        },
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "risk_coverage_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary, end="")


if __name__ == "__main__":
    main()
