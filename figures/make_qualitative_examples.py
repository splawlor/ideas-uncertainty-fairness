#!/usr/bin/env python3
"""Select and render four qualitative segmentation examples."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-disjoint-qualitative")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
import pandas as pd


TARGET_LABELS = {
    "L-Hippocampus": 13,
    "R-Hippocampus": 29,
    "L-Amygdala": 14,
    "R-Amygdala": 30,
    "L-Thalamus": 6,
    "R-Thalamus": 25,
    "L-Caudate": 7,
    "R-Caudate": 26,
    "L-Putamen": 8,
    "R-Putamen": 27,
    "L-Pallidum": 9,
    "R-Pallidum": 28,
    "L-Cerebral-WM": 1,
    "R-Cerebral-WM": 20,
    "Brain-Stem": 12,
}

ROLE_ORDER = [
    "expected_success",
    "detected_failure",
    "false_alarm",
    "confident_failure",
]
ROLE_SELECTION_ORDER = [
    "confident_failure",
    "false_alarm",
    "detected_failure",
    "expected_success",
]
ROLE_DISPLAY = {
    "expected_success": "High Dice, low entropy",
    "detected_failure": "Low Dice, high entropy",
    "false_alarm": "High Dice, unexpectedly high entropy",
    "confident_failure": "Low Dice, unexpectedly low entropy",
}

EXPECTED_GROUP_COUNTS = {0: 107, 1: 107, 2: 106, 3: 106, 4: 106}
EXPECTED_SUBJECTS = 532
HIGH_DICE_MINIMUM = 0.80
LOW_DICE_RANGE = (0.20, 0.60)
HIGH_PERCENTILE = 0.80
LOW_PERCENTILE = 0.20

PREFERRED_SUBJECT_MEAN_DICE = 0.55
RELAXED_SUBJECT_MEAN_DICE = 0.50
FALLBACK_SUBJECT_MEAN_DICE = 0.45
PREFERRED_CENTROID_DISTANCE_MM = 15.0
RELAXED_CENTROID_DISTANCE_MM = 25.0
FALLBACK_CENTROID_DISTANCE_MM = 35.0
PREFERRED_VOLUME_RATIO = (0.50, 2.00)
RELAXED_VOLUME_RATIO = (0.33, 3.00)
FALLBACK_VOLUME_RATIO = (0.25, 4.00)

ENTROPY_VMIN = 0.0
ENTROPY_VMAX = 1.3
ENTROPY_TICKS = [0.0, 0.5, 1.0, ENTROPY_VMAX]
GRID_ATOL_VOXELS = 1e-4
REFERENCE_COLOUR = "#78C679"
PREDICTION_COLOUR = "#00BFC4"
MASK_COLOUR = "#F28E2B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ideas-root",
        type=Path,
        default=Path(os.environ.get("IDEAS_ROOT", ".")),
    )
    parser.add_argument(
        "--uncertainty-dir",
        type=Path,
        required=True,
        help="Final disjoint directory containing the CSV and *_seg/_entropy NIfTIs",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--top-n-qc", type=int, default=5)
    parser.add_argument("--shortlist-per-role", type=int, default=80)
    parser.add_argument(
        "--exclude-subject",
        action="append",
        default=[],
        help="Subject to exclude for figure-level diversity; may be repeated",
    )
    return parser.parse_args()


def subject_paths(
    ideas_root: Path, uncertainty_dir: Path, subject_id: str
) -> dict[str, Path]:
    return {
        "raw": ideas_root
        / "nnunet_raw/Dataset001_IDEAS/imagesTr"
        / f"{subject_id}_0000.nii.gz",
        "reference": ideas_root
        / "nnunet_raw/Dataset001_IDEAS/labelsTr"
        / f"{subject_id}.nii.gz",
        "prediction": uncertainty_dir / f"{subject_id}_seg.nii.gz",
        "entropy": uncertainty_dir / f"{subject_id}_entropy.nii.gz",
    }


def validate_input_inventory(
    ideas_root: Path, uncertainty_dir: Path, table: pd.DataFrame
) -> dict[str, int]:
    counts = {
        "raw": len(
            list(
                (ideas_root / "nnunet_raw/Dataset001_IDEAS/imagesTr").glob(
                    "*_0000.nii.gz"
                )
            )
        ),
        "reference": len(
            list(
                (ideas_root / "nnunet_raw/Dataset001_IDEAS/labelsTr").glob(
                    "*.nii.gz"
                )
            )
        ),
        "prediction": len(list(uncertainty_dir.glob("*_seg.nii.gz"))),
        "entropy": len(list(uncertainty_dir.glob("*_entropy.nii.gz"))),
    }
    if set(counts.values()) != {EXPECTED_SUBJECTS}:
        raise RuntimeError(
            f"Expected {EXPECTED_SUBJECTS} raw/reference/prediction/entropy files; "
            f"found {counts}"
        )
    if len(table) != EXPECTED_SUBJECTS or table["subject_id"].nunique() != EXPECTED_SUBJECTS:
        raise RuntimeError(f"Unexpected subject table dimensions: {table.shape}")
    if table["subject_id"].duplicated().any():
        raise RuntimeError("Duplicate subject identifiers in the final disjoint CSV")

    groups = table["group"].astype(int).value_counts().sort_index().to_dict()
    if groups != EXPECTED_GROUP_COUNTS:
        raise RuntimeError(f"Unexpected disjoint group counts: {groups}")
    for row in table[["group", "models_used"]].itertuples(index=False):
        models = {int(value) for value in str(row.models_used).split("|")}
        if len(models) != 4 or int(row.group) in models or models != (set(range(5)) - {int(row.group)}):
            raise RuntimeError(
                f"Invalid model subset for group {row.group}: {row.models_used}"
            )
    return counts


def build_long_table(wide: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_id", "group", "models_used"}
    for region in TARGET_LABELS:
        required.update(
            {
                f"{region}_n_voxels",
                f"{region}_entropy_mean",
                f"{region}_dice",
            }
        )
    missing = sorted(required - set(wide.columns))
    if missing:
        raise RuntimeError(f"Final disjoint CSV is missing columns: {missing}")

    frames = []
    for region, label in TARGET_LABELS.items():
        frames.append(
            pd.DataFrame(
                {
                    "subject_id": wide["subject_id"].astype(str),
                    "group": wide["group"].astype(int),
                    "models_used": wide["models_used"].astype(str),
                    "region": region,
                    "region_id": label,
                    "reference_voxels": wide[f"{region}_n_voxels"],
                    "dice": wide[f"{region}_dice"],
                    "entropy": wide[f"{region}_entropy_mean"],
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)
    expected_rows = EXPECTED_SUBJECTS * len(TARGET_LABELS)
    if len(table) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} target rows; found {len(table)}")
    if table.duplicated(["subject_id", "region"]).any():
        raise RuntimeError("Duplicate subject-region rows in candidate table")
    if table[["dice", "entropy", "reference_voxels"]].isna().any().any():
        raise RuntimeError("Missing final Dice, entropy, or reference-voxel values")
    if not table["dice"].between(0.0, 1.0).all():
        raise RuntimeError("Dice values outside [0, 1]")
    if (table["entropy"] < 0).any() or (table["reference_voxels"] <= 0).any():
        raise RuntimeError("Negative entropy or non-positive target reference volume")

    table["dice_pct"] = table.groupby("region")["dice"].rank(
        method="average", pct=True
    )
    table["entropy_pct"] = table.groupby("region")["entropy"].rank(
        method="average", pct=True
    )
    table["error_pct"] = 1.0 - table["dice_pct"]
    table["subject_mean_dice"] = table.groupby("subject_id")["dice"].transform(
        "mean"
    )
    return table


def role_mask(table: pd.DataFrame, role: str) -> pd.Series:
    high_dice = (table["dice"] >= HIGH_DICE_MINIMUM) & (
        table["dice_pct"] >= HIGH_PERCENTILE
    )
    low_dice = table["dice"].between(*LOW_DICE_RANGE) & (
        table["dice_pct"] <= LOW_PERCENTILE
    )
    high_entropy = table["entropy_pct"] >= HIGH_PERCENTILE
    low_entropy = table["entropy_pct"] <= LOW_PERCENTILE
    masks = {
        "expected_success": high_dice & low_entropy,
        "detected_failure": low_dice & high_entropy,
        "false_alarm": high_dice & high_entropy,
        "confident_failure": low_dice & low_entropy,
    }
    return masks[role]


def role_score(table: pd.DataFrame, role: str) -> pd.Series:
    scores = {
        "expected_success": table["dice_pct"] + (1.0 - table["entropy_pct"]),
        "detected_failure": table["error_pct"] + table["entropy_pct"],
        "false_alarm": table["dice_pct"] + table["entropy_pct"],
        "confident_failure": table["error_pct"] + (1.0 - table["entropy_pct"]),
    }
    return scores[role]


def load_canonical_images(paths: dict[str, Path]) -> dict[str, nib.spatialimages.SpatialImage]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required NIfTI files:\n  " + "\n  ".join(missing))
    return {key: nib.as_closest_canonical(nib.load(path)) for key, path in paths.items()}


def validate_image_grids(
    images: dict[str, nib.spatialimages.SpatialImage], subject_id: str
) -> tuple[tuple[int, ...], dict[str, float]]:
    shapes = {tuple(image.shape) for image in images.values()}
    if len(shapes) != 1:
        raise RuntimeError(f"Canonical shape mismatch for {subject_id}: {shapes}")
    shape = next(iter(shapes))
    if len(shape) != 3:
        raise RuntimeError(f"Expected 3D arrays for {subject_id}; found {shape}")
    reference_affine = images["reference"].affine
    deviations = {}
    for key, image in images.items():
        transform = np.linalg.solve(reference_affine, image.affine)
        deviations[key] = float(np.max(np.abs(transform - np.eye(4))))
    failures = {key: value for key, value in deviations.items() if value > GRID_ATOL_VOXELS}
    if failures:
        detail = ", ".join(f"{key}={value:.6g}" for key, value in failures.items())
        raise RuntimeError(
            f"Voxel-grid mismatch for {subject_id} (>{GRID_ATOL_VOXELS:.1e} voxels): {detail}"
        )
    return shape, deviations


def spatial_metrics(
    ideas_root: Path, uncertainty_dir: Path, row: pd.Series
) -> dict[str, float]:
    paths = subject_paths(ideas_root, uncertainty_dir, str(row["subject_id"]))
    images = load_canonical_images(
        {"reference": paths["reference"], "prediction": paths["prediction"]}
    )
    validate_image_grids(images, str(row["subject_id"]))
    reference = np.asarray(images["reference"].dataobj)
    prediction = np.asarray(images["prediction"].dataobj)
    label = int(row["region_id"])
    reference_points = np.argwhere(reference == label)
    prediction_points = np.argwhere(prediction == label)
    if len(reference_points) == 0:
        raise RuntimeError(f"Reference label absent for {row['subject_id']}/{row['region']}")
    if len(prediction_points) == 0:
        return {
            "centroid_distance_mm": np.inf,
            "prediction_reference_volume_ratio": 0.0,
        }
    affine = images["reference"].affine
    reference_centroid = nib.affines.apply_affine(affine, reference_points.mean(axis=0))
    prediction_centroid = nib.affines.apply_affine(affine, prediction_points.mean(axis=0))
    return {
        "centroid_distance_mm": float(
            np.linalg.norm(reference_centroid - prediction_centroid)
        ),
        "prediction_reference_volume_ratio": float(
            len(prediction_points) / len(reference_points)
        ),
    }


def assign_spatial_rule(candidates: pd.DataFrame) -> pd.DataFrame:
    result = candidates.copy()
    ratio = result["prediction_reference_volume_ratio"]
    preferred = (
        (result["subject_mean_dice"] >= PREFERRED_SUBJECT_MEAN_DICE)
        & (result["centroid_distance_mm"] <= PREFERRED_CENTROID_DISTANCE_MM)
        & ratio.between(*PREFERRED_VOLUME_RATIO)
    )
    relaxed = (
        (result["subject_mean_dice"] >= RELAXED_SUBJECT_MEAN_DICE)
        & (result["centroid_distance_mm"] <= RELAXED_CENTROID_DISTANCE_MM)
        & ratio.between(*RELAXED_VOLUME_RATIO)
    )
    fallback = (
        (result["subject_mean_dice"] >= FALLBACK_SUBJECT_MEAN_DICE)
        & (result["centroid_distance_mm"] <= FALLBACK_CENTROID_DISTANCE_MM)
        & ratio.between(*FALLBACK_VOLUME_RATIO)
    )
    result["spatial_rule"] = np.select(
        [preferred, relaxed, fallback],
        ["preferred", "relaxed", "fallback"],
        default="excluded",
    )
    result["spatial_tier"] = result["spatial_rule"].map(
        {"preferred": 3, "relaxed": 2, "fallback": 1, "excluded": 0}
    )
    return result


def shortlist_role(
    table: pd.DataFrame,
    role: str,
    ideas_root: Path,
    uncertainty_dir: Path,
    maximum: int,
    excluded_subjects: set[str],
) -> pd.DataFrame:
    candidates = table.loc[role_mask(table, role)].copy()
    candidates["role"] = role
    candidates["role_score"] = role_score(candidates, role)
    candidates["excluded_for_figure_diversity"] = candidates["subject_id"].isin(
        excluded_subjects
    )
    candidates = candidates.loc[
        candidates["subject_mean_dice"] >= FALLBACK_SUBJECT_MEAN_DICE
    ].copy()
    candidates["subject_performance_tier"] = np.select(
        [
            candidates["subject_mean_dice"] >= PREFERRED_SUBJECT_MEAN_DICE,
            candidates["subject_mean_dice"] >= RELAXED_SUBJECT_MEAN_DICE,
        ],
        [3, 2],
        default=1,
    )
    candidates = candidates.sort_values(
        [
            "excluded_for_figure_diversity",
            "subject_performance_tier",
            "role_score",
            "subject_mean_dice",
            "subject_id",
            "region",
        ],
        ascending=[True, False, False, False, True, True],
    ).head(maximum)
    if candidates.empty:
        raise RuntimeError(f"No statistical candidates for role {role}")

    metrics = [
        spatial_metrics(ideas_root, uncertainty_dir, row)
        for _, row in candidates.iterrows()
    ]
    candidates = pd.concat(
        [candidates.reset_index(drop=True), pd.DataFrame(metrics)], axis=1
    )
    candidates = assign_spatial_rule(candidates)
    candidates = candidates.sort_values(
        [
            "excluded_for_figure_diversity",
            "spatial_tier",
            "role_score",
            "subject_mean_dice",
            "subject_id",
            "region",
        ],
        ascending=[True, False, False, False, True, True],
    ).reset_index(drop=True)
    candidates["candidate_rank"] = np.arange(1, len(candidates) + 1)
    if not (
        (candidates["spatial_tier"] > 0)
        & ~candidates["excluded_for_figure_diversity"]
    ).any():
        raise RuntimeError(
            f"No spatially eligible candidate for {role}; inspect its shortlist"
        )
    return candidates


def select_distinct_examples(shortlists: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pools = {}
    for role in ROLE_SELECTION_ORDER:
        pool = shortlists[role].loc[
            (shortlists[role]["spatial_tier"] > 0)
            & ~shortlists[role]["excluded_for_figure_diversity"]
        ].head(20)
        if pool.empty:
            raise RuntimeError(f"No eligible selection pool for {role}")
        pools[role] = [row for _, row in pool.iterrows()]

    best_rows = None
    best_key = None
    for combination in itertools.product(*(pools[role] for role in ROLE_SELECTION_ORDER)):
        subjects = {str(row["subject_id"]) for row in combination}
        regions = {str(row["region"]) for row in combination}
        if len(subjects) != 4 or len(regions) != 4:
            continue
        key = (
            sum(int(row["spatial_tier"]) for row in combination),
            sum(float(row["role_score"]) for row in combination),
            sum(float(row["subject_mean_dice"]) for row in combination),
            -sum(int(row["candidate_rank"]) for row in combination),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_rows = combination
    if best_rows is None:
        raise RuntimeError("Could not select four distinct subjects and regions")

    by_role = {str(row["role"]): row.copy() for row in best_rows}
    selected = pd.DataFrame([by_role[role] for role in ROLE_ORDER]).reset_index(drop=True)
    selected["selected_before_visual_qc"] = True
    return selected


def dice_for_label(prediction: np.ndarray, reference: np.ndarray, label: int) -> float:
    pred = prediction == label
    ref = reference == label
    denominator = int(pred.sum() + ref.sum())
    if denominator == 0:
        raise RuntimeError(f"Label {label} absent from prediction and reference")
    return float(2.0 * np.logical_and(pred, ref).sum() / denominator)


def largest_reference_coronal_slice(reference: np.ndarray, label: int) -> int:
    counts = (reference == label).sum(axis=(0, 2))
    index = int(np.argmax(counts))
    if counts[index] == 0:
        raise RuntimeError(f"Reference label {label} absent from every coronal slice")
    return index


def rotate_coronal(volume: np.ndarray, index: int) -> np.ndarray:
    return np.rot90(volume[:, index, :])


def crop_bounds(
    mask: np.ndarray, minimum: int = 96, padding: int = 18
) -> tuple[slice, slice]:
    points = np.argwhere(mask)
    if len(points) == 0:
        raise RuntimeError("Cannot crop an empty reference/prediction union")
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0) + 1
    y0, x0, y1, x1 = y0 - padding, x0 - padding, y1 + padding, x1 + padding
    height, width = mask.shape

    def expand(lo: int, hi: int, limit: int) -> tuple[int, int]:
        size = hi - lo
        if size < minimum:
            extra = minimum - size
            lo -= extra // 2
            hi += extra - extra // 2
        if lo < 0:
            hi -= lo
            lo = 0
        if hi > limit:
            lo -= hi - limit
            hi = limit
        return max(0, int(lo)), min(limit, int(hi))

    y0, y1 = expand(int(y0), int(y1), height)
    x0, x1 = expand(int(x0), int(x1), width)
    return slice(y0, y1), slice(x0, x1)


def display_limits(raw: np.ndarray) -> tuple[float, float]:
    values = raw[np.isfinite(raw) & (raw != 0)]
    if values.size == 0:
        values = raw[np.isfinite(raw)]
    if values.size == 0:
        raise RuntimeError("Raw MRI panel has no finite intensities")
    lo, hi = np.percentile(values, [1, 99])
    if not hi > lo:
        lo, hi = float(values.min()), float(values.max())
    if not hi > lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def prepare_case(
    row: pd.Series, ideas_root: Path, uncertainty_dir: Path
) -> tuple[dict[str, object], dict[str, object]]:
    subject_id = str(row["subject_id"])
    region = str(row["region"])
    label = int(row["region_id"])
    paths = subject_paths(ideas_root, uncertainty_dir, subject_id)
    images = load_canonical_images(paths)
    shape, deviations = validate_image_grids(images, subject_id)
    arrays = {key: np.asarray(image.dataobj) for key, image in images.items()}
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError(f"Non-finite values in a required NIfTI for {subject_id}")

    recomputed_dice = dice_for_label(arrays["prediction"], arrays["reference"], label)
    reference_mask_3d = arrays["reference"] == label
    recomputed_entropy = float(arrays["entropy"][reference_mask_3d].mean())
    if not np.isclose(recomputed_dice, float(row["dice"]), atol=1e-10, rtol=1e-8):
        raise RuntimeError(
            f"Dice mismatch for {subject_id}/{region}: "
            f"NIfTI={recomputed_dice}, CSV={row['dice']}"
        )
    if not np.isclose(recomputed_entropy, float(row["entropy"]), atol=1e-6, rtol=1e-6):
        raise RuntimeError(
            f"Entropy mismatch for {subject_id}/{region}: "
            f"NIfTI={recomputed_entropy}, CSV={row['entropy']}"
        )

    index = largest_reference_coronal_slice(arrays["reference"], label)
    slices = {key: rotate_coronal(array, index) for key, array in arrays.items()}
    ref_mask = slices["reference"] == label
    pred_mask = slices["prediction"] == label
    crop = crop_bounds(ref_mask | pred_mask)
    prepared = {key: value[crop] for key, value in slices.items()}
    prepared["ref_mask"] = ref_mask[crop]
    prepared["pred_mask"] = pred_mask[crop]
    prepared["slice_index"] = index
    prepared["display_limits"] = display_limits(prepared["raw"])
    audit = {
        "role": row["role"],
        "candidate_rank": int(row["candidate_rank"]),
        "subject_id": subject_id,
        "region": region,
        "region_id": label,
        "shape": "x".join(map(str, shape)),
        "canonical_coronal_slice": index,
        "dice_csv": float(row["dice"]),
        "dice_recomputed": recomputed_dice,
        "entropy_csv": float(row["entropy"]),
        "entropy_recomputed": recomputed_entropy,
        "max_grid_deviation_voxels": max(deviations.values()),
        "dice_matches": True,
        "entropy_matches": True,
        "grid_matches": True,
    }
    return prepared, audit


def plot_rows(
    rows: pd.DataFrame,
    prepared_cases: list[dict[str, object]],
    title: str,
    output_png: Path | None = None,
) -> plt.Figure:
    nrows = len(rows)
    fig, axes = plt.subplots(
        nrows,
        4,
        figsize=(12.8, 2.65 * nrows + 1.0),
        constrained_layout=True,
        squeeze=False,
    )
    column_titles = [
        "T1-weighted MRI",
        "Reference target",
        "Ensemble prediction",
        "Predictive entropy",
    ]
    for axis, column_title in zip(axes[0], column_titles):
        axis.set_title(column_title, fontweight="bold", pad=7)

    orange = ListedColormap([MASK_COLOUR])
    entropy_image = None
    for row_index, ((_, row), data) in enumerate(zip(rows.iterrows(), prepared_cases)):
        raw = data["raw"]
        vmin, vmax = data["display_limits"]
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])

        axes[row_index, 0].imshow(raw, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row_index, 1].imshow(raw, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row_index, 1].imshow(
            np.ma.masked_where(~data["ref_mask"], data["ref_mask"]),
            cmap=orange,
            alpha=0.72,
            interpolation="nearest",
        )
        axes[row_index, 2].imshow(raw, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row_index, 2].imshow(
            np.ma.masked_where(~data["pred_mask"], data["pred_mask"]),
            cmap=orange,
            alpha=0.72,
            interpolation="nearest",
        )
        entropy_image = axes[row_index, 3].imshow(
            data["entropy"],
            cmap="magma",
            vmin=ENTROPY_VMIN,
            vmax=ENTROPY_VMAX,
            interpolation="nearest",
        )
        if data["ref_mask"].any():
            axes[row_index, 3].contour(
                data["ref_mask"],
                levels=[0.5],
                colors=[REFERENCE_COLOUR],
                linewidths=1.1,
            )
        if data["pred_mask"].any():
            axes[row_index, 3].contour(
                data["pred_mask"],
                levels=[0.5],
                colors=[PREDICTION_COLOUR],
                linewidths=1.1,
            )

        role_label = ROLE_DISPLAY[str(row["role"])]
        row_text = (
            f"{chr(65 + row_index)}  {role_label}\n"
            f"{row['subject_id']}, {row['region']}\n"
            f"Dice={row['dice']:.3f}; entropy={row['entropy']:.3f}"
        )
        axes[row_index, 0].set_ylabel(
            row_text, rotation=0, ha="right", va="center", labelpad=12
        )

    colourbar = fig.colorbar(
        entropy_image,
        ax=axes[:, 3],
        fraction=0.028,
        pad=0.015,
        ticks=ENTROPY_TICKS,
        label="Predictive entropy",
    )
    colourbar.ax.set_yticklabels(["0.0", "0.5", "1.0", "1.3"])
    handles = [
        Line2D([0], [0], color=REFERENCE_COLOUR, lw=2, label="Reference contour"),
        Line2D([0], [0], color=PREDICTION_COLOUR, lw=2, label="Prediction contour"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    if output_png is not None:
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
    return fig


def prepare_rows(
    rows: pd.DataFrame, ideas_root: Path, uncertainty_dir: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prepared = []
    audits = []
    for _, row in rows.iterrows():
        case, audit = prepare_case(row, ideas_root, uncertainty_dir)
        prepared.append(case)
        audits.append(audit)
    return prepared, audits


def render_outputs(
    selected: pd.DataFrame,
    shortlists: dict[str, pd.DataFrame],
    ideas_root: Path,
    uncertainty_dir: Path,
    outdir: Path,
    top_n_qc: int,
) -> pd.DataFrame:
    selected_prepared, selected_audits = prepare_rows(
        selected, ideas_root, uncertainty_dir
    )
    fig = plot_rows(
        selected,
        selected_prepared,
        "Qualitative segmentation and uncertainty patterns",
        outdir / "qualitative_examples_disjoint_selected.png",
    )
    fig.savefig(
        outdir / "qualitative_examples_disjoint_selected.pdf", bbox_inches="tight"
    )
    plt.close(fig)

    all_audits = list(selected_audits)
    with PdfPages(outdir / "qualitative_examples_disjoint_qc.pdf") as pdf:
        for role in ROLE_ORDER:
            candidates = shortlists[role].loc[
                (shortlists[role]["spatial_tier"] > 0)
                & ~shortlists[role]["excluded_for_figure_diversity"]
            ].head(top_n_qc)
            selected_ids = set(
                zip(
                    selected.loc[selected["role"] == role, "subject_id"],
                    selected.loc[selected["role"] == role, "region"],
                )
            )
            present_ids = set(zip(candidates["subject_id"], candidates["region"]))
            if not selected_ids.issubset(present_ids):
                selected_row = selected.loc[selected["role"] == role]
                candidates = pd.concat([candidates, selected_row], ignore_index=True)
                candidates = candidates.drop_duplicates(["subject_id", "region"]).sort_values(
                    "candidate_rank"
                )
            prepared, audits = prepare_rows(candidates, ideas_root, uncertainty_dir)
            all_audits.extend(audits)
            role_slug = role.replace("_", "-")
            role_title = f"Visual-QC candidates: {ROLE_DISPLAY[role]}"
            role_fig = plot_rows(
                candidates,
                prepared,
                role_title,
                outdir / f"qc_{role_slug}.png",
            )
            pdf.savefig(role_fig, bbox_inches="tight")
            plt.close(role_fig)

    audit = pd.DataFrame(all_audits).drop_duplicates(
        ["role", "candidate_rank", "subject_id", "region"]
    )
    return audit.sort_values(["role", "candidate_rank"]).reset_index(drop=True)


def write_caption(selected: pd.DataFrame, outdir: Path) -> None:
    cases = []
    for index, row in selected.iterrows():
        cases.append(
            f"({chr(65 + index)}) {ROLE_DISPLAY[row['role']].lower()} "
            f"({row['subject_id']}, {row['region']}; Dice {row['dice']:.3f}, "
            f"mean entropy {row['entropy']:.3f})"
        )
    caption = (
        "Four qualitative examples from the leakage-free four-model disjoint ensemble: "
        + "; ".join(cases)
        + ". Each row shows the same cropped coronal slice as a T1-weighted MRI, "
        "the FreeSurfer-derived reference target, the ensemble prediction and predictive "
        "entropy. Orange indicates the selected structure; green and cyan contours show "
        "the reference and prediction, respectively. Entropy is displayed from 0 to 1.3. "
        "Cases were selected numerically using within-region Dice and entropy percentile "
        "bands, with the largest reference-region coronal cross-section used for display. "
        "They illustrate distinct uncertainty--error patterns but not their cohort frequency."
    )
    (outdir / "qualitative_examples_disjoint_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )
    latex = (
        "\\begin{figure*}[t]\n"
        "\\centering\n"
        "\\includegraphics[width=\\linewidth]{img/qualitative_examples_disjoint_selected.pdf}\n"
        f"\\caption{{{caption}}}\n"
        "\\label{fig:qualitative_examples}\n"
        "\\end{figure*}\n"
    )
    (outdir / "qualitative_examples_disjoint_insert.tex").write_text(
        latex, encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.top_n_qc < 1 or args.shortlist_per_role < args.top_n_qc:
        raise ValueError("Require shortlist-per-role >= top-n-qc >= 1")

    ideas_root = args.ideas_root.resolve()
    uncertainty_dir = args.uncertainty_dir.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = uncertainty_dir / "region_entropy_decomposed_reference_masked.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    wide = pd.read_csv(csv_path)
    inventory = validate_input_inventory(ideas_root, uncertainty_dir, wide)
    candidates = build_long_table(wide)
    excluded_subjects = set(map(str, args.exclude_subject))

    shortlists = {
        role: shortlist_role(
            candidates,
            role,
            ideas_root,
            uncertainty_dir,
            args.shortlist_per_role,
            excluded_subjects,
        )
        for role in ROLE_ORDER
    }
    selected = select_distinct_examples(shortlists)
    audit = render_outputs(
        selected,
        shortlists,
        ideas_root,
        uncertainty_dir,
        outdir,
        args.top_n_qc,
    )

    ranking_columns = [
        "role",
        "candidate_rank",
        "subject_id",
        "group",
        "models_used",
        "region_id",
        "region",
        "dice",
        "entropy",
        "dice_pct",
        "entropy_pct",
        "error_pct",
        "role_score",
        "subject_mean_dice",
        "centroid_distance_mm",
        "prediction_reference_volume_ratio",
        "spatial_rule",
        "spatial_tier",
        "excluded_for_figure_diversity",
    ]
    all_shortlists = pd.concat(
        [shortlists[role][ranking_columns] for role in ROLE_ORDER], ignore_index=True
    )
    all_shortlists.to_csv(outdir / "qualitative_disjoint_candidate_shortlists.csv", index=False)
    selected[ranking_columns + ["selected_before_visual_qc"]].to_csv(
        outdir / "qualitative_disjoint_selected_examples.csv", index=False
    )
    audit.to_csv(outdir / "qualitative_disjoint_array_verification.csv", index=False)
    write_caption(selected, outdir)

    metadata = {
        "input_csv": str(csv_path),
        "expected_subjects": EXPECTED_SUBJECTS,
        "target_regions": list(TARGET_LABELS),
        "input_file_counts": inventory,
        "excluded_subjects_for_figure_diversity": sorted(excluded_subjects),
        "selection_bands": {
            "high_dice_minimum": HIGH_DICE_MINIMUM,
            "low_dice_range": LOW_DICE_RANGE,
            "high_within_region_percentile": HIGH_PERCENTILE,
            "low_within_region_percentile": LOW_PERCENTILE,
        },
        "slice_rule": "largest reference-region coronal cross-section",
        "entropy_display_range": [ENTROPY_VMIN, ENTROPY_VMAX],
        "visual_qc_exclusions": [
            "gross acquisition artefact",
            "gross image/reference misregistration",
            "missing or unreadable required panel",
        ],
    }
    (outdir / "qualitative_disjoint_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "DISJOINT FOUR-CASE QUALITATIVE SELECTION CHECK",
        f"Input: {csv_path}",
        f"Subjects: {wide['subject_id'].nunique()}",
        f"Candidate subject-region rows: {len(candidates)}",
        f"Target regions: {len(TARGET_LABELS)}",
        f"Input file counts: {inventory}",
        f"Excluded subjects for figure diversity: {sorted(excluded_subjects)}",
        "Slice rule: largest reference-region coronal cross-section",
        f"Entropy display range: {ENTROPY_VMIN:.1f} to {ENTROPY_VMAX:.1f}",
        "",
        "Preselected examples before visual QC:",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"{ROLE_DISPLAY[row['role']]}: {row['subject_id']}, {row['region']}, "
            f"Dice={row['dice']:.6f}, entropy={row['entropy']:.6f}, "
            f"Dice percentile={row['dice_pct']:.3f}, "
            f"entropy percentile={row['entropy_pct']:.3f}, "
            f"subject mean Dice={row['subject_mean_dice']:.3f}, "
            f"centroid distance={row['centroid_distance_mm']:.2f} mm, "
            f"volume ratio={row['prediction_reference_volume_ratio']:.3f}, "
            f"spatial rule={row['spatial_rule']}, candidate rank={int(row['candidate_rank'])}"
        )
    lines.extend(
        [
            "",
            "All rendered Dice values matched direct NIfTI recomputation.",
            "All rendered entropy values matched direct reference-masked recomputation.",
            "All rendered raw, reference, prediction and entropy arrays matched in shape and voxel grid.",
            "Four distinct subjects and four distinct regions were selected.",
            "Poor segmentation and extreme uncertainty are not visual-QC exclusion criteria.",
            "Permissible exclusions: gross acquisition artefact, gross image/reference misregistration, or a missing/unreadable required panel.",
        ]
    )
    summary = "\n".join(lines) + "\n"
    (outdir / "qualitative_disjoint_summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
