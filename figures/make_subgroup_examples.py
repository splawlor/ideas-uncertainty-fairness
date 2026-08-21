#!/usr/bin/env python3
"""Select and render the subgroup examples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-disjoint-subgroups")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import make_qualitative_examples as base


EXPECTED_CLINICAL_SUBJECTS = 432
EXPECTED_OUTCOME_SUBJECTS = 417
PATHOLOGY_ORDER = ["HS", "OTHER", "DNT", "FCD", "CAV", "DUAL"]
PATHOLOGY_LABELS = {
    "HS": "HS",
    "OTHER": "Other (merged)",
    "DNT": "DNT",
    "FCD": "FCD",
    "CAV": "CAV",
    "DUAL": "Dual",
}
RARE_PATHOLOGY = {"GL", "TREBLE", "TBC"}

ENTROPY_VMIN = base.ENTROPY_VMIN
ENTROPY_VMAX = base.ENTROPY_VMAX
ENTROPY_TICKS = base.ENTROPY_TICKS
REFERENCE_COLOUR = base.REFERENCE_COLOUR
PREDICTION_COLOUR = base.PREDICTION_COLOUR
MASK_COLOUR = base.MASK_COLOUR

COMPARISONS = [
    {
        "panel": "A",
        "destination": "main",
        "variable": "Sex",
        "region": "R-Caudate",
        "region_id": 26,
        "title": "Right caudate by sex",
        "groups": [("F", "Female"), ("M", "Male")],
        "p_permutation": 0.003950,
        "q_fdr_105": 0.217864,
    },
    {
        "panel": "B",
        "destination": "main",
        "variable": "Pathology_grp",
        "region": "Brain-Stem",
        "region_id": 12,
        "title": "Brainstem by pathology",
        "groups": [(value, PATHOLOGY_LABELS[value]) for value in PATHOLOGY_ORDER],
        "p_permutation": 0.027049,
        "q_fdr_105": 0.454708,
    },
    {
        "panel": "C",
        "destination": "appendix",
        "variable": "Op_Side",
        "region": "L-Amygdala",
        "region_id": 14,
        "title": "Left amygdala by operation side",
        "groups": [("L", "Left"), ("R", "Right")],
        "p_permutation": 0.019449,
        "q_fdr_105": 0.454708,
    },
    {
        "panel": "D",
        "destination": "appendix",
        "variable": "Op_Type_collapsed",
        "region": "L-Cerebral-WM",
        "region_id": 1,
        "title": "Left cerebral white matter by operation type",
        "groups": [("Temporal", "Temporal"), ("Extra-temporal", "Extra-temporal")],
        "p_permutation": 0.052397,
        "q_fdr_105": 0.454708,
    },
    {
        "panel": "E",
        "destination": "appendix",
        "variable": "ILAE_Y1_seizure_free",
        "region": "R-Hippocampus",
        "region_id": 29,
        "title": "Right hippocampus by Year-1 seizure outcome",
        "groups": [(1.0, "Seizure-free (ILAE 1)"), (0.0, "Not seizure-free (2-5)")],
        "p_permutation": 0.140743,
        "q_fdr_105": 0.615750,
    },
]

FIXED_CROP_SIZE = {
    "R-Caudate": 96,
    "Brain-Stem": 112,
    "L-Amygdala": 80,
    "L-Cerebral-WM": 224,
    "R-Hippocampus": 80,
}


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
        help="Final disjoint directory containing the CSV and NIfTI outputs",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Defaults to IDEAS_ROOT/tables_metadata/Metadata_Release_Anon.csv",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--top-n-qc", type=int, default=4)
    parser.add_argument("--shortlist-per-group", type=int, default=12)
    return parser.parse_args()


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def collapse_operation_type(series: pd.Series) -> pd.Series:
    cleaned = clean_text(series)
    result = pd.Series(pd.NA, index=series.index, dtype="object")
    result.loc[cleaned.notna()] = "Extra-temporal"
    result.loc[cleaned.str.startswith("T", na=False)] = "Temporal"
    return result


def validate_and_deduplicate_metadata(metadata: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "ID" not in metadata.columns:
        raise RuntimeError("Clinical metadata is missing the ID column")
    data = metadata.copy()
    data["ID"] = pd.to_numeric(data["ID"], errors="coerce")
    if data["ID"].isna().any() or (data["ID"] % 1 != 0).any():
        raise RuntimeError("Clinical metadata contains missing or non-integer IDs")
    data["ID"] = data["ID"].astype(int)

    duplicate_rows = data.loc[data["ID"].duplicated(keep=False)]
    for identifier, rows in duplicate_rows.groupby("ID", sort=True):
        normalised = rows.fillna("<NA>").astype(str).drop_duplicates()
        if len(normalised) != 1:
            raise RuntimeError(
                f"Conflicting clinical metadata rows for ID={identifier}"
            )
    duplicate_count = int(data.duplicated("ID").sum())
    return data.drop_duplicates("ID", keep="first").copy(), duplicate_count


def load_analysis_table(
    wide: pd.DataFrame, metadata_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    metadata_raw = pd.read_csv(metadata_path)
    metadata, duplicate_count = validate_and_deduplicate_metadata(metadata_raw)

    metrics = wide.copy()
    extracted = metrics["subject_id"].astype(str).str.extract(r"^IDEAS_(\d+)$")[0]
    metrics["ID"] = pd.to_numeric(extracted, errors="coerce")
    if metrics["ID"].isna().any():
        bad = metrics.loc[metrics["ID"].isna(), "subject_id"].astype(str).tolist()
        raise RuntimeError(f"Could not parse subject IDs: {bad[:10]}")
    metrics["ID"] = metrics["ID"].astype(int)

    analysis = metrics.merge(
        metadata,
        on="ID",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_metadata"),
    )
    if len(analysis) != EXPECTED_CLINICAL_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_CLINICAL_SUBJECTS} matched clinical subjects; "
            f"found {len(analysis)}"
        )
    if analysis["subject_id"].nunique() != EXPECTED_CLINICAL_SUBJECTS:
        raise RuntimeError("Matched clinical table does not contain unique subject IDs")

    required = ["Sex", "Pathology", "Op_Side"]
    missing = [column for column in required if column not in analysis.columns]
    if missing:
        raise RuntimeError(f"Clinical metadata is missing columns: {missing}")

    analysis["Sex"] = clean_text(analysis["Sex"]).map(
        {"F": "F", "FEMALE": "F", "M": "M", "MALE": "M"}
    )
    analysis["Op_Side"] = clean_text(analysis["Op_Side"]).map(
        {"L": "L", "LEFT": "L", "R": "R", "RIGHT": "R"}
    )
    pathology = clean_text(analysis["Pathology"])
    analysis["Pathology_grp"] = pathology.replace(
        {value: "OTHER" for value in RARE_PATHOLOGY}
    )

    if "Op_Type_collapsed" in analysis.columns:
        analysis["Op_Type_collapsed"] = collapse_operation_type(
            analysis["Op_Type_collapsed"]
        )
    elif "Op_Type" in analysis.columns:
        analysis["Op_Type_collapsed"] = collapse_operation_type(analysis["Op_Type"])
    else:
        raise RuntimeError("Clinical metadata has neither Op_Type_collapsed nor Op_Type")

    if "ILAE_Y1_seizure_free" in analysis.columns:
        outcome = pd.to_numeric(analysis["ILAE_Y1_seizure_free"], errors="coerce")
    elif "ILAE_Year1" in analysis.columns:
        year1 = pd.to_numeric(analysis["ILAE_Year1"], errors="coerce")
        outcome = pd.Series(
            np.where(year1.isna(), np.nan, (year1 == 1).astype(float)),
            index=analysis.index,
        )
    else:
        raise RuntimeError(
            "Clinical metadata has neither ILAE_Y1_seizure_free nor ILAE_Year1"
        )
    analysis["ILAE_Y1_seizure_free"] = outcome

    unexpected_pathology = sorted(
        set(analysis["Pathology_grp"].dropna()) - set(PATHOLOGY_ORDER)
    )
    if unexpected_pathology:
        raise RuntimeError(f"Unexpected grouped pathology levels: {unexpected_pathology}")
    for variable in ["Sex", "Pathology_grp", "Op_Side", "Op_Type_collapsed"]:
        if analysis[variable].isna().any():
            raise RuntimeError(f"Missing or unrecognised values in {variable}")
    n_outcome = int(analysis["ILAE_Y1_seizure_free"].notna().sum())
    if n_outcome != EXPECTED_OUTCOME_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_OUTCOME_SUBJECTS} complete Year-1 outcomes; "
            f"found {n_outcome}"
        )

    merge_audit = {
        "metadata_rows_raw": int(len(metadata_raw)),
        "metadata_rows_after_exact_id_deduplication": int(len(metadata)),
        "exact_duplicate_metadata_rows_removed": duplicate_count,
        "matched_clinical_subjects": int(len(analysis)),
        "complete_year1_outcomes": n_outcome,
        "entropy_only_subjects": int(len(metrics) - len(analysis)),
    }
    return analysis, merge_audit


def select_candidates(
    analysis: pd.DataFrame, shortlist_per_group: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shortlists = []
    selected = []
    for comparison in COMPARISONS:
        variable = str(comparison["variable"])
        region = str(comparison["region"])
        entropy_column = f"{region}_entropy_mean"
        dice_column = f"{region}_dice"
        required = [variable, entropy_column, dice_column]
        missing = [column for column in required if column not in analysis.columns]
        if missing:
            raise RuntimeError(
                f"Missing columns for panel {comparison['panel']}: {missing}"
            )

        for group_value, group_label in comparison["groups"]:
            subset = analysis.loc[
                analysis[variable].eq(group_value),
                [
                    "subject_id",
                    "group",
                    "models_used",
                    entropy_column,
                    dice_column,
                ],
            ].dropna()
            if subset.empty:
                raise RuntimeError(
                    f"No complete subjects for {variable}={group_value!r}"
                )
            median = float(subset[entropy_column].median())
            ranked = subset.assign(
                abs_deviation_from_subgroup_median=(
                    subset[entropy_column].astype(float) - median
                ).abs()
            ).sort_values(
                ["abs_deviation_from_subgroup_median", "subject_id"],
                ascending=[True, True],
            )
            ranked = ranked.head(shortlist_per_group).reset_index(drop=True)
            ranked["candidate_rank"] = np.arange(1, len(ranked) + 1)
            ranked = ranked.rename(
                columns={entropy_column: "entropy", dice_column: "dice"}
            )
            ranked["panel"] = comparison["panel"]
            ranked["destination"] = comparison["destination"]
            ranked["variable"] = variable
            ranked["region"] = region
            ranked["region_id"] = int(comparison["region_id"])
            ranked["comparison_title"] = comparison["title"]
            ranked["group_value"] = str(group_value)
            ranked["group_label"] = group_label
            ranked["n_group"] = int(len(subset))
            ranked["subgroup_median"] = median
            ranked["p_group_adjusted_permutation"] = comparison["p_permutation"]
            ranked["q_fdr_105"] = comparison["q_fdr_105"]
            ranked["role"] = f"{comparison['panel']}_{group_label}"
            ranked["selected_before_visual_qc"] = ranked["candidate_rank"].eq(1)
            shortlists.append(ranked)
            selected.append(ranked.iloc[[0]].copy())

    all_shortlists = pd.concat(shortlists, ignore_index=True)
    all_selected = pd.concat(selected, ignore_index=True)
    return all_selected, all_shortlists


def fixed_square_crop(mask: np.ndarray, size: int) -> tuple[slice, slice]:
    points = np.argwhere(mask)
    if len(points) == 0:
        raise RuntimeError("Cannot crop an empty reference mask")
    centre_y, centre_x = np.rint(points.mean(axis=0)).astype(int)
    height, width = mask.shape
    size = int(min(size, height, width))

    def bounds(centre: int, limit: int) -> tuple[int, int]:
        start = int(centre - size // 2)
        end = start + size
        if start < 0:
            end -= start
            start = 0
        if end > limit:
            start -= end - limit
            end = limit
        return max(0, start), min(limit, end)

    y0, y1 = bounds(centre_y, height)
    x0, x1 = bounds(centre_x, width)
    return slice(y0, y1), slice(x0, x1)


def prepare_case(
    row: pd.Series, ideas_root: Path, uncertainty_dir: Path
) -> tuple[dict[str, object], dict[str, object]]:
    subject_id = str(row["subject_id"])
    region = str(row["region"])
    label = int(row["region_id"])
    paths = base.subject_paths(ideas_root, uncertainty_dir, subject_id)
    images = base.load_canonical_images(paths)
    shape, deviations = base.validate_image_grids(images, subject_id)
    arrays = {key: np.asarray(image.dataobj) for key, image in images.items()}
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError(f"Non-finite required NIfTI values for {subject_id}")

    recomputed_dice = base.dice_for_label(
        arrays["prediction"], arrays["reference"], label
    )
    reference_mask_3d = arrays["reference"] == label
    recomputed_entropy = float(arrays["entropy"][reference_mask_3d].mean())
    if not np.isclose(recomputed_dice, float(row["dice"]), atol=1e-10, rtol=1e-8):
        raise RuntimeError(
            f"Dice mismatch for {subject_id}/{region}: "
            f"NIfTI={recomputed_dice}, CSV={row['dice']}"
        )
    if not np.isclose(
        recomputed_entropy, float(row["entropy"]), atol=1e-6, rtol=1e-6
    ):
        raise RuntimeError(
            f"Entropy mismatch for {subject_id}/{region}: "
            f"NIfTI={recomputed_entropy}, CSV={row['entropy']}"
        )

    coronal_slice = base.largest_reference_coronal_slice(arrays["reference"], label)
    slices = {
        key: base.rotate_coronal(array, coronal_slice) for key, array in arrays.items()
    }
    ref_mask = slices["reference"] == label
    pred_mask = slices["prediction"] == label
    crop = fixed_square_crop(ref_mask, FIXED_CROP_SIZE[region])
    prepared = {key: value[crop] for key, value in slices.items()}
    prepared["ref_mask"] = ref_mask[crop]
    prepared["pred_mask"] = pred_mask[crop]
    prepared["display_limits"] = base.display_limits(prepared["raw"])
    prepared["slice_index"] = coronal_slice

    reference_points = np.argwhere(reference_mask_3d)
    prediction_points = np.argwhere(arrays["prediction"] == label)
    if len(prediction_points) == 0:
        centroid_distance_mm = np.inf
        prediction_reference_volume_ratio = 0.0
    else:
        affine = images["reference"].affine
        reference_centroid_voxels = reference_points.mean(axis=0)
        prediction_centroid_voxels = prediction_points.mean(axis=0)
        reference_centroid_mm = (
            affine[:3, :3] @ reference_centroid_voxels + affine[:3, 3]
        )
        prediction_centroid_mm = (
            affine[:3, :3] @ prediction_centroid_voxels + affine[:3, 3]
        )
        centroid_distance_mm = float(
            np.linalg.norm(reference_centroid_mm - prediction_centroid_mm)
        )
        prediction_reference_volume_ratio = float(
            len(prediction_points) / len(reference_points)
        )
    audit = {
        "panel": row["panel"],
        "variable": row["variable"],
        "group_label": row["group_label"],
        "candidate_rank": int(row["candidate_rank"]),
        "subject_id": subject_id,
        "region": region,
        "region_id": label,
        "shape": "x".join(map(str, shape)),
        "canonical_coronal_slice": coronal_slice,
        "dice_csv": float(row["dice"]),
        "dice_recomputed": recomputed_dice,
        "entropy_csv": float(row["entropy"]),
        "entropy_recomputed": recomputed_entropy,
        "max_grid_deviation_voxels": max(deviations.values()),
        "centroid_distance_mm": centroid_distance_mm,
        "prediction_reference_volume_ratio": prediction_reference_volume_ratio,
        "dice_matches": True,
        "entropy_matches": True,
        "grid_matches": True,
    }
    return prepared, audit


def draw_entropy_panel(
    axis: plt.Axes, row: pd.Series, prepared: dict[str, object]
) -> object:
    image = axis.imshow(
        prepared["entropy"],
        cmap="magma",
        vmin=ENTROPY_VMIN,
        vmax=ENTROPY_VMAX,
        interpolation="nearest",
    )
    if np.asarray(prepared["ref_mask"]).any():
        axis.contour(
            prepared["ref_mask"],
            levels=[0.5],
            colors=[REFERENCE_COLOUR],
            linewidths=1.0,
        )
    if np.asarray(prepared["pred_mask"]).any():
        axis.contour(
            prepared["pred_mask"],
            levels=[0.5],
            colors=[PREDICTION_COLOUR],
            linewidths=1.0,
        )
    axis.set_title(
        f"{row['group_label']}\n{row['subject_id']}", fontsize=8.2, pad=4
    )
    axis.text(
        0.5,
        -0.075,
        f"entropy = {row['entropy']:.3f}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def add_shared_legend(fig: plt.Figure, y: float = 0.01) -> None:
    handles = [
        Line2D([0], [0], color=REFERENCE_COLOUR, lw=2, label="Reference contour"),
        Line2D([0], [0], color=PREDICTION_COLOUR, lw=2, label="Prediction contour"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.5, y),
    )


def render_main_figure(
    selected: pd.DataFrame,
    prepared: dict[tuple[str, str], dict[str, object]],
    outdir: Path,
) -> None:
    fig = plt.figure(figsize=(8.0, 4.9), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        6,
        left=0.055,
        right=0.895,
        top=0.91,
        bottom=0.13,
        hspace=0.62,
        wspace=0.08,
    )
    image = None
    row_a = selected.loc[selected["panel"].eq("A")].reset_index(drop=True)
    for index, row in row_a.iterrows():
        axis = fig.add_subplot(grid[0, 2 + index])
        image = draw_entropy_panel(
            axis, row, prepared[(str(row["panel"]), str(row["group_label"]))]
        )
    row_b = selected.loc[selected["panel"].eq("B")].reset_index(drop=True)
    for index, row in row_b.iterrows():
        axis = fig.add_subplot(grid[1, index])
        image = draw_entropy_panel(
            axis, row, prepared[(str(row["panel"]), str(row["group_label"]))]
        )

    fig.text(0.055, 0.955, "A   Right caudate by sex", fontsize=9.5, weight="bold")
    fig.text(0.055, 0.515, "B   Brainstem by pathology", fontsize=9.5, weight="bold")
    colour_axis = fig.add_axes([0.92, 0.19, 0.014, 0.64])
    colourbar = fig.colorbar(image, cax=colour_axis, ticks=ENTROPY_TICKS)
    colourbar.set_label("Predictive entropy", fontsize=8)
    colourbar.ax.set_yticklabels(["0.0", "0.5", "1.0", "1.3"])
    colourbar.ax.tick_params(labelsize=7)
    add_shared_legend(fig, y=0.005)
    fig.savefig(outdir / "subgroup_qualitative_disjoint.pdf", facecolor="white")
    fig.savefig(
        outdir / "subgroup_qualitative_disjoint.png", dpi=300, facecolor="white"
    )
    plt.close(fig)


def render_appendix_figure(
    selected: pd.DataFrame,
    prepared: dict[tuple[str, str], dict[str, object]],
    outdir: Path,
) -> None:
    fig = plt.figure(figsize=(7.4, 9.8), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        2,
        left=0.08,
        right=0.88,
        top=0.94,
        bottom=0.075,
        hspace=0.78,
        wspace=0.18,
    )
    image = None
    panels = ["C", "D", "E"]
    headers = {
        "C": "C   Left amygdala by operation side",
        "D": "D   Left cerebral white matter by operation type",
        "E": "E   Right hippocampus by Year-1 seizure outcome",
    }
    row_axes: list[list[plt.Axes]] = []
    for row_index, panel in enumerate(panels):
        rows = selected.loc[selected["panel"].eq(panel)].reset_index(drop=True)
        axes_this_row = []
        for column_index, row in rows.iterrows():
            axis = fig.add_subplot(grid[row_index, column_index])
            image = draw_entropy_panel(
                axis, row, prepared[(str(row["panel"]), str(row["group_label"]))]
            )
            axes_this_row.append(axis)
        row_axes.append(axes_this_row)
    for panel, axes_this_row in zip(panels, row_axes):
        top = max(axis.get_position().y1 for axis in axes_this_row)
        fig.text(
            0.03,
            top + 0.040,
            headers[panel],
            fontsize=9.5,
            weight="bold",
            va="bottom",
        )

    colour_axis = fig.add_axes([0.91, 0.18, 0.018, 0.66])
    colourbar = fig.colorbar(image, cax=colour_axis, ticks=ENTROPY_TICKS)
    colourbar.set_label("Predictive entropy", fontsize=8)
    colourbar.ax.set_yticklabels(["0.0", "0.5", "1.0", "1.3"])
    colourbar.ax.tick_params(labelsize=7)
    add_shared_legend(fig, y=0.003)
    fig.savefig(outdir / "appendixD_subgroup_qualitative_disjoint.pdf", facecolor="white")
    fig.savefig(
        outdir / "appendixD_subgroup_qualitative_disjoint.png",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)


def plot_qc_rows(
    rows: pd.DataFrame, prepared_rows: list[dict[str, object]], title: str
) -> plt.Figure:
    nrows = len(rows)
    fig, axes = plt.subplots(
        nrows,
        4,
        figsize=(11.7, 2.5 * nrows + 0.8),
        constrained_layout=True,
        squeeze=False,
    )
    columns = [
        "T1-weighted MRI",
        "Reference target",
        "Ensemble prediction",
        "Predictive entropy",
    ]
    for axis, column in zip(axes[0], columns):
        axis.set_title(column, fontsize=9, weight="bold")
    orange = ListedColormap([MASK_COLOUR])
    entropy_image = None
    for row_index, ((_, row), prepared) in enumerate(
        zip(rows.iterrows(), prepared_rows)
    ):
        raw = prepared["raw"]
        vmin, vmax = prepared["display_limits"]
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].imshow(raw, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row_index, 1].imshow(raw, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row_index, 1].imshow(
            np.ma.masked_where(~prepared["ref_mask"], prepared["ref_mask"]),
            cmap=orange,
            alpha=0.72,
            interpolation="nearest",
        )
        axes[row_index, 2].imshow(raw, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row_index, 2].imshow(
            np.ma.masked_where(~prepared["pred_mask"], prepared["pred_mask"]),
            cmap=orange,
            alpha=0.72,
            interpolation="nearest",
        )
        entropy_image = axes[row_index, 3].imshow(
            prepared["entropy"],
            cmap="magma",
            vmin=ENTROPY_VMIN,
            vmax=ENTROPY_VMAX,
            interpolation="nearest",
        )
        if np.asarray(prepared["ref_mask"]).any():
            axes[row_index, 3].contour(
                prepared["ref_mask"],
                levels=[0.5],
                colors=[REFERENCE_COLOUR],
                linewidths=1.0,
            )
        if np.asarray(prepared["pred_mask"]).any():
            axes[row_index, 3].contour(
                prepared["pred_mask"],
                levels=[0.5],
                colors=[PREDICTION_COLOUR],
                linewidths=1.0,
            )
        label = (
            f"Rank {int(row['candidate_rank'])}: {row['subject_id']}\n"
            f"Dice={row['dice']:.3f}; entropy={row['entropy']:.3f}\n"
            f"|entropy - median|={row['abs_deviation_from_subgroup_median']:.4f}"
        )
        axes[row_index, 0].set_ylabel(
            label, rotation=0, ha="right", va="center", labelpad=10, fontsize=7.4
        )
    colourbar = fig.colorbar(
        entropy_image,
        ax=axes[:, 3],
        fraction=0.026,
        pad=0.015,
        ticks=ENTROPY_TICKS,
        label="Predictive entropy",
    )
    colourbar.ax.set_yticklabels(["0.0", "0.5", "1.0", "1.3"])
    fig.suptitle(title, fontsize=11, weight="bold")
    return fig


def render_and_audit(
    selected: pd.DataFrame,
    shortlists: pd.DataFrame,
    ideas_root: Path,
    uncertainty_dir: Path,
    outdir: Path,
    top_n_qc: int,
) -> pd.DataFrame:
    selected_prepared: dict[tuple[str, str], dict[str, object]] = {}
    audits: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        prepared, audit = prepare_case(row, ideas_root, uncertainty_dir)
        selected_prepared[(str(row["panel"]), str(row["group_label"]))] = prepared
        audits.append(audit)

    render_main_figure(selected, selected_prepared, outdir)
    render_appendix_figure(selected, selected_prepared, outdir)

    with PdfPages(outdir / "subgroup_qualitative_disjoint_qc.pdf") as pdf:
        group_keys = selected[["panel", "group_label"]].itertuples(index=False)
        for panel, group_label in group_keys:
            candidates = shortlists.loc[
                shortlists["panel"].eq(panel)
                & shortlists["group_label"].eq(group_label)
            ].head(top_n_qc)
            prepared_rows = []
            for _, row in candidates.iterrows():
                prepared, audit = prepare_case(row, ideas_root, uncertainty_dir)
                prepared_rows.append(prepared)
                audits.append(audit)
            comparison_title = str(candidates.iloc[0]["comparison_title"])
            figure = plot_qc_rows(
                candidates,
                prepared_rows,
                f"Visual-QC candidates: {comparison_title} - {group_label}",
            )
            pdf.savefig(figure)
            plt.close(figure)

    return (
        pd.DataFrame(audits)
        .drop_duplicates(["panel", "group_label", "candidate_rank", "subject_id"])
        .sort_values(["panel", "group_label", "candidate_rank"])
        .reset_index(drop=True)
    )


def write_captions(selected: pd.DataFrame, outdir: Path) -> None:
    main_caption = (
        "Qualitative subgroup examples from the leakage-free four-model disjoint "
        "ensemble. (A) Right-caudate entropy for the female and male subjects whose "
        "values were closest to their subgroup medians. (B) Brainstem entropy for the "
        "subject closest to the median in each grouped-pathology category; GL, TREBLE "
        "and TBC were combined as Other. Green and cyan contours show the reference "
        "and prediction. Reported values are mean predictive entropy within the full "
        "3D reference region; all panels use the same 0--1.3 scale. Cases were selected "
        "numerically before visual inspection. They illustrate the screened subgroup "
        "distributions and are not evidence for a subgroup association; neither "
        "comparison survived pooled FDR correction."
    )
    appendix_caption = (
        "Additional qualitative subgroup examples from the leakage-free four-model "
        "disjoint ensemble. (C) Left-amygdala entropy by operation side. (D) Left "
        "cerebral-white-matter entropy by operation type. (E) Right-hippocampal entropy "
        "by Year-1 seizure outcome. The subject closest to each subgroup median was "
        "selected numerically before visual inspection. Green and cyan contours show "
        "the reference and prediction, and reported values are mean predictive entropy "
        "within the full 3D reference region. All panels use the same 0--1.3 scale. "
        "None of these comparisons survived pooled FDR correction, and the examples "
        "are illustrative rather than evidence for or against a subgroup association."
    )
    (outdir / "subgroup_qualitative_disjoint_caption.txt").write_text(
        main_caption + "\n", encoding="utf-8"
    )
    (outdir / "appendixD_subgroup_qualitative_disjoint_caption.txt").write_text(
        appendix_caption + "\n", encoding="utf-8"
    )
    main_tex = (
        "\\begin{figure*}[t]\n"
        "\\centering\n"
        "\\includegraphics[width=\\linewidth]{img/subgroup_qualitative_disjoint.pdf}\n"
        f"\\caption{{{main_caption}}}\n"
        "\\label{fig:subgroup_qualitative}\n"
        "\\end{figure*}\n"
    )
    appendix_tex = (
        "\\begin{figure}[H]\n"
        "\\centering\n"
        "\\includegraphics[width=0.82\\linewidth]{img/appendixD_subgroup_qualitative_disjoint.pdf}\n"
        f"\\caption{{{appendix_caption}}}\n"
        "\\label{fig:appendix_subgroup}\n"
        "\\end{figure}\n"
    )
    (outdir / "subgroup_qualitative_disjoint_insert.tex").write_text(
        main_tex, encoding="utf-8"
    )
    (outdir / "appendixD_subgroup_qualitative_disjoint_insert.tex").write_text(
        appendix_tex, encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.top_n_qc < 1 or args.shortlist_per_group < args.top_n_qc:
        raise ValueError("Require shortlist-per-group >= top-n-qc >= 1")

    ideas_root = args.ideas_root.resolve()
    uncertainty_dir = args.uncertainty_dir.resolve()
    metadata_path = (
        args.metadata_csv.resolve()
        if args.metadata_csv is not None
        else ideas_root / "tables_metadata/Metadata_Release_Anon.csv"
    )
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = uncertainty_dir / "region_entropy_decomposed_reference_masked.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    wide = pd.read_csv(csv_path)
    inventory = base.validate_input_inventory(ideas_root, uncertainty_dir, wide)
    analysis, merge_audit = load_analysis_table(wide, metadata_path)
    selected, shortlists = select_candidates(analysis, args.shortlist_per_group)
    audit = render_and_audit(
        selected,
        shortlists,
        ideas_root,
        uncertainty_dir,
        outdir,
        args.top_n_qc,
    )

    output_columns = [
        "panel",
        "destination",
        "variable",
        "comparison_title",
        "group_value",
        "group_label",
        "n_group",
        "subgroup_median",
        "candidate_rank",
        "subject_id",
        "group",
        "models_used",
        "region_id",
        "region",
        "dice",
        "entropy",
        "abs_deviation_from_subgroup_median",
        "p_group_adjusted_permutation",
        "q_fdr_105",
        "selected_before_visual_qc",
    ]
    selected[output_columns].to_csv(
        outdir / "subgroup_qualitative_disjoint_selected_examples.csv", index=False
    )
    shortlists[output_columns].to_csv(
        outdir / "subgroup_qualitative_disjoint_candidate_shortlists.csv",
        index=False,
    )
    audit.to_csv(
        outdir / "subgroup_qualitative_disjoint_array_verification.csv", index=False
    )
    write_captions(selected, outdir)

    metadata = {
        "input_csv": str(csv_path),
        "clinical_metadata_csv": str(metadata_path),
        "input_file_counts": inventory,
        "clinical_merge_audit": merge_audit,
        "selection_rule": "closest regional entropy to subgroup median",
        "slice_rule": "largest reference-region coronal cross-section",
        "entropy_display_range": [ENTROPY_VMIN, ENTROPY_VMAX],
        "comparisons": COMPARISONS,
        "visual_qc_exclusions": [
            "gross acquisition artefact",
            "gross image/reference misregistration",
            "missing or unreadable required panel",
        ],
    }
    (outdir / "subgroup_qualitative_disjoint_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    duplicate_selected_subjects = sorted(
        selected.loc[
            selected["subject_id"].duplicated(keep=False), "subject_id"
        ].astype(str).unique()
    )
    lines = [
        "DISJOINT SUBGROUP AND APPENDIX D SELECTION CHECK",
        f"Input: {csv_path}",
        f"Clinical metadata: {metadata_path}",
        f"Disjoint subjects: {wide['subject_id'].nunique()}",
        f"Matched clinical subjects: {len(analysis)}",
        f"Complete Year-1 outcomes: {analysis['ILAE_Y1_seizure_free'].notna().sum()}",
        f"Input file counts: {inventory}",
        f"Exact duplicate metadata rows removed: {merge_audit['exact_duplicate_metadata_rows_removed']}",
        "Selection rule: closest regional entropy to subgroup median",
        "Slice rule: largest reference-region coronal cross-section",
        f"Entropy display range: {ENTROPY_VMIN:.1f} to {ENTROPY_VMAX:.1f}",
        "",
        "Preselected rank-1 examples before visual QC:",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"Panel {row['panel']} - {row['comparison_title']} - {row['group_label']}: "
            f"{row['subject_id']}, {row['region']}, n={int(row['n_group'])}, "
            f"subgroup median={row['subgroup_median']:.6f}, "
            f"entropy={row['entropy']:.6f}, Dice={row['dice']:.6f}, "
            f"absolute median deviation={row['abs_deviation_from_subgroup_median']:.6f}"
        )
    lines.extend(
        [
            "",
            f"Repeated selected subjects across subgroup panels: {duplicate_selected_subjects}",
            "All selected and rendered QC Dice values matched direct NIfTI recomputation.",
            "All selected and rendered QC entropy values matched direct reference-masked recomputation.",
            "All selected and rendered QC raw, reference, prediction and entropy arrays matched in shape and voxel grid.",
            "Cases were selected from the final disjoint predictions before visual inspection.",
            "No displayed RQ2 comparison survived pooled FDR correction across 105 tests.",
            "Poor segmentation and extreme uncertainty are not visual-QC exclusion criteria.",
            "Permissible exclusions: gross acquisition artefact, gross image/reference misregistration, or a missing/unreadable required panel.",
        ]
    )
    summary = "\n".join(lines) + "\n"
    (outdir / "subgroup_qualitative_disjoint_summary.txt").write_text(
        summary, encoding="utf-8"
    )
    print(summary, end="")


if __name__ == "__main__":
    main()
