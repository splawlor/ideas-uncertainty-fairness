#!/usr/bin/env python3
"""Generate Figure 3.1 and its label legend."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import figure_3_1_coronal as base


GRID_ATOL_VOXELS = 1e-4
EXPECTED_MODELS = {0, 1, 2, 3, 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the final disjoint Figure 3.1 and exact Appendix B label list."
    )
    parser.add_argument(
        "--ideas-root",
        type=Path,
        default=Path(os.environ.get("IDEAS_ROOT", ".")),
    )
    parser.add_argument(
        "--uncertainty-dir",
        type=Path,
        required=True,
        help="Directory containing segmentation, entropy and summary files.",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--subject", default="IDEAS_10")
    parser.add_argument(
        "--coronal-slice",
        type=int,
        default=133,
        help="Canonical coronal index. Default preserves the previously approved example.",
    )
    parser.add_argument("--entropy-max", type=float, default=1.3)
    return parser.parse_args()


def load_mapping(script_dir: Path) -> tuple[dict[int, dict[str, object]], set[int]]:
    mapping_rows = base.read_csv(script_dir / "label_mapping.csv")
    target_rows = base.read_csv(script_dir / "target_15_labels.csv")
    mapping: dict[int, dict[str, object]] = {
        int(row["remapped_id"]): {
            "name": row["label_name"],
            "rgb": tuple(int(row[channel]) for channel in ("R", "G", "B")),
        }
        for row in mapping_rows
    }
    target_ids = {int(row["remapped_id"]) for row in target_rows}
    if len(mapping) != 110 or set(mapping) != set(range(110)):
        raise RuntimeError("label_mapping.csv must contain remapped IDs 0 through 109")
    if len(target_ids) != 15:
        raise RuntimeError(f"Expected 15 target IDs; found {len(target_ids)}")
    for label_id, colour in base.TARGET_DISPLAY_RGB.items():
        if label_id not in mapping:
            raise RuntimeError(f"Target label {label_id} is missing from label_mapping.csv")
        mapping[label_id]["rgb"] = colour
    return mapping, target_ids


def subject_paths(
    ideas_root: Path, uncertainty_dir: Path, subject: str
) -> dict[str, Path]:
    return {
        "raw": ideas_root
        / "nnunet_raw/Dataset001_IDEAS/imagesTr"
        / f"{subject}_0000.nii.gz",
        "reference": ideas_root
        / "nnunet_raw/Dataset001_IDEAS/labelsTr"
        / f"{subject}.nii.gz",
        "prediction": uncertainty_dir / f"{subject}_seg.nii.gz",
        "entropy": uncertainty_dir / f"{subject}_entropy.nii.gz",
    }


def max_grid_deviation_voxels(
    reference: nib.spatialimages.SpatialImage,
    image: nib.spatialimages.SpatialImage,
) -> float:
    shape = np.asarray(reference.shape, dtype=float)
    corners = np.asarray(
        [
            [x, y, z, 1.0]
            for x in (0.0, shape[0] - 1.0)
            for y in (0.0, shape[1] - 1.0)
            for z in (0.0, shape[2] - 1.0)
        ]
    ).T
    transform = np.linalg.inv(reference.affine) @ image.affine
    mapped = (transform @ corners)[:3].T
    expected = corners[:3].T
    return float(np.max(np.abs(mapped - expected)))


def validate_grids(
    images: dict[str, nib.spatialimages.SpatialImage], subject: str
) -> dict[str, float]:
    reference = images["reference"]
    deviations: dict[str, float] = {}
    for name, image in images.items():
        if image.shape != reference.shape:
            raise RuntimeError(
                f"Canonical shape mismatch for {subject}/{name}: "
                f"{image.shape} vs {reference.shape}"
            )
        deviation = max_grid_deviation_voxels(reference, image)
        deviations[name] = deviation
        if deviation > GRID_ATOL_VOXELS:
            raise RuntimeError(
                f"Voxel-grid mismatch for {subject}/{name}: "
                f"{deviation:.6g} > {GRID_ATOL_VOXELS:.1e} voxels"
            )
    return deviations


def read_disjoint_row(csv_path: Path, subject: str) -> dict[str, str]:
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    matches: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("subject_id") == subject:
                matches.append(row)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one final CSV row for {subject}; found {len(matches)}")
    row = matches[0]
    group = int(row["group"])
    models = {int(value) for value in row["models_used"].split("|")}
    if group not in EXPECTED_MODELS:
        raise RuntimeError(f"Invalid disjoint group for {subject}: {group}")
    if models != EXPECTED_MODELS - {group}:
        raise RuntimeError(
            f"Invalid subject-specific model set for {subject}: group={group}, "
            f"models_used={sorted(models)}"
        )
    return row


def latex_escape(text: str) -> str:
    return base.latex_escape(text)


def target_clause(
    visible_target_ids: set[int], target_ids: set[int], mapping: dict[int, dict[str, object]]
) -> str:
    missing = sorted(target_ids - visible_target_ids)
    if not missing:
        return "All 15 target regions are represented and are included in the legend."
    missing_names = [base.clean_label(str(mapping[label]["name"])) for label in missing]
    if len(missing_names) == 1:
        return (
            f"The {missing_names[0].lower()} is not represented and is therefore "
            "omitted from the legend."
        )
    return (
        "The following target regions are not represented and are therefore omitted "
        "from the legend: " + ", ".join(name.lower() for name in missing_names) + "."
    )


def write_insert(
    path: Path,
    subject: str,
    target_count: int,
    label_count: int,
    clause: str,
) -> None:
    caption = (
        f"Qualitative comparison for {latex_escape(subject)} using the same coronal "
        "slice and crop throughout: (A) raw T1-weighted MRI, (B) FreeSurfer-derived "
        "reference segmentation, (C) leakage-free four-model disjoint-ensemble "
        "prediction, and (D) predictive entropy calculated from the averaged softmax "
        "output. Each subject was predicted using the four models that had not trained "
        "on that subject. Entropy uses the natural logarithm, is not normalised, and is "
        "displayed from $0$ to $1.3$; warmer colours indicate higher predictive "
        f"uncertainty. The legend contains the {target_count} target regions represented "
        f"on this slice, with left and right structures shown separately. {clause} "
        f"Appendix~\\ref{{ap:label_legend}} lists all {label_count} non-background "
        "label IDs represented by at least one pixel in the reference or prediction panel."
    )
    lines = [
        r"\begin{figure*}[!t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{img/figure_3_1_coronal.pdf}",
        f"\\caption{{{caption}}}",
        r"\label{fig:qualitative_example}",
        r"\end{figure*}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    (path.parent / "figure_3_1_disjoint_caption.txt").write_text(
        caption.replace("$", "").replace("~", " ") + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.entropy_max <= 0:
        raise ValueError("--entropy-max must be positive")
    ideas_root = args.ideas_root.resolve()
    uncertainty_dir = args.uncertainty_dir.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    mapping, target_ids = load_mapping(script_dir)

    paths = subject_paths(ideas_root, uncertainty_dir, args.subject)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))
    row = read_disjoint_row(
        uncertainty_dir / "region_entropy_decomposed_reference_masked.csv", args.subject
    )

    original_images = {name: nib.load(str(path)) for name, path in paths.items()}
    images = {
        name: nib.as_closest_canonical(image) for name, image in original_images.items()
    }
    deviations = validate_grids(images, args.subject)
    arrays = {name: np.asarray(image.dataobj) for name, image in images.items()}
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError(f"Non-finite values in required NIfTIs for {args.subject}")

    reference = np.rint(arrays["reference"]).astype(np.int16)
    prediction = np.rint(arrays["prediction"]).astype(np.int16)
    unknown = (set(np.unique(reference)) | set(np.unique(prediction))) - set(mapping)
    if unknown:
        raise RuntimeError(f"Unknown remapped label IDs: {sorted(unknown)}")
    if not 0 <= args.coronal_slice < reference.shape[1]:
        raise ValueError(
            f"Coronal slice must be between 0 and {reference.shape[1] - 1}"
        )

    index = args.coronal_slice
    slices = {
        "raw": arrays["raw"][:, index, :],
        "reference": reference[:, index, :],
        "prediction": prediction[:, index, :],
        "entropy": arrays["entropy"][:, index, :].astype(np.float32),
    }
    bounds = tuple(
        int(value)
        for value in base.crop_bounds(slices["reference"], slices["prediction"])
    )
    crops = {name: base.crop(array, bounds) for name, array in slices.items()}
    raw_scaled, raw_low, raw_high = base.normalise_t1(crops["raw"])
    raw_display = base.display_orientation(np.round(raw_scaled * 255).astype(np.uint8))
    raw_rgb = np.repeat(raw_display[..., None], 3, axis=2)
    ref_rgb = base.display_orientation(base.label_rgb(crops["reference"], mapping))
    pred_rgb = base.display_orientation(base.label_rgb(crops["prediction"], mapping))
    entropy_rgb = base.display_orientation(
        base.entropy_rgb(crops["entropy"], args.entropy_max)
    )

    reference_ids = set(int(value) for value in np.unique(crops["reference"])) - {0}
    prediction_ids = set(int(value) for value in np.unique(crops["prediction"])) - {0}
    visible_ids = reference_ids | prediction_ids
    visible_target_ids = visible_ids & target_ids
    all_entries = base.legend_entries(visible_ids, mapping)
    target_entries = base.legend_entries(visible_target_ids, mapping)
    clause = target_clause(visible_target_ids, target_ids, mapping)

    base.save_exact_png(outdir / "qualitative_raw.png", raw_rgb)
    base.save_exact_png(outdir / "qualitative_reference.png", ref_rgb)
    base.save_exact_png(outdir / "qualitative_prediction.png", pred_rgb)
    base.save_exact_png(outdir / "qualitative_uncertainty.png", entropy_rgb)
    base.make_legend(
        outdir / "figure_3_1_target_legend",
        target_entries,
        "Target regions represented on this slice (left and right listed separately)",
    )
    base.make_combined_figure(
        outdir / "figure_3_1_coronal",
        raw_rgb,
        ref_rgb,
        pred_rgb,
        entropy_rgb,
        target_entries,
        args.entropy_max,
    )
    base.make_appendix_tex(outdir / "appendixB_disjoint.tex", all_entries)
    write_insert(
        outdir / "figure_3_1_disjoint_insert.tex",
        args.subject,
        len(target_entries),
        len(all_entries),
        clause,
    )

    with (outdir / "figure_3_1_visible_labels.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "remapped_id",
                "label_name",
                "R",
                "G",
                "B",
                "reference_pixels",
                "prediction_pixels",
                "union_pixels",
                "is_target",
            ]
        )
        for label_id in sorted(visible_ids):
            colour = mapping[label_id]["rgb"]
            reference_pixels = int((crops["reference"] == label_id).sum())
            prediction_pixels = int((crops["prediction"] == label_id).sum())
            union_pixels = int(
                np.logical_or(
                    crops["reference"] == label_id, crops["prediction"] == label_id
                ).sum()
            )
            writer.writerow(
                [
                    label_id,
                    mapping[label_id]["name"],
                    *colour,
                    reference_pixels,
                    prediction_pixels,
                    union_pixels,
                    label_id in target_ids,
                ]
            )

    with (outdir / "figure_3_1_panel_metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "subject_id",
                "disjoint_group",
                "models_used",
                "canonical_coronal_slice",
                "x_start",
                "x_stop",
                "z_start",
                "z_stop",
                "entropy_min",
                "entropy_max",
                "t1_window_low",
                "t1_window_high",
                "visible_nonbackground_labels",
                "visible_target_labels",
                "max_grid_deviation_voxels",
            ]
        )
        writer.writerow(
            [
                args.subject,
                row["group"],
                row["models_used"],
                index,
                *bounds,
                0.0,
                args.entropy_max,
                raw_low,
                raw_high,
                len(visible_ids),
                len(visible_target_ids),
                max(deviations.values()),
            ]
        )

    reference_only = sorted(reference_ids - prediction_ids)
    prediction_only = sorted(prediction_ids - reference_ids)
    shared = sorted(reference_ids & prediction_ids)
    summary = [
        "DISJOINT FIGURE 3.1 AND APPENDIX B CHECK",
        f"Subject: {args.subject}",
        f"Final disjoint CSV: {uncertainty_dir / 'region_entropy_decomposed_reference_masked.csv'}",
        f"Disjoint group: {row['group']}",
        f"Models used: {row['models_used']}",
        f"Canonical image shape: {images['reference'].shape}",
        f"Canonical coronal slice: {index}",
        f"Crop bounds (x0, x1, z0, z1): {bounds}",
        f"Maximum voxel-grid deviation: {max(deviations.values()):.8g} voxels",
        f"Non-background labels in reference: {len(reference_ids)}",
        f"Non-background labels in prediction: {len(prediction_ids)}",
        f"Exact union listed in Appendix B: {len(visible_ids)}",
        f"Shared labels: {len(shared)}",
        f"Reference-only labels: {reference_only}",
        f"Prediction-only labels: {prediction_only}",
        f"Target labels represented in figure legend: {len(visible_target_ids)}/15",
        f"Missing target labels: {sorted(target_ids - visible_target_ids)}",
        "",
        "Every non-background label represented by at least one pixel is listed.",
        "No minimum-pixel visibility threshold was applied.",
        "Reference and prediction pixel counts are saved separately.",
        "The subject-specific four-model set excludes the subject's disjoint group.",
        "Raw, reference, prediction and entropy arrays matched in shape and voxel grid.",
        "All four displayed panels use the same canonical coronal slice and crop.",
    ]
    (outdir / "figure_3_1_disjoint_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print("\n".join(summary))


if __name__ == "__main__":
    main()
