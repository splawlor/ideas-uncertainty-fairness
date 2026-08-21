#!/usr/bin/env python3

import os
import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np


TARGET_DISPLAY_RGB = {
    1: (224, 224, 224),
    20: (128, 128, 128),
    6: (0, 128, 96),
    25: (102, 194, 165),
    7: (31, 119, 180),
    26: (23, 190, 207),
    8: (227, 26, 138),
    27: (247, 129, 191),
    9: (75, 50, 195),
    28: (140, 109, 222),
    13: (230, 171, 2),
    29: (255, 127, 14),
    14: (214, 39, 40),
    30: (165, 15, 21),
    12: (88, 125, 140),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a coronal T1, reference segmentation, ensemble prediction, "
            "entropy map, and an exact legend for every visible label."
        )
    )
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("IDEAS_ROOT", ".")))
    parser.add_argument("--subject", default="IDEAS_10")
    parser.add_argument("--slice", type=int, default=None, dest="coronal_slice")
    parser.add_argument("--entropy-max", type=float, default=1.3)
    parser.add_argument("--minimum-visible-pixels", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figure_3_1_coronal_generated"),
    )
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_image(path):
    image = nib.load(str(path))
    canonical = nib.as_closest_canonical(image)
    return image, canonical, np.asarray(canonical.dataobj)


def display_orientation(array_2d):
    if array_2d.ndim == 2:
        return np.flipud(array_2d.T)
    if array_2d.ndim == 3 and array_2d.shape[-1] in (3, 4):
        return np.flip(array_2d.transpose(1, 0, 2), axis=0)
    raise ValueError(f"Expected a 2D slice or an RGB/RGBA image, found {array_2d.shape}")


def normalise_t1(raw):
    finite = raw[np.isfinite(raw)]
    nonzero = finite[finite != 0]
    values = nonzero if nonzero.size else finite
    low, high = np.percentile(values, [1.0, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise RuntimeError("Could not calculate a valid T1 display window")
    scaled = np.clip((raw - low) / (high - low), 0.0, 1.0)
    return scaled, float(low), float(high)


def choose_slice(reference, prediction, target_ids, minimum_pixels):
    best = None
    for index in range(reference.shape[1]):
        ref_slice = reference[:, index, :]
        pred_slice = prediction[:, index, :]
        union = np.concatenate([ref_slice.ravel(), pred_slice.ravel()])
        labels, counts = np.unique(union[union > 0], return_counts=True)
        visible = {
            int(label): int(count)
            for label, count in zip(labels, counts)
            if count >= minimum_pixels
        }
        target_count = sum(label in target_ids for label in visible)
        foreground_pixels = int(np.logical_or(ref_slice > 0, pred_slice > 0).sum())
        score = (target_count, len(visible), foreground_pixels)
        if best is None or score > best[0]:
            best = (score, index)
    if best is None:
        raise RuntimeError("No foreground labels were found in any coronal slice")
    return best[1], best[0]


def crop_bounds(reference_slice, prediction_slice):
    foreground = np.logical_or(reference_slice > 0, prediction_slice > 0)
    coordinates = np.argwhere(foreground)
    if not coordinates.size:
        raise RuntimeError("The selected slice contains no foreground labels")
    x_min, z_min = coordinates.min(axis=0)
    x_max, z_max = coordinates.max(axis=0)
    span = max(x_max - x_min + 1, z_max - z_min + 1)
    padding = max(4, int(round(span * 0.055)))
    return (
        max(0, x_min - padding),
        min(reference_slice.shape[0], x_max + padding + 1),
        max(0, z_min - padding),
        min(reference_slice.shape[1], z_max + padding + 1),
    )


def crop(array, bounds):
    x0, x1, z0, z1 = bounds
    return array[x0:x1, z0:z1]


def label_rgb(label_slice, mapping):
    output = np.zeros(label_slice.shape + (3,), dtype=np.uint8)
    for label in np.unique(label_slice):
        label = int(label)
        if label == 0:
            continue
        if label not in mapping:
            raise RuntimeError(f"Label {label} is missing from label_mapping.csv")
        output[label_slice == label] = mapping[label]["rgb"]
    return output


def entropy_rgb(entropy_slice, maximum):
    normalised = np.clip(entropy_slice / maximum, 0.0, 1.0)
    rgba = plt.get_cmap("inferno")(normalised)
    rgb = np.round(rgba[..., :3] * 255).astype(np.uint8)
    rgb[~np.isfinite(entropy_slice)] = 0
    return rgb


def clean_label(name):
    return name.replace("-", " ").replace("WM", "white matter")


def legend_entries(visible_ids, mapping):
    return [
        (clean_label(mapping[label_id]["name"]), mapping[label_id]["rgb"], label_id, None)
        for label_id in sorted(visible_ids)
    ]


def save_exact_png(path, rgb):
    plt.imsave(path, rgb)


def make_legend(path_base, entries, title):
    columns = 4
    rows = max(1, math.ceil(len(entries) / columns))
    figure = plt.figure(figsize=(7.2, 0.38 + 0.28 * rows))
    handles = [
        Patch(
            facecolor=tuple(channel / 255 for channel in colour),
            edgecolor="black",
            linewidth=0.45,
            label=name,
        )
        for name, colour, _, _ in entries
    ]
    figure.legend(
        handles=handles,
        loc="center",
        ncol=columns,
        frameon=False,
        fontsize=7.0,
        title=title,
        title_fontsize=7.5,
        handlelength=1.5,
        columnspacing=1.4,
    )
    for extension in ("pdf", "png"):
        figure.savefig(
            path_base.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def make_combined_figure(path_base, raw_rgb, ref_rgb, pred_rgb, entropy_rgb_image,
                         target_entries, entropy_max):
    columns = 4
    legend_rows = max(1, math.ceil(len(target_entries) / columns))
    height = 3.05 + 0.25 * legend_rows
    figure, axes = plt.subplots(1, 4, figsize=(7.2, height))
    titles = (
        "A  T1-weighted MRI",
        "B  FreeSurfer reference",
        "C  Ensemble prediction",
        "D  Predictive entropy",
    )
    images = (raw_rgb, ref_rgb, pred_rgb, entropy_rgb_image)
    for axis, title, image in zip(axes, titles, images):
        axis.imshow(image, interpolation="nearest")
        axis.set_title(title, fontsize=8.2, fontweight="semibold", pad=4)
        axis.axis("off")

    colourbar = figure.colorbar(
        ScalarMappable(norm=Normalize(vmin=0.0, vmax=entropy_max), cmap="inferno"),
        ax=axes[3], orientation="horizontal", fraction=0.045, pad=0.03,
    )
    colourbar.set_label("Predictive entropy", fontsize=6.8)
    colourbar.ax.tick_params(labelsize=6.4, length=2)

    handles = [
        Patch(
            facecolor=tuple(channel / 255 for channel in colour),
            edgecolor="black",
            linewidth=0.45,
            label=name,
        )
        for name, colour, _, _ in target_entries
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=columns,
        frameon=False,
        fontsize=6.5,
        title="Visible target regions (left and right listed separately)",
        title_fontsize=7.2,
        handlelength=1.4,
        columnspacing=1.15,
    )
    bottom = min(0.50, 0.11 + 0.048 * legend_rows)
    figure.subplots_adjust(left=0.01, right=0.995, top=0.93, bottom=bottom, wspace=0.035)
    for extension in ("pdf", "png"):
        figure.savefig(
            path_base.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def latex_escape(text):
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
    return "".join(replacements.get(character, character) for character in text)


def make_appendix_tex(path, entries, figure_reference="fig:qualitative_example"):
    split_after = math.ceil(len(entries) / 2)
    lines = [
        r"\section{Labels Visible in the Qualitative Example}\label{ap:label_legend}",
        "",
        (
            "The labels visible in the reference or predicted segmentation in "
            f"Figure~\\ref{{{figure_reference}}} are listed below. Left and right "
            "structures are listed separately. Labels that do not appear in the "
            "displayed coronal slice are not included."
        ),
        "",
        r"\begingroup",
        r"\footnotesize",
        r"\raggedright",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.05ex}",
        r"\setlength{\fboxsep}{0pt}",
        r"\setlength{\fboxrule}{0.25pt}",
        r"\newcommand{\segentry}[5]{%",
        r"  \noindent\makebox[2.3em][r]{#1}\hspace{0.45em}%",
        r"  \fbox{\colorbox[RGB]{#2,#3,#4}{\phantom{\rule{1.8ex}{1.35ex}}}}%",
        r"  \hspace{0.55em}#5\par%",
        r"}",
        r"\textbf{ID}\hspace{0.7em}\textbf{Colour}\hspace{0.8em}\textbf{Visible label}\par",
        r"\vspace{0.35ex}",
    ]
    for index, (name, colour, label_id, _) in enumerate(entries):
        if index == split_after:
            lines.append(r"\newpage")
        lines.append(
            f"\\segentry{{{label_id}}}{{{colour[0]}}}{{{colour[1]}}}{{{colour[2]}}}"
            f"{{{latex_escape(name)}}}"
        )
    lines.extend([r"\endgroup", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    mapping_rows = read_csv(script_dir / "label_mapping.csv")
    target_rows = read_csv(script_dir / "target_15_labels.csv")
    mapping = {
        int(row["remapped_id"]): {
            "name": row["label_name"],
            "rgb": tuple(int(row[channel]) for channel in ("R", "G", "B")),
        }
        for row in mapping_rows
    }
    target_ids = {int(row["remapped_id"]) for row in target_rows}
    for label_id, colour in TARGET_DISPLAY_RGB.items():
        if label_id not in mapping:
            raise RuntimeError(f"Target label {label_id} is missing from label_mapping.csv")
        mapping[label_id]["rgb"] = colour

    subject = args.subject
    paths = {
        "raw": args.root / "nnunet_raw" / "Dataset001_IDEAS" / "imagesTr" / f"{subject}_0000.nii.gz",
        "reference": args.root / "nnunet_raw" / "Dataset001_IDEAS" / "labelsTr" / f"{subject}.nii.gz",
        "prediction": args.root / "uncertainty_output_internal_250" / f"{subject}_seg.nii.gz",
        "entropy": args.root / "uncertainty_output_internal_250" / f"{subject}_entropy.nii.gz",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    loaded = {name: load_image(path) for name, path in paths.items()}
    originals = {name: value[0] for name, value in loaded.items()}
    canonicals = {name: value[1] for name, value in loaded.items()}
    arrays = {name: value[2] for name, value in loaded.items()}

    reference_shape = canonicals["reference"].shape
    for name, image in canonicals.items():
        if image.shape != reference_shape:
            raise RuntimeError(f"Canonical shape mismatch for {name}: {image.shape} vs {reference_shape}")
        if not np.allclose(image.affine, canonicals["reference"].affine):
            raise RuntimeError(f"Canonical affine mismatch for {name}")

    reference = np.rint(arrays["reference"]).astype(np.int16)
    prediction = np.rint(arrays["prediction"]).astype(np.int16)
    unknown = (set(np.unique(reference)) | set(np.unique(prediction))) - set(mapping)
    if unknown:
        raise RuntimeError(f"Unknown remapped label IDs: {sorted(unknown)}")

    if args.coronal_slice is None:
        coronal_slice, selection_score = choose_slice(
            reference, prediction, target_ids, args.minimum_visible_pixels
        )
    else:
        coronal_slice = args.coronal_slice
        if not 0 <= coronal_slice < reference.shape[1]:
            raise ValueError(f"Coronal slice must be between 0 and {reference.shape[1] - 1}")
        selection_score = None

    raw_slice = arrays["raw"][:, coronal_slice, :]
    ref_slice = reference[:, coronal_slice, :]
    pred_slice = prediction[:, coronal_slice, :]
    entropy_slice = arrays["entropy"][:, coronal_slice, :].astype(np.float32)
    bounds = crop_bounds(ref_slice, pred_slice)

    raw_crop = crop(raw_slice, bounds)
    ref_crop = crop(ref_slice, bounds)
    pred_crop = crop(pred_slice, bounds)
    entropy_crop = crop(entropy_slice, bounds)
    raw_scaled, raw_low, raw_high = normalise_t1(raw_crop)

    raw_display = display_orientation(np.round(raw_scaled * 255).astype(np.uint8))
    raw_rgb = np.repeat(raw_display[..., None], 3, axis=2)
    ref_rgb = display_orientation(label_rgb(ref_crop, mapping))
    pred_rgb = display_orientation(label_rgb(pred_crop, mapping))
    entropy_display = display_orientation(entropy_rgb(entropy_crop, args.entropy_max))

    visible_counts = {}
    for label_id in sorted((set(np.unique(ref_crop)) | set(np.unique(pred_crop))) - {0}):
        count = int((ref_crop == label_id).sum() + (pred_crop == label_id).sum())
        if count >= args.minimum_visible_pixels:
            visible_counts[int(label_id)] = count
    all_visible_entries = legend_entries(set(visible_counts), mapping)
    visible_target_ids = set(visible_counts) & target_ids
    target_entries = legend_entries(visible_target_ids, mapping)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    save_exact_png(output / "qualitative_raw.png", raw_rgb)
    save_exact_png(output / "qualitative_reference.png", ref_rgb)
    save_exact_png(output / "qualitative_prediction.png", pred_rgb)
    save_exact_png(output / "qualitative_uncertainty.png", entropy_display)
    make_legend(
        output / "figure_3_1_target_legend",
        target_entries,
        "Visible target regions (left and right listed separately)",
    )
    make_combined_figure(
        output / "figure_3_1_coronal",
        raw_rgb,
        ref_rgb,
        pred_rgb,
        entropy_display,
        target_entries,
        args.entropy_max,
    )
    make_appendix_tex(output / "appendix_visible_labels.tex", all_visible_entries)

    with (output / "visible_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["remapped_id", "label_name", "R", "G", "B", "combined_reference_prediction_pixels"])
        for label_id in sorted(visible_counts):
            colour = mapping[label_id]["rgb"]
            writer.writerow([label_id, mapping[label_id]["name"], *colour, visible_counts[label_id]])

    with (output / "panel_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "canonical_coronal_slice", "x_start", "x_stop", "z_start", "z_stop", "entropy_min", "entropy_max", "t1_window_low", "t1_window_high"])
        writer.writerow([subject, coronal_slice, *bounds, 0.0, args.entropy_max, raw_low, raw_high])

    affine_equal_original = all(
        np.allclose(image.affine, originals["reference"].affine)
        for image in originals.values()
    )
    orientation_lines = [
        f"  {name}: {nib.aff2axcodes(image.affine)}"
        for name, image in originals.items()
    ]
    label_lines = [
        f"  {label_id}: {mapping[label_id]['name']} ({visible_counts[label_id]} pixels across reference and prediction)"
        for label_id in sorted(visible_counts)
    ]
    summary = [
        "FIGURE 3.1 CORONAL REPLACEMENT CHECK",
        f"Subject: {subject}",
        f"Canonical image shape: {reference_shape}",
        f"Selected canonical coronal slice: {coronal_slice}",
        f"Automatic selection score (target labels, all labels, foreground pixels): {selection_score}",
        f"Crop bounds (x0, x1, z0, z1): {bounds}",
        f"Original affines all equal: {affine_equal_original}",
        "Original orientation codes:",
        *orientation_lines,
        f"Visible target labels included in the figure legend: {len(target_entries)}",
        f"All visible labels included in the Appendix B source: {len(all_visible_entries)}",
        *label_lines,
        "",
        "The reference and prediction panels use the exact RGB values from label_mapping.csv.",
        "The figure legend contains only visible target regions.",
        "Appendix B contains the union of labels visible in the reference and prediction.",
        "Left and right structures are listed separately in both legends.",
        "The 15 target regions use separate left/right display colours in this figure.",
        "All four panels use the same canonical coronal slice and crop.",
    ]
    (output / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
