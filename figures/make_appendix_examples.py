#!/usr/bin/env python3
"""Select and render the additional Appendix C examples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-disjoint-appendix-c")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import make_qualitative_examples as base


ROLE_ORDER = [
    "expected_success",
    "detected_failure",
    "false_alarm",
    "moderate_disagreement",
]
ROLE_SELECTION_ORDER = [
    "moderate_disagreement",
    "moderate_disagreement",
    "detected_failure",
    "detected_failure",
    "false_alarm",
    "false_alarm",
    "expected_success",
    "expected_success",
]
ROLE_DISPLAY = {
    "expected_success": "High Dice, low entropy",
    "detected_failure": "Lower Dice, high entropy",
    "false_alarm": "High Dice, high entropy",
    "moderate_disagreement": "Moderate disagreement",
}

HIGH_DICE_MINIMUM = 0.80
LOW_DICE_RANGE = (0.35, 0.65)
MODERATE_DICE_RANGE = (0.65, 0.80)
HIGH_DICE_PERCENTILE = 0.75
LOW_DICE_PERCENTILE = 0.35
LOW_ENTROPY_PERCENTILE = 0.25
HIGH_ENTROPY_PERCENTILE = 0.75
DETECTED_ENTROPY_PERCENTILE = 0.65
MODERATE_ENTROPY_PERCENTILE = 0.60

CASE_LETTERS = list("ABCDEFGH")
BEAM_WIDTH = 1200
POOL_PER_ROLE = 28
REQUIRED_EXCLUDED_SUBJECTS = {
    "IDEAS_194",
    "IDEAS_4068",
    "IDEAS_4069",
    "IDEAS_151",
    "IDEAS_354",
    "IDEAS_4018",
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
        help="Final disjoint directory containing the CSV and *_seg/_entropy NIfTIs",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--top-n-qc", type=int, default=6)
    parser.add_argument("--shortlist-per-role", type=int, default=100)
    parser.add_argument(
        "--exclude-subject",
        action="append",
        default=[],
        help="Subject already used in a main-text figure; may be repeated",
    )
    return parser.parse_args()


def region_family(region: str) -> str:
    if region.startswith("L-") or region.startswith("R-"):
        return region[2:]
    return region


def laterality(region: str) -> str:
    if region.startswith("L-"):
        return "L"
    if region.startswith("R-"):
        return "R"
    return "midline"


def role_mask(table: pd.DataFrame, role: str) -> pd.Series:
    high_dice = (table["dice"] >= HIGH_DICE_MINIMUM) & (
        table["dice_pct"] >= HIGH_DICE_PERCENTILE
    )
    masks = {
        "expected_success": high_dice
        & (table["entropy_pct"] <= LOW_ENTROPY_PERCENTILE),
        "detected_failure": table["dice"].between(*LOW_DICE_RANGE)
        & (table["dice_pct"] <= LOW_DICE_PERCENTILE)
        & (table["entropy_pct"] >= DETECTED_ENTROPY_PERCENTILE),
        "false_alarm": high_dice
        & (table["entropy_pct"] >= HIGH_ENTROPY_PERCENTILE),
        "moderate_disagreement": table["dice"].between(*MODERATE_DICE_RANGE)
        & (table["entropy_pct"] >= MODERATE_ENTROPY_PERCENTILE),
    }
    return masks[role]


def bounded_closeness(values: pd.Series, centre: float, half_width: float) -> pd.Series:
    return (1.0 - (values - centre).abs() / half_width).clip(lower=0.0)


def role_score(table: pd.DataFrame, role: str) -> pd.Series:
    scores = {
        "expected_success": table["dice_pct"] + (1.0 - table["entropy_pct"]),
        "detected_failure": (1.0 - table["dice_pct"])
        + table["entropy_pct"]
        + 0.35 * bounded_closeness(table["dice"], 0.50, 0.15),
        "false_alarm": table["dice_pct"] + table["entropy_pct"],
        "moderate_disagreement": (1.0 - table["dice_pct"])
        + table["entropy_pct"]
        + 0.35 * bounded_closeness(table["dice"], 0.725, 0.075),
    }
    return scores[role]


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
        candidates["subject_mean_dice"] >= base.FALLBACK_SUBJECT_MEAN_DICE
    ].copy()
    candidates["subject_performance_tier"] = np.select(
        [
            candidates["subject_mean_dice"] >= base.PREFERRED_SUBJECT_MEAN_DICE,
            candidates["subject_mean_dice"] >= base.RELAXED_SUBJECT_MEAN_DICE,
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
        base.spatial_metrics(ideas_root, uncertainty_dir, row)
        for _, row in candidates.iterrows()
    ]
    candidates = pd.concat(
        [candidates.reset_index(drop=True), pd.DataFrame(metrics)], axis=1
    )
    candidates = base.assign_spatial_rule(candidates)
    candidates["region_family"] = candidates["region"].map(region_family)
    candidates["laterality"] = candidates["region"].map(laterality)
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
    eligible = candidates.loc[
        (candidates["spatial_tier"] > 0)
        & ~candidates["excluded_for_figure_diversity"]
    ]
    if eligible["subject_id"].nunique() < 2 or eligible["region"].nunique() < 2:
        raise RuntimeError(f"Fewer than two eligible candidates for {role}")
    return candidates


def state_score(rows: list[pd.Series]) -> tuple[float, ...]:
    families = {str(row["region_family"]) for row in rows}
    lateralities = [str(row["laterality"]) for row in rows]
    left_right_balance = min(lateralities.count("L"), lateralities.count("R"))
    return (
        float(sum(int(row["spatial_tier"]) for row in rows)),
        float(len(families)),
        float(left_right_balance),
        float(sum(float(row["role_score"]) for row in rows)),
        float(sum(float(row["subject_mean_dice"]) for row in rows)),
        float(-sum(int(row["candidate_rank"]) for row in rows)),
    )


def select_distinct_examples(shortlists: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pools: dict[str, list[pd.Series]] = {}
    for role in ROLE_ORDER:
        eligible = shortlists[role].loc[
            (shortlists[role]["spatial_tier"] > 0)
            & ~shortlists[role]["excluded_for_figure_diversity"]
        ].head(POOL_PER_ROLE)
        pools[role] = [row for _, row in eligible.iterrows()]
        if len(pools[role]) < 2:
            raise RuntimeError(f"No two-case selection pool for {role}")

    states: list[list[pd.Series]] = [[]]
    for role in ROLE_SELECTION_ORDER:
        expanded: list[list[pd.Series]] = []
        for rows in states:
            used_subjects = {str(row["subject_id"]) for row in rows}
            used_regions = {str(row["region"]) for row in rows}
            same_role_rows = [row for row in rows if str(row["role"]) == role]
            same_role_families = {
                str(row["region_family"])
                for row in same_role_rows
            }
            for candidate in pools[role]:
                if str(candidate["subject_id"]) in used_subjects:
                    continue
                if str(candidate["region"]) in used_regions:
                    continue
                if str(candidate["region_family"]) in same_role_families:
                    continue
                if same_role_rows and int(candidate["candidate_rank"]) <= max(
                    int(row["candidate_rank"]) for row in same_role_rows
                ):
                    continue
                expanded.append(rows + [candidate])
        if not expanded:
            raise RuntimeError(
                f"Could not extend eight-case selection at role {role}; "
                "inspect the candidate shortlists"
            )
        expanded.sort(key=state_score, reverse=True)
        deduplicated: list[list[pd.Series]] = []
        signatures: set[tuple[tuple[str, str], ...]] = set()
        for rows in expanded:
            signature = tuple(
                (str(row["subject_id"]), str(row["region"])) for row in rows
            )
            if signature in signatures:
                continue
            signatures.add(signature)
            deduplicated.append(rows)
            if len(deduplicated) >= BEAM_WIDTH:
                break
        states = deduplicated

    best = max(states, key=state_score)
    by_role: dict[str, list[pd.Series]] = {}
    for role in ROLE_ORDER:
        by_role[role] = sorted(
            [row.copy() for row in best if str(row["role"]) == role],
            key=lambda row: (int(row["candidate_rank"]), str(row["subject_id"])),
        )
        if len(by_role[role]) != 2:
            raise RuntimeError(f"Selection did not produce two examples for {role}")

    display_rows = [row for role in ROLE_ORDER for row in by_role[role]]
    selected = pd.DataFrame(display_rows).reset_index(drop=True)
    selected["case_letter"] = CASE_LETTERS
    selected["selected_before_visual_qc"] = True
    if selected["subject_id"].nunique() != 8:
        raise RuntimeError("Appendix C selection did not produce eight distinct subjects")
    if selected["region"].nunique() != 8:
        raise RuntimeError("Appendix C selection did not produce eight distinct sided regions")
    if selected.groupby("role").size().to_dict() != {role: 2 for role in ROLE_ORDER}:
        raise RuntimeError("Appendix C selection did not produce two cases per role")
    return selected


def draw_case_panels(
    axes: np.ndarray,
    data: dict[str, object],
    add_column_titles: bool = True,
) -> object:
    flat = np.asarray(axes).reshape(-1)
    if len(flat) != 4:
        raise ValueError("Expected exactly four panel axes")
    titles = [
        "T1-weighted MRI",
        "Reference target",
        "Ensemble prediction",
        "Predictive entropy",
    ]
    raw = data["raw"]
    vmin, vmax = data["display_limits"]
    orange = ListedColormap([base.MASK_COLOUR])
    for axis in flat:
        axis.set_xticks([])
        axis.set_yticks([])
    flat[0].imshow(raw, cmap="gray", vmin=vmin, vmax=vmax)
    flat[1].imshow(raw, cmap="gray", vmin=vmin, vmax=vmax)
    flat[1].imshow(
        np.ma.masked_where(~data["ref_mask"], data["ref_mask"]),
        cmap=orange,
        alpha=0.72,
        interpolation="nearest",
    )
    flat[2].imshow(raw, cmap="gray", vmin=vmin, vmax=vmax)
    flat[2].imshow(
        np.ma.masked_where(~data["pred_mask"], data["pred_mask"]),
        cmap=orange,
        alpha=0.72,
        interpolation="nearest",
    )
    entropy_image = flat[3].imshow(
        data["entropy"],
        cmap="magma",
        vmin=base.ENTROPY_VMIN,
        vmax=base.ENTROPY_VMAX,
        interpolation="nearest",
    )
    if data["ref_mask"].any():
        flat[3].contour(
            data["ref_mask"],
            levels=[0.5],
            colors=[base.REFERENCE_COLOUR],
            linewidths=1.1,
        )
    if data["pred_mask"].any():
        flat[3].contour(
            data["pred_mask"],
            levels=[0.5],
            colors=[base.PREDICTION_COLOUR],
            linewidths=1.1,
        )
    if add_column_titles:
        for axis, title in zip(flat, titles):
            axis.set_title(title, fontweight="bold", pad=6)
    return entropy_image


def contour_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=base.REFERENCE_COLOUR, lw=2, label="Reference contour"),
        Line2D([0], [0], color=base.PREDICTION_COLOUR, lw=2, label="Prediction contour"),
    ]


def plot_rows(
    rows: pd.DataFrame,
    prepared_cases: list[dict[str, object]],
    title: str,
    output_png: Path | None = None,
    show_candidate_rank: bool = False,
) -> plt.Figure:
    nrows = len(rows)
    fig, axes = plt.subplots(
        nrows,
        4,
        figsize=(12.8, 2.55 * nrows + 1.15),
        squeeze=False,
    )
    entropy_image = None
    for row_index, ((_, row), data) in enumerate(zip(rows.iterrows(), prepared_cases)):
        entropy_image = draw_case_panels(
            axes[row_index], data, add_column_titles=(row_index == 0)
        )
        prefix = (
            f"Rank {int(row['candidate_rank'])}  " if show_candidate_rank else ""
        )
        row_text = (
            f"{prefix}{row['subject_id']}, {row['region']}\n"
            f"Dice={row['dice']:.3f}; entropy={row['entropy']:.3f}\n"
            f"mean Dice={row['subject_mean_dice']:.3f}; {row['spatial_rule']}"
        )
        axes[row_index, 0].set_ylabel(
            row_text, rotation=0, ha="right", va="center", labelpad=12
        )

    fig.subplots_adjust(
        left=0.235,
        right=0.925,
        top=0.90,
        bottom=0.075,
        hspace=0.22,
        wspace=0.04,
    )
    colourbar = fig.colorbar(
        entropy_image,
        ax=axes[:, 3],
        fraction=0.028,
        pad=0.015,
        ticks=base.ENTROPY_TICKS,
        label="Predictive entropy",
    )
    colourbar.ax.set_yticklabels(["0.0", "0.5", "1.0", "1.3"])
    fig.legend(
        handles=contour_handles(),
        loc="lower center",
        bbox_to_anchor=(0.56, 0.012),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.985)
    if output_png is not None:
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
    return fig


def plot_individual_case(
    row: pd.Series,
    data: dict[str, object],
    output_pdf: Path,
    output_png: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(5.3, 5.8))
    draw_case_panels(axes, data, add_column_titles=True)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.08, hspace=0.20, wspace=0.08)
    fig.legend(
        handles=contour_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def render_outputs(
    selected: pd.DataFrame,
    shortlists: dict[str, pd.DataFrame],
    ideas_root: Path,
    uncertainty_dir: Path,
    outdir: Path,
    top_n_qc: int,
) -> pd.DataFrame:
    selected_prepared, selected_audits = base.prepare_rows(
        selected, ideas_root, uncertainty_dir
    )
    for (_, row), data in zip(selected.iterrows(), selected_prepared):
        letter = str(row["case_letter"])
        plot_individual_case(
            row,
            data,
            outdir / f"case_{letter}_grid.pdf",
            outdir / f"case_{letter}_grid.png",
        )

    all_audits = list(selected_audits)
    with PdfPages(outdir / "appendixC_disjoint_selected_overview.pdf") as selected_pdf:
        for role in ROLE_ORDER:
            role_rows = selected.loc[selected["role"] == role].reset_index(drop=True)
            prepared, _ = base.prepare_rows(role_rows, ideas_root, uncertainty_dir)
            role_fig = plot_rows(
                role_rows,
                prepared,
                f"Appendix C selected cases: {ROLE_DISPLAY[role]}",
                outdir / f"selected_{role.replace('_', '-')}.png",
            )
            selected_pdf.savefig(role_fig, bbox_inches="tight")
            plt.close(role_fig)

    with PdfPages(outdir / "appendixC_disjoint_qc.pdf") as qc_pdf:
        for role in ROLE_ORDER:
            candidates = shortlists[role].loc[
                (shortlists[role]["spatial_tier"] > 0)
                & ~shortlists[role]["excluded_for_figure_diversity"]
            ].head(top_n_qc)
            selected_role = selected.loc[selected["role"] == role]
            selected_ids = set(zip(selected_role["subject_id"], selected_role["region"]))
            present_ids = set(zip(candidates["subject_id"], candidates["region"]))
            if not selected_ids.issubset(present_ids):
                candidates = pd.concat([candidates, selected_role], ignore_index=True)
                candidates = candidates.drop_duplicates(["subject_id", "region"]).sort_values(
                    "candidate_rank"
                )
            prepared, audits = base.prepare_rows(
                candidates, ideas_root, uncertainty_dir
            )
            all_audits.extend(audits)
            role_fig = plot_rows(
                candidates,
                prepared,
                f"Appendix C visual-QC candidates: {ROLE_DISPLAY[role]}",
                outdir / f"qc_{role.replace('_', '-')}.png",
                show_candidate_rank=True,
            )
            qc_pdf.savefig(role_fig, bbox_inches="tight")
            plt.close(role_fig)

    audit = pd.DataFrame(all_audits).drop_duplicates(
        ["role", "candidate_rank", "subject_id", "region"]
    )
    return audit.sort_values(["role", "candidate_rank"]).reset_index(drop=True)


def human_region(region: str) -> str:
    if region == "Brain-Stem":
        return "brainstem"
    side = ""
    name = region
    if region.startswith("L-"):
        side, name = "left ", region[2:]
    elif region.startswith("R-"):
        side, name = "right ", region[2:]
    return side + name.replace("Cerebral-WM", "cerebral white matter").lower()


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_appendix_tex(selected: pd.DataFrame, outdir: Path) -> None:
    lines = [
        r"\section{Additional Qualitative Segmentation Examples}\label{ap:additional_examples}",
        "",
        (
            "The eight cases below are additional to the four examples in "
            "Figure~\\ref{fig:qualitative_examples} and were selected from the final "
            "leakage-free four-model disjoint ensemble after excluding all subjects "
            "used in the main-text qualitative figures. They comprise two examples "
            "each of high Dice with low entropy, lower Dice with high entropy, high "
            "Dice with high entropy, and moderate disagreement. The eight subjects "
            "and sided regions are distinct. Each case shows the T1-weighted MRI, "
            "FreeSurfer-derived reference target, ensemble prediction, and predictive "
            "entropy map for the same coronal slice. Orange marks the selected "
            "structure, while green and cyan contours show the reference and "
            "prediction. All entropy maps use the same 0 to 1.3 range. These selected "
            "examples illustrate uncertainty--error patterns, not their cohort frequency."
        ),
        "",
    ]
    for role_index, role in enumerate(ROLE_ORDER):
        lines.append(rf"\subsection*{{{ROLE_DISPLAY[role]}}}")
        lines.append("")
        role_rows = selected.loc[selected["role"] == role]
        for pair_index, (_, row) in enumerate(role_rows.iterrows()):
            letter = str(row["case_letter"])
            lines.extend(
                [
                    r"\noindent\begin{minipage}{\textwidth}",
                    r"\centering",
                    rf"\includegraphics[width=0.43\textwidth]{{img/case_{letter}_grid.pdf}}",
                    r"\par\smallskip",
                    (
                        rf"\raggedright\small\textbf{{Case {letter}:}} "
                        f"{latex_escape(str(row['subject_id']))}, "
                        f"{human_region(str(row['region']))}. "
                        rf"Dice $={row['dice']:.3f}$; entropy $={row['entropy']:.3f}$. "
                        f"{ROLE_DISPLAY[role]}."
                    ),
                    r"\end{minipage}",
                ]
            )
            if pair_index == 0:
                lines.extend(["", r"\vspace{1em}", ""])
        if role_index < len(ROLE_ORDER) - 1:
            lines.extend(["", r"\clearpage", ""])
    (outdir / "appendixC_disjoint.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.top_n_qc < 2 or args.shortlist_per_role < args.top_n_qc:
        raise ValueError("Require shortlist-per-role >= top-n-qc >= 2")

    ideas_root = args.ideas_root.resolve()
    uncertainty_dir = args.uncertainty_dir.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = uncertainty_dir / "region_entropy_decomposed_reference_masked.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    wide = pd.read_csv(csv_path)
    inventory = base.validate_input_inventory(ideas_root, uncertainty_dir, wide)
    candidates = base.build_long_table(wide)
    excluded_subjects = set(map(str, args.exclude_subject))
    missing_exclusions = REQUIRED_EXCLUDED_SUBJECTS - excluded_subjects
    if missing_exclusions:
        raise RuntimeError(
            "Appendix C must exclude every selected main-text qualitative subject; "
            f"missing --exclude-subject values: {sorted(missing_exclusions)}"
        )
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
        "region_family",
        "laterality",
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
    all_shortlists.to_csv(
        outdir / "appendixC_disjoint_candidate_shortlists.csv", index=False
    )
    selected[["case_letter"] + ranking_columns + ["selected_before_visual_qc"]].to_csv(
        outdir / "appendixC_disjoint_selected_examples.csv", index=False
    )
    audit.to_csv(outdir / "appendixC_disjoint_array_verification.csv", index=False)
    write_appendix_tex(selected, outdir)

    metadata = {
        "input_csv": str(csv_path),
        "expected_subjects": base.EXPECTED_SUBJECTS,
        "target_regions": list(base.TARGET_LABELS),
        "input_file_counts": inventory,
        "excluded_subjects_for_figure_diversity": sorted(excluded_subjects),
        "selection_bands": {
            "high_dice_minimum": HIGH_DICE_MINIMUM,
            "lower_dice_range": LOW_DICE_RANGE,
            "moderate_dice_range": MODERATE_DICE_RANGE,
            "high_dice_percentile": HIGH_DICE_PERCENTILE,
            "low_dice_percentile": LOW_DICE_PERCENTILE,
            "low_entropy_percentile": LOW_ENTROPY_PERCENTILE,
            "high_entropy_percentile": HIGH_ENTROPY_PERCENTILE,
            "detected_failure_entropy_percentile": DETECTED_ENTROPY_PERCENTILE,
            "moderate_disagreement_entropy_percentile": MODERATE_ENTROPY_PERCENTILE,
        },
        "selection_constraints": [
            "eight distinct subjects",
            "eight distinct sided regions",
            "two different region families within each pattern",
            "subjects from all main-text qualitative figures excluded",
        ],
        "slice_rule": "largest reference-region coronal cross-section",
        "entropy_display_range": [base.ENTROPY_VMIN, base.ENTROPY_VMAX],
        "visual_qc_exclusions": [
            "gross acquisition artefact",
            "gross image/reference misregistration",
            "missing or unreadable required panel",
        ],
    }
    (outdir / "appendixC_disjoint_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "DISJOINT APPENDIX C EIGHT-CASE SELECTION CHECK",
        f"Input: {csv_path}",
        f"Subjects: {wide['subject_id'].nunique()}",
        f"Candidate subject-region rows: {len(candidates)}",
        f"Target regions: {len(base.TARGET_LABELS)}",
        f"Input file counts: {inventory}",
        f"Excluded subjects for figure diversity: {sorted(excluded_subjects)}",
        "Slice rule: largest reference-region coronal cross-section",
        f"Entropy display range: {base.ENTROPY_VMIN:.1f} to {base.ENTROPY_VMAX:.1f}",
        "",
        "Preselected examples before visual QC:",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"Case {row['case_letter']} - {ROLE_DISPLAY[row['role']]}: "
            f"{row['subject_id']}, {row['region']}, "
            f"Dice={row['dice']:.6f}, entropy={row['entropy']:.6f}, "
            f"Dice percentile={row['dice_pct']:.3f}, "
            f"entropy percentile={row['entropy_pct']:.3f}, "
            f"subject mean Dice={row['subject_mean_dice']:.3f}, "
            f"centroid distance={row['centroid_distance_mm']:.2f} mm, "
            f"volume ratio={row['prediction_reference_volume_ratio']:.3f}, "
            f"spatial rule={row['spatial_rule']}, "
            f"candidate rank={int(row['candidate_rank'])}"
        )
    lines.extend(
        [
            "",
            "All rendered Dice values matched direct NIfTI recomputation.",
            "All rendered entropy values matched direct reference-masked recomputation.",
            "All rendered raw, reference, prediction and entropy arrays matched in shape and voxel grid.",
            "Eight distinct subjects and eight distinct sided regions were selected.",
            "Each pattern contains two different region families.",
            "Poor segmentation and extreme uncertainty are not visual-QC exclusion criteria.",
            "Permissible exclusions: gross acquisition artefact, gross image/reference misregistration, or a missing/unreadable required panel.",
        ]
    )
    summary = "\n".join(lines) + "\n"
    (outdir / "appendixC_disjoint_summary.txt").write_text(
        summary, encoding="utf-8"
    )
    print(summary, end="")


if __name__ == "__main__":
    main()
