"""Generate the RQ2 effect-size heatmaps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "supporting_tables"
    / "final_disjoint"
    / "rq2_all_tests.csv"
)
OUTPUT_PDF = ROOT / "img" / "rq2_effect_overview.pdf"
OUTPUT_PNG = ROOT / "img" / "rq2_effect_overview_preview.png"

REGIONS = [
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

REGION_LABELS = {
    "Brain-Stem": "Brainstem",
    "L-Cerebral-WM": "Left cerebral white matter",
    "R-Cerebral-WM": "Right cerebral white matter",
    "L-Thalamus": "Left thalamus",
    "R-Thalamus": "Right thalamus",
    "L-Caudate": "Left caudate",
    "R-Caudate": "Right caudate",
    "L-Putamen": "Left putamen",
    "R-Putamen": "Right putamen",
    "L-Pallidum": "Left pallidum",
    "R-Pallidum": "Right pallidum",
    "L-Hippocampus": "Left hippocampus",
    "R-Hippocampus": "Right hippocampus",
    "L-Amygdala": "Left amygdala",
    "R-Amygdala": "Right amygdala",
}

BINARY = [
    ("Sex", "Sex\nF vs M"),
    ("Op_Side", "Op. side\nL vs R"),
    ("Op_Type_collapsed", "Op. type\nExtra vs\ntemp."),
    ("ILAE_Y1_seizure_free", "Year 1\nNot free\nvs free"),
]

ORDERED = [
    ("Binned_Age_at_Scan", "Age band"),
    ("Number_ASMs", "Number of ASMs"),
]

PATHOLOGY = [("Pathology_grp", "Pathology")]

TEXT = "#263544"
GRID = "#D8DEE4"
DIVERGING = LinearSegmentedColormap.from_list(
    "muted_blue_orange", ["#C87533", "#F7F7F5", "#3E78A3"], N=256
)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "muted_blues", ["#F7F7F5", "#AFC9DC", "#3E78A3"], N=256
)


def matrix(data: pd.DataFrame, variables: list[tuple[str, str]]):
    values = np.full((len(REGIONS), len(variables)), np.nan)
    nominal = np.zeros_like(values, dtype=bool)
    for column, (variable, _) in enumerate(variables):
        subset = data[data["variable"] == variable].set_index("region")
        for row, region in enumerate(REGIONS):
            values[row, column] = float(subset.loc[region, "effect"])
            nominal[row, column] = (
                float(subset.loc[region, "p_group_adjusted_permutation"]) < 0.05
            )
    return values, nominal


def draw_heatmap(
    axis,
    values,
    nominal,
    labels,
    title,
    *,
    cmap,
    vmin,
    vmax,
    show_ylabels=False,
):
    image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axis.set_title(title, loc="left", fontsize=8.5, fontweight="bold", color=TEXT, pad=10)
    axis.set_xticks(np.arange(len(labels)), labels=labels, fontsize=7.0)
    axis.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=5)
    axis.set_yticks(np.arange(len(REGIONS)))
    if show_ylabels:
        axis.set_yticklabels([REGION_LABELS[r] for r in REGIONS], fontsize=7.6)
        axis.tick_params(axis="y", length=0, pad=5)
    else:
        axis.set_yticklabels([])
        axis.tick_params(axis="y", length=0)

    axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(REGIONS), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_edgecolor(GRID)
        spine.set_linewidth(0.8)

    rows, columns = np.where(nominal)
    axis.scatter(
        columns,
        rows,
        s=22,
        facecolors="none",
        edgecolors="#111111",
        linewidths=1.0,
        zorder=3,
    )
    return image


def main():
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    if len(data) != 105 or data[["variable", "region"]].duplicated().any():
        raise ValueError("Expected exactly 105 unique variable-region tests")
    if not set(REGIONS).issubset(set(data["region"])):
        raise ValueError("RQ2 source is missing one or more target regions")

    binary_values, binary_nominal = matrix(data, BINARY)
    ordered_values, ordered_nominal = matrix(data, ORDERED)
    pathology_values, pathology_nominal = matrix(data, PATHOLOGY)

    nominal_count = int(
        (data["p_group_adjusted_permutation"].astype(float) < 0.05).sum()
    )
    fdr_count = int((data["p_fdr_group_adjusted_105"].astype(float) < 0.05).sum())
    if nominal_count != 9 or fdr_count != 0:
        raise ValueError(
            f"Unexpected significance counts: nominal={nominal_count}, FDR={fdr_count}"
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )

    figure = plt.figure(figsize=(7.25, 5.30), facecolor="white")
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=[4.0, 2.0, 1.15],
        left=0.255,
        right=0.975,
        top=0.735,
        bottom=0.135,
        wspace=0.14,
    )
    axes = [figure.add_subplot(grid[0, i]) for i in range(3)]

    signed_limit = 0.20
    binary_image = draw_heatmap(
        axes[0],
        binary_values,
        binary_nominal,
        [label for _, label in BINARY],
        r"A   $r_{rb}$",
        cmap=DIVERGING,
        vmin=-signed_limit,
        vmax=signed_limit,
        show_ylabels=True,
    )
    draw_heatmap(
        axes[1],
        ordered_values,
        ordered_nominal,
        [label for _, label in ORDERED],
        r"B   $\rho$",
        cmap=DIVERGING,
        vmin=-signed_limit,
        vmax=signed_limit,
    )
    pathology_limit = 0.020
    pathology_image = draw_heatmap(
        axes[2],
        pathology_values,
        pathology_nominal,
        [label for _, label in PATHOLOGY],
        r"C   $\epsilon^2$",
        cmap=SEQUENTIAL,
        vmin=0.0,
        vmax=pathology_limit,
    )

    figure.suptitle(
        "Clinical and demographic screening across 15 target regions",
        x=0.255,
        y=0.975,
        ha="left",
        fontsize=10.8,
        fontweight="bold",
        color=TEXT,
    )
    figure.text(
        0.255,
        0.910,
        "Effect sizes from 105 group-adjusted tests; circles mark raw permutation p < 0.05. "
        "No test survived pooled FDR correction.",
        ha="left",
        va="top",
        fontsize=8.0,
        color="#4D5C68",
    )

    signed_bar_axis = figure.add_axes([0.315, 0.060, 0.310, 0.020])
    signed_bar = figure.colorbar(binary_image, cax=signed_bar_axis, orientation="horizontal")
    signed_bar.set_ticks([-0.20, 0.0, 0.20])
    signed_bar.ax.tick_params(labelsize=7, length=2, pad=2)
    signed_bar.set_label("Signed effect size", fontsize=7.3, color=TEXT, labelpad=2)
    signed_bar.outline.set_linewidth(0.6)

    pathology_bar_axis = figure.add_axes([0.770, 0.060, 0.155, 0.020])
    pathology_bar = figure.colorbar(
        pathology_image, cax=pathology_bar_axis, orientation="horizontal"
    )
    pathology_bar.set_ticks([0.0, 0.01, 0.02])
    pathology_bar.ax.tick_params(labelsize=7, length=2, pad=2)
    pathology_bar.set_label("Pathology effect size", fontsize=7.3, color=TEXT, labelpad=2)
    pathology_bar.outline.set_linewidth(0.6)

    nominal_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="none",
        markeredgecolor="#111111",
        markersize=5.2,
        label="Raw group-adjusted p < 0.05",
    )
    figure.legend(
        handles=[nominal_handle],
        loc="lower left",
        bbox_to_anchor=(0.010, 0.038),
        frameon=False,
        fontsize=7.5,
        handletextpad=0.4,
    )

    figure.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(OUTPUT_PNG, dpi=260, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


if __name__ == "__main__":
    main()
