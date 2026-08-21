"""Generate the restrained disjoint-ensemble pipeline for Figure 2.1."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PDF = ROOT / "img" / "disjoint_ensemble_pipeline.pdf"
OUTPUT_PNG = ROOT / "img" / "disjoint_ensemble_pipeline_preview.png"


TEXT = "#1D2A38"
LINE = "#586979"
BLUE_FILL = "#EAF2F8"
BLUE_EDGE = "#4B7EA5"
GREEN_FILL = "#E7F2EE"
GREEN_EDGE = "#4B8978"
GREY_FILL = "#F4F6F7"
GREY_EDGE = "#8A98A5"
EXCLUDE_FILL = "#F8EEEE"
EXCLUDE_EDGE = "#B45B5B"


def box(
    ax,
    x,
    y,
    width,
    height,
    text,
    *,
    facecolor="white",
    edgecolor=GREY_EDGE,
    fontsize=10.3,
    fontweight="normal",
    linewidth=1.35,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=linewidth,
        facecolor=facecolor,
        edgecolor=edgecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=TEXT,
        linespacing=1.18,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, *, connectionstyle="arc3", linewidth=1.45):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=linewidth,
        color=LINE,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def main():
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.20, 5.05))
    fig.patch.set_facecolor("white")
    ax.set_position([0.015, 0.020, 0.970, 0.960])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(
        ax,
        0.020,
        0.705,
        0.180,
        0.205,
        "Split 532 subjects\ninto five disjoint\ngroups",
        facecolor=BLUE_FILL,
        edgecolor=BLUE_EDGE,
        fontsize=9.7,
    )
    box(
        ax,
        0.245,
        0.705,
        0.190,
        0.205,
        "Train one model\non each group",
        facecolor=BLUE_FILL,
        edgecolor=BLUE_EDGE,
    )
    box(
        ax,
        0.480,
        0.705,
        0.190,
        0.205,
        "For each subject,\nexclude its group's\nmodel",
        facecolor="white",
        edgecolor=BLUE_EDGE,
        fontweight="normal",
    )
    box(
        ax,
        0.715,
        0.705,
        0.270,
        0.205,
        "Average the other four\nsoftmax outputs\n$\\bar{p}_s=\\frac{1}{4}\\sum_{m\\ne g}p_m$",
        facecolor=GREEN_FILL,
        edgecolor=GREEN_EDGE,
        fontsize=9.7,
    )

    arrow(ax, (0.200, 0.8075), (0.245, 0.8075))
    arrow(ax, (0.435, 0.8075), (0.480, 0.8075))
    arrow(ax, (0.670, 0.8075), (0.715, 0.8075))

    ax.text(
        0.020,
        0.635,
        "Example for a subject in group $G_0$",
        ha="left",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=TEXT,
    )
    chip_y = 0.515
    chip_h = 0.090
    chip_w = 0.125
    chip_gap = 0.012
    chip_x0 = 0.020
    chip_text = [
        ("$M_0$\nEXCLUDE", EXCLUDE_FILL, EXCLUDE_EDGE),
        ("$M_1$\nUSE", GREEN_FILL, GREEN_EDGE),
        ("$M_2$\nUSE", GREEN_FILL, GREEN_EDGE),
        ("$M_3$\nUSE", GREEN_FILL, GREEN_EDGE),
        ("$M_4$\nUSE", GREEN_FILL, GREEN_EDGE),
    ]
    for index, (label, fill, edge) in enumerate(chip_text):
        x = chip_x0 + index * (chip_w + chip_gap)
        box(
            ax,
            x,
            chip_y,
            chip_w,
            chip_h,
            label,
            facecolor=fill,
            edgecolor=edge,
            fontsize=7.8,
            linewidth=1.15,
        )
        if index == 0:
            ax.plot(
                [x + 0.020, x + chip_w - 0.020],
                [chip_y + 0.018, chip_y + chip_h - 0.018],
                color=EXCLUDE_EDGE,
                linewidth=1.5,
                zorder=4,
            )

    box(
        ax,
        0.110,
        0.090,
        0.215,
        0.165,
        "Ensemble segmentation\nand Dice error",
        facecolor=GREY_FILL,
        edgecolor=GREY_EDGE,
        fontsize=9.4,
    )
    box(
        ax,
        0.390,
        0.090,
        0.230,
        0.165,
        "Regional predictive\nentropy\n(reference masks)",
        facecolor=GREEN_FILL,
        edgecolor=GREEN_EDGE,
        fontsize=9.4,
    )
    box(
        ax,
        0.685,
        0.090,
        0.275,
        0.165,
        "Failure detection\n(predicted target voxels)",
        facecolor=GREY_FILL,
        edgecolor=GREY_EDGE,
        fontsize=9.4,
    )

    branch_y = 0.385
    ax.plot([0.2175, 0.850], [branch_y, branch_y], color=LINE, linewidth=1.45, zorder=1)
    ax.plot([0.850, 0.850], [0.705, branch_y], color=LINE, linewidth=1.45, zorder=1)
    arrow(ax, (0.2175, branch_y), (0.2175, 0.255))
    arrow(ax, (0.5050, branch_y), (0.5050, 0.255))
    arrow(ax, (0.8225, branch_y), (0.8225, 0.255))

    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_PNG, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
