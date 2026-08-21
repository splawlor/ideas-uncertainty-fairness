"""Generate the risk--coverage figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "supporting_tables"
    / "final_disjoint"
    / "risk_coverage_curve.csv"
)
OUTPUT_PDF = ROOT / "img" / "risk_coverage_curve.pdf"
OUTPUT_PNG = ROOT / "img" / "risk_coverage_curve_preview.png"


def main():
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    curve = pd.read_csv(INPUT)
    required = {
        "score",
        "target_coverage",
        "actual_coverage",
        "selective_risk_one_minus_dice",
        "selective_risk_ci_low",
        "selective_risk_ci_high",
    }
    if not required.issubset(curve.columns):
        raise ValueError(f"Risk-coverage source is missing: {required - set(curve.columns)}")

    styles = {
        "Predictive entropy": {"color": "#3E78A3", "linestyle": "-", "marker": "o"},
        "Maximum-softmax uncertainty": {
            "color": "#C87533",
            "linestyle": "--",
            "marker": "s",
        },
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
    for score_name, style in styles.items():
        rows = curve[curve["score"] == score_name].sort_values("actual_coverage")
        x = rows["actual_coverage"].to_numpy(float) * 100.0
        risk = rows["selective_risk_one_minus_dice"].to_numpy(float)
        low = rows["selective_risk_ci_low"].to_numpy(float)
        high = rows["selective_risk_ci_high"].to_numpy(float)
        axis.plot(x, risk, linewidth=2.1, markersize=4.5, label=score_name, **style)
        axis.fill_between(x, low, high, color=style["color"], alpha=0.14, linewidth=0)

    full_risk = float(
        curve.loc[
            np.isclose(curve["target_coverage"], 1.0),
            "selective_risk_one_minus_dice",
        ].iloc[0]
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
        "Risk-coverage for label-free subject triage",
        x=0.105,
        y=0.985,
        ha="left",
        va="top",
        fontsize=12,
    )
    figure.text(
        0.105,
        0.935,
        "Disjoint four-model ensemble; lower risk is better; shaded bands are 95% subject-bootstrap CIs",
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
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
