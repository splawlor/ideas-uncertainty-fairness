#!/usr/bin/env python3
"""Calculate the minimum detectable effects for selected RQ2 comparisons."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from scipy.optimize import brentq
from scipy.stats import nct, norm, t


COMPARISONS = [
    ("Sex: female vs male", 233, 199, False),
    ("Operation side: left vs right", 228, 204, False),
    ("ILAE year 1: seizure-free vs not", 242, 175, False),
    ("Operation type: temporal vs extra-temporal", 346, 86, False),
    ("Pathology: HS vs Other", 208, 82, True),
    ("Pathology: HS vs DNT", 208, 51, True),
    ("Pathology: HS vs FCD", 208, 39, True),
    ("Pathology: HS vs cavernoma", 208, 32, True),
    ("Pathology: HS vs dual pathology", 208, 20, True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--family-size", type=int, default=105)
    return parser.parse_args()


def exact_two_sample_power(effect: float, n1: int, n2: int, alpha: float) -> float:
    df = n1 + n2 - 2
    critical = t.ppf(1.0 - alpha / 2.0, df)
    noncentrality = effect * math.sqrt(n1 * n2 / (n1 + n2))
    value = nct.cdf(-critical, df, noncentrality) + nct.sf(critical, df, noncentrality)
    return float(value)


def solve_exact_mde(n1: int, n2: int, alpha: float, target_power: float) -> float:
    low = 0.0
    for step in range(1, 2001):
        high = step / 1000.0
        power = exact_two_sample_power(high, n1, n2, alpha)
        if math.isfinite(power) and power >= target_power:
            return float(
                brentq(
                    lambda effect: exact_two_sample_power(effect, n1, n2, alpha) - target_power,
                    low,
                    high,
                    xtol=1e-13,
                    rtol=1e-13,
                )
            )
        if math.isfinite(power):
            low = high
    raise RuntimeError(f"Could not bracket the MDE for n1={n1}, n2={n2}, alpha={alpha}")


def normal_approximation(n1: int, n2: int, alpha: float, target_power: float) -> float:
    return float(
        (norm.ppf(1.0 - alpha / 2.0) + norm.ppf(target_power))
        * math.sqrt(1.0 / n1 + 1.0 / n2)
    )


def latex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.power < 1.0:
        raise ValueError("Power must be between zero and one")
    if args.family_size < 1:
        raise ValueError("Family size must be positive")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    alpha_uncorrected = 0.05
    alpha_bonferroni = alpha_uncorrected / args.family_size
    rows = []
    for comparison, n1, n2, pairwise_approximation in COMPARISONS:
        exact_uncorrected = solve_exact_mde(n1, n2, alpha_uncorrected, args.power)
        exact_bonferroni = solve_exact_mde(n1, n2, alpha_bonferroni, args.power)
        normal_uncorrected = normal_approximation(n1, n2, alpha_uncorrected, args.power)
        normal_bonferroni = normal_approximation(n1, n2, alpha_bonferroni, args.power)
        rows.append(
            {
                "comparison": comparison,
                "n1": n1,
                "n2": n2,
                "target_power": args.power,
                "alpha_uncorrected": alpha_uncorrected,
                "alpha_bonferroni": alpha_bonferroni,
                "mde_d_exact_t_uncorrected": exact_uncorrected,
                "mde_d_exact_t_bonferroni": exact_bonferroni,
                "mde_d_normal_approx_uncorrected": normal_uncorrected,
                "mde_d_normal_approx_bonferroni": normal_bonferroni,
                "pathology_pairwise_approximation": pairwise_approximation,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out / "mde_sensitivity.csv", index=False)

    checks = result.set_index("comparison")
    expected_ranges = {
        "Sex: female vs male": (0.421, 0.422),
        "Operation side: left vs right": (0.420, 0.422),
        "Pathology: HS vs dual pathology": (1.028, 1.030),
    }
    for comparison, (lower, upper) in expected_ranges.items():
        value = checks.loc[comparison, "mde_d_exact_t_bonferroni"]
        if not lower <= value <= upper:
            raise AssertionError(f"Unexpected exact MDE for {comparison}: {value}")

    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{Conservative minimum detectable standardized differences for selected two-group RQ2 comparisons at {100*args.power:.0f}\% power. Values use a two-sided equal-variance noncentral-$t$ calculation. The Bonferroni threshold is $0.05/{args.family_size}=4.76\times10^{{-4}}$. Pathology rows are pairwise approximations to the prespecified six-level omnibus test and do not constitute separate confirmatory tests.}}",
        r"\label{tab:mde_sensitivity}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        rf"Comparison & $n_1$ & $n_2$ & MDE at $\alpha=0.05$ & MDE at $\alpha=0.05/{args.family_size}$ \\",
        r"\hline",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.comparison)} & {row.n1} & {row.n2} & "
            f"{row.mde_d_exact_t_uncorrected:.2f} & {row.mde_d_exact_t_bonferroni:.2f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    (out / "table_mde_sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    method = f"""# MDE sensitivity method

- Target power: {args.power:.2f}
- Two-sided uncorrected alpha: {alpha_uncorrected}
- Conservative Bonferroni alpha: 0.05/{args.family_size} = {alpha_bonferroni:.12g}
- Primary calculation: exact noncentral-t power for an equal-variance independent-samples t test.
- Effect scale: Cohen's standardized mean difference d.
- The CSV includes the normal approximation for comparison; the LaTeX table uses the exact noncentral-t calculation.
- Scope: approximate two-group sensitivity for selected binary and pathology contrasts in the 105-test RQ2 screening family. It is not a power analysis for the age/ASM Spearman tests, the six-level pathology omnibus test, or the separate 165-coefficient RQ3 family.
"""
    (out / "MDE_METHOD.md").write_text(method, encoding="utf-8")
    print(f"Wrote exact MDE table to {out}")
    print(result[["comparison", "mde_d_exact_t_uncorrected", "mde_d_exact_t_bonferroni"]].to_string(index=False))


if __name__ == "__main__":
    main()
