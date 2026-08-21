#!/usr/bin/env python3
"""Run the post hoc RQ1 sensitivity analyses."""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

PUBLISHED = {
    "L-Hippocampus":  (0.587,  0.522,  0.493),
    "R-Hippocampus":  (0.600,  0.495,  0.534),
    "L-Amygdala":     (0.587,  0.466,  0.109),
    "R-Amygdala":     (0.604,  0.451,  0.162),
    "L-Thalamus":     (0.690,  0.393,  0.493),
    "R-Thalamus":     (0.687,  0.395,  0.487),
    "L-Caudate":      (0.439,  0.543, -0.019),
    "R-Caudate":      (0.445,  0.551,  0.047),
    "L-Putamen":      (0.506,  0.559,  0.433),
    "R-Putamen":      (0.508,  0.553,  0.451),
    "L-Pallidum":     (0.505,  0.515,  0.239),
    "R-Pallidum":     (0.524,  0.490,  0.259),
    "L-Cerebral-WM":  (0.637,  0.627,  0.668),
    "R-Cerebral-WM":  (0.644,  0.615,  0.676),
    "Brain-Stem":     (0.774,  0.293,  0.570),
}
REGIONS = list(PUBLISHED)

PUBLISHED_MEAN_DICE = 0.582461
PUBLISHED_MEAN_ENTROPY = 0.498
PUBLISHED_MEAN_RHO = 0.373
PUBLISHED_N_POSITIVE = 14
PUBLISHED_N_SIG = 13

N_SUBJECTS, N_PATIENTS, N_CONTROLS, N_REGIONS = 532, 432, 100, 15
CONTROL_ID_LO, CONTROL_ID_HI = 4001, 4100

TOL_RHO = 0.002
TOL_MEAN = 0.002

SUFFIX_ENTROPY = "_entropy_mean"
SUFFIX_DICE = "_dice"
SUFFIX_VOXELS = "_n_voxels"
FORBIDDEN = "_pred_"

SUBJECT_CANDIDATES = ["subject", "subject_id", "subjectid", "case", "case_id",
                      "id", "patient", "patient_id", "ideas_id"]


def die(msg: str) -> None:
    print("\n*** ABORT ***\n" + msg + "\n", file=sys.stderr)
    sys.exit(1)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_subject_column(df: pd.DataFrame) -> str:
    norm = {_norm(c): c for c in df.columns}
    for cand in SUBJECT_CANDIDATES:
        if _norm(cand) in norm:
            return norm[_norm(cand)]
    for c in df.columns:
        vals = df[c].astype(str).head(20)
        if vals.str.contains(r"(?i)ideas", regex=True).all():
            return c
    die("Could not identify the subject column.\n"
        f"First 30 columns: {list(df.columns)[:30]}\n"
        f"Add its name to SUBJECT_CANDIDATES.")


def reshape_wide(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Select the 45 intended columns and melt to long form."""
    print("\n=== COLUMN SELECTION (wide format) ===")
    sub_col = find_subject_column(df)
    print(f"  subject column                 : '{sub_col}'")

    n_pred = sum(1 for c in df.columns if FORBIDDEN in str(c))
    print(f"  columns containing '{FORBIDDEN}'    : {n_pred}  (rejected outright)")

    selected, missing = {}, []
    for region in REGIONS:
        trio = {
            "_ent": f"{region}{SUFFIX_ENTROPY}",
            "_dice": f"{region}{SUFFIX_DICE}",
            "_vol": f"{region}{SUFFIX_VOXELS}",
        }
        for key, col in trio.items():
            if col not in df.columns:
                missing.append(col)
        selected[region] = trio

    if missing:
        stem = missing[0].split("_")[0]
        near = [c for c in df.columns if str(c).startswith(stem)][:20]
        die(f"{len(missing)} expected columns are absent, e.g. {missing[:6]}\n"
            f"Columns beginning '{stem}': {near}\n"
            f"Adjust SUFFIX_ENTROPY / SUFFIX_DICE / SUFFIX_VOXELS to match.")

    flat = [c for trio in selected.values() for c in trio.values()]
    bad = [c for c in flat if FORBIDDEN in c]
    if bad:
        die(f"Selected columns contain '{FORBIDDEN}': {bad}")
    print(f"  selected columns               : {len(flat)}  "
          f"(15 regions x 3)  OK")
    print(f"  example                        : "
          f"'{selected[REGIONS[0]]['_ent']}', "
          f"'{selected[REGIONS[0]]['_dice']}', "
          f"'{selected[REGIONS[0]]['_vol']}'")

    frames = []
    for region, trio in selected.items():
        part = df[[sub_col, trio["_ent"], trio["_dice"], trio["_vol"]]].copy()
        part.columns = ["_subject", "_ent", "_dice", "_vol"]
        part["_canon"] = region
        frames.append(part)
    long = pd.concat(frames, ignore_index=True)

    provenance = {
        "format": "wide",
        "subject_column": sub_col,
        "n_pred_columns_rejected": n_pred,
        "selected_columns": flat,
    }
    return long, provenance


def validate(long: pd.DataFrame) -> pd.DataFrame:
    print("\n=== STRUCTURAL VALIDATION ===")

    def to_int(v):
        m = re.search(r"(\d+)", str(v))
        return int(m.group(1)) if m else None

    long = long.copy()
    long["_sid"] = long["_subject"].map(to_int)
    if long["_sid"].isna().any():
        die("Could not parse a numeric subject ID from the subject column.")
    long["_sid"] = long["_sid"].astype(int)

    n_sub = long["_sid"].nunique()
    print(f"  unique subjects                : {n_sub}", end="")
    if n_sub != N_SUBJECTS:
        die(f"\nExpected {N_SUBJECTS}, found {n_sub}.")
    print("  OK")

    long["_is_control"] = long["_sid"].between(CONTROL_ID_LO, CONTROL_ID_HI)
    n_ctrl = long.loc[long["_is_control"], "_sid"].nunique()
    n_pat = long.loc[~long["_is_control"], "_sid"].nunique()
    print(f"  controls (IDEAS_4001-4100)     : {n_ctrl}", end="")
    if n_ctrl != N_CONTROLS:
        die(f"\nExpected {N_CONTROLS}, found {n_ctrl}.")
    print("  OK")
    print(f"  patients                       : {n_pat}", end="")
    if n_pat != N_PATIENTS:
        die(f"\nExpected {N_PATIENTS}, found {n_pat}.")
    print("  OK")

    print(f"  regions                        : {long['_canon'].nunique()}", end="")
    if long["_canon"].nunique() != N_REGIONS:
        die(f"\nExpected {N_REGIONS} regions.")
    print("  OK")

    pairs = len(long)
    print(f"  subject-region pairs           : {pairs}", end="")
    if pairs != N_SUBJECTS * N_REGIONS:
        die(f"\nExpected {N_SUBJECTS * N_REGIONS}, found {pairs}.")
    print("  OK")

    if long.duplicated(["_sid", "_canon"]).any():
        die("Duplicate (subject, region) rows.")

    for col, name in [("_ent", "entropy"), ("_dice", "dice"), ("_vol", "volume")]:
        long[col] = pd.to_numeric(long[col], errors="coerce").astype(float)
        finite = np.isfinite(long[col].values)
        n_bad = int((~finite).sum())
        print(f"  {name + ' finite':<30} : {len(long) - n_bad}/{len(long)}", end="")
        if n_bad:
            bad_rows = long.loc[~finite, ["_subject", "_canon", col]].head(10)
            die(f"\n{n_bad} non-finite {name} values (NaN or inf). Examples:\n"
                f"{bad_rows.to_string(index=False)}")
        print("  OK")

    long["_err"] = 1.0 - long["_dice"]

    lo, hi = float(long["_dice"].min()), float(long["_dice"].max())
    print(f"  dice range                     : [{lo:.3f}, {hi:.3f}]", end="")
    if lo < -1e-9 or hi > 1 + 1e-9:
        die("\nDice outside [0, 1]. Wrong column.")
    print("  OK")

    elo, ehi = float(long["_ent"].min()), float(long["_ent"].max())
    print(f"  entropy range                  : [{elo:.3f}, {ehi:.3f}]", end="")
    if elo < 0:
        die("\nNegative entropy. Wrong column.")
    if ehi > np.log(110):
        die(f"\nEntropy exceeds ln(110) = {np.log(110):.3f}. Wrong column.")
    print("  OK")

    if (long["_vol"] <= 0).any():
        die(f"{int((long['_vol'] <= 0).sum())} non-positive region sizes.")
    frac_int = float(np.mean(np.isclose(long["_vol"], np.round(long["_vol"]))))
    print(f"  volume positive                : yes  OK")
    print(f"  volume integer-valued          : {frac_int:.1%}", end="")
    if frac_int < 0.99:
        print("  <-- WARNING: looks like mm^3, not a voxel count.")
    else:
        print("  OK")

    return long


def reproduce_table31(long: pd.DataFrame) -> None:
    print("\n=== REPRODUCTION GATE (Table 3.1, all 532) ===")
    print(f"  {'Region':<16} {'dice':>16} {'entropy':>16} {'rho':>18}")
    failures, rhos, dices, ents = [], [], [], []
    for region in REGIONS:
        d = long[long["_canon"] == region]
        pd_, pe_, pr_ = PUBLISHED[region]
        gd = float(d["_dice"].mean())
        ge = float(d["_ent"].mean())
        gr = float(stats.spearmanr(d["_ent"].values, d["_err"].values).statistic)
        dices.append(gd); ents.append(ge); rhos.append(gr)
        ok_d, ok_e, ok_r = (abs(gd - pd_) <= TOL_RHO,
                            abs(ge - pe_) <= TOL_RHO,
                            abs(gr - pr_) <= TOL_RHO)
        if not (ok_d and ok_e and ok_r):
            failures.append(region)
        m = lambda ok: "ok" if ok else "XX"
        print(f"  {region:<16} {pd_:.3f}->{gd:.3f} {m(ok_d)}  "
              f"{pe_:.3f}->{ge:.3f} {m(ok_e)}  "
              f"{pr_:+.3f}->{gr:+.3f} {m(ok_r)}")

    md, me, mr = float(np.mean(dices)), float(np.mean(ents)), float(np.mean(rhos))
    print(f"\n  mean Dice     {PUBLISHED_MEAN_DICE:.6f} -> {md:.6f}")
    print(f"  mean entropy  {PUBLISHED_MEAN_ENTROPY:.3f} -> {me:.3f}")
    print(f"  mean rho      {PUBLISHED_MEAN_RHO:.3f} -> {mr:.3f}")
    if abs(md - PUBLISHED_MEAN_DICE) > TOL_MEAN:
        failures.append("MEAN DICE")
    if abs(mr - PUBLISHED_MEAN_RHO) > TOL_MEAN:
        failures.append("MEAN RHO")

    if failures:
        die("Reproduction FAILED for: " + ", ".join(failures) +
            "\n\nThe script is not reading the columns used for Table 3.1. Check:\n"
            "  - is this the reference-masked table (no '_pred_')?\n"
            "  - is '_entropy_mean' TOTAL predictive entropy, not\n"
            "    mean-member entropy or mutual information?\n"
            "  - are all 532 subjects present, controls included?")
    print("\n  REPRODUCTION PASSED -- proceeding.")


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q


def spearman(x, y):
    r = stats.spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y controlling for z."""
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    n = rx.size
    Z = np.column_stack([np.ones(n), rz])
    coef, *_ = np.linalg.lstsq(Z, np.column_stack([rx, ry]), rcond=None)
    res = np.column_stack([rx, ry]) - Z @ coef
    ex, ey = res[:, 0], res[:, 1]
    denom = np.sqrt(np.sum(ex ** 2) * np.sum(ey ** 2))
    if denom == 0:
        return np.nan, np.nan
    r = float(np.clip(np.sum(ex * ey) / denom, -1.0, 1.0))
    df = n - 3
    if df <= 0 or abs(r) >= 1.0:
        return r, np.nan
    t = r * np.sqrt(df / (1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), df))


def bootstrap_ci(fn, arrays, n_boot, rng, alpha=0.05):
    n = arrays[0].size
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        out[b] = fn(*[a[idx] for a in arrays])[0]
    out = out[np.isfinite(out)]
    if out.size == 0:
        return np.nan, np.nan
    return (float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))))


def run_analysis(long, label, adjust_volume, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for region in REGIONS:
        d = long[long["_canon"] == region]
        ent, err, vol = d["_ent"].values, d["_err"].values, d["_vol"].values
        if adjust_volume:
            r, p = partial_spearman(ent, err, vol)
            lo, hi = bootstrap_ci(partial_spearman, (ent, err, vol), n_boot, rng)
        else:
            r, p = spearman(ent, err)
            lo, hi = bootstrap_ci(spearman, (ent, err), n_boot, rng)
        rows.append({"analysis": label, "region": region, "n": len(d),
                     "rho": r, "ci_lo": lo, "ci_hi": hi, "p_raw": p,
                     "published_rho": PUBLISHED[region][2],
                     "seed": seed, "n_boot": n_boot})
    out = pd.DataFrame(rows)
    out["q_bh"] = bh_qvalues(out["p_raw"].values)
    out["delta_vs_published"] = out["rho"] - out["published_rho"]
    return out


def report(out: pd.DataFrame) -> dict:
    label = out["analysis"].iloc[0]
    print(f"\n=== {label} ===")
    print(f"  {'Region':<16} {'n':>4} {'rho':>7} {'95% CI':>18} "
          f"{'p':>10} {'q(BH)':>10} {'pub':>7} {'delta':>7}")
    for _, r in out.iterrows():
        sig = "*" if r["q_bh"] < 0.05 else " "
        print(f"  {r['region']:<16} {r['n']:>4} {r['rho']:>+7.3f} "
              f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] {r['p_raw']:>10.2e} "
              f"{r['q_bh']:>9.4f}{sig} {r['published_rho']:>+7.3f} "
              f"{r['delta_vs_published']:>+7.3f}")
    summary = {
        "analysis": label,
        "n_subjects": int(out["n"].iloc[0]),
        "mean_rho": float(out["rho"].mean()),
        "n_positive": int((out["rho"] > 0).sum()),
        "n_fdr_significant": int((out["q_bh"] < 0.05).sum()),
        "seed": int(out["seed"].iloc[0]),
        "n_boot": int(out["n_boot"].iloc[0]),
    }
    print(f"  mean regional rho              : {summary['mean_rho']:.3f} "
          f"(published: {PUBLISHED_MEAN_RHO:.3f})")
    print(f"  positive                       : {summary['n_positive']}/15 "
          f"(published: {PUBLISHED_N_POSITIVE})")
    print(f"  FDR-significant (q<0.05)       : "
          f"{summary['n_fdr_significant']}/15 (published: {PUBLISHED_N_SIG})")
    return summary


def build_synthetic_wide() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    sids = [f"IDEAS_{i}" for i in list(range(1, 433)) + list(range(4001, 4101))]
    data = {"subject_id": sids}
    for region in REGIONS:
        bd, be, _ = PUBLISHED[region]
        vol = rng.gamma(20, 100, len(sids))
        ent = np.maximum(be + rng.normal(0, 0.1, len(sids))
                         - 0.0002 * (vol - 2000), 0.0)
        dice = np.clip(bd + rng.normal(0, 0.1, len(sids)) - 0.6 * (ent - be), 0, 1)
        data[f"{region}{SUFFIX_ENTROPY}"] = ent
        data[f"{region}{SUFFIX_DICE}"] = dice
        data[f"{region}{SUFFIX_VOXELS}"] = vol.round()
        data[f"{region}_pred_entropy_mean"] = ent + 0.3
        data[f"{region}_mi_mean"] = ent * 0.6
    return pd.DataFrame(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--out-prefix", default="rq1_sensitivity")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    prov = {"seed": args.seed, "n_boot": args.boot}

    if args.self_test:
        raw = build_synthetic_wide()
        prov["input"] = "SYNTHETIC"
        print("SELF-TEST: synthetic wide table; reproduction gate skipped.")
    elif args.csv:
        raw = pd.read_csv(args.csv)
        prov["input"] = args.csv
        print(f"\nInput   : {args.csv}")
    else:
        die("Provide --csv <file> or --self-test.")

    print(f"Shape   : {raw.shape[0]} rows x {raw.shape[1]} columns")
    print(f"Seed    : {args.seed}   Bootstraps: {args.boot}")

    long, sel_prov = reshape_wide(raw)
    prov.update(sel_prov)

    long = validate(long)
    if not args.self_test:
        reproduce_table31(long)

    pat = long[~long["_is_control"]].copy()

    results = [
        run_analysis(pat, "A. Patients only (n=432), unadjusted",
                     False, args.boot, args.seed),
        run_analysis(long, "B1. Volume-adjusted, full cohort (n=532)",
                     True, args.boot, args.seed + 1),
        run_analysis(pat, "B2. Volume-adjusted, patients only (n=432)",
                     True, args.boot, args.seed + 2),
    ]
    summaries = [report(r) for r in results]

    summaries.insert(0, {
        "analysis": "Published Table 3.1 (n=532, unadjusted)",
        "n_subjects": N_SUBJECTS, "mean_rho": PUBLISHED_MEAN_RHO,
        "n_positive": PUBLISHED_N_POSITIVE,
        "n_fdr_significant": PUBLISHED_N_SIG,
        "seed": None, "n_boot": None,
    })

    reg_path = f"{args.out_prefix}_regional.csv"
    sum_path = f"{args.out_prefix}_summary.csv"
    prov_path = f"{args.out_prefix}_provenance.json"
    pd.concat(results, ignore_index=True).to_csv(reg_path, index=False)
    pd.DataFrame(summaries).to_csv(sum_path, index=False)
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)

    print(f"\nWrote {reg_path}   (45 regional rows)")
    print(f"Wrote {sum_path}    (summary incl. published reference)")
    print(f"Wrote {prov_path} (seed, bootstraps, selected columns)")
    print("\nBH applied separately within each analysis, across the 15 regions.")
    print("Point estimates and p/q are independent of --boot; only the CIs "
          "need the full 10,000.")


if __name__ == "__main__":
    main()
