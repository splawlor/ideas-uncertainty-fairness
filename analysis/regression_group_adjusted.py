import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

data_path, result_path, nuisance_path, model_path, summary_path = sys.argv[1:]

targets = [
    "Brain-Stem",
    "L-Cerebral-WM", "R-Cerebral-WM",
    "L-Thalamus", "R-Thalamus",
    "L-Caudate", "R-Caudate",
    "L-Putamen", "R-Putamen",
    "L-Pallidum", "R-Pallidum",
    "L-Hippocampus", "R-Hippocampus",
    "L-Amygdala", "R-Amygdala",
]

entropy_columns = [f"{r}_entropy_mean" for r in targets]
size_columns = [f"{r}_n_voxels" for r in targets]

required = [
    "subject_id", "group", "Sex", "Binned_Age_at_Scan",
    "Op_Side", "Pathology", "Op_Type_collapsed", "Number_ASMs",
    *entropy_columns, *size_columns,
]

d = pd.read_csv(data_path)
missing = [c for c in required if c not in d.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")
if d["subject_id"].duplicated().any():
    raise RuntimeError("Duplicate subject IDs found")

def clean(series):
    return series.astype("string").str.strip().str.upper()

def zscore(series, name):
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if values.isna().any():
        raise RuntimeError(f"Missing/non-numeric values in {name}")
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f"Cannot standardise {name}: SD={sd}")
    return (values - values.mean()) / sd

def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise RuntimeError("Invalid p-values supplied to BH correction")
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q = np.empty_like(q_ranked)
    q[order] = np.clip(q_ranked, 0, 1)
    return q

pathology_order = ["HS", "DNT", "FCD", "CAV", "DUAL", "OTHER"]
d["Pathology_grp"] = clean(d["Pathology"]).replace({
    "GL": "OTHER", "TREBLE": "OTHER", "TBC": "OTHER",
})
unexpected = sorted(set(d["Pathology_grp"].dropna()) - set(pathology_order))
if unexpected:
    raise RuntimeError(f"Unexpected pathology categories: {unexpected}")

d["Male"] = clean(d["Sex"]).map({
    "F": 0.0, "FEMALE": 0.0, "M": 1.0, "MALE": 1.0,
})
d["Right_side"] = clean(d["Op_Side"]).map({
    "L": 0.0, "LEFT": 0.0, "R": 1.0, "RIGHT": 1.0,
})
d["Extra_temporal"] = clean(d["Op_Type_collapsed"]).map({
    "TEMPORAL": 0.0,
    "EXTRA-TEMPORAL": 1.0,
    "EXTRA TEMPORAL": 1.0,
    "EXTRATEMPORAL": 1.0,
})
d["ASMs_numeric"] = pd.to_numeric(d["Number_ASMs"], errors="coerce")

age_text = d["Binned_Age_at_Scan"].astype("string").str.strip()
d["Age_lower"] = pd.to_numeric(
    age_text.str.extract(r"(-?\d+(?:\.\d+)?)")[0],
    errors="coerce",
)
bad_age = d["Binned_Age_at_Scan"].notna() & d["Age_lower"].isna()
if bad_age.any():
    values = sorted(d.loc[bad_age, "Binned_Age_at_Scan"].astype(str).unique())
    raise RuntimeError(f"Could not parse age bins: {values}")

group_numeric = pd.to_numeric(d["group"], errors="coerce")
if (group_numeric.dropna() % 1 != 0).any():
    raise RuntimeError("Non-integer disjoint-group values found")
d["Group"] = group_numeric.astype("Int64").astype("string")

clinical_columns = [
    "Pathology_grp", "Male", "Right_side", "Extra_temporal",
    "ASMs_numeric", "Age_lower", "Group",
]
analysis = d.loc[d[clinical_columns].notna().all(axis=1)].copy()

if len(analysis) != 432 or analysis["subject_id"].nunique() != 432:
    raise RuntimeError(
        "Expected 432 complete, unique epilepsy-patient rows; "
        f"found rows={len(analysis)}, "
        f"subjects={analysis['subject_id'].nunique()}"
    )

if set(analysis["Pathology_grp"]) != set(pathology_order):
    raise RuntimeError(
        "Unexpected pathology levels in clinical subset: "
        f"{sorted(analysis['Pathology_grp'].unique())}"
    )

expected_groups = ["0", "1", "2", "3", "4"]
if sorted(analysis["Group"].unique()) != expected_groups:
    raise RuntimeError(
        f"Expected groups {expected_groups}; "
        f"found {sorted(analysis['Group'].unique())}"
    )

analysis["Age_rank"] = analysis["Age_lower"].rank(method="dense")

entropy = analysis[entropy_columns].apply(pd.to_numeric, errors="coerce")
sizes = analysis[size_columns].apply(pd.to_numeric, errors="coerce")
if entropy.isna().any().any():
    raise RuntimeError("Missing/non-numeric target-region entropy values")
if sizes.isna().any().any() or (sizes <= 0).any().any():
    raise RuntimeError("Missing, non-numeric, or non-positive reference-mask sizes")

pathology_dummies = pd.get_dummies(
    pd.Categorical(
        analysis["Pathology_grp"],
        categories=pathology_order,
        ordered=True,
    ),
    prefix="Pathology",
    drop_first=True,
    dtype=float,
)
pathology_dummies.index = analysis.index

group_dummies = pd.get_dummies(
    pd.Categorical(
        analysis["Group"],
        categories=expected_groups,
        ordered=True,
    ),
    prefix="Group",
    drop_first=True,
    dtype=float,
)
group_dummies.index = analysis.index

primary_base = pd.concat([
    pathology_dummies,
    pd.DataFrame({
        "Age_rank_z": zscore(analysis["Age_rank"], "age-band rank"),
        "Male": analysis["Male"].astype(float),
        "Right_side": analysis["Right_side"].astype(float),
        "Extra_temporal": analysis["Extra_temporal"].astype(float),
        "ASMs_z": zscore(analysis["ASMs_numeric"], "number of ASMs"),
    }, index=analysis.index),
], axis=1)

base_terms = [
    "Pathology_DNT", "Pathology_FCD", "Pathology_CAV",
    "Pathology_DUAL", "Pathology_OTHER", "Age_rank_z",
    "Male", "Right_side", "Extra_temporal", "ASMs_z",
]
group_terms = ["Group_1", "Group_2", "Group_3", "Group_4"]

if primary_base.columns.tolist() != base_terms:
    raise RuntimeError(f"Unexpected primary terms: {primary_base.columns.tolist()}")
if group_dummies.columns.tolist() != group_terms:
    raise RuntimeError(f"Unexpected nuisance terms: {group_dummies.columns.tolist()}")

metadata = {
    "Pathology_DNT": ("Pathology", "DNT vs HS"),
    "Pathology_FCD": ("Pathology", "FCD vs HS"),
    "Pathology_CAV": ("Pathology", "CAV vs HS"),
    "Pathology_DUAL": ("Pathology", "Dual vs HS"),
    "Pathology_OTHER": ("Pathology", "Other vs HS"),
    "Age_rank_z": ("Age", "Age-band rank, per 1 SD"),
    "Male": ("Sex", "Male vs female"),
    "Right_side": ("Operation side", "Right vs left"),
    "Extra_temporal": ("Operation type", "Extra-temporal vs temporal"),
    "ASMs_z": ("Number of ASMs", "Number of ASMs, per 1 SD"),
    "Region_size_z": ("Region size", "Reference-mask voxel count, per 1 SD"),
}

coefficient_rows = []
nuisance_rows = []
model_rows = []

for region in targets:
    primary = primary_base.copy()
    primary["Region_size_z"] = zscore(
        analysis[f"{region}_n_voxels"],
        f"{region} reference-mask size",
    )
    primary_terms = [*base_terms, "Region_size_z"]

    design = sm.add_constant(
        pd.concat([primary, group_dummies], axis=1),
        has_constant="add",
    ).astype(float)
    outcome = analysis[f"{region}_entropy_mean"].astype(float)

    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise RuntimeError(f"Rank-deficient design matrix for {region}")

    conventional = sm.OLS(outcome, design, missing="raise").fit()
    hc3 = sm.OLS(outcome, design, missing="raise").fit(cov_type="HC3")
    conventional_ci = pd.DataFrame(
        conventional.conf_int(alpha=0.05),
        index=design.columns,
        columns=[0, 1],
    )
    hc3_ci = pd.DataFrame(
        hc3.conf_int(alpha=0.05),
        index=design.columns,
        columns=[0, 1],
    )

    for term in primary_terms:
        family, contrast = metadata[term]
        coefficient_rows.append({
            "region": region,
            "n": int(conventional.nobs),
            "predictor_family": family,
            "term": term,
            "contrast": contrast,
            "beta": conventional.params[term],
            "se_conventional": conventional.bse[term],
            "ci95_low_conventional": conventional_ci.loc[term, 0],
            "ci95_high_conventional": conventional_ci.loc[term, 1],
            "p_conventional": conventional.pvalues[term],
            "se_hc3": hc3.bse[term],
            "ci95_low_hc3": hc3_ci.loc[term, 0],
            "ci95_high_hc3": hc3_ci.loc[term, 1],
            "p_hc3": hc3.pvalues[term],
        })

    for term in group_terms:
        nuisance_rows.append({
            "region": region,
            "n": int(conventional.nobs),
            "term": term,
            "contrast": f"{term.replace('_', ' ')} vs Group 0",
            "beta": conventional.params[term],
            "p_conventional": conventional.pvalues[term],
            "p_hc3": hc3.pvalues[term],
            "included_in_165_fdr_family": False,
        })

    model_rows.append({
        "region": region,
        "n": int(conventional.nobs),
        "r_squared": conventional.rsquared,
        "adjusted_r_squared": conventional.rsquared_adj,
        "model_f_pvalue": conventional.f_pvalue,
    })

results = pd.DataFrame(coefficient_rows)
nuisance = pd.DataFrame(nuisance_rows)
models = pd.DataFrame(model_rows)

if len(results) != 165 or not (results.groupby("region").size() == 11).all():
    raise RuntimeError(f"Expected 165 primary coefficients; found {len(results)}")
if len(nuisance) != 60:
    raise RuntimeError(f"Expected 60 nuisance coefficients; found {len(nuisance)}")

results["p_fdr_conventional_165"] = bh_fdr(results["p_conventional"])
results["p_fdr_hc3_165"] = bh_fdr(results["p_hc3"])
results = results.sort_values(["p_conventional", "region", "term"])
nuisance = nuisance.sort_values(["region", "term"])
models = models.sort_values("region")

results.to_csv(result_path, index=False)
nuisance.to_csv(nuisance_path, index=False)
models.to_csv(model_path, index=False)

columns = [
    "region", "predictor_family", "contrast", "beta",
    "p_conventional", "p_fdr_conventional_165",
    "p_hc3", "p_fdr_hc3_165",
]

age_order = (
    analysis[["Binned_Age_at_Scan", "Age_lower", "Age_rank"]]
    .drop_duplicates()
    .sort_values("Age_rank")
)

summary_lines = [
    "RQ3 GROUP-ADJUSTED MULTIVARIABLE REGRESSION",
    "",
    "INPUT",
    f"Input rows: {len(d)}",
    f"Analysed epilepsy patients: {len(analysis)}",
    f"Regional models: {len(models)}",
    f"Primary coefficients: {len(results)} (15 x 11)",
    f"Group nuisance coefficients: {len(nuisance)} (15 x 4; excluded from FDR)",
    "Outcome: reference-masked total predictive entropy",
    "Reference groups: HS, female, left, temporal, disjoint Group 0",
    "Standardised with population SD: age-band rank, ASM count, regional reference-mask size",
    "ILAE outcome excluded from RQ3",
    "",
    "AGE-BAND ORDER",
    age_order.to_string(index=False),
    "",
    "INFERENCE SUMMARY",
    f"Conventional raw p < 0.05: {int((results.p_conventional < 0.05).sum())}/165",
    f"Conventional FDR q < 0.05: {int((results.p_fdr_conventional_165 < 0.05).sum())}/165",
    f"Minimum conventional q: {results.p_fdr_conventional_165.min():.8g}",
    f"HC3 raw p < 0.05: {int((results.p_hc3 < 0.05).sum())}/165",
    f"HC3 FDR q < 0.05: {int((results.p_fdr_hc3_165 < 0.05).sum())}/165",
    f"Minimum HC3 q: {results.p_fdr_hc3_165.min():.8g}",
    "",
    "MODEL FIT",
    f"Mean R-squared: {models.r_squared.mean():.8g}",
    f"Mean adjusted R-squared: {models.adjusted_r_squared.mean():.8g}",
    f"Median adjusted R-squared: {models.adjusted_r_squared.median():.8g}",
    "",
    "15 SMALLEST CONVENTIONAL P-VALUES",
    results.nsmallest(15, "p_conventional")[columns].to_string(index=False),
    "",
    "15 SMALLEST HC3 P-VALUES",
    results.nsmallest(15, "p_hc3")[columns].to_string(index=False),
    "",
    "FDR-SIGNIFICANT CONVENTIONAL RESULTS",
    (
        results.loc[results.p_fdr_conventional_165 < 0.05, columns].to_string(index=False)
        if (results.p_fdr_conventional_165 < 0.05).any()
        else "None"
    ),
    "",
    "FDR-SIGNIFICANT HC3 RESULTS",
    (
        results.loc[results.p_fdr_hc3_165 < 0.05, columns].to_string(index=False)
        if (results.p_fdr_hc3_165 < 0.05).any()
        else "None"
    ),
    "",
    f"Saved detailed results: {result_path}",
    f"Saved nuisance coefficients: {nuisance_path}",
    f"Saved model summary: {model_path}",
]

summary_text = "\n".join(summary_lines)
with open(summary_path, "w", encoding="utf-8") as file:
    file.write(summary_text + "\n")

print(summary_text)
print("\nSaved summary:", summary_path)
