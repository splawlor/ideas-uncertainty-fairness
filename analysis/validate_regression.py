import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera


data_path = sys.argv[1]
result_path = sys.argv[2]
nuisance_path = sys.argv[3]
model_path = sys.argv[4]
summary_path = sys.argv[5]

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

entropy_columns = [
    f"{region}_entropy_mean"
    for region in targets
]

size_columns = [
    f"{region}_n_voxels"
    for region in targets
]

required = [
    "subject_id",
    "group",
    "Sex",
    "Binned_Age_at_Scan",
    "Op_Side",
    "Pathology",
    "Op_Type_collapsed",
    "Number_ASMs",
    *entropy_columns,
    *size_columns,
]

d = pd.read_csv(data_path)

missing_columns = [
    column for column in required
    if column not in d.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing required columns: {missing_columns}"
    )

if d["subject_id"].duplicated().any():
    raise RuntimeError("Duplicate subject IDs found")


def clean_text(series):
    return (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )


def z_score_population(series, name):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)

    if values.isna().any():
        raise RuntimeError(
            f"Missing/non-numeric values in {name}"
        )

    standard_deviation = values.std(ddof=0)

    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise RuntimeError(
            f"Cannot standardise {name}: "
            f"SD={standard_deviation}"
        )

    return (values - values.mean()) / standard_deviation


def bh_fdr(p_values):
    p_values = np.asarray(p_values, dtype=float)

    if (
        np.isnan(p_values).any()
        or (p_values < 0).any()
        or (p_values > 1).any()
    ):
        raise RuntimeError("Invalid p-values supplied to BH correction")

    order = np.argsort(p_values)
    ranked = p_values[order]

    adjusted_ranked = (
        ranked
        * len(ranked)
        / np.arange(1, len(ranked) + 1)
    )

    adjusted_ranked = np.minimum.accumulate(
        adjusted_ranked[::-1]
    )[::-1]

    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(
        adjusted_ranked,
        0,
        1,
    )

    return adjusted


pathology = clean_text(d["Pathology"]).replace({
    "GL": "OTHER",
    "TREBLE": "OTHER",
    "TBC": "OTHER",
})

pathology_order = [
    "HS",
    "DNT",
    "FCD",
    "CAV",
    "DUAL",
    "OTHER",
]

unexpected_pathology = sorted(
    set(pathology.dropna()) - set(pathology_order)
)

if unexpected_pathology:
    raise RuntimeError(
        "Unexpected pathology categories: "
        f"{unexpected_pathology}"
    )

d["Pathology_grp"] = pathology

sex_map = {
    "F": 0.0,
    "FEMALE": 0.0,
    "M": 1.0,
    "MALE": 1.0,
}

side_map = {
    "L": 0.0,
    "LEFT": 0.0,
    "R": 1.0,
    "RIGHT": 1.0,
}

operation_type_map = {
    "TEMPORAL": 0.0,
    "EXTRA-TEMPORAL": 1.0,
    "EXTRA TEMPORAL": 1.0,
    "EXTRATEMPORAL": 1.0,
}

d["Male"] = clean_text(d["Sex"]).map(sex_map)
d["Right_side"] = clean_text(d["Op_Side"]).map(side_map)
d["Extra_temporal"] = clean_text(
    d["Op_Type_collapsed"]
).map(operation_type_map)

d["Number_ASMs_numeric"] = pd.to_numeric(
    d["Number_ASMs"],
    errors="coerce",
)

age_text = (
    d["Binned_Age_at_Scan"]
    .astype("string")
    .str.strip()
)

age_lower = pd.to_numeric(
    age_text.str.extract(r"(-?\d+(?:\.\d+)?)")[0],
    errors="coerce",
)

bad_age = (
    d["Binned_Age_at_Scan"].notna()
    & age_lower.isna()
)

if bad_age.any():
    raise RuntimeError(
        "Could not parse age bands: "
        + str(
            sorted(
                d.loc[
                    bad_age,
                    "Binned_Age_at_Scan",
                ].astype(str).unique()
            )
        )
    )

d["Age_band_text"] = age_text
d["Age_band_lower"] = age_lower

group_numeric = pd.to_numeric(
    d["group"],
    errors="coerce",
)

non_integer_group = (
    group_numeric.notna()
    & (group_numeric % 1 != 0)
)

if non_integer_group.any():
    raise RuntimeError("Non-integer disjoint-group values found")

d["group_clean"] = (
    group_numeric
    .astype("Int64")
    .astype("string")
)

expected_groups = ["0", "1", "2", "3", "4"]
observed_groups = sorted(d["group_clean"].dropna().unique())

if observed_groups != expected_groups:
    raise RuntimeError(
        f"Expected groups {expected_groups}; found {observed_groups}"
    )

clinical_columns = [
    "Pathology_grp",
    "Male",
    "Right_side",
    "Extra_temporal",
    "Number_ASMs_numeric",
    "Age_band_lower",
    "group_clean",
]

clinical_complete = d[clinical_columns].notna().all(axis=1)
analysis = d.loc[clinical_complete].copy()

if len(analysis) != 432:
    raise RuntimeError(
        "Expected 432 complete epilepsy-patient rows; "
        f"found {len(analysis)}"
    )

if analysis["subject_id"].nunique() != 432:
    raise RuntimeError(
        "Clinical analysis does not contain 432 unique subjects"
    )

analysis["Age_band_rank"] = analysis[
    "Age_band_lower"
].rank(method="dense")

age_level_table = (
    analysis[[
        "Age_band_text",
        "Age_band_lower",
        "Age_band_rank",
    ]]
    .drop_duplicates()
    .sort_values(["Age_band_rank", "Age_band_text"])
    .rename(columns={
        "Age_band_text": "age_text",
        "Age_band_lower": "age_lower",
        "Age_band_rank": "rank",
    })
)

if age_level_table["age_lower"].duplicated().any():
    raise RuntimeError(
        "Age bands do not have unique lower boundaries:\n"
        + age_level_table.to_string(index=False)
    )

if not set(analysis["Pathology_grp"]) == set(pathology_order):
    raise RuntimeError(
        "Clinical subset pathology levels do not match the "
        f"six prespecified levels: "
        f"{sorted(analysis['Pathology_grp'].unique())}"
    )

entropy_numeric = analysis[entropy_columns].apply(
    pd.to_numeric,
    errors="coerce",
)

size_numeric = analysis[size_columns].apply(
    pd.to_numeric,
    errors="coerce",
)

if entropy_numeric.isna().any().any():
    missing_entropy = entropy_numeric.isna().sum()
    raise RuntimeError(
        "Missing/non-numeric entropy values:\n"
        + missing_entropy[
            missing_entropy > 0
        ].to_string()
    )

if size_numeric.isna().any().any():
    missing_size = size_numeric.isna().sum()
    raise RuntimeError(
        "Missing/non-numeric reference-mask sizes:\n"
        + missing_size[
            missing_size > 0
        ].to_string()
    )

if (size_numeric <= 0).any().any():
    raise RuntimeError(
        "Non-positive reference-mask size found"
    )

pathology_categorical = pd.Categorical(
    analysis["Pathology_grp"],
    categories=pathology_order,
    ordered=True,
)

pathology_dummies = pd.get_dummies(
    pathology_categorical,
    prefix="Pathology",
    drop_first=True,
    dtype=float,
)
pathology_dummies.index = analysis.index

group_categorical = pd.Categorical(
    analysis["group_clean"],
    categories=expected_groups,
    ordered=True,
)

group_dummies = pd.get_dummies(
    group_categorical,
    prefix="Group",
    drop_first=True,
    dtype=float,
)
group_dummies.index = analysis.index

continuous_primary = pd.DataFrame(
    {
        "Age_band_rank_z": z_score_population(
            analysis["Age_band_rank"],
            "age-band rank",
        ),
        "Male": analysis["Male"].astype(float),
        "Right_side": analysis["Right_side"].astype(float),
        "Extra_temporal": analysis["Extra_temporal"].astype(float),
        "Number_ASMs_z": z_score_population(
            analysis["Number_ASMs_numeric"],
            "number of ASMs",
        ),
    },
    index=analysis.index,
)

primary_base = pd.concat(
    [
        pathology_dummies,
        continuous_primary,
    ],
    axis=1,
)

base_primary_names = [
    "Pathology_DNT",
    "Pathology_FCD",
    "Pathology_CAV",
    "Pathology_DUAL",
    "Pathology_OTHER",
    "Age_band_rank_z",
    "Male",
    "Right_side",
    "Extra_temporal",
    "Number_ASMs_z",
]

if primary_base.columns.tolist() != base_primary_names:
    raise RuntimeError(
        "Unexpected primary design columns: "
        f"{primary_base.columns.tolist()}"
    )

if group_dummies.columns.tolist() != [
    "Group_1",
    "Group_2",
    "Group_3",
    "Group_4",
]:
    raise RuntimeError(
        "Unexpected group dummy columns: "
        f"{group_dummies.columns.tolist()}"
    )

term_metadata = {
    "Pathology_DNT": (
        "Pathology",
        "DNT vs HS",
    ),
    "Pathology_FCD": (
        "Pathology",
        "FCD vs HS",
    ),
    "Pathology_CAV": (
        "Pathology",
        "CAV vs HS",
    ),
    "Pathology_DUAL": (
        "Pathology",
        "Dual vs HS",
    ),
    "Pathology_OTHER": (
        "Pathology",
        "Other vs HS",
    ),
    "Age_band_rank_z": (
        "Age",
        "Age-band rank (per 1 SD)",
    ),
    "Male": (
        "Sex",
        "Male vs female",
    ),
    "Right_side": (
        "Operation side",
        "Right vs left",
    ),
    "Extra_temporal": (
        "Operation type",
        "Extra-temporal vs temporal",
    ),
    "Number_ASMs_z": (
        "Number of ASMs",
        "Number of ASMs (per 1 SD)",
    ),
    "Region_size_z": (
        "Region size",
        "Reference-mask voxel count (per 1 SD)",
    ),
}

coefficient_rows = []
nuisance_rows = []
model_rows = []

for region in targets:
    entropy_column = f"{region}_entropy_mean"
    size_column = f"{region}_n_voxels"

    primary = primary_base.copy()
    primary["Region_size_z"] = z_score_population(
        analysis[size_column],
        f"{region} reference-mask size",
    )

    primary_names = [
        *base_primary_names,
        "Region_size_z",
    ]

    if len(primary_names) != 11:
        raise RuntimeError(
            f"Expected 11 primary coefficients for {region}"
        )

    design = pd.concat(
        [primary, group_dummies],
        axis=1,
    )
    design = sm.add_constant(
        design,
        has_constant="add",
    ).astype(float)

    outcome = pd.to_numeric(
        analysis[entropy_column],
        errors="raise",
    ).astype(float)

    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise RuntimeError(
            f"Rank-deficient design matrix for {region}"
        )

    model = sm.OLS(
        outcome,
        design,
        missing="raise",
    ).fit()

    robust = sm.OLS(
        outcome,
        design,
        missing="raise",
    ).fit(cov_type="HC3")

    robust_parameters = pd.Series(
        robust.params,
        index=model.params.index,
    )
    robust_standard_errors = pd.Series(
        robust.bse,
        index=model.params.index,
    )
    robust_p_values = pd.Series(
        robust.pvalues,
        index=model.params.index,
    )

    conventional_ci = model.conf_int(alpha=0.05)
    robust_ci = pd.DataFrame(
        robust.conf_int(alpha=0.05),
        index=model.params.index,
        columns=[0, 1],
    )

    for term in primary_names:
        family, label = term_metadata[term]

        coefficient_rows.append({
            "region": region,
            "n": int(model.nobs),
            "predictor_family": family,
            "term": term,
            "contrast": label,
            "beta": model.params[term],
            "se_conventional": model.bse[term],
            "ci95_low_conventional": conventional_ci.loc[term, 0],
            "ci95_high_conventional": conventional_ci.loc[term, 1],
            "p_conventional": model.pvalues[term],
            "se_hc3": robust_standard_errors[term],
            "ci95_low_hc3": robust_ci.loc[term, 0],
            "ci95_high_hc3": robust_ci.loc[term, 1],
            "p_hc3": robust_p_values[term],
        })

    for term in group_dummies.columns:
        nuisance_rows.append({
            "region": region,
            "n": int(model.nobs),
            "term": term,
            "contrast": f"{term.replace('_', ' ')} vs Group 0",
            "beta": model.params[term],
            "se_conventional": model.bse[term],
            "p_conventional": model.pvalues[term],
            "se_hc3": robust_standard_errors[term],
            "p_hc3": robust_p_values[term],
            "included_in_165_fdr_family": False,
        })

    influence = model.get_influence()
    cooks_distance = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag
    studentized = influence.resid_studentized_external

    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(
        model.resid,
        model.model.exog,
    )

    jb_statistic, jb_p, residual_skew, residual_kurtosis = (
        jarque_bera(model.resid)
    )

    model_rows.append({
        "region": region,
        "n": int(model.nobs),
        "r_squared": model.rsquared,
        "adjusted_r_squared": model.rsquared_adj,
        "aic": model.aic,
        "bic": model.bic,
        "condition_number": np.linalg.cond(
            model.model.exog
        ),
        "breusch_pagan_lm": bp_lm,
        "breusch_pagan_lm_p": bp_lm_p,
        "breusch_pagan_f": bp_f,
        "breusch_pagan_f_p": bp_f_p,
        "jarque_bera": jb_statistic,
        "jarque_bera_p": jb_p,
        "residual_skew": residual_skew,
        "residual_kurtosis": residual_kurtosis,
        "max_cooks_distance": np.max(cooks_distance),
        "n_cooks_distance_gt_4_over_n": int(
            (cooks_distance > 4.0 / model.nobs).sum()
        ),
        "max_leverage": np.max(leverage),
        "max_abs_externally_studentized_residual": np.max(
            np.abs(studentized)
        ),
    })

results = pd.DataFrame(coefficient_rows)
nuisance = pd.DataFrame(nuisance_rows)
models = pd.DataFrame(model_rows)

if len(results) != 165:
    raise RuntimeError(
        f"Expected 165 primary coefficients; found {len(results)}"
    )

if len(nuisance) != 60:
    raise RuntimeError(
        f"Expected 60 group nuisance coefficients; found {len(nuisance)}"
    )

if not (results.groupby("region").size() == 11).all():
    raise RuntimeError(
        "Not every regional model supplied 11 primary coefficients"
    )

results["p_fdr_conventional_165"] = bh_fdr(
    results["p_conventional"]
)
results["p_fdr_hc3_165"] = bh_fdr(
    results["p_hc3"]
)

results = results.sort_values(
    ["p_conventional", "region", "term"]
).reset_index(drop=True)

nuisance = nuisance.sort_values(
    ["region", "term"]
).reset_index(drop=True)

models = models.sort_values("region").reset_index(drop=True)

results.to_csv(result_path, index=False)
nuisance.to_csv(nuisance_path, index=False)
models.to_csv(model_path, index=False)

summary_columns = [
    "region",
    "predictor_family",
    "contrast",
    "beta",
    "p_conventional",
    "p_fdr_conventional_165",
    "p_hc3",
    "p_fdr_hc3_165",
]

top_conventional = results.nsmallest(
    15,
    "p_conventional",
)[summary_columns]

top_hc3 = results.nsmallest(
    15,
    "p_hc3",
)[summary_columns]

significant_conventional = results[
    results["p_fdr_conventional_165"] < 0.05
][summary_columns]

significant_hc3 = results[
    results["p_fdr_hc3_165"] < 0.05
][summary_columns]

pathology_counts = (
    analysis["Pathology_grp"]
    .value_counts()
    .reindex(pathology_order)
)

group_counts = (
    analysis["group_clean"]
    .value_counts()
    .reindex(expected_groups)
)

summary_lines = [
    "RQ3 GROUP-ADJUSTED MULTIVARIABLE REGRESSION",
    "",
    "INPUT",
    f"Input rows: {len(d)}",
    f"Complete epilepsy-patient rows: {len(analysis)}",
    f"Unique analysed subjects: {analysis['subject_id'].nunique()}",
    f"Regional models: {len(models)}",
    f"Primary coefficients: {len(results)} (15 x 11)",
    f"Group nuisance coefficients: {len(nuisance)} (15 x 4; excluded from FDR family)",
    "Outcome: reference-masked total predictive entropy",
    "Reference-mask size: region-specific *_n_voxels",
    "Reference groups: HS, female, left, temporal, disjoint Group 0",
    "Standardisation: age-band rank, ASM count, and regional reference-mask size; population SD (ddof=0)",
    "ILAE outcome: excluded from RQ3",
    "",
    "AGE-BAND ORDER",
    age_level_table.to_string(index=False),
    "",
    "PATHOLOGY COUNTS",
    pathology_counts.to_string(),
    "",
    "DISJOINT-GROUP COUNTS IN ANALYSIS",
    group_counts.to_string(),
    "",
    "INFERENCE SUMMARY",
    f"Conventional OLS raw p < 0.05: {int((results['p_conventional'] < 0.05).sum())}/165",
    f"Conventional OLS FDR q < 0.05: {int((results['p_fdr_conventional_165'] < 0.05).sum())}/165",
    f"Minimum conventional FDR q: {results['p_fdr_conventional_165'].min():.8g}",
    f"HC3 raw p < 0.05: {int((results['p_hc3'] < 0.05).sum())}/165",
    f"HC3 FDR q < 0.05: {int((results['p_fdr_hc3_165'] < 0.05).sum())}/165",
    f"Minimum HC3 FDR q: {results['p_fdr_hc3_165'].min():.8g}",
    "",
    "MODEL FIT SUMMARY",
    f"Mean R-squared: {models['r_squared'].mean():.8g}",
    f"Mean adjusted R-squared: {models['adjusted_r_squared'].mean():.8g}",
    f"Median adjusted R-squared: {models['adjusted_r_squared'].median():.8g}",
    f"Adjusted R-squared range: {models['adjusted_r_squared'].min():.8g} to {models['adjusted_r_squared'].max():.8g}",
    "",
    "15 SMALLEST CONVENTIONAL OLS P-VALUES",
    top_conventional.to_string(index=False),
    "",
    "15 SMALLEST HC3 P-VALUES",
    top_hc3.to_string(index=False),
    "",
    "FDR-SIGNIFICANT CONVENTIONAL OLS RESULTS",
    (
        significant_conventional.to_string(index=False)
        if len(significant_conventional)
        else "None"
    ),
    "",
    "FDR-SIGNIFICANT HC3 RESULTS",
    (
        significant_hc3.to_string(index=False)
        if len(significant_hc3)
        else "None"
    ),
    "",
    "MODEL DIAGNOSTICS",
    models.to_string(index=False),
    "",
    f"Saved 165-coefficient results: {result_path}",
    f"Saved group nuisance coefficients: {nuisance_path}",
    f"Saved model fit and diagnostics: {model_path}",
]

summary_text = "\n".join(summary_lines)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:
    file.write(summary_text + "\n")

print(summary_text)
print("\nSaved summary:", summary_path)
