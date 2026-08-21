# Predictive Uncertainty in Brain MRI Segmentation

This repository contains the scripts used for my BSc thesis, *Predictive
Uncertainty in Brain MRI Segmentation: Assessing Clinical and Demographic
Variation in Epilepsy Patients*.

The scripts were written during the project and collected here so the final
analysis can be inspected and reproduced. This is thesis research code rather
than a general-purpose software package.

## Data

The project uses the IDEAS epilepsy dataset. MRI scans, clinical tables, model
weights, full subject-level results and generated NIfTI files are not included
in this repository.

To run the full pipeline, the IDEAS data must first be prepared for nnU-Net.
Some paths or options will probably need to be changed for a different computer
or cluster.

## Final model setup

The final analysis uses five models trained on five separate subject groups.
For a subject in group `i`, model `i` is left out and the other four models are
used for the ensemble prediction. This means that none of the models used for a
subject's prediction trained on that subject.

The groups were created from the five validation sets in
`splits/splits_final.json`. The resulting training split is in
`splits/splits_final_disjoint.json`.

## Main scripts

- `training/make_disjoint_splits.py`: creates and checks the final training
  split.
- `training/nnUNetTrainer_250epochs_disjoint.py`: custom nnU-Net trainer used
  for the five models.
- `inference/predict_disjoint.py`: runs the four-model prediction and calculates
  the uncertainty measures and Dice scores.
- `analysis/disjoint_analysis.py`: runs the group checks, RQ1 analysis,
  uncertainty decomposition and failure-detection analysis.
- `analysis/regression_group_adjusted.py`: runs the group-adjusted RQ3 models.
- `analysis/validate_regression.py`: independently checks the RQ3 results.
- `analysis/risk_coverage.py`: runs the risk-coverage analysis.
- `analysis/mde_sensitivity.py`: calculates the minimum detectable effects.
- `analysis/make_tables.py`: creates the final thesis tables.
- `scripts/rq1_sensitivity.py`: runs the patient-only and region-size-adjusted
  RQ1 sensitivity analyses.
- `figures/`: scripts used to generate the thesis figures.

The training and inference scripts are run first. The analysis and figure
scripts then use the resulting CSV or NIfTI files.

## Running the split and models

The split can first be checked without writing anything:

```bash
python training/make_disjoint_splits.py --source splits/splits_final.json
```

It can then be written to the nnU-Net preprocessed dataset folder with:

```bash
python training/make_disjoint_splits.py \
  --source splits/splits_final.json \
  --write \
  --out /path/to/nnUNet_preprocessed/Dataset001_IDEAS/splits_final_disjoint.json
```

After installing the custom trainer in the nnU-Net trainer variants folder,
folds 0-4 are trained with:

```bash
nnUNetv2_train 1 2d FOLD -tr nnUNetTrainer_250epochs_disjoint
```

Inference is run with:

```bash
python inference/predict_disjoint.py \
  --input /path/to/nnunet_raw/Dataset001_IDEAS/imagesTr \
  --labels /path/to/nnunet_raw/Dataset001_IDEAS/labelsTr \
  --splits /path/to/nnunet_preprocessed/Dataset001_IDEAS/splits_final_disjoint.json \
  --model_dir /path/to/nnunet_results/Dataset001_IDEAS/nnUNetTrainer_250epochs_disjoint__nnUNetPlans__2d \
  --output /path/to/uncertainty_output_disjoint_final
```

`--test` can be used to run one subject from each group before starting the
full inference job. `--save_maps` also saves the entropy and segmentation maps
used for the qualitative figures.

## Supporting results

`supporting_tables/final_disjoint/` contains compact versions of the final RQ1,
RQ2, RQ3, uncertainty decomposition and failure-detection results. These are
included so the main reported values can be checked without publishing the
restricted subject-level data.

## Environment

The main software used was nnU-Net v2.4.2, PyTorch 2.1.2, Python 3.11, NumPy,
pandas, SciPy, statsmodels, matplotlib and nibabel. Training and most analysis
were run on the Hábrók cluster.

`environment.yml` is the environment snapshot from Hábrók. It contains
cluster-specific CUDA packages, so it may not recreate correctly on another
system.

## Note

Although the model training subjects are disjoint, the models still use the
same architecture, preprocessing, optimisation and source population. They are
therefore not treated as statistically independent. The regional subgroup
analysis also uses reference masks, so it is not a prospective quality-control
score. These limitations are discussed in the thesis.
