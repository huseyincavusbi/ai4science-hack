# Synthetic Treatment Effect Estimation

You are given a synthetic observational study of model-improvement runs. Each
training row describes a prior run, whether it received an adaptive intervention,
and the measured improvement after the run. Treatment assignment is not random:
it depends on the observed run features.

Estimate the individual treatment effect for every test row: the expected change
in `outcome` if the adaptive intervention is applied instead of not applied for
that same unit.

The test file contains covariates only. Your `run.sh` must write
`predictions.csv` with exactly these columns:

```csv
id,prediction
test_0001,0.58
test_0002,0.74
```

`prediction` must be one numeric treatment-effect estimate for the corresponding
test `id`. The primary metric is PEHE against hidden per-unit effects, so lower
per-unit effect error is better. The scorer also reports absolute percent error
of the average treatment effect. Oracle treatment effects receive `score=1`,
`pehe=0`, and `abs_pct_error_of_ate=0`.

## Files

- `data/train.csv`: covariates, observed treatment assignment, and observed
  outcome for previous runs.
- `data/test.csv`: covariates for units to predict.
- `sample_submission.csv`: required output format.

## Columns

- `arch_family`: binary architecture-family indicator.
- `model_scale`: standardized model scale.
- `data_quality`: input-data quality index.
- `pretrain_overlap`: overlap between the run data and pretraining mixture.
- `baseline_loss`: pre-intervention evaluation loss.
- `compute_budget`: relative compute budget for the run.
- `eval_noise`: estimated evaluation noise level.
- `treatment`: observed adaptive-intervention indicator, present only in
  training data.
- `outcome`: observed post-run improvement, present only in training data.
