# Colored Digit Domain Generalization

Build a binary classifier for colored handwritten digit images under
distribution shift.

The training set comes from two labeled source domains. The test set comes from
a held-out domain with a different relationship between visual nuisance factors
and the binary digit label. Your goal is to improve held-out-domain accuracy
while keeping the solution efficient enough to run inside the task runtime.

Training labels are provided. Test labels are hidden.

## Files

- `data/train.csv`: combined training manifest with `id`, `image_path`, and
  noisy binary `label`.
- `data/train_source0.csv`: first labeled source-domain training manifest.
- `data/train_source1.csv`: second labeled source-domain training manifest.
- `data/test.csv`: held-out-domain test manifest with `id` and `image_path`.
- `data/images.zip`: RGB PNG images referenced by both manifests. The
  `image_path` values are member paths inside this archive. If you extract the
  archive at the task root, the same paths become ordinary files on disk.
- `sample_submission.csv`: valid output format for all test IDs.

The binary labels are noisy digit-class labels. Do not assume that a single
visual cue is stable across domains.

## Submission

Your `run.sh` is called with `--task`, `--output`, and `--output-dir`. Write
your predictions to:

```text
<output-dir>/predictions.csv
```

The file must contain exactly two columns:

```csv
id,prediction
test_0000,0
test_0001,1
```

`prediction` must be `0` or `1` for every row in `data/test.csv`.

## Scoring

The scorer compares `predictions.csv` against hidden labels for the held-out
domain and reports held-out accuracy. The normalized score maps 50% accuracy to
`0` and perfect accuracy to `1`.
