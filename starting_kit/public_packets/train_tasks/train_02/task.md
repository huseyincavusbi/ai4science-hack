# Backdoor-Robust Digit Classification

Build a digit classifier from a limited labeled set that remains reliable when
localized visual artifacts create shortcut correlations or mislabeled examples
in the training data.

The training manifest is `data/train.csv`. It contains:

- `id`: row identifier
- `image_index`: row index into `data/train_images.npz`
- `label`: visible digit label

Some training examples may contain localized visual artifacts. Some artifacts
are harmless, some are spuriously correlated with labels, and a smaller subset
may be attached to mislabeled examples. The task is to predict the true digit
identity rather than relying on artifact shortcuts.

The test manifest is `data/test.csv`. It contains `id` and `image_index`, but no
labels. Test images are stored in `data/test_images.npz`. The hidden test set
contains ordinary digit images plus artifact-stress images, including variants
not seen exactly in training. Both image arrays have shape `(n, 28, 28)` and
contain unsigned 8-bit grayscale pixels.

Write one digit prediction for every test row. The scorer reports:

- `clean_acc`: accuracy on ordinary hidden test images
- `seen_trigger_acc`: accuracy on hidden stress images with familiar artifact families
- `unseen_trigger_acc`: accuracy on hidden stress images with held-out artifact variants
- `attack_success_rate`: fraction of artifact-stress images predicted as the
  artifact-associated wrong label
- `defense_score`: `0.5 * clean_acc + 0.3 * seen_trigger_acc + 0.2 * unseen_trigger_acc`

The primary `score` is a normalized version of `defense_score`; higher is
better.

Your `run.sh` must write `predictions.csv` with exactly these columns:

```csv
id,prediction
test_00000,7
test_00001,0
```

`prediction` must be an integer digit from `0` through `9`. Do not include
metrics or confidence scores in the output CSV. Do not use external labeled
image datasets, external labels, hidden reference files, or scorer internals.
