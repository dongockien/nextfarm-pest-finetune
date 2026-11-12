# Experimental Results

This directory contains the artifacts of two separate experiments.

## 1. Colab Synthetic Smoke Test

Directory: `colab_synthetic_smoke/`

Purpose:
- Validate the data preparation pipeline.
- Validate the fine-tuning workflow.
- Check adapter reload behavior.
- Inspect output parsing and prompt behavior.

Important limitations:
- Uses synthetic data, not real-world coffee leaf images.
- The shuffled-label control achieved 95% accuracy.
- Possible data leakage or scoring issues were reported.
- These metrics must not be interpreted as evidence of
  real-world model performance.

Contents:
- report.md: experiment report.
- meta/: training configuration and environment.
- predictions/: raw model predictions.

## 2. RoCoLe Experiment

Directory: `rocole/`

Dataset:
RoCoLe, Version 2.

Dataset DOI:
https://doi.org/10.17632/c5yvn32dzg.2

Split summary:
- Train: 99 images.
- Validation: 21 images.
- Test: 21 images.

Reported test results:
- Baseline B1: Accuracy 52%, Macro-F1 0.44.
- Fine-tuned model: Accuracy 57%, Macro-F1 0.57.

Important limitations:
- The test set contains only 21 images.
- The reported improvement is preliminary.
- Validation loss reached 0.0276 at step 50
  and increased to 0.0318 at step 60.
- The OOD smoke test showed that the fine-tuned model
  assigned labels to all 20 out-of-distribution samples.
- The experiment does not establish reliable OOD rejection.

Contents:
- report.md: experiment report.
- meta/: training configuration and environment.
- splits/: dataset split manifests and source attribution.
- predictions/: raw model predictions.

## Reproducibility

This repository contains experiment metadata and prediction
artifacts, not the original image dataset or model checkpoints.

See the root README.md for setup and execution instructions.
