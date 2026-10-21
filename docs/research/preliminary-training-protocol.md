# Preliminary Training Protocol

This document defines the settings that must remain shared across the nine preliminary experiments.

## Experimental matrix

| Assigned model | High group | Medium group | Low group |
| --- | --- | --- | --- |
| ResNet-18 | Train and evaluate | Train and evaluate | Train and evaluate |
| EfficientNet-B0 | Train and evaluate | Train and evaluate | Train and evaluate |
| EfficientNet-B3 | Train and evaluate | Train and evaluate | Train and evaluate |

Each teammate owns one row. Running all three groups produces three experiments per teammate and nine experiments for the team.

## What must be identical

The committed file `configs/preliminary_experiment.json` is the source of truth for:

- Seed: 42
- Random initialization: `weights=None`
- Input size: 224x224
- Fixed 70/15/15 image splits
- Batch size and data-loader settings
- Training augmentations and normalization
- Weighted cross-entropy loss
- AdamW optimizer, learning rate, weight decay, scheduler, and epoch count
- Validation model-selection rule
- Evaluation and latency code

Do not copy the training loop into separate notebooks. If the protocol changes, update the shared configuration once and record the reason in the pull request.

Different architectures cannot contain identical initial tensors because their layers have different shapes. The fair comparison uses the same initialization policy and seed, not identical weight values.

## Before official training

1. Merge the dataset split and 224x224 cache pull requests.
2. Confirm the three cache zip checksums against `cache_meta.json`.
3. Run one smoke test for each architecture.
4. Confirm the values in `configs/preliminary_experiment.json` with the team and Professor Kumar.
5. Tag or record the exact Git commit used for all official runs.

Once the first official experiment begins, do not change the protocol for another model. A necessary correction means all affected experiments must be rerun.

The loader also verifies the merged metadata: seed 42, matching split-manifest hashes, mineral groups, 224x224 image size, padding color `[124, 116, 104]`, class-to-label mappings, per-split image counts, and each zip's SHA-256 checksum.

## Outputs

Small JSON results belong in `results/preliminary/` and may be committed. Large `.pt` checkpoints stay in the shared Drive folder and are ignored by Git.

Each run records:

- Configuration and seed
- Training and validation history
- Accuracy, macro F1, top-3 accuracy, worst-class recall, 10-bin ECE, and per-class metrics
- Confusion matrix
- Training time and CPU inference latency
- Parameter counts
- Software versions
- Exact Git commit and hardware name
- SHA-256 checksum of the best checkpoint

The aggregate file combines the three group macro F1 scores into the proposal's Priority-Weighted Score and records their cross-group standard deviation.

## Smoke test

A smoke test runs one epoch and verifies that the data, model, metrics, and output paths work. It is not an official experiment and its numbers must not be included in the comparison.

