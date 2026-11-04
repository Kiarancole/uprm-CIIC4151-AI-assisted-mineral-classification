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

## Official Run Checklist

1. Review and approve `configs/preliminary_experiment.json` before beginning official training.
2. Confirm that `high_224.zip`, `medium_224.zip`, and `low_224.zip` are available in `/content/drive/MyDrive/CSE_Capstone_Data/cache_224`.
3. Agree on one Git commit hash. Every teammate must train using that exact version.
4. Use the same Colab GPU type when comparing assimilation time. A T4 is the preferred shared target. If different GPUs are used, the training times are not directly comparable.
5. Run one smoke test for each architecture before beginning official experiments. Smoke-test results are not included in the comparison.
6. Once official training begins, do not change the configuration for an individual model. If a shared setting changes, all affected experiments must be rerun.

The loader verifies the dataset seed, split-manifest hashes, mineral groups, image size, padding color, class labels, image counts, and cache checksums before training.

## Running the Experiments

1. Open `notebooks/03_train_assigned_model.ipynb` in Google Colab and select a GPU runtime.
2. Run the setup cells to mount Drive, download the agreed repository version, and install the shared training package.
3. Change only `MODEL_NAME` to the assigned architecture and confirm the shared Drive paths.
4. Run the one-group smoke test.
5. After the smoke test succeeds, run the assigned model on all three mineral groups. The smoke-test results are not official and must not be included in the comparison.
6. Keep large `.pt` checkpoints in Google Drive. Copy the small JSON results into `results/preliminary/` for version control.

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

After all nine official experiments are available, generate the team comparison with:

```bash
python scripts/compare_preliminary.py
```

The script verifies that all runs used the same configuration and reports whether their Git commits and training hardware match. It writes the combined results to `results/preliminary/team_comparison.json`.

