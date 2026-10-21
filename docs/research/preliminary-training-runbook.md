# Preliminary Training Runbook

This is the short procedure every teammate should follow. The notebook is only an entry point; the actual training logic is shared in `src/model/` so nobody has a slightly different copy.

## Team assignments

| Teammate | Model value to enter |
| --- | --- |
| ResNet-18 owner | `resnet18` |
| EfficientNet-B0 owner | `efficientnet_b0` |
| EfficientNet-B3 owner | `efficientnet_b3` |

Each owner runs their model on `high`, `medium`, and `low`. This creates nine team experiments in total.

## Before training

1. Make sure the dataset-split and 224x224 cache pull requests have been merged.
2. Confirm that `high_224.zip`, `medium_224.zip`, and `low_224.zip` are in `/content/drive/MyDrive/CSE_Capstone_Data/cache_224`, matching notebook `02_make_cache.ipynb`.
3. Put `cache_meta.json` in `notebooks/cache_224/` in the repository.
4. Ask the team to review `configs/preliminary_experiment.json`. Values such as 20 epochs, batch size 16, AdamW, and learning rate 0.001 are the proposed shared starting protocol and must be approved before official runs.
5. Agree on one Git commit hash. Everyone must train from that exact version.

For a fair assimilation-time comparison, use the same GPU type for all nine runs (the proposed target is a Colab T4). The pipeline records the hardware name. If Colab assigns a different GPU, reconnect to an appropriate runtime or clearly state that the recorded training times are not directly comparable.

## In Google Colab

1. Open `notebooks/03_train_assigned_model.ipynb` in Colab.
2. Select a GPU runtime.
3. Run the setup cells.
4. Change only `MODEL_NAME` to your assigned value.
5. Confirm the three `CSE_Capstone_Data` Drive paths for caches, checkpoints, and results. Change them only if the team deliberately uses a different shared folder.
6. Run the one-group smoke test. Its metrics are not official.
7. If it succeeds, run all three official experiments.
8. The notebook saves both checkpoints and results in Drive so a disconnected runtime does not erase them. Copy only your small JSON result files into the repository's matching `results/preliminary/` folder and commit them. Leave the large `.pt` checkpoints in Drive.

Do not edit the seed, splits, transforms, epochs, optimizer, loss, or metrics for your individual model. A protocol improvement should be discussed and changed once in the shared configuration. If a setting changes after official training begins, rerun every affected experiment.

## What the pipeline records

- Accuracy, macro F1, Top-3 accuracy, worst-class recall, and 10-bin ECE
- Precision, recall, F1, and support for every mineral
- Confusion matrix
- Training time and CPU inference latency
- Total and trainable parameter counts
- Best epoch and model-checkpoint checksum
- Shared configuration and software versions

The three group macro F1 scores are also combined into the proposed Priority-Weighted Score: 50% high priority, 30% medium priority, and 20% low priority. Their population standard deviation is recorded as the cross-group consistency measure.

After all nine official results are copied into the repository, create the team summary with:

```bash
python scripts/compare_preliminary.py
```

The script refuses to compare runs with different shared configurations, reports whether the Git commits and hardware match, and writes `results/preliminary/team_comparison.json`. The team still makes the final candidate decision using the proposal's PWS, recall, complexity, and latency rules.

## Recommended official reading

The complete implementation-to-source mapping is in [`training-code-references.md`](training-code-references.md). The most useful starting points are:

- [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [TorchVision ImageFolder](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)
- [TorchVision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [TorchVision EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)
- [TorchVision EfficientNet-B3](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b3.html)
- [PyTorch: Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

These links are preferable to copying code from unrelated tutorials because they document the exact library used by this project.
