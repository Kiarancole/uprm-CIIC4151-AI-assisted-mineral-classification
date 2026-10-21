# Training Code References

This page lists the sources used to design the shared training scaffold. The code follows official library documentation whenever possible so the team can verify what each PyTorch component does.

## Beginner overview

- [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) introduces tensors, datasets, data loaders, model construction, automatic differentiation, optimization, and saving models.
- [PyTorch Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) shows the overall training and evaluation workflow used by the scaffold.
- [Optimization loop](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) explains the forward pass, loss calculation, `zero_grad`, backpropagation, and optimizer update.

## Data and reproducibility

- [TorchVision `ImageFolder`](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) explains the `<class>/<image>` directory format loaded in `data.py`.
- [PyTorch data loading](https://docs.pytorch.org/docs/stable/data.html) documents `Dataset`, `DataLoader`, batching, shuffling, workers, and worker initialization.
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html) explain random seeds, deterministic operations, cuDNN settings, and DataLoader worker seeding used in `reproducibility.py`.
- [TorchVision transforms](https://docs.pytorch.org/vision/stable/transforms.html) document the flips, rotations, tensor conversion, and normalization used for training images.

## Model architectures

- [TorchVision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [TorchVision EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)
- [TorchVision EfficientNet-B3](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b3.html)

The three builders accept a `weights` argument. The scaffold deliberately uses `weights=None`, meaning no pretrained ImageNet parameters are loaded. Supplying `num_classes=5` creates the correct output size for each five-mineral group.

## Training components

- [Cross-entropy loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) documents the classification loss and optional class weights used for imbalanced mineral groups.
- [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) documents the shared optimizer and weight decay.
- [CosineAnnealingLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) documents the learning-rate scheduler.
- [Automatic mixed precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) explain `autocast` and `GradScaler`, which reduce GPU memory use and often accelerate training.
- [Saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) recommends saving a model's `state_dict`, which is how the best validation checkpoint is stored.

## Evaluation

- [Scikit-learn precision, recall, and F-score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html) documents the per-class statistics used to calculate macro F1 and worst-class recall.
- [Scikit-learn Top-K accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html) documents Top-3 accuracy.
- Guo et al., [*On Calibration of Modern Neural Networks*](https://proceedings.mlr.press/v70/guo17a.html), describes confidence calibration and Expected Calibration Error. The proposal specifies a ten-bin ECE calculation.

The performance thresholds and Priority-Weighted Score do not come from these software references. They come from the team's approved project proposal and are stored in `configs/preliminary_experiment.json` so they remain identical for all experiments.
