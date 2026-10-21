# Model Component

This directory contains the single shared pipeline used for all preliminary model experiments.

- `config.py`: loads and validates the frozen experiment configuration.
- `data.py`: verifies and extracts the shared 224x224 caches and builds data loaders.
- `models.py`: constructs ResNet-18, EfficientNet-B0, or EfficientNet-B3 with random weights.
- `reproducibility.py`: applies the shared random seed and deterministic settings.
- `metrics.py`: calculates the same metrics for every experiment.
- `trainer.py`: contains the common training, validation, checkpoint, timing, and result logic.

Team members should not create separate training implementations. Select the assigned model through `scripts/run_preliminary.py` or the shared Colab notebook.

See [`docs/research/preliminary-training-protocol.md`](../../docs/research/preliminary-training-protocol.md) for the comparison rules.
