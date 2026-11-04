# Notebooks

This directory contains Jupyter and Google Colab notebooks used throughout the project lifecycle.

## Purpose

Notebooks in this folder are used for:

- Dataset analysis
- Model training
- Model comparison
- Evaluation

## Guidelines

- Notebooks should have clear, descriptive filenames.
- Notebooks must **not** contain credentials, API keys, or private data.
- Training notebooks should call the shared code in `src/model/`; they must not contain separate model-specific training loops.

## Order

1. `00_Capstone_Mineral_Identification.ipynb`: download the source dataset.
2. `01_make_splits.ipynb`: produce the fixed 70/15/15 split manifest.
3. `02_make_cache.ipynb`: produce the verified 224x224 group archives.
4. `03_train_assigned_model.ipynb`: select one model and run the shared pipeline on all three groups.
