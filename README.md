# AI-Assisted Mineral Identification

An academic capstone project exploring the use of convolutional neural networks (CNNs) to identify mineral specimens from photographs.

## Overview

The project will compare image-classification models trained on mineral photographs and make the selected model available through a web application. Users will upload a photograph of a mineral specimen and receive likely mineral classifications, confidence scores, and basic reference information.

The classifier is intended to provide preliminary visual suggestions. It will not replace expert identification or laboratory testing.

## Main Goals

- Train and compare multiple CNN architectures.
- Study how the selection of mineral classes affects classification performance.
- Evaluate the models using independently collected images.
- Deploy the selected model to a remote prediction service.
- Build a web interface for uploading images and viewing results.
- Document the reasoning, experiments, and limitations behind the final approach.

## Approach

The initial training data will come from the [Mineral Photos dataset on Kaggle](https://www.kaggle.com/datasets/floriangeillon/mineral-photos). Separate validation and test images will be collected from external sources and, if available, resources from the University of Puerto Rico at Mayagüez geology department.

Model development will begin in Google Colab using Python, PyTorch, and TorchVision. Candidate models will be trained from random weight initialization and compared using measures such as accuracy, macro F1 score, per-class performance, model complexity, and training and inference time.

The final system will use a web interface connected to a remotely hosted model. Geographic heatmaps and drilling recommendations may be considered as future work after the image-classification system is completed.

## Status

The project is currently under development. The supported mineral classes, final model, success targets, and deployment platform will be determined through experimentation and consultation with the project advisor.

## Team

Developed by a three-person senior software engineering capstone team at the University of Puerto Rico at Mayagüez under the guidance of Professor Kumar.
