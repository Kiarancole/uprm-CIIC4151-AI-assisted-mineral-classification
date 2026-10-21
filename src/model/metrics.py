"""Shared evaluation functions used by every architecture."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, top_k_accuracy_score


def expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    """Measure the confidence-versus-accuracy gap using equal-width bins.

    A low ECE means that predictions made with, for example, about 80% confidence
    are correct about 80% of the time. The proposal specifies ten bins.
    """
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        in_bin = (confidence > lower) & (confidence <= upper)
        if np.any(in_bin):
            bin_accuracy = float(np.mean(correct[in_bin]))
            bin_confidence = float(np.mean(confidence[in_bin]))
            error += float(np.mean(in_bin)) * abs(bin_accuracy - bin_confidence)
    return float(error)


def classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    calibration_bins: int = 10,
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(len(class_names))
    # These are calculated separately for every mineral. Macro F1 below gives
    # each mineral equal importance, even when class sizes are different.
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )
    per_class = {
        class_name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(class_names)
    }
    # Top-3 matches the planned interface, which can show several likely minerals.
    top3 = top_k_accuracy_score(targets, probabilities, k=min(3, len(class_names)), labels=labels)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(np.mean(f1)),
        "worst_class_recall": float(np.min(recall)),
        "top3_accuracy": float(top3),
        "expected_calibration_error": expected_calibration_error(targets, probabilities, calibration_bins),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
    }


def priority_weighted_score(group_macro_f1: dict[str, float], weights: dict[str, float]) -> float:
    """Combine group macro F1 values using the proposal's priority weights."""
    missing = set(weights) - set(group_macro_f1)
    if missing:
        raise ValueError(f"Missing macro F1 results for groups: {sorted(missing)}")
    return float(sum(float(weights[group]) * float(group_macro_f1[group]) for group in weights))

