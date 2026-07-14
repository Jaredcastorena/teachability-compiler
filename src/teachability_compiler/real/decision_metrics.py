"""Decision-validity metrics for transition-predictor rankings."""

from __future__ import annotations

import numpy as np


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Kendall tau-a with ties counted as neither concordant nor discordant."""
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)

    if x_array.ndim != 1 or y_array.ndim != 1:
        raise ValueError("kendall_tau expects two one-dimensional arrays")
    if x_array.shape != y_array.shape:
        raise ValueError("kendall_tau expects arrays with identical shape")

    n = int(x_array.shape[0])
    if n < 2:
        raise ValueError("kendall_tau expects at least two values")

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            product = (x_array[i] - x_array[j]) * (y_array[i] - y_array[j])
            if product > 0.0:
                concordant += 1
            elif product < 0.0:
                discordant += 1

    denominator = n * (n - 1) / 2.0
    return float((concordant - discordant) / denominator)


def decision_metrics(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> dict[str, float]:
    """Compute top-k, rank-correlation, and regret metrics for a decision set."""
    true_array = np.asarray(true_values, dtype=np.float64)
    predicted_array = np.asarray(predicted_values, dtype=np.float64)

    if true_array.ndim != 1 or predicted_array.ndim != 1:
        raise ValueError("decision_metrics expects two one-dimensional arrays")
    if true_array.shape != predicted_array.shape:
        raise ValueError("decision_metrics expects arrays with identical shape")

    n = int(true_array.shape[0])
    if n < 3:
        raise ValueError("decision_metrics expects at least three candidate values")

    true_argmax = int(np.argmax(true_array))
    predicted_argmax = int(np.argmax(predicted_array))

    true_top3 = set(int(index) for index in np.argsort(true_array)[-3:])
    predicted_top3 = set(int(index) for index in np.argsort(predicted_array)[-3:])

    return {
        "top1_agreement": float(true_argmax == predicted_argmax),
        "top3_recall": float(len(true_top3 & predicted_top3) / 3.0),
        "kendall_tau": kendall_tau(true_array, predicted_array),
        "selected_regret": float(np.max(true_array) - true_array[predicted_argmax]),
    }
