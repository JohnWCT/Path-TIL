"""Validation-tuned probability thresholds for hard TIL estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def hard_til_from_labels(labels):
    """Compute Positive / (Positive + Negative); NaN if denominator is zero."""
    labels = pd.Series(labels).astype(str)
    positive = int((labels == "positive").sum())
    negative = int((labels == "negative").sum())
    denominator = positive + negative
    if denominator == 0:
        return np.nan
    return positive / float(denominator)


def predict_labels_with_threshold(probabilities, threshold, labels=("positive", "negative", "other")):
    """Assign positive if P(positive) >= threshold, else argmax over the rest.

    This keeps the positive decision threshold explicit while still producing a
    three-way label for hard TIL aggregation.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise ValueError("probabilities must have shape [N, {0}]".format(len(labels)))
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1]")
    positive_index = list(labels).index("positive")
    predicted = []
    for row in probabilities:
        if row[positive_index] >= threshold:
            predicted.append(labels[positive_index])
        else:
            masked = row.copy()
            masked[positive_index] = -np.inf
            predicted.append(labels[int(np.argmax(masked))])
    return np.asarray(predicted)


def til_absolute_error(y_true_labels, y_pred_labels):
    gt = hard_til_from_labels(y_true_labels)
    pred = hard_til_from_labels(y_pred_labels)
    if not (np.isfinite(gt) and np.isfinite(pred)):
        return np.nan
    return abs(gt - pred)


def tune_positive_threshold(validation_frame, grid=None):
    """Tune positive probability threshold on validation predictions only."""
    required = {
        "y_true_label",
        "prob_positive",
        "prob_negative",
        "prob_other",
    }
    missing = sorted(required - set(validation_frame.columns))
    if missing:
        raise ValueError("validation frame missing columns: {0}".format(missing))
    if validation_frame.empty:
        raise ValueError("validation frame is empty")
    if grid is None:
        grid = np.round(np.linspace(0.1, 0.9, 17), 4)
    probabilities = validation_frame[
        ["prob_positive", "prob_negative", "prob_other"]
    ].to_numpy(dtype=np.float64)
    y_true = validation_frame["y_true_label"].tolist()
    best_threshold = None
    best_error = np.inf
    rows = []
    for threshold in grid:
        predicted = predict_labels_with_threshold(probabilities, float(threshold))
        error = til_absolute_error(y_true, predicted)
        rows.append({"threshold": float(threshold), "abs_error": error})
        if np.isfinite(error) and (
            error < best_error
            or (
                np.isclose(error, best_error)
                and (best_threshold is None or threshold < best_threshold)
            )
        ):
            best_error = float(error)
            best_threshold = float(threshold)
    if best_threshold is None:
        raise ValueError("Unable to tune threshold: no finite validation TIL error")
    return best_threshold, best_error, pd.DataFrame(rows)


def apply_threshold_to_frame(frame, threshold):
    """Return a copy with thresholded hard predictions and updated correctness."""
    probabilities = frame[
        ["prob_positive", "prob_negative", "prob_other"]
    ].to_numpy(dtype=np.float64)
    predicted = predict_labels_with_threshold(probabilities, threshold)
    result = frame.copy()
    result["y_pred_label"] = predicted
    result["y_pred_idx"] = [
        {"positive": 0, "negative": 1, "other": 2}[label] for label in predicted
    ]
    if "y_true_label" in result.columns:
        result["correct"] = result["y_true_label"].astype(str) == result[
            "y_pred_label"
        ].astype(str)
    result["positive_threshold"] = float(threshold)
    return result
