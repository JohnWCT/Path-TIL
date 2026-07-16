"""Sparse multiclass loss builders for HNSCC methodology experiments."""

from __future__ import annotations

import numpy as np


LOSS_NAMES = (
    "weighted_ce",
    "focal_gamma1",
    "focal_gamma2",
    "class_balanced_focal",
    "label_smoothing_ce",
    "logit_adjusted_ce",
    "sparse_ce",
)


def class_frequencies(labels, num_classes):
    """Return per-class counts and frequencies for integer labels."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        raise ValueError("labels must not be empty")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("labels must be in [0, num_classes)")
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    frequencies = counts / float(labels.size)
    return counts, frequencies


def effective_number_weights(counts, beta=0.9999):
    """Class-balanced weights from Cui et al. effective number of samples."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("counts must be a non-empty 1-D array")
    if not (0.0 <= beta < 1.0):
        raise ValueError("beta must be in [0, 1)")
    effective = 1.0 - np.power(beta, np.maximum(counts, 1.0))
    weights = (1.0 - beta) / np.maximum(effective, 1e-12)
    weights = weights / weights.mean()
    return weights


def logit_adjustment(frequencies, tau=1.0):
    """Logit adjustment offsets: tau * log(pi)."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies must be a non-empty 1-D array")
    if np.any(frequencies <= 0):
        raise ValueError("frequencies must be positive")
    if tau < 0:
        raise ValueError("tau must be non-negative")
    return tau * np.log(frequencies)


def build_keras_loss(tf, name, num_classes, class_weights=None, frequencies=None):
    """Return a Keras-compatible sparse multiclass loss callable."""
    if name not in LOSS_NAMES:
        raise ValueError("Unknown loss: {0}; expected one of {1}".format(name, LOSS_NAMES))
    epsilon = tf.keras.backend.epsilon()

    def sparse_ce(y_true, y_pred):
        labels = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        probabilities = tf.clip_by_value(
            tf.cast(y_pred, tf.float32), epsilon, 1.0 - epsilon
        )
        one_hot = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
        return -tf.reduce_sum(one_hot * tf.math.log(probabilities), axis=-1)

    if name == "sparse_ce":
        loss = sparse_ce
        loss.__name__ = "sparse_ce"
        return loss

    if name == "weighted_ce":
        if class_weights is None:
            raise ValueError("weighted_ce requires class_weights")
        weight_tensor = tf.constant(
            [float(class_weights[index]) for index in range(num_classes)],
            dtype=tf.float32,
        )

        def weighted_ce(y_true, y_pred):
            labels = tf.reshape(tf.cast(y_true, tf.int32), [-1])
            base = sparse_ce(y_true, y_pred)
            sample_weight = tf.gather(weight_tensor, labels)
            return base * sample_weight

        weighted_ce.__name__ = "weighted_ce"
        return weighted_ce

    if name in ("focal_gamma1", "focal_gamma2", "class_balanced_focal"):
        gamma = 1.0 if name == "focal_gamma1" else 2.0
        if name == "class_balanced_focal":
            if class_weights is None:
                raise ValueError("class_balanced_focal requires class_weights")
            weight_tensor = tf.constant(
                [float(class_weights[index]) for index in range(num_classes)],
                dtype=tf.float32,
            )
        else:
            weight_tensor = None

        def focal(y_true, y_pred):
            labels = tf.reshape(tf.cast(y_true, tf.int32), [-1])
            probabilities = tf.clip_by_value(
                tf.cast(y_pred, tf.float32), epsilon, 1.0
            )
            row = tf.range(tf.shape(labels)[0], dtype=tf.int32)
            true_probability = tf.gather_nd(
                probabilities, tf.stack([row, labels], axis=1)
            )
            loss_value = -tf.pow(1.0 - true_probability, gamma) * tf.math.log(
                true_probability
            )
            if weight_tensor is not None:
                loss_value = loss_value * tf.gather(weight_tensor, labels)
            return loss_value

        focal.__name__ = name
        return focal

    if name == "label_smoothing_ce":
        smoothing = 0.1

        def label_smoothing_ce(y_true, y_pred):
            labels = tf.reshape(tf.cast(y_true, tf.int32), [-1])
            probabilities = tf.clip_by_value(
                tf.cast(y_pred, tf.float32), epsilon, 1.0 - epsilon
            )
            one_hot = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
            soft = one_hot * (1.0 - smoothing) + (
                smoothing / float(num_classes)
            )
            return -tf.reduce_sum(soft * tf.math.log(probabilities), axis=-1)

        label_smoothing_ce.__name__ = "label_smoothing_ce"
        return label_smoothing_ce

    if name == "logit_adjusted_ce":
        if frequencies is None:
            raise ValueError("logit_adjusted_ce requires frequencies")
        adjustments = tf.constant(
            logit_adjustment(frequencies, tau=1.0).astype(np.float32),
            dtype=tf.float32,
        )

        def logit_adjusted_ce(y_true, y_pred):
            labels = tf.reshape(tf.cast(y_true, tf.int32), [-1])
            probabilities = tf.clip_by_value(
                tf.cast(y_pred, tf.float32), epsilon, 1.0 - epsilon
            )
            logits = tf.math.log(probabilities) - adjustments
            log_probs = logits - tf.reduce_logsumexp(logits, axis=-1, keepdims=True)
            one_hot = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
            return -tf.reduce_sum(one_hot * log_probs, axis=-1)

        logit_adjusted_ce.__name__ = "logit_adjusted_ce"
        return logit_adjusted_ce

    raise ValueError("Unhandled loss: {0}".format(name))
