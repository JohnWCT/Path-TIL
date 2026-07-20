"""L2-SP regularization helpers for anti-forgetting during fine-tuning."""

from __future__ import annotations


def snapshot_trainable_weights(model):
    """Capture a frozen copy of the current trainable weights."""
    import tensorflow as tf

    return [tf.identity(variable) for variable in model.trainable_variables]


def l2sp_penalty(model, theta_star):
    """Sum of squared deviations from the source-domain snapshot."""
    import tensorflow as tf

    penalty = tf.constant(0.0, dtype=tf.float32)
    for current, source in zip(model.trainable_variables, theta_star):
        penalty += tf.reduce_sum(tf.square(current - source))
    return penalty
