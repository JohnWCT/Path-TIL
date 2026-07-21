import unittest

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - host without TF
    tf = None

from path_til.l2sp import (
    l2sp_penalty,
    matched_l2sp_pairs,
    snapshot_trainable_weights,
    snapshot_weights_by_name,
)


@unittest.skipIf(tf is None, "tensorflow not installed")
class L2SPTests(unittest.TestCase):
    def test_l2sp_penalty_zero_for_identical_weights(self):
        model = tf.keras.Sequential([tf.keras.layers.Dense(3, input_shape=(4,))])
        theta_star = snapshot_trainable_weights(model)
        penalty = l2sp_penalty(model, theta_star)
        self.assertEqual(float(penalty.numpy()), 0.0)

    def test_l2sp_penalty_positive_after_weight_change(self):
        model = tf.keras.Sequential([tf.keras.layers.Dense(3, input_shape=(4,))])
        theta_star = snapshot_trainable_weights(model)
        for variable in model.trainable_variables:
            variable.assign_add(tf.ones_like(variable) * 0.01)
        penalty = l2sp_penalty(model, theta_star)
        self.assertGreater(float(penalty.numpy()), 0.0)

    def test_name_keyed_snapshot_survives_trainable_set_change(self):
        inputs = tf.keras.Input(shape=(4,))
        hidden = tf.keras.layers.Dense(8, name="hidden")(inputs)
        outputs = tf.keras.layers.Dense(3, name="head")(hidden)
        model = tf.keras.Model(inputs, outputs)
        theta_star = snapshot_weights_by_name(model)

        model.get_layer("hidden").trainable = False
        penalty_head_only = l2sp_penalty(model, theta_star)
        self.assertEqual(float(penalty_head_only.numpy()), 0.0)

        model.get_layer("hidden").trainable = True
        for variable in model.trainable_variables:
            variable.assign_add(tf.ones_like(variable) * 0.01)
        pairs = matched_l2sp_pairs(model, theta_star)
        self.assertEqual(len(pairs), len(model.trainable_variables))
        penalty_full = l2sp_penalty(model, theta_star)
        self.assertGreater(float(penalty_full.numpy()), 0.0)
