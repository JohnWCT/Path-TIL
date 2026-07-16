#!/usr/bin/env python3
"""Unit tests for path_til.losses helpers and Keras builders."""

import unittest

import numpy as np


class LossHelperTests(unittest.TestCase):
    def test_class_frequencies(self):
        from path_til.losses import class_frequencies

        counts, frequencies = class_frequencies([0, 0, 1, 2], num_classes=3)
        np.testing.assert_array_equal(counts, [2, 1, 1])
        np.testing.assert_allclose(frequencies, [0.5, 0.25, 0.25])

    def test_effective_number_weights_positive_mean_one(self):
        from path_til.losses import effective_number_weights

        weights = effective_number_weights([100, 10, 5], beta=0.999)
        self.assertEqual(weights.shape, (3,))
        self.assertTrue(np.all(weights > 0))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_logit_adjustment(self):
        from path_til.losses import logit_adjustment

        offsets = logit_adjustment([0.5, 0.25, 0.25], tau=1.0)
        np.testing.assert_allclose(offsets, np.log([0.5, 0.25, 0.25]))


class KerasLossSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import tensorflow as tf

        cls.tf = tf

    def test_focal_and_logit_adjusted_shapes(self):
        from path_til.losses import build_keras_loss

        tf = self.tf
        y_true = tf.constant([0, 1, 2], dtype=tf.int32)
        y_pred = tf.constant(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.2, 0.6],
            ],
            dtype=tf.float32,
        )
        for name in (
            "sparse_ce",
            "focal_gamma1",
            "focal_gamma2",
            "label_smoothing_ce",
        ):
            loss = build_keras_loss(tf, name, num_classes=3)
            values = loss(y_true, y_pred).numpy()
            self.assertEqual(values.shape, (3,))
            self.assertTrue(np.all(np.isfinite(values)))

        weighted = build_keras_loss(
            tf,
            "weighted_ce",
            num_classes=3,
            class_weights={0: 2.0, 1: 1.0, 2: 1.0},
        )
        self.assertEqual(weighted(y_true, y_pred).numpy().shape, (3,))

        balanced = build_keras_loss(
            tf,
            "class_balanced_focal",
            num_classes=3,
            class_weights={0: 2.0, 1: 1.0, 2: 0.5},
        )
        self.assertEqual(balanced(y_true, y_pred).numpy().shape, (3,))

        adjusted = build_keras_loss(
            tf,
            "logit_adjusted_ce",
            num_classes=3,
            frequencies=np.array([0.1, 0.4, 0.5]),
        )
        self.assertEqual(adjusted(y_true, y_pred).numpy().shape, (3,))


if __name__ == "__main__":
    unittest.main()
