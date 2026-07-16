#!/usr/bin/env python3
"""Unit tests for path_til.samplers."""

import unittest

import numpy as np

from path_til.samplers import sample_indices


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        self.cases = np.array(
            ["a", "a", "b", "b", "b", "c", "c", "c", "c", "c"]
        )

    def test_random_sampler_deterministic(self):
        first = sample_indices(
            self.labels, self.cases, "random", size=8, seed=7
        )
        second = sample_indices(
            self.labels, self.cases, "random", size=8, seed=7
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 8)

    def test_class_balanced_covers_all_classes(self):
        selected = sample_indices(
            self.labels, self.cases, "class_balanced", size=12, seed=1
        )
        seen = set(self.labels[selected])
        self.assertEqual(seen, {0, 1, 2})

    def test_case_balanced_covers_all_cases(self):
        selected = sample_indices(
            self.labels, self.cases, "case_balanced", size=12, seed=2
        )
        seen = set(self.cases[selected])
        self.assertEqual(seen, {"a", "b", "c"})

    def test_hybrid_length(self):
        selected = sample_indices(
            self.labels,
            self.cases,
            "hybrid_class_case",
            size=10,
            seed=3,
            class_ratio=0.4,
        )
        self.assertEqual(len(selected), 10)


if __name__ == "__main__":
    unittest.main()
