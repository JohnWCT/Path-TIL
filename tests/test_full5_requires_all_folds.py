import unittest

from path_til.backbone_metrics import has_all_full5_folds, required_full5_folds


class Full5RequiresAllFoldsTests(unittest.TestCase):
    def test_full5_requires_folds_0_to_4(self):
        folds = {0, 1, 2, 3, 4}
        observed = {0, 1, 2, 3, 4}
        self.assertEqual(observed, folds)
        self.assertEqual(required_full5_folds(), folds)
        self.assertTrue(has_all_full5_folds(observed))
        self.assertFalse(has_all_full5_folds({0, 1}))


if __name__ == "__main__":
    unittest.main()
