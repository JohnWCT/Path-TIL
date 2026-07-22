import unittest

from path_til.backbone_decision import BackboneMetrics, decide_backbone_status


class BackboneDecisionTests(unittest.TestCase):
    def setUp(self):
        self.ref = BackboneMetrics(
            name="irv2",
            positive_auc=0.8848,
            positive_prc=0.4196,
            macro_ovr_auc=0.9173,
            weighted_ovr_auc=0.9288,
        )

    def test_backbone_replace_requires_macro_and_weighted_guardrails(self):
        cand = BackboneMetrics(
            name="efficientnetv2_s",
            positive_auc=0.8983,
            positive_prc=0.5448,
            macro_ovr_auc=0.9002,
            weighted_ovr_auc=0.9054,
        )
        decision = decide_backbone_status(cand, self.ref)
        self.assertEqual(decision.decision, "positive_specialist_pending_full5")
        self.assertIn("macro_ovr_auc_decreased", decision.reasons)
        self.assertIn("weighted_ovr_auc_clearly_decreased", decision.reasons)

    def test_backbone_can_replace_when_all_guardrails_pass(self):
        cand = BackboneMetrics(
            name="new_backbone",
            positive_auc=0.8950,
            positive_prc=0.4300,
            macro_ovr_auc=0.9180,
            weighted_ovr_auc=0.9290,
            fold_auc_gap=0.01,
            fold_prc_gap=0.02,
        )
        decision = decide_backbone_status(cand, self.ref)
        self.assertEqual(decision.decision, "replace_candidate")

    def test_drop_when_positive_not_improved(self):
        cand = BackboneMetrics(
            name="weak",
            positive_auc=0.80,
            positive_prc=0.30,
            macro_ovr_auc=0.92,
            weighted_ovr_auc=0.93,
        )
        decision = decide_backbone_status(cand, self.ref)
        self.assertEqual(decision.decision, "drop")
        self.assertIn("positive_auc_or_prc_not_improved", decision.reasons)


if __name__ == "__main__":
    unittest.main()
