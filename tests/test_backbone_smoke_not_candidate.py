import unittest

from path_til.backbone_decision import is_forbidden_smoke_promotion


class SmokeNotCandidateTests(unittest.TestCase):
    def test_smoke_phase_cannot_replace_candidate(self):
        phase = "B4"
        decision = "replace_candidate"
        self.assertTrue(is_forbidden_smoke_promotion(phase, decision))
        self.assertFalse(is_forbidden_smoke_promotion("B6", "replace_candidate"))
        self.assertFalse(is_forbidden_smoke_promotion("B4", "positive_specialist_pending_full5"))


if __name__ == "__main__":
    unittest.main()
