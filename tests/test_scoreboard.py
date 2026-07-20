#!/usr/bin/env python3
"""Unit tests for path_til.scoreboard."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from path_til.experiment_registry import CANDIDATE_REFERENCE
from path_til.scoreboard import (
    apply_candidate_comparison,
    backfill_positive_prc,
    build_scoreboard,
    load_oof_metrics,
    render_scoreboard_markdown,
    write_scoreboard,
)


def _write_eval_summary(directory, positive_auc, positive_prc=None, hard_mae=0.2):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    patch = {
        "positive_vs_rest_auc_positive_binary": {
            "status": "ok",
            "value": positive_auc,
        },
        "ovr_auc_macro": {"status": "ok", "value": positive_auc + 0.02},
        "ovr_auc_weighted": {"status": "ok", "value": positive_auc + 0.03},
        "accuracy": {"status": "ok", "value": 0.75},
        "f1_macro": {"status": "ok", "value": 0.65},
    }
    if positive_prc is not None:
        patch["positive_vs_rest_average_precision_positive_binary"] = {
            "status": "ok",
            "value": positive_prc,
        }
    payload = {
        "patch_metrics": patch,
        "slide_metrics": {"mae": hard_mae, "soft_mae": hard_mae + 0.01},
        "n_patches": 100,
        "n_cases": 10,
    }
    with (directory / "eval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class ScoreboardTests(unittest.TestCase):
    def test_backfill_positive_prc_from_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oof = root / "oof"
            oof.mkdir()
            _write_eval_summary(oof, positive_auc=0.8, positive_prc=None)
            frame = pd.DataFrame(
                {
                    "y_true_idx": [0, 1, 2, 0, 1],
                    "prob_positive": [0.9, 0.1, 0.2, 0.8, 0.05],
                    "prob_negative": [0.05, 0.8, 0.3, 0.1, 0.9],
                    "prob_other": [0.05, 0.1, 0.5, 0.1, 0.05],
                }
            )
            frame.to_csv(oof / "oof_predictions.csv", index=False)
            value = backfill_positive_prc(oof)
            self.assertIsNotNone(value)
            self.assertGreater(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_apply_candidate_comparison_marks_current_candidate(self):
        row = {
            "experiment_id": CANDIDATE_REFERENCE["name"],
            "positive_auc": CANDIDATE_REFERENCE["positive_auc"],
            "positive_prc": CANDIDATE_REFERENCE["positive_prc"],
            "macro_ovr_auc": CANDIDATE_REFERENCE["macro_ovr_auc"],
            "weighted_ovr_auc": CANDIDATE_REFERENCE["weighted_ovr_auc"],
        }
        out = apply_candidate_comparison(row)
        self.assertEqual(out["vs_candidate"], "current_candidate")
        self.assertEqual(out["delta_auc"], 0.0)

    def test_build_scoreboard_chapters_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            registry = root / "registry.yaml"
            registry.write_text(
                """
candidate_id: demo_candidate
chapters:
  - id: leaderboard
    title: Leaderboard
    include_all: true
  - id: demo_chapter
    title: Demo chapter
    experiments:
      - id: demo_candidate
        display_name: Demo candidate
        oof_path: exp_a
      - id: demo_worse
        display_name: Demo worse
        oof_path: exp_b
""".strip(),
                encoding="utf-8",
            )
            _write_eval_summary(
                results / "exp_a",
                positive_auc=0.9,
                positive_prc=0.5,
            )
            _write_eval_summary(
                results / "exp_b",
                positive_auc=0.7,
                positive_prc=0.3,
            )
            payload = build_scoreboard(
                registry,
                results,
                candidate={
                    "name": "demo_candidate",
                    "positive_auc": 0.9,
                    "positive_prc": 0.5,
                    "macro_ovr_auc": 0.92,
                    "weighted_ovr_auc": 0.93,
                    "hard_til_mae": 0.2,
                },
            )
            self.assertEqual(len(payload["chapters"]), 1)
            self.assertEqual(len(payload["leaderboard"]), 2)
            md = render_scoreboard_markdown(payload)
            self.assertIn("HNSCC Living Scoreboard", md)
            self.assertIn("Demo chapter", md)
            self.assertIn("current_candidate", md)

    def test_write_scoreboard_outputs_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            registry = root / "registry.yaml"
            registry.write_text(
                """
candidate_id: only_one
chapters:
  - id: only
    title: Only
    experiments:
      - id: only_one
        display_name: Only one
        oof_path: only
""".strip(),
                encoding="utf-8",
            )
            _write_eval_summary(results / "only", 0.85, 0.4)
            out_md = root / "scoreboard.md"
            out_csv = root / "scoreboard.csv"
            write_scoreboard(
                registry,
                results,
                out_md,
                output_csv=out_csv,
                candidate={
                    "name": "only_one",
                    "positive_auc": 0.85,
                    "positive_prc": 0.4,
                    "macro_ovr_auc": 0.87,
                    "weighted_ovr_auc": 0.88,
                    "hard_til_mae": 0.15,
                },
            )
            self.assertTrue(out_md.is_file())
            self.assertTrue(out_csv.is_file())
            metrics = load_oof_metrics(results / "only")
            self.assertAlmostEqual(metrics["positive_auc"], 0.85)


if __name__ == "__main__":
    unittest.main()
