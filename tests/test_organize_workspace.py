#!/usr/bin/env python3
"""Unit tests for workspace organization helpers."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from path_til.paths import (
    PRETRAINED_MODEL_NAME,
    baseline_path,
    ensure_layout_dirs,
    pretrained_model_path,
    result_path,
)


def _load_organize_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "organize_workspace.py"
    spec = importlib.util.spec_from_file_location("organize_workspace", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ORG = _load_organize_module()


class PathsHelpersTests(unittest.TestCase):
    def test_result_and_baseline_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = ensure_layout_dirs(root)
            self.assertTrue(dirs["results"].is_dir())
            self.assertTrue(dirs["baselines"].is_dir())
            self.assertTrue(dirs["tests_legacy"].is_dir())

            weight = dirs["baselines"] / PRETRAINED_MODEL_NAME
            weight.write_bytes(b"fake-h5")
            found = pretrained_model_path(root)
            self.assertEqual(found, weight)

            self.assertEqual(
                ORG.destination_for(root, "results_demo", "results"),
                root / "results" / "results_demo",
            )
            self.assertEqual(
                ORG.destination_for(root, "qupath_results_heavy", "baselines"),
                root / "baselines" / "qupath_results_heavy",
            )

    def test_classify_entry(self):
        self.assertIsNone(ORG.classify_entry("results"))
        self.assertEqual(ORG.classify_entry("results_foo"), "results")
        self.assertEqual(ORG.classify_entry("inference_results"), "results")
        self.assertEqual(ORG.classify_entry("qupath_results_heavy"), "baselines")
        self.assertEqual(ORG.classify_entry(PRETRAINED_MODEL_NAME), "baselines")
        self.assertEqual(
            ORG.classify_entry("InceptionResNetV2_testing.py"), "tests_legacy"
        )
        self.assertIsNone(ORG.classify_entry("scripts"))


class OrganizeWorkspaceTests(unittest.TestCase):
    def _seed_root(self, root):
        (root / "results").mkdir()
        (root / "results" / "keep_me.txt").write_text("stay", encoding="utf-8")
        (root / "results_exp_a").mkdir()
        (root / "results_exp_a" / "metrics.json").write_text("{}", encoding="utf-8")
        (root / "inference_results").mkdir()
        (root / "inference_results" / "out.csv").write_text("a\n", encoding="utf-8")
        (root / "qupath_results_heavy").mkdir()
        (root / "qupath_results_heavy" / "fold01_stage2_best.h5").write_bytes(b"w")
        (root / PRETRAINED_MODEL_NAME).write_bytes(b"pretrained")
        (root / "InceptionResNetV2_testing.py").write_text("# test\n", encoding="utf-8")
        (root / "scripts").mkdir()
        (root / "path_til").mkdir()
        (root / "dockerfile").write_text("FROM scratch\n", encoding="utf-8")

    def test_dry_run_does_not_change_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed_root(root)
            before = sorted(p.name for p in root.iterdir())
            actions = ORG.plan_moves(root)
            self.assertGreaterEqual(len(actions), 4)
            self.assertTrue(all(a["status"] == "move" for a in actions))
            after = sorted(p.name for p in root.iterdir())
            self.assertEqual(before, after)
            self.assertTrue((root / "results_exp_a").is_dir())
            self.assertFalse((root / "results" / "results_exp_a").exists())

    def test_apply_moves_without_root_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed_root(root)
            actions = ORG.plan_moves(root)
            payload = ORG.apply_moves(root, actions, create_symlinks=False)
            self.assertEqual(payload["error_count"], 0)
            self.assertGreaterEqual(payload["moved_count"], 4)

            dest = root / "results" / "results_exp_a"
            self.assertTrue(dest.is_dir())
            self.assertTrue((dest / "metrics.json").is_file())
            self.assertFalse((root / "results_exp_a").exists())
            self.assertFalse((root / "results_exp_a").is_symlink())

            base = root / "baselines" / PRETRAINED_MODEL_NAME
            self.assertTrue(base.is_file())
            self.assertFalse((root / PRETRAINED_MODEL_NAME).exists())

            legacy = root / "tests" / "legacy" / "InceptionResNetV2_testing.py"
            self.assertTrue(legacy.is_file())
            self.assertFalse((root / "InceptionResNetV2_testing.py").exists())

            self.assertEqual(
                (root / "results" / "keep_me.txt").read_text(encoding="utf-8"),
                "stay",
            )

            manifest = Path(payload["manifest_path"])
            self.assertTrue(manifest.is_file())
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(data["moved_count"], payload["moved_count"])
            self.assertFalse(data["create_symlinks"])

            again = ORG.plan_moves(root)
            self.assertEqual(again, [])

    def test_optional_symlink_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed_root(root)
            actions = ORG.plan_moves(root)
            payload = ORG.apply_moves(root, actions, create_symlinks=True)
            self.assertEqual(payload["error_count"], 0)
            link = root / "results_exp_a"
            self.assertTrue(link.is_symlink())
            self.assertEqual(
                link.resolve(), (root / "results" / "results_exp_a").resolve()
            )

    def test_conflict_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed_root(root)
            conflict_dir = root / "results" / "results_exp_a"
            conflict_dir.mkdir(parents=True)
            (conflict_dir / "existing.txt").write_text("nope", encoding="utf-8")

            actions = ORG.plan_moves(root)
            conflict = [a for a in actions if a["name"] == "results_exp_a"]
            self.assertEqual(len(conflict), 1)
            self.assertEqual(conflict[0]["status"], "conflict")

            payload = ORG.apply_moves(root, actions, create_symlinks=False)
            self.assertGreaterEqual(payload["error_count"], 1)
            self.assertTrue((root / "results_exp_a" / "metrics.json").is_file())
            self.assertEqual(
                (conflict_dir / "existing.txt").read_text(encoding="utf-8"),
                "nope",
            )
            self.assertFalse((conflict_dir / "metrics.json").exists())


class ModulePathExportsTests(unittest.TestCase):
    def test_module_level_helpers_are_paths(self):
        self.assertIsInstance(result_path("results_demo"), Path)
        self.assertIsInstance(baseline_path("qupath_results_heavy"), Path)
        self.assertTrue(str(result_path("results_demo")).endswith("results_demo"))


if __name__ == "__main__":
    unittest.main()
