#!/usr/bin/env python3
"""Summarize why backbone smoke / B5 runs fail multiclass guardrails."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.candidate import reference_metrics  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize backbone failure modes.")
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_metrics(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    for name in ("metrics.json", "smoke_summary.json", "oof_metrics.json"):
        for candidate in (path / name, path / "smoke_summary" / name):
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(path)


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ref = reference_metrics()
    rows = []
    for experiment in args.experiments:
        payload = load_metrics(Path(experiment))
        name = payload.get("experiment") or Path(experiment).name
        modes = []
        if float(payload.get("macro_ovr_auc") or 0) < float(ref["macro_ovr_auc"]):
            modes.append("macro_ovr_drop")
        if float(payload.get("weighted_ovr_auc") or 0) < float(ref["weighted_ovr_auc"]) - 0.01:
            modes.append("weighted_ovr_drop")
        if (payload.get("fold_auc_gap") or 0) > 0.05:
            modes.append("fold_auc_unstable")
        if (payload.get("fold_prc_gap") or 0) > 0.08:
            modes.append("fold_prc_unstable")
        if float(payload.get("negative_auc") or 1) < 0.90:
            modes.append("negative_auc_weak")
        if float(payload.get("other_auc") or 1) < 0.90:
            modes.append("other_auc_weak")
        if float(payload.get("positive_auc") or 0) > float(ref["positive_auc"]) and modes:
            modes.append("positive_specialist_tradeoff")
        rows.append(
            {
                "experiment": name,
                "positive_auc": payload.get("positive_auc"),
                "positive_prc": payload.get("positive_prc"),
                "macro_ovr_auc": payload.get("macro_ovr_auc"),
                "weighted_ovr_auc": payload.get("weighted_ovr_auc"),
                "negative_auc": payload.get("negative_auc"),
                "other_auc": payload.get("other_auc"),
                "fold_auc_gap": payload.get("fold_auc_gap"),
                "failure_modes": ";".join(modes) if modes else "none",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "backbone_failure_modes.csv", index=False)
    lines = ["# Backbone Failure Modes", ""]
    for row in rows:
        lines.append("## {0}".format(row["experiment"]))
        lines.append("")
        lines.append("- failure_modes: `{0}`".format(row["failure_modes"]))
        lines.append(
            "- positive AUC/PRC = {0:.4f} / {1:.4f}".format(
                float(row["positive_auc"] or 0), float(row["positive_prc"] or 0)
            )
        )
        lines.append(
            "- macro / weighted OVR = {0:.4f} / {1:.4f}".format(
                float(row["macro_ovr_auc"] or 0), float(row["weighted_ovr_auc"] or 0)
            )
        )
        lines.append("")
    (output / "backbone_failure_modes.md").write_text("\n".join(lines), encoding="utf-8")
    print("Failure-mode summary ->", output)


if __name__ == "__main__":
    main()
