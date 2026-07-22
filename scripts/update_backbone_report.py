#!/usr/bin/env python3
"""Update backbone B5 / full5 markdown reports from result artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.candidate import reference_metrics  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Update backbone B5/B6 reports.")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument(
        "--b5-selection",
        default="results/results_backbone_b5_selection/backbone_b5_selection.csv",
    )
    parser.add_argument(
        "--full5-comparison",
        default="results/results_backbone_full5_comparison/backbone_candidate_comparison.csv",
    )
    return parser.parse_args()


def fmt(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "TBD"
    try:
        return "{0:.4f}".format(float(value))
    except (TypeError, ValueError):
        return str(value)


def write_b5_report(path: Path, selection_csv: Path) -> None:
    ref = reference_metrics()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# HNSCC Backbone B5 Hyperparameter Repair Report",
        "",
        "> Auto-updated by `scripts/update_backbone_report.py` at {0}.".format(stamp),
        "",
        "## Reference",
        "",
        "IRV2 + source mix 0.50:0.50:",
        "",
        "- positive AUC = {0}".format(fmt(ref["positive_auc"])),
        "- positive PRC = {0}".format(fmt(ref["positive_prc"])),
        "- macro OVR AUC = {0}".format(fmt(ref["macro_ovr_auc"])),
        "- weighted OVR AUC = {0}".format(fmt(ref["weighted_ovr_auc"])),
        "",
        "## Purpose",
        "",
        "B4 smoke showed EfficientNetV2-S and ConvNeXt-Tiny improved positive AUC / PRC",
        "but decreased macro / weighted OVR AUC. B5 tests whether small hyperparameter",
        "changes can preserve positive gains while repairing multiclass degradation.",
        "",
        "## Experiments",
        "",
        "| backbone | config | positive AUC | positive PRC | macro OVR AUC | weighted OVR AUC | fold gap | decision |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    promote = []
    specialist = []
    drop = []
    selected_path = selection_csv.parent / "selected_for_full5.txt"
    if selected_path.is_file():
        promote = [
            line.strip()
            for line in selected_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if selection_csv.is_file():
        frame = pd.read_csv(selection_csv)
        for _, row in frame.iterrows():
            lines.append(
                "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |".format(
                    row.get("backbone", ""),
                    row.get("experiment_name", ""),
                    fmt(row.get("positive_auc")),
                    fmt(row.get("positive_prc")),
                    fmt(row.get("macro_ovr_auc")),
                    fmt(row.get("weighted_ovr_auc")),
                    fmt(row.get("fold_auc_gap")),
                    row.get("decision", "TBD"),
                )
            )
            decision = row.get("decision")
            name = row.get("experiment_name")
            if decision == "replace_candidate":
                if name not in promote:
                    promote.append(name)
            elif decision == "positive_specialist_pending_full5":
                specialist.append(name)
            else:
                drop.append(name)
    else:
        lines.append("| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |")

    def bullets(items):
        return ["- {0}".format(item) for item in items] if items else ["- none"]

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "### Selected for B6 (≤1 per backbone)",
            "",
        ]
    )
    lines.extend(bullets(promote))
    lines.extend(["", "### positive-specialist (not auto-replace)", ""])
    lines.extend(bullets(specialist))
    lines.extend(["", "### drop", ""])
    lines.extend(bullets(drop))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_full5_report(path: Path, comparison_csv: Path) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# HNSCC Backbone Full 5-fold Report",
        "",
        "> Auto-updated at {0}.".format(stamp),
        "",
        "## Rule",
        "",
        "Smoke results are not candidate-level evidence.",
        "A backbone can replace IRV2 only after full 5-fold OOF and external lock-box confirmation.",
        "",
        "## Full 5-fold Results",
        "",
        "| model | positive AUC | positive PRC | macro OVR AUC | weighted OVR AUC | decision |",
        "|---|---:|---:|---:|---:|---|",
    ]
    replace = []
    specialist = []
    drop = []
    if comparison_csv.is_file():
        frame = pd.read_csv(comparison_csv)
        for _, row in frame.iterrows():
            name = row.get("experiment_name", "")
            decision = row.get("keep_or_drop", "TBD")
            lines.append(
                "| {0} | {1} | {2} | {3} | {4} | {5} |".format(
                    name,
                    fmt(row.get("hnscc_oof_positive_auc")),
                    fmt(row.get("hnscc_oof_positive_prc")),
                    fmt(row.get("hnscc_oof_macro_ovr_auc")),
                    fmt(row.get("hnscc_oof_weighted_ovr_auc")),
                    decision,
                )
            )
            if decision == "keep":
                replace.append(name)
            elif "macro" in str(row.get("notes", "")) or "weighted" in str(row.get("notes", "")):
                specialist.append(name)
            else:
                drop.append(name)
    else:
        lines.append("| TBD | TBD | TBD | TBD | TBD | TBD |")

    def bullets(items):
        return ["- {0}".format(item) for item in items] if items else ["- none yet"]

    lines.extend(["", "## Interpretation", "", "### Replace IRV2", ""])
    lines.extend(bullets(replace))
    lines.extend(["", "### Pareto / positive-specialist", ""])
    lines.extend(bullets(specialist))
    lines.extend(["", "### Drop", ""])
    lines.extend(bullets(drop))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision_log(path: Path) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = """# HNSCC Backbone Decision Log

> Updated: {stamp}

## Locked candidate

- Backbone: InceptionResNetV2
- Source mix: 0.50:0.50
- OOF positive AUC / PRC: 0.8848 / 0.4196

## B4 smoke

- EfficientNetV2-S and ConvNeXt-Tiny: positive-specialist trend, pending B5/B6
- Not promoted: macro / weighted OVR decreased; smoke is not candidate-level evidence

## B5 / B6 / B7

- B5: repair macro/weighted OVR while keeping positive gains
- B6: at most one config per backbone for full 5-fold
- B7: external lock-box only after B6; report-only, never used for tuning
""".format(
        stamp=stamp
    )
    path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    docs = Path(args.docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    write_b5_report(docs / "hnscc_backbone_b5_report.md", Path(args.b5_selection))
    write_full5_report(docs / "hnscc_backbone_full5_report.md", Path(args.full5_comparison))
    write_decision_log(docs / "hnscc_backbone_decision_log.md")
    print("Updated backbone reports in", docs)


if __name__ == "__main__":
    main()
