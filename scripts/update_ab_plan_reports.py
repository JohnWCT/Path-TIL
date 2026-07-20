#!/usr/bin/env python3
"""Fill A/B plan markdown reports from orchestrator JSON/CSV outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from path_til.candidate import reference_metrics  # noqa: E402


WORKSPACE = Path("/workspace")
RESULTS = WORKSPACE / "results"


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value) -> str:
    if value is None:
        return "TBD"
    try:
        return "{0:.4f}".format(float(value))
    except (TypeError, ValueError):
        return str(value)


def update_external_report(path: Path) -> None:
    tcga = load_json(RESULTS / "results_tcga_internal_r50_50" / "tcga_internal_metrics.json")
    external_csv = RESULTS / "results_external_testset_r50_50" / "external_summary.csv"
    external = pd.read_csv(external_csv) if external_csv.is_file() else pd.DataFrame()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# HNSCC External Lock-Box Report",
        "",
        "> Auto-updated by `scripts/update_ab_plan_reports.py` at {0}.".format(stamp),
        "> External results are report-only and must not be used for tuning.",
        "",
        "## Reference (HNSCC OOF candidate)",
        "",
        "```text",
        "positive AUC = {0}".format(fmt(reference_metrics()["positive_auc"])),
        "positive PRC = {0}".format(fmt(reference_metrics()["positive_prc"])),
        "```",
        "",
        "## TCGA Internal (`dataset/test`)",
        "",
        "| metric | value |",
        "|---|---:|",
        "| positive AUC | {0} |".format(fmt(tcga.get("positive_auc"))),
        "| positive PRC | {0} |".format(fmt(tcga.get("positive_prc"))),
        "| macro OVR AUC | {0} |".format(fmt(tcga.get("macro_ovr_auc"))),
        "| weighted OVR AUC | {0} |".format(fmt(tcga.get("weighted_ovr_auc"))),
        "",
        "## External Summary",
        "",
        "| dataset | n_patches | positive AUC | positive PRC |",
        "|---|---:|---:|---:|",
    ]
    if external.empty:
        lines.append("| TBD | TBD | TBD | TBD |")
    else:
        for _, row in external.iterrows():
            lines.append(
                "| {0} | {1} | {2} | {3} |".format(
                    row["dataset"],
                    int(row["n_patches"]),
                    fmt(row["positive_auc"]),
                    fmt(row["positive_prc"]),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_stability_report(path: Path) -> None:
    summary_path = RESULTS / "results_candidate_stability_r50_50" / "candidate_stability_summary.json"
    per_seed = RESULTS / "results_candidate_stability_r50_50" / "per_seed_summary.csv"
    summary = load_json(summary_path)
    frame = pd.read_csv(per_seed) if per_seed.is_file() else pd.DataFrame()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# HNSCC Candidate Stability Report",
        "",
        "> Auto-updated at {0}.".format(stamp),
        "",
        "## Aggregate",
        "",
        "```text",
        "mean_positive_auc: {0}".format(fmt(summary.get("mean_positive_auc"))),
        "std_positive_auc: {0}".format(fmt(summary.get("std_positive_auc"))),
        "mean_positive_prc: {0}".format(fmt(summary.get("mean_positive_prc"))),
        "std_positive_prc: {0}".format(fmt(summary.get("std_positive_prc"))),
        "```",
        "",
        "## Per seed",
        "",
        "| experiment | positive AUC | positive PRC | macro OVR AUC | weighted OVR AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    if frame.empty:
        lines.append("| TBD | TBD | TBD | TBD | TBD |")
    else:
        for _, row in frame.iterrows():
            lines.append(
                "| {0} | {1} | {2} | {3} | {4} |".format(
                    row.get("experiment", ""),
                    fmt(row.get("positive_auc")),
                    fmt(row.get("positive_prc")),
                    fmt(row.get("macro_ovr_auc")),
                    fmt(row.get("weighted_ovr_auc")),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", default=str(PROJECT_ROOT / "docs"))
    args = parser.parse_args()
    docs = Path(args.docs_dir)
    update_external_report(docs / "hnscc_external_lockbox_report.md")
    update_stability_report(docs / "hnscc_candidate_stability_report.md")
    print("Updated markdown reports in {0}".format(docs))


if __name__ == "__main__":
    main()
