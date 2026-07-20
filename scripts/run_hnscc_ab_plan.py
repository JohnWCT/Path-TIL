#!/usr/bin/env python3
"""Orchestrate HNSCC Plan A (robustness) and Plan B (backbone replacement) in Docker."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.gpu_profile import detect_gpu_profile, write_profile_snapshot  # noqa: E402


WORKSPACE = Path("/workspace")
RESULTS = WORKSPACE / "results"
BASELINES = WORKSPACE / "baselines"
LOG_ROOT = RESULTS / "results_ab_plan_orchestrator"

CANDIDATE_MODEL = RESULTS / "results_method_source_mix_tcga_r50_50"
CANDIDATE_OOF = RESULTS / "results_oof_with_prc" / "source_mix_tcga_r50_50"
PRETRAINED_IRV2 = BASELINES / "best_InceptionResNetV2_model.h5"

PHASE_ORDER = ("B1", "A1", "A2", "A3", "A4", "B2", "B3", "B4", "SUMMARY")


def parse_args():
    parser = argparse.ArgumentParser(description="Run HNSCC Plan A + Plan B with GPU-tuned settings.")
    parser.add_argument(
        "--phase",
        choices=list(PHASE_ORDER) + ["ALL"],
        default="ALL",
        help="Run one phase or ALL (default)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip steps whose primary output marker already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    return parser.parse_args()


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = "[{0}] {1}".format(stamp, msg)
    print(line, flush=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with (LOG_ROOT / "orchestrator.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    log("CMD: {0}".format(" ".join(cmd)))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def marker_exists(path: Path) -> bool:
    return path.is_file() or path.is_dir()


def phase_b1(profile: dict, dry_run: bool, skip_existing: bool) -> None:
    """Build TCGA train/test manifests (B-line prerequisite)."""
    train_csv = WORKSPACE / "tcga_train_dataset.csv"
    test_csv = WORKSPACE / "tcga_test_dataset.csv"
    if skip_existing and train_csv.is_file() and test_csv.is_file():
        log("B1 skip: manifests exist")
        return
    run_cmd(
        [
            "python3",
            "scripts/prepare_labeled_patch_csv.py",
            "--root",
            "dataset/train",
            "--output",
            str(train_csv),
            "--path-prefix",
            "/workspace",
        ],
        dry_run,
    )
    run_cmd(
        [
            "python3",
            "scripts/prepare_labeled_patch_csv.py",
            "--root",
            "dataset/test",
            "--output",
            str(test_csv),
            "--path-prefix",
            "/workspace",
        ],
        dry_run,
    )


def phase_a1(profile: dict, dry_run: bool, skip_existing: bool) -> None:
    """TCGA internal holdout eval – source-domain retention check."""
    out = RESULTS / "results_tcga_internal_r50_50"
    marker = out / "tcga_internal_metrics.json"
    if skip_existing and marker.is_file():
        log("A1 skip: {0}".format(marker))
        return
    run_cmd(
        [
            "python3",
            "scripts/eval_tcga_internal.py",
            "--model-dir",
            str(CANDIDATE_MODEL),
            "--test-root",
            "dataset/test",
            "--output-dir",
            str(out),
            "--stage",
            "selected",
            "--batch-size",
            str(profile["batch_size_eval"]),
            "--image-workers",
            str(profile["image_workers"]),
        ],
        dry_run,
    )


def phase_a2(profile: dict, dry_run: bool, skip_existing: bool) -> None:
    """External lock-box eval – report only, no tuning."""
    out = RESULTS / "results_external_testset_r50_50"
    marker = out / "external_summary.csv"
    if skip_existing and marker.is_file():
        log("A2 skip: {0}".format(marker))
        return
    run_cmd(
        [
            "python3",
            "scripts/eval_external_testset.py",
            "--model-dir",
            str(CANDIDATE_MODEL),
            "--testset-root",
            "dataset/Testset",
            "--output-dir",
            str(out),
            "--stage",
            "selected",
            "--batch-size",
            str(profile["batch_size_eval"]),
            "--image-workers",
            str(profile["image_workers"]),
        ],
        dry_run,
    )
    if not dry_run:
        run_cmd(["python3", "scripts/update_ab_plan_reports.py"], dry_run=False)


def train_source_mix(
    config: str,
    output_dir: Path,
    seed: int | None,
    profile: dict,
    dry_run: bool,
    skip_existing: bool,
) -> None:
    marker = output_dir / "fold04" / "fold_metrics.json"
    if skip_existing and marker.is_file():
        log("skip training (complete): {0}".format(output_dir))
        return
    cmd = [
        "python3",
        "scripts/train_hnscc_source_mix.py",
        "--config",
        config,
        "--csv-hnscc",
        "qupath_dataset.csv",
        "--csv-tcga",
        "tcga_train_dataset.csv",
        "--fold-csv",
        "folds_hnscc_group5.csv",
        "--pretrained",
        str(PRETRAINED_IRV2),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(profile["batch_size_train_irv2"]),
        "--image-workers",
        str(profile["image_workers"]),
        "--fit-workers",
        str(profile["fit_workers"]),
        "--use-multiprocessing",
        "on" if profile["use_multiprocessing"] else "off",
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    run_cmd(cmd, dry_run)


def train_l2sp(config: str, output_dir: Path, profile: dict, dry_run: bool, skip_existing: bool) -> None:
    marker = output_dir / "fold04" / "fold_metrics.json"
    if skip_existing and marker.is_file():
        log("skip L2-SP (complete): {0}".format(output_dir))
        return
    run_cmd(
        [
            "python3",
            "scripts/train_hnscc_l2sp.py",
            "--config",
            config,
            "--csv-hnscc",
            "qupath_dataset.csv",
            "--csv-tcga",
            "tcga_train_dataset.csv",
            "--fold-csv",
            "folds_hnscc_group5.csv",
            "--pretrained",
            str(PRETRAINED_IRV2),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            str(profile["batch_size_train_irv2"]),
            "--image-workers",
            str(profile["image_workers"]),
            "--fit-workers",
            str(profile["fit_workers"]),
            "--use-multiprocessing",
            "on" if profile["use_multiprocessing"] else "off",
        ],
        dry_run,
    )


def eval_oof(pred_dir: Path, oof_name: str, dry_run: bool, skip_existing: bool) -> None:
    out = RESULTS / "results_oof_with_prc" / oof_name
    marker = out / "oof_predictions.csv"
    if skip_existing and marker.is_file():
        log("skip OOF (exists): {0}".format(out))
        return
    run_cmd(
        [
            "python3",
            "scripts/eval_hnscc_oof.py",
            "--pred-dir",
            str(pred_dir),
            "--csv",
            "qupath_dataset.csv",
            "--fold-csv",
            "folds_hnscc_group5.csv",
            "--stage",
            "selected",
            "--output",
            str(out),
        ],
        dry_run,
    )


def phase_a3(profile: dict, dry_run: bool, skip_existing: bool) -> None:
    """Seed stability: 7 and 21 (42 OOF already exists)."""
    seeds = (
        ("seed7", "configs/method_source_mix_tcga_r50_50_seed7.yaml", 7),
        ("seed21", "configs/method_source_mix_tcga_r50_50_seed21.yaml", 21),
    )
    for name, config, seed in seeds:
        out = RESULTS / "results_method_source_mix_tcga_r50_50_{0}".format(name)
        train_source_mix(config, out, seed, profile, dry_run, skip_existing)
        eval_oof(out, "source_mix_tcga_r50_50_{0}".format(name), dry_run, skip_existing)


def phase_a4(profile: dict, dry_run: bool, skip_existing: bool) -> None:
    """L2-SP ablation at three lambda values."""
    configs = (
        ("1e-5", "configs/method_l2sp_r50_50_lambda_1e-5.yaml"),
        ("1e-4", "configs/method_l2sp_r50_50_lambda_1e-4.yaml"),
        ("1e-3", "configs/method_l2sp_r50_50_lambda_1e-3.yaml"),
    )
    for tag, config in configs:
        out = RESULTS / "results_method_l2sp_r50_50_lambda_{0}".format(tag)
        train_l2sp(config, out, profile, dry_run, skip_existing)
        eval_oof(out, "l2sp_r50_50_lambda_{0}".format(tag), dry_run, skip_existing)


def phase_b2(profile: dict, dry_run: bool, skip_existing: bool) -> None:
    out = BASELINES / "source_pretrain_efficientnetv2_s"
    marker = out / "source_val_metrics.json"
    if skip_existing and marker.is_file():
        log("B2 skip: {0}".format(marker))
        return
    run_cmd(
        [
            "python3",
            "scripts/pretrain_source_backbone.py",
            "--config",
            "configs/source_pretrain_efficientnetv2_s.yaml",
            "--train-csv",
            "tcga_train_dataset.csv",
            "--val-csv",
            "tcga_test_dataset.csv",
            "--output-dir",
            str(out),
            "--batch-size",
            str(profile["batch_size_pretrain"]),
            "--image-workers",
            str(profile["image_workers"]),
        ],
        dry_run,
    )


def phase_b3(profile: dict, dry_run: bool, skip_existing: bool) -> None:
    out = BASELINES / "source_pretrain_convnext_tiny"
    marker = out / "source_val_metrics.json"
    if skip_existing and marker.is_file():
        log("B3 skip: {0}".format(marker))
        return
    run_cmd(
        [
            "python3",
            "scripts/pretrain_source_backbone.py",
            "--config",
            "configs/source_pretrain_convnext_tiny.yaml",
            "--train-csv",
            "tcga_train_dataset.csv",
            "--val-csv",
            "tcga_test_dataset.csv",
            "--output-dir",
            str(out),
            "--batch-size",
            str(profile["batch_size_pretrain"]),
            "--image-workers",
            str(profile["image_workers"]),
        ],
        dry_run,
    )


def phase_b4(profile: dict, dry_run: bool, skip_existing: bool) -> None:
    """Fold 0+1 smoke for both backbones."""
    smokes = (
        (
            "efficientnetv2_s",
            BASELINES / "source_pretrain_efficientnetv2_s" / "source_pretrained_efficientnetv2_s_best.h5",
            RESULTS / "results_backbone_smoke_efficientnetv2_s",
        ),
        (
            "convnext_tiny",
            BASELINES / "source_pretrain_convnext_tiny" / "source_pretrained_convnext_tiny_best.h5",
            RESULTS / "results_backbone_smoke_convnext_tiny",
        ),
    )
    for backbone, pretrained, out in smokes:
        marker = out / "fold01" / "fold_metrics.json"
        if skip_existing and marker.is_file():
            log("B4 skip: {0}".format(out))
            continue
        if not dry_run and not pretrained.is_file():
            log("B4 waiting for pretrained weights: {0}".format(pretrained))
            continue
        run_cmd(
            [
                "python3",
                "scripts/train_hnscc_backbone_source_mix.py",
                "--backbone",
                backbone,
                "--csv-hnscc",
                "qupath_dataset.csv",
                "--csv-tcga",
                "tcga_train_dataset.csv",
                "--fold-csv",
                "folds_hnscc_group5.csv",
                "--pretrained",
                str(pretrained),
                "--output-dir",
                str(out),
                "--folds",
                "0",
                "1",
                "--source-mix-ratio",
                "0.50",
                "--aug",
                "heavy",
                "--hne-norm",
                "off",
                "--class-weight",
                "on",
            ],
            dry_run,
        )


def phase_summary(profile: dict, dry_run: bool, skip_existing: bool) -> None:
    out = RESULTS / "results_candidate_stability_r50_50"
    experiments = [
        CANDIDATE_OOF,
        RESULTS / "results_oof_with_prc" / "source_mix_tcga_r50_50_seed7",
        RESULTS / "results_oof_with_prc" / "source_mix_tcga_r50_50_seed21",
    ]
    existing = [str(path) for path in experiments if path.is_dir()]
    if len(existing) < 1:
        log("SUMMARY skip: no OOF directories yet")
        return
    cmd = [
        "python3",
        "scripts/summarize_candidate_stability.py",
        "--experiments",
        *existing,
        "--output",
        str(out),
    ]
    run_cmd(cmd, dry_run)

    compare_out = RESULTS / "results_backbone_candidate_comparison"
    smoke_oofs = [
        RESULTS / "results_oof_with_prc" / "backbone_efficientnetv2_s_smoke",
        RESULTS / "results_oof_with_prc" / "backbone_convnext_tiny_smoke",
    ]
    ext_dirs = [
        RESULTS / "results_external_testset_r50_50",
    ]
    if CANDIDATE_OOF.is_dir():
        run_cmd(
            [
                "python3",
                "scripts/compare_backbone_and_candidate.py",
                "--reference",
                str(CANDIDATE_OOF),
                "--experiments",
                *[str(path) for path in smoke_oofs if path.is_dir()],
                "--external-results",
                *[str(path) for path in ext_dirs if path.is_dir()],
                "--output",
                str(compare_out),
            ],
            dry_run,
        )
    run_cmd(["python3", "scripts/update_ab_plan_reports.py"], dry_run)


def run_phase(name: str, profile: dict, dry_run: bool, skip_existing: bool) -> None:
    log("=== Phase {0} start ===".format(name))
    started = time.time()
    dispatch = {
        "B1": phase_b1,
        "A1": phase_a1,
        "A2": phase_a2,
        "A3": phase_a3,
        "A4": phase_a4,
        "B2": phase_b2,
        "B3": phase_b3,
        "B4": phase_b4,
        "SUMMARY": phase_summary,
    }
    dispatch[name](profile, dry_run, skip_existing)
    log("=== Phase {0} done ({1:.1f}s) ===".format(name, time.time() - started))


def main():
    args = parse_args()
    profile = detect_gpu_profile()
    write_profile_snapshot(LOG_ROOT / "gpu_profile.json", profile)
    log("GPU profile: {0}".format(json.dumps(profile, sort_keys=True)))

    phases = list(PHASE_ORDER) if args.phase == "ALL" else [args.phase]
    for phase in phases:
        run_phase(phase, profile, args.dry_run, args.skip_existing)
    log("Orchestrator finished.")


if __name__ == "__main__":
    main()
