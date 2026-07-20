#!/usr/bin/env python3
"""Mix TCGA pretrained-train patches into HNSCC GroupCV train splits only."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_base_module():
    path = PROJECT_ROOT / "scripts" / "train_hnscc_groupcv_irv2.py"
    spec = importlib.util.spec_from_file_location(
        "train_hnscc_groupcv_irv2", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_base_module()

from path_til.experiment_registry import load_method_config  # noqa: E402
from path_til.hnscc import LABELS  # noqa: E402


LABEL_TO_IDX = {label: index for index, label in enumerate(LABELS)}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune HNSCC GroupCV while mixing TCGA patches into each fold "
            "train split only. Held-out HNSCC val/test stay pure HNSCC."
        )
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "method_source_mix_tcga.yaml"),
    )
    parser.add_argument("--csv-hnscc", default="qupath_dataset.csv")
    parser.add_argument("--csv-tcga", default="tcga_train_dataset.csv")
    parser.add_argument("--fold-csv", default="folds_hnscc_group5.csv")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hnscc-ratio", type=float, default=None)
    parser.add_argument("--tcga-ratio", type=float, default=None)
    parser.add_argument("--fold", action="append", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs-stage1", type=int, default=None)
    parser.add_argument("--epochs-stage2", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-workers", type=int, default=None)
    parser.add_argument("--fit-workers", type=int, default=None)
    parser.add_argument(
        "--use-multiprocessing",
        choices=("on", "off"),
        default=None,
    )
    return parser.parse_args()


def validate_ratio(hnscc_ratio, tcga_ratio):
    if hnscc_ratio < 0 or tcga_ratio < 0:
        raise ValueError("ratios must be non-negative")
    total = hnscc_ratio + tcga_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            "hnscc_ratio + tcga_ratio must equal 1.0; got {0}".format(total)
        )


def load_tcga_csv(path):
    frame = pd.read_csv(path)
    required = {"image_path", "label", "case_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("{0} missing columns: {1}".format(path, missing))
    invalid = sorted(set(frame["label"].astype(str)) - set(LABELS))
    if invalid:
        raise ValueError("{0} has invalid labels: {1}".format(path, invalid))
    frame = frame.copy()
    frame["case_id"] = frame["case_id"].astype(str)
    frame["label"] = frame["label"].astype(str)
    frame["image_path"] = frame["image_path"].astype(str)
    frame["source"] = "tcga"
    return frame


def stratified_sample(frame, n_target, seed):
    if n_target <= 0:
        return frame.iloc[0:0].copy()
    if n_target >= len(frame):
        return frame.sample(n=n_target, replace=True, random_state=seed).reset_index(
            drop=True
        )
    rng = np.random.RandomState(seed)
    counts = frame["label"].value_counts()
    labels = list(counts.index)
    raw = {
        label: n_target * float(counts[label]) / float(len(frame)) for label in labels
    }
    alloc = {label: int(np.floor(raw[label])) for label in labels}
    remainder = n_target - sum(alloc.values())
    order = sorted(labels, key=lambda label: raw[label] - alloc[label], reverse=True)
    for label in order[:remainder]:
        alloc[label] += 1
    parts = []
    for label in labels:
        take = alloc[label]
        if take <= 0:
            continue
        subset = frame[frame["label"] == label]
        replace = take > len(subset)
        chosen = subset.sample(
            n=take, replace=replace, random_state=int(rng.randint(1e9))
        )
        parts.append(chosen)
    mixed = pd.concat(parts, ignore_index=True)
    return mixed.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_mixed_frame(hnscc, tcga, hnscc_ratio, tcga_ratio, seed):
    # Size the TCGA pool from the full HNSCC set so every fold can reuse it.
    n_tcga = int(round(len(hnscc) * tcga_ratio / hnscc_ratio)) if hnscc_ratio else 0
    sampled = stratified_sample(tcga, n_tcga, seed)
    hnscc = hnscc.copy()
    hnscc["source"] = "hnscc"
    mixed = pd.concat([hnscc, sampled], ignore_index=True)
    return mixed, sampled


def split_indices_with_source(frame, assignments, fold):
    fold_rows = assignments[assignments["fold"] == fold]
    role_by_case = dict(zip(fold_rows["case_id"].astype(str), fold_rows["role"]))
    roles = []
    for _, row in frame.iterrows():
        if row.get("source") == "tcga":
            roles.append("train")
            continue
        role = role_by_case.get(str(row["case_id"]))
        if role is None:
            raise ValueError(
                "Fold {0} has unmapped HNSCC case {1}".format(fold, row["case_id"])
            )
        roles.append(role)
    roles = pd.Series(roles, index=frame.index)
    return {
        role: np.flatnonzero(roles.to_numpy() == role)
        for role in ("train", "val", "test")
    }


def build_namespace(cli, method):
    hnscc_ratio = (
        cli.hnscc_ratio
        if cli.hnscc_ratio is not None
        else float(method.get("hnscc_ratio", 0.75))
    )
    tcga_ratio = (
        cli.tcga_ratio
        if cli.tcga_ratio is not None
        else float(method.get("tcga_ratio", 0.25))
    )
    validate_ratio(hnscc_ratio, tcga_ratio)
    cpu = max(1, os.cpu_count() or 1)
    image_workers = (
        cli.image_workers
        if cli.image_workers is not None
        else min(14, max(4, cpu - 2))
    )
    fit_workers = (
        cli.fit_workers if cli.fit_workers is not None else max(2, min(6, cpu // 3))
    )
    use_multiprocessing = (
        cli.use_multiprocessing
        if cli.use_multiprocessing is not None
        else ("on" if fit_workers > 1 else "off")
    )
    return argparse.Namespace(
        csv=cli.csv_hnscc,
        fold_csv=cli.fold_csv,
        pretrained=cli.pretrained,
        output_dir=cli.output_dir,
        fold=cli.fold,
        aug=method.get("aug", "heavy"),
        hne_norm=method.get("hne_norm", "off"),
        class_weight=method.get("class_weight", "on"),
        epochs_stage1=(
            cli.epochs_stage1
            if cli.epochs_stage1 is not None
            else int(method.get("epochs_stage1", 30))
        ),
        epochs_stage2=(
            cli.epochs_stage2
            if cli.epochs_stage2 is not None
            else int(method.get("epochs_stage2", 30))
        ),
        batch_size=(
            cli.batch_size
            if cli.batch_size is not None
            else int(method.get("batch_size", 32))
        ),
        seed=int(method.get("seed", 42)),
        image_workers=image_workers,
        fit_workers=fit_workers,
        use_multiprocessing=use_multiprocessing,
        mixed_precision="off",
        patience=10,
        dry_run=cli.dry_run,
        baseline_only=False,
        method=method,
        loss=method.get("loss", "weighted_ce"),
        sampler=method.get("sampler", "random"),
        stage_policy=method.get(
            "stage_policy", "validation_multiclass_auc"
        ),
        disable_aug_components=None,
        hnscc_ratio=hnscc_ratio,
        tcga_ratio=tcga_ratio,
        csv_tcga=cli.csv_tcga,
    )


def install_source_mix_split():
    original_split = base.split_indices

    def split_indices(frame, assignments, fold):
        if "source" in frame.columns and (frame["source"] == "tcga").any():
            return split_indices_with_source(frame, assignments, fold)
        return original_split(frame, assignments, fold)

    base.split_indices = split_indices


def make_source_mix_fold_config(args, frame, assignments, fold, folds, summary):
    details = base.fold_split_details(frame, assignments, fold)
    indices = base.split_indices(frame, assignments, fold)
    train_labels = frame.iloc[indices["train"]]["label"].tolist()
    # Augment config train totals with TCGA mix counts for transparency.
    tcga_train = frame.iloc[indices["train"]]
    tcga_train = tcga_train[tcga_train["source"] == "tcga"]
    details = copy.deepcopy(details)
    details["train"]["tcga_cases"] = sorted(tcga_train["case_id"].astype(str).unique())
    details["train"]["tcga_total"] = int(len(tcga_train))
    details["train"]["hnscc_total"] = int(details["train"]["total"])
    details["train"]["total"] = int(len(indices["train"]))
    tcga_counts = tcga_train["label"].value_counts().reindex(LABELS, fill_value=0)
    details["train"]["tcga_class_counts"] = {
        label: int(tcga_counts[label]) for label in LABELS
    }
    details["train"]["class_counts"] = {
        label: int(
            details["train"]["class_counts"][label]
            + details["train"]["tcga_class_counts"][label]
        )
        for label in LABELS
    }
    weights = (
        base.balanced_class_weights(train_labels)
        if base.on_off(args.class_weight)
        else None
    )
    return {
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key not in ("fold",)
        },
        "selected_folds": folds,
        "fold": fold,
        "label_to_index": LABEL_TO_IDX,
        "splits": details,
        "class_weights": weights,
        "image_count": int(len(frame)),
        "case_count": int(frame["case_id"].nunique()),
        "source_mix": summary,
        "preprocessing": {
            "color": "cv2 BGR to RGB",
            "hne_norm": base.on_off(args.hne_norm),
            "hne_condition": "mean < 230 and std > 15",
            "hne_parameters": {"Io": 240, "alpha": 1, "beta": 0.15},
            "resize": [base.IMG_SIZE, base.IMG_SIZE],
            "model_scale": "float32 / 255",
        },
    }


def main():
    cli = parse_args()
    method = load_method_config(cli.config)
    if cli.seed is not None:
        method["seed"] = int(cli.seed)
    args = build_namespace(cli, method)
    base.validate_args(args)
    install_source_mix_split()

    hnscc = base.load_hnscc_csv(args.csv, expected_cases=10)
    assignments = base.load_assignments(args.fold_csv)
    base.validate_fold_assignments(hnscc, assignments, n_folds=5)
    tcga = load_tcga_csv(args.csv_tcga)
    mixed, sampled = build_mixed_frame(
        hnscc, tcga, args.hnscc_ratio, args.tcga_ratio, args.seed
    )
    folds = base.selected_folds(args.fold)

    # Path existence check: allow /workspace paths when running in Docker.
    missing = [p for p in mixed["image_path"] if not Path(p).is_file()]
    if missing:
        remapped = []
        for path in mixed["image_path"]:
            candidate = path
            if path.startswith("/workspace/"):
                candidate = str(PROJECT_ROOT / path[len("/workspace/") :])
            remapped.append(candidate)
        mixed = mixed.copy()
        mixed["image_path"] = remapped
        missing = [p for p in mixed["image_path"] if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing {0} images; example={1}".format(len(missing), missing[0])
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mixed.to_csv(output_dir / "mixed_frame.csv", index=False)
    sampled.to_csv(output_dir / "tcga_sampled.csv", index=False)
    summary = {
        "hnscc_ratio": args.hnscc_ratio,
        "tcga_ratio": args.tcga_ratio,
        "n_hnscc": int(len(hnscc)),
        "n_tcga_pool": int(len(tcga)),
        "n_tcga_sampled": int(len(sampled)),
        "n_mixed": int(len(mixed)),
        "tcga_label_counts": {
            str(key): int(value)
            for key, value in sampled["label"].value_counts().to_dict().items()
        },
        "note": (
            "TCGA patches come from dataset/train (pretrained source). "
            "dataset/test is TCGA internal holdout; dataset/Testset is external "
            "CPTAC/RUMC evaluation and is NOT mixed into training."
        ),
    }
    with (output_dir / "source_mix_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("Source-mix summary: {0}".format(summary))

    for fold in folds:
        indices = base.split_indices(mixed, assignments, fold)
        print(
            "Fold {0}: train={1} (HNSCC={2}, TCGA={3}) val={4} test={5}".format(
                fold,
                len(indices["train"]),
                int((mixed.iloc[indices["train"]]["source"] == "hnscc").sum()),
                int((mixed.iloc[indices["train"]]["source"] == "tcga").sum()),
                len(indices["val"]),
                len(indices["test"]),
            )
        )
        if (mixed.iloc[indices["val"]]["source"] != "hnscc").any():
            raise RuntimeError("Validation leaked non-HNSCC rows")
        if (mixed.iloc[indices["test"]]["source"] != "hnscc").any():
            raise RuntimeError("Test leaked non-HNSCC rows")

    if args.dry_run:
        print("Source-mix dry run complete -> {0}".format(output_dir))
        return

    import tensorflow as tf

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf = base.configure_tensorflow(args)

    images, preprocessing_report = base.load_all_images(
        mixed["image_path"].tolist(),
        base.on_off(args.hne_norm),
        args.image_workers,
    )
    auc_class = base.sparse_auc_class(tf)
    for fold in folds:
        config = make_source_mix_fold_config(
            args, mixed, assignments, fold, folds, summary
        )
        config["preprocessing_report"] = preprocessing_report
        fold_dir = output_dir / "fold{0:02d}".format(fold)
        fold_dir.mkdir(parents=True, exist_ok=True)
        base.write_json(fold_dir / "config.json", config)
        base.train_fold(
            tf, auc_class, args, mixed, assignments, images, fold, config
        )
    print("Source-mix training complete: {0}".format(output_dir))


if __name__ == "__main__":
    main()
