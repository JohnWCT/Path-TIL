#!/usr/bin/env python3
"""Train and evaluate selected backbone full 5-fold OOF runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.backbone_metrics import has_all_full5_folds  # noqa: E402
from path_til.gpu_profile import detect_gpu_profile  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run backbone full 5-fold training + OOF.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-oof", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], dry_run: bool) -> None:
    print("CMD:", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def output_dir_for(config: Path, output_root: Path) -> Path:
    name = config.stem.replace("_full5_selected", "_full5_selected")
    return output_root / "results_{0}".format(name)


def main():
    args = parse_args()
    profile = detect_gpu_profile()
    output_root = Path(args.output_root)

    for config in args.configs:
        config_path = Path(config)
        out = output_dir_for(config_path, output_root)
        folds_present = []
        for fold in range(5):
            if (out / "fold{0:02d}".format(fold) / "fold_metrics.json").is_file():
                folds_present.append(fold)
        if args.skip_existing and has_all_full5_folds(folds_present):
            print("Skip existing full5 training:", out)
        else:
            run(
                [
                    "python3",
                    "scripts/train_hnscc_backbone_source_mix.py",
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(out),
                    "--folds",
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "--batch-size",
                    str(profile["batch_size_train_backbone"]),
                    "--image-workers",
                    str(profile["image_workers"]),
                ],
                args.dry_run,
            )

        if args.skip_oof:
            continue
        oof_out = output_root / "results_oof_with_prc" / config_path.stem
        oof_marker = oof_out / "oof_predictions.csv"
        if args.skip_existing and oof_marker.is_file():
            print("Skip existing OOF:", oof_out)
            continue
        run(
            [
                "python3",
                "scripts/eval_hnscc_oof.py",
                "--pred-dir",
                str(out),
                "--csv",
                "qupath_dataset.csv",
                "--fold-csv",
                "folds_hnscc_group5.csv",
                "--stage",
                "selected",
                "--output",
                str(oof_out),
            ],
            args.dry_run,
        )
    print("Full5 orchestration complete.")


if __name__ == "__main__":
    main()
