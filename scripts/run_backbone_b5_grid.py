#!/usr/bin/env python3
"""Run B5 fold-0+1 repair grid for one or more backbone configs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_til.gpu_profile import detect_gpu_profile  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run backbone B5 fold 0+1 grid.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def config_tag(path: Path) -> str:
    name = path.stem
    for prefix in (
        "backbone_efficientnetv2_s_b5_",
        "backbone_convnext_tiny_b5_",
        "backbone_",
    ):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def run(cmd: list[str], dry_run: bool) -> None:
    print("CMD:", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    profile = detect_gpu_profile()

    for config in args.configs:
        config_path = Path(config)
        tag = config_tag(config_path)
        out = output_root / tag
        smoke_marker = out / "smoke_summary" / "metrics.json"
        fold_markers = [
            out / "fold{0:02d}".format(fold) / "fold_metrics.json" for fold in args.folds
        ]
        if args.skip_existing and smoke_marker.is_file() and all(p.is_file() for p in fold_markers):
            print("Skip existing B5 run:", out)
            continue
        run(
            [
                "python3",
                "scripts/train_hnscc_backbone_source_mix.py",
                "--config",
                str(config_path),
                "--output-dir",
                str(out),
                "--folds",
                *[str(fold) for fold in args.folds],
                "--batch-size",
                str(profile["batch_size_train_backbone"]),
                "--image-workers",
                str(profile["image_workers"]),
            ],
            args.dry_run,
        )
        run(
            [
                "python3",
                "scripts/eval_backbone_smoke.py",
                "--pred-dir",
                str(out),
                "--output",
                str(out / "smoke_summary"),
                "--folds",
                *[str(fold) for fold in args.folds],
                "--experiment-name",
                config_path.stem,
            ],
            args.dry_run,
        )
    print("B5 grid complete ->", output_root)


if __name__ == "__main__":
    main()
