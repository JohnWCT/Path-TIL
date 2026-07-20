"""Detect GPU/CPU capacity and return throughput-oriented training/eval defaults."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _gpu_memory_mb() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(output.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def detect_gpu_profile(target_utilization: float = 0.90) -> dict:
    """Return batch/worker settings tuned for high GPU utilization on a single GPU."""
    cpu = max(1, os.cpu_count() or 8)
    gpu_mb = _gpu_memory_mb()
    image_workers = max(4, min(cpu - 2, 14))
    # Heavy imgaug + fork workers duplicate large in-memory arrays; keep fit
    # workers single-process to avoid SIGKILL/OOM while still feeding GPU via batch size.
    fit_workers = 1

    profile = {
        "cpu_count": cpu,
        "gpu_memory_mb": gpu_mb,
        "target_utilization": target_utilization,
        "image_workers": image_workers,
        "fit_workers": fit_workers,
        "use_multiprocessing": False,
        "batch_size_eval": 128,
        "batch_size_train_irv2": 48,
        "batch_size_train_backbone": 64,
        "batch_size_pretrain": 64,
        "prefetch_batches": 4,
        "mixed_precision_backbone": gpu_mb is not None and gpu_mb >= 16000,
    }

    if gpu_mb is not None and gpu_mb >= 40000:
        profile.update(
            {
                "batch_size_eval": 256,
                "batch_size_train_irv2": 96,
                "batch_size_train_backbone": 128,
                "batch_size_pretrain": 128,
                "prefetch_batches": 8,
            }
        )
    elif gpu_mb is not None and gpu_mb >= 20000:
        profile.update(
            {
                "batch_size_eval": 192,
                "batch_size_train_irv2": 64,
                "batch_size_train_backbone": 96,
                "batch_size_pretrain": 96,
                "prefetch_batches": 6,
            }
        )

    return profile


def write_profile_snapshot(path: Path, profile: dict, extra: dict | None = None) -> None:
    payload = dict(profile)
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
