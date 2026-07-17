"""Canonical workspace paths for Path-TIL.

Docker TIL mounts the repo at ``/workspace``. New code should prefer these
helpers so experiment outputs, baseline weights, and tests stay consistent
after ``scripts/organize_workspace.py`` relocates legacy root clutter.
"""

from __future__ import annotations

import os
from pathlib import Path

PRETRAINED_MODEL_NAME = "best_InceptionResNetV2_model.h5"


def detect_repo_root(start=None):
    """Resolve the Path-TIL repo root.

    Preference order:
    1. ``PATH_TIL_ROOT`` environment variable
    2. ``/workspace`` when it looks like this repo (Docker TIL default)
    3. Walk parents from ``start`` (or this file) looking for markers
    """
    env = os.environ.get("PATH_TIL_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    workspace = Path("/workspace")
    if _looks_like_repo(workspace):
        return workspace.resolve()

    here = Path(start or __file__).resolve()
    for candidate in (here, *here.parents):
        if _looks_like_repo(candidate):
            return candidate
    # Fallback: package lives at <root>/path_til/paths.py
    return Path(__file__).resolve().parents[1]


def _looks_like_repo(path):
    path = Path(path)
    return (path / "path_til").is_dir() and (
        (path / "scripts").is_dir() or (path / "dockerfile").is_file()
    )


REPO_ROOT = detect_repo_root()
RESULTS_DIR = REPO_ROOT / "results"
BASELINES_DIR = REPO_ROOT / "baselines"
TESTS_DIR = REPO_ROOT / "tests"
TESTS_LEGACY_DIR = TESTS_DIR / "legacy"


def result_path(name):
    """Return ``results/<name>`` under the repo root."""
    name = str(name).strip().strip("/\\")
    if not name:
        raise ValueError("result name must be non-empty")
    return RESULTS_DIR / name


def baseline_path(name):
    """Return ``baselines/<name>`` under the repo root."""
    name = str(name).strip().strip("/\\")
    if not name:
        raise ValueError("baseline name must be non-empty")
    return BASELINES_DIR / name


def pretrained_model_path(root=None):
    """Locate the TILScout pretrained ``.h5`` weight file.

    Prefer ``baselines/<name>``. Fall back to a legacy root-level file only if
    it still exists (no compatibility symlinks are expected).
    """
    root = Path(root) if root is not None else REPO_ROOT
    candidates = (
        root / "baselines" / PRETRAINED_MODEL_NAME,
        root / PRETRAINED_MODEL_NAME,
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def ensure_layout_dirs(root=None):
    """Create the standard top-level layout directories if missing."""
    root = Path(root) if root is not None else REPO_ROOT
    for path in (
        root / "results",
        root / "baselines",
        root / "tests" / "legacy",
    ):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "results": root / "results",
        "baselines": root / "baselines",
        "tests_legacy": root / "tests" / "legacy",
    }
