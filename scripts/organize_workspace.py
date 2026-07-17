#!/usr/bin/env python3
"""Reorganize Path-TIL root clutter into results/, baselines/, and tests/legacy/.

Default mode is dry-run: print the planned moves without touching the disk.
Pass ``--apply`` only after reviewing the plan (large experiment trees).

Designed for Docker TIL::

    docker exec TIL python3 /workspace/scripts/organize_workspace.py --dry-run
    docker exec TIL python3 /workspace/scripts/organize_workspace.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow ``python3 scripts/organize_workspace.py`` without installing the package.
_REPO_CANDIDATE = Path(__file__).resolve().parents[1]
if str(_REPO_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(_REPO_CANDIDATE))

from path_til.paths import (  # noqa: E402
    PRETRAINED_MODEL_NAME,
    detect_repo_root,
    ensure_layout_dirs,
)

MANIFEST_NAME = "organize_manifest.json"
LEGACY_TEST_SCRIPT = "InceptionResNetV2_testing.py"


def classify_entry(name):
    """Return destination category for a root-level entry, or None to skip.

    Categories:
      - results   -> <root>/results/<name>
      - baselines -> <root>/baselines/<name>
      - tests_legacy -> <root>/tests/legacy/<name>
    """
    if name == "results":
        # Parent bucket itself; never move into itself.
        return None
    if name == "inference_results" or name.startswith("results_"):
        return "results"
    if name == PRETRAINED_MODEL_NAME or name.startswith("qupath_results"):
        return "baselines"
    if name == LEGACY_TEST_SCRIPT:
        return "tests_legacy"
    return None


def destination_for(root, name, category):
    root = Path(root)
    if category == "results":
        return root / "results" / name
    if category == "baselines":
        return root / "baselines" / name
    if category == "tests_legacy":
        return root / "tests" / "legacy" / name
    raise ValueError("Unknown category: {0}".format(category))


def _is_nonempty(path):
    path = Path(path)
    if not path.exists():
        return False
    if path.is_file() or path.is_symlink():
        return True
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False


def plan_moves(root):
    """Build an ordered list of move actions for ``root``."""
    root = Path(root).resolve()
    actions = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        name = entry.name
        category = classify_entry(name)
        if category is None:
            continue
        # Already relocated (or only a compatibility symlink remains).
        if entry.is_symlink():
            target = destination_for(root, name, category)
            try:
                resolved = entry.resolve()
            except OSError:
                resolved = None
            if resolved is not None and resolved == target.resolve():
                continue
        dest = destination_for(root, name, category)
        # Skip if already living at the destination (same inode path).
        try:
            if entry.resolve() == dest.resolve() and entry.exists():
                continue
        except OSError:
            pass
        if dest.exists() and entry.resolve() != dest.resolve():
            status = "conflict"
            reason = "destination_exists"
        else:
            status = "move"
            reason = None
        actions.append(
            {
                "name": name,
                "category": category,
                "source": str(entry),
                "destination": str(dest),
                "status": status,
                "reason": reason,
                "is_dir": entry.is_dir() and not entry.is_symlink(),
            }
        )
    return actions


def _safe_move(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Destination already exists: {0}".format(destination))
    try:
        source.rename(destination)
    except OSError:
        shutil.move(str(source), str(destination))
    return destination


def _relative_symlink(link_path, target_path):
    link_path = Path(link_path)
    target_path = Path(target_path)
    if link_path.exists() or link_path.is_symlink():
        raise FileExistsError("Symlink path already exists: {0}".format(link_path))
    rel = os.path.relpath(str(target_path), start=str(link_path.parent))
    link_path.symlink_to(rel, target_is_directory=target_path.is_dir())
    return rel


def apply_moves(root, actions, create_symlinks=False, manifest_path=None):
    """Execute planned moves. Conflicts are skipped, never overwritten.

    By default no compatibility symlinks are created; callers should update
    CLI paths to ``results/``, ``baselines/``, and ``tests/legacy/``.
    """
    root = Path(root).resolve()
    ensure_layout_dirs(root)
    applied = []
    errors = []

    for action in actions:
        record = dict(action)
        if action["status"] == "conflict":
            record["applied"] = False
            errors.append(record)
            applied.append(record)
            continue

        source = Path(action["source"])
        dest = Path(action["destination"])
        if not source.exists() and not source.is_symlink():
            record["applied"] = False
            record["status"] = "missing"
            record["reason"] = "source_missing"
            errors.append(record)
            applied.append(record)
            continue

        try:
            _safe_move(source, dest)
            record["moved"] = True
            if create_symlinks:
                rel = _relative_symlink(source, dest)
                record["symlink"] = rel
                record["symlink_created"] = True
            else:
                record["symlink_created"] = False
            record["applied"] = True
            record["status"] = "applied"
        except Exception as exc:  # noqa: BLE001 - surface any IO failure in manifest
            record["applied"] = False
            record["status"] = "error"
            record["reason"] = str(exc)
            errors.append(record)
        applied.append(record)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "create_symlinks": create_symlinks,
        "actions": applied,
        "error_count": len(errors),
        "moved_count": sum(1 for item in applied if item.get("applied")),
    }

    if manifest_path is None:
        manifest_path = root / "results" / MANIFEST_NAME
    else:
        manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    payload["manifest_path"] = str(manifest_path)
    return payload


def format_plan(actions, root):
    lines = [
        "Workspace organize plan",
        "root: {0}".format(root),
        "actions: {0}".format(len(actions)),
        "",
    ]
    if not actions:
        lines.append("(nothing to move)")
        return "\n".join(lines)

    width = max(len(a["name"]) for a in actions)
    for action in actions:
        marker = "MOVE" if action["status"] == "move" else "SKIP"
        lines.append(
            "[{0}] {1:<{width}}  {2}  ->  {3}".format(
                marker,
                action["name"],
                action["category"],
                action["destination"],
                width=width,
            )
        )
        if action["reason"]:
            lines.append("       reason: {0}".format(action["reason"]))
    conflicts = sum(1 for a in actions if a["status"] == "conflict")
    lines.append("")
    lines.append(
        "summary: move={0} conflict={1}".format(
            len(actions) - conflicts, conflicts
        )
    )
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Move experiment outputs, baseline models, and legacy test scripts "
            "into results/, baselines/, and tests/legacy/."
        )
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Repo root (default: detect /workspace or PATH_TIL_ROOT)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only (default when --apply is omitted)",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Perform moves and write organize_manifest.json",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help=(
            "With --apply, also create compatibility symlinks at old root paths "
            "(off by default; prefer updating CLI paths instead)"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest output path (default: <root>/results/organize_manifest.json)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text table",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else detect_repo_root()
    if not root.is_dir():
        parser.error("root is not a directory: {0}".format(root))

    actions = plan_moves(root)
    if args.json and not args.apply:
        print(json.dumps({"root": str(root), "actions": actions}, indent=2))
    else:
        print(format_plan(actions, root))

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to execute.")
        return 0 if not any(a["status"] == "conflict" for a in actions) else 2

    payload = apply_moves(
        root,
        actions,
        create_symlinks=args.symlink,
        manifest_path=args.manifest,
    )
    print(
        "\nApplied: moved={0} errors={1} manifest={2}".format(
            payload["moved_count"],
            payload["error_count"],
            payload["manifest_path"],
        )
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    return 0 if payload["error_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
