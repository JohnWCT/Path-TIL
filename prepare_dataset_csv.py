# -*- coding: utf-8 -*-
"""
prepare_dataset_csv.py
======================
掃描影像目錄，產出 InceptionResNetV2_QuPath_2stage.py 可用之 CSV（欄位: case_id, image_path, label）。

兩種預設流程::

  1) qupath（預設）
     QuPathOutput/HXXXX/patches/ 下之子資料夾對應 TIL 標籤。

  2) cptac_luad
     Testset/CPTAC_LUAD/ 下 A_positive → positive, B_negative → negative, C_other → other
     （與 best_InceptionResNetV2_model.h5 / TILscout 之神經元順序一致，用於對照「資料集差異」實驗）

使用範例::

    python3 prepare_dataset_csv.py
    python3 prepare_dataset_csv.py --preset qupath --output /workspace/Path_TIL/qupath_dataset.csv
    python3 prepare_dataset_csv.py --preset cptac_luad
    python3 prepare_dataset_csv.py --preset cptac_luad --cptac-dir /workspace/Path_TIL/Testset/CPTAC_LUAD \\
        --output /workspace/Path_TIL/cptac_luad_dataset.csv
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

# ========= QuPath 預設 =========
DEFAULT_QUPATH_DIR = "/workspace/Path_TIL/QuPathOutput"
DEFAULT_QUPATH_CSV = "/workspace/Path_TIL/qupath_dataset.csv"

QUPATH_LABEL_MAP = {
    "BenignOrDysplasia": "other",
    "Non-tumor": "other",
    "TIL-positive": "positive",
    "TIL-negative": "negative",
}

# ========= CPTAC_LUAD 預設（Testset/CPTAC_LUAD）=========
DEFAULT_CPTAC_DIR = "/workspace/Path_TIL/Testset/CPTAC_LUAD"
DEFAULT_CPTAC_CSV = "/workspace/Path_TIL/cptac_luad_dataset.csv"

# 資料夾名稱 → 與 2stage / TILscout 一致之小寫三類
CPTAC_FOLDER_LABEL_MAP = {
    "A_positive": "positive",
    "B_negative": "negative",
    "C_other": "other",
}

_IMAGE_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


def _collect_images_in_dir(dir_path: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(dir_path, "*.*")))
    return [p for p in paths if p.lower().endswith(_IMAGE_EXTS)]


def build_qupath_records(qupath_dir: str) -> list[dict]:
    records: list[dict] = []
    case_dirs = sorted(glob.glob(os.path.join(qupath_dir, "H*")))
    print(f"[qupath] 找到 {len(case_dirs)} 個病例資料夾 (H*)")

    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)
        patches_dir = os.path.join(case_dir, "patches")
        if not os.path.isdir(patches_dir):
            print(f"  ⚠ {case_id}: patches 不存在，跳過")
            continue

        for subfolder, target_label in QUPATH_LABEL_MAP.items():
            sub_path = os.path.join(patches_dir, subfolder)
            if not os.path.isdir(sub_path):
                continue
            for img_path in _collect_images_in_dir(sub_path):
                records.append(
                    {
                        "case_id": case_id,
                        "image_path": img_path,
                        "label": target_label,
                    }
                )
        print(f"  ✓ {case_id}: 累計 {len(records)} 筆")

    return records


def build_cptac_luad_records(cptac_root: str) -> list[dict]:
    """
    掃描 cptac_root 底下第一層子資料夾 A_positive / B_negative / C_other。
    case_id 統一為資料集目錄 basename（例如 CPTAC_LUAD），便於與 QuPath 多病例格式區分。
    """
    records: list[dict] = []
    case_id = os.path.basename(os.path.normpath(cptac_root))
    if not os.path.isdir(cptac_root):
        print(f"[cptac_luad] ⚠ 根目錄不存在: {cptac_root}")
        return records

    print(f"[cptac_luad] 根目錄: {cptac_root} (case_id={case_id})")

    for folder, target_label in CPTAC_FOLDER_LABEL_MAP.items():
        sub_path = os.path.join(cptac_root, folder)
        if not os.path.isdir(sub_path):
            print(f"  ⚠ 缺少子資料夾: {folder} → 跳過")
            continue
        imgs = _collect_images_in_dir(sub_path)
        for img_path in imgs:
            records.append(
                {
                    "case_id": case_id,
                    "image_path": img_path,
                    "label": target_label,
                }
            )
        print(f"  ✓ {folder} → {target_label}: {len(imgs)} 張")

    return records


def _print_stats(df: pd.DataFrame) -> None:
    print(f"\n======== 資料集統計 ========")
    print(f"總筆數: {len(df)}")
    print(f"\n各類別數量:")
    print(df["label"].value_counts().to_string())
    print(f"\n各 case_id 數量:")
    print(df["case_id"].value_counts().sort_index().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="產出 case_id, image_path, label 之訓練用 CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python3 prepare_dataset_csv.py
  python3 prepare_dataset_csv.py --preset cptac_luad
  python3 prepare_dataset_csv.py --preset cptac_luad --output /workspace/Path_TIL/my_cptac.csv
""",
    )
    parser.add_argument(
        "--preset",
        choices=("qupath", "cptac_luad"),
        default="qupath",
        help="qupath=QuPathOutput/H*/patches；cptac_luad=Testset/CPTAC_LUAD 下 A_/B_/C_ 三類",
    )
    parser.add_argument(
        "--qupath-dir",
        default=DEFAULT_QUPATH_DIR,
        help="QuPath 根目錄（內含 HXXXX）",
    )
    parser.add_argument(
        "--cptac-dir",
        default=DEFAULT_CPTAC_DIR,
        help="CPTAC_LUAD 根目錄（內含 A_positive, B_negative, C_other）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="輸出 CSV 路徑（未指定則依 preset 使用預設檔名）",
    )
    args = parser.parse_args()

    if args.preset == "qupath":
        out = args.output or DEFAULT_QUPATH_CSV
        records = build_qupath_records(args.qupath_dir)
    else:
        out = args.output or DEFAULT_CPTAC_CSV
        records = build_cptac_luad_records(args.cptac_dir)

    if not records:
        print("\n❌ 未掃描到任何影像，未寫入 CSV。請確認目錄結構與 --preset 是否相符。")
        raise SystemExit(1)

    df = pd.DataFrame(records, columns=["case_id", "image_path", "label"])
    _print_stats(df)
    df.to_csv(out, index=False)
    print(f"\n✓ CSV 已儲存至: {out}")
    print(
        "  標籤為小寫 positive/negative/other，與 InceptionResNetV2_QuPath_2stage.py 及 "
        "TILscout ['Positive','Negative','Other'] 索引順序一致。"
    )


if __name__ == "__main__":
    main()
