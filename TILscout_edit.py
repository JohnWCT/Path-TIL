# -*- coding: utf-8 -*-
"""
TILScout
@author: Huibo Zhang
"""

"""
### patch generation
"""
# ==== 與 TILscout.py 差異紀錄（人工比對） ====
# 1) 參數化與可配置化
#    - 新增 argparse，改為可由 CLI 指定 patch_size / slide_dir / slide_ext /
#      model_file / batch_size / num_threads / results_path。
#    - 原版多數路徑與參數為固定常數。
#
# 2) 影像倍率與 patch 產生
#    - 新增 get_objective_power()：優先讀 objective-power，失敗時改由 mpp-x 推估。
#    - DeepZoom tile_size 由固定 150 改為 PATCH_SIZE（可配置）。
#    - patch 子資料夾命名從 label[:-4] 改為 os.path.splitext(label)[0]（較穩定）。
#    - tile 命名由 "{col}_{row}" 改為 "TILE_{col}_{row}"。
#    - 另外回傳並保存每張 slide 的 zoom_factor（供後續 CSV 座標縮放）。
#
# 3) 預測與 TIL score
#    - 模型路徑改為 args.model_file；類別欄位統一由 CLASS_NAMES 管理。
#    - 座標抽取 regex 由 (\d+_\d+) 改為 TILE_(\d+_\d+)（對應新檔名格式）。
#    - TIL 計分核心邏輯仍相同：
#         Positive 主導且大於其餘類別；Negative 主導且大於其餘類別；
#         score = Positive / (Positive + Negative)。
#    - 新增每張 slide 的 CSV 輸出（x/y 會乘上 PATCH_SIZE 與 zoom_factor，並輸出各類別置信度）。
#
# 4) TIL map 與防呆
#    - draw_til_map 改為傳入 label，避免從 Patch_id 反推目錄。
#    - 顏色改為由 CLASS_NAMES 對應；加入 adjust_factor 變數與 try/except 防呆。
#
# 5) 流程一致性結論
#    - 高層流程一致：patch 生成 -> 分類預測 -> TIL score 計算 -> map 輸出 -> 刪除 patch 暫存。
#    - 差異主要是「參數化/健壯性/輸出擴充（CSV 與座標縮放）」，
#      非核心 TIL score 計算公式改變。
# ============================================

import openslide
import numpy as np
import tifffile as tiff
from openslide.deepzoom import DeepZoomGenerator
import os
import glob
import concurrent.futures
import threading
from normalize_HnE import norm_HnE
import logging
import argparse
import json

# ---- 錯誤收集（跳過後於流程結尾一次回報）----
_error_lock = threading.Lock()
_slide_errors = []   # [{"slide": str, "reason": str}, ...]
_patch_errors = []   # [{"slide": str, "tile": str, "reason": str}, ...]


def record_slide_error(slide_base: str, reason: str):
    with _error_lock:
        _slide_errors.append({"slide": slide_base, "reason": reason})


def record_patch_error(slide_base: str, tile_name: str, reason: str):
    with _error_lock:
        _patch_errors.append({"slide": slide_base, "tile": tile_name, "reason": reason})


_MAX_PATCH_ERRORS_PRINT = 500


def finalize_error_report():
    """於 patch 生成結束後或程式結尾呼叫：輸出摘要並寫入 results。"""
    report_path = os.path.join(results_path, "tilscout_pipeline_errors.txt")
    lines = []
    lines.append("=== TILscout 管線錯誤／略過項目彙總 ===\n")
    lines.append(f"略過或失敗的 slide 數: {len(_slide_errors)}\n")
    for e in _slide_errors:
        lines.append(f"  [SLIDE] {e['slide']}: {e['reason']}\n")
    lines.append(f"略過的 patch（單一 tile）數: {len(_patch_errors)}\n")
    for i, e in enumerate(_patch_errors):
        if i >= _MAX_PATCH_ERRORS_PRINT:
            lines.append(
                f"  ... 其餘 {len(_patch_errors) - _MAX_PATCH_ERRORS_PRINT} 筆 patch 請見 log tile_processing.log\n"
            )
            break
        lines.append(f"  [PATCH] {e['slide']} / {e['tile']}: {e['reason']}\n")
    text = "".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"錯誤彙總已寫入: {report_path}")

# Define argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='TILScout - Tumor Infiltrating Lymphocyte Analysis')
    parser.add_argument('--patch_size', type=int, default=150, help='Patch size for tile generation (default: 150)')
    parser.add_argument('--slide_dir', type=str, default="WSI_example", help='Directory containing slide files (default: WSI_example)')
    parser.add_argument('--slide_ext', type=str, default="*.svs", help='Slide file extension pattern (default: *.svs)')
    parser.add_argument('--model_file', type=str, default='baselines/best_InceptionResNetV2_model.h5', help='Model file path (default: baselines/best_InceptionResNetV2_model.h5)')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for prediction (default: 1000)')
    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads for processing (default: 1)')
    parser.add_argument('--results_path', type=str, default="./results", help='Results directory path (default: ./results)')
    parser.add_argument(
        '--generic-tiff-default-objective',
        type=float,
        default=40.0,
        help='generic-tiff 無法從 TIFF Resolution 推算物鏡倍率時的預設值（預設 40，與本資料集 Philips mpp=0.25 對齊）',
    )
    return parser.parse_args()

# python TILscout_edit.py --slide_dir WSI_example --slide_ext "*.tiff" --batch_size 100 --num_threads 1

# Parse arguments
args = parse_arguments()

# Define constants
SIZE = 224  
BATCH_SIZE = args.batch_size  
NUM_THREADS = args.num_threads
PATCH_SIZE = args.patch_size
CLASS_NAMES = ['Positive', 'Negative', 'Other']

# Define results folder path
results_path = os.path.abspath(args.results_path)
if not os.path.exists(results_path):
    os.makedirs(results_path)

PATCH_PROGRESS_FILE = os.path.join(results_path, "patch_progress.json")
_patch_progress_lock = threading.Lock()


def load_patch_progress():
    if not os.path.exists(PATCH_PROGRESS_FILE):
        return {}
    try:
        with open(PATCH_PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Warning: 無法讀取 patch 進度檔，將重新建立: {e}")
        return {}


patch_progress = load_patch_progress()


def save_patch_progress():
    tmp_path = f"{PATCH_PROGRESS_FILE}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(patch_progress, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, PATCH_PROGRESS_FILE)


def mark_patch_completed(slide_base, zoom_factor):
    with _patch_progress_lock:
        patch_progress[slide_base] = {
            "status": "completed",
            "zoom_factor": zoom_factor,
        }
        save_patch_progress()


def get_completed_patch_zoom_factor(slide_base, slide_patch_dir):
    with _patch_progress_lock:
        entry = patch_progress.get(slide_base, {})
    if entry.get("status") != "completed":
        return None
    if not os.path.isdir(slide_patch_dir):
        return None
    return entry.get("zoom_factor", 1.0)

# Function to process images and generate patches
def detect_slide_vendor(slide: openslide.OpenSlide) -> str:
    """依 OpenSlide vendor 判斷掃描儀／TIFF 類型。"""
    return slide.properties.get(openslide.PROPERTY_NAME_VENDOR, "unknown").strip().lower()


def objective_from_mpp(mpp_x: float) -> float:
    """與原 TILScout 相同：objective ≈ 10 / mpp（mpp 單位 µm/pixel）。"""
    if mpp_x <= 0:
        raise ValueError("MPP value cannot be zero or negative.")
    return 10.0 / mpp_x


def get_objective_power_philips(slide: openslide.OpenSlide) -> float:
    """
    Philips WSI（及具標準 OpenSlide MPP 標籤之格式）：
    1) openslide.objective-power
    2) openslide.mpp-x → 10/mpp
    """
    try:
        return float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    except (KeyError, ValueError):
        print("Info: 'openslide.objective-power' not found. Trying to calculate from MPP.")
        mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        return objective_from_mpp(mpp_x)


def _mpp_from_tiff_resolution_tags(slide: openslide.OpenSlide) -> float:
    """
    generic-tiff（常見 tifffile 匯出）常無 openslide.mpp-x，但留有 TIFF Resolution：
      tiff.XResolution / tiff.YResolution + tiff.ResolutionUnit

    換算（µm/pixel）：
      centimeter → 10000 / pixels_per_cm
      inch       → 25400 / pixels_per_inch
      millimeter → 1000 / pixels_per_mm
    """
    x_res = float(slide.properties["tiff.XResolution"])
    y_res = float(slide.properties["tiff.YResolution"])
    unit = slide.properties.get("tiff.ResolutionUnit", "").strip().lower()
    if x_res <= 0 or y_res <= 0:
        raise ValueError("TIFF resolution must be positive.")

    if unit == "centimeter":
        mpp_x = 10000.0 / x_res
        mpp_y = 10000.0 / y_res
    elif unit == "inch":
        mpp_x = 25400.0 / x_res
        mpp_y = 25400.0 / y_res
    elif unit in ("millimeter", "millimetre"):
        mpp_x = 1000.0 / x_res
        mpp_y = 1000.0 / y_res
    else:
        raise ValueError(f"Unsupported tiff.ResolutionUnit: {unit!r}")

    return (mpp_x + mpp_y) / 2.0


def get_objective_power_generic_tiff(slide: openslide.OpenSlide) -> float:
    """
    generic-tiff 專用物鏡倍率推估：
    1) 由 TIFF Resolution 標籤換算 mpp，再 objective = 10/mpp
    2) 若標籤缺失，使用 --generic-tiff-default-objective（預設 40x）
    """
    try:
        mpp = _mpp_from_tiff_resolution_tags(slide)
        objective = objective_from_mpp(mpp)
        print(
            f"Info: generic-tiff objective from TIFF resolution tags: "
            f"{objective:.2f}x (mpp≈{mpp:.4f})"
        )
        return objective
    except (KeyError, ValueError) as e:
        fallback = float(args.generic_tiff_default_objective)
        print(
            f"Warning: generic-tiff cannot read TIFF resolution ({e}); "
            f"using --generic-tiff-default-objective={fallback:.2f}x"
        )
        return fallback


def get_objective_power(slide: openslide.OpenSlide) -> float:
    """
    依掃描儀／TIFF 類型選擇物鏡倍率讀取方式：
      philips      → get_objective_power_philips（維持原邏輯）
      generic-tiff → get_objective_power_generic_tiff（TIFF Resolution 換算）
      其他         → 先 Philips 路徑，失敗再 generic-tiff 路徑
    """
    vendor = detect_slide_vendor(slide)
    if vendor == "philips":
        return get_objective_power_philips(slide)
    if vendor == "generic-tiff":
        return get_objective_power_generic_tiff(slide)

    print(f"Info: unknown vendor '{vendor}', trying Philips then generic-tiff methods.")
    try:
        return get_objective_power_philips(slide)
    except (KeyError, ValueError, ZeroDivisionError):
        return get_objective_power_generic_tiff(slide)

def process_image(directory_path, patch_path):
    label = os.path.basename(directory_path)
    slide_base = os.path.splitext(label)[0]
    print("Processing:", label)
    # Remove file extension properly
    file_name = os.path.join(patch_path, slide_base)
    completed_zoom_factor = get_completed_patch_zoom_factor(slide_base, file_name)
    if completed_zoom_factor is not None:
        print(f"Skip patching completed slide: {slide_base}")
        return completed_zoom_factor
    os.makedirs(file_name, exist_ok=True)

    slide = None
    try:
        slide = openslide.OpenSlide(directory_path)
    except Exception as e:
        msg = f"無法開啟 slide: {e}"
        print(f"Error opening slide at {directory_path}: {msg}")
        record_slide_error(slide_base, msg)
        return None

    try:
        vendor = detect_slide_vendor(slide)
        print(f"Slide vendor/format: {vendor}")
        try:
            objective = get_objective_power(slide)
        except ValueError as e:
            record_slide_error(slide_base, str(e))
            return None
        print(f"Determined objective power: {objective:.2f}x")

        tiles = DeepZoomGenerator(slide, tile_size=PATCH_SIZE, overlap=0, limit_bounds=False)
        level = tiles.level_count - 1 if objective < 40.0 else tiles.level_count - 2
        cols, rows = tiles.level_tiles[level]

        zoom_factor = round(
            tiles.level_dimensions[tiles.level_count - 1][0] / tiles.level_dimensions[level][0], 2
        )

        for row in range(rows):
            for col in range(cols):
                process_tile(tiles, level, col, row, file_name, label, slide_base)

        mark_patch_completed(slide_base, zoom_factor)
        return zoom_factor
    finally:
        if slide is not None:
            slide.close()

# Configure logging
logging.basicConfig(level=logging.INFO, filename='tile_processing.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
def process_tile(tiles, level, col, row, file_name, label, slide_base):
    tile_name = f"TILE_{col}_{row}"
    try:
        temp_tile = tiles.get_tile(level, (col, row))
        temp_tile_RGB = temp_tile.convert('RGB')
        temp_tile_np = np.array(temp_tile_RGB)
        if temp_tile_np.mean() < 230 and temp_tile_np.std() > 15:
            try:
                norm_img, _, _ = norm_HnE(temp_tile_np, Io=240, alpha=1, beta=0.15)
            except Exception as e:
                logging.error(f"norm_HnE failed {slide_base} {tile_name}: {str(e)}")
                record_patch_error(slide_base, tile_name, f"norm_HnE: {e}")
                return
            out_path = os.path.join(file_name, f"{tile_name}_{label[:23]}_norm.tif")
            try:
                tiff.imwrite(out_path, norm_img)
            except Exception as e:
                logging.error(f"save tile failed {slide_base} {tile_name}: {str(e)}")
                record_patch_error(slide_base, tile_name, f"imwrite: {e}")
                return
            logging.info(f"Processing tile number: {tile_name}")
        else:
            logging.info(f"Skipping tile: {tile_name}")
    except Exception as e:
        logging.error(f"Error processing tile {tile_name}: {str(e)}")
        record_patch_error(slide_base, tile_name, str(e))

patch_path = os.path.abspath("./patch") #Create a temporary folder ./patch for storing patches
if not os.path.exists(patch_path):
    os.makedirs(patch_path)
directory_paths = glob.glob(os.path.join(args.slide_dir, args.slide_ext))

# Store zoom factors for each slide
zoom_factors = {}

with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = [executor.submit(process_image, path, patch_path) for path in directory_paths]
    for path, future in zip(directory_paths, futures):
        label = os.path.splitext(os.path.basename(path))[0]
        try:
            zoom_factor = future.result()
        except Exception as e:
            record_slide_error(label, f"執行緒意外錯誤: {e}")
            continue
        if zoom_factor is not None:
            zoom_factors[label] = zoom_factor


"""
### patch prediction and TIL score calculation
"""
import cv2
from tensorflow import keras
import pandas as pd
import gc
from concurrent.futures import ThreadPoolExecutor


# Load model for inference only. compile=False avoids requiring training-only
# custom metrics such as SparseMulticlassAUC during deserialization.
prediction_model = keras.models.load_model(args.model_file, compile=False)
_prediction_lock = threading.Lock()

def process_directory(directory_path, zoom_factor):
    label = os.path.splitext(os.path.basename(directory_path))[0]  # Remove extension
    img_paths = glob.glob(os.path.join(directory_path, "*.tif"))
    num_images = len(img_paths)
    if num_images == 0:
        print(f"略過推論（0 個 .tif patch）: {label}")
        record_slide_error(label, "推論階段：patch 資料夾內無 .tif（整張略過或產圖全失）。")
        return None
    num_batches = np.ceil(num_images / BATCH_SIZE).astype(int)
    pred_ID, Y_proba = pd.DataFrame(), pd.DataFrame()

    for batch in range(num_batches):
        batch_img_paths = img_paths[batch * BATCH_SIZE: min((batch + 1) * BATCH_SIZE, num_images)]
        pred_images = [cv2.cvtColor(cv2.resize(cv2.imread(img_path, cv2.IMREAD_COLOR), (SIZE, SIZE)), cv2.COLOR_BGR2RGB) for img_path in batch_img_paths]
        X_pred = np.array(pred_images) / 255.0

        # Keras/TensorFlow model.predict is not safe to call concurrently on
        # the same shared model, especially on GPU.
        with _prediction_lock:
            y_proba = prediction_model.predict(X_pred, verbose=0)
        batch_pred_ID = pd.DataFrame(batch_img_paths, columns=['Patch_id'])
        batch_Y_proba = pd.DataFrame(y_proba, columns=CLASS_NAMES)

        pred_ID = pd.concat([pred_ID, batch_pred_ID], ignore_index=True)
        Y_proba = pd.concat([Y_proba, batch_Y_proba], ignore_index=True)

        del pred_images, X_pred
        gc.collect()

    # Combine results
    results = pd.concat([pred_ID, Y_proba], axis=1)
    results['MaxCategory'] = results[CLASS_NAMES].idxmax(axis=1)
    # Extract coordinates from TILE_X_Y pattern
    results['Coordinates'] = results['Patch_id'].str.extract(r'TILE_(\d+_\d+)')
    results[['X', 'Y']] = results['Coordinates'].str.split('_', expand=True).astype(int)
    
    # Calculate TIL scores
    TIL_positive = results[results[CLASS_NAMES[0]] > results[CLASS_NAMES[1:]].max(axis=1)]
    TIL_negative = results[results[CLASS_NAMES[1]] > results[CLASS_NAMES[::2]].max(axis=1)]
    #Other = results[results[CLASS_NAMES[2]] > results[CLASS_NAMES[:2]].max(axis=1)]

    denom = TIL_negative.shape[0] + TIL_positive.shape[0]
    if denom == 0:
        print(f"略過 TIL 分數（無 Positive/Negative 主導 patch）: {label}")
        score = float("nan")
    else:
        score = TIL_positive.shape[0] / denom
    result_text = f"{label}: TIL Score = {score:.4f}" if not np.isnan(score) else f"{label}: TIL Score = NaN (no pos/neg)"
    print(result_text)
    with open(os.path.join(results_path, f"{label}_TIL_score.txt"), 'w') as f:
        print(label, score, file=f)
    
    # Generate CSV output with the specified format
    csv_results = pd.DataFrame()
    csv_results['x'] = (results['X'] * PATCH_SIZE * zoom_factor).astype(int)
    csv_results['y'] = (results['Y'] * PATCH_SIZE * zoom_factor).astype(int)
    csv_results['prediction'] = results['MaxCategory']
    csv_results['confidence'] = results[CLASS_NAMES].max(axis=1)
    csv_results['Truth'] = ''
    print(f"{label} zoom factor = {zoom_factor} <-----")
    
    # Add confidence columns for each class
    for i, class_name in enumerate(CLASS_NAMES):
        csv_results[f'{class_name}_conf'] = results[class_name]
    
    csv_results[f'Not_{CLASS_NAMES[0]}_conf'] = 1 - results[CLASS_NAMES[0]]
    
    # Save CSV file
    csv_filename = os.path.join(results_path, f"{label}.csv")
    csv_results.to_csv(csv_filename, index=False)
    print(f"CSV results saved to: {csv_filename}")
    
    return results


def safe_process_directory(directory_path, zoom_factor):
    label = os.path.splitext(os.path.basename(directory_path))[0]
    try:
        return process_directory(directory_path, zoom_factor)
    except Exception as e:
        msg = f"推論階段錯誤: {e}"
        print(f"{label}: {msg}")
        record_slide_error(label, msg)
        return None

# Processing all directories
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    directory_paths = glob.glob(os.path.join("patch", "*"))  
    zoom_factor_list = []
    for p in directory_paths:
        label = os.path.splitext(os.path.basename(p))[0]
        zoom_factor_list.append(zoom_factors.get(label, 1.0))  # Default to 1.0 if not found
    results_list = list(executor.map(safe_process_directory, directory_paths, zoom_factor_list))



"""
### TIL map generation
"""
import matplotlib.pyplot as plt

def draw_til_map(results, label):
    if results is None or len(results) == 0:
        return
    # Extraction of the directory name
    label1 = label
    #print("Generating TIL map for:", label1)
    #colors = {'Positive': 'purple', 'Negative': 'red', 'Other': 'pink'}
    colors = {CLASS_NAMES[0]: 'purple', CLASS_NAMES[1]: 'red', CLASS_NAMES[2]: 'pink'}
    max_dims = results[['X', 'Y']].max()
    adjust_factor = 7.1
    plt.figure(figsize=(max_dims['X']/adjust_factor, max_dims['Y']/adjust_factor))  # Adjust figure size dynamically

    for category, color in colors.items():
        category_data = results[results['MaxCategory'] == category]
        plt.scatter(x=category_data.X, y=category_data.Y, s=48, color=color, marker='s')

    plt.title(f"{label1} TIL Map", fontsize=max_dims['X']/adjust_factor)  # Title size adjusts with figure size
    plt.gca().set_aspect('equal')
    plt.xlim(0, max_dims['X'] + 5)
    plt.ylim(max_dims['Y'] + 5, 0)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{results_path}/{label1}_TIL_map.pdf")
    plt.close()
    print("TIL map is generated and stored in the results directory for:", label)

# Extract labels from directory paths（與 results 對齊，略過 None）
labels = [os.path.splitext(os.path.basename(p))[0] for p in directory_paths]
try:
    for res, lab in zip(results_list, labels):
        if res is None or len(res) == 0:
            print(f"TIL map 略過（無預測資料）: {lab}")
            continue
        draw_til_map(res, lab)
except Exception as e:
    print(f"Error generating TIL maps: {str(e)}")

finalize_error_report()

#delete temporary folder patch
import shutil
patch_path = os.path.abspath("./patch")
shutil.rmtree(patch_path)

