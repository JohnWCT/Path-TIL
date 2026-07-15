# Path-TIL

本專案延續 [TILScout](https://github.com/huibozh/TILScout)（Huibo Zhang），在 Whole Slide Image（WSI）上進行腫瘤浸潤淋巴細胞（TIL）分析，並擴充 **可參數化推論管線**、**QuPath / CPTAC 資料集訓練與評估**、以及 **分數彙整工具**。

原始 TILScout 以學術非商業用途免費提供；本儲存庫之新增程式碼採 [MIT License](LICENSE)。使用上游預訓練權重與方法時，請一併引用 TILScout 相關文獻與授權說明。

## 功能概覽

| 模組 | 說明 |
|------|------|
| `TILscout.py` | 上游 TILScout 推論流程（固定路徑與參數） |
| `TILscout_edit.py` | 強化版：CLI 參數、H&E 標準化、per-patch CSV、錯誤彙總、TIL map |
| `normalize_HnE.py` | 與 TILScout 一致的 H&E 正規化（Macenko 風格） |
| `prepare_dataset_csv.py` | 由 QuPath 或 CPTAC_LUAD 目錄產生訓練用 CSV |
| `InceptionResNetV2_QuPath_2stage.py` | 兩階段微調（與 TILscout 推論前處理對齊） |
| `InceptionResNetV2_cross_val*.py` | K-fold 交叉驗證訓練 |
| `InceptionResNetV2_testing.py` | 使用已訓練 fold 模型評估資料夾 |
| `inference_InceptionResNetV2.py` / `inference_cptac_luad.py` | 推論與外部資料集評估 |
| `aggregate_til_scores.py` | 將多張 slide 的 `*_TIL_score.txt` 彙整為 CSV |

## 環境需求

- Python 3.8+（建議與 Docker 映像一致）
- GPU（建議，用於 TensorFlow 推論與訓練）
- OpenSlide 系統函式庫（WSI 讀取）

### Python 套件（與 TILScout / `dockerfile` 對齊）

| 套件 | 建議版本 |
|------|----------|
| tensorflow | 2.10.x |
| openslide-python | 1.2.0 |
| opencv-python-headless | 4.7.0.72 |
| scikit-learn | 1.2.1 |
| pandas | 1.4.4 |
| matplotlib | 3.7.0 |
| numpy | 1.23.5 |
| tifffile | 2023.4.12（TILscout patch 寫入） |
| imgaug | 0.4.0（`--aug heavy`） |
| scikit-image | 0.19.3（`--aug heavy` 之 HED 增強） |

`InceptionResNetV2_QuPath_2stage.py` 的 `--aug none` / `light` / `medium` 僅使用 TensorFlow 內建增強，不需 imgaug；`--aug heavy` 才需要上表後兩項。

### Docker（選用）

```bash
docker build -f dockerfile -t path-til .
docker run -it --gpus all \
  --name TIL \
  --ipc=host \
  -v "$(pwd)":/workspace/Path_TIL \
  -v /path/to/dataset:/workspace/dataset \
  -w /workspace/Path_TIL \
  path-til
```

容器預設進入 `/bin/bash`。若既有容器將整個專案直接掛到 `/workspace`，請將以下指令中的 `/workspace/Path_TIL` 改為 `/workspace`；資料集仍建議獨立掛載至 `/workspace/dataset`。

## 安裝

```bash
git clone <your-repo-url>
cd Path_TIL
pip install openslide-python==1.2.0 opencv-python-headless==4.7.0.72 \
  scikit-learn==1.2.1 pandas==1.4.4 matplotlib==3.7.0 numpy==1.23.5 \
  tifffile==2023.4.12 imgaug==0.4.0 scikit-image==0.19.3
# 另需安裝與 GPU 對應的 TensorFlow 2.10
```

### 預訓練模型

請從 [TILScout](https://github.com/huibozh/TILScout) 取得 `best_InceptionResNetV2_model.h5`，置於專案根目錄（`.gitignore` 已排除大型 `.h5` 檔，請勿提交至 Git）。

## 輸入資料集

資料集壓縮檔放在 `/workspace/dataset/`。解壓後保留原始 ZIP，並在相同目錄產生 `train/`、`test/`、`Testset/` 與 `QuPathOutput/`：

```text
dataset/
├── Dateset_TCGA.zip                  # TCGA 訓練／內部測試集壓縮檔
├── Testset.zip                       # 外部測試集壓縮檔
├── tvgh_hnscc_TIL_patch_label.zip    # TVGH HNSCC QuPath 標註壓縮檔
│
├── train/                            # TCGA 訓練集，共 72,272 個 .tif patch
│   ├── A_positive/                   # 18,071
│   ├── B_negative/                   # 26,251
│   └── C_other/                      # 27,950
├── test/                             # TCGA 內部測試集，共 18,216 個 .tif patch
│   ├── A_positive/                   # 4,031
│   ├── B_negative/                   # 7,974
│   └── C_other/                      # 6,211
│
├── Testset/                          # 外部測試集，共 9,000 個 .tif patch
│   ├── CPTAC_LUAD/                   # 3,000
│   │   ├── A_Positive/               # 224
│   │   ├── B_Negative/               # 1,357
│   │   └── C_Other/                  # 1,419
│   ├── CPTAC_LUSC/                   # 3,000
│   │   ├── A_Positive/               # 291
│   │   ├── B_Negative/               # 1,769
│   │   └── C_Other/                  # 940
│   └── RUMC-BRCA/                    # 3,000
│       ├── A_Positive/               # 162
│       ├── B_Negative/               # 1,219
│       └── C_Other/                  # 1,619
│
└── QuPathOutput/                     # TVGH HNSCC，共 10 個 case
    └── HXXXX/
        ├── labeled_detections.csv    # QuPath 標註／偵測資料
        └── patches/
            ├── TIL-positive/         # 模型類別 positive
            ├── TIL-negative/         # 模型類別 negative
            ├── Non-tumor/            # 映射為模型類別 other
            └── BenignOrDysplasia/    # 若該 case 存在，映射為 other
```

TVGH HNSCC 的 10 個 case 為 `H0002`、`H0003`、`H0005`、`H0006`、`H0007`、`H0008`、`H0013`、`H0018`、`H0041`、`H0103`；合計 5,492 個 PNG patch 與 10 個 CSV。解壓後所有資料集合計 104,990 個檔案（99,488 個 TIF、5,492 個 PNG、10 個 CSV）。

### 在既有 Docker 容器解壓縮

以下指令在 `TIL` 容器內以最多 3 個 worker 平行解壓三個互不相依的 ZIP。解壓縮是 CPU／儲存裝置 I/O 工作，不使用 GPU：

```bash
docker exec TIL python3 -c "
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

root = Path('/workspace/dataset')
archives = sorted(root.glob('*.zip'))
if not archives:
    raise FileNotFoundError(f'No ZIP archives found in {root}')

def extract(archive):
    with ZipFile(archive) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f'Corrupted file in {archive.name}: {bad}')
        for member in z.namelist():
            parts = Path(member).parts
            if Path(member).is_absolute() or '..' in parts:
                raise RuntimeError(f'Unsafe path in {archive.name}: {member}')
        z.extractall(root)
    print(f'Extracted: {archive.name}')

with ThreadPoolExecutor(max_workers=min(3, len(archives))) as pool:
    list(pool.map(extract, archives))
"
```

此程式會先驗證 ZIP CRC 與成員路徑，再以最多 3 個 worker 平行解壓。若重複執行，現有同名檔案可能被覆寫。

## HNSCC case-level GroupCV

`InceptionResNetV2_QuPath_2stage.py` 保留作為 patch-level baseline。HNSCC 泛化性實驗請使用新的 case-level GroupCV 流程，確保同一 case 的 patches 不會跨 train、validation、test：

```text
每個 fold：7 train cases + 1 validation case + 2 held-out test cases
5 folds：每個 case 恰好作為 held-out test 一次
模型選擇：只使用 validation AUC
最終評估：合併 5 folds 的 held-out test predictions
```

### 1. 建立 QuPath manifest

```bash
docker exec TIL bash -lc '
cd /workspace/Path_TIL
python3 prepare_dataset_csv.py \
  --preset qupath \
  --qupath-dir /workspace/dataset/QuPathOutput \
  --output /workspace/Path_TIL/qupath_dataset.csv
'
```

### 2. 建立 deterministic GroupCV folds

程式會枚舉 10 cases 的 test pair 分組，近似平衡每 fold 的三類 patches 與總大小；validation case 也會以全域搜尋選取，且各 fold 不重複。

```bash
docker exec TIL bash -lc '
cd /workspace/Path_TIL
python3 scripts/make_hnscc_group_folds.py \
  --csv qupath_dataset.csv \
  --output folds_hnscc_group5.csv \
  --n-folds 5 \
  --seed 42
'
```

輸出：

```text
folds_hnscc_group5.csv   # fold,case_id,role
folds_hnscc_group5.json  # 每 fold cases、class counts 與搜尋 objective
```

### 3. Dry-run 與單 fold smoke test

Dry-run 只驗證 manifest、影像路徑、folds、class weights 與輸出設定，不載入 TensorFlow：

```bash
docker exec TIL bash -lc '
cd /workspace/Path_TIL
python3 scripts/train_hnscc_groupcv_irv2.py \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --pretrained best_InceptionResNetV2_model.h5 \
  --output-dir results_groupcv_dry_run \
  --fold 0 \
  --dry-run
'
```

GPU smoke test（Stage 1/2 各 1 epoch）：

```bash
docker exec TIL bash -lc '
cd /workspace/Path_TIL
python3 scripts/train_hnscc_groupcv_irv2.py \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --pretrained best_InceptionResNetV2_model.h5 \
  --output-dir results_groupcv_smoke \
  --fold 0 \
  --aug heavy \
  --hne-norm on \
  --class-weight on \
  --epochs-stage1 1 \
  --epochs-stage2 1 \
  --batch-size 32
'
```

腳本使用 GPU memory growth；`none/light/medium` 使用 `tf.data`、`AUTOTUNE` 與 prefetch，`heavy` 使用 imgaug `Sequence`。GPU 使用率取決於影像 I/O、H&E normalization、augmentation 與 batch size，無法保證固定比例；請用 `nvidia-smi` 觀察後逐步調高 `--batch-size` 或 `--fit-workers`，避免 OOM。

### 4. 完整 5-fold 訓練

省略 `--fold` 即依序執行全部 folds：

```bash
docker exec TIL bash -lc '
cd /workspace/Path_TIL
python3 scripts/train_hnscc_groupcv_irv2.py \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --pretrained best_InceptionResNetV2_model.h5 \
  --output-dir results_groupcv_irv2 \
  --aug heavy \
  --hne-norm on \
  --class-weight on \
  --epochs-stage1 30 \
  --epochs-stage2 30 \
  --batch-size 32
'
```

每個 `foldXX/` 會輸出 Stage 1/2 最佳模型、學習曲線、設定、metrics，以及 validation/test 逐 patch predictions；held-out test 僅在該 fold 訓練完成後評估一次。

### 5. OOF 與 slide-level TIL 評估

```bash
docker exec TIL bash -lc '
cd /workspace/Path_TIL
python3 scripts/eval_hnscc_oof.py \
  --pred-dir results_groupcv_irv2 \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --output results_groupcv_irv2/oof_summary
'
```

輸出：

```text
oof_predictions.csv           # 5 folds 的完整 held-out predictions
patch_auc_summary.csv         # positive-vs-rest 與 macro/weighted OVR 等
slide_til_score_summary.csv   # 每 case GT/Pred TIL、MAE、相關係數
eval_summary.json             # 主要結果摘要
```

slide-level TIL score 使用 hard prediction：

\[
\text{TIL score} = \frac{\text{Positive}}{\text{Positive}+\text{Negative}}
\]

### 6. 測試

```bash
docker exec TIL bash -lc '
cd /workspace/Path_TIL
python3 -m unittest discover -s tests -v
'
```

## 快速開始

### 1. WSI TIL 推論（建議使用強化版）

將 `.svs` 等 WSI 放入 `WSI_example/`（或自訂目錄）：

```bash
python TILscout_edit.py \
  --slide_dir WSI_example \
  --slide_ext "*.svs" \
  --model_file best_InceptionResNetV2_model.h5 \
  --results_path ./results
```

輸出包含：

- `{slide_id}_TIL_score.txt` — TIL 分數
- `{slide_id}.csv` — 各 patch 座標與類別置信度
- `{slide_id}_TIL_map.pdf` — 可視化 TIL map
- `tilscout_pipeline_errors.txt` — 略過或失敗項目彙總

### 2. 彙整多張 slide 分數

```bash
python aggregate_til_scores.py --results_dir results_fold04_stage2_tvgh
# 預設輸出：<results_dir>/til_scores_summary.csv
```

### 3. 由 QuPath 標註建立訓練 CSV

```bash
python prepare_dataset_csv.py --preset qupath \
  --qupath-dir QuPathOutput \
  --output qupath_dataset.csv
```

### 4. 兩階段訓練（與 TILscout 前處理對齊）

```bash
python InceptionResNetV2_QuPath_2stage.py \
  --csv qupath_dataset.csv \
  --pretrained best_InceptionResNetV2_model.h5 \
  --output-dir qupath_results \
  --folds 5 --aug medium
```

### 5. K-fold 模型測試

```bash
python InceptionResNetV2_testing.py \
  --test_dir /path/to/labeled_patches \
  --model_dir .
```

## 專案結構（精簡）

```
Path_TIL/
├── TILscout.py              # 上游腳本
├── TILscout_edit.py         # 參數化推論管線
├── normalize_HnE.py
├── prepare_dataset_csv.py
├── InceptionResNetV2_QuPath_2stage.py
├── InceptionResNetV2_*.py   # 訓練 / CV / 測試
├── inference_*.py
├── aggregate_til_scores.py
├── path_til/
│   └── hnscc.py             # GroupCV、OOF 與 TIL 共用邏輯
├── scripts/
│   ├── make_hnscc_group_folds.py
│   ├── train_hnscc_groupcv_irv2.py
│   └── eval_hnscc_oof.py
├── tests/
│   └── test_hnscc_groupcv.py
├── dockerfile
├── LICENSE
└── README.md
```

執行時產生的 `results_*`、`qupath_results*`、`patch/`、日誌與大型資料檔已列於 `.gitignore`。

## TIL 分數計算

與 TILScout 相同：對每個 patch 以 InceptionResNetV2 分類為 Positive / Negative / Other；在 Positive 或 Negative 為主導類別且高於其餘類別時計入分子分母，最終

\[
\text{TIL score} = \frac{\text{Positive}}{\text{Positive} + \text{Negative}}
\]

詳見 `TILscout_edit.py` 內註解與上游說明。

## 引用與致謝

若使用本專案或 TILScout 方法，請引用 TILScout 原文與 [huibozh/TILScout](https://github.com/huibozh/TILScout)。

## 授權

- 本儲存庫新增程式碼：[MIT License](LICENSE)
- 上游 TILScout 程式與模型：請遵循其儲存庫之學術使用條款
