# HNSCC 實驗總結（Master Report）

> 2026-07-23 · 敘事唯一正文。詳細數字亦見 `results/results_master_stagewise_metrics/`。

## 1. 共通訓練與評估框架（只說明一次）

後續各章只寫**該階段相對此框架的差異**。

| 項目 | 共通設定 |
|---|---|
| 目標資料 | HNSCC patches（`dataset/QuPathOutput` → `qupath_dataset.csv`），三類 positive / negative / other |
| 切分 | case-level **GroupCV 5-fold**（同 case 不跨 train/val/test）。共 **10 cases**（H0002, H0003, H0005, H0006, H0007, H0008, H0013, H0018, H0041, H0103）；每 fold：**train 7** / **val 1** / **test 2**；每個 case 恰好當 held-out test 一次（`folds_hnscc_group5.csv`） |
| 訓練階段 | Stage1（較高 LR）→ Stage2（較低 LR）；推論 checkpoint 由 **validation-selected** 決定 |
| 評估 | HNSCC OOF patch-level OvR **AUROC / AUPRC**（pos / neg / other + macro / micro）；決策主看 positive，並看 neg/other 與 macro/micro |
| External | CPTAC_LUAD / LUSC、RUMC-BRCA 為 **report-only**，不回頭調參 |

**兩種「舊 IRV2 權重」對照（勿與 Stage 名稱混淆）**

| 權重檔 | 是什麼 | 對應本報告的結果代號 |
|---|---|---|
| **`baselines/best_InceptionResNetV2_model.h5`** | TILScout 公開預訓練 IRV2（未在本案 GroupCV 上 fine-tune 的起點） | Baseline 表的 **Stage0 pretrained** = 直接用此檔推論；**Stage1 / Stage2 / validation-selected** = 以此檔為初始化，再對 HNSCC GroupCV fine-tune 後的結果。§3 以降 IRV2 實驗同此起點。 |
| **`baselines/qupath_results_heavy/`**（例：`fold04_stage2_best.h5`） | 舊腳本 `InceptionResNetV2_QuPath_2stage.py`、**patch-level StratifiedKFold** 在 QuPath 上 train 出的 fold 權重（非本案 GroupCV） | **不在**現行 Baseline 四列裡。歷史對照名 **old_base**（`results/results_old_base_qupath_fold04_hne_on`）：固定載入 `qupath_results_heavy/fold04_stage2_best.h5`，只在 GroupCV held-out 上推論、**不再** fine-tune。pos AUROC ≈ **0.8912**（H&E on）。 |

**兩種預訓練來源（勿混淆）**

1. **IRV2 起點**：上表 `best_InceptionResNetV2_model.h5` → 再對 HNSCC GroupCV fine-tune（本報告主線）。  
2. **新 backbone source pretrain**：先在 TCGA **`dataset/train`** 訓練、**`dataset/test`** 驗證 → 再對 HNSCC（`QuPathOutput`）GroupCV fine-tune，train 可混 TCGA（source mix）。`dataset/Testset` 永不進訓練。

---

## 2. Baseline（H&E on / heavy / weight on）

**訓練流程（本階段）**

- Backbone：**IRV2**，載入舊預訓練權重。  
- 資料：僅 **HNSCC `QuPathOutput`**（**無** TCGA source mix）。  
- 前處理：H&E **on**、heavy aug、class weight **on**。  
- 比較：同一訓練 run 的 Stage0（未 fine-tune）/ Stage1 / Stage2 / validation-selected。

**結論**：fine-tune 優於 Stage0；Stage2 的 pos AUROC 最高，validation-selected 較均衡（後續仍採 selected）。

**表中 model 名稱意義**（同一 Baseline run 的不同 checkpoint；OOF 皆為各 fold held-out test）

| model | 意義 |
|---|---|
| **Stage0 pretrained** | **尚未**對該 fold 的 HNSCC train 做 fine-tune：直接用載入的舊 IRV2 權重推論（無本 fold 訓練、無 aug）。用來對照「只靠預訓練」的表現。 |
| **Stage1** | 兩階段 fine-tune 的 **第 1 階段**最佳權重（通常較高 LR、較多層凍結或較短 warmup 設定依訓練腳本）；依該 fold **validation** 指標存 `stage1_best`。 |
| **Stage2** | **第 2 階段**最佳權重（接續 Stage1、較低 LR 繼續調）；依該 fold **validation** 存 `stage2_best`。 |
| **validation-selected** | 每個 fold **只看 validation**（通常 validation multiclass / Keras AUC），在 Stage1 vs Stage2 中擇一作為該 fold 正式推論模型，再合併成 OOF。test 不參與選擇。 |

#### AUROC

| model | pos | neg | other | macro | micro |
|---|---:|---:|---:|---:|---:|
| Stage0 pretrained | 0.7815 | 0.8852 | 0.7654 | 0.8107 | 0.7965 |
| Stage1 | 0.8211 | 0.8722 | 0.8729 | 0.8554 | 0.8555 |
| Stage2 | 0.8512 | 0.8417 | 0.8508 | 0.8479 | 0.8486 |
| validation-selected | 0.8199 | 0.8589 | 0.8644 | 0.8477 | 0.8599 |

#### AUPRC

| model | pos | neg | other | macro | micro |
|---|---:|---:|---:|---:|---:|
| Stage0 pretrained | 0.2036 | 0.8399 | 0.7928 | 0.6121 | 0.6727 |
| Stage1 | 0.3119 | 0.8105 | 0.8587 | 0.6604 | 0.7607 |
| Stage2 | 0.3942 | 0.7314 | 0.8048 | 0.6435 | 0.7065 |
| validation-selected | 0.3644 | 0.8310 | 0.8111 | 0.6688 | 0.7480 |

---

## 3. Recipe（A1–A3）

**訓練流程（本階段）**

- 與 Baseline 相同：IRV2 + 舊預訓練 → **僅 HNSCC** GroupCV fine-tune（仍無 source mix）。  
- **差異**：掃 H&E on/off、heavy/medium、class weight on/off；其餘固定。

**結論**：鎖定 **H&E off + heavy + weight on**（pos 0.8555；neg/other/macro/micro 同步提升）。heavy ≫ medium；weight on 優於 off。

#### AUROC

| model | pos | neg | other | macro | micro |
|---|---:|---:|---:|---:|---:|
| H&E on / heavy / weight on | 0.8199 | 0.8589 | 0.8644 | 0.8477 | 0.8599 |
| **H&E off / heavy / weight on** | **0.8555** | **0.9082** | **0.9127** | **0.8922** | **0.9011** |
| H&E off / medium / weight on | 0.8020 | 0.8876 | 0.9050 | 0.8649 | 0.8831 |
| H&E off / heavy / weight off | 0.8226 | 0.9036 | 0.8947 | 0.8736 | 0.9100 |

#### AUPRC

| model | pos | neg | other | macro | micro |
|---|---:|---:|---:|---:|---:|
| H&E on / heavy / weight on | 0.3644 | 0.8310 | 0.8111 | 0.6688 | 0.7480 |
| **H&E off / heavy / weight on** | **0.3817** | **0.8757** | **0.9023** | **0.7199** | **0.8319** |
| H&E off / medium / weight on | 0.3132 | 0.8493 | 0.8973 | 0.6866 | 0.8066 |
| H&E off / heavy / weight off | 0.3419 | 0.8669 | 0.8760 | 0.6949 | 0.8404 |

---

## 4. Stage 選擇（B2）

**訓練流程（本階段）**

- 固定 Recipe：**H&E off / heavy / weight on**、IRV2、**僅 HNSCC**（無 source mix）。  
- **差異**：只改「用哪個 stage checkpoint 做 OOF」（fixed S1 / S2 / validation-selected / 其他 policy）。

**結論**：**validation-selected** 的 pos / macro / micro 最佳；保留 selected，不做 calibration。

#### AUROC

| model | pos | neg | other | macro | micro |
|---|---:|---:|---:|---:|---:|
| fixed Stage1 | 0.8136 | 0.8856 | 0.8800 | 0.8597 | 0.8728 |
| fixed Stage2 | 0.8508 | 0.9086 | 0.9078 | 0.8891 | 0.8940 |
| **validation-selected** | **0.8555** | **0.9082** | **0.9127** | **0.8922** | **0.9011** |
| validation positive AUC | 0.8495 | 0.8961 | 0.9017 | 0.8824 | 0.8933 |
| composite positive+macro | 0.8495 | 0.8961 | 0.9017 | 0.8824 | 0.8933 |

#### AUPRC

| model | pos | neg | other | macro | micro |
|---|---:|---:|---:|---:|---:|
| fixed Stage1 | 0.2846 | 0.8240 | 0.8718 | 0.6602 | 0.7891 |
| fixed Stage2 | 0.3655 | 0.8678 | 0.8995 | 0.7110 | 0.8175 |
| **validation-selected** | **0.3817** | **0.8757** | **0.9023** | **0.7199** | **0.8319** |
| validation positive AUC | 0.3762 | 0.8587 | 0.8816 | 0.7055 | 0.8157 |
| composite positive+macro | 0.3762 | 0.8587 | 0.8816 | 0.7055 | 0.8157 |

---

## 5. Methodology（loss / sampler / aug LOO）

**訓練流程（本階段）**

- 基準 = §3–§4 的 no-mix candidate（IRV2 + Recipe + selected，**純 HNSCC**）。  
- **差異**：只換 loss（focal / logit-adjusted）、sampler（balanced）或 heavy-aug 元件消融；仍無 source mix。

**結論**：相對 no-mix candidate **全部不如基準** → 全 drop。

#### AUROC

| model | pos | neg | other | macro | micro |
|---|---:|---:|---:|---:|---:|
| **no-mix candidate** | **0.8555** | **0.9082** | **0.9127** | **0.8922** | **0.9011** |
| focal γ=1 | 0.8463 | 0.8954 | 0.8839 | 0.8752 | 0.8999 |
| focal γ=2 | 0.8271 | 0.9074 | 0.9025 | 0.8790 | 0.9128 |
| logit-adjusted CE | 0.8338 | 0.9019 | 0.8925 | 0.8761 | 0.9053 |
| balanced sampler | 0.8210 | 0.8905 | 0.8858 | 0.8658 | 0.8614 |
| aug without geometric | 0.8317 | 0.8955 | 0.8841 | 0.8704 | 0.8972 |
| aug without cutout | 0.8302 | 0.8979 | 0.8883 | 0.8721 | 0.9020 |
| aug without blur/noise | 0.8175 | 0.9080 | 0.9018 | 0.8758 | 0.9087 |
| aug without HED | 0.8174 | 0.9006 | 0.8999 | 0.8727 | 0.9072 |

#### AUPRC

| model | pos | neg | other | macro | micro |
|---|---:|---:|---:|---:|---:|
| **no-mix candidate** | **0.3817** | **0.8757** | **0.9023** | **0.7199** | **0.8319** |
| focal γ=1 | 0.3279 | 0.7755 | 0.8822 | 0.6619 | 0.7984 |
| focal γ=2 | 0.3463 | 0.8743 | 0.8834 | 0.7013 | 0.8458 |
| logit-adjusted CE | 0.3380 | 0.8147 | 0.8844 | 0.6790 | 0.8182 |
| balanced sampler | 0.3318 | 0.8530 | 0.8632 | 0.6827 | 0.7704 |
| aug without geometric | 0.3080 | 0.7810 | 0.8820 | 0.6570 | 0.7966 |
| aug without cutout | 0.3419 | 0.7912 | 0.8776 | 0.6702 | 0.8045 |
| aug without blur/noise | 0.3409 | 0.8726 | 0.8826 | 0.6987 | 0.8387 |
| aug without HED | 0.3225 | 0.8173 | 0.8937 | 0.6778 | 0.8253 |

---

## 6. Source mix（IRV2）

**訓練流程（本階段）**

- Backbone：仍為 **IRV2** + 舊預訓練。  
- **差異（相對 §5）**：GroupCV train 改為 **HNSCC `QuPathOutput` + TCGA `dataset/train` 比例混入**（validation / held-out test 仍只有該 fold 的 HNSCC）。  
- 掃比例：0.90:0.10 / 0.75:0.25 / **0.50:0.50** / 0.25:0.75。

**結論**：鎖定 **0.50:0.50**（pos AUROC 最高 0.8848）。**0.25:0.75** 為 Pareto（pos AUPRC 更高）。

#### AUROC

| model | pos | neg | other | macro | micro |
|---|---:|---:|---:|---:|---:|
| HNSCC:TCGA 0.90:0.10 | 0.8606 | 0.9318 | 0.9260 | 0.9061 | 0.9336 |
| HNSCC:TCGA 0.75:0.25 | 0.8655 | 0.9319 | 0.9260 | 0.9078 | 0.9317 |
| **HNSCC:TCGA 0.50:0.50** | **0.8848** | **0.9358** | 0.9313 | **0.9173** | 0.9382 |
| HNSCC:TCGA 0.25:0.75 | 0.8809 | 0.9353 | **0.9330** | 0.9164 | **0.9396** |

#### AUPRC

| model | pos | neg | other | macro | micro |
|---|---:|---:|---:|---:|---:|
| HNSCC:TCGA 0.90:0.10 | 0.3829 | 0.8924 | 0.9194 | 0.7316 | 0.8805 |
| HNSCC:TCGA 0.75:0.25 | 0.3998 | 0.8862 | **0.9281** | 0.7381 | 0.8786 |
| **HNSCC:TCGA 0.50:0.50** | 0.4196 | 0.9032 | 0.9273 | 0.7501 | 0.8895 |
| HNSCC:TCGA 0.25:0.75 | **0.4503** | **0.9170** | 0.9206 | **0.7626** | **0.8923** |

---

## 7. Seed 穩定性（IRV2 0.50:0.50）

**訓練流程（本階段）**

- 與 §6 鎖定設定完全相同（IRV2 + Recipe + source mix **0.50:0.50**）。  
- **差異**：只改 random seed（42 / 7 / 21）。

**結論**：mean pos AUROC **0.8712 ± 0.0101**；可接受，維持 seed 42 為鎖定代表。

#### AUROC

| model | pos | neg | other | macro | micro |
|---|---:|---:|---:|---:|---:|
| seed 42 (locked) | 0.8848 | 0.9358 | 0.9313 | 0.9173 | 0.9382 |
| seed 7 | 0.8604 | 0.9265 | 0.9198 | 0.9022 | 0.9315 |
| seed 21 | 0.8685 | 0.9204 | 0.9169 | 0.9019 | 0.9303 |

#### AUPRC

| model | pos | neg | other | macro | micro |
|---|---:|---:|---:|---:|---:|
| seed 42 (locked) | 0.4196 | 0.9032 | 0.9273 | 0.7501 | 0.8895 |
| seed 7 | 0.3905 | 0.8998 | 0.9169 | 0.7357 | 0.8824 |
| seed 21 | 0.4056 | 0.8970 | 0.9103 | 0.7377 | 0.8796 |

---

## 8. L2-SP（相對 IRV2 0.50:0.50）

**訓練流程（本階段）**

- 與 §6 鎖定設定相同。  
- **差異**：Stage2 加 L2-SP（往 source／預訓練權重拉回），掃 λ。

**結論**：三個 λ **全部不如**無 L2-SP → drop。

#### AUROC

| model | pos | neg | other | macro | micro |
|---|---:|---:|---:|---:|---:|
| **IRV2 0.50:0.50 (no L2-SP)** | **0.8848** | **0.9358** | **0.9313** | **0.9173** | **0.9382** |
| L2-SP λ=1e-3 | 0.8628 | 0.9323 | 0.9262 | 0.9071 | 0.9348 |
| L2-SP λ=1e-4 | 0.8283 | 0.9211 | 0.9103 | 0.8866 | 0.9202 |
| L2-SP λ=1e-5 | 0.8319 | 0.9243 | 0.9140 | 0.8901 | 0.9229 |

#### AUPRC

| model | pos | neg | other | macro | micro |
|---|---:|---:|---:|---:|---:|
| **IRV2 0.50:0.50 (no L2-SP)** | **0.4196** | 0.9032 | **0.9273** | **0.7501** | **0.8895** |
| L2-SP λ=1e-3 | 0.4095 | **0.9055** | 0.9217 | 0.7456 | 0.8869 |
| L2-SP λ=1e-4 | 0.3540 | 0.8920 | 0.9074 | 0.7178 | 0.8667 |
| L2-SP λ=1e-5 | 0.3575 | 0.8937 | 0.9146 | 0.7219 | 0.8717 |

---

## 9. Backbone（EfficientNetV2-S / ConvNeXt-Tiny）

### 訓練流程（本階段，與 IRV2 路線不同）

```text
[Step A] Source pretrain（TCGA only）
  資料：dataset/train 訓練，dataset/test 驗證
  產出：baselines/source_pretrain_{efficientnetv2_s|convnext_tiny}/*.h5
  （不碰 QuPathOutput，不碰 dataset/Testset）

[Step B] HNSCC GroupCV fine-tune + source mix
  初始化：載入 Step A 的 source-pretrained 權重
  目標：dataset/QuPathOutput（HNSCC）GroupCV
  train 混入：TCGA dataset/train patches（比例依 config）
  配方：H&E off / heavy / class weight on / validation-selected
```

**是否已測 source mix？是。** B4 smoke、B5×12、B6 full5 的 EfficientNet / ConvNeXt **皆為「source pretrain → QuPathOutput fine-tune + source mix」**，不是只在 HNSCC 上從頭訓。

| full5 入圍設定 | mix（HNSCC:TCGA） | 其他差異 |
|---|---|---|
| EfficientNetV2-S `h4_more_tcga` | **0.25:0.75**（偏多 TCGA replay） | B5 入圍後跑滿 5-fold |
| ConvNeXt-Tiny `h6_low_lr` | **0.50:0.50** | Stage2 LR 較低 |
| IRV2（對照） | **0.50:0.50** | 舊 IRV2 權重，無 Step A 新 backbone pretrain |
| **old_base**（歷史對照） | —（無本案 GroupCV fine-tune / 無 source mix） | 固定 `baselines/qupath_results_heavy/fold04_stage2_best.h5`；H&E **on**；只在 GroupCV held-out 上推論 |

B5 亦含 EfficientNet **0.50:0.50**（如 `h1`）等設定；full5 最終選的是 **0.25:0.75** 的 EfficientNet（fold0+1 綜合較佳），**不是**未做 mix。

**結論（B6）**：依「pos AUROC 必須超越 IRV2」規則 **不取代**。EfficientNet 在 neg/other/macro/micro 與 pos AUPRC 較強，但 **pos AUROC 0.8792 < 0.8848**；ConvNeXt pos 較高但 macro/micro 與 neg/other 下降（positive-specialist）。**old_base** 的 pos/macro/micro 數字更高，但來自分割不同的舊 patch-level 訓練權重、且未經本案 GroupCV fine-tune，**不作主候選**（僅歷史對照）。

#### AUROC

| model | pos | neg | other | macro | micro |
|---|---:|---:|---:|---:|---:|
| old_base（`qupath_results_heavy` fold04_stage2，H&E on） | 0.8912 | 0.9544 | 0.9566 | 0.9341 | 0.9558 |
| **IRV2 0.50:0.50** | 0.8848 | 0.9358 | 0.9313 | 0.9173 | 0.9382 |
| EfficientNetV2-S full5（mix 0.25:0.75） | 0.8792 | **0.9447** | **0.9426** | **0.9222** | **0.9416** |
| ConvNeXt-Tiny full5（mix 0.50:0.50） | **0.8904** | 0.9243 | 0.9225 | 0.9124 | 0.9270 |

#### AUPRC

| model | pos | neg | other | macro | micro |
|---|---:|---:|---:|---:|---:|
| old_base（`qupath_results_heavy` fold04_stage2，H&E on） | 0.4605 | 0.9341 | 0.9582 | 0.7843 | 0.9241 |
| **IRV2 0.50:0.50** | 0.4196 | 0.9032 | 0.9273 | 0.7501 | 0.8895 |
| EfficientNetV2-S full5（mix 0.25:0.75） | 0.4747 | **0.9192** | **0.9465** | **0.7801** | **0.9024** |
| ConvNeXt-Tiny full5（mix 0.50:0.50） | **0.4858** | 0.8975 | 0.9165 | 0.7666 | 0.8712 |

### External（IRV2 lock-box，report-only）

| dataset | pos AUROC | pos AUPRC |
|---|---:|---:|
| TCGA internal | 0.9950 | 0.9952 |
| CPTAC_LUAD | 0.9886 | 0.8923 |
| CPTAC_LUSC | 0.9904 | 0.9748 |
| RUMC-BRCA | 0.9972 | 0.9629 |

---

## 10. 最終鎖定

```text
KEEP: IRV2 + H&E off + heavy + class weight on + source mix 0.50:0.50
OOF: pos AUROC 0.8848 / AUPRC 0.4196
     neg 0.9358 / other 0.9313
     macro AUROC 0.9173 / micro 0.9382
     macro AUPRC 0.7501 / micro 0.8895

NOTE: EfficientNetV2-S full5 已完成 source pretrain + QuPathOutput source-mix fine-tune
      （選定 mix 0.25:0.75）；multiclass/AUPRC 較佳，但 pos AUROC 未超越 → 不取代主候選。
DO NOT replace primary candidate with EfficientNet or ConvNeXt
```

指標再生：`docker exec -w /workspace TIL python3 scripts/compile_stagewise_multiclass_metrics.py`
