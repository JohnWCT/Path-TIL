# HNSCC 方法學優化工作流程

## 1. 摘要

本輪依《Path-TIL 方法學優化 IDE 操作手冊》完成可重現實驗骨架，並以 **positive-vs-rest AUC／PRC** 為主要判斷完成完整 5-fold 比較。

**目前最佳候選（已由 Source mix 取代舊設定）**：

```text
H&E off + heavy + class weight on + validation multiclass AUC
+ Source mix：各 fold train 混入 TCGA `dataset/train` patches（HNSCC:TCGA ≈ 0.75:0.25）
val／test 仍為純 HNSCC
positive AUC = 0.8655
positive PRC = 0.3998
hard TIL MAE = 0.1678（參考，不作 keep/drop）
輸出：results/results_method_source_mix_tcga/
OOF：results/results_oof_with_prc/source_mix_tcga/
```

舊參考（已被取代）：

```text
H&E off + heavy + class weight on（無 source mix）
positive AUC = 0.8555
positive PRC = 0.3817
hard TIL MAE = 0.1428（參考）
```

成功標準（主要判斷 = AUC / PRC；相對當時比較參考）：

```text
positive AUC > 參考
positive PRC > 參考
macro OVR AUC 不下降
weighted OVR AUC 不明顯下降（< 0.01）
```

說明：`dataset/QuPathOutput` 並非每張 WSI 都完整 patching，slide TIL 分母不完整，因此 **hard/soft TIL MAE 只作參考診斷**，不納入 keep/drop。

## 2. 新增結構

```text
configs/method_*.yaml
path_til/losses.py
path_til/samplers.py
path_til/stage_selection.py
path_til/til_threshold.py
path_til/experiment_registry.py
path_til/paths.py
path_til/backbones/
scripts/train_hnscc_method.py
scripts/organize_workspace.py
scripts/prepare_tcga_train_csv.py
scripts/compare_hnscc_stage_policies.py
scripts/tune_hnscc_thresholds.py
scripts/ablate_heavy_augmentation.py
scripts/train_hnscc_source_mix.py
scripts/compare_hnscc_methodology.py
scripts/summarize_methodology_table.py
tests/test_losses.py
tests/test_samplers.py
tests/test_stage_selection.py
tests/test_til_threshold.py
tests/test_methodology_registry.py
tests/test_organize_workspace.py
```

## 3. 最終比較結果（AUC／PRC）

完整表：[`hnscc_methodology_comparison_table.md`](hnscc_methodology_comparison_table.md)  
數值來源：`results/results_methodology_comparison_auc_prc_full/`、`results/results_oof_with_prc/`

本表以舊 candidate（無 source mix）為比較參考；Source mix 為唯一通過 keep 標準者。

| 方法 | Positive AUC | Positive PRC | 決策 |
|---|---:|---:|---|
| **source_mix_tcga**（train 混 TCGA；val/test 純 HNSCC） | **0.8655** | **0.3998** | **keep → 新候選** |
| candidate（H&E off / heavy / weight on / val multiclass AUC） | 0.8555 | 0.3817 | 舊 reference（已被取代） |
| threshold TIL（validation-tuned） | 0.8555 | 0.3817 | drop（未嚴格優於） |
| fixed Stage 2 | 0.8508 | 0.3655 | drop |
| focal γ=1 | 0.8463 | 0.3279 | drop |
| logit-adjusted CE | 0.8338 | 0.3380 | drop |
| leave-one-out：without geometric | 0.8317 | 0.3080 | drop |
| leave-one-out：without cutout | 0.8302 | 0.3419 | drop |
| focal γ=2 | 0.8271 | 0.3463 | drop |
| balanced sampler | 0.8210 | 0.3318 | drop |
| leave-one-out：without blur/noise | 0.8175 | 0.3409 | drop |
| leave-one-out：without HED | 0.8174 | 0.3225 | drop |

解讀：

1. **Source mix 同時提升 AUC（+0.0100）與 PRC（+0.0181）**，且 macro／weighted OVR AUC 上升，為本輪唯一 keep。
2. Stage policy／threshold TIL 在 ranking 指標上未優於 validation multiclass AUC。
3. Focal、logit-adjusted、balanced sampler 皆降低 AUC／PRC。
4. Heavy-aug leave-one-out（去掉 geometric／HED／blur-noise／cutout）皆變差，支持保留完整 heavy pipeline。
5. Source mix 的 hard TIL MAE（0.1678）略高於舊候選（0.1428），但 MAE 僅參考、不作 keep/drop。
6. 後續比較請以 Source mix 為新 reference（見 `path_til/experiment_registry.py` 的 `CANDIDATE_REFERENCE`）。

### 完成確認

| 項目 | 狀態 |
|---|---|
| Stage policy 離線比較 | 完成 |
| Threshold TIL | 完成 |
| Focal γ=1 / γ=2 完整 5-fold | 完成 |
| Logit-adjusted CE 完整 5-fold | 完成 |
| Balanced sampler 完整 5-fold | 完成 |
| Heavy aug component ablation（`--disable-aug-component`） | 完成 |
| AUC／PRC 統一比較表 | 完成 |
| Source mix（TCGA `dataset/train`） | **完成（keep；已升為新候選）** |
| L2-SP / backbone / EWC | 未做 |

## 4. 執行方式

### Loss / sampler 訓練

```bash
docker exec TIL bash -lc '
cd /workspace
python3 scripts/train_hnscc_method.py \
  --config configs/method_focal_loss_gamma1.yaml \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --pretrained baselines/best_InceptionResNetV2_model.h5 \
  --output-dir results/results_method_focal_gamma1
'
```

同樣可替換：

- `configs/method_focal_loss_gamma2.yaml`
- `configs/method_logit_adjusted_ce.yaml`
- `configs/method_balanced_sampler.yaml`

訓練後：

```bash
python3 scripts/eval_hnscc_oof.py \
  --pred-dir results/results_method_focal_gamma1 \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --stage selected \
  --output results/results_oof_with_prc/focal_gamma1
```

### Heavy aug leave-one-out

```bash
python3 scripts/ablate_heavy_augmentation.py \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --pretrained baselines/best_InceptionResNetV2_model.h5 \
  --output-root results/results_method_heavy_aug_ablation \
  --skip-full
```

### Stage policy／Threshold TIL

```bash
python3 scripts/compare_hnscc_stage_policies.py \
  --pred-dir results/results_groupcv_nohne_heavy \
  --output results/results_method_stage_policy

python3 scripts/tune_hnscc_thresholds.py \
  --pred-dir results/results_groupcv_nohne_heavy \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --output results/results_oof_with_prc/threshold_til
```

### Source mix

1. 先準備 TCGA train manifest（來自預訓練來源 `dataset/train`）：

```bash
python3 scripts/prepare_tcga_train_csv.py \
  --train-root dataset/train \
  --output tcga_train_dataset.csv \
  --rewrite-prefix /workspace
```

2. 訓練（只混入各 fold 的 train；held-out HNSCC val/test 不變）：

```bash
python3 scripts/train_hnscc_source_mix.py \
  --config configs/method_source_mix_tcga.yaml \
  --csv-hnscc qupath_dataset.csv \
  --csv-tcga tcga_train_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --pretrained baselines/best_InceptionResNetV2_model.h5 \
  --output-dir results/results_method_source_mix_tcga
```

3. OOF：

```bash
python3 scripts/eval_hnscc_oof.py \
  --pred-dir results/results_method_source_mix_tcga \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --stage selected \
  --output results/results_oof_with_prc/source_mix_tcga
```

注意：`dataset/test` 是 TCGA 內部 holdout；`dataset/Testset` 是外部 CPTAC/RUMC 評估集，**不**混入訓練。

### 統一比較

```bash
python3 scripts/compare_hnscc_methodology.py \
  --reference results/results_oof_with_prc/candidate_selected \
  --experiments \
    results/results_oof_with_prc/focal_gamma1 \
    results/results_oof_with_prc/focal_gamma2 \
    results/results_oof_with_prc/logit_adjusted_ce \
    results/results_oof_with_prc/balanced_sampler \
    results/results_oof_with_prc/source_mix_tcga \
    results/results_oof_with_prc/aug_without_geometric \
    results/results_oof_with_prc/aug_without_hed \
    results/results_oof_with_prc/aug_without_blur_noise \
    results/results_oof_with_prc/aug_without_cutout \
    results/results_oof_with_prc/stage_fixed_stage2 \
    results/results_oof_with_prc/threshold_til \
  --output results/results_methodology_comparison_auc_prc_full
```

比較決策只看 AUC／PRC（與 macro／weighted OVR AUC 守門）；`hard_til_mae` 仍會寫入表內作參考。

## 5. 測試

```bash
docker exec -w /workspace TIL python3 -m unittest discover -s tests -v
```

工作區整理（預設 dry-run）：

```bash
docker exec TIL python3 /workspace/scripts/organize_workspace.py --dry-run
```

## 6. 後續可選

1. 以 Source mix 為新 reference，做獨立外部 cohort（`dataset/Testset`）final confirmation
2. L2-SP／backbone registry／EWC
3. 人工檢查 high-confidence false positives（見 GroupCV 報告）

只有通過成功標準（AUC／PRC）才取代目前 candidate。
