# HNSCC 方法學優化工作流程

## 1. 摘要

本輪依《Path-TIL 方法學優化 IDE 操作手冊》完成可重現實驗骨架，並以 **positive-vs-rest AUC／PRC** 為主要判斷完成完整 5-fold 比較。

候選參考（**未有實驗通過成功標準，維持不取代**）：

```text
H&E off + heavy + class weight on + validation multiclass AUC
positive AUC = 0.8555
positive PRC = 0.3817
hard TIL MAE = 0.1428（參考，不作 keep/drop）
```

成功標準（主要判斷 = AUC / PRC）：

```text
positive AUC > 0.8555
positive PRC > 0.3817
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
path_til/backbones/
scripts/train_hnscc_method.py
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
```

## 3. 最終比較結果（AUC／PRC）

完整表：[`hnscc_methodology_comparison_table.md`](hnscc_methodology_comparison_table.md)  
數值來源：`results_methodology_comparison_auc_prc_full/`、`results_oof_with_prc/`

| 方法 | Positive AUC | Positive PRC | 決策 |
|---|---:|---:|---|
| **candidate**（H&E off / heavy / weight on / val multiclass AUC） | **0.8555** | **0.3817** | reference |
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

1. **沒有任何方法同時提升 AUC 與 PRC**；candidate 維持最佳。
2. Stage policy／threshold TIL 在 ranking 指標上未優於 validation multiclass AUC。
3. Focal、logit-adjusted、balanced sampler 皆降低 AUC／PRC。
4. Heavy-aug leave-one-out（去掉 geometric／HED／blur-noise／cutout）皆變差，支持保留完整 heavy pipeline。
5. TIL MAE 僅作參考；本輪 keep/drop 不依賴 MAE。

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
| Source mix（TCGA） | 未做（缺 `tcga_train_dataset.csv`） |
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
  --pretrained best_InceptionResNetV2_model.h5 \
  --output-dir results_method_focal_gamma1
'
```

同樣可替換：

- `configs/method_focal_loss_gamma2.yaml`
- `configs/method_logit_adjusted_ce.yaml`
- `configs/method_balanced_sampler.yaml`

訓練後：

```bash
python3 scripts/eval_hnscc_oof.py \
  --pred-dir results_method_focal_gamma1 \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --stage selected \
  --output results_oof_with_prc/focal_gamma1
```

### Heavy aug leave-one-out

```bash
python3 scripts/ablate_heavy_augmentation.py \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --pretrained best_InceptionResNetV2_model.h5 \
  --output-root results_method_heavy_aug_ablation \
  --skip-full
```

### Stage policy／Threshold TIL

```bash
python3 scripts/compare_hnscc_stage_policies.py \
  --pred-dir results_groupcv_nohne_heavy \
  --output results_method_stage_policy

python3 scripts/tune_hnscc_thresholds.py \
  --pred-dir results_groupcv_nohne_heavy \
  --csv qupath_dataset.csv \
  --fold-csv folds_hnscc_group5.csv \
  --output results_oof_with_prc/threshold_til
```

### Source mix

需要先準備 `tcga_train_dataset.csv`。腳本會只把 TCGA patches 混入各 fold 的 train，不動 held-out HNSCC test。

### 統一比較

```bash
python3 scripts/compare_hnscc_methodology.py \
  --reference results_oof_with_prc/candidate_selected \
  --experiments \
    results_oof_with_prc/focal_gamma1 \
    results_oof_with_prc/focal_gamma2 \
    results_oof_with_prc/logit_adjusted_ce \
    results_oof_with_prc/balanced_sampler \
    results_oof_with_prc/aug_without_geometric \
    results_oof_with_prc/aug_without_hed \
    results_oof_with_prc/aug_without_blur_noise \
    results_oof_with_prc/aug_without_cutout \
    results_oof_with_prc/stage_fixed_stage2 \
    results_oof_with_prc/threshold_til \
  --output results_methodology_comparison_auc_prc_full
```

比較決策只看 AUC／PRC（與 macro／weighted OVR AUC 守門）；`hard_til_mae` 仍會寫入表內作參考。

## 5. 測試

```bash
python3 -m unittest discover -s tests -v
```

## 6. 後續可選

1. 準備 `tcga_train_dataset.csv` 後跑 source mix
2. L2-SP／backbone registry／EWC
3. 獨立外部 cohort 做 final confirmation（避免同一 OOF 上多次選模的 optimism）

只有通過成功標準（AUC／PRC）才取代目前 candidate。
