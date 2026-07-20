# HNSCC 方法學優化工作流程

## 1. 摘要

本輪依《Path-TIL 方法學優化 IDE 操作手冊》完成可重現實驗骨架，並以 **positive-vs-rest AUC／PRC** 為主要判斷完成完整 5-fold 比較。

**目前最佳候選（Source mix 比例 ablation 後更新）**：

```text
H&E off + heavy + class weight on + validation multiclass AUC
+ Source mix：HNSCC:TCGA = 0.50:0.50（各 fold train；val/test 純 HNSCC）
positive AUC = 0.8848
positive PRC = 0.4196
hard TIL MAE = 0.1781（參考，不作 keep/drop）
輸出：results/results_method_source_mix_tcga_r50_50/
OOF：results/results_oof_with_prc/source_mix_tcga_r50_50/
config：configs/method_source_mix_tcga_r50_50.yaml
```

先前候選（已被 0.50:0.50 取代）：

```text
Source mix 0.75:0.25 → AUC 0.8655 / PRC 0.3998
no-mix（H&E off / heavy / weight on）→ AUC 0.8555 / PRC 0.3817
```

Pareto 備選（相對 0.75:0.25 亦 keep，但 AUC 低於 0.50:0.50）：

```text
Source mix 0.25:0.75 → AUC 0.8809 / PRC 0.4503（最高 PRC）
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
path_til/scoreboard.py
path_til/candidate.py
path_til/l2sp.py
path_til/external_eval.py
path_til/backbone_registry.py
path_til/model_factory.py
path_til/source_pretrain.py
path_til/backbones/
scripts/train_hnscc_method.py
scripts/eval_tcga_internal.py
scripts/eval_external_testset.py
scripts/train_hnscc_l2sp.py
scripts/pretrain_source_backbone.py
scripts/train_hnscc_backbone_source_mix.py
scripts/compare_backbone_and_candidate.py
scripts/summarize_candidate_stability.py
scripts/prepare_labeled_patch_csv.py
scripts/organize_workspace.py
scripts/prepare_tcga_train_csv.py
scripts/build_hnscc_scoreboard.py
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
tests/test_scoreboard.py
tests/test_l2sp.py
tests/test_external_eval.py
tests/test_candidate_config.py
tests/test_backbone_registry.py
tests/test_model_factory.py
tests/test_source_pretrain_config.py
tests/test_no_lockbox_leakage.py
docs/hnscc_external_lockbox_report.md
docs/hnscc_candidate_stability_report.md
```

## 3. 最終比較結果（AUC／PRC）

**Living scoreboard（分主題、持續更新）：[`hnscc_living_scoreboard.md`](hnscc_living_scoreboard.md)**  
登錄：`configs/scoreboard_experiments.yaml`  
再生（Docker TIL）：

```bash
docker exec -w /workspace TIL python3 scripts/build_hnscc_scoreboard.py \
  --registry configs/scoreboard_experiments.yaml \
  --results-root results \
  --output-md docs/hnscc_living_scoreboard.md \
  --output-csv results/results_methodology_comparison_scoreboard/scoreboard.csv
```

方法學摘要表：[`hnscc_methodology_comparison_table.md`](hnscc_methodology_comparison_table.md)（指向 scoreboard）  
敘事報告：[`hnscc_groupcv_optimization_report.md`](hnscc_groupcv_optimization_report.md)

本表以舊 no-mix candidate 為比較參考（方法學全實驗）。Source mix 比例細節見下節。

| 方法 | Positive AUC | Positive PRC | 決策 |
|---|---:|---:|---|
| **source_mix 0.50:0.50** | **0.8848** | **0.4196** | **keep → 目前候選** |
| source_mix 0.25:0.75 | 0.8809 | 0.4503 | keep（Pareto；最高 PRC） |
| source_mix 0.75:0.25 | 0.8655 | 0.3998 | 曾 keep；現為比例 ablation reference |
| source_mix 0.90:0.10 | 0.8606 | 0.3829 | drop（相對 0.75:0.25） |
| candidate no-mix | 0.8555 | 0.3817 | 舊 reference |
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

1. **Source mix 0.50:0.50 為目前最佳**（最高 AUC，且 PRC 優於 0.75:0.25／no-mix）。
2. **0.25:0.75 為 PRC 備選**；與 0.50:0.50 互不嚴格支配。
3. Stage policy／threshold TIL、focal、logit-adjusted、balanced sampler、heavy-aug leave-one-out 皆未優於當時 no-mix 候選。
4. TIL MAE 僅參考；本輪 keep/drop 不依賴 MAE。
5. 後續比較請以 `CANDIDATE_REFERENCE`（0.50:0.50）為準。

### Source mix：不同 TCGA 混入量比較

數值來源：`results/results_methodology_comparison_source_mix_ratios/`  
比較參考：先前候選 **0.75:0.25**（AUC 0.8655／PRC 0.3998）。

| 設定 | HNSCC:TCGA | TCGA 張數 | Positive AUC | Positive PRC | ΔAUC | ΔPRC | 決策 |
|---|---:|---:|---:|---:|---:|---:|---|
| no-mix（更早候選） | 1.00 : 0.00 | 0 | 0.8555 | 0.3817 | — | — | 已被 source mix 取代 |
| source_mix 輕量 | 0.90 : 0.10 | 610 | 0.8606 | 0.3829 | −0.0050 | −0.0170 | **drop**（低於 0.75:0.25） |
| source_mix（先前候選） | 0.75 : 0.25 | 1831 | 0.8655 | 0.3998 | 0 | 0 | 本表 reference |
| **source_mix 半量** | **0.50 : 0.50** | **5492** | **0.8848** | **0.4196** | **+0.0192** | **+0.0197** | **keep → 新候選**（最高 AUC） |
| source_mix 偏 TCGA | 0.25 : 0.75 | 16476 | 0.8809 | 0.4503 | +0.0153 | +0.0505 | **keep**（最高 PRC；Pareto 備選） |

解讀：

1. **0.10 TCGA 不足**：相對 0.25 略差，不能取代 0.75:0.25。
2. **0.50 與 0.75 皆 keep**：同時提升 AUC 與 PRC；兩者互不嚴格支配（0.50 較高 AUC，0.75 較高 PRC）。
3. **新候選取 0.50:0.50**：以最高 positive AUC 為主，並仍提升 PRC；輸出目錄見上。
4. 若應用更在意 rare-positive 的 PRC，可改用 **0.25:0.75**（需外部 cohort 再確認）。
5. 混入量愈大 hard TIL MAE 略升（僅參考）；選模仍只看 AUC／PRC。

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
| Source mix（TCGA `dataset/train`，0.75:0.25） | 完成（曾 keep；現為比例表 reference） |
| Source mix 多比例（0.9:0.1／0.5:0.5／0.25:0.75） | **完成** |
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

## 6. 後續路線（A 線穩健優化 + B 線 backbone 替換）

已實作 IDE 手冊骨架；**所有新實驗仍以 Source mix 0.50:0.50 為 reference**（AUC 0.8848／PRC 0.4196）。

### A 線：維持 IRV2 主幹

| 步驟 | 腳本 | 說明 |
|---|---|---|
| A1 | `scripts/eval_tcga_internal.py` | `dataset/test` TCGA internal holdout |
| A2 | `scripts/eval_external_testset.py` | `dataset/Testset` lock-box（**只報告、不調參**） |
| A3 | `scripts/train_hnscc_source_mix.py` + seed configs | seed 7／21 穩定性 |
| A4 | `scripts/train_hnscc_l2sp.py` | L2-SP λ = 1e-5／1e-4／1e-3 |
| 彙整 | `scripts/summarize_candidate_stability.py` | seed mean±std |

報告模板：[`hnscc_external_lockbox_report.md`](hnscc_external_lockbox_report.md)、[`hnscc_candidate_stability_report.md`](hnscc_candidate_stability_report.md)

```bash
# A1–A2（需已訓練 results/results_method_source_mix_tcga_r50_50）
docker exec -w /workspace TIL python3 scripts/eval_tcga_internal.py \
  --model-dir results/results_method_source_mix_tcga_r50_50 \
  --test-root dataset/test \
  --output-dir results/results_tcga_internal_r50_50

docker exec -w /workspace TIL python3 scripts/eval_external_testset.py \
  --model-dir results/results_method_source_mix_tcga_r50_50 \
  --testset-root dataset/Testset \
  --output-dir results/results_external_testset_r50_50
```

### B 線：backbone 替換（EfficientNetV2-S、ConvNeXt-Tiny 優先）

| 步驟 | 腳本 | 說明 |
|---|---|---|
| B1 | `scripts/prepare_labeled_patch_csv.py` | `tcga_train_dataset.csv`／`tcga_test_dataset.csv` |
| B2–B3 | `scripts/pretrain_source_backbone.py` | `dataset/train` 訓練、`dataset/test` 驗證 |
| B4 | `scripts/train_hnscc_backbone_source_mix.py` | fold 0+1 smoke（0.50:0.50 mix） |
| B5–B7 | hyperparam configs + 5-fold + external | 僅入圍 backbone |
| 比較 | `scripts/compare_backbone_and_candidate.py` | 統一 decision 表 |

Swin-Tiny 保留 placeholder config；第一輪以 TensorFlow backbone 為主。

### 新增模組（本 commit）

```text
path_til/candidate.py, l2sp.py, external_eval.py, backbone_registry.py, model_factory.py, source_pretrain.py
scripts/eval_tcga_internal.py, eval_external_testset.py, train_hnscc_l2sp.py, pretrain_source_backbone.py, ...
configs/method_l2sp_*.yaml, method_source_mix_tcga_r50_50_seed*.yaml, source_pretrain_*.yaml
tests/test_l2sp.py, test_external_eval.py, test_no_lockbox_leakage.py, ...
```

EWC 排在 A 線 + B 線 backbone 篩選之後；僅在 source-domain 明顯下降且 L2-SP 無效時再考慮。

只有通過成功標準（AUC／PRC + TCGA／external 不明顯崩壞）才取代目前 candidate。
