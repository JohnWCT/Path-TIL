# HNSCC Backbone Full 5-fold Report

> Updated: 2026-07-23

## Rule

Smoke / B5 fold-0+1 results are not candidate-level evidence.  
A backbone can replace IRV2 only after full 5-fold OOF **and** external lock-box confirmation, with all guardrails passed.

## Full 5-fold Results（HNSCC OOF）

| model | positive AUC | positive PRC | macro OVR AUC | weighted OVR AUC | decision |
|---|---:|---:|---:|---:|---|
| **IRV2 candidate** (0.50:0.50) | **0.8848** | 0.4196 | 0.9173 | 0.9288 | locked reference |
| EfficientNetV2-S full5 (`h4_more_tcga`) | 0.8792 | **0.4747** | **0.9222** | **0.9375** | drop（AUC 未超越） |
| ConvNeXt-Tiny full5 (`h6_low_lr`) | **0.8904** | **0.4858** | 0.9124 | 0.9203 | drop（macro OVR 下降） |

## External lock-box（B7，report-only）

| model | CPTAC_LUAD AUC | CPTAC_LUSC AUC | RUMC-BRCA AUC |
|---|---:|---:|---:|
| IRV2 candidate | 0.9886 | 0.9904 | 0.9972 |
| EfficientNetV2-S full5 | 0.9909 | 0.9986 | 0.9934 |
| ConvNeXt-Tiny full5 | 0.9875 | 0.9982 | 0.9916 |

External 未崩壞；但依規則**不可用 external 回頭改超參數**，也不能單獨用來取代 candidate。

## Interpretation

### Replace IRV2

- **none** — 沒有任何 backbone 同時滿足 positive AUC/PRC 提升 + macro/weighted guardrails。

### Pareto / positive-specialist（可保留作對照，不取代主候選）

- **ConvNeXt-Tiny full5**：positive AUC/PRC 高於 IRV2，但 macro OVR 略降。
- **EfficientNetV2-S full5**：PRC 與 multiclass OVR 優於 IRV2，但 positive AUC 略低。

### Drop as primary candidate

- 兩組都不取代 IRV2 + source mix 0.50:0.50。

## Artifacts

- `results/results_oof_with_prc/backbone_efficientnetv2_s_full5_selected/`
- `results/results_oof_with_prc/backbone_convnext_tiny_full5_selected/`
- `results/results_backbone_full5_comparison/backbone_candidate_comparison.csv`
- `results/results_external_testset_backbone_*_full5_selected/`
