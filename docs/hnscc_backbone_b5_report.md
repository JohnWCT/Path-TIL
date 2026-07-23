# HNSCC Backbone B5 Hyperparameter Repair Report

> Auto-updated by `scripts/update_backbone_report.py` at 2026-07-23 01:28 UTC.

## Reference

IRV2 + source mix 0.50:0.50:

- positive AUC = 0.8848
- positive PRC = 0.4196
- macro OVR AUC = 0.9173
- weighted OVR AUC = 0.9288

## Purpose

B4 smoke showed EfficientNetV2-S and ConvNeXt-Tiny improved positive AUC / PRC
but decreased macro / weighted OVR AUC. B5 tests whether small hyperparameter
changes can preserve positive gains while repairing multiclass degradation.

## Experiments

| backbone | config | positive AUC | positive PRC | macro OVR AUC | weighted OVR AUC | fold gap | decision |
|---|---|---:|---:|---:|---:|---:|---|
| convnext_tiny | backbone_convnext_tiny_b5_h6_low_lr | 0.9033 | 0.5708 | 0.9012 | 0.9035 | 0.0575 | positive_specialist_pending_full5 |
| convnext_tiny | backbone_convnext_tiny_b5_h5_macro_stage | 0.9024 | 0.5783 | 0.8990 | 0.9017 | 0.0587 | positive_specialist_pending_full5 |
| convnext_tiny | backbone_convnext_tiny_b5_h3_lower_pos_weight | 0.9015 | 0.5546 | 0.9012 | 0.9017 | 0.0576 | positive_specialist_pending_full5 |
| convnext_tiny | backbone_convnext_tiny_b5_h1 | 0.9013 | 0.5694 | 0.9019 | 0.9036 | 0.0548 | positive_specialist_pending_full5 |
| efficientnetv2_s | backbone_efficientnetv2_s_b5_h4_more_tcga | 0.9001 | 0.5541 | 0.9047 | 0.9102 | 0.0557 | positive_specialist_pending_full5 |
| efficientnetv2_s | backbone_efficientnetv2_s_b5_h1 | 0.8983 | 0.5443 | 0.9002 | 0.9054 | 0.0741 | positive_specialist_pending_full5 |
| efficientnetv2_s | backbone_efficientnetv2_s_b5_h5_macro_stage | 0.8980 | 0.5462 | 0.9007 | 0.9058 | 0.0742 | positive_specialist_pending_full5 |
| efficientnetv2_s | backbone_efficientnetv2_s_b5_h6_low_lr | 0.8977 | 0.5424 | 0.8998 | 0.9052 | 0.0708 | positive_specialist_pending_full5 |
| efficientnetv2_s | backbone_efficientnetv2_s_b5_h3_lower_pos_weight | 0.8957 | 0.5344 | 0.9001 | 0.9054 | 0.0695 | positive_specialist_pending_full5 |
| convnext_tiny | backbone_convnext_tiny_b5_h4_more_tcga | 0.8931 | 0.5400 | 0.9011 | 0.9029 | 0.0669 | positive_specialist_pending_full5 |
| convnext_tiny | backbone_convnext_tiny_b5_h2_label_smoothing | 0.8913 | 0.5474 | 0.8973 | 0.9012 | 0.0591 | positive_specialist_pending_full5 |
| efficientnetv2_s | backbone_efficientnetv2_s_b5_h2_label_smoothing | 0.8872 | 0.5474 | 0.9009 | 0.9068 | 0.0893 | positive_specialist_pending_full5 |

## Decision

### Selected for B6 (≤1 per backbone)

- backbone_convnext_tiny_b5_h6_low_lr
- backbone_efficientnetv2_s_b5_h4_more_tcga

### positive-specialist (not auto-replace)

- backbone_convnext_tiny_b5_h6_low_lr
- backbone_convnext_tiny_b5_h5_macro_stage
- backbone_convnext_tiny_b5_h3_lower_pos_weight
- backbone_convnext_tiny_b5_h1
- backbone_efficientnetv2_s_b5_h4_more_tcga
- backbone_efficientnetv2_s_b5_h1
- backbone_efficientnetv2_s_b5_h5_macro_stage
- backbone_efficientnetv2_s_b5_h6_low_lr
- backbone_efficientnetv2_s_b5_h3_lower_pos_weight
- backbone_convnext_tiny_b5_h4_more_tcga
- backbone_convnext_tiny_b5_h2_label_smoothing
- backbone_efficientnetv2_s_b5_h2_label_smoothing

### drop

- none

### Interpretation

- **B5 無法修復 macro / weighted OVR**：12 組皆為 `positive_specialist_pending_full5`（無 `replace_candidate`）。
- 入圍 B6：`h6_low_lr`（ConvNeXt）、`h4_more_tcga`（EfficientNet）。
- B6/B7 結論見 [`hnscc_backbone_full5_report.md`](hnscc_backbone_full5_report.md)：兩組皆不取代 IRV2 candidate。
