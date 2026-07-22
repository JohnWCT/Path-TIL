# HNSCC A/B Plan Execution Status

> Updated: 2026-07-22  
> Orchestrator log: `results/results_ab_plan_orchestrator/`

## Completed

| Phase | Output | Result |
|---|---|---|
| B1–B4 / A1–A4 / SUMMARY | 見先前報告 | OK |
| **B5** EfficientNet ×6 | `results_backbone_b5_grid_efficientnetv2_s/` | OK |
| **B5** ConvNeXt ×6 | `results_backbone_b5_grid_convnext_tiny/` | OK |
| **B5 selection** | `results_backbone_b5_selection/` | 2 configs → B6 |

## B5 conclusion

- 12/12 完成；**無人達到 `replace_candidate`**（macro/weighted OVR 仍低於 IRV2）。
- 入圍 B6：
  - ConvNeXt-Tiny `h6_low_lr`
  - EfficientNetV2-S `h4_more_tcga`

## In progress / next

| Phase | Status |
|---|---|
| B6 full 5-fold（入圍 2 組） | starting |
| B7 external lock-box | after B6 |

## Decision

- **Keep** IRV2 + source mix 0.50:0.50 as locked candidate.
- B5: positive-specialist only; proceed to B6 for confirmation, not auto-replace.

## Artifacts

- `results/results_backbone_b5_selection/backbone_b5_selection.csv`
- `docs/hnscc_backbone_b5_report.md`
- `docs/hnscc_backbone_decision_log.md`
