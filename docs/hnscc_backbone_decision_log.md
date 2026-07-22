# HNSCC Backbone Decision Log

> Updated: 2026-07-22

## Locked candidate

- Backbone: InceptionResNetV2
- Source mix: 0.50:0.50
- OOF positive AUC / PRC: 0.8848 / 0.4196
- macro / weighted OVR AUC: 0.9173 / 0.9288

## B4 smoke

- EfficientNetV2-S / ConvNeXt-Tiny: positive AUC/PRC ↑, macro/weighted OVR ↓
- Label: **positive-specialist trend**; not promoted

## B5 repair grid（完成）

12 / 12 fold-0+1 runs finished. **No config repaired macro/weighted OVR enough to reach `replace_candidate`.**  
All remain `positive_specialist_pending_full5`.

### Selected for B6（每 backbone 1 組）

| backbone | config | positive AUC | positive PRC | macro OVR | weighted OVR | fold AUC gap |
|---|---|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | `b5_h6_low_lr` | 0.9033 | 0.5708 | 0.9012 | 0.9035 | 0.0575 |
| EfficientNetV2-S | `b5_h4_more_tcga` | 0.9001 | 0.5541 | 0.9047 | 0.9102 | 0.0557 |

Artifacts:

- `results/results_backbone_b5_selection/backbone_b5_selection.csv`
- `results/results_backbone_b5_selection/selected_for_full5.txt`
- `docs/hnscc_backbone_b5_report.md`

## B6 / B7

- B6: full 5-fold for the two selected configs（academic / Pareto comparison; not auto-replace）
- B7: external lock-box only after B6; report-only
