# HNSCC A/B Plan Execution Status

> Updated: 2026-07-23  
> All planned A/B + B5/B6/B7 phases complete.

## Completed

| Phase | Result |
|---|---|
| A1–A4 / B1–B4 | OK（先前完成） |
| B5 repair grid ×12 | OK；無人 replace |
| B6 full5（入圍 2 組） | OK |
| B7 external lock-box | OK（report-only） |
| Reports + comparison | OK |

## Final decision

**Keep IRV2 + source mix 0.50:0.50 as locked candidate.**

| model | HNSCC AUC | HNSCC PRC | macro OVR | decision |
|---|---:|---:|---:|---|
| IRV2 candidate | 0.8848 | 0.4196 | 0.9173 | **locked** |
| EfficientNet full5 | 0.8792 | 0.4747 | 0.9222 | drop as primary |
| ConvNeXt full5 | 0.8904 | 0.4858 | 0.9124 | specialist / not replace |

## Key artifacts

- `docs/hnscc_backbone_b5_report.md`
- `docs/hnscc_backbone_full5_report.md`
- `docs/hnscc_backbone_decision_log.md`
- `results/results_backbone_full5_comparison/backbone_candidate_comparison.csv`
- `results/results_backbone_b5_selection/`
