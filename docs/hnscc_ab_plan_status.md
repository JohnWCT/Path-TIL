# HNSCC A/B Plan Execution Status

> Updated: 2026-07-22  
> Verified: all A1–A4 / B1–B4 / SUMMARY markers present; working tree clean after refresh.  

> Orchestrator log: `results/results_ab_plan_orchestrator/orchestrator.log`

## All phases complete

| Phase | Output | Result |
|---|---|---|
| B1 | TCGA manifests | OK |
| A1 | TCGA internal | AUC 0.9950 / PRC 0.9952 |
| A2 | External lock-box | AUC ≥ 0.9886（report only） |
| A3 | Seed 7 / 21 | mean AUC 0.8712 ± 0.0101 |
| A4 | L2-SP ×3 | all **drop** |
| B2 | EfficientNetV2-S pretrain | source val AUC 0.9999 |
| B3 | ConvNeXt-Tiny pretrain | source val AUC 0.9998 |
| B4 | Backbone smoke fold 0+1 | EfficientNet AUC 0.8983；ConvNeXt 0.9004 |
| SUMMARY | stability + comparison + reports | OK |

## Decision

- Keep IRV2 + source mix 0.50:0.50 as locked candidate.
- Backbone smoke: positive-specialist trend only; macro/weighted OVR dropped.
- Next: **B5 repair grid** (6 configs × 2 backbones, fold 0+1), then select ≤1 per backbone for B6 full5.

## B5 / B6 / B7 workflow

See:
- `docs/hnscc_backbone_b5_report.md`
- `docs/hnscc_backbone_full5_report.md`
- `docs/hnscc_backbone_decision_log.md`

```bash
# EfficientNetV2-S B5 grid
docker exec -w /workspace TIL python3 scripts/run_backbone_b5_grid.py \
  --configs configs/backbone_efficientnetv2_s_b5_h*.yaml \
  --folds 0 1 \
  --output-root results/results_backbone_b5_grid_efficientnetv2_s \
  --skip-existing

# ConvNeXt-Tiny B5 grid
docker exec -w /workspace TIL python3 scripts/run_backbone_b5_grid.py \
  --configs configs/backbone_convnext_tiny_b5_h*.yaml \
  --folds 0 1 \
  --output-root results/results_backbone_b5_grid_convnext_tiny \
  --skip-existing
```


## Artifacts

- `results/results_backbone_candidate_comparison/backbone_candidate_comparison.csv`
- `docs/hnscc_external_lockbox_report.md`
- `docs/hnscc_candidate_stability_report.md`

## Re-run

```bash
docker exec -w /workspace TIL python3 scripts/run_hnscc_ab_plan.py --skip-existing
```
