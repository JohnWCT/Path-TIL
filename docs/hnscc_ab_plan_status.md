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

- **Keep** InceptionResNetV2 + source mix 0.50:0.50 as locked candidate.
- **Drop** all L2-SP λ settings tested.
- **Backbone smoke** shows higher fold-0+1 AUC/PRC but lower macro OVR AUC vs candidate → proceed to **5-fold only if** team wants to invest GPU; not auto-promoted.

## Artifacts

- `results/results_backbone_candidate_comparison/backbone_candidate_comparison.csv`
- `docs/hnscc_external_lockbox_report.md`
- `docs/hnscc_candidate_stability_report.md`

## Re-run

```bash
docker exec -w /workspace TIL python3 scripts/run_hnscc_ab_plan.py --skip-existing
```
