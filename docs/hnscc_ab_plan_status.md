# HNSCC A/B Plan Execution Status

> Auto-maintained status for Plan A (robustness) + Plan B (backbone).  
> Orchestrator log: `results/results_ab_plan_orchestrator/run.log`

## Completed

| Phase | Output | Notes |
|---|---|---|
| B1 | `tcga_train_dataset.csv`, `tcga_test_dataset.csv` | manifests |
| A1 | `results/results_tcga_internal_r50_50/` | positive AUC 0.9950 / PRC 0.9952 |
| A2 | `results/results_external_testset_r50_50/` | lock-box only; see external report |

## In progress / pending

| Phase | Status | Notes |
|---|---|---|
| A3 seed7 | fold 0–3 done; fold 4 training | resumed after SIGKILL (multiprocessing OOM) |
| A3 seed21 | pending | after seed7 OOF |
| A4 L2-SP ×3 | pending | |
| B2 EfficientNet pretrain | pending | |
| B3 ConvNeXt pretrain | pending | |
| B4 smoke fold 0+1 | pending | |
| SUMMARY | pending | stability + backbone compare |

## Resume command (Docker)

```bash
docker exec -w /workspace TIL python3 scripts/run_hnscc_ab_plan.py --skip-existing
```

## Fix applied (2026-07-20)

- Disabled Keras `multiprocessing` fit workers (OOM on fold 4).
- Orchestrator resumes **per missing fold** instead of restarting full 5-fold runs.
