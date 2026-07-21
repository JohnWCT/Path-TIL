# HNSCC A/B Plan Execution Status

> Updated: 2026-07-21  
> Orchestrator log: `results/results_ab_plan_orchestrator/run.log`

## Completed

| Phase | Output | Notes |
|---|---|---|
| B1 | `tcga_train_dataset.csv`, `tcga_test_dataset.csv` | manifests |
| A1 | `results/results_tcga_internal_r50_50/` | positive AUC **0.9950** / PRC **0.9952** |
| A2 | `results/results_external_testset_r50_50/` | lock-box only; external AUCs ≥ 0.9886 |
| A3 seed7 | `results_method_source_mix_tcga_r50_50_seed7` + OOF | AUC 0.8604 / PRC 0.3905 |
| A3 seed21 | `results_method_source_mix_tcga_r50_50_seed21` + OOF | AUC 0.8685 / PRC 0.4056 |

### Seed stability (A3)

```text
seed42 (candidate): AUC 0.8848 / PRC 0.4196
seed7:              AUC 0.8604 / PRC 0.3905
seed21:             AUC 0.8685 / PRC 0.4056
mean_positive_auc: 0.8712  std: 0.0101
mean_positive_prc: 0.4052  std: 0.0119
```

Interpretation: candidate is moderately seed-sensitive (~0.01 AUC std). Report mean±std; do not treat single-seed 0.8848 as exact.

## In progress / pending

| Phase | Status | Notes |
|---|---|---|
| A4 L2-SP ×3 | running | name-keyed snapshot fix applied (stage2 shape mismatch) |
| B2 EfficientNetV2-S pretrain | pending | after A4 |
| B3 ConvNeXt-Tiny pretrain | pending | |
| B4 smoke fold 0+1 | pending | |
| SUMMARY | pending | |

## Fixes applied

1. Disabled Keras multiprocessing fit workers (OOM/SIGKILL on seed7 fold04).
2. Orchestrator resumes **per missing fold**.
3. L2-SP: snapshot source weights by normalized name; match pairs in Python before TF graph (fixes stage2 unfreeze shape error).

## Resume

```bash
docker exec -w /workspace TIL python3 scripts/run_hnscc_ab_plan.py --skip-existing
```
