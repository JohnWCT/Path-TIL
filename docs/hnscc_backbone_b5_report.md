# HNSCC Backbone B5 Hyperparameter Repair Report

> Auto-updated by `scripts/update_backbone_report.py` at 2026-07-22 03:44 UTC.

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
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Decision

### promote to B6

- none

### positive-specialist only

- none

### drop

- none
