# Quench2026 Pipeline Guide

This guide covers the new optimization workflow built on top of
`naspinn_baseline_with_quench_2026_data.py`.

## What is implemented

- Baseline training with Adam.
- Optional refinement stages from best Adam checkpoint:
  - L-BFGS
  - PSO (via `pymoo`)
- Architecture search methods:
  - NSGA-II
  - NSGA-III
  - Bayesian Optimization
- Resume-safe run state and extra logs so interrupted runs continue without restarting from zero.

## Entry points

- `python NAS-PINNS2/NAS_PINNs_quench_nsga2.py ...`
- `python NAS-PINNS2/NAS_PINNs_quench_nsga3.py ...`
- `python NAS-PINNS2/NAS_PINNs_quench_bayesian.py ...`
- `python NAS-PINNS2/NAS_PINNs_quench_pipeline.py ...` (runs all selected methods)

## Recommended full run

```bash
python NAS-PINNS2/NAS_PINNs_quench_pipeline.py \
  --methods nsga2,nsga3,bayesian \
  --epochs 5000 \
  --proxy-epochs 300 \
  --use-pso-final \
  --save-dir NAS-PINNS2/results/quench2026/pipeline
```

## Resume behavior

- Method-level state:
  - `<save_dir>/<method>/run_state.json`
- Pipeline-level state:
  - `<save_dir>/pipeline_state.json`
- Search cache (reused evaluations):
  - `<save_dir>/<method>/search_cache.json`
- Logs:
  - `<save_dir>/logs/pipeline.log`
  - `<save_dir>/<method>/logs/<method>.log`
  - Trial logs in `search_trials/*/train.log`
  - Final training log in `final/*/train.log`

If a run crashes or the process is killed, re-running the same command continues from the saved state.

## Core outputs

- Per-method:
  - `search_result.json`
  - `search_population.csv`
  - `method_summary.csv`
  - `final/<arch>/stage_summary.csv`
  - `final/<arch>/run_meta.json`
- Pipeline:
  - `comparison.csv`
  - `stage_comparison.csv`
