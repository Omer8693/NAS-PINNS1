#!/usr/bin/env bash
set -euo pipefail

# Example improved baseline run (separate from original scripts/results).
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
python naspinn_baseline_with_quench_2026_data_improved.py \
  --save-dir results/quench2026/improved_baseline/L5_N96 \
  --layers 5 \
  --base-neurons 96 \
  --epochs 3000 \
  --phase1-epochs 1200 \
  --lbfgs-max-iter 500 \
  --use-pso \
  --pso-iters 8 \
  --pso-swarm 16 \
  --force-final

# Improved physical metrics/heatmaps (skips baseline delta files by default).
python quench2026_physical_heatmaps_improved.py \
  --results-root results/quench2026 \
  --out-dir results/quench2026/report_pack_improved/physical_heatmaps

# Improved report pack (panelized runtime plot).
python quench2026_report_pack_improved.py \
  --results-root results/quench2026 \
  --out-dir results/quench2026/report_pack_improved

