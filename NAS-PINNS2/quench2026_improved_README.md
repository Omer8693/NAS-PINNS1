# Quench2026 Improved (Safe Copy)

This variant keeps original scripts unchanged and applies improvements in separate files.

## New files
- `naspinn_baseline_with_quench_2026_data_improved.py`
- `quench2026_physical_heatmaps_improved.py`
- `quench2026_report_pack_improved.py`
- `run_quench2026_improved.sh`

## What is improved
- Temperature/displacement data loss is normalized to reference statistics.
- Adam training uses 2 phases (data-focused warmup + balanced phase).
- Stage selection uses composite score:
  - `score = objective + w_temp * normalized_temp_mae + w_disp * normalized_disp_mae`
- Physical comparison keeps fair displacement scope (`x=0` paper reference points).
- Delta heatmaps are generated against baseline; baseline self-delta files are skipped by default.
- Runtime-vs-objective plot is panelized by group.

## Quick run
```bash
cd NAS-PINNS2
./run_quench2026_improved.sh
```

## Manual run examples
```bash
cd NAS-PINNS2
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python naspinn_baseline_with_quench_2026_data_improved.py \
  --save-dir results/quench2026/improved_baseline/L5_N96 \
  --layers 5 --base-neurons 96 \
  --epochs 3000 --phase1-epochs 1200 \
  --lbfgs-max-iter 500 --use-pso --force-final

python quench2026_physical_heatmaps_improved.py \
  --results-root results/quench2026 \
  --out-dir results/quench2026/report_pack_improved/physical_heatmaps

python quench2026_report_pack_improved.py \
  --results-root results/quench2026 \
  --out-dir results/quench2026/report_pack_improved
```
