# NAS-PINNs Comparative Project

This repository contains two main experiment tracks:

1. Burgers + Poisson (multi-case, classic pipeline)
2. Advection + Burgers2D (paper-profile, resume-enabled pipeline)

Core comparison methods:

- `naspinn` (baseline)
- `nsga2`
- `nsga3`
- `bayesian`

## Quench2026 (NAS-PINNS2) Quick Link

Quench2026 deneylerinin güncel ve detaylı dokümantasyonu:
- `NAS-PINNS2/README.md`

Bu rehberde şunlar var:
- script/kod haritası
- çıktı klasörlerinin anlamı
- hangi soruya hangi CSV/PNG bakılacağı
- önerilen çalıştırma sırası
- strict fair same-architecture kıyas akışı

## Entry Scripts

### Burgers

- `python NAS_PINNs_burgers.py`
- `python NAS_PINNs_burgers_nsga2.py`
- `python NAS_PINNs_burgers_nsga3.py`
- `python NAS_PINNs_burgers_bayesian.py`

### Poisson

- `python NAS_PINNs_poisson.py`
- `python NAS_PINNs_poisson_nsga2.py`
- `python NAS_PINNs_poisson_nsga3.py`
- `python NAS_PINNs_poisson_bayesian.py`

### Advection

- `python NAS_PINNs_advection.py`
- `python NAS_PINNs_advection_nsga2.py`
- `python NAS_PINNs_advection_nsga3.py`
- `python NAS_PINNs_advection_bayesian.py`

### Burgers2D

- `python NAS_PINNs_burgers2d.py`
- `python NAS_PINNs_burgers2d_nsga2.py`
- `python NAS_PINNs_burgers2d_nsga3.py`
- `python NAS_PINNs_burgers2d_bayesian.py`

## Main Runners

### 1) Burgers + Poisson sequential runner

```bash
python run_pipeline.py
```

Useful options:

- `python run_pipeline.py --quick`
- `python run_pipeline.py --repeats 3`
- `python run_pipeline.py --burgers-stage lbfgs --poisson-stage pso`

### 2) Advection + Burgers2D resume runner

```bash
./run_advection_burgers2d_remaining.sh
```

This script resumes missing runs and automatically skips completed ones.

Important environment variables:

- `RUN_ROOT`: existing run directory (if empty, the most recent run is selected automatically)
- `REQUIRE_GPU=1`: fail-fast if CUDA is not available
- `PROFILE=paper_baseline|ours_fast`
- `REPEATS`, `SEED`, `BETA_LIST`
- `RUN_FAMILIES=advection,burgers2d`
- `ADVECTION_METHODS=naspinn,nsga2,nsga3,bayesian`
- `BURGERS2D_METHODS=naspinn,nsga2,nsga3,bayesian`

Example (only burgers2d search chain):

```bash
RUN_ROOT=results/pipeline_runs/20260303_222758_advection_burgers2d \
RUN_FAMILIES=burgers2d \
BURGERS2D_METHODS=nsga2,nsga3,bayesian \
REQUIRE_GPU=1 \
./run_advection_burgers2d_remaining.sh
```

## Output Layout

### Classic pipeline (Burgers + Poisson)

- `results/pipeline_runs/<timestamp>/artifacts/rep_XX/burgers/...`
- `results/pipeline_runs/<timestamp>/artifacts/rep_XX/poisson/...`
- `results/pipeline_runs/<timestamp>/summary.csv`
- `results/pipeline_runs/<timestamp>/summary.json`

### Advection + Burgers2D pipeline

- `results/pipeline_runs/<timestamp>_advection_burgers2d/artifacts/advection/...`
- `results/pipeline_runs/<timestamp>_advection_burgers2d/artifacts/burgers2d/...`
- `results/pipeline_runs/<timestamp>_advection_burgers2d/logs/*.log`

Typical files inside each method/case directory:

- `metrics.csv`
- `run_time.txt`
- `loss_curve*.png` / `result_comparison*.png`
- stage directories: `stage_adam/`, `stage_lbfgs/`, `stage_pso/`, `stage_best/` (depending on method)

## Current Result Exports

CSV files generated from all completed records:

- `results/pipeline_runs/poisson_burgers_completed_all.csv`
- `results/pipeline_runs/poisson_burgers_completed_summary_by_stage.csv`
- `results/pipeline_runs/best_only_list.csv`

Note: These export filenames keep legacy naming, but currently they include
completed records from `poisson + burgers + advection + burgers2d`.
