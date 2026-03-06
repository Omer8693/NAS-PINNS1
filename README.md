# NAS-PINNs Comparative Project

Bu repo iki ana deney hattini icerir:

1. Burgers + Poisson (multi-case, klasik pipeline)
2. Advection + Burgers2D (paper-profile, resume destekli pipeline)

Temel karsilastirma yontemleri:

- `naspinn` (baseline)
- `nsga2`
- `nsga3`
- `bayesian`

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

Yararlilar:

- `python run_pipeline.py --quick`
- `python run_pipeline.py --repeats 3`
- `python run_pipeline.py --burgers-stage lbfgs --poisson-stage pso`

### 2) Advection + Burgers2D resume runner

```bash
./run_advection_burgers2d_remaining.sh
```

Bu script eksik runlari resume eder, tamamlananlari otomatik atlar.

Onemli ortam degiskenleri:

- `RUN_ROOT`: mevcut run klasoru (bos ise en guncel run otomatik secilir)
- `REQUIRE_GPU=1`: CUDA yoksa fail-fast
- `PROFILE=paper_baseline|ours_fast`
- `REPEATS`, `SEED`, `BETA_LIST`
- `RUN_FAMILIES=advection,burgers2d`
- `ADVECTION_METHODS=naspinn,nsga2,nsga3,bayesian`
- `BURGERS2D_METHODS=naspinn,nsga2,nsga3,bayesian`

Ornek (yalniz burgers2d search zinciri):

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

Her method/case klasorunde tipik olarak:

- `metrics.csv`
- `run_time.txt`
- `loss_curve*.png` / `result_comparison*.png`
- stage klasorleri: `stage_adam/`, `stage_lbfgs/`, `stage_pso/`, `stage_best/` (methode gore)

## Current Result Exports

Tum tamamlanmis kayitlardan uretilen csv dosyalari:

- `results/pipeline_runs/poisson_burgers_completed_all.csv`
- `results/pipeline_runs/poisson_burgers_completed_summary_by_stage.csv`
- `results/pipeline_runs/best_only_list.csv`

Not: Bu export dosyalari adlarinda eski isim kalmis olsa da su anda
`poisson + burgers + advection + burgers2d` tamamlanmis kayitlarini birlestirir.
