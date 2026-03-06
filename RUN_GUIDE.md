# Run Guide (Latest)

This file provides directly runnable commands for the latest workflows.

## 1) Burgers + Poisson pipeline

Run all jobs sequentially:

```bash
python run_pipeline.py
```

Quick smoke test:

```bash
python run_pipeline.py --quick
```

Set repeat count:

```bash
python run_pipeline.py --repeats 3
```

Equation-specific stage selection:

```bash
python run_pipeline.py --burgers-stage lbfgs --poisson-stage pso
```

Notes:

- On Burgers, `pso` is active only for NAS-PINN baseline.
- Burgers NSGA2/NSGA3/Bayesian follow the Adam/L-BFGS path.

## 2) Advection + Burgers2D resume pipeline

Default resume run:

```bash
./run_advection_burgers2d_remaining.sh
```

Require GPU (recommended):

```bash
REQUIRE_GPU=1 ./run_advection_burgers2d_remaining.sh
```

Resume a specific run directory:

```bash
RUN_ROOT=results/pipeline_runs/20260303_222758_advection_burgers2d \
REQUIRE_GPU=1 \
./run_advection_burgers2d_remaining.sh
```

Run only selected family/methods:

```bash
RUN_ROOT=results/pipeline_runs/20260303_222758_advection_burgers2d \
RUN_FAMILIES=burgers2d \
BURGERS2D_METHODS=nsga2,nsga3,bayesian \
REQUIRE_GPU=1 \
./run_advection_burgers2d_remaining.sh
```

Filter variables:

- `RUN_FAMILIES=advection,burgers2d`
- `ADVECTION_METHODS=naspinn,nsga2,nsga3,bayesian`
- `BURGERS2D_METHODS=naspinn,nsga2,nsga3,bayesian`

## 3) Manual script order

### Burgers

1. `python NAS_PINNs_burgers.py --multi-nu --nu-list 0.01,0.04,0.07`
2. `python NAS_PINNs_burgers_nsga2.py --multi-nu --nu-list 0.01,0.04,0.07`
3. `python NAS_PINNs_burgers_nsga3.py --multi-nu --nu-list 0.01,0.04,0.07`
4. `python NAS_PINNs_burgers_bayesian.py --multi-nu --nu-list 0.01,0.04,0.07`

### Poisson

1. `python NAS_PINNs_poisson.py --multi-domain --domain-list rectangular,circle,lshape,flower,annulus`
2. `python NAS_PINNs_poisson_nsga2.py --multi-domain --domain-list rectangular,circle,lshape,flower,annulus --skip-pso`
3. `python NAS_PINNs_poisson_nsga3.py --multi-domain --domain-list rectangular,circle,lshape,flower,annulus --skip-pso`
4. `python NAS_PINNs_poisson_bayesian.py --multi-domain --domain-list rectangular,circle,lshape,flower,annulus --skip-pso`

## 4) Outputs and logs

Classic pipeline:

- `results/pipeline_runs/<timestamp>/summary.csv`
- `results/pipeline_runs/<timestamp>/summary.json`
- `results/pipeline_runs/<timestamp>/artifacts/rep_XX/burgers/...`
- `results/pipeline_runs/<timestamp>/artifacts/rep_XX/poisson/...`

Advection + Burgers2D pipeline:

- `results/pipeline_runs/<timestamp>_advection_burgers2d/logs/*.log`
- `results/pipeline_runs/<timestamp>_advection_burgers2d/artifacts/advection/...`
- `results/pipeline_runs/<timestamp>_advection_burgers2d/artifacts/burgers2d/...`

## 5) Result export files

Merged CSVs for completed records:

- `results/pipeline_runs/poisson_burgers_completed_all.csv`
- `results/pipeline_runs/poisson_burgers_completed_summary_by_stage.csv`
- `results/pipeline_runs/best_only_list.csv`

Note: These exports currently include completed records from
`poisson + burgers + advection + burgers2d`.
