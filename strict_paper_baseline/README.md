# Strict Paper Baseline (Isolated)

This folder is fully isolated from your existing pipeline logic.

- No existing training/optimizer file is modified.
- It only calls your current entry scripts with fixed paper settings.
- All outputs/logs are written under `results/strict_paper_baseline/...`.

## Scope

Strict baseline targets the equations explicitly used from the original NAS-PINN paper in your project:

1. 1D Burgers (paper protocol)
2. 1D Advection (beta sweep)
3. 2D Burgers

Poisson is not included in this strict-paper runner because your Poisson setup (multi irregular domains) is project-specific and not the same benchmark protocol.

## Locked Baseline Settings

### Burgers 1D
- `paper_nus = 0.1,0.07,0.04`
- `repeats = 5`
- `train_nt = 21`
- `train_nx = 250`
- `test_nt = 21`
- `test_nx = 500`
- `epochs = 15000`
- `stage = lbfgs`
- `seed = 42`

### Advection 1D
- `paper_betas = 1.0,0.5,0.1`
- `repeats = 5`
- `train_nt = 40`
- `train_nx = 120`
- `test_nt = 40`
- `test_nx = 120`
- `epochs = 12000`
- `layers = 4`
- `base_neurons = 128`
- `stage = lbfgs`
- `seed = 42`

### Burgers 2D
- `repeats = 5`
- `train_nt = 20`
- `train_nx = 25`
- `train_ny = 25`
- `test_nt = 41`
- `test_nx = 500`
- `test_ny = 500`
- `slice_times = 0,1,2`
- `epochs = 12000`
- `layers = 5`
- `base_neurons = 128`
- `stage = lbfgs`
- `seed = 42`

## Commands

Dry-run:

```bash
python strict_paper_baseline/run_strict_baseline.py --dry-run
```

Run strict NAS-PINN baseline:

```bash
python strict_paper_baseline/run_strict_baseline.py
```

Run comparison with current optimizers (same paper settings):

```bash
python strict_paper_baseline/run_compare_current_optimizers.py
```
