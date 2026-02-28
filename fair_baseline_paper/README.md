# Fair Paper Baseline (Isolated)

This folder provides an isolated baseline runner that does **not** modify any
existing training code.

All outputs/logs are written under:

- `results/fair_baseline_paper/<timestamp>/...`

## Purpose

- Keep your current code path untouched.
- Run a separate "paper-parameter baseline" path.
- Use fixed baseline settings per equation/method.

## What Is Locked

- Fixed seed: `42` (for baseline consistency).
- Paper-style settings are used where available:
  - Burgers 1D: `paper_protocol`, `nu=0.1,0.07,0.04`, repeats=5.
  - Advection 1D: `beta=1.0,0.5,0.1`, train/test grids `40x120`, repeats=5.
  - Burgers 2D: train `20x25x25`, test `41x500x500`, repeats=5.
- Poisson has no direct NAS-PINN paper protocol in this repository, so a fixed
  project baseline is used (same domains and fixed options).

## Usage

From repository root:

```bash
bash fair_baseline_paper/run_all.sh --dry-run
```

Run all equations and methods:

```bash
bash fair_baseline_paper/run_all.sh
```

Run subset:

```bash
bash fair_baseline_paper/run_all.sh \
  --equations advection,burgers2d \
  --methods naspinn,nsga2,nsga3,bayesian
```

Stop on first error:

```bash
bash fair_baseline_paper/run_all.sh --stop-on-error
```
