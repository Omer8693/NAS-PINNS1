# NAS-PINN: Neural Architecture Search for PINNs (Burgers + Poisson)

## Project Overview

This repository contains the **current active workflow** for NAS-based PINN training on:

- **Burgers equation**
- **Poisson equation**

Supported search/training methods:

- **NAS-PINN** (`naspinn`)
- **NSGA-II**
- **NSGA-III**
- **Bayesian Optimization**

The codebase is organized so that root scripts are lightweight entrypoints, while implementation lives under `optimizers/`.

---

## Current Repository Structure

```text
NAS-PINNS1/
├── NAS_PINNs_burgers.py
├── NAS_PINNs_burgers_nsga2.py
├── NAS_PINNs_burgers_nsga3.py
├── NAS_PINNs_burgers_bayesian.py
├── NAS_PINNs_poisson.py
├── NAS_PINNs_poisson_nsga2.py
├── NAS_PINNs_poisson_nsga3.py
├── NAS_PINNs_poisson_bayesian.py
├── optimizers/
│   ├── burgers/
│   │   ├── naspinn.py
│   │   ├── nsga2.py
│   │   ├── nsga3.py
│   │   ├── bayesian.py
│   │   └── plots.py
│   └── poisson/
│       ├── common.py
│       ├── naspinn.py
│       ├── nsga2.py
│       ├── nsga3.py
│       ├── bayesian.py
│       └── plots.py
├── migrate_results.py
├── RUN_GUIDE.md
├── legacy_previous_work/
└── results/
```

Notes:

- Plot code is centralized per domain in `optimizers/burgers/plots.py` and `optimizers/poisson/plots.py`.
- This means style/color updates are made once and affect all methods automatically.

---

## How to Run

### Burgers (single run)

- `python NAS_PINNs_burgers.py`
- `python NAS_PINNs_burgers_nsga2.py`
- `python NAS_PINNs_burgers_nsga3.py`
- `python NAS_PINNs_burgers_bayesian.py`

### Burgers (3 viscosity comparison)

- `python NAS_PINNs_burgers.py --multi-nu`
- `python NAS_PINNs_burgers_nsga2.py --multi-nu`
- `python NAS_PINNs_burgers_nsga3.py --multi-nu`
- `python NAS_PINNs_burgers_bayesian.py --multi-nu`

Default viscosity list: `0.01,0.04,0.07` (override with `--nu-list`).

Tested command example:

- `python NAS_PINNs_burgers.py --multi-nu --nu-list 0.01,0.04,0.07`

Important terminal note:

- Do not paste markdown link syntax into terminal (for example `[NAS_PINNs_burgers.py](...)`).
- Use plain command text only.

### Poisson (single run)

- `python NAS_PINNs_poisson.py`
- `python NAS_PINNs_poisson_nsga2.py`
- `python NAS_PINNs_poisson_nsga3.py`
- `python NAS_PINNs_poisson_bayesian.py`

### Poisson (3 seed comparison)

- `python NAS_PINNs_poisson.py --multi-seed`
- `python NAS_PINNs_poisson_nsga2.py --multi-seed`
- `python NAS_PINNs_poisson_nsga3.py --multi-seed`
- `python NAS_PINNs_poisson_bayesian.py --multi-seed`

Default seed list: `42,43,44` (override with `--seed-list`).

> Use `PINNs` in filenames (e.g. `NAS_PINNs_burgers.py`), not `PINNS`.

---

## Output Layout

Results are written under:

- `results/burgers/naspinn`
- `results/burgers/nsga2`
- `results/burgers/nsga3`
- `results/burgers/bayesian`
- `results/poisson/naspinn`
- `results/poisson/nsga2`
- `results/poisson/nsga3`
- `results/poisson/bayesian`

Typical outputs include:

- prediction/exact/error plots
- comparison CSV/PNG for multi-run mode
    - `viscosity_comparison.csv/.png` (Burgers)
    - `seed_comparison.csv/.png` (Poisson)
- runtime log: `run_time.txt`

Poisson methods also save loss-vs-epoch figures (loss curve PNG files).

For Burgers `--multi-nu` runs:

- `nu_0.010`, `nu_0.040`, `nu_0.070` subfolders are generated.
- Exact `burgers_shock.mat` comparison plots are only valid for dataset viscosity (`0.01/pi`).
- For different `nu` values, code skips that exact-comparison step and still saves learned-solution plots.

---

## Plot Styling (Current)

- Plot logic is separated from training scripts to keep main files shorter.
- Colormap is intentionally simplified to a cleaner palette via shared plot modules.
- If you want a new style, edit only:
    - `optimizers/burgers/plots.py`
    - `optimizers/poisson/plots.py`

---

## PSO Extension (Standalone)

A separate PSO-based comparison pipeline is added without changing core method scripts.

Core PSO modules:

- `optimizers/pso/fuzzy_pso.py` (PSO implementation with `w`, `c1`, `c2`, adaptive update)
- `optimizers/pso/runner.py` (baseline + PSO search + best rerun + comparison export)

Standalone entrypoints (one per method):

- `NAS_PINNs_burgers_pso.py`
- `NAS_PINNs_burgers_nsga2_pso.py`
- `NAS_PINNs_burgers_nsga3_pso.py`
- `NAS_PINNs_burgers_bayesian_pso.py`
- `NAS_PINNs_poisson_pso.py`
- `NAS_PINNs_poisson_nsga2_pso.py`
- `NAS_PINNs_poisson_nsga3_pso.py`
- `NAS_PINNs_poisson_bayesian_pso.py`

PSO outputs are saved under `results/pso_compare/<target>/` including:

- `baseline/`
- `pso_evals/`
- `pso_best/`
- `pso_comparison.csv`
- `best_params.json`

---

## Toplu Çalıştırma (Otomasyon)

`run_overnight.py` dosyası ile tüm yöntemler ve parametre kombinasyonları otomatik olarak çalıştırılır.

Temel komut:

```bash
python3 run_overnight.py
```

PSO hariç çalıştırmak için:

```bash
python3 run_overnight.py --no-pso
```

Hızlı test için:

```bash
python3 run_overnight.py --quick
```

Hata durumunda durması ve onarım için:

```bash
python3 run_overnight.py --stop-on-error
```

**YENİ:** Hata oluşursa, log dosyasının son 20 satırı otomatik olarak gösterilir. Onarım yaptıktan sonra Enter'a basarak script kaldığı yerden devam eder. Eğer tekrar deneme de başarısız olursa script durur.

Özetle:

- Hata olursa log satırları gösterilir
- Onarım sonrası Enter ile devam edilir
- Job tekrar denenir
- Başarılı olursa devam, başarısızsa durur

Çıktılar ve sonuçlar `results/` altında kaydedilir.

---

## Legacy Code

Old pipeline files are archived in:

- `legacy_previous_work/`

They are kept for reference and are not part of the active workflow.

