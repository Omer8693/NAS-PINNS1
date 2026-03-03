# paper_strict_scratch

Bu klasor mevcut proje koduna dokunmadan, NAS-PINN paper protokolune gore sifirdan ayrik bir pipeline sunar.

## Icerik

- `equations.py`: paper denklemleri/formulleri
  - Burgers 1D
  - Advection 1D
  - Burgers 2D
  - Poisson 2D (domain: rectangular/circle/lshape/flower/annulus)
- `model.py`: NAS-PINN supernet (op + mask relaxation)
- `search.py`: NSGA-II / NSGA-III / Bayesian architecture search
- `trainer.py`: Adam-best -> LBFGS ve Adam-best -> PSO stage akisi
- `runner.py`: case/repeat orchestrator, CSV/JSON ozetleri
- `run_naspinn.py`: baseline NAS-PINN
- `run_nsga2.py`: NSGA-II tabanli arama + ayni stage akisi
- `run_nsga3.py`: NSGA-III tabanli arama + ayni stage akisi
- `run_bayesian.py`: Bayesian tabanli arama + ayni stage akisi
- `run_full_suite.sh`: tum equation x method kombinasyonlarini (poisson dahil) sirali calistirir

## Paper-Locked Varsayilanlar

- Burgers1D: `nu = [0.01, 0.04, 0.07]`, repeats=5
- Advection1D: `beta = [1.0, 0.5, 0.1]`, repeats=5
- Burgers2D: tek case, repeats=5
- Poisson: `domain = [rectangular, circle, lshape, flower, annulus]`, repeats=1
- Advection train/test grid: `40 x 120`
- Burgers2D train grid: `20 x 25 x 25`, test grid: `41 x 500 x 500`
- Poisson parametreleri (orijinal NAS-PINN Poisson ayarlari):
  - `base_neurons=110`, `hidden_layers=5`, `mask_levels=[30,50,70,90,110]`
  - `epochs=30000`, `inner_lr=1e-3`, `outer_lr=3e-4`, `outer_every=5`
  - `lbfgs_max_iter=6000`, `pso_iters=8`, `pso_swarm=16`, `pso_span=0.25`
  - PSO: pymoo adaptif fuzzy-PSO (fallback: bounded basic PSO)
  - `n_col=4000`, `n_bc=400`, `test_grid=500`
- Stage mantigi: Adam-best -> LBFGS ve Adam-best -> PSO (ayri stage), en iyi stage rel L2 ile secilir

## Ornek Komutlar

Baseline:

```bash
python paper_strict_scratch/run_naspinn.py --equation advection1d \
  --save-dir results/paper_strict_scratch/advection1d/naspinn
```

NSGA-II:

```bash
python paper_strict_scratch/run_nsga2.py --equation advection1d \
  --save-dir results/paper_strict_scratch/advection1d/nsga2
```

NSGA-III:

```bash
python paper_strict_scratch/run_nsga3.py --equation burgers2d \
  --save-dir results/paper_strict_scratch/burgers2d/nsga3
```

Bayesian:

```bash
python paper_strict_scratch/run_bayesian.py --equation burgers1d \
  --save-dir results/paper_strict_scratch/burgers1d/bayesian
```

Poisson (ornek):

```bash
python paper_strict_scratch/run_nsga3.py --equation poisson \
  --save-dir results/paper_strict_scratch/poisson/nsga3
```

## Hızlı smoke-test icin

```bash
python paper_strict_scratch/run_nsga2.py --equation advection1d \
  --repeats 1 --cases 1.0 --epochs 1 --proxy-epochs 1 \
  --pop-size 4 --n-gen 1 --skip-lbfgs --skip-pso
```

## Cikti Yapisi

Her run icinde:

- `stage_adam/loss_history.csv`
- `stage_lbfgs/loss_history.csv` (skip edilmediyse)
- `stage_pso/loss_history.csv` (skip edilmediyse)
- `results_summary.csv`
- `run_meta.json`

Gorseller:

- Tum denklemler icin `result_comparison.png`:
  - Exact
  - Predicted
  - `|Pred-Exact|` isi haritasi (Relative L2 bilgisi baslikta)
- Burgers2D icin ek:
  - `slice_t_0.00_comparison.png`
  - `slice_t_1.00_comparison.png`
  - `slice_t_2.00_comparison.png`
- Her stage icin:
  - `loss_curve.png`
  - `l2_error.txt`
  - `metrics.csv`
- En iyi stage:
  - `stage_best/` (secili stage dosyalari)
  - `stage_best/selected_stage.txt`

Root `save-dir` altinda:

- `summary_<equation>_<method>.csv`
- `summary_<equation>_<method>.json`
