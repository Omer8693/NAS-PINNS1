# Run Guide (Latest)

Bu dosya guncel akislara gore direkt calistirilabilir komut setidir.

## 1) Burgers + Poisson pipeline

Tumunu sirali calistir:

```bash
python run_pipeline.py
```

Hizli smoke:

```bash
python run_pipeline.py --quick
```

Tekrar sayisi:

```bash
python run_pipeline.py --repeats 3
```

Denklem bazli stage secimi:

```bash
python run_pipeline.py --burgers-stage lbfgs --poisson-stage pso
```

Not:

- Burgers tarafinda `pso` sadece NAS-PINN baseline icin aktif.
- Burgers NSGA2/NSGA3/Bayesian akisi Adam/L-BFGS cizgisindadir.

## 2) Advection + Burgers2D resume pipeline

Varsayilan resume calistirma:

```bash
./run_advection_burgers2d_remaining.sh
```

GPU zorunlu (onerilen):

```bash
REQUIRE_GPU=1 ./run_advection_burgers2d_remaining.sh
```

Belirli run klasorunde devam:

```bash
RUN_ROOT=results/pipeline_runs/20260303_222758_advection_burgers2d \
REQUIRE_GPU=1 \
./run_advection_burgers2d_remaining.sh
```

Sadece belirli aile/method sec:

```bash
RUN_ROOT=results/pipeline_runs/20260303_222758_advection_burgers2d \
RUN_FAMILIES=burgers2d \
BURGERS2D_METHODS=nsga2,nsga3,bayesian \
REQUIRE_GPU=1 \
./run_advection_burgers2d_remaining.sh
```

Filtre degiskenleri:

- `RUN_FAMILIES=advection,burgers2d`
- `ADVECTION_METHODS=naspinn,nsga2,nsga3,bayesian`
- `BURGERS2D_METHODS=naspinn,nsga2,nsga3,bayesian`

## 3) Manuel script sirasi

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

## 4) Ciktilar ve loglar

Classic pipeline:

- `results/pipeline_runs/<timestamp>/summary.csv`
- `results/pipeline_runs/<timestamp>/summary.json`
- `results/pipeline_runs/<timestamp>/artifacts/rep_XX/burgers/...`
- `results/pipeline_runs/<timestamp>/artifacts/rep_XX/poisson/...`

Advection + Burgers2D pipeline:

- `results/pipeline_runs/<timestamp>_advection_burgers2d/logs/*.log`
- `results/pipeline_runs/<timestamp>_advection_burgers2d/artifacts/advection/...`
- `results/pipeline_runs/<timestamp>_advection_burgers2d/artifacts/burgers2d/...`

## 5) Sonuc export dosyalari

Tamamlanmis kayit birlestirme csv:

- `results/pipeline_runs/poisson_burgers_completed_all.csv`
- `results/pipeline_runs/poisson_burgers_completed_summary_by_stage.csv`
- `results/pipeline_runs/best_only_list.csv`

Not: Bu exportlar su an `poisson + burgers + advection + burgers2d`
tamamlanmis kayitlarini birlikte tutar.
