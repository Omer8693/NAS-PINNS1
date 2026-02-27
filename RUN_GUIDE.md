# Run Guide (Guncel)

Bu dosya yeni akisa gore net komut listesidir.

## 1) Tum projeyi tek komutla sirali calistir

```bash
python run_pipeline.py
```

### Hizli test

```bash
python run_pipeline.py --quick
```

### 3 tekrarli deney

```bash
python run_pipeline.py --repeats 3
```

### Stage secimi

```bash
python run_pipeline.py --stage adam
python run_pipeline.py --stage lbfgs
python run_pipeline.py --stage pso
```

Denklem bazli ayri stage:

```bash
python run_pipeline.py --burgers-stage adam --poisson-stage pso
python run_pipeline.py --burgers-stage lbfgs --poisson-stage lbfgs
```

Not:

- Burgers tarafinda `pso` NAS-PINN baseline icin calisir.
- Burgers NSGA2/NSGA3/Bayesian tarafi Adam/L-BFGS akisindadir.

## 2) Manuel calistirma sirasi

Asagidaki sira dogrudan hocaya gosterilecek karsilastirma duzenidir.

1. `python NAS_PINNs_burgers.py --multi-nu --nu-list 0.01,0.04,0.07`
2. `python NAS_PINNs_burgers_nsga2.py --multi-nu --nu-list 0.01,0.04,0.07`
3. `python NAS_PINNs_burgers_nsga3.py --multi-nu --nu-list 0.01,0.04,0.07`
4. `python NAS_PINNs_burgers_bayesian.py --multi-nu --nu-list 0.01,0.04,0.07`
5. `python NAS_PINNs_poisson.py --multi-domain --domain-list rectangular,circle,lshape,flower,annulus`
6. `python NAS_PINNs_poisson_nsga2.py --multi-domain --domain-list rectangular,circle,lshape,flower,annulus --skip-pso`
7. `python NAS_PINNs_poisson_nsga3.py --multi-domain --domain-list rectangular,circle,lshape,flower,annulus --skip-pso`
8. `python NAS_PINNs_poisson_bayesian.py --multi-domain --domain-list rectangular,circle,lshape,flower,annulus --skip-pso`

## 3) Hoca sorusu: "kac kez run ediliyor?"

Iki seviye var:

- Top-level run: kac ana script cagiriliyor (pipeline'da 8 adet x repeats)
- Sub-experiment run: her scriptin icindeki nu/domain parcali deney adedi

Varsayilan durumda:

- Burgers: 4 yontem x 3 nu = 12 alt deney
- Poisson: 4 yontem x 5 domain = 20 alt deney
- Toplam alt deney = 32 (repeats=1 icin)

`run_pipeline.py` bunu otomatik yazdirir.

## 4) Ciktilar nerede?

### Burgers

- `results/burgers/naspinn`
- `results/burgers/nsga2`
- `results/burgers/nsga3`
- `results/burgers/bayesian`

NAS-PINN icin stage klasorleri:

- `stage_adam/`
- `stage_lbfgs/`
- `stage_pso/` (PSO aciksa)
- `stage_summary.csv`

### Poisson

- `results/poisson/naspinn`
- `results/poisson/nsga2`
- `results/poisson/nsga3`
- `results/poisson/bayesian`

NAS-PINN icin stage klasorleri:

- `stage_adam/`
- `stage_lbfgs/`
- `stage_pso/` (PSO aciksa)
- `stage_summary.csv`

Pipeline log/ozet:

- `results/pipeline_runs/<timestamp>/summary.csv`
- `results/pipeline_runs/<timestamp>/summary.json`
