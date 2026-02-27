# NAS-PINNs Comparative Project (Burgers + Poisson)

Bu repo toplantı sonrası hedefe göre düzenlendi:

- Baseline: NAS-PINN
- Arama yöntemleri: NSGA-II, NSGA-III, Bayesian
- Denklem seti: Burgers + Poisson
- Poisson için domain karşılaştırması
- Burgers için viscosity karşılaştırması
- Standart çıktı: loss grafiği, heatmap karşılaştırmaları, CSV/metric/log

## Active Entry Scripts

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

## Multi-Case Runs

### Burgers (multi-nu)

Örnek:

```bash
python NAS_PINNs_burgers.py --multi-nu --nu-list 0.01,0.04,0.07
```

Aynı şekilde NSGA2/NSGA3/Bayesian scriptlerinde de `--multi-nu` + `--nu-list` var.

### Poisson (multi-domain)

Örnek:

```bash
python NAS_PINNs_poisson.py --multi-domain --domain-list rectangular,circle,lshape,flower,annulus
```

Aynı şekilde NSGA2/NSGA3/Bayesian scriptlerinde de `--multi-domain` + `--domain-list` var.

## Output Layout

### Burgers

- `results/burgers/naspinn`
- `results/burgers/nsga2`
- `results/burgers/nsga3`
- `results/burgers/bayesian`

Tipik dosyalar:

- `loss_curve.png`
- `burgers_heatmap.png`
- `result_comparison.png` (Exact / Pred / |Pred-Exact|)
- `burgers_exact_vs_pred_time_slices.png` (yalnız `nu=0.01`)
- `l2_error.txt`
- `run_time.txt`
- `metrics.csv`
- `viscosity_comparison.csv`

NAS-PINN stage bazlı çıktı klasörleri:

- `stage_adam/`
- `stage_lbfgs/` (L-BFGS açıksa)
- `stage_pso/` (PSO açıksa)
- `stage_summary.csv`

### Poisson

- `results/poisson/naspinn`
- `results/poisson/nsga2`
- `results/poisson/nsga3`
- `results/poisson/bayesian`

Tipik dosyalar:

- `poisson_*_loss_curve.png`
- `result_comparison.png` (Exact / Pred / |Pred-Exact|)
- `results_summary.csv`
- `run_time.txt`
- `metrics.csv`
- `domain_comparison.csv`

NAS-PINN stage bazlı çıktı klasörleri:

- `stage_adam/`
- `stage_lbfgs/` (L-BFGS açıksa)
- `stage_pso/` (PSO açıksa)
- `stage_summary.csv`

## Sequential Runner

Tüm scriptleri sırayla çalıştırmak için:

```bash
python run_pipeline.py
```

Hızlı smoke:

```bash
python run_pipeline.py --quick
```

Tekrar sayısı (hoca sorusu için run sayısı burada görünür):

```bash
python run_pipeline.py --repeats 3
```

Script başlangıçta iki sayı basar:

- `Top-level job count`: çalıştırılacak ana script adedi
- `Estimated sub-experiment count`: `nu/domain` bazlı toplam alt deney adedi

## Stage Mode

`run_pipeline.py` stage seçimini iki denklem için ayrı verebilir:

- `--burgers-stage {adam,lbfgs,pso}`
- `--poisson-stage {adam,lbfgs,pso}`

Örnek:

```bash
python run_pipeline.py --burgers-stage lbfgs --poisson-stage pso
```

Notlar:

- Burgers tarafında `pso` NAS-PINN baseline için aktiftir; NSGA2/NSGA3/Bayesian Burgers scriptleri Adam/L-BFGS akışındadır.
- `--stage` parametresi geri uyumluluk için var ve iki tarafa aynı stage’i uygular.

## Notes

- Burgers exact karşılaştırması veri seti uyumu nedeniyle `nu=0.01` için üretilir.
- Poisson tarafı domain bazlı karşılaştırma için normalize edildi.
