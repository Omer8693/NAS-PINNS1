# Run Guide (Güncel)

Bu projede çalıştırılacak ana dosyalar kök dizindeki `NAS_PINNs_*.py` dosyalarıdır.
`optimizers/` klasörü iç implementasyondur.

> Not: Dosya adında `PINNs` kullanılır, `PINNS` değil.

## Burgers

### Tekli çalıştırma

- NAS-PINN: `python NAS_PINNs_burgers.py`
- NSGA-II: `python NAS_PINNs_burgers_nsga2.py`
- NSGA-III: `python NAS_PINNs_burgers_nsga3.py`
- Bayesian: `python NAS_PINNs_burgers_bayesian.py`

### 3 viskoziteyi tek komutta çalıştırma

- NAS-PINN: `python NAS_PINNs_burgers.py --multi-nu`
- NSGA-II: `python NAS_PINNs_burgers_nsga2.py --multi-nu`
- NSGA-III: `python NAS_PINNs_burgers_nsga3.py --multi-nu`
- Bayesian: `python NAS_PINNs_burgers_bayesian.py --multi-nu`

Varsayılan viskozite listesi: `0.01,0.04,0.07` (gerekirse `--nu-list` ile değiştir).

Doğrulanmış örnek komut:

- `python NAS_PINNs_burgers.py --multi-nu --nu-list 0.01,0.04,0.07`

Terminal notu:

- Komutu markdown link formatında yapıştırma (`[...](...)` kullanma).
- Düz metin komut çalıştır: `python NAS_PINNs_burgers.py ...`

Üretilen klasör/çıktılar:

- Alt klasörler: `nu_*/`
- Karşılaştırma: `viscosity_comparison.csv`, `viscosity_comparison.png`
- Runtime: `run_time.txt`

Ek not (Burgers `--multi-nu`):

- `burgers_shock.mat` dosyası `0.01/pi` viskoziteye karşılık gelir.
- `nu` farklıysa exact-mat kıyası atlanır; heatmap/time-slice ve özet çıktılar yine üretilir.

Sonuç kökleri:

- `results/burgers/naspinn`
- `results/burgers/nsga2`
- `results/burgers/nsga3`
- `results/burgers/bayesian`

## Poisson

### Tekli çalıştırma

- NAS-PINN: `python NAS_PINNs_poisson.py`
- NSGA-II: `python NAS_PINNs_poisson_nsga2.py`
- NSGA-III: `python NAS_PINNs_poisson_nsga3.py`
- Bayesian: `python NAS_PINNs_poisson_bayesian.py`

### 3 seed karşılaştırmasını tek komutta çalıştırma

Poisson denkleminde viskozite olmadığı için 3 farklı seed kıyası kullanılır:

- NAS-PINN: `python NAS_PINNs_poisson.py --multi-seed`
- NSGA-II: `python NAS_PINNs_poisson_nsga2.py --multi-seed`
- NSGA-III: `python NAS_PINNs_poisson_nsga3.py --multi-seed`
- Bayesian: `python NAS_PINNs_poisson_bayesian.py --multi-seed`

Varsayılan seed listesi: `42,43,44` (gerekirse `--seed-list` ile değiştir).

Üretilen klasör/çıktılar:

- Alt klasörler: `seed_*/`
- Karşılaştırma: `seed_comparison.csv`, `seed_comparison.png`
- Runtime: `run_time.txt`

Sonuç kökleri:

- `results/poisson/naspinn`
- `results/poisson/nsga2`
- `results/poisson/nsga3`
- `results/poisson/bayesian`

## Loss/Epoch Grafikleri

Poisson akışlarında loss grafiği otomatik kaydedilir:

- naspinn: `poisson_naspinn_loss_curve.png`
- nsga2: `poisson_nsga2_loss_curve.png`
- nsga3: `poisson_nsga3_loss_curve.png`
- bayesian: `bayes_poisson_loss_curve.png`

## Plot Kodlarının Yeri

Plot kodları ana eğitim dosyalarından ayrılmıştır:

- `optimizers/burgers/plots.py`
- `optimizers/poisson/plots.py`

Renk/stil değişikliği yapmak için sadece bu iki dosyayı düzenlemek yeterlidir.

## PSO Karşılaştırma (Ayrı Kod, Mevcut Akışı Bozmaz)

PSO tarafı ayrı dosyalar olarak eklendi; mevcut ana eğitim dosyaları değişmeden kalır.

### Burgers PSO

- NAS-PINN: `python NAS_PINNs_burgers_pso.py`
- NSGA-II: `python NAS_PINNs_burgers_nsga2_pso.py`
- NSGA-III: `python NAS_PINNs_burgers_nsga3_pso.py`
- Bayesian: `python NAS_PINNs_burgers_bayesian_pso.py`

### Poisson PSO

- NAS-PINN: `python NAS_PINNs_poisson_pso.py`
- NSGA-II: `python NAS_PINNs_poisson_nsga2_pso.py`
- NSGA-III: `python NAS_PINNs_poisson_nsga3_pso.py`
- Bayesian: `python NAS_PINNs_poisson_bayesian_pso.py`

### PSO Parametreleri (verdiğin yapı ile)

- `--pop-size`
- `--generations`
- `--w`, `--c1`, `--c2`
- `--adaptive`
- `--initial-velocity` (`random`/`zero`)
- `--max-velocity-rate`
- `--pertube-best`
- `--base-args` (hedef scriptlere ek argüman geçirmek için)

### Kaydedilen Çıktılar

Her hedef için:

- `results/pso_compare/<target>/baseline/`
- `results/pso_compare/<target>/pso_evals/`
- `results/pso_compare/<target>/pso_best/`
- `results/pso_compare/<target>/pso_comparison.csv`
- `results/pso_compare/<target>/best_params.json`

`pso_comparison.csv` içinde baseline ve PSO-en-iyi sonuçları birlikte tutulur.

---

## Batch Run (Automation)

To automatically run all method and parameter combinations, use the `run_overnight.py` script in the root directory.

Basic command:

```bash
python3 run_overnight.py
```

To exclude PSO jobs:

```bash
python3 run_overnight.py --no-pso
```

For quick testing:

```bash
python3 run_overnight.py --quick
```

To stop on error and allow repair:

```bash
python3 run_overnight.py --stop-on-error
```

**NEW:** If an error occurs, the last 20 lines of the log file are automatically displayed. After you repair the issue, press Enter to continue the script from where it left off. If the retry also fails, the script stops.

Summary:

- If an error occurs, log lines are shown
- After repair, press Enter to continue
- The job is retried
- If successful, the script continues; if not, it stops

Outputs and results are saved under `results/`.
