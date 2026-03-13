# PINN Quenching Project — Dosya Yapısı

## Proje Amacı
Mortensen et al. (2026) FEM tabanlı alüminyum subframe quenching analizinin
PINN (Physics-Informed Neural Network) ile yeniden üretimi.
- FEM'in zaman adımlı çözümü → adaptif zaman atlama (TemporalSkipScheduler)
- DARTS NAS → NSGA-II, NSGA-III, Bayesian Optimization
- Adam eğitimi → L-BFGS ve PSO ile bağımsız ince ayar
- Tüm paper çıktıları baseline_data.py'de saklanır ve otomatik karşılaştırılır

---

## Dosyalar

```
pinn_project/
├── main.py                   # Ana runner: NAS → eğitim → görselleştirme → baseline
├── requirements.txt          # torch, pymoo, scikit-optimize, pyswarms, matplotlib
│
└── src/
    ├── physics_model.py      # Fizik denklemleri (Mortensen 2026 Eq. 3,5,6)
    │                         #  + WaterQuenchHTC (4 bölgeli film boiling modeli)
    │                         #  + TemporalSkipScheduler (projenin ana yeniliği)
    │
    ├── pinn_network.py       # PINNNet (NAS-PINN tabanlı)
    │                         #  + PINNLoss (fizik + sınır + başlangıç + veri)
    │                         #  + CollocationSampler (kollokasiyon noktaları)
    │
    ├── arch_search.py        # NAS optimizer'ları
    │                         #  + NSGA-II  (pymoo)
    │                         #  + NSGA-III (pymoo, Das-Dennis ref dirs)
    │                         #  + Bayesian (scikit-optimize GP)
    │
    ├── trainers.py           # Eğitim modülleri
    │                         #  + AdamTrainer    (Phase 1: bulk training)
    │                         #  + LBFGSFinetuner (Phase 2: Adam'dan bağımsız)
    │                         #  + PSOFinetuner   (Phase 3: Adam'dan bağımsız)
    │                         #  + full_training_pipeline
    │
    └── baseline_data.py      # Paper baseline verileri (Mortensen 2026)
                              #  + Table 1: A356 malzeme parametreleri
                              #  + Table 2: sertlik-deplasman ilişkisi
                              #  + Fig 15: As-cast distorsiyonları
                              #  + Fig 16: Creep distorsiyonları
                              #  + Fig 17/18: Quenching distorsiyonları (3 katman)
                              #  + Fig 19/20: Optimize raf (5 katman)
                              #  + Fig 21/22: Cross-member CMM noktaları
                              #  + Fig 7: Su tankı sıcaklığı
                              #  + MortensenBaseline: karşılaştırma + görselleştirme
```

---

## Çalıştırma

```bash
pip install torch pymoo scikit-optimize matplotlib numpy

# Yalnızca baseline grafiklerini çiz (eğitim gerektirmez)
python main.py --baseline_only

# NSGA-II ile tam deney
python main.py --optimizer nsga2 --adam_epochs 3000 --run_lbfgs --run_pso

# NSGA-III ile
python main.py --optimizer nsga3 --nas_pop 20 --nas_gen 30

# Bayesian Optimization ile
python main.py --optimizer bayesian --nas_calls 30

# Tüm optimizerleri karşılaştır
python main.py --compare_all --adam_epochs 2000
```

---

## Çıktı Dosyaları

Her optimizer için `results/{optimizer}/`:
- `exact_vs_pred.png`          — analitik vs PINN sıcaklık haritası
- `loss_l2_comparison.png`     — Adam/L-BFGS/PSO kayıp ve L2 grafikleri
- `temporal_skip_stats.png`    — zaman atlama istatistikleri
- `baseline_comparison.png`    — Mortensen paper ölçüm/FEM/PINN karşılaştırması
- `model.pt`                   — kaydedilmiş model ağırlıkları
- `best_arch.json`             — NAS tarafından bulunan en iyi mimari

Genel:
- `results/{optimizer}_results.json` — tüm metrikler
- `results/comparison.json`          — optimizer karşılaştırması (--compare_all)
- `results/baseline/`                — paper baseline grafikleri

---

## baseline_data.py Kullanımı

```python
from src.baseline_data import MortensenBaseline
import numpy as np

bl = MortensenBaseline()
bl.summary()    # tüm veri setlerini listele

# Herhangi bir veri setini al
data = bl.get("fig17_ht")
print(data["points"])           # CMM nokta isimleri
print(data["Measured_Bottom"])  # ölçülen değerler [mm]
print(data["Simulated_Bottom"]) # Mortensen FEM değerleri [mm]

# PINN tahminleriyle karşılaştır
pinn_preds = np.array(...)   # model'den alınan tahminler
metrics = bl.compare_with_pinn(pinn_preds, "fig17_ht", layer="Bottom")
# → MAE, RMSE, MaxErr, R² hem PINN hem FEM için yazdırılır

# Grafik oluştur
bl.plot_comparison(pinn_preds, "fig17_ht", "Bottom",
                   optimizer_name="PINN_NSGA2", save_path="results/")
```
