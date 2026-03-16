# Level 1 — Single-Shot Global PINN

Mevcut NAS-PINNS3 kodunun referans klasoru.
Ana kaynak kodlar `../src/` ve `../problems/` altındadır.
Sonuçlar `../results/run2/` içindedir.

## Ne yapar

PINN tüm [0, T_END] zaman aralığını tek seferde öğrenir.
T(x, y, t) doğrudan tahmin edilir.
NAS ile en iyi mimari seçilir (NSGA2, NSGA3, Bayesian).

## Mevcut sonuçlar (run2)

| Optimizer | Mimari       | MAE (°C) | L2_rel | NAS süresi |
|-----------|--------------|----------|--------|------------|
| NSGA-II   | 3×153, tanh  | 132.6    | 0.252  | 1583s      |
| NSGA-III  | 3×75, tanh   | 270.3    | 0.513  | 1609s      |
| Bayesian  | 5×151, relu  | 39.1     | 0.076  | 150s       |

## Çalıştırma

```bash
cd ..
python main.py --optimizer bayesian --problem quenching --time_mode full
```

## Sonraki adım

Level 2'ye geçmek için: `../level2_timestepper/`
