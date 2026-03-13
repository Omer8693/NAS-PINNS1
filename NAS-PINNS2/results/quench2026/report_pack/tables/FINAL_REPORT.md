# Quench2026 - Pointwise Comparison Against Paper Baseline

## Evaluation Set (Only Paper Data Points)
- Temperature points from paper: `times=[0,5,10,15,20,25,30]`, `depths=[0.0,0.3,0.6,0.9]` -> total `28` points.
- Displacement points from paper: `layers=[-2.1,-0.8,0.3,1.2,1.8]` at `x=0` -> total `5` points.
- All errors below are computed only on these paper points.

## Paper Baseline Values (Reference)
- Temperature range (degC): `[60.000000, 97.757454]`
- Displacement layers (mm): `[-2.1, -0.8, 0.3, 1.2, 1.8]`

## Comparison Table (Paper Baseline as Ground Truth)

| Method | Architecture | Runtime (min) | Temp L2 Error (RMSE, degC) | Disp L2 Error (RMSE, mm, x=0) | Best Objective | Rank | Quality |
|---|---|---:|---:|---:|---:|---:|---|
| paper_baseline | paper_reference | N/A | 0.000000 | 0.000000 | N/A | - | Ground Truth |
| nsga2_lbfgs5000 | L6_N132 | 30.970 | 455.303131 | 0.070567 | 0.004928 | 1 | Excellent |
| baseline_lbfgs5000 | L5_N96 | 20.328 | 455.269989 | 0.394618 | 0.008149 | 2 | Very Close |
| bayesian_lbfgs5000 | L5_N121 | 25.912 | 455.301758 | 0.570024 | 0.021403 | 3 | Close |
| nsga3_lbfgs5000 | L6_N141 | 29.041 | 455.308411 | 0.835625 | 0.015123 | 4 | Close |
| baseline_original | L5_N96 | 10.787 | 450.810974 | 31.351157 | 3.260912 | 5 | Far |

## Yorum (Neden Daha Iyi / Neden Kotulesti)
- **NSGA2 (L6_N132):** En iyi. Disp L2 en dusuk (`0.070567 mm`), bu nedenle paper noktalarina en yakin cozum.
- **Baseline+LBFGS5000 (L5_N96):** Ikinci en iyi. Runtime iyi (`20.33 min`) ve disp L2 makul (`0.394618 mm`), ancak NSGA2 kadar yakin degil.
- **Bayesian+LBFGS5000 (L5_N121):** Runtime iyi (`25.91 min`), fakat disp L2 baseline+LBFGS5000'dan daha yuksek (`0.570024 mm`).
- **NSGA3+LBFGS5000 (L6_N141):** Disp L2 daha da yuksek (`0.835625 mm`), bu nedenle paper'a yakinlik azalıyor.
- **Baseline original:** En hizli ama disp L2 cok yuksek (`31.351157 mm`), bu nedenle paper noktalarindan belirgin sekilde uzak.

## Net Sonuc
- Karsilastirma tamamen paper baseline veri noktalarina gore yapildi (28 sicaklik + 5 deplasman noktasi).
- Bu kritere gore en iyi yontem: **NSGA2 + LBFGS5000**.
- Sicaklik L2 tum yontemlerde birbirine yakin; ayrimi asil olarak displacement L2 belirliyor.
