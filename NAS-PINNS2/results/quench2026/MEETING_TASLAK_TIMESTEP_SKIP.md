# Quench2026 - Time-Step Atlama Odakli Proje Taslagi

## 1) Proje Amaci (Net Tanim)
Paper'daki FEM tabanli zaman-ilerleme mantigini referans alarak, NAS-PINN ile tum zaman adimlarini kullanmadan (step atlayarak) benzer veya daha iyi dogruluk elde etmek ve toplam egitim suresini azaltmak.

Hedef: `daha az time-step + daha kisa runtime + benzer/iyi hata`.

## 2) Arastirma Sorusu
- Tum time-step'leri gezmek zorunda miyiz?
- Ornegin 1-2-3-4 yerine 1-3-4 (veya daha seyrek) kullanirsak,
  - sicaklik/deplasman hatasi korunuyor mu?
  - runtime anlamli sekilde dusuyor mu?

## 2.1 Paper Time-Step Seti (Sabit Referans)
Bu calismada referans zaman noktasi mutlaka paper setinden alinacaktir:
- `t = [0, 5, 10, 15, 20, 25, 30]` saniye

Ve sicaklik icin depth referanslari:
- `depth = [0.0, 0.3, 0.6, 0.9]` metre

Not: Kullanilan tum "step atlama" senaryolari bu paper zaman setinin alt-kumeleri olarak tanimlanacaktir.

## 3) Yontem (NSGA2/NSGA3/Bayesian ile)
Ayni NAS-PINN egitim mantigi korunur; mimari arama yontemi olarak:
- NSGA2
- NSGA3
- Bayesian Optimization

Bu yontemler mevcut kodda zaten var. Mimari arama + final asama (Adam->LBFGS->opsiyonel PSO) akisi aynen kullanilabilir.

## 4) Matematiksel Cerceve (Kisa)
Tam zaman kumesi: `T_full = {t0, t1, ..., tN}`

Alt-kume (atlanmis): `T_sub subset T_full`, `|T_sub| < |T_full|`

Toplam loss:
`L = w_physics*L_physics(T_sub) + w_ic*L_ic + w_bc*L_bc + w_data*L_data`

Karsilastirma hedefi:
- `runtime(T_sub) < runtime(T_full)`
- `error(T_sub) <= error(T_full) + epsilon`

## 5) Mevcut Kodda Ne Hazir?
`naspinn_baseline_with_quench_2026_data.py` tarafinda:
- `--n-time-steps`: Adam asamasinda zaman dongusu sayisi
- `--lbfgs-time-steps`: LBFGS asamasinda zaman dongusu sayisi

Pipeline tarafinda (`NSGA2/NSGA3/Bayesian`) bu argumanlar zaten pass-through olarak kullaniliyor.

Not:
- Mevcut haliyle "ozel indeks" (ornek: sadece 1,3,4) secimi yok.
- Ancak `n-time-steps` degerini dusurerek pratikte time-step azaltma/atlama etkisi uygulanabiliyor.

## 6) Hemen Uygulanabilir Deney Matrisi
Ayni seed ve ayni search butcesiyle:

- Exp-A (Full'e yakin): `n-time-steps=10`, `lbfgs-time-steps=4`
- Exp-B (Orta azalma): `n-time-steps=6`, `lbfgs-time-steps=3`
- Exp-C (Agresif): `n-time-steps=4`, `lbfgs-time-steps=2`
- Exp-D (Cok agresif): `n-time-steps=3`, `lbfgs-time-steps=1`

Her exp icin:
- NSGA2
- NSGA3
- Bayesian

Boylece 4 x 3 karsilastirma seti olusur.

## 7) Basari Kriteri
Asagidaki kosullardan en az biri saglanirsa "basarili":
- Runtime'da >= %30 azalma ve hata artisi <= %5
- Runtime benzerken hata iyilesmesi
- Runtime ve hata birlikte iyilesme (ideal)

## 8) Olculecek Metrikler
- `best_objective`
- `temp_rmse`, `disp_rmse` (paper referans noktalarinda)
- `run_time_seconds`
- Secilen asama (`adam/lbfgs/pso`)

## 9) Calistirma Komutlari

### 9.1 Tek yontem (ornek NSGA2)
```bash
python NAS-PINNS2/NAS_PINNs_quench_nsga2.py \
  --save-dir NAS-PINNS2/results/quench2026/timestep_skip/nsga2_expB \
  --n-time-steps 6 \
  --lbfgs-time-steps 3 \
  --epochs 5000 \
  --proxy-epochs 300 \
  --temp-ref-t0-mode align_ic
```

### 9.2 Tum yontemler pipeline
```bash
python NAS-PINNS2/NAS_PINNs_quench_pipeline.py \
  --save-dir NAS-PINNS2/results/quench2026/timestep_skip/expB \
  --methods nsga2,nsga3,bayesian \
  --n-time-steps 6 \
  --lbfgs-time-steps 3 \
  --epochs 5000 \
  --proxy-epochs 300 \
  --temp-ref-t0-mode align_ic
```

## 10) Hoca Icin Kisa Sonuc Cumlesi
"Bu calismada FEM referans zaman davranisini baseline alip, NAS-PINN tarafinda tum time-step'leri gezmeden calisarak runtime'i dusurmeyi ve dogrulugu korumayi hedefliyoruz. NSGA2/NSGA3/Bayesian ile hem mimariyi hem zaman adimi yogunlugunu deneysel olarak optimize edecegiz."

## 11) Bir Sonraki Teknik Adim (Opsiyonel Kod Gelistirme)
Gercek "1->3->4" gibi ozel atlama icin koda su parametre eklenebilir:
- `--time-step-indices 0,2,3,5,...`

Bu eklenirse sadece adim sayisi degil, hangi adimlarin ziyaret edilecegi de dogrudan kontrol edilir.
