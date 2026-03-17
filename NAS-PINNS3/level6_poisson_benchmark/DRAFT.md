# Level 6 — Poisson-Assisted Enhancement of Quenching PINN
# (FR: Level 5 çözümünü Poisson denklem kısıtıyla iyileştirme)

---

## 1. Fizik Bağlantısı — Neden Poisson Quenching'e Yardımcı Olur?

### Isı Denklemi (bizim problemimiz):
```
ρCp · ∂T/∂t  =  k · (∂²T/∂x²  +  ∂²T/∂y²)
ρCp · ∂T/∂t  =  k · ΔT
```

Her **sabit t = t₀ zaman diliminde** bu denklemi yeniden düzenlersek:

```
-ΔT(x,y)  =  -(ρCp/k) · ∂T/∂t|_{t₀}  =  f(x,y,t₀)
```

**Bu tam olarak bir 2D Poisson denklemine dönüşüyor.**
- Sağ taraf  `f = -(ρCp/k)·∂T/∂t`  kaynak terimi (source term)
- Sol taraf  `-ΔT`  spatial Laplacian

Sonuç:  **Isı denklemi = her zaman diliminde ayrı bir Poisson problemi.**

### Neden mevcut PINN bunu tam yakalamıyor?
Level 5 ağı `T(x,y,t)` çıktı olarak öğreniyor ama:
- Spatial yapı (x,y) ile temporal yapı (t) iç içe geçiyor
- Sınıra yakın bölgelerde (`x≈0, x≈1.3m, y≈0, y≈0.6m`) HTC Robin BC yüksek gradyanlar yaratıyor
- Ağ bu yüksek-gradyanlı sınır katmanlarını öğrenmekte zorlanıyor → Level 5 hataları büyük sınır bölgelerinde

**Poisson** bu spatial gradyanları açık bir fizik kısıtı olarak modelleyebilir.

---

## 2. Üç Yaklaşım Seçeneği

### Yaklaşım A — Quasi-Static Poisson Auxiliary Loss (Önerilen) ⭐
```
Fikir: Her epoch'ta bazı zaman dilimlerini sabit tutarak Poisson residual'ı
       ayrı bir kayıp terimi olarak ekle.

L_total = L_heat  +  λ_p · L_poisson_slices

L_poisson_slices = (1/K) Σ_{k=1}^{K} || ΔT(x,y,t_k) + (ρCp/k)·∂T/∂t|_{t_k} ||²

Neden yardımcı olur:
- Aslında heat equation = Poisson, ama ağırlıklandırma farklı
- λ_p ile Poisson kısıtını güçlendirmek = spatial accuracy'yi artırmak
- Heat eq. loss'ta ρCp·∂T/∂t - k·ΔT = 0 tek terime bakıyor
- Poisson loss'ta spatial Laplacian ayrıca penalize ediliyor
  → ağ hem temporal hem spatial gradyanları daha dengeli öğreniyor

Uygulama:
  Level 5 ağını yükle → Poisson auxiliary ekle → fine-tune
```
**Beklenen iyileştirme:** L2 %10-25 düşüşü (özellikle sınır bölgelerinde)

---

### Yaklaşım B — Poisson Spatial Pretraining (Transfer Learning)
```
Fikir: Önce (x,y) spatial encoder'ı Poisson ile eğit,
       sonra quenching üzerinde fine-tune yap.

Aşama 1 (Poisson): T_spatial(x,y) öğren — quenching geometrisinde (-Δu = f)
                   Çeşitli f(x,y) örnekleri: Gaussian blobs, sinus, random
                   BCs: Robin tip (HTC sabit tutularak)  ← quenching BC'sine yakın

Aşama 2 (Quenching):  T_spatial ağırlıklarını başlangıç noktası olarak al
                       Temporal boyutu ekle: T(x,y,t) = net(concat([x,y,t]))
                       Full quenching eğitimi (cosine LR)

Neden yardımcı olur:
  - Poisson çözümleri ağa bu domain'in Green's function yapısını öğretiyor
  - Robin BC'yi Poisson aşamasında öğrenen ağ, quenching'e daha hazır başlıyor
  - Level 5'teki 3 optimizer farklı mimari buldu → hepsini aynı pretrained
    spatial encoder ile başlatmak sonuçları yaklaştırabilir
```
**Beklenen iyileştirme:** NSGA-III için büyük (L2 0.274→<0.1 hedef), Bayesian için küçük

---

### Yaklaşım C — Poisson-Moded Decomposition  (Karmaşık)
```
Fikir: T(x,y,t) = T̄(x,y) + T'(x,y,t)  olarak ayır
  - T̄(x,y):  time-averaged temperature → Poisson denklemini çözer (-ΔT̄ = f̄)
  - T'(x,y,t): dalgalanma → basitleştirilmiş heat eq. çözer

Ağlar: 2 ayrı network → birlikte eğit
Avantaj: Spatial ve temporal öğrenme ayrıştırılıyor
Dezavantaj: 2 ağ → eğitim karmaşıklığı 2×, interface tasarımı zor
```
**Bu taslakta ilerlemiyoruz — aşırı mühendislik riskli**

---

## 3. Önerilen Tasarım: Yaklaşım A (Quasi-Static Poisson Auxiliary)

### 3.1 Kayıp Fonksiyonu

```
L_total = λ_heat · L_heat  +  λ_ic · L_ic  +  λ_bc · L_bc  +  λ_p · L_poisson

L_heat    = || ρCp·∂T/∂t  -  k·ΔT ||²          ← Level 1-5'te var
L_ic      = || T(x,y,0) - 540°C ||²             ← Level 1-5'te var
L_bc      = || Robin BC koşulu ||²               ← Level 1-5'te var

L_poisson = (1/K) Σ_{k=1}^{K} || ΔT(x,y,t_k) + (ρCp/k)·∂T/∂t|_{t_k} ||²
                                 ↑                 ↑
                              -ΔT olmalı    source term (geçici türev)
```

Zaman dilimleri: K=5 sabit noktada  `t = {0.5, 2, 5, 10, 20}` saniye

### 3.2 Başlangıç Noktası
- Level 5'in en iyi modeli yükleniyor (cosine LR Adam, 20K epoch)
- Küçük lr (`1e-5 → 1e-6`) ile fine-tune ediliyor
- Poisson ağırlığı λ_p kademeli artırılıyor:
  `0 → 0.1 → 0.5 → 1.0` (warm-up stratejisi — ani jump hatalı)

### 3.3 Eğitim Parametreleri
```python
ADAM_EPOCHS    = 5_000     # fine-tune (küçük, Level 5 üstüne ekleniyor)
LR_MAX         = 1e-5      # Level 5 son lr'ından başla
LR_MIN         = 1e-6
K_TIME_SLICES  = 5         # Poisson time slices
T_SLICES       = [0.5, 2.0, 5.0, 10.0, 20.0]  # saniye
LAMBDA_P_FINAL = 1.0       # Poisson ağırlığı son değer
LAMBDA_P_WARMUP= 500       # bu epoch'a kadar lineer artır
```

### 3.4 Neden Bu İşe Yarar?

| Durum | Level 5 Loss | Level 6 Loss Ek |
|-------|-------------|-----------------|
| İç bölge | R_heat küçük ama spatial teremi küçümsüyor | Poisson L_p spatial ΔT'yi ayrıca zorluyor |
| Sınır yakını | Robin BC ağırlıklı | Poisson source term sınır gradyanını da kalibre ediyor |
| t≈0 (başlangıç) | IC loss dominant | Poisson t=0.5s'de neredeyse Laplace → ağı stabilize ediyor |

---

## 4. Oluşturulacak Dosyalar

```
level6_poisson_benchmark/
├── DRAFT.md                  ← bu dosya (güncellenmiş)
├── src/
│   ├── poisson_aux_loss.py   ← Poisson auxiliary loss hesaplama
│   └── level6_finetune.py    ← Level 5 model yükleme + fine-tune
├── main_level6.py            ← 3 optimizer için çalıştırıcı
└── plot_results.py           ← Level 5 vs Level 6 karşılaştırma grafikleri
```

**`poisson_aux_loss.py` ana fonksiyon:**
```python
def poisson_auxiliary_loss(model, t_slices, domain_bounds) -> torch.Tensor:
    """
    Her t=t_k için:
      1. (x,y,t_k) noktaları sample et
      2. ΔT(x,y,t_k) = ∂²T/∂x² + ∂²T/∂y²  hesapla (autograd)
      3. (ρCp/k) · ∂T/∂t hesapla (autograd)
      4. R_poisson = ΔT + (ρCp/k)·∂T/∂t  →  sıfır olmalı
    """
```

---

## 5. Beklenen Sonuçlar

### Quenching Level 5 → Level 6 iyileştirme tahmini:

| Optimizer | Level 5 L2 | Level 6 L2 (tahmini) | İyileşme |
|-----------|-----------|---------------------|---------|
| Bayesian  | 0.030     | 0.020 – 0.025       | ~20-30% |
| NSGA-II   | 0.055     | 0.035 – 0.045       | ~20-30% |
| NSGA-III  | 0.274     | 0.150 – 0.200       | ~25-40% |

> Not: NSGA-III mimarisi küçük (14K params) → kapasitesi yetersiz kalabilir.
> Büyük iyileşme için Yaklaşım B (pretraining) daha iyi olabilir.

---

## 6. Karşılaştırma Grafikler

1. `l6_l2_progression.png`  — Level 1 → 2 → 3 → 4 → 5 → 6 zinciri
2. `l6_vs_l5_heatmap.png`   — Spatial hata haritası: L5 vs L6 yan yana
3. `l6_poisson_residual.png` — Poisson residual R_p eğitim boyunca
4. `l6_summary_table.png`    — Tüm levelların özet tablosu

---

## 7. Uygulama Sırası

```
Adım 1: level6/src/poisson_aux_loss.py
Adım 2: level6/src/level6_finetune.py  (Level 5 model yükle + fine-tune)
Adım 3: level6/main_level6.py          (CLI: --optimizer all)
Adım 4: level6/plot_results.py         (karşılaştırma grafikleri)
```

Çalıştırma komutu (taslak):
```bash
cd /home/coder/NAS-PINNS1/NAS-PINNS3
python level6_poisson_benchmark/main_level6.py --optimizer all
```
