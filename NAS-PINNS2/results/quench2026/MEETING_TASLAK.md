# Quench2026 - Hoca Toplantisi Konusma Taslagi

## 1) 30-60 saniyelik acilis
- Bu calismada hedefimiz, quenching problemi icin paper referansini baseline kabul ederek 3 arama yontemini karsilastirmakti: `NSGA2`, `NSGA3`, `Bayesian`.
- Mimarileri bu yontemlerle bulup, egitimde `Adam -> LBFGS -> (opsiyonel) PSO` asamalarini uyguladik.
- Son olarak sonuclari paper referans noktalarinda `exact vs pred` karsilastirmasi ile degerlendirdik.

## 2) Paper'dan hangi verileri kullandik?

### 2.1 Table 1 (A356 flow stress empirical settings)
Kullanilan sicaklik-bagimli parametreler:

| T (C) | F(T) MPa | n(T) | m(T) |
|---:|---:|---:|---:|
| 0 | 22.0 | 0.3 | 0.0 |
| 50 | 21.5 | 0.3 | 0.0 |
| 100 | 21.0 | 0.3 | 0.0 |
| 150 | 20.0 | 0.3 | 0.006 |
| 200 | 18.9 | 0.27 | 0.016 |
| 250 | 17.5 | 0.21 | 0.04 |
| 300 | 16.4 | 0.15 | 0.10 |
| 350 | 16.0 | 0.125 | 0.15 |
| 400 | 15.6 | 0.02 | 0.19 |
| 450 | 12.3 | 0.0 | 0.22 |
| 500 | 7.2 | 0.0 | 0.25 |
| 550 | 2.0 | 0.0 | 0.277 |

### 2.2 Sicaklik referans noktasi (paper baseline)
- Zaman noktasi: `t = [0, 5, 10, 15, 20, 25, 30] s`
- Derinlik noktasi: `depth = [0.0, 0.3, 0.6, 0.9] m`
- Toplam sicaklik referansi: `7 x 4 = 28 nokta`

### 2.3 Distortion (deplasman) referansi
- Katman bazli referans: `[-2.1, -0.8, 0.3, 1.2, 1.8] mm`
- Karsilastirma `x=0` hattinda yapildi
- Toplam deplasman referansi: `5 nokta`

### 2.4 Diger kritik fiziksel sabitler
- Solution-treatment sicakligi: `540 C`
- Su banyosu sicakligi: `25 C`
- Hardening esigi: `420 C`

## 3) Modelde hangi denklemleri/losslari kullandik?

### 3.1 Cikti degiskenleri
Model cikti kanallari:
- `T` (temperature)
- `ux`, `uy` (deplasman bileşenleri)

### 3.2 Isi denklemi reziduali (PINN physics)
- Isi reziduali:
  - `rho * cp * dT/dt - k * (d2T/dx2 + d2T/dy2)`

### 3.3 Viskoplastik surrogate terimi
- Table 1'den interpolasyonla `F(T), n(T), m(T)`
- Surrogate rezidual formu:
  - `sigma_bar - F(T) * phi^n(T) * eps_p_dot^m(T)`

### 3.4 Loss yapisi
- Toplam loss:
  - `L = w_physics*L_physics + w_ic*L_ic + w_bc*L_bc + w_data*L_data`
- Son ayar (guncel):
  - `w_physics=50`
  - `w_ic=1e-3`
  - `w_bc=1e-18`
  - `w_data=1e-2` (yukseltildi)

## 4) Optimizasyonlarda ne yaptik?

### 4.1 Search (mimari arama)
- Yontemler: `NSGA2`, `NSGA3`, `Bayesian`
- Search butcesi (hizlandirilmis):
  - `pop_size=8`, `n_gen=5`
  - `bo_init_points=4`, `bo_iters=5`
- Proxy egitimde genelde Adam kisa kosu ile mimari puanlandi.

### 4.2 Final training/refinement
- Her secilen mimari icin asamalar:
  - `Adam`
  - `LBFGS`
  - `PSO` (opsiyonel)
- Cikti karsilastirmalarinda paper referans noktalarina gore RMSE/MAE hesaplandi.

## 5) Neleri iyilestirdik?
- Cok sayida dağinik rapor dosyasini sadeleştirip tek rapor formatina indirdik.
- Tum eski/yanlis heatmap-grafik ciktilarini temizledik.
- `Exact vs Pred` yan yana heatmap/curve formatini ekledik.
- Kritik duzeltme:
  - `t=0` referans noktasi `IC` ile uyumsuzdu (60-98 C vs 540 C).
  - Bunu cozmeye yonelik yeni mod eklendi: `--temp-ref-t0-mode align_ic`.
- Data etkisini arttirmak icin `w_data` default `1e-5 -> 1e-2` yapildi.

## 6) Su anki karsilastirma ozeti (eldeki checkpointlerle)
- Displacement tarafinda en iyi yakinlik: `NSGA2 + LBFGS5000`.
- Sicaklik tarafinda tum yontemlerde hata hala yuksek (model + referans uyumu daha fazla kalibrasyon istiyor).
- Ozet: mimari arama deplasmanda fayda verdi; sicaklik tarafi icin loss dengesi ve referans modelleme daha fazla iyilestirme gerektiriyor.

## 7) Onemli sinirlar / acik teknik risk
- Sicaklik referansi icin kullandigimiz set paper'daki noktalarin projedeki mevcut temsilidir; birebir fiziksel alan tanimi netlestirilmeli.
- GPU mevcut; ancak calistirma ortaminda GPU baglantisinin dogru dogrulanmasi (device binding) kritik.
- Yeni ayarlar (`align_ic`, `w_data=1e-2`) ile full retrain henuz tamamlanmadi; mevcut sayilar eski checkpointlerden geliyor.

## 8) Hoca sorarsa kisa cevaplar

### Soru: Baseline tam olarak ne?
- Bu calismada baseline iki sekilde ayrildi:
  - **Paper baseline**: referans data noktalarinin kendisi (ground truth).
  - **Model baseline**: sabit mimari (L5_N96) ile egitilen model.
- Karsilastirmayi paper baseline'a gore yaptik.

### Soru: Mimariler nasil bulundu?
- NSGA2/NSGA3/Bayesian search ile layers-neurons uzayinda proxy objective minimizasyonu yapildi.

### Soru: Hangi metrikle kiyas yaptiniz?
- Paper noktalarinda:
  - Temperature RMSE/MAE
  - Displacement RMSE/MAE (x=0)
  - Runtime

### Soru: Neden sicaklik kotu?
- Ana nedenler:
  - t=0 referans-IC uyumsuzlugu (duzeltildi)
  - loss agirlik dengesi (w_data dusuktu, yukseltildi)
  - fiziksel referans tanimi/kalibrasyon ihtiyaci

## 9) Toplanti sonrasi net aksiyon plani
1. Tum kosulari dogrudan GPU bagli calistir (ornek: `CUDA_VISIBLE_DEVICES=0` + logdan `torch.cuda.is_available()` kontrolu).
2. Yeni ayarlarla full rerun:
   - `--temp-ref-t0-mode align_ic`
   - `--w-data 1e-2` (gerekirse 1e-1 ablation)
3. Yeniden exact-vs-pred heatmap + final tablo olustur.
4. Paper baseline'a gore final ranking ve runtime trade-off raporunu guncelle.
