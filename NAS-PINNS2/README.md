# NAS-PINNS2 Quench2026 Rehberi

Bu klasördeki en büyük karmaşa noktası şudur:
- `search` (mimari arama) çıktıları
- `optimizer` (Adam/LBFGS/PSO) çıktıları
- `fair same-arch` (aynı mimaride adil kıyas) çıktıları

Bu README, hangi dosyanın ne anlattığını hızlıca bulman için hazırlanmıştır.

## 1) Kod Haritası

Ana eğitim kodu:
- `naspinn_baseline_with_quench_2026_data.py`

Arama/pipeline giriş noktaları:
- `NAS_PINNs_quench_pipeline.py` (nsga2/nsga3/bayesian birlikte)
- `NAS_PINNs_quench_nsga2.py`
- `NAS_PINNs_quench_nsga3.py`
- `NAS_PINNs_quench_bayesian.py`

Analiz ve yardımcı scriptler:
- `quench2026_winner.py` -> baseline + yöntem winner tablosu
- `quench2026_search_compare.py` -> search algoritması kıyası
- `quench2026_best_adam_lbfgs5000.py` -> her yöntemin en iyi Adam mimarisine LBFGS5000/PSO
- `quench2026_baseline_adam_refine.py` -> baseline için Adam checkpoint'ten LBFGS5000/PSO
- `quench2026_fair_same_arch.py` -> tek mimaride strict fair kıyas (seed bazlı)
- `quench2026_report_pack.py` -> tablo + grafik + markdown rapor paketi
- `quench2026_physical_heatmaps.py` -> fiziksel hata heatmapleri (`|T_pred-T_ref|`, `|u_pred-u_ref|`)

## 2) Sonuç Klasörleri (Ne İçin?)

Kök:
- `results/quench2026`

Ana klasörler:
- `pipeline/`
Bu klasör search + final pipeline çalışmalarıdır (nsga2/nsga3/bayesian).
- `baseline/`
Klasik baseline koşusu (örn. `L5_N96`).
- `best_adam_lbfgs5000/`
Her yöntemin en iyi Adam mimarisine ayrı LBFGS5000 ve PSO iyileştirmeleri.
- `baseline_adam_refine5000/`
Baseline Adam checkpoint'ten LBFGS5000 ve PSO iyileştirmeleri.
- `fair_same_arch/` ve `fair_same_arch_refine/`
Tek sabit mimaride (örn. `L6_N132`) strict adil kıyas (çoklu seed).
- `report_pack/`
Raporlanacak tüm nihai tablo/grafikler.

Deneysel/legacy klasörler:
- `search_fair_v2/`
- `adam_5000/`

## 3) Hangi Soruya Hangi Dosya?

Genel kalite sıralaması (yöntem+mimari):
- `results/quench2026/best_architectures_summary_ranked.csv`

Baseline vs yöntemler (refine sonrası):
- `results/quench2026/best_adam_lbfgs5000/baseline_vs_refined_table.csv`
- `results/quench2026/best_adam_lbfgs5000/comparison_new_refinements.csv`

Search algoritması kıyası (mimari bulma kalitesi):
- `results/quench2026/pipeline/search_algorithm_comparison.csv`
- `results/quench2026/pipeline/search_algorithm_comparison_budget8.csv`

Strict same-architecture optimizer kıyası:
- `results/quench2026/fair_same_arch_refine/L6_N132_per_seed.csv`
- `results/quench2026/fair_same_arch_refine/L6_N132_aggregate_mean_std.csv`

Fiziksel doğruluk (heatmap/MAE/RMSE):
- `results/quench2026/report_pack/physical_heatmaps/physical_accuracy_summary.csv`
- `results/quench2026/report_pack/physical_heatmaps/<scenario>/temp_error_heatmap.png`
- `results/quench2026/report_pack/physical_heatmaps/<scenario>/disp_error_heatmap.png`

Tek yerden okunacak rapor:
- `results/quench2026/report_pack/report.md`

## 4) En Pratik Çalışma Sırası

1. Pipeline search/final:
```bash
python NAS_PINNs_quench_pipeline.py \
  --save-dir results/quench2026/pipeline \
  --methods nsga2,nsga3,bayesian
```

2. En iyi Adam mimarilerini LBFGS5000/PSO ile iyileştir:
```bash
python quench2026_best_adam_lbfgs5000.py --run
```

3. Baseline için aynı işlemi yap:
```bash
python quench2026_baseline_adam_refine.py --run
```

4. Strict fair same-arch kıyası (önerilen):
```bash
python quench2026_fair_same_arch.py \
  --layers 6 --neurons 132 \
  --seeds 42,43,44 \
  --lbfgs-max-iter 5000 \
  --run
```

5. Rapor ve grafikleri üret:
```bash
python quench2026_report_pack.py
python quench2026_physical_heatmaps.py
```

## 5) Hızlı Kontrol Komutları

Aktif süreç var mı:
```bash
pgrep -af 'naspinn_baseline_with_quench_2026_data.py|NAS_PINNs_quench_pipeline.py|quench2026_'
```

Tek bir koşunun logunu canlı izle:
```bash
tail -f results/quench2026/fair_same_arch_refine/seed_42/fair_run.log
```

## 6) Terimler (Karışmasın)

- `search comparison`:
Mimari arama algoritmalarının kıyası (NSGA2/NSGA3/Bayesian).

- `optimizer comparison`:
Aynı mimaride Adam/LBFGS/PSO kıyası.

- `strict fair same-arch`:
Tüm karşılaştırmayı tek mimaride ve çoklu seed ile yapma.

## 7) Bu Projede Öncelikli Okunacak 4 Çıktı

Eğer her şeyi değil sadece karar dosyalarını görmek istiyorsan:
1. `results/quench2026/report_pack/report.md`
2. `results/quench2026/fair_same_arch_refine/L6_N132_aggregate_mean_std.csv`
3. `results/quench2026/best_architectures_summary_ranked.csv`
4. `results/quench2026/report_pack/physical_heatmaps/physical_accuracy_summary.csv`

