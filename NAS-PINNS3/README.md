# NAS-PINN: Neural Architecture Search for Physics-Informed Neural Networks
### (Fizik Bilgili Sinir Ağları için Sinir Mimarisi Arama)

A 4-level framework for automatic neural architecture search applied to the aluminum subframe water quenching problem from baseline_paper [2026].
(A356 alüminyum subframe su quenching problemine uygulanan 4 kademeli otomatik sinir mimarisi arama çerçevesi.)

---

## Problem (Problem)

Modeling the thermal field during **water quenching** of an A356 aluminum subframe casting.
(A356 alüminyum subframe dökümünün su quenching sürecindeki termal alanın modellenmesi.)

- Domain (Alan): 1.3 m × 0.6 m, 2D cross-section
- Time (Süre): 0 → 30 s
- Initial temperature (Başlangıç sıcaklığı): 540 °C (SHT exit / SHT çıkışı)
- Water temperature (Su sıcaklığı): 20 °C
- Reference (Referans): baseline_paper [2026], Table 1 — A356 material constants (A356 malzeme sabitleri)

---

## Architecture — 4 Levels (Mimari — 4 Kademe)

```
Level 1  →  Level 2  →  Level 3  →  Level 4
Global       Skip         Hybrid       Distortion
PINN         Operator     FEM+PINN     Mechanics
```

### Level 1 — Single-Shot PINN (Tek Seferlik PINN)

A single global PINN trained over the entire time domain [0, T_END].
(Tüm zaman aralığı [0, T_END] üzerinde eğitilen tek global PINN.)

NAS searches over: layers ∈ {3–6}, neurons ∈ {64–160}, activation ∈ {tanh, relu, sin, gelu, swish}.
Three optimizers compared: **Bayesian**, **NSGA-II**, **NSGA-III**.

| Optimizer | Architecture | L2_rel ↓ | MAE (°C) ↓ | NAS time (s) |
|-----------|-------------|-----------|------------|--------------|
| **Bayesian** ★ | 5×151 relu | **0.076** | **39.1** | 150 |
| NSGA-II | 3×153 tanh | 0.252 | 132.6 | 1583 |
| NSGA-III | 3×75 tanh | 0.513 | 270.3 | 1609 |
| ★ Target (Hedef) | — | < 0.100 | < 50.0 | — |

Best model (En iyi model): **Bayesian — 5×151 relu** (L2=0.076, MAE=39°C ✓)

### Level 2 — Time-Stepping Operator (Zaman Adım Operatörü)

Learns the mapping T_n → T_{n+k} to enable temporal skipping.
(T_n → T_{n+k} eşlemesini öğrenerek zaman adım atlamayı mümkün kılar.)

Input: (x, y, t_local_norm, T_prev_norm), Output: T_next.
Evaluated for skip ∈ {1, 2, 4, 6}.

| Skip k | Steps used | L2_rel | MAE (°C) | Runtime (s) | Speedup |
|--------|-----------|--------|----------|-------------|---------|
| 1 | 21/21 | 0.126 | 43.6 | 183.8 | 1.0× |
| **2** | 11/21 | **0.108** | **33.2** | **90.5** | **2.0×** |
| 4 | 6/21 | 0.204 | 57.5 | 46.1 | 4.0× |
| 6 | 4/21 | 0.294 | 93.6 | 27.7 | 6.6× |

Best trade-off (En iyi denge): **skip=2** — 2× speedup with better accuracy than skip=1.

### Level 3 — Hybrid FEM + PINN (Hibrit FEM + PINN)

Alternates between FEM anchors and PINN interpolation based on PDE residual.
(PDE rezidüeli temelinde FEM ankorlar ve PINN interpolasyon arasında geçiş yapar.)

- FEM triggered when (FEM tetiklenir): residual > threshold (0.1)
- PINN used otherwise (aksi durumda PINN): faster but less accurate
- Adaptive skip (Uyarlanabilir atlama): max_skip=4 or max_skip=20

Results (threshold=0.1, max_skip=4):
(Sonuçlar, eşik=0.1, maks_atlama=4):

| Optimizer | FEM steps | PINN steps | Skip rate | Wall time (s) |
|-----------|----------|------------|-----------|---------------|
| Bayesian | 4/20 | 16/20 | 80% | 111 s |
| NSGA-II | 4/20 | 16/20 | 80% | 93 s |
| NSGA-III | 4/20 | 16/20 | 80% | 92 s |

80% of steps handled by PINN — significant cost reduction vs. full FEM.
(Adımların %80'i PINN tarafından işlendi — tam FEM'e kıyasla önemli maliyet azalması.)

### Level 4 — Distortion Mechanics (Distorsiyon Mekaniği)

2D plane-stress FEM converts T(x,y) → displacement u(x,y) → |δ| at 24 CMM measurement points.
(2D düzlem gerilme FEM, T(x,y)'yi → yer değiştirme u(x,y) → 24 CMM ölçüm noktasında |δ|'ye dönüştürür.)

- Element type (Eleman tipi): Q4 bilinear isoparametric
- Integration (İntegrasyon): 2×2 Gauss
- Solver (Çözücü): Conjugate Gradient (Konjuge Gradyan)
- Boundary conditions (Sınır koşulları): Penalty method — rigid body motion suppressed

| Source | Mean |δ| (mm) | Max |δ| (mm) | MAE vs. measured (mm) |
|--------|--------------|-------------|----------------------|
| Paper FEM (baseline_paper Fig18) | 0.781 | 2.10 | 0.179 (internal) |
| PINN — Bayesian | 1.67 | 2.45 | 1.14 |
| PINN — NSGA-II | 1.42 | 2.10 | 0.97 |
| PINN — NSGA-III | 0.94 | 1.56 | 0.71 |

CMM point data digitized from paper (Kağıttan sayısallaştırılan CMM verisi): 24 points, Bottom layer, ±0.05 mm precision.

---

## Repository Structure (Depo Yapısı)

```
NAS-PINNS3/
├── main.py                         # Top-level CLI entry point (Üst düzey CLI giriş noktası)
├── run_quenching_tuned_queue.sh    # Batch experiment runner (Toplu deney çalıştırıcı)
│
├── src/                            # Core NAS-PINN framework (Temel NAS-PINN çerçevesi)
│   ├── config.py                   # Global constants: material, domain, NAS search space
│   ├── pinn_network.py             # PINNNet: MLP with configurable activations
│   ├── physics_model.py            # Heat equation residuals, HTC boiling curve
│   ├── trainers.py                 # Adam / L-BFGS / PSO three-phase training
│   ├── baseline_data.py            # Paper data: Tables 1-2, Figures 7, 15-22
│   ├── arch_search.py              # NAS infrastructure: decode, evaluate, pymoo interface
│   ├── opt_nsga2.py                # NSGA-II (pop=24, gen=16)
│   ├── opt_nsga3.py                # NSGA-III (ref_dirs=10)
│   ├── opt_bayesian.py             # Bayesian Optimization (4 init + 12 iter)
│   └── experiment_runner.py        # Orchestration: single and multi-optimizer runs
│
├── problems/                       # Benchmark problem definitions (Kıyaslama problem tanımları)
│   ├── base.py                     # Abstract PINNProblem base class
│   ├── quenching.py                # A356 aluminum quenching (main problem / ana problem)
│   ├── burgers.py                  # 1D Burgers equation
│   ├── poisson.py                  # 2D Poisson equation
│   └── allen_cahn.py               # 1D+t Allen-Cahn equation
│
├── level1_single_shot/             # Level 1: Global PINN (Küresel PINN)
│   ├── plot_results.py             # Visualize comparison.json → 3 figures + JSON
│   └── README.md                   # Level 1 quick reference
│
├── level2_timestepper/             # Level 2: Skip operator PINN (Atlama operatörü PINN)
│   ├── main_level2.py              # CLI: generate skip_table.json
│   ├── plot_results.py             # Skip vs L2 / runtime curves
│   └── src/
│       ├── ts_model.py             # TimeStepperPINN: 4-input MLP
│       ├── ts_trainer.py           # Window-based training with teacher forcing
│       ├── ts_nas.py               # Grid-search NAS for optimal skip factor
│       └── ts_evaluate.py          # Evaluate skip ∈ {1,2,4,6,...}
│
├── level3_hybrid_fem/              # Level 3: Adaptive hybrid loop (Uyarlanabilir hibrit döngü)
│   ├── main_level3.py              # CLI: run hybrid for 3 architectures
│   ├── plot_results.py             # Residual trace, step distribution, CMM bars
│   └── src/
│       ├── fem_interface.py        # FEMCheckpoint: save/load T-field snapshots
│       ├── hybrid_runner.py        # Main FEM ↔ PINN alternation loop
│       ├── adaptive_skip.py        # Residual-based FEM/PINN selector
│       └── mechanical.py          # Thermal strain → CMM distortion estimate
│
├── level4_distortion/              # Level 4: 2D FEM distortion (2D FEM distorsiyonu)
│   ├── main_distortion.py          # CLI: T_field → FEM solve → CMM |δ|
│   ├── plot_results.py             # CMM bar chart, displacement components, map
│   ├── plot_paper_comparison.py    # Signed bar chart vs paper Fig17/18
│   └── src/
│       ├── plane_stress_fem.py     # PlaneStressFEM: Q4 elements, CG solver
│       ├── thermal_field.py        # Analytical T(x,y) profiles: gradient/cosine/parabolic
│       └── cmm_points.py           # 24 CMM labels + paper reference values
│
└── results/                        # All output files (Tüm çıktı dosyaları)
    ├── baseline/                   # Paper reference images (Referans makale görüntüleri)
    ├── run2/                       # Level 1 results: comparison.json + model weights
    │   ├── comparison.json
    │   ├── plots/                  # level1_accuracy.png, runtime.png, summary_table.png
    │   └── quenching/{bayesian,nsga2,nsga3}/
    │       ├── best_arch.json
    │       ├── results.json
    │       └── model.pt
    ├── level2_timestepper/results/ # skip_table.json + curves
    ├── level3_hybrid_fem/results/  # thr0.1_skip{4,20}/ — history + PNGs
    └── level4_distortion/results/  # level4_distortion.json + CMM PNGs + all_results_summary.json
```

---

## Quick Start (Hızlı Başlangıç)

### Run all 3 optimizers — Level 1 (3 optimizer çalıştır — Kademe 1)
```bash
cd NAS-PINNS3
python main.py --compare_all --problem quenching
```

### Run a single optimizer (Tek optimizer çalıştır)
```bash
python main.py --optimizer bayesian --problem quenching --time_mode full
python main.py --optimizer nsga2    --problem quenching --time_mode fixed --fixed_skip 2
python main.py --optimizer nsga3    --problem quenching --time_mode adaptive
```

### Level 1 — Visualize results (Sonuçları görselleştir)
```bash
python level1_single_shot/plot_results.py
# Output (Çıktı): results/run2/plots/
```

### Level 2 — Time-stepping (Zaman adımlama)
```bash
python level2_timestepper/main_level2.py --skip_values 1 2 4 6
python level2_timestepper/plot_results.py
```

### Level 3 — Hybrid FEM+PINN (Hibrit FEM+PINN)
```bash
python level3_hybrid_fem/main_level3.py --threshold 0.1 --max_skip 4
python level3_hybrid_fem/plot_results.py
```

### Level 4 — Distortion mechanics (Distorsiyon mekaniği)
```bash
python level4_distortion/main_distortion.py --nx 20 --ny 10 --profile gradient
python level4_distortion/plot_results.py
python level4_distortion/plot_paper_comparison.py
# Output (Çıktı): level4_distortion/results/
```

### Extract paper baseline data (Makale temel verilerini çıkar)
```bash
python main.py --baseline_only
# Output (Çıktı): results/baseline/
```

---

## NAS Search Space (NAS Arama Uzayı)

| Parameter (Parametre) | Range (Aralık) |
|----------------------|----------------|
| Hidden layers (Gizli katmanlar) | 3 – 6 |
| Neurons per layer (Katman başına nöron) | 64 – 160 |
| Activation function (Aktivasyon fonksiyonu) | tanh, relu, sin, gelu, swish |
| Optimizer (Optimizer) | Bayesian, NSGA-II, NSGA-III |

Training pipeline (Eğitim hattı): **Adam** (bulk) → **L-BFGS** (refinement) → **PSO** (global search)

---

## Material Properties — A356 Aluminum (Malzeme Özellikleri — A356 Alüminyum)
*Source (Kaynak): baseline_paper [2026], Table 1*

| Property (Özellik) | Value (Değer) |
|--------------------|---------------|
| Thermal conductivity (Isıl iletkenlik) K | 151 W/m·K |
| Volumetric heat capacity (Hacimsel ısı kapasitesi) ρ·Cₚ | 2.43×10⁶ J/m³·K |
| Elastic modulus (Elastisite modülü) E | 70 GPa |
| Poisson's ratio (Poisson oranı) ν | 0.33 |
| Thermal expansion (Termal genleşme) β | 2.34×10⁻⁵ /K |
| Stress-free temperature (Stres referans sıcaklığı) T_ref | 540 °C |

---

## Dependencies (Bağımlılıklar)

```bash
pip install torch numpy scipy matplotlib pymoo bayesian-optimization
```

| Package | Purpose (Amaç) |
|---------|----------------|
| `torch` | Neural network training (Sinir ağı eğitimi) |
| `numpy`, `scipy` | Numerical FEM solver (Sayısal FEM çözücü) |
| `matplotlib` | Visualization (Görselleştirme) |
| `pymoo` | NSGA-II / NSGA-III multi-objective optimization (Çok amaçlı optimizasyon) |
| `bayesian-optimization` | Gaussian Process NAS (Gauss Süreci NAS) |

---

## Reference (Referans)

> baseline_paper [2026] — *Water quenching simulation of A356 aluminum subframe:
> FEM modeling, distortion prediction, and CMM validation.*
> (A356 alüminyum subframe su quenching simülasyonu: FEM modelleme, distorsiyon tahmini ve CMM doğrulaması.)
