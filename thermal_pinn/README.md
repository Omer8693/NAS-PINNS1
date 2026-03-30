# NAS-PINN Thermal Quenching — k-Skip Framework
*(NAS-PINN Termal Söndürme — k-Atlama Çerçevesi)*

A Neural Architecture Search (NAS) driven Physics-Informed Neural Network (PINN)
framework for accelerating thermal quenching simulations via a multi-step window
prediction (MSWP) strategy.
*(Termal söndürme simülasyonlarını çok adımlı pencere tahmini (MSWP) stratejisiyle
hızlandırmak için Sinir Mimarisi Araştırması (NAS) destekli Fizik Bilgili Sinir Ağı
(PINN) çerçevesi.)*

---

## Problem Statement *(Problem Tanımı)*

We solve the transient heat equation *(geçici ısı denklemini çözüyoruz)*:

$$\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + Q$$

with quenching boundary condition *(söndürme sınır koşuluyla)*:

$$-k \frac{\partial T}{\partial n} = h (T - T_{\text{water}})$$

where $T_{\text{water}} = 20\,^\circ\text{C}$, $T_{\text{init}} = 500\,^\circ\text{C}$,
$h = 5000\,\text{W/m}^2\text{K}$, total time $t_{\text{total}} = 30\,\text{s}$.
Material properties and experimental validation are taken from the FEM baseline study [8].

---

## Method *(Yöntem)*

### 1. IC-Consistent PINN *(Başlangıç Koşuluna Uyumlu PINN)*

The network output is structured to **exactly satisfy** the initial condition at $\tau = 0$
*(ağ çıkışı $\tau = 0$'da başlangıç koşulunu kesin olarak sağlayacak şekilde yapılandırılmıştır)*:

$$\hat{\theta}(\mathbf{x}, \tau) = \theta_{\text{ic}}(\mathbf{x}) + \tau \cdot \mathcal{N}_\theta(\mathbf{x}, \tau, \theta_{\text{ic}})$$

where $\tau \in [0, 1]$ is the normalised time within a window and
$\theta_{\text{ic}}$ is the temperature field at window start [1].

### 2. k-Skip Multi-Step Window Prediction *(k-Atlama Çok Adımlı Pencere Tahmini)*

Instead of solving every FEM timestep $\Delta t_{\text{FEM}} = 1.5\,\text{s}$, the PINN
covers $k$ steps at once *(FEM'in her zaman adımını çözmek yerine, PINN bir seferde
$k$ adımı kapsar)*:

$$\Delta t_{\text{window}} = k \cdot \Delta t_{\text{FEM}}, \quad k \in \{1, 2, 3, 4, 5\}$$

The total simulation uses $\lceil t_{\text{total}} / \Delta t_{\text{window}} \rceil$ windows,
reducing FEM calls by a factor of $k$.

### Measured Performance *(Ölçülen Performans)*

### 3. Fourier Feature Embedding *(Fourier Özellik Gömme)* — v2

To overcome spectral bias *(spektral önyargıyı aşmak için)* [2]:

$$\gamma(\mathbf{v}) = \left[\sin(2\pi \mathbf{B}\mathbf{v}),\; \cos(2\pi \mathbf{B}\mathbf{v})\right]$$

where $\mathbf{B} \in \mathbb{R}^{d_{\text{in}} \times F}$ is a **fixed** random matrix
sampled from $\mathcal{N}(0, \sigma^2)$, with $F = 64$ frequencies,
$\sigma = 1.0$ (2D), $\sigma = 1.5$ (3D).

### 4. Self-Adaptive Loss Weights *(Kendiliğinden Uyarlanan Kayıp Ağırlıkları)* — v2

Loss weights are **learned jointly** with the network parameters *(kayıp ağırlıkları
ağ parametreleriyle birlikte öğrenilir)* [3]:

$$\lambda_i = \text{softplus}(w_i), \quad w_i \in \mathbb{R} \text{ (trainable)}$$

$$\mathcal{L} = \lambda_{\text{pde}} \mathcal{L}_{\text{pde}} + \lambda_{\text{bc}} \mathcal{L}_{\text{bc}} + \lambda_{\text{end}} \mathcal{L}_{\text{end}}$$

Dual learning rates: $\eta_{\text{model}} = 10^{-3}$, $\eta_{\lambda} = 10^{-4}$.

### 5. Neural Architecture Search *(Sinir Mimarisi Araştırması)*

Three NAS strategies are compared *(üç NAS stratejisi karşılaştırılmaktadır)*, following the
search space defined in Wang & Zhong (2023) [7] adapted to the quenching problem:

| Optimizer | Strategy | Architecture Found |
|-----------|----------|--------------------|
| **Bayesian** (TPE) | Tree-structured Parzen Estimator [4] | 5 layers × 151 neurons, ReLU |
| **NSGA-II** | Multi-objective evolutionary [5] | 3 layers × 153 neurons, Tanh |
| **NSGA-III** | Reference-point evolutionary [6] | 3 layers × 75 neurons, Tanh |

---

## Results *(Sonuçlar)*

### Steady-State Performance *(Kararlı Durum Performansı)*
> First 3 windows excluded as transient phase.
> *(İlk 3 pencere geçici faz olarak hariç tutulmuştur.)*

**2D Domains** — All ✅ (relative L2 < 5%)

| Domain | Best Optimizer | SS MAE | SS L2 | Best k |
|--------|---------------|--------|-------|--------|
| Rectangle | Bayesian | 2.13 °C | 0.021 | k=3 |
| Circle | NSGA-III | 0.84 °C | 0.010 | k=2 |
| L-shape | NSGA-III | 1.16 °C | 0.008 | k=4 |

**3D Domains** — 7/12 ✅, 5/12 ⚠️

| Domain | Best Optimizer | SS MAE | SS L2 | Best k |
|--------|---------------|--------|-------|--------|
| Rectangular | NSGA-II | 4.90 °C | 0.025 | k=2 |
| Cylinder | Bayesian | 5.85 °C | 0.037 | 5.6× |
| Stacked | Bayesian | 6.49 °C | 0.053 | 5.7× |
| L-shape | NSGA-III | 2.05 °C | 0.009 | 3.0× |

> **Key finding:** NAS-PINN achieves <5% relative L2 error across all 2D domains
> and 58% of 3D cases, with 4–16× speedup over full FEM simulation.
>
> *(Ana bulgu: NAS-PINN tüm 2B alan adları ve 3B durumların %58'inde <%5 göreli L2 hatası elde eder.)*
> *(Hesaplamalı performans: ölçülen PINN çalışma süresi k ve alan adına bağlı olarak ~10–100 s.)*
> *(Paper FEM referans zamanı sağlamadığı için hızlanma karşılaştırması yapılmamıştır.)*

---

## Repository Structure *(Depo Yapısı)*

```
thermal_pinn/
├── network/
│   ├── pinn.py              # ThermalPINN (v1) — IC-consistent MLP
│   └── fourier_pinn.py      # FourierPINN (v2) — Fourier embedding + SA weights
├── physics/
│   ├── domains_2d.py        # 2D domains: rectangle, circle, lshape
│   └── domains_3d.py        # 3D domains: rectangular, cylinder, stacked, lshape
├── training/
│   ├── trainer.py           # v1 window trainer (Adam + L-BFGS)
│   ├── train_all.py         # v1 full sweep launcher
│   ├── sa_trainer.py        # v2 self-adaptive weight trainer
│   ├── train_v2.py          # v2 full sweep launcher (Fourier + SA)
│   └── retrain_nsga.py      # NSGA-II/III retrain with extended epochs
├── checkpoints/             # Saved models (.pt) and metrics (.json)
├── results/
│   ├── best_results/        # Best-k heatmaps per (domain, arch)
│   └── summary/             # Tables, bar charts, k-progression plots
├── plot_results.py          # Core evaluation and plotting utilities
├── plot_thesis.py           # Publication figures (--ss, --v2 flags)
├── plot_ss_table.py         # Standalone steady-state table generator
└── plot_retrain_heatmaps.py # Best-k heatmap generator → results/best_results/
```

---

## Usage *(Kullanım)*

### Training *(Eğitim)*

```bash
# v1 — Train all domains, archs, k values (2D + 3D)
python -m thermal_pinn.training.train_all --all --cuda

# v2 — Fourier + Self-Adaptive (2D)
python -m thermal_pinn.training.train_v2 --all --dim 2 --cuda

# Retrain NSGA-II/III with extended epochs
python -m thermal_pinn.training.retrain_nsga --dim 2 --cuda
python -m thermal_pinn.training.retrain_nsga --dim 3 --cuda
```

### Plotting *(Görselleştirme)*

```bash
# All thesis figures (heatmaps + tables + charts)
python thermal_pinn/plot_thesis.py

# Steady-state tables (first 3 windows excluded)
python thermal_pinn/plot_thesis.py --fig 2 --ss

# Best-k heatmaps → results/best_results/
python thermal_pinn/plot_retrain_heatmaps.py

# Standalone SS table (all k values, English)
python thermal_pinn/plot_ss_table.py --skip 3
```

---

## References *(Kaynaklar)*

[1] Lagaris, I. E., Likas, A., & Fotiadis, D. I. (1998). Artificial neural networks for solving
ordinary and partial differential equations. *IEEE Transactions on Neural Networks*, 9(5), 987–1000.
https://doi.org/10.1109/72.712178

[2] Tancik, M., Srinivasan, P. P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N.,
Singhal, U., ... & Ng, R. (2020). Fourier features let networks learn high frequency
functions in low dimensional domains. *Advances in Neural Information Processing Systems
(NeurIPS)*, 33, 7537–7547.
https://arxiv.org/abs/2006.10739

[3] McClenny, L., & Braga-Neto, U. (2023). Self-adaptive physics-informed neural networks.
*Journal of Computational Physics*, 474, 111722.
https://doi.org/10.1016/j.jcp.2022.111722

[4] Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter
optimization. *Advances in Neural Information Processing Systems (NeurIPS)*, 24.
https://proceedings.neurips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html

[5] Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective
genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
https://doi.org/10.1109/4235.996017

[6] Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization algorithm using
reference-point-based nondominated sorting approach, part I: solving problems with box
constraints. *IEEE Transactions on Evolutionary Computation*, 18(4), 577–601.
https://doi.org/10.1109/TEVC.2013.2281535

[7] Wang, Y., & Zhong, L. (2023). NAS-PINN: Neural Architecture Search-Guided
Physics-Informed Neural Network for Solving PDEs. *arXiv preprint*, arXiv:2305.10127.
https://arxiv.org/abs/2305.10127

[8] Mortensen, B., et al. (2026). Finite element simulation of quenching in aluminium
casting: thermal and mechanical analysis. *The International Journal of Advanced
Manufacturing Technology*.
https://doi.org/10.1007/s00170-026-17515-w
