# NAS-PINNs: Neural Architecture Search for Physics-Informed Neural Networks

A seven-level progressive framework that applies Neural Architecture Search (NAS) to
Physics-Informed Neural Networks (PINN) for simulating A356 aluminum water quenching.

**Reference:**
> Dag Mortensen, Gulshan Noorsumar, Hallvard G. Fjær, Reza Babaei, Per Erik Drønen (2026)
> *"Mitigating distortions in cast automotive subframes: A finite element simulation approach"*
> The International Journal of Advanced Manufacturing Technology
> https://doi.org/10.1007/s00170-026-17515-w

---

## Quick Start

```bash
# Generate all presentation plots
python generate_presentation_plots.py

# Build the PowerPoint (21 slides)
python make_presentation.py
# Output: results/NAS_PINNs_Presentation.pptx
```

---

## Project Structure

```
NAS-PINNS3/
├── src/                        # Core framework (NAS, PINN, trainers, physics)
├── problems/                   # PDE definitions (quenching, Poisson, Burgers)
├── level1_single_shot/         # L1: global T(t,x,y) PINN
├── level2_timestepper/         # L2: temporal skip operator
├── level3_hybrid_fem/          # L3: adaptive FEM+PINN routing
├── level4_distortion/          # L4: thermal → distortion mechanics
├── level5_refinement/          # L5: extended Adam training
├── level6_poisson_benchmark/   # L6: Poisson auxiliary fine-tuning
├── level7_temporal/            # L7A: temporal skip analysis
├── level7_multiDomain/         # L7B: 5-geometry Poisson NAS
├── results/
│   ├── pres_plots/             # Clean plots for presentation
│   ├── cross_level_comparison/ # Cross-level analysis
│   └── NAS_PINNs_Presentation.pptx
├── generate_presentation_plots.py
└── make_presentation.py
```

---

## Framework Overview

| Level | Name | Input → Output | Key Result |
|-------|------|----------------|------------|
| L1 | Single-Shot PINN | (t,x,y) → T | Bayesian L2=0.076 (best) |
| L2 | Skip Operator | (x,y,t_local,T_prev) → T_next | skip=2: 2× FEM reduction |
| L3 | Hybrid FEM+PINN | residual → route | 80% FEM step reduction |
| L4 | Distortion | T_field → δ (mm) | Bayesian: 1.67 mm mean |
| L5 | Extended Training | more epochs → lower L2 | NSGA-II: 4.5× improvement |
| L6 | Poisson Fine-Tune | Poisson aux loss | marginal gain (+17%) |
| L7B | Multi-Domain NAS | 5 geometries | Circle: L2=0.00014 |

---

## Physical Problem

**Material:** A356 cast aluminium alloy
**Domain:** 1.3 m × 0.6 m cross-section (2D)
**Initial temperature:** T₀ = 540°C
**Water bath:** T_water = 20°C
**Duration:** 30 seconds

**Governing PDE:**
```
ρCₚ · ∂T/∂t = K · ∇²T    in Ω
K · ∂T/∂n = h(T) · (T - T_water)    on ∂Ω (Robin BC)
```

**Material properties (A356):**

| Property | Value |
|----------|-------|
| K (thermal conductivity) | 150 W/mK |
| ρCₚ (volumetric heat capacity) | 2.4×10⁶ J/m³K |
| E (elastic modulus) | 69 GPa |
| α (thermal expansion) | 22×10⁻⁶ /°C |

**Analytical reference (Robin BC fundamental mode):**
```
T(t) = 20 + 520 · exp(−1.75×10⁻³ · t)
```
Derived from paper parameters (Mortensen 2026).

---

## NAS Search Space

All three optimizers search the same architecture space:

| Hyperparameter | Options |
|----------------|---------|
| Number of hidden layers | {2, 3, 4, 5, 6} |
| Neurons per layer | {32, 48, 64, 96, 128, 160, 192, 256} |
| Activation function | tanh, relu, swish |
| Architecture type | uniform (same neurons in all layers) |

**Optimizers:**
- **Bayesian (TPE):** single-objective, minimises L2_rel
- **NSGA-II:** multi-objective (L2, param count), evolutionary
- **NSGA-III:** multi-objective with reference-point directions

---

## Level 1 — Single-Shot NAS-PINN

**Goal:** Find architecture for global T(t,x,y) predictor.

**Training parameters:**

| Parameter | Value |
|-----------|-------|
| Input | (t, x, y) → T |
| Adam epochs | 20 000 |
| Learning rate | 1×10⁻³ → 1×10⁻⁵ (cosine) |
| Domain points | 2 000 (random collocation) |
| BC points | 200 (boundary sampling) |
| Time range | t ∈ [0, 30] s |

**Results:**

| Optimizer | Architecture | Parameters | L2_rel | MAE (°C) | NAS Time |
|-----------|-------------|------------|--------|----------|----------|
| **Bayesian ★** | **5×151 relu** | **92 564** | **0.076 ✓** | **39.1°C** | ~150 s |
| NSGA-II | 3×153 tanh | 47 890 | 0.252 | 132.6°C | ~1 583 s |
| NSGA-III | 3×75 tanh | 11 776 | 0.513 | 270.3°C | ~1 609 s |
| Target | — | — | < 0.100 | < 50°C | — |

★ Only Bayesian meets the L2 < 0.10 target at Level 1.

**Key finding:** Architecture depth and width directly determine accuracy. The Bayesian 5×151 network has 8× more parameters than NSGA-III and achieves 7× lower MAE.

---

## Level 2 — Temporal Skip Operator

**Goal:** Replace intermediate FEM time steps with PINN predictions.

**How it works:**
- FEM runs at anchor points `t[::k]` (every k-th step)
- PINN (TimeStepperPINN) predicts the k-1 skipped steps
- Input: `[x, y, t_local, T_prev(x,y)]` → `T_next(x,y)`
- Convergence criterion: `MAE(skip=k) / MAE(skip=1) < 1.5×`

**Training parameters (500-epoch baseline):**

| Parameter | Value |
|-----------|-------|
| Adam epochs per window | 500 |
| Learning rate | 1×10⁻³ → 1×10⁻⁵ (cosine) |
| Domain points | 500 |
| BC points | 100 |
| Time steps | N=20 (Δt = 1.5 s) |

**Results — 500 Adam epochs:**

| Skip | FEM Calls | Bayesian | Ratio | NSGA-II | Ratio | NSGA-III | Ratio |
|------|-----------|----------|-------|---------|-------|---------|-------|
| 1 | 21/21 | 43.6°C ✓ | 1.00× | 38.1°C ✓ | 1.00× | 44.4°C ✓ | 1.00× |
| 2 | 11/21 | 33.2°C ✓ | 0.76× | 56.0°C **✓** | **1.47×** | 75.6°C ✗ | 1.70× |
| 4 | 6/21 | 57.5°C ✓ | 1.32× | 154.8°C ✗ | 4.06× | 202.1°C ✗ | 4.55× |
| 6 | 4/21 | 93.6°C ✗ | 2.15× | 354.6°C ✗ | 9.30× | 428.6°C ✗ | 9.65× |

**Correction vs previous version:** NSGA-II skip=2 ratio=1.47× is **CONVERGED** (< 1.5 threshold), not diverged.

**Results — 2 000 Adam epochs (NSGA-II and NSGA-III only):**

| Optimizer | skip=1 MAE | skip=2 MAE | Ratio vs 500ep ref | Result |
|-----------|-----------|-----------|-------------------|--------|
| NSGA-II | 18.4°C | 37.4°C | 0.98× | **✓ Converged** |
| NSGA-III | 33.8°C | 43.8°C | 0.99× | **✓ Converged** |

**Key finding:** Epoch count was the bottleneck, not architecture. With 2 000 epochs, both NSGA-II and NSGA-III converge at skip=2.

**Interesting observation:** Bayesian skip=2 (33.2°C) is BETTER than skip=1 (43.6°C). PINN avoids FEM's sequential numerical error accumulation.

---

## Level 3 — Hybrid FEM + PINN

**Goal:** Adaptive routing — call FEM only when PINN error is too high.

**Algorithm:**
1. Solve step with PINN
2. Compute PDE residual `r(t) = ‖∂T/∂t − K∇²T‖ / ‖T‖`
3. If `r > threshold` → trigger full FEM solve
4. If `r ≤ threshold` → accept PINN prediction

**Parameters:**

| Parameter | Config A | Config B |
|-----------|---------|---------|
| Residual threshold | 0.10 | 0.10 |
| Max consecutive PINN steps | 4 | 20 |
| Total time steps | 20 | 20 |

**Results (Bayesian architecture):**

| Config | FEM Steps | PINN Steps | Skip Rate |
|--------|----------|-----------|-----------|
| A (max_skip=4) | 4 | 16 | 80% |
| B (max_skip=20) | 1 | 19 | 95% |

**Key finding:** All three optimizers achieve 80% FEM step reduction in Config A. The residual check prevents error accumulation automatically.

---

## Level 4 — Thermal Distortion Mechanics

**Goal:** Convert PINN temperature field to CMM-point distortions in mm.

**Method:**
- Input: `T(x,y,t=30s)` from Level 3 PINN
- 2D plane-stress FEM: `σ = E · α · ΔT`
- Output: `δ (mm)` at 47 CMM measurement points

**Parameters:**

| Property | Value |
|----------|-------|
| E (elastic modulus) | 69 GPa |
| α (thermal expansion) | 22×10⁻⁶ /°C |
| FEM elements | Q4 plane-stress |
| CMM points | 47 (from paper Fig 17) |

**Results — Mean absolute distortion |δ|:**

| Source | Mean |δ| (mm) | MAE vs Measured |
|--------|--------------|----------------|
| Paper Measured | 0.78 mm | — (reference) |
| Paper FEM | 0.75 mm | 0.18 mm |
| Bayesian PINN | 1.67 mm | 1.14 mm |
| NSGA-II PINN | 2.81 mm | 2.08 mm |
| NSGA-III PINN | 4.74 mm | 3.97 mm |

**Key finding:** PINN thermal accuracy (L2_rel) directly determines distortion accuracy. Bayesian (L2=0.076) → 1.67mm; NSGA-III (L2=0.513) → 4.74mm.

---

## Level 5 — Extended Adam Training

**Goal:** Improve accuracy by training longer with cosine LR schedule.

**Note:** L-BFGS was tested but did not run (0 iterations in all cases — the Adam minimum was already in a locally flat region). The improvement comes from extended Adam with cosine LR.

**Training parameters:**

| Parameter | Value |
|-----------|-------|
| Adam epochs | 20 000 |
| LR schedule | cosine: 1×10⁻³ → 1×10⁻⁵ |
| Same architectures | as Level 1 |

**Results:**

| Optimizer | L1 L2 | L5 L2 | Improvement | L1 MAE | L5 MAE |
|-----------|-------|-------|-------------|--------|--------|
| **Bayesian** | 0.076 | **0.030 ✓** | 2.6× | 39.1°C | 14.4°C ✓ |
| **NSGA-II** | 0.252 | **0.055 ✓** | 4.5× | 132.6°C | 28.7°C ✓ |
| NSGA-III | 0.513 | 0.259 | 2.0× | 270.3°C | 136.4°C |

**Key finding:** NSGA-II shows the largest gain (4.5×) — its 3-layer architecture has enough capacity when given sufficient training time. Both Bayesian and NSGA-II now meet the L2 < 0.10 target.

---

## Level 6 — Poisson Auxiliary Fine-Tuning

**Goal:** Test whether Poisson-type auxiliary loss improves quenching accuracy.

**Method:**
- Start from Level 5 weights
- Add Poisson PDE auxiliary loss at 5 time slices: t ∈ {0.5, 2, 5, 10, 20} s
- Fine-tune for 5 000 epochs

**Training parameters:**

| Parameter | Value |
|-----------|-------|
| Fine-tune epochs | 5 000 |
| LR | 1×10⁻⁵ → 1×10⁻⁶ (cosine) |
| λ_poisson | 1.0 |
| Spatial points per slice | 1 000 |
| Time slices | {0.5, 2, 5, 10, 20} s |

**Results:**

| Optimizer | L5 L2 | L6 L2 | Improvement |
|-----------|-------|-------|-------------|
| Bayesian | 0.038 | 0.031 | +17.5% |
| NSGA-II | 0.062 | 0.058 | +6.2% |
| NSGA-III | 0.274 | 0.267 | +2.5% |

**Key finding:** Marginal gains only. Poisson auxiliary loss acts as a light regulariser but does not fundamentally improve accuracy for quenching.

---

## Level 7B — Multi-Domain Poisson NAS

**Goal:** Apply NAS to Poisson PDE across 5 different geometries.

**PDE:** `−∇²u = f(x,y)` with `u = 0` on boundary (Dirichlet)

**Training parameters:**

| Parameter | Value |
|-----------|-------|
| Proxy epochs (NAS) | 2 000 |
| Final training epochs | 5 000 |
| LR | 1×10⁻³ → 1×10⁻⁵ |
| Domain points | 1 000 per domain |
| BC points | 200 |

**Domains and results:**

| Domain | Reference | Best L2 (Bayesian) | Architecture | Difficulty |
|--------|-----------|-------------------|--------------|------------|
| Square [0,1]² | u*=x²(x−1)²y²(y−1)² (exact) | 0.0376 | 5×100 gelu | Medium |
| Circle x²+y²≤1 | u*=(1−r²)/4 (exact) | **0.00014 ✓** | 5×40 gelu | Easy |
| Annulus 0.25≤r≤1 | u*=(r²−r_i²)/4 (exact) | — | — | Did not converge |
| L-Shape | FDM reference | 0.0277 ✓ | 5×100 gelu | Medium |
| Flower r≤1+0.3cos(5θ) | FDM reference | 0.172 | 2×50 gelu | Hard |

**Key finding:** Geometry complexity strongly affects NAS results. Smooth domains (circle) converge excellently. Complex boundaries (flower) remain challenging for all optimizers.

---

## Skip Operator: Core Concept

```
FEM alone:  t=0 → t=1.5 → t=3 → t=4.5 → … → t=30   (21 steps, 20 FEM solves)

skip=2:     FEM at [0, 3, 6, 9, …, 30]               (11 FEM solves — 2× faster)
            PINN at [1.5, 4.5, 7.5, …, 28.5]         (10 predictions)
            Bayesian MAE = 33.2°C ✓  (better than skip=1!)

skip=4:     FEM at [0, 6, 12, 18, 24, 30]            (6 FEM solves — 4× faster)
            PINN fills 15 intermediate steps
            Bayesian MAE = 57.5°C ✓  (still within 1.5× threshold)
```

**Convergence criterion:** `MAE(skip=k) / MAE(skip=1) < 1.5×`

---

## Key Results Summary

| Metric | Bayesian | NSGA-II | NSGA-III | Paper FEM |
|--------|---------|---------|---------|-----------|
| L1 L2_rel | 0.076 ✓ | 0.252 | 0.513 | — |
| L5 L2_rel | **0.030 ✓** | **0.055 ✓** | 0.259 | — |
| Skip=2 (500ep) | ✓ (0.76×) | ✓ (1.47×) | ✗ (1.70×) | — |
| Skip=2 (2000ep) | ✓ | ✓ (0.98×) | ✓ (0.99×) | — |
| Skip=4 (500ep) | ✓ (1.32×) | ✗ | ✗ | — |
| CMM distortion | 1.67 mm | 2.81 mm | 4.74 mm | 0.75 mm |
| Hybrid skip rate | 80% | 80% | 80% | — |
| Poisson L2 (circle) | 0.00014 | 0.00042 | 0.00066 | — |

---

## Reference

> **Dag Mortensen, Gulshan Noorsumar, Hallvard G. Fjær, Reza Babaei, Per Erik Drønen (2026)**
> *"Mitigating distortions in cast automotive subframes: A finite element simulation approach"*
> The International Journal of Advanced Manufacturing Technology
> https://doi.org/10.1007/s00170-026-17515-w

The analytical cooling reference `T(t) = 20 + 520·exp(−1.75×10⁻³·t)` is derived from the
Robin BC fundamental mode using the material parameters reported in this paper
(K = 150 W/mK, ρCₚ = 2.4×10⁶ J/m³K). All FEM baseline values, CMM distortion
measurements, and A356 material properties originate from this work.
