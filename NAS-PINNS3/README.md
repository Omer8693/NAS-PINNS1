# NAS-PINNs: Neural Architecture Search for Physics-Informed Neural Networks

A **7-level progressive framework** applying NAS to thermal simulation of A356 aluminum water quenching — with temporal skip operator, hybrid FEM+PINN routing, L-BFGS refinement, and multi-domain generalization.

---

## Core Thesis

> **PINN with NAS-optimized architecture can replace sequential FEM time steps — solving in fewer solver calls without accuracy loss.**

FEM requires 20 mandatory sequential solves (t = 1.5 s each). The NAS-PINN skip operator reduces this to 11 solves (skip=2, 2× speedup) for the Bayesian architecture, with equal or better accuracy.

---

## Problem

**A356 aluminum subframe water quenching** (baseline_paper [2026]):

| Parameter | Value |
|-----------|-------|
| Domain | 1.3 m × 0.6 m (2D cross-section) |
| Time range | 0 → 30 s |
| Initial temperature T₀ | 540 °C |
| Water temperature T_w | 20 °C |
| PDE | ρCp·∂T/∂t = K·∇²T + Robin BC |
| HTC | Nonlinear h(T) — nucleate/film boiling |

---

## Seven-Level Framework

```
L1 → L2 → L3 → L4 → L5 → L6 → L7
```

| Level | Name | Description |
|-------|------|-------------|
| **1** | Single-Shot PINN | Global T(t,x,y) trained over entire time domain |
| **2** | Skip Operator | Window PINN: FEM at anchors, PINN fills skipped steps |
| **3** | Hybrid FEM+PINN | Adaptive residual-driven routing between FEM and PINN |
| **4** | Distortion Mechanics | Plane-stress FEM: T-field → CMM distortion |
| **5** | L-BFGS Refinement | Second-order polish applied after Adam convergence |
| **6** | Poisson Benchmark | Generalization test on a different PDE family |
| **7** | Multi-Domain | 7A: temporal skip analysis · 7B: 5-geometry Poisson NAS |

---

## Key Results

### Level 1 — Single-Shot NAS-PINN

| Optimizer | Architecture | L2_rel ↓ | MAE (°C) ↓ | NAS time |
|-----------|-------------|-----------|------------|----------|
| **Bayesian** ★ | 5×151 relu | **0.076** | **39.1** | ~180 s |
| NSGA-II | 3×153 tanh | 0.252 | 132.6 | ~120 s |
| NSGA-III | 3×75 tanh | 0.513 | 270.3 | ~110 s |
| Target | — | < 0.100 | < 50.0 | — |

### Level 2 — Temporal Skip Operator (Bayesian architecture)

| Skip k | FEM Steps | L2_rel | MAE (°C) | Speedup |
|--------|-----------|--------|----------|---------|
| 1 | 21/21 | 0.126 | 43.6 | 1.0× |
| **2** ★ | **11/21** | **0.108** | **33.2** | **2.0×** |
| 4 | 6/21 | 0.204 | 57.5 | 4.0× |
| 6 | 4/21 | 0.294 | 93.6 | 6.6× |

**Only Bayesian converges at skip=2** (MAE ratio 0.76×). NSGA-II ratio=1.47×, NSGA-III ratio=1.76× (threshold: < 1.5×).

### Level 3 — Hybrid FEM+PINN (all optimizers)

| Optimizer | FEM steps | PINN steps | Skip rate |
|-----------|-----------|------------|-----------|
| All three | 4/20 | 16/20 | **80%** |

Adaptive residual routing achieves 80% FEM reduction for all architectures.

### Level 5 — L-BFGS Refinement

| Optimizer | Adam L2 | L-BFGS L2 | Improvement |
|-----------|---------|-----------|-------------|
| Bayesian | 0.076 | **0.030** | 2.6× |
| NSGA-II | 0.252 | **0.055** | 4.5× |
| NSGA-III | 0.513 | 0.259 | 2.0× |

### Level 6 — Poisson PDE Benchmark

| Optimizer | L2_rel |
|-----------|--------|
| Bayesian | **0.0083** ✓ |
| NSGA-II | 0.0412 ✓ |
| NSGA-III | 0.1890 (marginal) |

Same NAS architecture generalizes to a different PDE without re-search.

### Level 4 — Distortion at CMM Points

| Source | Mean |δ| (mm) |
|--------|--------------|
| Bayesian PINN | 1.67 |
| NSGA-II PINN | 2.81 |
| NSGA-III PINN | 4.74 |

---

## Repository Structure

```
NAS-PINNS3/
├── main.py                              # Top-level CLI entry point
├── make_presentation.py                 # Generate PowerPoint (12 slides)
├── compare_all_levels.py                # Cross-level comparison pipeline
├── finalize_after_5k.py                 # Update plots after 5000-epoch run
│
├── src/                                 # Core NAS-PINN framework
│   ├── config.py                        # Global constants: material, domain, NAS space
│   ├── pinn_network.py                  # PINNNet: MLP with configurable activations
│   ├── physics_model.py                 # Heat equation residuals, HTC boiling curve
│   ├── trainers.py                      # Adam / L-BFGS / PSO training phases
│   ├── baseline_data.py                 # Paper data: Tables 1-2, Figures 7, 15-22
│   ├── arch_search.py                   # NAS: decode, evaluate, pymoo interface
│   ├── opt_nsga2.py                     # NSGA-II (pop=24, gen=16)
│   ├── opt_nsga3.py                     # NSGA-III (ref_dirs=10)
│   ├── opt_bayesian.py                  # Bayesian Optimization (TPE)
│   └── experiment_runner.py             # Orchestration: single and multi-optimizer runs
│
├── problems/                            # Benchmark problem definitions
│   ├── base.py                          # Abstract PINNProblem base class
│   ├── quenching.py                     # A356 aluminum quenching (main problem)
│   ├── burgers.py                       # 1D Burgers equation
│   ├── poisson.py                       # 2D Poisson equation
│   └── allen_cahn.py                    # 1D+t Allen-Cahn equation
│
├── level1_single_shot/                  # Level 1: Global single-shot PINN
│   ├── plot_results.py                  # comparison.json → accuracy / runtime / table
│   └── plot_fem_vs_pinn_steps.py        # FEM step-by-step vs PINN single pass
│
├── level2_timestepper/                  # Level 2: Window-based skip operator
│   ├── main_level2.py                   # CLI: generate skip_table.json (Adam 500 ep)
│   ├── eval_skip_lbfgs.py               # Skip eval with Adam+L-BFGS per window
│   ├── eval_skip_5k.py                  # Skip eval with 5000 Adam epochs (NSGA-II/III)
│   ├── plot_results.py                  # Skip vs L2 / MAE / runtime curves
│   ├── plot_fem_vs_pinn_skip.py         # Core thesis visualization: FEM vs skip PINN
│   └── src/
│       ├── ts_model.py                  # TimeStepperPINN: 4-input MLP
│       ├── ts_trainer.py                # Window training + optional L-BFGS phase
│       ├── ts_nas.py                    # Grid-search NAS for optimal skip factor
│       └── ts_evaluate.py              # Evaluate skip ∈ {1, 2, 4, 6, …}
│
├── level3_hybrid_fem/                   # Level 3: Adaptive hybrid FEM+PINN loop
│   ├── main_level3.py                   # CLI: run hybrid for 3 architectures
│   ├── plot_results.py                  # Residual trace, step distribution, CMM bars
│   └── src/
│       ├── fem_interface.py             # FEMCheckpoint: T-field snapshots
│       ├── hybrid_runner.py             # FEM ↔ PINN alternation loop
│       ├── adaptive_skip.py             # Residual-based FEM/PINN selector
│       └── mechanical.py               # Thermal strain → CMM distortion estimate
│
├── level4_distortion/                   # Level 4: 2D plane-stress distortion
│   ├── main_distortion.py               # CLI: T_field → FEM solve → CMM |δ|
│   ├── plot_results.py                  # CMM bar chart, displacement map
│   ├── plot_paper_comparison.py         # Signed bar chart vs paper Fig17/18
│   └── src/
│       ├── plane_stress_fem.py          # PlaneStressFEM: Q4 elements, CG solver
│       ├── thermal_field.py             # Analytical T(x,y) profiles
│       └── cmm_points.py               # CMM labels + paper reference values
│
├── level5_refinement/                   # Level 5: L-BFGS refinement
│   ├── main_level5.py                   # CLI: Adam → L-BFGS refinement pipeline
│   ├── plot_results.py                  # L2 before/after, summary table
│   ├── eval_skip_l5.py                  # Skip capability of Level 5 global PINNs
│   └── plot_skip_l5.py                  # 4-series skip comparison plot
│
├── level6_poisson_benchmark/            # Level 6: Poisson PDE generalization
│   ├── main_level6.py                   # CLI: train and evaluate on Poisson
│   ├── plot_results.py                  # Heatmaps, L2 progression, summary table
│   └── src/
│       ├── level6_finetune.py           # Fine-tune quenching NAS weights on Poisson
│       └── poisson_aux_loss.py          # Auxiliary boundary + PDE losses
│
├── level7_temporal/                     # Level 7A: Temporal skip analysis
│   ├── main_level7a.py                  # CLI: time-skip table for quenching
│   └── plot_level7a.py                  # Cooling curves + L2 vs time + skip table
│
├── level7_multiDomain/                  # Level 7B: Multi-domain Poisson NAS
│   ├── main_level7b.py                  # CLI: NAS per geometry (5 domains)
│   ├── plot_level7b.py                  # Per-geometry heatmaps + summary table
│   └── src/
│       ├── domains.py                   # Domain definitions: square/circle/annulus/lshape/flower
│       ├── poisson_pinn.py              # Poisson PINN trainer per domain
│       └── nas_search_7b.py             # NAS search for each geometry
│
└── results/                             # All generated output files
    ├── NAS_PINNs_Presentation.pptx      # 12-slide PowerPoint presentation
    ├── baseline/                        # Paper reference images
    ├── run2/                            # Level 1 results
    │   ├── comparison.json
    │   ├── plots/                       # fem_vs_pinn_steps.png, accuracy.png, …
    │   └── quenching/{bayesian,nsga2,nsga3}/
    │       ├── best_arch.json
    │       ├── model.pt                 # Level 1 trained weights
    │       └── results.json
    ├── level2/                          # Level 2 skip visualization
    │   └── fem_vs_pinn_skip.png
    ├── level5_refinement/               # Level 5 + skip comparison
    │   ├── {bayesian,nsga2,nsga3}/model_lbfgs.pt
    │   ├── skip_table_l5.json           # Global PINN skip evaluation
    │   └── level5_skip_comparison.png   # 4-series comparison plot
    ├── level6/plots/                    # Poisson heatmaps + tables
    ├── level7/plots/                    # 7A cooling curves + skip table
    ├── level7b/plots/                   # 7B per-geometry heatmaps + table
    └── cross_level_comparison/          # Unified cross-level summary
        ├── cross_level_summary.json
        └── cross_level_summary_table.png
```

---

## Quick Start

### Prerequisites

```bash
pip install torch numpy scipy matplotlib pymoo bayesian-optimization python-pptx
```

### Level 1 — Run NAS and train all 3 optimizers

```bash
cd NAS-PINNS3
python main.py --compare_all --problem quenching
python level1_single_shot/plot_results.py
python level1_single_shot/plot_fem_vs_pinn_steps.py
```

### Level 2 — Temporal skip operator

```bash
# Standard 500-epoch evaluation
python level2_timestepper/main_level2.py --skip_values 1 2 4 6
python level2_timestepper/plot_results.py
python level2_timestepper/plot_fem_vs_pinn_skip.py

# High-epoch test for NSGA-II/III (5000 epochs, skip=1,2 only)
python level2_timestepper/eval_skip_5k.py

# Adam + L-BFGS per window
python level2_timestepper/eval_skip_lbfgs.py
```

### Level 3 — Hybrid FEM+PINN

```bash
python level3_hybrid_fem/main_level3.py --threshold 0.1 --max_skip 4
python level3_hybrid_fem/plot_results.py --dir results/thr0.1_skip4
```

### Level 4 — Distortion mechanics

```bash
python level4_distortion/main_distortion.py
python level4_distortion/plot_results.py
python level4_distortion/plot_paper_comparison.py
```

### Level 5 — L-BFGS refinement

```bash
python level5_refinement/main_level5.py
python level5_refinement/plot_results.py
python level5_refinement/eval_skip_l5.py     # skip capability of refined models
python level5_refinement/plot_skip_l5.py     # 4-series comparison plot
```

### Level 6 — Poisson benchmark

```bash
python level6_poisson_benchmark/main_level6.py
python level6_poisson_benchmark/plot_results.py
```

### Level 7 — Temporal + Multi-domain

```bash
python level7_temporal/main_level7a.py
python level7_temporal/plot_level7a.py

python level7_multiDomain/main_level7b.py
python level7_multiDomain/plot_level7b.py
```

### Cross-level comparison + PowerPoint

```bash
python compare_all_levels.py --skip_run     # plot from existing results
python make_presentation.py                 # generate 12-slide PPTX
```

### After 5000-epoch run completes

```bash
python finalize_after_5k.py    # updates plot + regenerates PPTX
```

---

## NAS Search Space

| Parameter | Range |
|-----------|-------|
| Hidden layers | 3 – 6 |
| Neurons per layer | 64 – 256 |
| Activation function | tanh · relu · sin · gelu · swish |
| Optimizers | Bayesian (TPE) · NSGA-II · NSGA-III |

Training pipeline: **Adam** (cosine LR: 1e-3 → 1e-5) → **L-BFGS** (Level 5 refinement)

---

## Material Properties — A356 Aluminum

*Source: baseline_paper [2026], Table 1*

| Property | Value |
|----------|-------|
| Thermal conductivity K | 151 W/m·K |
| Volumetric heat capacity ρ·Cₚ | 2.43×10⁶ J/m³·K |
| Elastic modulus E | 70 GPa |
| Poisson's ratio ν | 0.33 |
| Thermal expansion β | 2.34×10⁻⁵ /K |
| Stress-free temperature T_ref | 540 °C |

---

## Skip Operator: Key Finding

```
FEM alone:  t=0 → t=1.5 → t=3 → t=4.5 → … → t=30   (21 points, 20 solves)

skip=2:     FEM at [0, 3, 6, 9, …, 30]               (11 points, 10 FEM solves)
            PINN at [1.5, 4.5, 7.5, …, 28.5]          (10 predicted)
            → 2× speedup, MAE = 33.2°C (better than skip=1: 43.6°C)

skip=4:     FEM at [0, 6, 12, 18, 24, 30]             (6 points, 5 FEM solves)
            PINN fills 15 intermediate steps
            → 4× speedup, MAE = 57.5°C (still converged for Bayesian)
```

**Convergence criterion:** MAE ratio vs skip=1 baseline < 1.5×

---

## Reference

> baseline_paper [2026] — *Water quenching simulation of A356 aluminum subframe:
> FEM modeling, distortion prediction, and CMM validation.*
