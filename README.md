# NAS-PINN: Neural Architecture Search for Physics-Informed Neural Networks

## 🎯 Project Overview

This repository implements **Neural Architecture Search (NAS)** for Physics-Informed Neural Networks (PINNs) applied to the **Burgers Equation** - a fundamental nonlinear PDE in fluid dynamics.

### Key Features
- ✅ **3 NAS Methods**: NSGA-II, NSGA-III, Bayesian Optimization
- ✅ **Automatic Architecture Discovery**: Optimizes network depth, width, and learning rate
- ✅ **Multi-Regime Benchmark**: Tests across 3 viscosity values (ν = 0.01, 0.04, 0.07)
- ✅ **Comprehensive Evaluation**: L2 error, MSE, RMSE, MAE, parameter count
- ✅ **Rich Visualizations**: Heatmaps, snapshots, error distributions, loss curves

---

## 📐 Problem Formulation

### Burgers Equation
```
∂u/∂t + u·∂u/∂x - (ν/π)·∂²u/∂x² = 0
```

**Domain:**
- Spatial: x ∈ [-1, 1]
- Temporal: t ∈ [0, 1]

**Initial Condition:**
```
u(0, x) = -sin(πx)
```

**Boundary Conditions:**
```
u(t, -1) = 0
u(t, +1) = 0
```

**Viscosity Values:**
- ν = 0.01 → Highly advective (shock-like structures, most challenging)
- ν = 0.04 → Moderate regime (balanced advection-diffusion)
- ν = 0.07 → Diffusion-dominated (smooth solution, easier)

---

## 🏗️ Architecture

### File Structure
```
NAS-PINNS1/
│
├── naspinn.py              # Core PINN implementation
├── nsga2_search.py         # NSGA-II optimization
├── nsga3_search.py         # NSGA-III optimization
├── bayes_opt_search.py     # Bayesian optimization
├── lbfgs_refine.py         # Stage 2 refinement (Adam + L-BFGS)
├── pso_refine.py           # Stage 3 refinement (Adam + PSO)
├── final_comparison.py     # Final stage comparison and plots
├── run_all.py              # Main execution script
├── README.md               # This file
│
└── results/                # Generated results
     ├── adam/               # Stage 1 outputs
     ├── adam_lbfgs/         # Stage 2 outputs
     ├── adam_pso/           # Stage 3 outputs
     ├── visualizations/     # Final comparison plots
     └── MASTER_COMPARISON.csv
```

## 🔄 Workflow (Current Logic)

The project runs sequentially in this order:

1. **Stage 1 – Adam NAS Search**
    - Finds best architectures with NSGA-II, NSGA-III, Bayesian Optimization.
2. **Stage 2 – L-BFGS Refinement**
    - Re-trains Stage 1 best architectures using Adam + L-BFGS.
3. **Stage 3 – PSO Refinement**
    - Optimizes training hyperparameters (learning rate, λ_PDE factor) with PSO.
4. **Final Comparison**
    - Compares all stages and methods, creates summary tables and plots.

You can run the full pipeline from `run_all.py` (it triggers Stage 1 → Stage 2 → Stage 3 → Final Comparison in sequence).

### Core Components

#### 1. `naspinn.py`
- `ResidualBurgerPINN`: PINN class with optional residual connections and tanh activation
- `generate_data()`: Creates training data (collocation, boundary, initial points)
- `pde_loss()`: Computes PDE residual using automatic differentiation
- `train_pinn()`: Trains model using Adam optimizer (default 1200 epochs, lr=3e-4)
- Training progress is printed every 100 epochs in terminal
- `compute_mean_l2_error()`: Evaluates time-averaged relative L2 error
- Visualization functions for heatmaps, snapshots, comparisons

#### 2. `nsga2_search.py`
- Multi-objective optimization (minimize L2 error + parameters)
- Population: 10, Generations: 5 (fast mode)
- Search space: 4-8 hidden layers, 48-256 neurons/layer
- Also optimizes learning rate, residual toggle, and PDE loss weight factor

#### 3. `nsga3_search.py`
- Many-objective optimization (L2 error + parameters + training time)
- Population: 10, Generations: 5 (fast mode)
- Reference directions for Pareto front
- Additional learning rate optimization

#### 4. `bayes_opt_search.py`
- Gaussian Process-based optimization
- Default: 10 iterations, 2 initial random points
- In `run_all.py` fast preset: 8 iterations, 2 initial random points
- Optimizes: layers, neurons, learning rate

#### 5. `lbfgs_refine.py`
- Loads best Stage 1 architectures from `results/adam/...`
- Refines each model with Adam + L-BFGS
- Saves outputs to `results/adam_lbfgs/...`

#### 6. `pso_refine.py`
- Loads best Stage 1 architectures from `results/adam/...`
- Uses PSO to optimize `(learning_rate, lambda_pde_factor)`
- Retrains and saves outputs to `results/adam_pso/...`

#### 7. `final_comparison.py`
- Loads all stage summaries (`adam`, `adam_lbfgs`, `adam_pso`)
- Produces consolidated metrics and visual comparisons
- Writes `results/MASTER_COMPARISON.csv` and plots under `results/visualizations/`

---

