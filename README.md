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
├── run_all.py              # Main execution script
├── README.md               # This file
│
└── results/                # Generated results
    ├── nu_0.01/
    │   ├── nsga2/
    │   ├── nsga3/
    │   └── bayesian/
    ├── nu_0.04/
    ├── nu_0.07/
    └── comparison.csv
```

### Core Components

#### 1. `naspinn.py`
- `BurgerPINN`: Neural network class with tanh activation
- `generate_data()`: Creates training data (collocation, boundary, initial points)
- `pde_loss()`: Computes PDE residual using automatic differentiation
- `train_pinn()`: Trains model using Adam optimizer (2000 epochs, lr=5e-4)
- `compute_l2_error()`: Evaluates relative L2 error
- Visualization functions for heatmaps, snapshots, comparisons

#### 2. `nsga2_search.py`
- Multi-objective optimization (minimize L2 error + parameters)
- Population: 20, Generations: 10
- Crossover rate: 0.8, Mutation rate: 0.01
- Search space: 2-8 layers, 20-80 neurons/layer

#### 3. `nsga3_search.py`
- Many-objective optimization (L2 error + parameters + training time)
- Population: 50, Generations: 10
- Reference directions for Pareto front
- Additional learning rate optimization

#### 4. `bayes_opt_search.py`
- Gaussian Process-based optimization
- 20 iterations, 5 initial random points
- Optimizes: layers, neurons, learning rate

---

