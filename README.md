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

## 🚀 Installation

### Requirements
```bash
Python >= 3.8
PyTorch >= 1.10
NumPy >= 1.20
Matplotlib >= 3.3
Pandas >= 1.3
pymoo >= 0.5.0
bayesian-optimization >= 1.2.0
```

### Setup
```bash
# Clone repository
git clone <repository-url>
cd NAS-PINNS1

# Install dependencies
pip install torch numpy matplotlib pandas pymoo bayesian-optimization

# Verify installation
python -c "import torch; import pymoo; print('Setup complete!')"
```

---

## 💻 Usage

### Quick Start
```bash
python run_all.py
```

This runs all 3 NAS methods across all 3 viscosity values and generates:
- ✅ Trained models
- ✅ Visualizations (heatmaps, snapshots, comparisons)
- ✅ CSV results
- ✅ Loss curves

**Expected Runtime:** 6-8 hours on standard CPU

---

### Advanced Usage

#### Run Single Method
```python
from nsga2_search import run_nsga2

# Search for viscosity ν=0.01
best = run_nsga2(nu=0.01, pop=20, ngen=10)
print(f"Best architecture: {best['architecture']}")
print(f"L2 Error: {best['l2_error']:.6e}")
```

#### Train Custom Architecture
```python
from naspinn import BurgerPINN, generate_data, train_pinn

# Define custom architecture
architecture = [2, 64, 64, 64, 1]  # Input(2) → 3 hidden(64) → Output(1)
model = BurgerPINN(architecture).double()

# Generate data
data = generate_data(n_collocation=10000, n_boundary=100, n_initial=100)

# Train
nu = 0.04
train_pinn(model, nu, data, epochs=2000, lr=5e-4)
```

#### Evaluate Model
```python
from naspinn import evaluate_model

metrics = evaluate_model(model, nu=0.04)
print(f"L2 Error: {metrics['l2_error']:.6e}")
print(f"RMSE: {metrics['rmse']:.6e}")
print(f"MAE: {metrics['mae']:.6e}")
```

---

## 📊 Results

### Output Structure
```
results/
├── nu_0.01/
│   ├── nsga2/
│   │   ├── heatmap.png
│   │   ├── snapshots.png
│   │   ├── comparison.png
│   │   ├── error_dist.png
│   │   ├── loss_curve.png
│   │   └── model.pt
│   ├── nsga3/
│   ├── bayesian/
│   └── results.csv
├── nu_0.04/
├── nu_0.07/
└── comparison.csv          # Overall comparison
```

### Sample Results (Expected)

| ν    | Method    | Architecture         | L2 Error | Parameters | Search Time |
|------|-----------|---------------------|----------|------------|-------------|
| 0.01 | NSGA-II   | [2,80,75,56,1]      | ~10⁻²    | ~15K       | ~2000s      |
| 0.01 | NSGA-III  | [2,78,64,52,44,1]   | ~10⁻²    | ~18K       | ~3500s      |
| 0.01 | Bayesian  | [2,72,68,1]         | ~10⁻²    | ~12K       | ~2500s      |
| 0.04 | NSGA-II   | [2,64,56,1]         | ~10⁻³    | ~8K        | ~1800s      |
| 0.07 | NSGA-II   | [2,48,32,1]         | ~10⁻⁴    | ~4K        | ~1500s      |

---

## 📈 Visualization Examples

### 1. Solution Heatmap
Contour plot showing u(x,t) evolution over space-time domain.

### 2. Solution Snapshots
Velocity profiles at t = [0.0, 0.25, 0.5, 0.75, 1.0] compared with analytical solution.

### 3. Final Time Comparison
PINN prediction vs analytical solution at t=1.

### 4. Pointwise Error Distribution
Spatial distribution of absolute error |u_pred - u_exact|.

### 5. Training Loss Curve
Loss evolution over 2000 epochs (log scale).

---

## 🔬 Methodology

### Training Configuration
```python
epochs = 2000
learning_rate = 5e-4
optimizer = Adam
activation = tanh
collocation_points = 10000
boundary_points = 100 (per boundary)
initial_points = 100
```

### NAS Configuration

**NSGA-II:**
- Objectives: L2 error, parameter count
- Population: 20
- Generations: 10
- Crossover: 0.8
- Mutation: 0.01

**NSGA-III:**
- Objectives: L2 error, parameters, training time
- Population: 50
- Generations: 10
- Reference directions: Das-Dennis (3-obj, 6 partitions)

**Bayesian Optimization:**
- Surrogate: Gaussian Process
- Acquisition: Upper Confidence Bound
- Iterations: 20
- Initial samples: 5

### Search Space
```python
n_layers: [2, 8]
neurons_per_layer: [20, 80]
learning_rate: [1e-4, 1e-2]  # NSGA-III & Bayesian only
```

---

## 📚 References

### Core Papers
1. **Physics-Informed Neural Networks:**
   - Raissi et al. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*.

2. **Neural Architecture Search:**
   - Elsken et al. (2019). "Neural Architecture Search: A Survey." *JMLR*.

3. **Multi-Objective Optimization:**
   - Deb & Jain (2014). "An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach." *IEEE TEVC*.

### Burgers Equation
- Burgers, J. M. (1948). "A mathematical model illustrating the theory of turbulence." *Advances in Applied Mechanics*.
- Cole, J. D. (1951). "On a quasi-linear parabolic equation occurring in aerodynamics." *Quarterly of Applied Mathematics*.

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Use CPU instead
model = BurgerPINN(architecture)  # Don't use .cuda()
```

#### 2. NaN in Training
- Reduce learning rate: `lr=1e-4`
- Check viscosity value (very low ν can be unstable)
- Increase collocation points

#### 3. High L2 Error
- Increase epochs: `epochs=5000`
- Try deeper/wider network
- Adjust data sampling distribution

#### 4. Slow Training
- Reduce collocation points: `n_collocation=5000`
- Reduce generations: `ngen=5`
- Use GPU if available

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a pull request

**Areas for improvement:**
- [ ] Add more PDE benchmarks (Heat, Wave, Navier-Stokes)
- [ ] GPU acceleration
- [ ] Parallel NAS execution
- [ ] Advanced visualization (interactive plots)
- [ ] Hyperparameter sensitivity analysis

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{nas_pinn_burgers,
  title={NAS-PINN: Neural Architecture Search for Burgers Equation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/NAS-PINNS1}
}
```

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- PyMOO library for multi-objective optimization
- PyTorch for automatic differentiation
- Bayesian Optimization library
- Original NAS-PINN authors

---

## 📞 Support

For questions or issues:
- 📧 Open an issue on GitHub
- 💬 Contact: your.email@example.com
- 📖 Check documentation in code comments

---

## 🗺️ Roadmap

### Version 1.0 (Current)
- ✅ Burgers equation implementation
- ✅ NSGA-II, NSGA-III, Bayesian optimization
- ✅ Comprehensive visualization

### Version 1.1 (Planned)
- [ ] GPU acceleration
- [ ] Additional PDE benchmarks
- [ ] Real-time visualization
- [ ] Hyperparameter tuning guide

### Version 2.0 (Future)
- [ ] Multi-GPU distributed training
- [ ] Advanced NAS algorithms (DARTS, ENAS)
- [ ] Interactive web dashboard
- [ ] Pre-trained model zoo

---

**Last Updated:** February 2024  
**Version:** 1.0.0  
**Status:** Active Development 🚀