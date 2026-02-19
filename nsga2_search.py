import numpy as np
import time
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from tqdm import tqdm

from naspinn import (
    ResidualBurgerPINN,
    generate_data,
    train_pinn,
    compute_mean_l2_error
)

# ════════════════════════════════════════════════════════════════════════════════
# NSGA-II Problem Definition
# ════════════════════════════════════════════════════════════════════════════════

class BurgersNASProblem(ElementwiseProblem):
    """
    Multi-objective optimization problem for NAS-PINN
    Objectives: minimize (mean L2 error, parameter count)
    """
    def __init__(self, nu):
        self.max_hidden_layers = 8
        # Decision variables:
        # [0]: n_layers (4-8)
        # [1-8]: neurons per layer (48-256)
        # [9]: learning rate (1e-4 to 8e-3)
        # [10]: use_residual (0 or 1)
        # [11]: lambda_pde_factor (60-140, scales default 100.0)
        
        n_var = 1 + self.max_hidden_layers + 1 + 1 + 1
        xl = np.array([4] + [48]*8 + [1e-4] + [0] + [60.0])
        xu = np.array([self.max_hidden_layers] + [256]*8 + [8e-3] + [1] + [140.0])
        
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)
        self.nu = nu
        self.eval_count = 0

    def _evaluate(self, x, out, *args, **kwargs):
        self.eval_count += 1
        
        # Parse decision variables
        n_layers = int(np.clip(round(x[0]), 4, self.max_hidden_layers))
        neurons = [int(round(x[i])) for i in range(1, 1 + n_layers)]
        lr = float(x[-3])
        use_residual = bool(round(x[-2]))
        lambda_pde_factor = float(x[-1])
        
        # Build architecture
        architecture = [2] + neurons + [1]
        
        try:
            # Create model
            model = ResidualBurgerPINN(architecture, use_residual=use_residual).double()
            data = generate_data(n_collocation=15000, n_boundary=200, n_initial=200)
            
            # Train: Adam-only (NAS stage, fast evaluation)
            model, _ = train_pinn(
                model,
                self.nu,
                data,
                epochs=1000,
                lr=lr,
                lambda_pde=100.0 * lambda_pde_factor,
                lambda_bc=20.0,
                lambda_ic=5.0,
                patience=120,
                verbose=True,
                use_lbfgs=False
            )
            
            # Evaluate: time-averaged L2 error
            mean_l2, _ = compute_mean_l2_error(model, self.nu, nt=80, nx=300)
            params = sum(p.numel() for p in model.parameters())
            
            # Bounds check
            if np.isnan(mean_l2) or mean_l2 > 1.0:
                mean_l2 = 1.0
            
        except Exception as e:
            print(f"  ⚠️  Error evaluating architecture {architecture}: {str(e)[:80]}")
            mean_l2 = 1.0
            params = 20000
        
        out["F"] = [mean_l2, float(params)]


# ════════════════════════════════════════════════════════════════════════════════
# Callback for Progress Tracking
# ════════════════════════════════════════════════════════════════════════════════

class NSGAIICallback:
    def __init__(self, n_gen):
        self.n_gen = n_gen
        self.pbar = tqdm(total=n_gen, desc="📊 Generation Progress", unit="gen")

    def __call__(self, algorithm):
        self.pbar.update(1)
        
        if hasattr(algorithm, 'pop') and algorithm.pop is not None:
            F = algorithm.pop.get("F")
            best_idx = np.argmin(F[:, 0])
            print(f"\n  Gen {algorithm.n_gen:2d}/{self.n_gen} → "
                  f"Best mean L2: {F[best_idx,0]:.2e}   Params: {int(F[best_idx,1]):,}")

    def close(self):
        self.pbar.close()


# ════════════════════════════════════════════════════════════════════════════════
# Main NSGA-II Run Function
# ════════════════════════════════════════════════════════════════════════════════

def run_nsga2(nu, pop=10, ngen=5):
    """
    Run NSGA-II for NAS-PINN architecture search
    
    Args:
        nu: viscosity coefficient
        pop: population size
        ngen: number of generations
    
    Returns:
        dict: best solution found
        pymoo result object
    """
    print("\n" + "="*90)
    print("🔍 METHOD: NSGA-II (Multi-Objective Genetic Algorithm)")
    print("="*90)
    print(f"  Viscosity: ν = {nu:.3f}")
    print(f"  Population Size: {pop}")
    print(f"  Generations: {ngen}")
    print(f"  Training: {1000} epochs Adam (no L-BFGS, fast evaluation)")
    print(f"  Objective: Minimize (mean L2 error, parameter count)")
    print(f"  Collocation Points: 15,000")
    print(f"  Domain: x ∈ [-1, 1], t ∈ [0, 1]")
    print("="*90 + "\n")
    
    # Algorithm setup
    algorithm = NSGA2(
        pop_size=pop,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(eta=25),
        eliminate_duplicates=True
    )
    
    problem = BurgersNASProblem(nu)
    termination = get_termination("n_gen", ngen)
    
    start_time = time.time()
    callback = NSGAIICallback(ngen)
    
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=False,
        callback=callback
    )
    
    callback.close()
    search_time = time.time() - start_time
    
    # Extract best solution
    best_idx = np.argmin(res.F[:, 0])
    max_hidden_layers = 8
    n_layers = int(np.clip(round(res.X[best_idx][0]), 4, max_hidden_layers))
    neurons = [int(round(res.X[best_idx][i])) for i in range(1, 1 + n_layers)]
    lr = float(res.X[best_idx][-3])
    use_residual = bool(round(res.X[best_idx][-2]))
    lambda_pde_factor = float(res.X[best_idx][-1])
    
    architecture = [2] + neurons + [1]
    
    # Print results
    print("\n" + "="*90)
    print("🏆 BEST ARCHITECTURE FOUND")
    print("="*90)
    print(f"  Architecture         : {architecture}")
    print(f"  Residual Connections : {use_residual}")
    print(f"  Learning Rate        : {lr:.6f}")
    print(f"  λ_PDE Factor         : {lambda_pde_factor:.2f} → effective λ_PDE = {100*lambda_pde_factor:.1f}")
    print(f"  Mean L2 Error        : {res.F[best_idx,0]:.2e}")
    print(f"  Total Parameters     : {int(res.F[best_idx,1]):,}")
    print(f"  Search Time          : {search_time:.1f} seconds")
    print("="*90 + "\n")
    
    return {
        'architecture': architecture,
        'layers': n_layers,
        'neurons': neurons,
        'use_residual': use_residual,
        'lr': lr,
        'lambda_pde_factor': lambda_pde_factor,
        'mean_l2_error': res.F[best_idx, 0],
        'params': int(res.F[best_idx, 1]),
        'search_time': search_time,
        'pareto_front': res.F,
        'pareto_solutions': res.X
    }, res


if __name__ == "__main__":
    viscosities = [0.01]
    
    for nu in viscosities:
        print(f"\n{'#'*90}")
        print(f"  NSGA-II NAS-PINN Search – ν = {nu}")
        print(f"{'#'*90}\n")
        
        best_result, pymoo_result = run_nsga2(nu, pop=10, ngen=5)
        
        print(f"✅ Search completed. Best architecture: {best_result['architecture']}")