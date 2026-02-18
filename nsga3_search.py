import numpy as np
import time
import torch
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
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
# NSGA-III Problem Definition
# ════════════════════════════════════════════════════════════════════════════════

class BurgersNASProblem3(ElementwiseProblem):
    """
    Multi-objective optimization problem for NAS-PINN
    Objectives: minimize (mean L2 error, parameter count, training time)
    """
    def __init__(self, nu):
        # Decision variables:
        # [0]: n_layers (4-10)
        # [1-8]: neurons per layer (48-256)
        # [9]: learning rate (1e-4 to 8e-3)
        # [10]: use_residual (0 or 1)
        # [11]: lambda_pde_factor (60-140)
        
        n_var = 1 + 8 + 1 + 1 + 1
        xl = np.array([4] + [48]*8 + [1e-4] + [0] + [60.0])
        xu = np.array([10] + [256]*8 + [8e-3] + [1] + [140.0])
        
        super().__init__(n_var=n_var, n_obj=3, xl=xl, xu=xu)
        self.nu = nu
        self.eval_count = 0

    def _evaluate(self, x, out, *args, **kwargs):
        self.eval_count += 1
        
        # Parse decision variables
        n_layers = int(round(x[0]))
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
            t0 = time.time()
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
            train_time = time.time() - t0
            
            # Evaluate: time-averaged L2 error
            mean_l2, _ = compute_mean_l2_error(model, self.nu, nt=80, nx=300)
            params = sum(p.numel() for p in model.parameters())
            
            # Bounds check
            if np.isnan(mean_l2) or mean_l2 > 1.0:
                mean_l2 = 1.0
            if np.isnan(train_time):
                train_time = 1000.0
            
        except Exception as e:
            print(f"  ⚠️  Error evaluating architecture {architecture}: {str(e)[:80]}")
            mean_l2 = 1.0
            params = 20000
            train_time = 1000.0
        
        out["F"] = [mean_l2, float(params), float(train_time)]


# ════════════════════════════════════════════════════════════════════════════════
# Callback for Progress Tracking
# ════════════════════════════════════════════════════════════════════════════════

class NSGA3Callback:
    """
    Callback for monitoring NSGA-III progress
    """
    def __init__(self, n_gen):
        self.n_gen = n_gen
        self.gen_count = 0
        self.pbar = tqdm(total=n_gen, desc="📊 Generation Progress", unit="gen")
        self.gen_history = []

    def __call__(self, algorithm):
        self.gen_count = algorithm.n_gen
        self.pbar.update(1)
        
        if hasattr(algorithm, 'pop') and algorithm.pop is not None:
            F = algorithm.pop.get("F")
            best_error_idx = np.argmin(F[:, 0])
            best_params_idx = np.argmin(F[:, 1])
            best_time_idx = np.argmin(F[:, 2])
            
            best_error = F[best_error_idx, 0]
            best_params = int(F[best_params_idx, 1])
            best_time = F[best_time_idx, 2]
            
            avg_error = np.mean(F[:, 0])
            avg_params = np.mean(F[:, 1])
            avg_time = np.mean(F[:, 2])
            
            self.gen_history.append({
                'generation': algorithm.n_gen,
                'best_error': best_error,
                'best_params': best_params,
                'best_time': best_time,
                'avg_error': avg_error,
                'avg_params': avg_params,
                'avg_time': avg_time,
                'front_size': len(F)
            })
            
            print(f"\n  ⭐ Generation {algorithm.n_gen}/{self.n_gen}:")
            print(f"     🎯 Best mean L2 Error: {best_error:.2e} (avg: {avg_error:.2e})")
            print(f"     🔧 Best Params: {best_params:,} (avg: {avg_params:.0f})")
            print(f"     ⏱️  Best Time: {best_time:.2f}s (avg: {avg_time:.2f}s)")
            print(f"     📊 Front Size: {len(F)}")

    def close(self):
        self.pbar.close()


# ════════════════════════════════════════════════════════════════════════════════
# Main NSGA-III Run Function
# ════════════════════════════════════════════════════════════════════════════════

def run_nsga3(nu, pop=24, ngen=12):
    """
    Run NSGA-III for NAS-PINN architecture search
    
    Args:
        nu: viscosity coefficient
        pop: population size
        ngen: number of generations
    
    Returns:
        dict: best solution found
        pymoo result object
    """
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=5)
    
    print("\n" + "="*90)
    print("🔍 METHOD: NSGA-III (Many-Objective Genetic Algorithm)")
    print("="*90)
    print(f"  Viscosity: ν = {nu:.3f}")
    print(f"  Population Size: {pop}")
    print(f"  Reference Directions: {len(ref_dirs)}")
    print(f"  Generations: {ngen}")
    print(f"  Training: {1000} epochs Adam (no L-BFGS, fast evaluation)")
    print(f"  Objectives: [mean L2 Error, Parameters, Training Time]")
    print(f"  Collocation Points: 15,000")
    print(f"  Domain: x ∈ [-1, 1], t ∈ [0, 1]")
    print("="*90 + "\n")
    
    # Ensure pop_size >= ref_dirs
    if pop < len(ref_dirs):
        print(f"  ⚠️  pop_size={pop} < ref_dirs={len(ref_dirs)}")
        pop = len(ref_dirs)
        print(f"  → Adjusted pop_size to {pop}\n")
    
    # Algorithm setup
    algorithm = NSGA3(
        pop_size=pop,
        ref_dirs=ref_dirs,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(eta=25),
        eliminate_duplicates=True
    )
    
    problem = BurgersNASProblem3(nu)
    termination = get_termination("n_gen", ngen)
    
    start_time = time.time()
    callback = NSGA3Callback(ngen)
    
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
    
    # Extract best solution (by mean L2 error)
    best_idx = np.argmin(res.F[:, 0])
    n_layers = int(round(res.X[best_idx][0]))
    neurons = [int(round(res.X[best_idx][i])) for i in range(1, 1 + n_layers)]
    lr = float(res.X[best_idx][-3])
    use_residual = bool(round(res.X[best_idx][-2]))
    lambda_pde_factor = float(res.X[best_idx][-1])
    
    architecture = [2] + neurons + [1]
    
    # Print results
    print("\n" + "="*90)
    print("🏆 BEST ARCHITECTURE FOUND (by mean L2 error)")
    print("="*90)
    print(f"  Architecture         : {architecture}")
    print(f"  Residual Connections : {use_residual}")
    print(f"  Learning Rate        : {lr:.6f}")
    print(f"  λ_PDE Factor         : {lambda_pde_factor:.2f} → effective λ_PDE = {100*lambda_pde_factor:.1f}")
    print(f"  Mean L2 Error        : {res.F[best_idx,0]:.2e}")
    print(f"  Total Parameters     : {int(res.F[best_idx,1]):,}")
    print(f"  Training Time        : {res.F[best_idx,2]:.2f}s")
    print(f"  Search Time          : {search_time:.1f}s")
    print(f"  Pareto Front Size    : {len(res.F)}")
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
        'train_time': res.F[best_idx, 2],
        'search_time': search_time,
        'pareto_front': res.F,
        'pareto_solutions': res.X
    }, res


if __name__ == "__main__":
    viscosities = [0.01]
    
    for nu in viscosities:
        print(f"\n{'#'*90}")
        print(f"  NSGA-III NAS-PINN Search – ν = {nu}")
        print(f"{'#'*90}\n")
        
        best_result, pymoo_result = run_nsga3(nu, pop=24, ngen=12)
        
        print(f"✅ Search completed. Best architecture: {best_result['architecture']}")