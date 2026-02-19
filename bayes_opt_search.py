import numpy as np
import time
import torch
from bayes_opt import BayesianOptimization
from tqdm import tqdm

from naspinn import (
    ResidualBurgerPINN,
    generate_data,
    train_pinn,
    compute_mean_l2_error
)

# ════════════════════════════════════════════════════════════════════════════════
# Objective Function for Bayesian Optimization
# ════════════════════════════════════════════════════════════════════════════════

class BayesianNASOptimizer:
    """
    Bayesian Optimization for NAS-PINN hyperparameter search
    Maximizes: -mean_l2_error (minimize error as maximization problem)
    """
    def __init__(self, nu, verbose=True):
        self.nu = nu
        self.verbose = verbose
        self.eval_count = 0
        self.eval_history = []
        
    def objective_function(self, n_layers, n1, n2, n3, n4, n5, n6, n7, n8, lr, use_residual, lambda_pde_factor):
        """
        Objective function for Bayesian optimization
        Inputs are continuous; we'll round/convert as needed
        
        Args:
            n_layers: number of hidden layers (4-10)
            n1-n8: neurons per layer (48-256)
            lr: learning rate (1e-4 to 8e-3)
            use_residual: 0 or 1 (boolean)
            lambda_pde_factor: 60-140 (scales default 100.0)
        
        Returns:
            -mean_l2_error (negative because we maximize)
        """
        self.eval_count += 1
        
        # Convert to integers/booleans
        n_layers = int(round(n_layers))
        neurons = [
            int(round(n1)),
            int(round(n2)),
            int(round(n3)),
            int(round(n4)),
            int(round(n5)),
            int(round(n6)),
            int(round(n7)),
            int(round(n8))
        ]
        neurons = neurons[:n_layers]  # Keep only n_layers neurons
        
        lr = float(lr)
        use_residual = bool(round(use_residual))
        lambda_pde_factor = float(lambda_pde_factor)
        
        architecture = [2] + neurons + [1]
        
        try:
            # Create and train model
            model = ResidualBurgerPINN(architecture, use_residual=use_residual).double()
            data = generate_data(n_collocation=15000, n_boundary=200, n_initial=200)
            
            # Train: Adam-only
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
            
            # Evaluate
            mean_l2, _ = compute_mean_l2_error(model, self.nu, nt=80, nx=300)
            params = sum(p.numel() for p in model.parameters())
            
            # Handle NaN
            if np.isnan(mean_l2) or mean_l2 > 1.0:
                mean_l2 = 1.0
            
            # Store history (for analysis)
            self.eval_history.append({
                'eval': self.eval_count,
                'architecture': str(architecture),
                'n_layers': n_layers,
                'lr': lr,
                'use_residual': use_residual,
                'lambda_pde_factor': lambda_pde_factor,
                'mean_l2_error': mean_l2,
                'params': params,
                'train_time': train_time
            })
            
            if self.verbose and self.eval_count % 1 == 0:
                print(f"  [Eval {self.eval_count}] L2: {mean_l2:.2e}, "
                      f"Arch: {architecture}, Params: {params:,}")
            
            # Return negative error (maximize = minimize error)
            return -mean_l2
            
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Error evaluating: {str(e)[:80]}")
            self.eval_history.append({
                'eval': self.eval_count,
                'architecture': str(architecture),
                'n_layers': n_layers,
                'lr': lr,
                'use_residual': use_residual,
                'lambda_pde_factor': lambda_pde_factor,
                'mean_l2_error': 1.0,
                'params': 20000,
                'train_time': 0.0
            })
            return 0.0  # Very bad score


# ════════════════════════════════════════════════════════════════════════════════
# Main Bayesian Run Function
# ════════════════════════════════════════════════════════════════════════════════

def run_bayes(nu, n_iter=10, init_points=2):
    """
    Run Bayesian Optimization for NAS-PINN
    
    Args:
        nu: viscosity coefficient
        n_iter: total number of iterations (init_points + random exploration)
        init_points: initial random exploration points
    
    Returns:
        dict: best solution found
        optimizer: BayesianOptimization object
    """
    print("\n" + "="*90)
    print("🔍 METHOD: Bayesian Optimization")
    print("="*90)
    print(f"  Viscosity: ν = {nu:.3f}")
    print(f"  Total Iterations: {n_iter}")
    print(f"  Initial Random Points: {init_points}")
    print(f"  Gaussian Process Iterations: {n_iter - init_points}")
    print(f"  Training: {1000} epochs Adam (no L-BFGS, fast evaluation)")
    print(f"  Objective: Maximize -mean_L2_error (minimize error)")
    print(f"  Collocation Points: 15,000")
    print(f"  Domain: x ∈ [-1, 1], t ∈ [0, 1]")
    print("="*90 + "\n")
    
    # Setup optimizer
    optimizer_obj = BayesianNASOptimizer(nu, verbose=True)
    
    # Define parameter bounds
    pbounds = {
        'n_layers': (4, 10),
        'n1': (48, 256),
        'n2': (48, 256),
        'n3': (48, 256),
        'n4': (48, 256),
        'n5': (48, 256),
        'n6': (48, 256),
        'n7': (48, 256),
        'n8': (48, 256),
        'lr': (1e-4, 8e-3),
        'use_residual': (0, 1),
        'lambda_pde_factor': (60, 140)
    }
    
    # Create Bayesian Optimizer
    optimizer = BayesianOptimization(
        f=optimizer_obj.objective_function,
        pbounds=pbounds,
        random_state=42,
        allow_duplicate_points=False
    )
    
    # Progress bar
    pbar = tqdm(total=n_iter, desc="📊 Bayesian Iterations", unit="iter")
    
    # Initial random exploration
    optimizer.maximize(init_points=init_points, n_iter=0)
    pbar.update(init_points)
    
    # Gaussian Process guided exploration
    for i in range(n_iter - init_points):
        optimizer.maximize(init_points=0, n_iter=1)
        pbar.update(1)
        
        # Print progress
        best_val = optimizer.max['target']
        print(f"\n  📈 Iteration {init_points + i + 1}/{n_iter} → Best -L2: {best_val:.2e}")
    
    pbar.close()
    
    # Extract best solution
    best_params = optimizer.max['params']
    best_target = optimizer.max['target']
    best_mean_l2 = -best_target
    
    n_layers = int(round(best_params['n_layers']))
    neurons = [
        int(round(best_params['n1'])),
        int(round(best_params['n2'])),
        int(round(best_params['n3'])),
        int(round(best_params['n4'])),
        int(round(best_params['n5'])),
        int(round(best_params['n6'])),
        int(round(best_params['n7'])),
        int(round(best_params['n8']))
    ]
    neurons = neurons[:n_layers]
    
    lr = best_params['lr']
    use_residual = bool(round(best_params['use_residual']))
    lambda_pde_factor = best_params['lambda_pde_factor']
    
    architecture = [2] + neurons + [1]
    
    # Count parameters
    try:
        model = ResidualBurgerPINN(architecture, use_residual=use_residual).double()
        params = sum(p.numel() for p in model.parameters())
    except:
        params = 20000
    
    # Print results
    print("\n" + "="*90)
    print("🏆 BEST SOLUTION FOUND (by Bayesian Optimization)")
    print("="*90)
    print(f"  Architecture         : {architecture}")
    print(f"  Residual Connections : {use_residual}")
    print(f"  Learning Rate        : {lr:.6f}")
    print(f"  λ_PDE Factor         : {lambda_pde_factor:.2f} → effective λ_PDE = {100*lambda_pde_factor:.1f}")
    print(f"  Mean L2 Error        : {best_mean_l2:.2e}")
    print(f"  Total Parameters     : {params:,}")
    print(f"  Total Evaluations    : {optimizer_obj.eval_count}")
    print("="*90 + "\n")
    
    return {
        'architecture': architecture,
        'layers': n_layers,
        'neurons': neurons,
        'use_residual': use_residual,
        'lr': lr,
        'lambda_pde_factor': lambda_pde_factor,
        'mean_l2_error': best_mean_l2,
        'params': params,
        'n_evals': optimizer_obj.eval_count,
        'eval_history': optimizer_obj.eval_history
    }, optimizer


if __name__ == "__main__":
    viscosities = [0.01]
    
    for nu in viscosities:
        print(f"\n{'#'*90}")
        print(f"  Bayesian Optimization NAS-PINN Search – ν = {nu}")
        print(f"{'#'*90}\n")
        
        best_result, optimizer = run_bayes(nu, n_iter=10, init_points=2)
        
        print(f"✅ Search completed. Best architecture: {best_result['architecture']}")

def evaluate_architecture(arch_encoding, data, nu, device='cpu'):
    """
    Evaluate a single architecture with detailed error logging
    """
    try:
        # Build model
        model = ResidualBurgerPINN(arch_encoding, use_residual=True).double()
        
        if model is None:
            print(f"  ⚠️  Model building failed for {arch_encoding}")
            return 1e6  # Return large penalty
        
        # Train model
        model, history = train_pinn(
            model, nu, data,
            epochs=1000,
            lr=3e-4,
            lambda_pde=100.0,
            lambda_bc=20.0,
            lambda_ic=5.0,
            patience=100,
            verbose=True,
            use_lbfgs=False
        )
        
        # Compute final loss
        final_loss = history[-1] if history else 1e6
        param_count = sum(p.numel() for p in model.parameters())
        
        # Multi-objective: minimize loss and parameters
        objective = final_loss + 0.0001 * param_count
        
        return -objective  # Negative for maximization
        
    except RuntimeError as e:
        print(f"  ⚠️  Runtime error for {arch_encoding}: {str(e)[:80]}")
        return 1e6
    except Exception as e:
        print(f"  ⚠️  Unexpected error for {arch_encoding}: {str(e)[:80]}")
        return 1e6