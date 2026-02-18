import os
import time
import torch
import pandas as pd
import numpy as np
from pyswarm import pso
from tqdm import tqdm

from naspinn import (
    ResidualBurgerPINN,
    generate_data,
    train_pinn,
    compute_mean_l2_error,
    plot_loss_curve,
    plot_solution_heatmap,
    plot_solution_snapshots,
    plot_comparison_at_final_time,
    plot_error_distribution,
    set_seed
)

# ════════════════════════════════════════════════════════════════════════════════
# Save outputs
# ════════════════════════════════════════════════════════════════════════════════

def save_stage_outputs(stage, method, nu, model, loss_history, mean_l2_error, arch, extra=None):
    """Save Stage 3 (Adam+PSO) outputs"""
    output_dir = f"results/{stage}/{method}/nu_{nu}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  📸 Generating visualizations...")
    
    # Plots
    try:
        plot_loss_curve(loss_history, f"{output_dir}/loss_curve.png")
        plot_solution_heatmap(model, nu, f"{output_dir}/heatmap.png")
        plot_solution_snapshots(model, nu, f"{output_dir}/snapshots.png")
        plot_comparison_at_final_time(model, nu, f"{output_dir}/comparison.png")
        plot_error_distribution(model, nu, f"{output_dir}/error_dist.png")
        print(f"     ✅ Plots saved")
    except Exception as e:
        print(f"     ⚠️  Plot error: {str(e)[:60]}")
    
    # Model
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': arch,
            'loss_history': loss_history,
            'mean_l2_error': mean_l2_error,
            'extra': extra
        }, f"{output_dir}/model.pt")
        print(f"     ✅ Model saved")
    except Exception as e:
        print(f"     ⚠️  Model save error: {str(e)[:60]}")
    
    # CSV
    try:
        row = {
            'stage': stage,
            'method': method,
            'nu': nu,
            'architecture': str(arch),
            'mean_l2_error': mean_l2_error,
        }
        if extra:
            row.update(extra)
        
        pd.DataFrame([row]).to_csv(f"{output_dir}/results.csv", index=False)
        print(f"     ✅ Results CSV saved")
    except Exception as e:
        print(f"     ⚠️  CSV save error: {str(e)[:60]}")


# ════════════════════════════════════════════════════════════════════════════════
# PSO Objective: Optimize Learning Rate & Lambda_PDE
# ════════════════════════════════════════════════════════════════════════════════

class PSO_Evaluator:
    def __init__(self, arch, use_residual, nu):
        self.arch = arch
        self.use_residual = use_residual
        self.nu = nu
        self.eval_count = 0
        self.history = []
    
    def objective(self, params):
        """
        PSO minimizes this function
        params = [lr, lambda_pde_factor]
        """
        self.eval_count += 1
        lr = float(params[0])
        lambda_pde_factor = float(params[1])
        
        try:
            model = ResidualBurgerPINN(self.arch, use_residual=self.use_residual).double()
            data = generate_data(n_collocation=15000, n_boundary=200, n_initial=200)
            
            model, _ = train_pinn(
                model,
                self.nu,
                data,
                epochs=500,  # Short training for PSO iterations
                lr=lr,
                lambda_pde=100.0 * lambda_pde_factor,
                lambda_bc=20.0,
                lambda_ic=5.0,
                patience=100,
                verbose=False,
                use_lbfgs=False
            )
            
            mean_l2, _ = compute_mean_l2_error(model, self.nu, nt=80, nx=300)
            
            if np.isnan(mean_l2):
                mean_l2 = 1.0
            
            self.history.append({
                'eval': self.eval_count,
                'lr': lr,
                'lambda_pde_factor': lambda_pde_factor,
                'mean_l2_error': mean_l2
            })
            
            return mean_l2
        
        except Exception as e:
            return 1.0


# ════════════════════════════════════════════════════════════════════════════════
# Main: Load best models and optimize with PSO
# ════════════════════════════════════════════════════════════════════════════════

def main():
    viscosities = [0.01, 0.04, 0.07]
    methods = ['nsga2', 'nsga3', 'bayes']
    all_results = []
    
    print("\n" + "="*100)
    print(" 🧠 NAS-PINNs: Stage 3 – Adam + PSO Hyperparameter Optimization")
    print("="*100)
    print(f"\nFor each best architecture:")
    print(f"  1. Use PSO to optimize: [learning_rate, lambda_pde_factor]")
    print(f"  2. Train final model with optimized hyperparameters")
    print(f"  3. Evaluate & Save\n")
    print("="*100 + "\n")
    
    total_tasks = len(viscosities) * len(methods)
    pbar = tqdm(total=total_tasks, desc="📊 PSO Optimization", unit="model")
    
    for nu in viscosities:
        print(f"\n{'='*100}")
        print(f"STAGE 3: ADAM + PSO – ν = {nu}")
        print(f"{'='*100}\n")
        
        nu_results = []
        
        for method in methods:
            print(f"{'─'*100}")
            print(f"PSO-optimizing {method.upper()} best architecture...\n")
            
            try:
                # Load Stage 1 results
                stage1_csv = f"results/adam/{method}/nu_{nu}/results.csv"
                if not os.path.exists(stage1_csv):
                    print(f"  ⚠️  Stage 1 results not found")
                    pbar.update(1)
                    continue
                
                df = pd.read_csv(stage1_csv)
                arch_str = df['architecture'].iloc[0]
                arch = eval(arch_str)
                use_residual = df.get('use_residual', [True]).iloc[0] if 'use_residual' in df else True
                
                print(f"  Architecture: {arch}")
                print(f"  Use Residual: {use_residual}\n")
                
                # PSO optimization
                print(f"  🐝 Running PSO (20 particles, 10 iterations)...")
                evaluator = PSO_Evaluator(arch, use_residual, nu)
                
                xopt, fopt = pso(
                    evaluator.objective,
                    lb=[1e-4, 60.0],      # Learning rate min, lambda_pde_factor min
                    ub=[8e-3, 140.0],     # Learning rate max, lambda_pde_factor max
                    maxiter=10,
                    swarmsize=20,
                    debug=False,
                    processes=1
                )
                
                best_lr = xopt[0]
                best_lambda_pde_factor = xopt[1]
                best_pso_error = fopt
                
                print(f"  ✅ PSO completed")
                print(f"     Best LR: {best_lr:.6f}")
                print(f"     Best λ_PDE factor: {best_lambda_pde_factor:.2f}")
                print(f"     Best error (PSO): {best_pso_error:.2e}\n")
                
                # Final training with optimized hyperparams
                print(f"  ⏳ Final training with optimized hyperparameters...")
                t0 = time.time()
                model = ResidualBurgerPINN(arch, use_residual=use_residual).double()
                data = generate_data(n_collocation=15000, n_boundary=200, n_initial=200)
                
                model, loss_history = train_pinn(
                    model,
                    nu,
                    data,
                    epochs=1000,
                    lr=best_lr,
                    lambda_pde=100.0 * best_lambda_pde_factor,
                    lambda_bc=20.0,
                    lambda_ic=5.0,
                    patience=120,
                    verbose=False,
                    use_lbfgs=False
                )
                train_time = time.time() - t0
                
                print(f"  ✅ Final training completed ({train_time:.1f}s)\n")
                
                # Evaluate
                print(f"  📊 Computing L2 error...")
                mean_l2, _ = compute_mean_l2_error(model, nu, nt=80, nx=300)
                params = sum(p.numel() for p in model.parameters())
                
                print(f"  ✅ Mean L2 error: {mean_l2:.2e}\n")
                
                # Save
                print(f"  💾 Saving results...")
                save_stage_outputs(
                    stage="adam_pso",
                    method=method,
                    nu=nu,
                    model=model,
                    loss_history=loss_history,
                    mean_l2_error=mean_l2,
                    arch=arch,
                    extra={
                        'lr': best_lr,
                        'lambda_pde_factor': best_lambda_pde_factor,
                        'params': params,
                        'use_residual': use_residual,
                        'train_time': train_time,
                        'pso_error': best_pso_error,
                        'pso_evals': evaluator.eval_count
                    }
                )
                
                nu_results.append({
                    'method': method.upper(),
                    'stage': 'adam_pso',
                    'nu': nu,
                    'architecture': str(arch),
                    'mean_l2_error': mean_l2,
                    'pso_error': best_pso_error,
                    'params': params,
                    'train_time': train_time,
                    'lr': best_lr,
                    'lambda_pde_factor': best_lambda_pde_factor
                })
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
            
            pbar.update(1)
        
        # Save nu summary
        if nu_results:
            df_nu = pd.DataFrame(nu_results)
            df_nu.to_csv(f"results/adam_pso/nu_{nu}_summary.csv", index=False)
            all_results.extend(nu_results)
    
    pbar.close()
    
    # Overall summary
    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv("results/stage3_adam_pso_summary.csv", index=False)
        
        print(f"\n{'='*100}")
        print("📋 STAGE 3 SUMMARY")
        print(f"{'='*100}\n")
        print(df_all.to_string(index=False))
        print(f"\n✅ Stage 3 results saved: results/stage3_adam_pso_summary.csv")
        print(f"{'='*100}\n")


if __name__ == "__main__":
    main()