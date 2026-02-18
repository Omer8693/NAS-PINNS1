import os
import time
import torch
import pandas as pd
import numpy as np
import traceback
from tqdm import tqdm

from nsga2_search import run_nsga2
from nsga3_search import run_nsga3
from bayes_opt_search import run_bayes
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
# Utility: Save stage outputs
# ════════════════════════════════════════════════════════════════════════════════

def save_stage_outputs(stage, method, nu, model, loss_history, mean_l2_error, arch, extra=None):
    """
    Save outputs for each stage (adam / adam_lbfgs / adam_pso)
    
    stage: 'adam' | 'adam_lbfgs' | 'adam_pso'
    method: 'nsga2' | 'nsga3' | 'bayes'
    """
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
        print(f"     ✅ Model saved ({output_dir}/model.pt)")
    except Exception as e:
        print(f"     ⚠️  Model save error: {str(e)[:60]}")
    
    # Results CSV
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
# Main Loop
# ════════════════════════════════════════════════════════════════════════════════

def main():
    viscosities = [0.01, 0.04, 0.07]
    all_results = []
    
    print("="*100)
    print(" 🧠 NAS-PINNs: Neural Architecture Search for Physics-Informed Neural Networks")
    print("           Stage 1: Adam Baseline (Fast NAS Evaluation)")
    print("="*100)
    print(f"\nEquation: u_t + u*u_x - (ν/π)*u_xx = 0")
    print(f"Domain: x ∈ [-1, 1], t ∈ [0, 1]")
    print(f"Viscosities: ν = {viscosities}")
    print(f"Training: {1000} epochs Adam (no L-BFGS)")
    print(f"Collocation: 15,000 points\n")
    print("="*100 + "\n")
    
    # Main progress bar
    main_pbar = tqdm(total=len(viscosities)*3, desc="📊 Overall Progress", unit="search", position=0)
    
    for nu_idx, nu in enumerate(viscosities):
        print(f"\n{'='*100}")
        print(f"STAGE 1: ADAM BASELINE – ν = {nu} [{nu_idx+1}/{len(viscosities)}]")
        print(f"{'='*100}\n")
        
        nu_results = []
        
        # ───────────────────────────────────────────────────────────────────────────
        # NSGA-II
        # ───────────────────────────────────────────────────────────────────────────
        print(f"{'─'*100}\n")
        main_pbar.set_description(f"📊 ν={nu} NSGA-II (Adam)")
        
        try:
            print("🔍 Running NSGA-II search...\n")
            best_nsga2, _ = run_nsga2(nu, pop=10, ngen=5)
            
            arch = best_nsga2['architecture']
            print(f"\n✅ NSGA-II search completed")
            print(f"   Best architecture: {arch}")
            print(f"   Mean L2 error: {best_nsga2['mean_l2_error']:.2e}\n")
            
            # Save outputs
            print(f"💾 Saving NSGA-II results (Adam stage)...")
            save_stage_outputs(
                stage="adam",
                method="nsga2",
                nu=nu,
                model=None,  # Will be reloaded if needed
                loss_history=[],
                mean_l2_error=best_nsga2['mean_l2_error'],
                arch=arch,
                extra={
                    'lr': best_nsga2['lr'],
                    'params': best_nsga2['params'],
                    'use_residual': best_nsga2['use_residual'],
                    'search_time': best_nsga2['search_time']
                }
            )
            
            nu_results.append({
                'method': 'NSGA-II',
                'stage': 'adam',
                'nu': nu,
                'architecture': str(arch),
                'mean_l2_error': best_nsga2['mean_l2_error'],
                'params': best_nsga2['params'],
                'search_time': best_nsga2['search_time']
            })
            
        except Exception as e:
            print(f"❌ NSGA-II error: {str(e)}")
            traceback.print_exc()
        
        main_pbar.update(1)
        
        # ───────────────────────────────────────────────────────────────────────────
        # NSGA-III
        # ───────────────────────────────────────────────────────────────────────────
        print(f"\n{'─'*100}\n")
        main_pbar.set_description(f"📊 ν={nu} NSGA-III (Adam)")
        
        try:
            print("🔍 Running NSGA-III search...\n")
            best_nsga3, _ = run_nsga3(nu, pop=20, ngen=5)
            
            arch = best_nsga3['architecture']
            print(f"\n✅ NSGA-III search completed")
            print(f"   Best architecture: {arch}")
            print(f"   Mean L2 error: {best_nsga3['mean_l2_error']:.2e}\n")
            
            print(f"💾 Saving NSGA-III results (Adam stage)...")
            save_stage_outputs(
                stage="adam",
                method="nsga3",
                nu=nu,
                model=None,
                loss_history=[],
                mean_l2_error=best_nsga3['mean_l2_error'],
                arch=arch,
                extra={
                    'lr': best_nsga3['lr'],
                    'params': best_nsga3['params'],
                    'use_residual': best_nsga3['use_residual'],
                    'search_time': best_nsga3['search_time']
                }
            )
            
            nu_results.append({
                'method': 'NSGA-III',
                'stage': 'adam',
                'nu': nu,
                'architecture': str(arch),
                'mean_l2_error': best_nsga3['mean_l2_error'],
                'params': best_nsga3['params'],
                'search_time': best_nsga3['search_time']
            })
            
        except Exception as e:
            print(f"❌ NSGA-III error: {str(e)}")
            traceback.print_exc()
        
        main_pbar.update(1)
        
        # ───────────────────────────────────────────────────────────────────────────
        # Bayesian
        # ───────────────────────────────────────────────────────────────────────────
        print(f"\n{'─'*100}\n")
        main_pbar.set_description(f"📊 ν={nu} Bayesian (Adam)")
        
        try:
            print("🔍 Running Bayesian Optimization...\n")
            best_bayes, _ = run_bayes(nu, n_iter=10, init_points=3)
            
            arch = best_bayes['architecture']
            print(f"\n✅ Bayesian search completed")
            print(f"   Best architecture: {arch}")
            print(f"   Mean L2 error: {best_bayes['mean_l2_error']:.2e}\n")
            
            print(f"💾 Saving Bayesian results (Adam stage)...")
            save_stage_outputs(
                stage="adam",
                method="bayes",
                nu=nu,
                model=None,
                loss_history=[],
                mean_l2_error=best_bayes['mean_l2_error'],
                arch=arch,
                extra={
                    'lr': best_bayes['lr'],
                    'params': best_bayes['params'],
                    'use_residual': best_bayes['use_residual'],
                    'n_evals': best_bayes['n_evals']
                }
            )
            
            nu_results.append({
                'method': 'Bayesian',
                'stage': 'adam',
                'nu': nu,
                'architecture': str(arch),
                'mean_l2_error': best_bayes['mean_l2_error'],
                'params': best_bayes['params'],
                'n_evals': best_bayes['n_evals']
            })
            
        except Exception as e:
            print(f"❌ Bayesian error: {str(e)}")
            traceback.print_exc()
        
        main_pbar.update(1)
        
        # Save CSV for this viscosity
        if nu_results:
            df_nu = pd.DataFrame(nu_results)
            os.makedirs(f"results/adam", exist_ok=True)
            df_nu.to_csv(f"results/adam/nu_{nu}_summary.csv", index=False)
            all_results.extend(nu_results)
        
        print(f"\n{'='*100}")
        print(f"✅ STAGE 1 completed for ν = {nu}")
        print(f"{'='*100}\n")
    
    main_pbar.close()
    
    # Overall summary
    if all_results:
        df_all = pd.DataFrame(all_results)
        os.makedirs("results", exist_ok=True)
        df_all.to_csv("results/stage1_adam_summary.csv", index=False)
        
        print(f"\n{'='*100}")
        print("📋 STAGE 1 SUMMARY")
        print(f"{'='*100}\n")
        print(df_all.to_string(index=False))
        print(f"\n✅ Stage 1 results saved: results/stage1_adam_summary.csv")
        print(f"{'='*100}\n")


if __name__ == "__main__":
    main()