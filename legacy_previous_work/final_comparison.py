import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════════
# Load all results
# ════════════════════════════════════════════════════════════════════════════════

def load_all_results():
    """Load Stage 1, 2, 3 results from CSV files"""
    
    stages = ['adam', 'adam_lbfgs', 'adam_pso']
    all_data = []
    
    for stage in stages:
        # Try to load summary CSV
        summary_file = f"results/stage{stages.index(stage)+1}_{stage}_summary.csv"
        
        if os.path.exists(summary_file):
            df = pd.read_csv(summary_file)
            df['stage'] = stage
            all_data.append(df)
            print(f"✅ Loaded: {summary_file} ({len(df)} rows)")
        else:
            print(f"⚠️  Not found: {summary_file}")
    
    if not all_data:
        print("❌ No results found!")
        return None

    df = pd.concat(all_data, ignore_index=True)

    # Normalize method names across stages/files
    method_map = {
        'NSGA2': 'NSGA-II',
        'NSGA-II': 'NSGA-II',
        'NSGA3': 'NSGA-III',
        'NSGA-III': 'NSGA-III',
        'BAYES': 'Bayesian',
        'Bayesian': 'Bayesian'
    }
    df['method'] = df['method'].astype(str).str.strip().map(method_map).fillna(df['method'])

    return df


# ════════════════════════════════════════════════════════════════════════════════
# Summary Statistics
# ════════════════════════════════════════════════════════════════════════════════

def print_summary_stats(df):
    """Print summary statistics for each stage and method"""
    
    print("\n" + "="*120)
    print("📊 SUMMARY STATISTICS BY STAGE")
    print("="*120 + "\n")
    
    for stage in ['adam', 'adam_lbfgs', 'adam_pso']:
        df_stage = df[df['stage'] == stage]
        
        if df_stage.empty:
            continue
        
        print(f"\n{'─'*120}")
        print(f"STAGE: {stage.upper()}")
        print(f"{'─'*120}\n")
        
        for method in ['NSGA-II', 'NSGA-III', 'Bayesian']:
            df_method = df_stage[df_stage['method'] == method]
            
            if df_method.empty:
                continue
            
            print(f"  {method}:")
            
            for nu in sorted(df_method['nu'].unique()):
                df_nu = df_method[df_method['nu'] == nu]
                
                mean_l2 = df_nu['mean_l2_error'].mean()
                std_l2 = df_nu['mean_l2_error'].std()
                params = df_nu['params'].mean()
                
                print(f"    ν = {nu}: L2 = {mean_l2:.2e} (±{std_l2:.2e}), Params = {params:,.0f}")
        
        print()


# ════════════════════════════════════════════════════════════════════════════════
# Comparison Table
# ════════════════════════════════════════════════════════════════════════════════

def create_comparison_table(df):
    """Create comprehensive comparison table"""
    
    print("\n" + "="*150)
    print("📋 COMPREHENSIVE COMPARISON TABLE")
    print("="*150 + "\n")
    
    for nu in sorted(df['nu'].unique()):
        df_nu = df[df['nu'] == nu]
        
        print(f"\n{'─'*150}")
        print(f"ν = {nu}")
        print(f"{'─'*150}\n")
        
        # Pivot table: method × stage
        pivot_data = []
        
        for method in ['NSGA-II', 'NSGA-III', 'Bayesian']:
            row = {'Method': method}
            
            for stage in ['adam', 'adam_lbfgs', 'adam_pso']:
                df_cell = df_nu[(df_nu['method'] == method) & (df_nu['stage'] == stage)]
                
                if not df_cell.empty:
                    l2 = df_cell['mean_l2_error'].iloc[0]
                    row[f"{stage.upper()}"] = f"{l2:.2e}"
                else:
                    row[f"{stage.upper()}"] = "N/A"
            
            pivot_data.append(row)
        
        df_pivot = pd.DataFrame(pivot_data)
        print(df_pivot.to_string(index=False))
    
    print("\n" + "="*150 + "\n")


# ════════════════════════════════════════════════════════════════════════════════
# Visualizations
# ════════════════════════════════════════════════════════════════════════════════

def create_visualizations(df):
    """Create comparison plots"""
    
    sns.set_style("whitegrid")
    
    # 1. L2 Error Comparison (Bar plot)
    print("\n📊 Creating visualization: L2 Error Comparison (Bar)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for nu_idx, nu in enumerate(sorted(df['nu'].unique())):
        df_nu = df[df['nu'] == nu]
        
        pivot = df_nu.pivot_table(
            index='method',
            columns='stage',
            values='mean_l2_error',
            aggfunc='mean'
        )
        
        pivot.plot(kind='bar', ax=axes[nu_idx], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[nu_idx].set_title(f'ν = {nu}', fontsize=12, fontweight='bold')
        axes[nu_idx].set_ylabel('Mean L2 Error', fontsize=11)
        axes[nu_idx].set_xlabel('Method', fontsize=11)
        axes[nu_idx].set_yscale('log')
        axes[nu_idx].legend(title='Stage', fontsize=9)
        axes[nu_idx].tick_params(axis='x', rotation=45)
        axes[nu_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results/visualizations', exist_ok=True)
    plt.savefig('results/visualizations/01_l2_error_comparison_bar.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: results/visualizations/01_l2_error_comparison_bar.png")
    plt.close()
    
    # 2. L2 Error Improvement (Line plot)
    print("📊 Creating visualization: L2 Error Improvement (Line)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    stages_order = ['adam', 'adam_lbfgs', 'adam_pso']
    stage_labels = ['Adam', 'Adam+LBFGS', 'Adam+PSO']
    
    for nu_idx, nu in enumerate(sorted(df['nu'].unique())):
        df_nu = df[df['nu'] == nu]
        ax = axes[nu_idx]
        
        for method in ['NSGA-II', 'NSGA-III', 'Bayesian']:
            df_method = df_nu[df_nu['method'] == method]
            
            l2_values = []
            for stage in stages_order:
                df_cell = df_method[df_method['stage'] == stage]
                if not df_cell.empty:
                    l2_values.append(df_cell['mean_l2_error'].iloc[0])
                else:
                    l2_values.append(np.nan)
            
            ax.plot(stage_labels, l2_values, marker='o', label=method, linewidth=2, markersize=8)
        
        ax.set_title(f'ν = {nu}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean L2 Error', fontsize=11)
        ax.set_xlabel('Stage', fontsize=11)
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig('results/visualizations/02_l2_error_improvement_line.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: results/visualizations/02_l2_error_improvement_line.png")
    plt.close()
    
    # 3. Improvement Percentage
    print("📊 Creating visualization: L2 Error Improvement %...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for nu_idx, nu in enumerate(sorted(df['nu'].unique())):
        df_nu = df[df['nu'] == nu]
        ax = axes[nu_idx]
        
        improvement_data = []
        labels = []
        
        for method in ['NSGA-II', 'NSGA-III', 'Bayesian']:
            df_method = df_nu[df_nu['method'] == method]
            
            adam_l2 = df_method[df_method['stage'] == 'adam']['mean_l2_error'].iloc[0] if not df_method[df_method['stage'] == 'adam'].empty else 1.0
            lbfgs_l2 = df_method[df_method['stage'] == 'adam_lbfgs']['mean_l2_error'].iloc[0] if not df_method[df_method['stage'] == 'adam_lbfgs'].empty else 1.0
            pso_l2 = df_method[df_method['stage'] == 'adam_pso']['mean_l2_error'].iloc[0] if not df_method[df_method['stage'] == 'adam_pso'].empty else 1.0
            
            # Improvement from Adam
            lbfgs_improvement = ((adam_l2 - lbfgs_l2) / adam_l2) * 100
            pso_improvement = ((adam_l2 - pso_l2) / adam_l2) * 100
            
            improvement_data.append([lbfgs_improvement, pso_improvement])
            labels.append(method)
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, [d[0] for d in improvement_data], width, label='LBFGS', color='#ff7f0e')
        ax.bar(x + width/2, [d[1] for d in improvement_data], width, label='PSO', color='#2ca02c')
        
        ax.set_title(f'ν = {nu}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Improvement (%)', fontsize=11)
        ax.set_xlabel('Method', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=9)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/03_l2_error_improvement_percent.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: results/visualizations/03_l2_error_improvement_percent.png")
    plt.close()
    
    # 4. Parameter Count Comparison
    print("📊 Creating visualization: Parameter Count...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_pivot = df.groupby(['stage', 'method'])['params'].mean().reset_index()
    pivot = df_pivot.pivot(index='method', columns='stage', values='params')
    
    # Reorder columns
    pivot = pivot[['adam', 'adam_lbfgs', 'adam_pso']]
    pivot.columns = ['Adam', 'Adam+LBFGS', 'Adam+PSO']
    
    pivot.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'], width=0.7)
    ax.set_title('Average Parameter Count by Method & Stage', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Parameters', fontsize=11)
    ax.set_xlabel('Method', fontsize=11)
    ax.legend(title='Stage', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/04_parameter_count.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: results/visualizations/04_parameter_count.png")
    plt.close()
    
    # 5. Stage-wise Improvement Heatmap
    print("📊 Creating visualization: Stage-wise Improvement Heatmap...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    for nu_idx, nu in enumerate(sorted(df['nu'].unique())):
        df_nu = df[df['nu'] == nu]
        
        improvement_matrix = []
        methods = ['NSGA-II', 'NSGA-III', 'Bayesian']
        stages = ['Adam→LBFGS', 'LBFGS→PSO', 'Adam→PSO']
        
        for method in methods:
            df_method = df_nu[df_nu['method'] == method]
            
            adam_l2 = df_method[df_method['stage'] == 'adam']['mean_l2_error'].iloc[0] if not df_method[df_method['stage'] == 'adam'].empty else 1.0
            lbfgs_l2 = df_method[df_method['stage'] == 'adam_lbfgs']['mean_l2_error'].iloc[0] if not df_method[df_method['stage'] == 'adam_lbfgs'].empty else 1.0
            pso_l2 = df_method[df_method['stage'] == 'adam_pso']['mean_l2_error'].iloc[0] if not df_method[df_method['stage'] == 'adam_pso'].empty else 1.0
            
            imp_lbfgs = ((adam_l2 - lbfgs_l2) / adam_l2) * 100
            imp_pso = ((lbfgs_l2 - pso_l2) / lbfgs_l2) * 100 if lbfgs_l2 > 0 else 0
            imp_adam_pso = ((adam_l2 - pso_l2) / adam_l2) * 100
            
            improvement_matrix.append([imp_lbfgs, imp_pso, imp_adam_pso])
        
        im = axes[nu_idx].imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=100)
        axes[nu_idx].set_xticks(np.arange(len(stages)))
        axes[nu_idx].set_yticks(np.arange(len(methods)))
        axes[nu_idx].set_xticklabels(stages, fontsize=9)
        axes[nu_idx].set_yticklabels(methods, fontsize=9)
        axes[nu_idx].set_title(f'ν = {nu}', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(methods)):
            for j in range(len(stages)):
                text = axes[nu_idx].text(j, i, f'{improvement_matrix[i][j]:.1f}%',
                                        ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=axes[nu_idx], label='Improvement (%)')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/05_improvement_heatmap.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: results/visualizations/05_improvement_heatmap.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# Best Model Summary
# ════════════════════════════════════════════════════════════════════════════════

def print_best_models(df):
    """Print best models overall and per stage"""
    
    print("\n" + "="*120)
    print("🏆 BEST MODELS SUMMARY")
    print("="*120 + "\n")
    
    # Best overall (all stages, all methods, all nu)
    best_overall_idx = df['mean_l2_error'].idxmin()
    best_overall = df.loc[best_overall_idx]
    
    print(f"🥇 BEST OVERALL:")
    print(f"   Stage:     {best_overall['stage'].upper()}")
    print(f"   Method:    {best_overall['method']}")
    print(f"   ν:         {best_overall['nu']}")
    print(f"   L2 Error:  {best_overall['mean_l2_error']:.2e}")
    print(f"   Params:    {best_overall['params']:,}")
    print()
    
    # Best per stage
    for stage in ['adam', 'adam_lbfgs', 'adam_pso']:
        df_stage = df[df['stage'] == stage]
        
        if df_stage.empty:
            continue
        
        best_idx = df_stage['mean_l2_error'].idxmin()
        best = df_stage.loc[best_idx]
        
        stage_name = {'adam': 'Adam Only', 'adam_lbfgs': 'Adam+LBFGS', 'adam_pso': 'Adam+PSO'}[stage]
        
        print(f"🏆 BEST IN {stage_name}:")
        print(f"   Method:    {best['method']}")
        print(f"   ν:         {best['nu']}")
        print(f"   L2 Error:  {best['mean_l2_error']:.2e}")
        print(f"   Params:    {best['params']:,}")
        print()


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*120)
    print(" 🎯 NAS-PINNs: FINAL COMPARISON (3 Stages)")
    print("="*120 + "\n")
    
    # Load results
    print("📂 Loading results from all stages...\n")
    df = load_all_results()
    
    if df is None:
        return
    
    print(f"\n✅ Total records loaded: {len(df)}")
    print(f"   Stages: {sorted(df['stage'].unique())}")
    print(f"   Methods: {sorted(df['method'].unique())}")
    print(f"   Viscosities: {sorted(df['nu'].unique())}\n")
    
    # Print summaries
    print_summary_stats(df)
    create_comparison_table(df)
    print_best_models(df)
    
    # Create visualizations
    print("\n" + "="*120)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*120 + "\n")
    
    create_visualizations(df)
    
    # Save master comparison CSV
    print("\n💾 Saving master comparison CSV...")
    df_sorted = df.sort_values(['nu', 'method', 'stage']).reset_index(drop=True)
    df_sorted.to_csv('results/MASTER_COMPARISON.csv', index=False)
    print("   ✅ Saved: results/MASTER_COMPARISON.csv\n")
    
    print("="*120)
    print("✅ FINAL COMPARISON COMPLETED!")
    print("="*120)
    print(f"\nResults saved in:")
    print(f"  - results/MASTER_COMPARISON.csv (Master table)")
    print(f"  - results/visualizations/ (5 comparison plots)")
    print(f"  - results/adam/, results/adam_lbfgs/, results/adam_pso/ (Stage outputs)")
    print("\n" + "="*120 + "\n")


if __name__ == "__main__":
    main()