"""
thermal_pinn/training/train_all.py
====================================
Main training entry point.

For each combination of (domain × architecture × k_skip), trains a ThermalPINN
over the full 30-second quenching simulation and saves:
    - checkpoint: checkpoints/<domain>_<arch>_k<k>.pt
    - metrics:    checkpoints/<domain>_<arch>_k<k>_metrics.json

After all k values are trained for a given (domain, arch), the best k is
selected by minimum mean MAE and recorded in checkpoints/registry.json.
All k checkpoints are retained for comparison.

Usage
-----
    # Train a single combination (quick test):
    python -m thermal_pinn.training.train_all --domain rectangle --arch nsga2 --k 3

    # Sweep all k values for one domain/arch:
    python -m thermal_pinn.training.train_all --domain cylinder --arch bayesian --sweep_k

    # Full run: all domains × all archs × all k (long, use GPU):
    python -m thermal_pinn.training.train_all --all --cuda

    # Only 3D domains:
    python -m thermal_pinn.training.train_all --all --dim 3 --cuda

k-skip strategy
---------------
k=1 : PINN covers one FEM step (Δt=1.5s)  — highest accuracy, most FEM calls
k=2 : PINN covers 2 FEM steps (3.0s)
k=3 : PINN covers 3 FEM steps (4.5s)      — recommended default
k=4 : PINN covers 4 FEM steps (6.0s)
k=5 : PINN covers 5 FEM steps (7.5s)      — fewest FEM calls, highest PINN load

Best k is the one with minimum mean MAE across all windows.
For large Bi numbers (Bi > 10), k=3 typically gives the best MAE/cost trade-off
based on the Level 10 MSWP sweep results (see level10_multiwindow/results/).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Allow running as script from repo root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent))

from thermal_pinn.physics.domains_2d import make_domain, DOMAINS_2D
from thermal_pinn.physics.domains_3d import make_domain_3d, DOMAINS_3D
from thermal_pinn.network.pinn       import make_pinn, ARCH_CONFIGS
from thermal_pinn.training.trainer   import (
    train_window, evaluate_window, K_VALUES
)

# ──────────────────────────────────────────────────────────────────────────────
# Simulation constants
# ──────────────────────────────────────────────────────────────────────────────
T_TOTAL      = 30.0    # s  — total quenching time (Mortensen et al., 2026)
DT_FEM       = 1.5     # s  — FEM time step (matches Level 2–8 experiments)
N_FEM_STEPS  = int(T_TOTAL / DT_FEM)   # 20 steps
T_WATER      = 20.0
DELTA_T      = 520.0

TRAIN_KWARGS = {
    "n_domain":  1500,
    "n_bc":      300,
    "n_epochs":  800,
    "lr":        1e-3,
    "lr_min":    1e-5,
    "lbfgs_iters": 50,
    "use_end_supervision": True,
}

CHECKPOINT_DIR = _ROOT / "checkpoints"


def theta_from_domain(domain, coords_np: np.ndarray, t: float) -> np.ndarray:
    """
    Return dimensionless IC temperature θ = (T − T_w) / ΔT at time t.
    Dispatches to 2D or 3D domain .T() method.
    """
    if coords_np.shape[1] == 2:
        T_vals = domain.T(coords_np[:, 0], coords_np[:, 1], t)
    else:
        T_vals = domain.T(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], t)
    T_vals = np.where(np.isfinite(T_vals), T_vals, T_WATER)
    return ((T_vals - T_WATER) / DELTA_T).astype(np.float32)


def run_one(domain_name: str, arch: str, k: int, dim: int,
            device: torch.device, verbose: bool = True) -> dict:
    """
    Train a single (domain, arch, k) combination.

    For each anchor in [0, T_TOTAL] with step k·DT_FEM:
        1. Build θ_ic from domain's analytical/FD solution at t_anchor
        2. Train PINN on [t_anchor, t_anchor + k·DT_FEM]
        3. Evaluate MAE against reference at t_anchor + k·DT_FEM

    Returns metrics dict with per-window MAEs and the saved checkpoint path.
    """
    if dim == 2:
        domain = make_domain(domain_name)
    else:
        domain = make_domain_3d(domain_name)

    model = make_pinn(dim=dim, arch=arch, device=device)

    dt_window = k * DT_FEM
    anchors   = np.arange(0.0, T_TOTAL - dt_window / 2, dt_window)

    window_results = []
    t0_total = time.time()

    for i, t_start in enumerate(anchors):
        t_end = min(t_start + dt_window, T_TOTAL)
        if verbose:
            print(f"  [{i+1}/{len(anchors)}] Window [{t_start:.1f} → {t_end:.1f}s]  "
                  f"k={k}  {domain_name}/{arch}", flush=True)

        def theta_ic_fn(coords_np, _t=t_start):
            return theta_from_domain(domain, coords_np, _t)

        def theta_end_fn(coords_np, _t=t_end):
            return theta_from_domain(domain, coords_np, _t)

        train_info = train_window(
            model=model, domain=domain,
            t_start=t_start, t_end=t_end,
            theta_ic_fn=theta_ic_fn,
            theta_end_fn=theta_end_fn,
            device=device,
            rng_seed=i,
            **TRAIN_KWARGS,
        )

        eval_info = evaluate_window(
            model=model, domain=domain,
            t_start=t_start, t_end=t_end,
            theta_ic_fn=theta_ic_fn,
            device=device,
        )

        if verbose:
            print(f"      MAE={eval_info['mae_C']:.2f}°C  "
                  f"L2={eval_info['l2_rel']:.4f}  "
                  f"({train_info['runtime_s']:.0f}s)", flush=True)

        window_results.append({
            "t_start":    float(t_start),
            "t_end":      float(t_end),
            "mae_C":      eval_info["mae_C"],
            "l2_rel":     eval_info["l2_rel"],
            "runtime_s":  train_info["runtime_s"],
        })

    mean_mae = float(np.mean([w["mae_C"] for w in window_results]))
    total_t  = time.time() - t0_total

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{domain_name}_{arch}_k{k}_dim{dim}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "arch":        arch,
        "dim":         dim,
        "domain":      domain_name,
        "k":           k,
        "mean_mae":    mean_mae,
        "windows":     window_results,
    }, ckpt_path)

    result = {
        "domain":     domain_name,
        "arch":       arch,
        "k":          k,
        "dim":        dim,
        "mean_mae":   mean_mae,
        "n_windows":  len(window_results),
        "windows":    window_results,
        "total_s":    total_t,
        "checkpoint": str(ckpt_path),
    }

    metrics_path = CHECKPOINT_DIR / f"{domain_name}_{arch}_k{k}_dim{dim}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"  → Mean MAE={mean_mae:.2f}°C  saved: {ckpt_path.name}", flush=True)

    return result


def sweep_k(domain_name: str, arch: str, dim: int,
            k_values: list[int], device: torch.device) -> dict:
    """
    Train all k values for a given (domain, arch), then select best k.

    All checkpoints are saved. The best k (minimum mean MAE) is recorded
    in the returned summary dict and written to registry.json.
    """
    print(f"\n{'='*60}")
    print(f"  K-SWEEP  domain={domain_name}  arch={arch}  dim={dim}D")
    print(f"  k values: {k_values}")
    print(f"{'='*60}", flush=True)

    results_by_k = {}
    for k in k_values:
        res = run_one(domain_name, arch, k, dim, device)
        results_by_k[k] = res

    # Select best k
    best_k   = min(results_by_k, key=lambda k: results_by_k[k]["mean_mae"])
    best_mae = results_by_k[best_k]["mean_mae"]

    print(f"\n  K-SWEEP RESULT  domain={domain_name}  arch={arch}")
    print(f"  {'k':>4}  {'mean MAE':>10}  {'n_windows':>10}")
    for k, r in sorted(results_by_k.items()):
        marker = " ← best" if k == best_k else ""
        print(f"  {k:>4}  {r['mean_mae']:>10.2f}°C  {r['n_windows']:>10}{marker}")

    return {
        "domain":     domain_name,
        "arch":       arch,
        "dim":        dim,
        "best_k":     best_k,
        "best_mae":   best_mae,
        "all_k":      {str(k): r["mean_mae"] for k, r in results_by_k.items()},
        "checkpoint_best": results_by_k[best_k]["checkpoint"],
    }


def update_registry(entry: dict):
    """Append/update entry in checkpoints/registry.json."""
    reg_path = CHECKPOINT_DIR / "registry.json"
    registry = []
    if reg_path.exists():
        with open(reg_path) as f:
            registry = json.load(f)

    key = (entry["domain"], entry["arch"], entry["dim"])
    registry = [r for r in registry
                if not (r["domain"] == key[0] and
                        r["arch"]   == key[1] and
                        r["dim"]    == key[2])]
    registry.append(entry)

    with open(reg_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"  Registry updated: {reg_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train ThermalPINN for all domain/arch/k combinations."
    )
    parser.add_argument("--domain",   default="rectangle",
                        choices=list(DOMAINS_2D) + list(DOMAINS_3D))
    parser.add_argument("--arch",     default="nsga2",
                        choices=list(ARCH_CONFIGS))
    parser.add_argument("--k",        type=int, default=3,
                        help="Single k value (used without --sweep_k)")
    parser.add_argument("--sweep_k",  action="store_true",
                        help="Train all k values and select best")
    parser.add_argument("--k_values", nargs="+", type=int, default=K_VALUES,
                        help="k values for sweep (default: 1 2 3 4 5)")
    parser.add_argument("--dim",      type=int, default=2, choices=[2, 3])
    parser.add_argument("--all",      action="store_true",
                        help="Train all domain × arch × k combinations")
    parser.add_argument("--cuda",     action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    if args.all:
        domains_2d = list(DOMAINS_2D.keys())
        domains_3d = list(DOMAINS_3D.keys())
        archs      = list(ARCH_CONFIGS.keys())

        for dim, domains in [(2, domains_2d), (3, domains_3d)]:
            for domain_name in domains:
                for arch in archs:
                    entry = sweep_k(domain_name, arch, dim, args.k_values, device)
                    update_registry(entry)
    elif args.sweep_k:
        entry = sweep_k(args.domain, args.arch, args.dim, args.k_values, device)
        update_registry(entry)
    else:
        result = run_one(args.domain, args.arch, args.k, args.dim, device)
        update_registry({
            "domain":          result["domain"],
            "arch":            result["arch"],
            "dim":             result["dim"],
            "best_k":          result["k"],
            "best_mae":        result["mean_mae"],
            "all_k":           {str(result["k"]): result["mean_mae"]},
            "checkpoint_best": result["checkpoint"],
        })


if __name__ == "__main__":
    main()
