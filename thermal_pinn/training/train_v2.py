"""
thermal_pinn/training/train_v2.py
===================================
v2 training entry point — FourierPINN + Self-Adaptive loss weights.

Identical structure to train_all.py but uses:
    - FourierPINN  (network/fourier_pinn.py)  instead of ThermalPINN
    - train_window_sa (training/sa_trainer.py) instead of train_window

Output files:
    checkpoints/<domain>_<arch>_k<k>_dim<d>_v2.pt
    checkpoints/<domain>_<arch>_k<k>_dim<d>_v2_metrics.json
    checkpoints/registry_v2.json

These files are completely separate from v1 (train_all.py) outputs, so
existing checkpoints / results / plots are untouched.

Usage
-----
    # Single combination:
    python -m thermal_pinn.training.train_v2 --domain rectangle --arch nsga2 --k 3

    # Sweep all k for one domain/arch:
    python -m thermal_pinn.training.train_v2 --domain cylinder --arch nsga3 --sweep_k

    # Full run — all domains × archs × k (2D + 3D):
    python -m thermal_pinn.training.train_v2 --all --cuda

    # Only 3D (where v2 improvement is expected largest):
    python -m thermal_pinn.training.train_v2 --all --dim 3 --cuda

    # Quick smoke-test (1 epoch, skip L-BFGS):
    python -m thermal_pinn.training.train_v2 --domain rectangle --arch nsga3 \\
           --k 1 --n_epochs 5 --lbfgs_iters 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent))

from thermal_pinn.physics.domains_2d  import make_domain,    DOMAINS_2D
from thermal_pinn.physics.domains_3d  import make_domain_3d, DOMAINS_3D
from thermal_pinn.network.pinn        import ARCH_CONFIGS
from thermal_pinn.network.fourier_pinn import make_fourier_pinn
from thermal_pinn.training.sa_trainer  import (
    SelfAdaptiveWeights, train_window_sa, evaluate_window_sa
)
from thermal_pinn.training.trainer    import K_VALUES

# ──────────────────────────────────────────────────────────────────────────────
# Simulation constants (same as train_all.py)
# ──────────────────────────────────────────────────────────────────────────────
T_TOTAL     = 30.0
DT_FEM      = 1.5
N_FEM_STEPS = int(T_TOTAL / DT_FEM)    # 20
T_WATER     = 20.0
DELTA_T     = 520.0

# Default training hyperparameters (same counts as v1 for fair comparison)
TRAIN_KWARGS_DEFAULT = {
    "n_domain":           1500,
    "n_bc":               300,
    "n_epochs":           800,
    "lr":                 1e-3,
    "lr_lambda":          1e-4,    # SA weight learning rate (10× slower)
    "lr_min":             1e-5,
    "lbfgs_iters":        50,
    "use_end_supervision": True,
}

CHECKPOINT_DIR = _ROOT / "checkpoints"


def theta_from_domain(domain, coords_np: np.ndarray, t: float) -> np.ndarray:
    """Dimensionless IC temperature θ = (T − T_w) / ΔT at time t."""
    if coords_np.shape[1] == 2:
        T_vals = domain.T(coords_np[:, 0], coords_np[:, 1], t)
    else:
        T_vals = domain.T(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], t)
    T_vals = np.where(np.isfinite(T_vals), T_vals, T_WATER)
    return ((T_vals - T_WATER) / DELTA_T).astype(np.float32)


def run_one_v2(domain_name: str, arch: str, k: int, dim: int,
               device: torch.device,
               train_kwargs: dict | None = None,
               verbose: bool = True) -> dict:
    """
    Train a single (domain, arch, k) combination with FourierPINN + SA weights.

    Saves:
        {domain}_{arch}_k{k}_dim{dim}_v2.pt
        {domain}_{arch}_k{k}_dim{dim}_v2_metrics.json

    Returns the metrics dict.
    """
    kwargs = dict(TRAIN_KWARGS_DEFAULT)
    if train_kwargs:
        kwargs.update(train_kwargs)

    if dim == 2:
        domain = make_domain(domain_name)
    else:
        domain = make_domain_3d(domain_name)

    model      = make_fourier_pinn(dim=dim, arch=arch, device=device)
    sa_weights = SelfAdaptiveWeights().to(device)

    if verbose:
        print(f"\n  [v2] {domain_name}/{arch}/k={k}/dim={dim}D  "
              f"model={model!r}", flush=True)

    dt_window = k * DT_FEM
    anchors   = np.arange(0.0, T_TOTAL - dt_window / 2, dt_window)

    window_results = []
    t0_total = time.time()

    for i, t_start in enumerate(anchors):
        t_end = min(t_start + dt_window, T_TOTAL)
        if verbose:
            print(f"    [{i+1}/{len(anchors)}] [{t_start:.1f}→{t_end:.1f}s]",
                  end="  ", flush=True)

        def theta_ic_fn(coords_np, _t=t_start):
            return theta_from_domain(domain, coords_np, _t)

        def theta_end_fn(coords_np, _t=t_end):
            return theta_from_domain(domain, coords_np, _t)

        train_info = train_window_sa(
            model=model, sa_weights=sa_weights,
            domain=domain,
            t_start=t_start, t_end=t_end,
            theta_ic_fn=theta_ic_fn,
            theta_end_fn=theta_end_fn,
            device=device,
            rng_seed=i,
            **kwargs,
        )

        eval_info = evaluate_window_sa(
            model=model, domain=domain,
            t_start=t_start, t_end=t_end,
            theta_ic_fn=theta_ic_fn,
            device=device,
        )

        lam = train_info["final_weights"]
        if verbose:
            print(f"MAE={eval_info['mae_C']:.2f}°C  L2={eval_info['l2_rel']:.4f}  "
                  f"({train_info['runtime_s']:.0f}s)  "
                  f"λ=[{lam['lam_pde']:.2f},{lam['lam_bc']:.2f},{lam['lam_end']:.2f}]",
                  flush=True)

        window_results.append({
            "t_start":      float(t_start),
            "t_end":        float(t_end),
            "mae_C":        eval_info["mae_C"],
            "l2_rel":       eval_info["l2_rel"],
            "runtime_s":    train_info["runtime_s"],
            "final_weights": lam,
        })

    mean_mae = float(np.mean([w["mae_C"]  for w in window_results]))
    mean_l2  = float(np.mean([w["l2_rel"] for w in window_results]))
    total_t  = time.time() - t0_total

    # ── Save checkpoint ────────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{domain_name}_{arch}_k{k}_dim{dim}_v2.pt"
    torch.save({
        "model_state":      model.state_dict(),
        "sa_weights_state": sa_weights.state_dict(),
        "arch":             arch,
        "dim":              dim,
        "domain":           domain_name,
        "k":                k,
        "mean_mae":         mean_mae,
        "mean_l2":          mean_l2,
        "windows":          window_results,
        "version":          "v2_fourier_sa",
    }, ckpt_path)

    result = {
        "domain":     domain_name,
        "arch":       arch,
        "k":          k,
        "dim":        dim,
        "mean_mae":   mean_mae,
        "mean_l2":    mean_l2,
        "n_windows":  len(window_results),
        "windows":    window_results,
        "total_s":    total_t,
        "checkpoint": str(ckpt_path),
        "version":    "v2_fourier_sa",
    }

    metrics_path = CHECKPOINT_DIR / f"{domain_name}_{arch}_k{k}_dim{dim}_v2_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"  → Mean MAE={mean_mae:.2f}°C  Mean L2={mean_l2:.4f}  "
              f"saved: {ckpt_path.name}", flush=True)

    return result


def sweep_k_v2(domain_name: str, arch: str, dim: int,
               k_values: list[int], device: torch.device,
               train_kwargs: dict | None = None) -> dict:
    """Train all k values and return sweep summary."""
    print(f"\n{'='*65}")
    print(f"  [v2] K-SWEEP  domain={domain_name}  arch={arch}  dim={dim}D")
    print(f"  k values: {k_values}")
    print(f"{'='*65}", flush=True)

    results_by_k = {}
    for k in k_values:
        res = run_one_v2(domain_name, arch, k, dim, device, train_kwargs)
        results_by_k[k] = res

    best_k   = min(results_by_k, key=lambda k: results_by_k[k]["mean_mae"])
    best_mae = results_by_k[best_k]["mean_mae"]

    print(f"\n  K-SWEEP RESULT  domain={domain_name}  arch={arch}")
    print(f"  {'k':>4}  {'mean MAE':>10}  {'mean L2':>9}")
    for k, r in sorted(results_by_k.items()):
        marker = " ← best" if k == best_k else ""
        print(f"  {k:>4}  {r['mean_mae']:>10.2f}°C  {r['mean_l2']:>9.4f}{marker}")

    return {
        "domain":     domain_name,
        "arch":       arch,
        "dim":        dim,
        "best_k":     best_k,
        "best_mae":   best_mae,
        "best_l2":    results_by_k[best_k]["mean_l2"],
        "all_k":      {str(k): {"mae": r["mean_mae"], "l2": r["mean_l2"]}
                       for k, r in results_by_k.items()},
        "checkpoint_best": results_by_k[best_k]["checkpoint"],
        "version":    "v2_fourier_sa",
    }


def update_registry_v2(entry: dict):
    """Append/update entry in checkpoints/registry_v2.json."""
    reg_path = CHECKPOINT_DIR / "registry_v2.json"
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
    print(f"  Registry updated: {reg_path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train FourierPINN + SA weights (v2) for all domain/arch/k."
    )
    parser.add_argument("--domain",   default="rectangle",
                        choices=list(DOMAINS_2D) + list(DOMAINS_3D))
    parser.add_argument("--arch",     default="nsga2",
                        choices=list(ARCH_CONFIGS))
    parser.add_argument("--k",        type=int, default=3)
    parser.add_argument("--dim",      type=int, default=2, choices=[2, 3])
    parser.add_argument("--sweep_k",  action="store_true")
    parser.add_argument("--k_values", nargs="+", type=int, default=K_VALUES)
    parser.add_argument("--all",      action="store_true",
                        help="Train all domain × arch × k (2D + 3D unless --dim given)")
    parser.add_argument("--cuda",     action="store_true")
    # Training hyperparameter overrides
    parser.add_argument("--n_epochs",    type=int, default=None)
    parser.add_argument("--lbfgs_iters", type=int, default=None)
    parser.add_argument("--n_domain",    type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Build optional training kwarg overrides
    train_kwargs = {}
    if args.n_epochs    is not None: train_kwargs["n_epochs"]    = args.n_epochs
    if args.lbfgs_iters is not None: train_kwargs["lbfgs_iters"] = args.lbfgs_iters
    if args.n_domain    is not None: train_kwargs["n_domain"]    = args.n_domain

    if args.all:
        dims_to_run = [2, 3] if args.dim not in [2, 3] else [args.dim]
        # Override: --dim given with --all limits to that dimension
        if "--dim" in sys.argv:
            dims_to_run = [args.dim]

        for dim in dims_to_run:
            domains = list(DOMAINS_2D.keys()) if dim == 2 else list(DOMAINS_3D.keys())
            for domain_name in domains:
                for arch in list(ARCH_CONFIGS.keys()):
                    entry = sweep_k_v2(domain_name, arch, dim,
                                       args.k_values, device, train_kwargs)
                    update_registry_v2(entry)

    elif args.sweep_k:
        entry = sweep_k_v2(args.domain, args.arch, args.dim,
                           args.k_values, device, train_kwargs)
        update_registry_v2(entry)

    else:
        result = run_one_v2(args.domain, args.arch, args.k, args.dim,
                            device, train_kwargs)
        update_registry_v2({
            "domain":    result["domain"],
            "arch":      result["arch"],
            "dim":       result["dim"],
            "best_k":    result["k"],
            "best_mae":  result["mean_mae"],
            "best_l2":   result["mean_l2"],
            "all_k":     {str(result["k"]): {"mae": result["mean_mae"],
                                              "l2":  result["mean_l2"]}},
            "checkpoint_best": result["checkpoint"],
            "version":   "v2_fourier_sa",
        })


if __name__ == "__main__":
    main()
