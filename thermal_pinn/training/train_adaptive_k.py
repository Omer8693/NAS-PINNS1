"""
thermal_pinn/training/train_adaptive_k.py
==========================================
Adaptive-k training: dynamically adjusts the FEM skip count k per window
based on the measured MAE from the previous window.

Strategy
--------
    k starts at k_init (default 3).
    After each window:
        • MAE < thresh_up   → k = min(k+1, k_max)   (model doing well → skip more)
        • MAE > thresh_down → k = max(k-1, k_min)   (model struggling → be careful)
        • otherwise         → keep current k

The thresholds are in °C (actual temperature error):
    thresh_up   = 2.0°C  (model accurate enough to take bigger steps)
    thresh_down = 5.0°C  (model error too large, reduce step size)

These can be overridden via CLI arguments.

Why adaptive k?
---------------
Fixed k treats every time window identically despite the fact that gradient
steepness varies — the quench shock at t=0 demands small k while the later
quasi-steady phase tolerates large k.  Adaptive k exploits this by taking
conservative steps early and aggressive steps late, maximising FEM savings
while maintaining accuracy.

The FEM saving is quantified as:
    saving% = 1 - (n_windows_adaptive / n_windows_k1)
where n_windows_k1 = T_total / DT_FEM = 20 (the maximum possible).

Usage
-----
    # Quick test — one domain/arch:
    python -m thermal_pinn.training.train_adaptive_k \\
        --domain lshape --arch nsga2 --dim 3 --cuda

    # Compare adaptive vs fixed-k across all 3D domains:
    python -m thermal_pinn.training.train_adaptive_k \\
        --all --dim 3 --cuda

    # Custom thresholds:
    python -m thermal_pinn.training.train_adaptive_k \\
        --domain rectangle --arch nsga2 --thresh_up 1.5 --thresh_down 4.0

Output
------
    checkpoints/<domain>_<arch>_adaptive_dim<dim>_metrics.json
      {
        "domain": ..., "arch": ..., "dim": ...,
        "mean_mae": ...,
        "fem_savings_pct": ...,    # % of FEM calls saved vs k=1
        "k_schedule": [3,3,4,4,5,5,...],
        "windows": [{t_start, t_end, k_used, mae_C, l2_rel, runtime_s}, ...]
      }
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

from thermal_pinn.physics.domains_2d import make_domain, DOMAINS_2D
from thermal_pinn.physics.domains_3d import make_domain_3d, DOMAINS_3D
from thermal_pinn.network.pinn import make_pinn, ARCH_CONFIGS
from thermal_pinn.training.trainer import train_window, evaluate_window, K_VALUES

# ──────────────────────────────────────────────────────────────────────────────
# Simulation constants (same as train_all.py)
# ──────────────────────────────────────────────────────────────────────────────
T_TOTAL     = 30.0
DT_FEM      = 1.5
N_FEM_STEPS = int(T_TOTAL / DT_FEM)   # 20
T_WATER     = 20.0
DELTA_T     = 520.0

TRAIN_KWARGS = {
    "n_domain":  1500,
    "n_bc":      300,
    "n_epochs":  800,
    "lr":        1e-3,
    "lr_min":    1e-5,
    "lbfgs_iters": 50,
    "use_end_supervision": True,
}

K_MIN_DEFAULT = 1
K_MAX_DEFAULT = 5
K_INIT_DEFAULT = 3
THRESH_UP_DEFAULT   = 2.0   # °C  — increase k if MAE below this
THRESH_DOWN_DEFAULT = 5.0   # °C  — decrease k if MAE above this

CHECKPOINT_DIR = _ROOT / "checkpoints"


def theta_from_domain(domain, coords_np: np.ndarray, t: float) -> np.ndarray:
    if coords_np.shape[1] == 2:
        T_vals = domain.T(coords_np[:, 0], coords_np[:, 1], t)
    else:
        T_vals = domain.T(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], t)
    T_vals = np.where(np.isfinite(T_vals), T_vals, T_WATER)
    return ((T_vals - T_WATER) / DELTA_T).astype(np.float32)


def run_adaptive(domain_name: str, arch: str, dim: int,
                 device: torch.device,
                 k_init: int = K_INIT_DEFAULT,
                 k_min: int  = K_MIN_DEFAULT,
                 k_max: int  = K_MAX_DEFAULT,
                 thresh_up: float   = THRESH_UP_DEFAULT,
                 thresh_down: float = THRESH_DOWN_DEFAULT,
                 verbose: bool = True) -> dict:
    """
    Adaptive-k training loop for one (domain, arch) combination.

    The window boundaries are determined dynamically: each window starts
    where the previous one ended, with width k·DT_FEM for the current k.

    Returns metrics dict suitable for JSON serialisation.
    """
    if dim == 2:
        domain = make_domain(domain_name)
    else:
        domain = make_domain_3d(domain_name)

    model = make_pinn(dim=dim, arch=arch, device=device)

    train_kwargs = dict(TRAIN_KWARGS)
    if dim == 3 and domain_name == "lshape":
        train_kwargs["n_domain"] = 3000
        train_kwargs["n_bc"]     = 500

    k          = k_init
    t_now      = 0.0
    window_idx = 0
    window_results = []
    k_schedule     = []
    t0_total       = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  ADAPTIVE-k  domain={domain_name}  arch={arch}  dim={dim}D")
        print(f"  k_init={k_init}  k∈[{k_min},{k_max}]  "
              f"thresh↑={thresh_up}°C  thresh↓={thresh_down}°C")
        print(f"{'='*60}", flush=True)

    while t_now < T_TOTAL - 1e-6:
        dt_window = k * DT_FEM
        t_end     = min(t_now + dt_window, T_TOTAL)
        actual_k  = round((t_end - t_now) / DT_FEM)   # actual FEM steps in this window
        window_idx += 1

        if verbose:
            print(f"  [{window_idx:>2}] [{t_now:.1f}→{t_end:.1f}s]  k={k}  "
                  f"{domain_name}/{arch}", flush=True)

        t_start = t_now

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
            rng_seed=window_idx,
            **train_kwargs,
        )

        eval_info = evaluate_window(
            model=model, domain=domain,
            t_start=t_start, t_end=t_end,
            theta_ic_fn=theta_ic_fn,
            device=device,
        )

        mae = eval_info["mae_C"]
        if verbose:
            print(f"      MAE={mae:.2f}°C  L2={eval_info['l2_rel']:.4f}  "
                  f"({train_info['runtime_s']:.0f}s)", flush=True)

        window_results.append({
            "window":    window_idx,
            "t_start":   float(t_start),
            "t_end":     float(t_end),
            "k_used":    k,
            "mae_C":     mae,
            "l2_rel":    eval_info["l2_rel"],
            "runtime_s": train_info["runtime_s"],
        })
        k_schedule.append(k)

        # ── Adaptive k update rule ───────────────────────────────────────────
        prev_k = k
        if mae < thresh_up and k < k_max:
            k += 1
        elif mae > thresh_down and k > k_min:
            k -= 1

        if verbose and k != prev_k:
            direction = "↑" if k > prev_k else "↓"
            print(f"      k {direction} {prev_k}→{k}  "
                  f"(MAE={mae:.2f}°C, thresh={'↑' if k>prev_k else '↓'}"
                  f"={'%.1f'%thresh_up if k>prev_k else '%.1f'%thresh_down}°C)",
                  flush=True)

        t_now = t_end

    # ── Summary ──────────────────────────────────────────────────────────────
    mean_mae      = float(np.mean([w["mae_C"] for w in window_results]))
    n_windows     = len(window_results)
    # FEM savings: how many FEM calls did we save vs fixed k=1 (20 windows)?
    # Adaptive used n_windows windows; k=1 would need N_FEM_STEPS=20.
    fem_savings_pct = (1.0 - n_windows / N_FEM_STEPS) * 100.0

    total_t = time.time() - t0_total

    if verbose:
        print(f"\n  RESULT  mean MAE={mean_mae:.2f}°C  "
              f"windows={n_windows}  FEM savings={fem_savings_pct:.1f}%")
        print(f"  k schedule: {k_schedule}", flush=True)

    result = {
        "domain":          domain_name,
        "arch":            arch,
        "dim":             dim,
        "mean_mae":        mean_mae,
        "n_windows":       n_windows,
        "fem_savings_pct": fem_savings_pct,
        "k_schedule":      k_schedule,
        "k_init":          k_init,
        "thresh_up":       thresh_up,
        "thresh_down":     thresh_down,
        "total_s":         total_t,
        "windows":         window_results,
    }

    # Save metrics
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{domain_name}_{arch}_adaptive_dim{dim}"
    metrics_path = CHECKPOINT_DIR / f"{tag}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save model checkpoint
    ckpt_path = CHECKPOINT_DIR / f"{tag}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "arch":        arch,
        "dim":         dim,
        "domain":      domain_name,
        "k_schedule":  k_schedule,
        "mean_mae":    mean_mae,
    }, ckpt_path)

    if verbose:
        print(f"  → Saved: {metrics_path.name}", flush=True)

    return result


def compare_with_fixed(domain_name: str, arch: str, dim: int,
                       adaptive_result: dict) -> None:
    """
    Load fixed-k metrics from registry and print side-by-side comparison.
    Requires that train_all.py has already been run for this combination.
    """
    reg_path = CHECKPOINT_DIR / "registry.json"
    if not reg_path.exists():
        print("  [compare] registry.json not found — skipping comparison")
        return

    with open(reg_path) as f:
        registry = json.load(f)

    match = next(
        (r for r in registry
         if r["domain"] == domain_name and r["arch"] == arch and r["dim"] == dim),
        None
    )
    if match is None:
        print(f"  [compare] No registry entry for {domain_name}/{arch}/{dim}D")
        return

    best_k   = match["best_k"]
    best_mae = match["best_mae"]
    # Fixed k=best_k uses T_TOTAL/DT_FEM/best_k windows
    n_fixed  = int(np.ceil(T_TOTAL / (best_k * DT_FEM)))
    fixed_savings = (1.0 - n_fixed / N_FEM_STEPS) * 100.0

    print(f"\n  ── Comparison: {domain_name}/{arch} dim={dim}D ──")
    print(f"  {'Method':<22} {'MAE (°C)':>10} {'Windows':>9} {'FEM saved':>11}")
    print(f"  {'-'*55}")
    print(f"  {'Fixed k='+str(best_k):<22} {best_mae:>10.2f} "
          f"{n_fixed:>9} {fixed_savings:>10.1f}%")
    print(f"  {'Adaptive k':<22} {adaptive_result['mean_mae']:>10.2f} "
          f"{adaptive_result['n_windows']:>9} "
          f"{adaptive_result['fem_savings_pct']:>10.1f}%")

    mae_delta = adaptive_result["mean_mae"] - best_mae
    sav_delta = adaptive_result["fem_savings_pct"] - fixed_savings
    sign_mae  = "+" if mae_delta >= 0 else ""
    sign_sav  = "+" if sav_delta >= 0 else ""
    print(f"  {'Δ (adaptive − fixed)':<22} {sign_mae}{mae_delta:>9.2f} "
          f"{'':>9} {sign_sav}{sav_delta:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive-k PINN training — dynamically adjusts FEM skip count."
    )
    parser.add_argument("--domain", default="lshape",
                        choices=list(DOMAINS_2D) + list(DOMAINS_3D))
    parser.add_argument("--arch",   default="nsga2",
                        choices=list(ARCH_CONFIGS))
    parser.add_argument("--dim",    type=int, default=3, choices=[2, 3])
    parser.add_argument("--all",    action="store_true",
                        help="Run adaptive-k for all domain × arch combinations")
    parser.add_argument("--k_init",     type=int,   default=K_INIT_DEFAULT)
    parser.add_argument("--k_min",      type=int,   default=K_MIN_DEFAULT)
    parser.add_argument("--k_max",      type=int,   default=K_MAX_DEFAULT)
    parser.add_argument("--thresh_up",  type=float, default=THRESH_UP_DEFAULT,
                        help="MAE threshold below which k increases (°C)")
    parser.add_argument("--thresh_down", type=float, default=THRESH_DOWN_DEFAULT,
                        help="MAE threshold above which k decreases (°C)")
    parser.add_argument("--cuda",       action="store_true")
    parser.add_argument("--no_compare", action="store_true",
                        help="Skip comparison with fixed-k registry entry")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    kwargs = dict(
        k_init=args.k_init, k_min=args.k_min, k_max=args.k_max,
        thresh_up=args.thresh_up, thresh_down=args.thresh_down,
        device=device,
    )

    if args.all:
        domains = list(DOMAINS_2D.keys()) if args.dim == 2 else list(DOMAINS_3D.keys())
        archs   = list(ARCH_CONFIGS.keys())
        for domain_name in domains:
            for arch in archs:
                result = run_adaptive(domain_name, arch, args.dim, **kwargs)
                if not args.no_compare:
                    compare_with_fixed(domain_name, arch, args.dim, result)
    else:
        result = run_adaptive(args.domain, args.arch, args.dim, **kwargs)
        if not args.no_compare:
            compare_with_fixed(args.domain, args.arch, args.dim, result)


if __name__ == "__main__":
    main()
