"""
thermal_pinn/training/train_adaptive_k_v2.py
============================================
Accuracy-first adaptive-k training.

What changes vs v1
------------------
The original adaptive controller reacts only to the previous window MAE and can
start too aggressively in the early quench stage. This v2 controller is more
conservative:

    - force small k during the first few windows (burn-in)
    - require multiple consecutive good windows before increasing k
    - reduce k immediately when the model struggles
    - cap the maximum allowed k by time:
          early   -> small windows
          middle  -> moderate windows
          late    -> larger windows allowed

The goal is not maximum FEM savings at any cost, but a better
accuracy-versus-savings trade-off, especially for 3D cases.

Usage
-----
    # Recommended first run on GPU:
    python -m thermal_pinn.training.train_adaptive_k_v2 \
        --domain cylinder --arch bayesian --dim 3 --cuda

    # Quick smoke test:
    python -m thermal_pinn.training.train_adaptive_k_v2 \
        --domain cylinder --arch bayesian --dim 3 --device cpu \
        --n-epochs 1 --n-domain 16 --n-bc 8 --n-eval 16 \
        --lbfgs-iters 1 --max-windows 3 --no_compare
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

from thermal_pinn.network.pinn import ARCH_CONFIGS, make_pinn
from thermal_pinn.physics.domains_2d import DOMAINS_2D, make_domain
from thermal_pinn.physics.domains_3d import DOMAINS_3D, make_domain_3d
from thermal_pinn.training.trainer import evaluate_window, train_window

T_TOTAL = 30.0
DT_FEM = 1.5
N_FEM_STEPS = int(T_TOTAL / DT_FEM)
T_WATER = 20.0
DELTA_T = 520.0

TRAIN_KWARGS = {
    "n_domain": 1500,
    "n_bc": 300,
    "n_epochs": 800,
    "lr": 1e-3,
    "lr_min": 1e-5,
    "lbfgs_iters": 50,
    "use_end_supervision": True,
}

K_MIN_DEFAULT = 1
K_MAX_DEFAULT = 5
K_INIT_DEFAULT = 1
THRESH_UP_DEFAULT = 7.0
THRESH_DOWN_DEFAULT = 10.0
BURNIN_WINDOWS_DEFAULT = 3
BURNIN_K_DEFAULT = 1
PROMOTE_STREAK_DEFAULT = 2
EARLY_UNTIL_DEFAULT = 10.0
MID_UNTIL_DEFAULT = 18.0
EARLY_K_MAX_DEFAULT = 2
MID_K_MAX_DEFAULT = 3
LATE_K_MAX_DEFAULT = 5
TAG_DEFAULT = "adaptive_v2"

CHECKPOINT_DIR = _ROOT / "checkpoints"


def theta_from_domain(domain, coords_np: np.ndarray, t: float) -> np.ndarray:
    """Return dimensionless anchor temperature theta = (T - T_w) / DeltaT."""
    if coords_np.shape[1] == 2:
        T_vals = domain.T(coords_np[:, 0], coords_np[:, 1], t)
    else:
        T_vals = domain.T(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], t)
    T_vals = np.where(np.isfinite(T_vals), T_vals, T_WATER)
    return ((T_vals - T_WATER) / DELTA_T).astype(np.float32)


def resolve_device(args: argparse.Namespace) -> torch.device:
    """Choose CPU/GPU device with an explicit CUDA failure if requested."""
    if args.device is not None:
        return torch.device(args.device)
    if args.cuda:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but no GPU is visible to PyTorch.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def time_phase_k_max(t_now: float, args: argparse.Namespace) -> int:
    """Return the maximum allowed k for the current physical time."""
    if t_now < args.early_until:
        return args.early_k_max
    if t_now < args.mid_until:
        return args.mid_k_max
    return args.late_k_max


def build_train_kwargs(args: argparse.Namespace, dim: int, domain_name: str) -> dict:
    """Merge defaults with CLI overrides and legacy lshape3d override."""
    kw = dict(TRAIN_KWARGS)
    if dim == 3 and domain_name == "lshape":
        kw["n_domain"] = 3000
        kw["n_bc"] = 500

    for key in ("n_domain", "n_bc", "n_epochs", "lr", "lr_min", "lbfgs_iters"):
        value = getattr(args, key)
        if value is not None:
            kw[key] = value
    return kw


def compare_with_fixed(domain_name: str, arch: str, dim: int, adaptive_result: dict) -> None:
    """Load fixed-k metrics from registry and print side-by-side comparison."""
    reg_path = CHECKPOINT_DIR / "registry.json"
    if not reg_path.exists():
        print("  [compare] registry.json not found - skipping comparison")
        return

    with open(reg_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    match = next(
        (
            row for row in registry
            if row["domain"] == domain_name and row["arch"] == arch and row["dim"] == dim
        ),
        None,
    )
    if match is None:
        print(f"  [compare] No registry entry for {domain_name}/{arch}/{dim}D")
        return

    best_k = match["best_k"]
    best_mae = match["best_mae"]
    n_fixed = int(np.ceil(T_TOTAL / (best_k * DT_FEM)))
    fixed_savings = (1.0 - n_fixed / N_FEM_STEPS) * 100.0

    print(f"\n  -- Comparison: {domain_name}/{arch} dim={dim}D --")
    print(f"  {'Method':<22} {'MAE (C)':>10} {'Windows':>9} {'FEM saved':>11}")
    print(f"  {'-' * 55}")
    print(f"  {'Fixed k=' + str(best_k):<22} {best_mae:>10.2f} {n_fixed:>9} {fixed_savings:>10.1f}%")
    print(
        f"  {'Adaptive k v2':<22} {adaptive_result['mean_mae']:>10.2f} "
        f"{adaptive_result['n_windows']:>9} {adaptive_result['fem_savings_pct']:>10.1f}%"
    )

    mae_delta = adaptive_result["mean_mae"] - best_mae
    sav_delta = adaptive_result["fem_savings_pct"] - fixed_savings
    sign_mae = "+" if mae_delta >= 0 else ""
    sign_sav = "+" if sav_delta >= 0 else ""
    print(f"  {'Delta (adaptive-fixed)':<22} {sign_mae}{mae_delta:>9.2f} {'':>9} {sign_sav}{sav_delta:>9.1f}%")


def run_adaptive_v2(
    domain_name: str,
    arch: str,
    dim: int,
    *,
    device: torch.device,
    args: argparse.Namespace,
    verbose: bool = True,
) -> dict:
    """Adaptive-k v2 training loop for one domain/arch combination."""
    domain = make_domain(domain_name) if dim == 2 else make_domain_3d(domain_name)
    model = make_pinn(dim=dim, arch=arch, device=device)
    train_kwargs = build_train_kwargs(args, dim, domain_name)

    k = args.k_init
    t_now = 0.0
    window_idx = 0
    good_streak = 0
    window_results = []
    k_schedule = []
    t0_total = time.time()

    if verbose:
        print(f"\n{'=' * 72}")
        print(f"  ADAPTIVE-k v2  domain={domain_name}  arch={arch}  dim={dim}D")
        print(
            f"  burnin={args.burnin_windows} windows @ k={args.burnin_k}  "
            f"k_init={args.k_init}  k in [{args.k_min},{args.k_max}]"
        )
        print(
            f"  thresh_up={args.thresh_up}C  thresh_down={args.thresh_down}C  "
            f"promote_streak={args.promote_streak}"
        )
        print(
            f"  phase caps: t<{args.early_until}s -> k<={args.early_k_max}, "
            f"t<{args.mid_until}s -> k<={args.mid_k_max}, late -> k<={args.late_k_max}"
        )
        print(f"{'=' * 72}", flush=True)

    while t_now < T_TOTAL - 1e-6:
        if args.max_windows is not None and window_idx >= args.max_windows:
            break

        current_k_cap = min(args.k_max, time_phase_k_max(t_now, args))
        if window_idx < args.burnin_windows:
            k = min(max(args.burnin_k, args.k_min), current_k_cap)
            controller_mode = "burn-in"
        else:
            k = min(max(k, args.k_min), current_k_cap)
            controller_mode = "adaptive"

        dt_window = k * DT_FEM
        t_end = min(t_now + dt_window, T_TOTAL)
        actual_k = round((t_end - t_now) / DT_FEM)
        window_idx += 1

        if verbose:
            print(
                f"  [{window_idx:>2}] [{t_now:.1f}->{t_end:.1f}s]  "
                f"k={k} (cap={current_k_cap}, mode={controller_mode})  {domain_name}/{arch}",
                flush=True,
            )

        t_start = t_now

        def theta_ic_fn(coords_np, _t=t_start):
            return theta_from_domain(domain, coords_np, _t)

        def theta_end_fn(coords_np, _t=t_end):
            return theta_from_domain(domain, coords_np, _t)

        train_info = train_window(
            model=model,
            domain=domain,
            t_start=t_start,
            t_end=t_end,
            theta_ic_fn=theta_ic_fn,
            theta_end_fn=theta_end_fn,
            device=device,
            rng_seed=window_idx,
            **train_kwargs,
        )

        eval_info = evaluate_window(
            model=model,
            domain=domain,
            t_start=t_start,
            t_end=t_end,
            theta_ic_fn=theta_ic_fn,
            n_eval=args.n_eval,
            device=device,
        )

        mae = float(eval_info["mae_C"])
        if verbose:
            print(
                f"      MAE={mae:.2f}C  L2={eval_info['l2_rel']:.4f}  "
                f"({train_info['runtime_s']:.0f}s)",
                flush=True,
            )

        window_results.append(
            {
                "window": window_idx,
                "t_start": float(t_start),
                "t_end": float(t_end),
                "k_used": actual_k,
                "k_cap": current_k_cap,
                "mae_C": mae,
                "l2_rel": eval_info["l2_rel"],
                "runtime_s": train_info["runtime_s"],
                "controller_mode": controller_mode,
            }
        )
        k_schedule.append(actual_k)

        prev_k = actual_k
        if window_idx <= args.burnin_windows:
            next_k = max(args.k_min, min(args.burnin_k, args.k_max))
            good_streak = 0
        else:
            if mae <= args.thresh_up:
                good_streak += 1
            else:
                good_streak = 0

            if mae >= args.thresh_down:
                next_k = max(args.k_min, prev_k - 1)
                good_streak = 0
            elif good_streak >= args.promote_streak:
                next_cap = min(args.k_max, time_phase_k_max(t_end, args))
                next_k = min(next_cap, prev_k + 1)
                if next_k > prev_k:
                    good_streak = 0
            else:
                next_k = prev_k

        if verbose and next_k != prev_k:
            direction = "up" if next_k > prev_k else "down"
            print(
                f"      k {direction} {prev_k}->{next_k}  "
                f"(MAE={mae:.2f}C, streak={good_streak})",
                flush=True,
            )

        k = next_k
        t_now = t_end

    mean_mae = float(np.mean([row["mae_C"] for row in window_results])) if window_results else float("nan")
    n_windows = len(window_results)
    fem_savings_pct = (1.0 - n_windows / N_FEM_STEPS) * 100.0
    total_s = time.time() - t0_total

    if verbose:
        print(f"\n  RESULT  mean MAE={mean_mae:.2f}C  windows={n_windows}  FEM savings={fem_savings_pct:.1f}%")
        print(f"  k schedule: {k_schedule}", flush=True)

    result = {
        "domain": domain_name,
        "arch": arch,
        "dim": dim,
        "tag": args.tag,
        "mean_mae": mean_mae,
        "n_windows": n_windows,
        "fem_savings_pct": fem_savings_pct,
        "k_schedule": k_schedule,
        "k_init": args.k_init,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "thresh_up": args.thresh_up,
        "thresh_down": args.thresh_down,
        "burnin_windows": args.burnin_windows,
        "burnin_k": args.burnin_k,
        "promote_streak": args.promote_streak,
        "early_until": args.early_until,
        "mid_until": args.mid_until,
        "early_k_max": args.early_k_max,
        "mid_k_max": args.mid_k_max,
        "late_k_max": args.late_k_max,
        "total_s": total_s,
        "n_eval": args.n_eval,
        "train_kwargs": train_kwargs,
        "windows": window_results,
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{domain_name}_{arch}_{args.tag}_dim{dim}"
    metrics_path = CHECKPOINT_DIR / f"{tag}_metrics.json"
    ckpt_path = CHECKPOINT_DIR / f"{tag}.pt"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    torch.save(
        {
            "model_state": model.state_dict(),
            "arch": arch,
            "dim": dim,
            "domain": domain_name,
            "k_schedule": k_schedule,
            "mean_mae": mean_mae,
        },
        ckpt_path,
    )

    if verbose:
        print(f"  -> Saved: {metrics_path.name}", flush=True)

    return result


def build_parser() -> argparse.ArgumentParser:
    """CLI parser for adaptive-k v2."""
    domain_choices = list(dict.fromkeys(list(DOMAINS_2D) + list(DOMAINS_3D)))
    parser = argparse.ArgumentParser(
        description="Adaptive-k PINN training v2 - conservative controller."
    )
    parser.add_argument("--domain", default="cylinder", choices=domain_choices)
    parser.add_argument("--arch", default="bayesian", choices=list(ARCH_CONFIGS))
    parser.add_argument("--dim", type=int, default=3, choices=[2, 3])
    parser.add_argument("--all", action="store_true", help="Run all domain x arch combinations.")

    parser.add_argument("--k_init", type=int, default=K_INIT_DEFAULT)
    parser.add_argument("--k_min", type=int, default=K_MIN_DEFAULT)
    parser.add_argument("--k_max", type=int, default=K_MAX_DEFAULT)
    parser.add_argument("--thresh_up", type=float, default=THRESH_UP_DEFAULT)
    parser.add_argument("--thresh_down", type=float, default=THRESH_DOWN_DEFAULT)
    parser.add_argument("--burnin_windows", type=int, default=BURNIN_WINDOWS_DEFAULT)
    parser.add_argument("--burnin_k", type=int, default=BURNIN_K_DEFAULT)
    parser.add_argument("--promote_streak", type=int, default=PROMOTE_STREAK_DEFAULT)
    parser.add_argument("--early_until", type=float, default=EARLY_UNTIL_DEFAULT)
    parser.add_argument("--mid_until", type=float, default=MID_UNTIL_DEFAULT)
    parser.add_argument("--early_k_max", type=int, default=EARLY_K_MAX_DEFAULT)
    parser.add_argument("--mid_k_max", type=int, default=MID_K_MAX_DEFAULT)
    parser.add_argument("--late_k_max", type=int, default=LATE_K_MAX_DEFAULT)

    parser.add_argument("--n-domain", dest="n_domain", type=int, default=None)
    parser.add_argument("--n-bc", dest="n_bc", type=int, default=None)
    parser.add_argument("--n-epochs", dest="n_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-min", dest="lr_min", type=float, default=None)
    parser.add_argument("--lbfgs-iters", dest="lbfgs_iters", type=int, default=None)
    parser.add_argument("--n-eval", dest="n_eval", type=int, default=500)
    parser.add_argument("--max-windows", type=int, default=None)

    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--device", default=None, help="Explicit torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--tag", default=TAG_DEFAULT, help="Output tag for checkpoint and metrics filenames.")
    parser.add_argument("--no_compare", action="store_true", help="Skip comparison with fixed-k registry entry.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = resolve_device(args)
    verbose = not args.quiet

    print(f"Device: {device}", flush=True)

    if args.all:
        domains = list(DOMAINS_2D.keys()) if args.dim == 2 else list(DOMAINS_3D.keys())
        archs = list(ARCH_CONFIGS.keys())
        for domain_name in domains:
            for arch in archs:
                result = run_adaptive_v2(domain_name, arch, args.dim, device=device, args=args, verbose=verbose)
                if not args.no_compare:
                    compare_with_fixed(domain_name, arch, args.dim, result)
        return

    result = run_adaptive_v2(args.domain, args.arch, args.dim, device=device, args=args, verbose=verbose)
    if not args.no_compare:
        compare_with_fixed(args.domain, args.arch, args.dim, result)


if __name__ == "__main__":
    main()
