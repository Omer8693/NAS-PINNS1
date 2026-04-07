"""
thermal_pinn/training/train_3d_improved_warmstart.py
====================================================
Warm-start variant of the corrected 3D training path.

Strategy
--------
Window 1  : cold-start with the full improved3d budget
Window 2+ : warm-start from the previous window weights with
            fewer epochs, lower LR, and optionally fewer L-BFGS steps

This keeps the corrected 3D scaling from `train_3d_improved.py` while adding
the practical runtime/continuation idea from the legacy warm-start flow.

Usage
-----
    # Recommended first run on GPU:
    python -m thermal_pinn.training.train_3d_improved_warmstart \
        --domain cylinder --arch bayesian --k 1 --cuda

    # Sweep only k=1 and k=2:
    python -m thermal_pinn.training.train_3d_improved_warmstart \
        --domain cylinder --arch bayesian --sweep_k --k_values 1 2 --cuda

    # Tiny smoke test:
    python -m thermal_pinn.training.train_3d_improved_warmstart \
        --domain rectangular --arch bayesian --k 1 --device cpu \
        --cold-epochs 1 --warm-epochs 1 \
        --n-domain 16 --n-bc 8 --n-eval 16 \
        --lbfgs-iters 1 --lbfgs-iters-warm 1 \
        --max-windows 2
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
from thermal_pinn.physics.domains_3d import DOMAINS_3D, make_domain_3d
from thermal_pinn.training.train_3d_improved import (
    CHECKPOINT_DIR,
    DEFAULT_K_VALUES,
    DEFAULT_TRAIN_CONFIG,
    DOMAIN_CONFIG_OVERRIDES,
    DT_FEM,
    T_TOTAL,
    evaluate_window_improved,
    resolve_device,
    theta_from_domain,
    train_window_improved,
    update_registry,
)
from thermal_pinn.training.trainer import W_BC, W_END, W_IC, W_PDE


def build_warm_config(args: argparse.Namespace, domain_name: str) -> dict:
    """Merge 3D defaults, per-domain overrides, and warm-start CLI overrides."""
    config = dict(DEFAULT_TRAIN_CONFIG)
    config.update(DOMAIN_CONFIG_OVERRIDES.get(domain_name, {}))

    for key in ("n_domain", "n_bc", "lr_min", "n_eval", "lbfgs_iters"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    cold_epochs = args.cold_epochs if args.cold_epochs is not None else config["n_epochs"]
    lr_cold = args.lr_cold if args.lr_cold is not None else config["lr"]

    warm_epochs_default = max(500, int(round(cold_epochs * 0.4)))
    warm_lr_default = min(3e-4, lr_cold * 0.5)
    lbfgs_warm_default = max(25, int(config["lbfgs_iters"] // 2))

    config["n_epochs_cold"] = cold_epochs
    config["n_epochs_warm"] = args.warm_epochs if args.warm_epochs is not None else warm_epochs_default
    config["lr_cold"] = lr_cold
    config["lr_warm"] = args.lr_warm if args.lr_warm is not None else warm_lr_default
    config["lbfgs_iters_warm"] = (
        args.lbfgs_iters_warm if args.lbfgs_iters_warm is not None else lbfgs_warm_default
    )
    config["use_end_supervision"] = not args.no_end_supervision
    return config


def run_one_ws(
    domain_name: str,
    arch: str,
    k: int,
    *,
    device: torch.device,
    run_tag: str,
    max_windows: int | None,
    verbose: bool,
    args: argparse.Namespace,
) -> dict:
    """Train a single 3D domain/architecture/k configuration with warm-start."""
    domain = make_domain_3d(domain_name)
    model = make_pinn(dim=3, arch=arch, device=device)
    config = build_warm_config(args, domain_name)

    dt_window = k * DT_FEM
    anchors = np.arange(0.0, T_TOTAL - dt_window / 2, dt_window)
    if max_windows is not None:
        anchors = anchors[:max_windows]

    if verbose:
        print(f"\n{'=' * 78}")
        print(f"3D IMPROVED WARMSTART  domain={domain_name}  arch={arch}  k={k}")
        print(f"device={device}  windows={len(anchors)}  tag={run_tag}")
        print(
            f"n_domain={config['n_domain']}  n_bc={config['n_bc']}  "
            f"cold_epochs={config['n_epochs_cold']}  warm_epochs={config['n_epochs_warm']}  "
            f"n_eval={config['n_eval']}"
        )
        print(
            f"lr_cold={config['lr_cold']:.1e}  lr_warm={config['lr_warm']:.1e}  "
            f"lbfgs_cold={config['lbfgs_iters']}  lbfgs_warm={config['lbfgs_iters_warm']}"
        )
        print(f"{'=' * 78}", flush=True)

    window_results = []
    t0_total = time.time()

    for i, t_start in enumerate(anchors):
        t_end = min(t_start + dt_window, T_TOTAL)
        warm = i > 0
        n_epochs = config["n_epochs_warm"] if warm else config["n_epochs_cold"]
        lr = config["lr_warm"] if warm else config["lr_cold"]
        lbfgs_iters = config["lbfgs_iters_warm"] if warm else config["lbfgs_iters"]

        if verbose:
            mode = (
                f"warm({n_epochs}ep,lr={lr:.0e},lbfgs={lbfgs_iters})"
                if warm else
                f"cold({n_epochs}ep,lr={lr:.0e},lbfgs={lbfgs_iters})"
            )
            print(
                f"  [{i + 1}/{len(anchors)}] Window [{t_start:.1f} -> {t_end:.1f}s]  "
                f"{mode}  {domain_name}/{arch}/k{k}",
                flush=True,
            )

        def theta_ic_fn(coords_np: np.ndarray, _t=t_start) -> np.ndarray:
            return theta_from_domain(domain, coords_np, _t)

        def theta_end_fn(coords_np: np.ndarray, _t=t_end) -> np.ndarray:
            return theta_from_domain(domain, coords_np, _t)

        train_info = train_window_improved(
            model=model,
            domain=domain,
            t_start=t_start,
            t_end=t_end,
            theta_ic_fn=theta_ic_fn,
            theta_end_fn=theta_end_fn,
            n_domain=config["n_domain"],
            n_bc=config["n_bc"],
            n_epochs=n_epochs,
            lr=lr,
            lr_min=config["lr_min"],
            lbfgs_iters=lbfgs_iters,
            use_end_supervision=config["use_end_supervision"],
            n_eval=config["n_eval"],
            device=device,
            rng_seed=args.seed_base + i,
            w_pde=args.w_pde,
            w_ic=args.w_ic,
            w_bc=args.w_bc,
            w_end=args.w_end,
        )

        eval_info = evaluate_window_improved(
            model=model,
            domain=domain,
            t_end=t_end,
            theta_ic_fn=theta_ic_fn,
            n_eval=config["n_eval"],
            device=device,
        )

        if verbose:
            print(
                f"      MAE={eval_info['mae_C']:.2f}C  "
                f"L2={eval_info['l2_rel']:.4f}  "
                f"train={train_info['runtime_s']:.0f}s",
                flush=True,
            )

        window_results.append(
            {
                "t_start": float(t_start),
                "t_end": float(t_end),
                "mae_C": eval_info["mae_C"],
                "l2_rel": eval_info["l2_rel"],
                "runtime_s": train_info["runtime_s"],
                "warm": warm,
                "n_epochs": n_epochs,
                "lr": lr,
                "lbfgs_iters": lbfgs_iters,
            }
        )

    mean_mae = float(np.mean([w["mae_C"] for w in window_results]))
    total_s = time.time() - t0_total

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{domain_name}_{arch}_k{k}_dim3_{run_tag}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "arch": arch,
            "dim": 3,
            "domain": domain_name,
            "k": k,
            "mean_mae": mean_mae,
            "windows": window_results,
            "train_config": config,
            "warm_start": True,
            "run_tag": run_tag,
        },
        ckpt_path,
    )

    result = {
        "domain": domain_name,
        "arch": arch,
        "k": k,
        "dim": 3,
        "mean_mae": mean_mae,
        "n_windows": len(window_results),
        "windows": window_results,
        "total_s": total_s,
        "checkpoint": str(ckpt_path),
        "train_config": config,
        "warm_start": True,
        "weights": {
            "w_pde": args.w_pde,
            "w_ic": args.w_ic,
            "w_bc": args.w_bc,
            "w_end": args.w_end,
        },
        "run_tag": run_tag,
    }

    metrics_path = CHECKPOINT_DIR / f"{domain_name}_{arch}_k{k}_dim3_{run_tag}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"  -> Mean MAE={mean_mae:.2f}C  saved: {ckpt_path.name}", flush=True)

    return result


def sweep_k_ws(
    domain_name: str,
    arch: str,
    *,
    k_values: list[int],
    device: torch.device,
    run_tag: str,
    max_windows: int | None,
    verbose: bool,
    args: argparse.Namespace,
) -> dict:
    """Run a warm-start k sweep and keep the best 3D configuration."""
    if verbose:
        print(f"\n{'=' * 78}")
        print(f"3D IMPROVED WARMSTART K-SWEEP  domain={domain_name}  arch={arch}")
        print(f"k values: {k_values}")
        print(f"{'=' * 78}", flush=True)

    results_by_k = {}
    for k in k_values:
        results_by_k[k] = run_one_ws(
            domain_name,
            arch,
            k,
            device=device,
            run_tag=run_tag,
            max_windows=max_windows,
            verbose=verbose,
            args=args,
        )

    best_k = min(results_by_k, key=lambda kk: results_by_k[kk]["mean_mae"])
    best_mae = results_by_k[best_k]["mean_mae"]

    if verbose:
        print(f"\n3D IMPROVED WARMSTART K-SWEEP RESULT  domain={domain_name}  arch={arch}")
        print(f"{'k':>4}  {'mean MAE':>10}  {'n_windows':>10}")
        for k, res in sorted(results_by_k.items()):
            marker = " <- best" if k == best_k else ""
            print(f"{k:>4}  {res['mean_mae']:>10.2f}C  {res['n_windows']:>10}{marker}")

    return {
        "domain": domain_name,
        "arch": arch,
        "dim": 3,
        "best_k": best_k,
        "best_mae": best_mae,
        "all_k": {str(k): res["mean_mae"] for k, res in results_by_k.items()},
        "checkpoint_best": results_by_k[best_k]["checkpoint"],
        "run_tag": run_tag,
        "warm_start": True,
    }


def build_parser() -> argparse.ArgumentParser:
    """CLI parser for the improved 3D warm-start training entry point."""
    parser = argparse.ArgumentParser(
        description="Improved 3D ThermalPINN warm-start training with corrected scaling."
    )
    parser.add_argument("--domain", default="cylinder", choices=list(DOMAINS_3D))
    parser.add_argument("--arch", default="bayesian", choices=list(ARCH_CONFIGS))
    parser.add_argument("--k", type=int, default=1, help="Single k value without --sweep_k.")
    parser.add_argument("--sweep_k", action="store_true", help="Run all k values listed in --k_values.")
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=DEFAULT_K_VALUES,
        help="k values for the sweep (default: 1 2).",
    )
    parser.add_argument("--all", action="store_true", help="Run all 3D domains x architectures.")
    parser.add_argument("--cuda", action="store_true", help="Require CUDA.")
    parser.add_argument("--device", default=None, help="Explicit torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--tag", default="improved3d_ws", help="Suffix tag for checkpoints and metrics.")
    parser.add_argument("--max-windows", type=int, default=None, help="Limit windows for quick smoke tests.")
    parser.add_argument("--seed-base", type=int, default=0, help="Base RNG seed for window sampling.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging.")

    parser.add_argument("--n-domain", dest="n_domain", type=int, default=None)
    parser.add_argument("--n-bc", dest="n_bc", type=int, default=None)
    parser.add_argument("--n-eval", dest="n_eval", type=int, default=None)
    parser.add_argument("--lr-min", dest="lr_min", type=float, default=None)
    parser.add_argument("--lbfgs-iters", dest="lbfgs_iters", type=int, default=None)
    parser.add_argument("--lbfgs-iters-warm", dest="lbfgs_iters_warm", type=int, default=None)
    parser.add_argument("--cold-epochs", type=int, default=None)
    parser.add_argument("--warm-epochs", type=int, default=None)
    parser.add_argument("--lr-cold", dest="lr_cold", type=float, default=None)
    parser.add_argument("--lr-warm", dest="lr_warm", type=float, default=None)
    parser.add_argument("--no-end-supervision", action="store_true")

    parser.add_argument("--w-pde", type=float, default=W_PDE)
    parser.add_argument("--w-ic", type=float, default=W_IC)
    parser.add_argument("--w-bc", type=float, default=W_BC)
    parser.add_argument("--w-end", type=float, default=W_END)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = resolve_device(args)
    if device.type == "cuda":
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    verbose = not args.quiet
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)

    if args.all:
        for domain_name in DOMAINS_3D:
            for arch in ARCH_CONFIGS:
                entry = sweep_k_ws(
                    domain_name,
                    arch,
                    k_values=args.k_values,
                    device=device,
                    run_tag=args.tag,
                    max_windows=args.max_windows,
                    verbose=verbose,
                    args=args,
                )
                update_registry(entry)
        return

    if args.sweep_k:
        entry = sweep_k_ws(
            args.domain,
            args.arch,
            k_values=args.k_values,
            device=device,
            run_tag=args.tag,
            max_windows=args.max_windows,
            verbose=verbose,
            args=args,
        )
        update_registry(entry)
        return

    result = run_one_ws(
        args.domain,
        args.arch,
        args.k,
        device=device,
        run_tag=args.tag,
        max_windows=args.max_windows,
        verbose=verbose,
        args=args,
    )
    update_registry(
        {
            "domain": result["domain"],
            "arch": result["arch"],
            "dim": result["dim"],
            "best_k": result["k"],
            "best_mae": result["mean_mae"],
            "all_k": {str(result["k"]): result["mean_mae"]},
            "checkpoint_best": result["checkpoint"],
            "run_tag": args.tag,
            "warm_start": True,
        }
    )


if __name__ == "__main__":
    main()
