"""
thermal_pinn/training/train_3d_improved.py
==========================================
3D-focused training entry point with physically consistent coordinate scaling.

Why this script exists
----------------------
The baseline 3D training path reuses the 2D-oriented normalisation and training
budget almost unchanged. For centered domains such as the cylinder, that can
mis-scale the input coordinates; for anisotropic 3D domains, the PDE and Robin
boundary residuals also benefit from using the true physical axis lengths.

This script keeps the original project intact while providing a separate,
practical training path for 3D experiments:
    - true bounding-box normalisation, including centred cylinder coordinates
    - PDE residual scaled by per-axis physical lengths
    - Robin BC residual computed with the physical gradient
    - larger 3D-oriented default budgets
    - GPU-friendly CLI and separate output directory

Usage
-----
    # Recommended quick 3D run on GPU:
    python -m thermal_pinn.training.train_3d_improved \
        --domain cylinder --arch bayesian --k 1 --cuda

    # Sweep only k=1 and k=2 across one geometry/arch:
    python -m thermal_pinn.training.train_3d_improved \
        --domain rectangular --arch nsga2 --sweep_k --cuda

    # Full 3D sweep across all geometries and all NAS architectures:
    python -m thermal_pinn.training.train_3d_improved \
        --all --cuda

    # Tiny smoke test:
    python -m thermal_pinn.training.train_3d_improved \
        --domain rectangular --arch bayesian --k 1 \
        --n-epochs 1 --n-domain 16 --n-bc 8 --n-eval 16 \
        --lbfgs-iters 1 --max-windows 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.optim as optim

# Allow running as script from repo root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent))

from thermal_pinn.network.pinn import ARCH_CONFIGS, ThermalPINN, make_pinn
from thermal_pinn.physics.domains_3d import DOMAINS_3D, make_domain_3d
from thermal_pinn.training.trainer import (
    DELTA_T,
    H_CONV,
    K_THERM,
    T_WATER,
    W_BC,
    W_END,
    W_IC,
    W_PDE,
)

T_TOTAL = 30.0
DT_FEM = 1.5
DEFAULT_K_VALUES = [1, 2]

DEFAULT_TRAIN_CONFIG = {
    "n_domain": 10000,
    "n_bc": 2000,
    "n_epochs": 1800,
    "lr": 8e-4,
    "lr_min": 1e-5,
    "lbfgs_iters": 100,
    "n_eval": 3000,
    "use_end_supervision": True,
}

DOMAIN_CONFIG_OVERRIDES = {
    "rectangular": {
        "n_domain": 8000,
        "n_bc": 1600,
        "n_epochs": 1600,
        "n_eval": 2500,
    },
    "cylinder": {
        "n_domain": 12000,
        "n_bc": 2400,
        "n_epochs": 1800,
        "n_eval": 4000,
    },
    "stacked": {
        "n_domain": 10000,
        "n_bc": 2000,
        "n_epochs": 1800,
        "n_eval": 3000,
    },
    "lshape": {
        "n_domain": 14000,
        "n_bc": 2800,
        "n_epochs": 2200,
        "n_eval": 4000,
    },
}

CHECKPOINT_DIR = _ROOT / "checkpoints" / "improved3d"


def theta_from_domain(domain, coords_np: np.ndarray, t: float) -> np.ndarray:
    """Return dimensionless anchor temperature theta = (T - T_w) / DeltaT."""
    T_vals = domain.T(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], t)
    T_vals = np.where(np.isfinite(T_vals), T_vals, T_WATER)
    return ((T_vals - T_WATER) / DELTA_T).astype(np.float32)


def domain_bounds(domain) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the physical bounding box for a 3D domain.

    The cylinder uses centered x-y coordinates, so its lower bound is not zero.
    """
    if hasattr(domain, "R") and hasattr(domain, "Hz"):
        lo = np.array([-float(domain.R), -float(domain.R), 0.0], dtype=np.float32)
        hi = np.array([+float(domain.R), +float(domain.R), float(domain.Hz)], dtype=np.float32)
        return lo, hi

    lo = np.zeros(3, dtype=np.float32)
    hi = np.array(
        [
            float(getattr(domain, "Lx", 1.0)),
            float(getattr(domain, "Ly", 1.0)),
            float(getattr(domain, "Lz", getattr(domain, "Hz", 1.0))),
        ],
        dtype=np.float32,
    )
    return lo, hi


def normalise_coords(coords: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map physical coordinates to [0, 1]^3 using the true bounding box."""
    span = (hi - lo).astype(np.float32)
    span[span < 1e-8] = 1.0
    coords_norm = ((coords - lo) / span).astype(np.float32)
    return coords_norm, span


def pde_residual_physical(
    model: ThermalPINN,
    coords: torch.Tensor,
    tau: torch.Tensor,
    theta_ic: torch.Tensor,
    dt_window: float,
    alpha: float,
    span_t: torch.Tensor,
) -> torch.Tensor:
    """
    Compute PDE residual using the physical Laplacian.

    If x_norm = (x - lo) / span, then:
        d2/dx_phys^2 = (1/span^2) * d2/dx_norm^2
    """
    theta = model(coords, tau, theta_ic)

    dtheta_dtau = torch.autograd.grad(
        theta,
        tau,
        grad_outputs=torch.ones_like(theta),
        create_graph=True,
        retain_graph=True,
    )[0]

    grads = torch.autograd.grad(
        theta,
        coords,
        grad_outputs=torch.ones_like(theta),
        create_graph=True,
        retain_graph=True,
    )[0]

    inv_span2 = (1.0 / (span_t ** 2)).view(1, -1)
    laplacian_phys = torch.zeros_like(theta)
    for i in range(coords.shape[1]):
        d2_norm = torch.autograd.grad(
            grads[:, i:i + 1],
            coords,
            grad_outputs=torch.ones_like(grads[:, i:i + 1]),
            create_graph=True,
            retain_graph=True,
        )[0][:, i:i + 1]
        laplacian_phys = laplacian_phys + inv_span2[:, i:i + 1] * d2_norm

    alpha_eff = alpha * dt_window
    return (dtheta_dtau - alpha_eff * laplacian_phys).pow(2).mean()


def robin_bc_loss_physical(
    model: ThermalPINN,
    bc_coords: torch.Tensor,
    bc_normals: torch.Tensor,
    tau_bc: torch.Tensor,
    theta_ic_bc: torch.Tensor,
    h: float,
    span_t: torch.Tensor,
) -> torch.Tensor:
    """
    Robin BC loss with a physically scaled normal derivative.

    grad_phys = grad_norm / span
    dtheta/dn = grad_phys dot n_hat
    """
    theta = model(bc_coords, tau_bc, theta_ic_bc)
    grads_norm = torch.autograd.grad(
        theta,
        bc_coords,
        grad_outputs=torch.ones_like(theta),
        create_graph=True,
        retain_graph=True,
    )[0]

    grads_phys = grads_norm / span_t.view(1, -1)
    dtheta_dn = (grads_phys * bc_normals).sum(dim=1, keepdim=True)

    bi_approx = max(h / K_THERM, 1.0)
    return ((dtheta_dn + (h / K_THERM) * theta) / bi_approx).pow(2).mean()


def train_window_improved(
    model: ThermalPINN,
    domain,
    t_start: float,
    t_end: float,
    theta_ic_fn: Callable,
    theta_end_fn: Callable | None = None,
    *,
    n_domain: int,
    n_bc: int,
    n_epochs: int,
    lr: float,
    lr_min: float,
    lbfgs_iters: int,
    use_end_supervision: bool,
    n_eval: int,
    device: torch.device,
    rng_seed: int,
    w_pde: float,
    w_ic: float,
    w_bc: float,
    w_end: float,
) -> dict:
    """
    Train on one window using corrected 3D coordinate handling.

    `n_eval` is accepted here for config symmetry and stored in the result, even
    though evaluation itself is performed in `evaluate_window_improved`.
    """
    del n_eval

    alpha = getattr(domain, "alpha", 6.25e-5)
    h = getattr(domain, "h", H_CONV)
    dt_window = t_end - t_start
    rng = np.random.default_rng(rng_seed)

    xi, yi, zi = domain.sample_interior(n_domain, rng=rng)
    coords_np = np.stack([xi, yi, zi], axis=1)

    bx, by, bz, nx_, ny_, nz_ = domain.sample_boundary(n_bc, rng=rng)
    bc_np = np.stack([bx, by, bz], axis=1)
    normals_np = np.stack([nx_, ny_, nz_], axis=1).astype(np.float32)

    lo, hi = domain_bounds(domain)
    coords_norm, span = normalise_coords(coords_np, lo, hi)
    bc_norm, _ = normalise_coords(bc_np, lo, hi)

    tau_np = rng.uniform(0.0, 1.0, (n_domain, 1)).astype(np.float32)
    tau_bc_np = rng.uniform(0.0, 1.0, (n_bc, 1)).astype(np.float32)

    theta_ic_vals = theta_ic_fn(coords_np).reshape(-1, 1).astype(np.float32)
    theta_ic_bc = theta_ic_fn(bc_np).reshape(-1, 1).astype(np.float32)

    def to_t(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=device)

    coords_t = to_t(coords_norm).requires_grad_(True)
    tau_t = to_t(tau_np).requires_grad_(True)
    theta_ic_t = to_t(theta_ic_vals)
    bc_t = to_t(bc_norm).requires_grad_(True)
    tau_bc_t = to_t(tau_bc_np).requires_grad_(True)
    theta_ic_bc_t = to_t(theta_ic_bc)
    normals_t = to_t(normals_np)
    span_t = to_t(span)

    theta_end_t = None
    tau_end_t = None
    if use_end_supervision and theta_end_fn is not None:
        theta_end_vals = theta_end_fn(coords_np).reshape(-1, 1).astype(np.float32)
        theta_end_t = to_t(theta_end_vals)
        tau_end_t = torch.ones(n_domain, 1, dtype=torch.float32, device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=lr_min,
    )
    loss_history: list[float] = []
    t0 = time.time()
    model.train()

    for _ in range(n_epochs):
        optimizer.zero_grad()

        loss_pde = pde_residual_physical(
            model,
            coords_t,
            tau_t,
            theta_ic_t,
            dt_window,
            alpha,
            span_t,
        )
        tau_zero = torch.zeros(n_domain, 1, dtype=torch.float32, device=device)
        pred_ic = model(coords_t, tau_zero, theta_ic_t)
        loss_ic = ((pred_ic - theta_ic_t) ** 2).mean()
        loss_bc_term = robin_bc_loss_physical(
            model,
            bc_t,
            normals_t,
            tau_bc_t,
            theta_ic_bc_t,
            h,
            span_t,
        )

        total_loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc_term
        if theta_end_t is not None and tau_end_t is not None:
            pred_end = model(coords_t, tau_end_t, theta_ic_t)
            loss_end = ((pred_end - theta_end_t) ** 2).mean()
            total_loss = total_loss + w_end * loss_end

        total_loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(float(total_loss.detach()))

    if lbfgs_iters > 0:
        lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_iters,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

        def closure():
            lbfgs.zero_grad()
            loss_pde = pde_residual_physical(
                model,
                coords_t,
                tau_t,
                theta_ic_t,
                dt_window,
                alpha,
                span_t,
            )
            pred_ic = model(
                coords_t,
                torch.zeros(n_domain, 1, dtype=torch.float32, device=device),
                theta_ic_t,
            )
            loss_ic = ((pred_ic - theta_ic_t) ** 2).mean()
            loss_bc_term = robin_bc_loss_physical(
                model,
                bc_t,
                normals_t,
                tau_bc_t,
                theta_ic_bc_t,
                h,
                span_t,
            )
            total = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc_term
            if theta_end_t is not None and tau_end_t is not None:
                total = total + w_end * ((model(coords_t, tau_end_t, theta_ic_t) - theta_end_t) ** 2).mean()
            total.backward()
            return total

        try:
            lbfgs.step(closure)
        except RuntimeError:
            pass

    return {
        "loss_history": loss_history,
        "runtime_s": time.time() - t0,
        "bbox_lo": lo.tolist(),
        "bbox_hi": hi.tolist(),
        "span": span.tolist(),
    }


def evaluate_window_improved(
    model: ThermalPINN,
    domain,
    t_end: float,
    theta_ic_fn: Callable,
    *,
    n_eval: int,
    device: torch.device,
) -> dict:
    """Evaluate one trained window using the corrected bounding-box normalisation."""
    rng = np.random.default_rng(42)
    xi, yi, zi = domain.sample_interior(n_eval, rng=rng)
    coords_np = np.stack([xi, yi, zi], axis=1)
    T_ref = domain.T(xi, yi, zi, t_end)

    lo, hi = domain_bounds(domain)
    coords_norm, _ = normalise_coords(coords_np, lo, hi)
    theta_ic = theta_ic_fn(coords_np).reshape(-1, 1).astype(np.float32)
    tau_one = np.ones((n_eval, 1), dtype=np.float32)

    def to_t(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        theta_pred = model(
            to_t(coords_norm),
            to_t(tau_one),
            to_t(theta_ic),
        ).cpu().numpy().flatten()

    T_pred = T_WATER + DELTA_T * theta_pred
    valid = np.isfinite(T_ref)
    mae = float(np.mean(np.abs(T_pred[valid] - T_ref[valid])))
    l2 = float(
        np.linalg.norm(T_pred[valid] - T_ref[valid]) /
        (np.linalg.norm(T_ref[valid] - T_WATER) + 1e-10)
    )
    return {
        "mae_C": mae,
        "l2_rel": l2,
        "T_pred": T_pred,
        "T_ref": T_ref,
        "coords": coords_np,
    }


def build_run_config(args: argparse.Namespace, domain_name: str) -> dict:
    """Merge global defaults, per-domain defaults, and CLI overrides."""
    config = dict(DEFAULT_TRAIN_CONFIG)
    config.update(DOMAIN_CONFIG_OVERRIDES.get(domain_name, {}))

    for key in ("n_domain", "n_bc", "n_epochs", "lr", "lr_min", "lbfgs_iters", "n_eval"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    config["use_end_supervision"] = not args.no_end_supervision
    return config


def run_one(
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
    """Train a single 3D domain/architecture/k configuration."""
    domain = make_domain_3d(domain_name)
    model = make_pinn(dim=3, arch=arch, device=device)
    config = build_run_config(args, domain_name)

    dt_window = k * DT_FEM
    anchors = np.arange(0.0, T_TOTAL - dt_window / 2, dt_window)
    if max_windows is not None:
        anchors = anchors[:max_windows]

    if verbose:
        print(f"\n{'=' * 72}")
        print(f"3D IMPROVED RUN  domain={domain_name}  arch={arch}  k={k}")
        print(f"device={device}  windows={len(anchors)}  tag={run_tag}")
        print(
            f"n_domain={config['n_domain']}  n_bc={config['n_bc']}  "
            f"n_epochs={config['n_epochs']}  n_eval={config['n_eval']}  "
            f"lbfgs_iters={config['lbfgs_iters']}"
        )
        print(f"{'=' * 72}", flush=True)

    window_results = []
    t0_total = time.time()

    for i, t_start in enumerate(anchors):
        t_end = min(t_start + dt_window, T_TOTAL)
        if verbose:
            print(
                f"  [{i + 1}/{len(anchors)}] Window [{t_start:.1f} -> {t_end:.1f}s]  "
                f"{domain_name}/{arch}/k{k}",
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
            n_epochs=config["n_epochs"],
            lr=config["lr"],
            lr_min=config["lr_min"],
            lbfgs_iters=config["lbfgs_iters"],
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
            "weights": {
                "w_pde": args.w_pde,
                "w_ic": args.w_ic,
                "w_bc": args.w_bc,
                "w_end": args.w_end,
            },
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


def sweep_k(
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
    """Run a k sweep and keep the best 3D configuration."""
    if verbose:
        print(f"\n{'=' * 72}")
        print(f"3D IMPROVED K-SWEEP  domain={domain_name}  arch={arch}")
        print(f"k values: {k_values}")
        print(f"{'=' * 72}", flush=True)

    results_by_k = {}
    for k in k_values:
        results_by_k[k] = run_one(
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
        print(f"\n3D IMPROVED K-SWEEP RESULT  domain={domain_name}  arch={arch}")
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
    }


def update_registry(entry: dict) -> None:
    """Append or replace one entry in the improved 3D registry."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    reg_path = CHECKPOINT_DIR / "registry.json"

    registry = []
    if reg_path.exists():
        with open(reg_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

    key = (entry["domain"], entry["arch"], entry["dim"], entry["run_tag"])
    registry = [
        row for row in registry
        if not (
            row.get("domain") == key[0]
            and row.get("arch") == key[1]
            and row.get("dim") == key[2]
            and row.get("run_tag") == key[3]
        )
    ]
    registry.append(entry)

    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"Registry updated: {reg_path}", flush=True)


def resolve_device(args: argparse.Namespace) -> torch.device:
    """Choose CPU/GPU device with a user-friendly failure mode."""
    if args.device is not None:
        return torch.device(args.device)
    if args.cuda:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but no GPU is visible to PyTorch.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_parser() -> argparse.ArgumentParser:
    """CLI parser for the improved 3D training entry point."""
    parser = argparse.ArgumentParser(
        description="Improved 3D ThermalPINN training with corrected scaling."
    )
    parser.add_argument("--domain", default="cylinder", choices=list(DOMAINS_3D))
    parser.add_argument("--arch", default="bayesian", choices=list(ARCH_CONFIGS))
    parser.add_argument("--k", type=int, default=1, help="Single k value without --sweep_k.")
    parser.add_argument(
        "--sweep_k",
        action="store_true",
        help="Run all k values listed in --k_values and keep the best.",
    )
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
    parser.add_argument("--tag", default="improved3d", help="Suffix tag for checkpoints and metrics.")
    parser.add_argument("--max-windows", type=int, default=None, help="Limit windows for quick smoke tests.")
    parser.add_argument("--seed-base", type=int, default=0, help="Base RNG seed for window sampling.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging.")

    parser.add_argument("--n-domain", dest="n_domain", type=int, default=None)
    parser.add_argument("--n-bc", dest="n_bc", type=int, default=None)
    parser.add_argument("--n-epochs", dest="n_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-min", dest="lr_min", type=float, default=None)
    parser.add_argument("--lbfgs-iters", dest="lbfgs_iters", type=int, default=None)
    parser.add_argument("--n-eval", dest="n_eval", type=int, default=None)
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
                entry = sweep_k(
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
        entry = sweep_k(
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

    result = run_one(
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
        }
    )


if __name__ == "__main__":
    main()
