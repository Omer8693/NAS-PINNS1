"""
thermal_pinn/training/train_warmstart.py
=========================================
Warm-start variant of the k-skip MSWP training.

Strategy
--------
Window 1  : cold-start — random init, full Adam (N_EPOCHS_COLD=800) + L-BFGS
Window 2+ : warm-start — carry previous weights, short Adam (N_EPOCHS_WARM=300)
            with a lower LR (LR_WARM=3e-4). Because the physics changes only
            gradually between windows, the network is already near the solution;
            fewer gradient steps are enough to adapt to the new IC/window.

This stays within the paper baseline (Mortensen et al., 2026):
  - FEM still provides θ_ic at every window boundary (no PINN-only rollout)
  - FEM → PINN → FEM → PINN pattern unchanged
  - k-skip logic identical to train_all.py

Output
------
  checkpoints/<domain>_<arch>_k<k>_dim<dim>_ws_metrics.json
  checkpoints/<domain>_<arch>_k<k>_dim<dim>_ws.pt

Usage
-----
    # Test: rectangle / nsga2 / k=3
    python -m thermal_pinn.training.train_warmstart \\
        --domain rectangle --arch nsga2 --k 3 --dim 2

    # Sweep all k for one combo
    python -m thermal_pinn.training.train_warmstart \\
        --domain rectangle --arch nsga2 --sweep_k --dim 2

    # All domains × archs × k (full run)
    python -m thermal_pinn.training.train_warmstart --all --cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent))

from thermal_pinn.physics.domains_2d import make_domain, DOMAINS_2D
from thermal_pinn.physics.domains_3d import make_domain_3d, DOMAINS_3D
from thermal_pinn.network.pinn        import make_pinn, ARCH_CONFIGS
from thermal_pinn.training.trainer    import (
    _pde_residual, _robin_bc_loss, _normalise_coords,
    evaluate_window, K_VALUES,
    W_PDE, W_IC, W_BC, W_END,
    K_THERM, H_CONV,
)

# ── Simulation constants (identical to train_all.py) ─────────────────────────
T_TOTAL     = 30.0
DT_FEM      = 1.5
N_FEM_STEPS = int(T_TOTAL / DT_FEM)   # 20
T_WATER     = 20.0
DELTA_T     = 520.0

CHECKPOINT_DIR = _ROOT / "checkpoints"

# ── Warm-start hyper-params (defaults; overridden by CLI args) ────────────────
N_EPOCHS_COLD = 800    # window 1: full training (same as train_all.py)
N_EPOCHS_WARM = 300    # window 2+: fine-tune only (overridden by --warm_epochs)
LR_COLD       = 1e-3
LR_WARM       = 3e-4   # warm LR start (overridden by --warm_lr)
LR_MIN        = 1e-5
LBFGS_ITERS   = 50
N_DOMAIN      = 1500
N_BC          = 300


def theta_from_domain(domain, coords_np: np.ndarray, t: float) -> np.ndarray:
    if coords_np.shape[1] == 2:
        T_vals = domain.T(coords_np[:, 0], coords_np[:, 1], t)
    else:
        T_vals = domain.T(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], t)
    T_vals = np.where(np.isfinite(T_vals), T_vals, T_WATER)
    return ((T_vals - T_WATER) / DELTA_T).astype(np.float32)


def train_window_ws(model: torch.nn.Module,
                    domain,
                    t_start: float,
                    t_end: float,
                    theta_ic_fn,
                    theta_end_fn=None,
                    warm: bool = False,
                    n_epochs_warm: int = N_EPOCHS_WARM,
                    lr_warm: float = LR_WARM,
                    device: torch.device | None = None,
                    rng_seed: int = 0) -> dict:
    """
    Train on a single window.

    Parameters
    ----------
    warm          : If True, use warm-start hyper-params (window 2+).
    n_epochs_warm : Number of epochs for warm windows (overrides default).
    lr_warm       : Starting LR for warm windows (overrides default).
    """
    device    = device or next(model.parameters()).device
    alpha     = getattr(domain, "alpha", 6.25e-5)
    h         = getattr(domain, "h", H_CONV)
    dim       = model.dim
    dt_window = t_end - t_start
    rng       = np.random.default_rng(rng_seed)

    n_epochs = n_epochs_warm if warm else N_EPOCHS_COLD
    lr       = lr_warm       if warm else LR_COLD

    # ── Collocation points ────────────────────────────────────────────────────
    if dim == 2:
        xi, yi    = domain.sample_interior(N_DOMAIN, rng=rng)
        coords_np = np.stack([xi, yi], axis=1)
        bx, by, nx_, ny_ = domain.sample_boundary(N_BC, rng=rng)
        bc_np     = np.stack([bx, by],   axis=1)
        norms_np  = np.stack([nx_, ny_], axis=1)
    else:
        xi, yi, zi = domain.sample_interior(N_DOMAIN, rng=rng)
        coords_np  = np.stack([xi, yi, zi], axis=1)
        bx, by, bz, nx_, ny_, nz_ = domain.sample_boundary(N_BC, rng=rng)
        bc_np      = np.stack([bx, by, bz],     axis=1)
        norms_np   = np.stack([nx_, ny_, nz_],  axis=1)

    def to_t(a): return torch.tensor(a, dtype=torch.float32, device=device)

    coords_norm = _normalise_coords(coords_np, domain)
    bc_norm     = _normalise_coords(bc_np,     domain)

    tau_np  = rng.uniform(0, 1, (N_DOMAIN, 1)).astype(np.float32)
    tau_bc  = rng.uniform(0, 1, (N_BC,     1)).astype(np.float32)

    theta_ic_vals = theta_ic_fn(coords_np).reshape(-1, 1).astype(np.float32)
    theta_ic_bc   = theta_ic_fn(bc_np).reshape(-1, 1).astype(np.float32)

    coords_t      = to_t(coords_norm).requires_grad_(True)
    tau_t         = to_t(tau_np).requires_grad_(True)
    theta_ic_t    = to_t(theta_ic_vals)
    bc_t          = to_t(bc_norm).requires_grad_(True)
    tau_bc_t      = to_t(tau_bc).requires_grad_(True)
    theta_ic_bc_t = to_t(theta_ic_bc)
    norms_t       = to_t(norms_np)

    theta_end_t = None
    if theta_end_fn is not None:
        theta_end_vals = theta_end_fn(coords_np).reshape(-1, 1).astype(np.float32)
        theta_end_t    = to_t(theta_end_vals)
        tau_end_t      = torch.ones(N_DOMAIN, 1, device=device)

    # ── Phase 1: Adam ─────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=LR_MIN)
    loss_history = []
    t0 = time.time()
    model.train()

    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss_pde = _pde_residual(model, coords_t, tau_t, theta_ic_t,
                                 dt_window, alpha)
        tau_zero = torch.zeros(N_DOMAIN, 1, device=device)
        pred_ic  = model(coords_t, tau_zero, theta_ic_t)
        loss_ic  = ((pred_ic - theta_ic_t) ** 2).mean()
        loss_bc  = _robin_bc_loss(model, bc_t, norms_t, tau_bc_t,
                                  theta_ic_bc_t, h)
        loss     = W_PDE * loss_pde + W_IC * loss_ic + W_BC * loss_bc
        if theta_end_t is not None:
            pred_end = model(coords_t, tau_end_t, theta_ic_t)
            loss     = loss + W_END * ((pred_end - theta_end_t) ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(float(loss.detach()))

    # ── Phase 2: L-BFGS ───────────────────────────────────────────────────────
    if LBFGS_ITERS > 0:
        lbfgs = optim.LBFGS(model.parameters(),
                             max_iter=LBFGS_ITERS,
                             line_search_fn="strong_wolfe",
                             tolerance_grad=1e-7,
                             tolerance_change=1e-9)

        def closure():
            lbfgs.zero_grad()
            lp = _pde_residual(model, coords_t, tau_t, theta_ic_t,
                               dt_window, alpha)
            pi_ = model(coords_t, torch.zeros(N_DOMAIN, 1, device=device), theta_ic_t)
            li  = ((pi_ - theta_ic_t) ** 2).mean()
            lb  = _robin_bc_loss(model, bc_t, norms_t, tau_bc_t,
                                 theta_ic_bc_t, h)
            total = W_PDE * lp + W_IC * li + W_BC * lb
            if theta_end_t is not None:
                total = total + W_END * ((model(coords_t, tau_end_t, theta_ic_t)
                                         - theta_end_t) ** 2).mean()
            total.backward()
            return total

        try:
            lbfgs.step(closure)
        except RuntimeError:
            pass

    return {"loss_history": loss_history, "runtime_s": time.time() - t0}


def run_one_ws(domain_name: str, arch: str, k: int, dim: int,
               device: torch.device,
               n_epochs_warm: int = N_EPOCHS_WARM,
               lr_warm: float = LR_WARM,
               tag: str = "ws",
               verbose: bool = True) -> dict:
    """
    Train (domain, arch, k) with warm-start strategy.

    Window 1  → cold-start (800 epochs, lr=1e-3)
    Window 2+ → warm-start (n_epochs_warm epochs, lr_warm), weights inherited

    tag : suffix for output files, e.g. 'ws', 'ws_500ep', 'ws_highlr'
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
        warm  = (i > 0)   # window 1 is always cold

        if verbose:
            mode = f"warm({n_epochs_warm}ep,lr={lr_warm:.0e})" if warm else "cold(800ep)"
            print(f"  [{i+1}/{len(anchors)}] [{t_start:.1f}→{t_end:.1f}s]  "
                  f"k={k}  {mode}  {domain_name}/{arch}", flush=True)

        def theta_ic_fn(coords_np, _t=t_start):
            return theta_from_domain(domain, coords_np, _t)

        def theta_end_fn(coords_np, _t=t_end):
            return theta_from_domain(domain, coords_np, _t)

        train_info = train_window_ws(
            model=model, domain=domain,
            t_start=t_start, t_end=t_end,
            theta_ic_fn=theta_ic_fn,
            theta_end_fn=theta_end_fn,
            warm=warm,
            n_epochs_warm=n_epochs_warm,
            lr_warm=lr_warm,
            device=device,
            rng_seed=i,
        )

        eval_info = evaluate_window(
            model=model, domain=domain,
            t_start=t_start, t_end=t_end,
            theta_ic_fn=theta_ic_fn,
            device=device,
        )

        ep = n_epochs_warm if warm else N_EPOCHS_COLD
        if verbose:
            print(f"      MAE={eval_info['mae_C']:.2f}°C  "
                  f"L2={eval_info['l2_rel']:.4f}  "
                  f"({train_info['runtime_s']:.0f}s, {ep}ep)", flush=True)

        window_results.append({
            "t_start":   float(t_start),
            "t_end":     float(t_end),
            "mae_C":     eval_info["mae_C"],
            "l2_rel":    eval_info["l2_rel"],
            "runtime_s": train_info["runtime_s"],
            "warm":      warm,
        })

    mean_mae = float(np.mean([w["mae_C"] for w in window_results]))
    total_t  = time.time() - t0_total

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path    = CHECKPOINT_DIR / f"{domain_name}_{arch}_k{k}_dim{dim}_{tag}.pt"
    metrics_path = CHECKPOINT_DIR / f"{domain_name}_{arch}_k{k}_dim{dim}_{tag}_metrics.json"

    torch.save({
        "model_state": model.state_dict(),
        "arch": arch, "dim": dim,
        "domain": domain_name, "k": k,
        "mean_mae": mean_mae, "windows": window_results,
    }, ckpt_path)

    result = {
        "domain": domain_name, "arch": arch, "k": k, "dim": dim,
        "mean_mae": mean_mae, "n_windows": len(window_results),
        "windows": window_results, "total_s": total_t,
        "checkpoint": str(ckpt_path),
        "warm_start": True,
        "tag": tag,
        "n_epochs_cold": N_EPOCHS_COLD,
        "n_epochs_warm": n_epochs_warm,
        "lr_warm": lr_warm,
    }

    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"  → Mean MAE={mean_mae:.2f}°C  saved: {ckpt_path.name}", flush=True)

    return result


def compare_cold_vs_warm(domain_name: str, arch: str, k: int, dim: int,
                         *tags: str):
    """
    Print side-by-side comparison: cold-start vs one or more warm-start tags.
    tags: e.g. 'ws', 'ws_500ep', 'ws_highlr'
    """
    cold_p = CHECKPOINT_DIR / f"{domain_name}_{arch}_k{k}_dim{dim}_metrics.json"
    if not cold_p.exists():
        print("  Cold-start metrics not found — run train_all.py first.")
        return

    cold   = json.load(open(cold_p))
    warms  = {}
    for tag in tags:
        p = CHECKPOINT_DIR / f"{domain_name}_{arch}_k{k}_dim{dim}_{tag}_metrics.json"
        if p.exists():
            warms[tag] = json.load(open(p))
        else:
            print(f"  [{tag}] not found — skipping.")

    if not warms:
        return

    header_tags = "  ".join(f"{'L2 '+t:>12}" for t in warms)
    print(f"\n{'='*75}")
    print(f"  COLD vs WARM  {domain_name}  {arch}  k={k}  dim={dim}D")
    print(f"{'='*75}")
    print(f"  {'Win':>3}  {'t[s]':>10}  {'L2 cold':>10}  {header_tags}")

    for wi, cw in enumerate(cold["windows"]):
        row = (f"  {wi+1:>3}  [{cw['t_start']:.1f}→{cw['t_end']:.1f}]  "
               f"{cw['l2_rel']*100:>8.2f}%")
        for tag, wdata in warms.items():
            if wi < len(wdata["windows"]):
                ww    = wdata["windows"][wi]
                delta = ww["l2_rel"] - cw["l2_rel"]
                flag  = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "≈")
                row += f"  {ww['l2_rel']*100:>8.2f}% {flag}"
        print(row)

    print(f"  {'Mean MAE:':>20}  {cold['mean_mae']:>8.2f}°C  " +
          "  ".join(f"{w['mean_mae']:>8.2f}°C" for w in warms.values()))
    tot_c = sum(w["runtime_s"] for w in cold["windows"])
    print(f"  {'Total runtime:':>20}  {tot_c:>8.0f}s   " +
          "  ".join(f"{sum(w['runtime_s'] for w in wd['windows']):>8.0f}s "
                    f"({100*(sum(w['runtime_s'] for w in wd['windows'])-tot_c)/tot_c:+.0f}%)"
                    for wd in warms.values()))
    print(f"{'='*75}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Warm-start k-skip PINN training."
    )
    parser.add_argument("--domain",       default="rectangle",
                        choices=list(DOMAINS_2D) + list(DOMAINS_3D))
    parser.add_argument("--arch",         default="nsga2",
                        choices=list(ARCH_CONFIGS))
    parser.add_argument("--k",            type=int, default=3)
    parser.add_argument("--sweep_k",      action="store_true")
    parser.add_argument("--k_values",     nargs="+", type=int, default=K_VALUES)
    parser.add_argument("--dim",          type=int, default=2, choices=[2, 3])
    parser.add_argument("--all",          action="store_true",
                        help="All domains × archs × k")
    parser.add_argument("--warm_epochs",  type=int, default=N_EPOCHS_WARM,
                        help=f"Epochs for warm windows (default {N_EPOCHS_WARM})")
    parser.add_argument("--warm_lr",      type=float, default=LR_WARM,
                        help=f"Start LR for warm windows (default {LR_WARM})")
    parser.add_argument("--tag",          default="ws",
                        help="Output file suffix, e.g. ws_500ep, ws_highlr")
    parser.add_argument("--compare",      action="store_true",
                        help="Print cold vs warm comparison after training")
    parser.add_argument("--compare_tags", nargs="+", default=None,
                        help="Extra tags to include in comparison table")
    parser.add_argument("--cuda",         action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available())
                          else "cpu")
    print(f"[train_warmstart]  device={device}  "
          f"warm_epochs={args.warm_epochs}  warm_lr={args.warm_lr}  "
          f"tag={args.tag}", flush=True)

    def _run(domain_name, arch, k, dim):
        run_one_ws(domain_name, arch, k, dim, device,
                   n_epochs_warm=args.warm_epochs,
                   lr_warm=args.warm_lr,
                   tag=args.tag)
        if args.compare:
            tags = [args.tag] + (args.compare_tags or [])
            compare_cold_vs_warm(domain_name, arch, k, dim, *tags)

    if args.all:
        combos = ([(d, 2) for d in DOMAINS_2D] +
                  [(d, 3) for d in DOMAINS_3D])
        for domain_name, dim in combos:
            for arch in ARCH_CONFIGS:
                for k in args.k_values:
                    _run(domain_name, arch, k, dim)
    elif args.sweep_k:
        for k in args.k_values:
            _run(args.domain, args.arch, k, args.dim)
    else:
        _run(args.domain, args.arch, args.k, args.dim)


if __name__ == "__main__":
    main()
