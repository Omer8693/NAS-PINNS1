"""
poisson_pinn.py — Poisson PINN Solver (Domain-Agnostic)
=========================================================
Solves  -Δu = f(x,y)  on Ω,  u = 0  on ∂Ω.

Loss:
  L_pde = mean( (-Δu_pred - f)² ) / PDE_SCALE
  L_bc  = mean( u_pred_boundary² )
  L_tot = L_pde + W_BC * L_bc

Network input: (x, y) — 2 inputs, 1 output.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from level7_multiDomain.src.domains import BaseDomain

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W_BC       = 10.0    # BC penalty weight
N_INTERIOR = 2000    # collocation points per epoch
N_BOUNDARY = 500
N_VAL      = 1000    # validation points
PDE_SCALE  = 1.0     # normalisation (f~O(1) for unit source)


# ── Network ───────────────────────────────────────────────────────────────────

class PoissonNet(nn.Module):
    """Simple fully-connected network for 2-D Poisson."""

    def __init__(self, hidden_sizes: list, activation: str = "tanh"):
        super().__init__()
        act_map = {
            "tanh":  nn.Tanh,
            "relu":  nn.ReLU,
            "gelu":  nn.GELU,
            "sin":   lambda: _Sin(),
        }
        act_cls = act_map.get(activation, nn.Tanh)

        layers = []
        in_dim = 2
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), act_cls()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        # Xavier init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class _Sin(nn.Module):
    def forward(self, x): return torch.sin(x)


# ── Loss ──────────────────────────────────────────────────────────────────────

def pde_loss(model: PoissonNet,
             xy_int: torch.Tensor,
             f_vals: torch.Tensor) -> torch.Tensor:
    """
    -Δu = f   →   loss = mean((d²u/dx² + d²u/dy² + f)²)
    """
    xy_int.requires_grad_(True)
    u    = model(xy_int)                          # [N,1]
    grads = torch.autograd.grad(u, xy_int,
                                grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]   # [N,2]
    ux, uy = grads[:, 0:1], grads[:, 1:2]

    uxx = torch.autograd.grad(ux, xy_int,
                               grad_outputs=torch.ones_like(ux),
                               create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(uy, xy_int,
                               grad_outputs=torch.ones_like(uy),
                               create_graph=True)[0][:, 1:2]

    laplacian = uxx + uyy                         # Δu
    residual  = laplacian + f_vals                # Δu + f = 0  →  -Δu = f
    return (residual**2).mean() / PDE_SCALE


def bc_loss(model: PoissonNet, xy_bc: torch.Tensor) -> torch.Tensor:
    u_bc = model(xy_bc)
    return (u_bc**2).mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train(model:    PoissonNet,
          domain:   BaseDomain,
          n_epochs: int   = 3000,
          lr_start: float = 1e-3,
          lr_end:   float = 1e-5,
          seed:     int   = 42,
          verbose:  bool  = False) -> Dict:
    """
    Train a PoissonNet on the given domain.
    Returns dict with loss_history, val_l2_history, final metrics.
    """
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr_start)

    # Cosine LR annealing
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=lr_end
    )

    # Fixed validation set
    domain_val = domain.__class__(seed=99)
    xy_val_np  = domain_val.sample_interior(N_VAL)
    xy_val     = torch.tensor(xy_val_np, device=DEVICE)

    loss_hist   = []
    val_l2_hist = []

    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        model.train()
        # Fresh collocation every epoch
        xy_int_np = domain.sample_interior(N_INTERIOR)
        xy_bc_np  = domain.sample_boundary(N_BOUNDARY)

        xy_int = torch.tensor(xy_int_np, device=DEVICE, requires_grad=True)
        xy_bc  = torch.tensor(xy_bc_np,  device=DEVICE)
        f_np   = domain.f_rhs(xy_int_np)
        f_ten  = torch.tensor(f_np, device=DEVICE).unsqueeze(1)

        opt.zero_grad()
        l_pde = pde_loss(model, xy_int, f_ten)
        l_bc  = bc_loss(model, xy_bc)
        loss  = l_pde + W_BC * l_bc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        loss_hist.append(float(loss))

        # Validation every 200 epochs — always use PDE residual during
        # training/search so NAS can differentiate architectures even before
        # the model has converged.  Final L2 vs exact is computed separately
        # in run_one() after training completes.
        if epoch % 200 == 0 or epoch == n_epochs:
            model.eval()
            eps = 1e-3
            xy_v = torch.tensor(xy_val_np, device=DEVICE)
            dx   = torch.tensor([[eps, 0.0]], device=DEVICE)
            dy   = torch.tensor([[0.0, eps]], device=DEVICE)
            with torch.no_grad():
                u_c  = model(xy_v)
                u_px = model(xy_v + dx); u_mx = model(xy_v - dx)
                u_py = model(xy_v + dy); u_my = model(xy_v - dy)
            lap  = (u_px + u_mx + u_py + u_my - 4.0 * u_c) / (eps ** 2)
            f_v  = torch.tensor(domain.f_rhs(xy_val_np),
                                device=DEVICE).unsqueeze(1)
            residual = lap + f_v     # should be 0 when -Δu = f
            l2 = float(torch.sqrt(torch.mean(residual ** 2)))

            val_l2_hist.append((epoch, l2))
            if verbose:
                lr_now = sched.get_last_lr()[0]
                print(f"  epoch {epoch:5d}/{n_epochs}  "
                      f"loss={float(loss):.4e}  val={l2:.6f}  "
                      f"lr={lr_now:.2e}")

    elapsed = time.perf_counter() - t0
    model.eval()
    return {
        "loss_history":    loss_hist,
        "val_l2_history":  val_l2_hist,
        "final_val":       val_l2_hist[-1][1] if val_l2_hist else 999.0,
        "elapsed_s":       elapsed,
    }


# ── Eval grid ─────────────────────────────────────────────────────────────────

def eval_on_grid(model:  PoissonNet,
                 domain: BaseDomain,
                 n:      int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate u(x,y) on a uniform grid clipped to domain.
    Returns (xx, yy, uu) all shape [n, n] with NaN outside domain.
    """
    lo, hi = domain.bbox()
    xs = np.linspace(lo[0], hi[0], n, dtype=np.float32)
    ys = np.linspace(lo[1], hi[1], n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")   # [n,n]
    xy_flat = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    inside = domain.is_inside(xy_flat)

    model_device = next(model.parameters()).device
    xy_t = torch.tensor(xy_flat, device=model_device)
    with torch.no_grad():
        u_flat = model(xy_t).squeeze().cpu().numpy()

    u_flat[~inside] = np.nan
    uu = u_flat.reshape(n, n)
    return xx, yy, uu
