"""
thermal_pinn/training/sa_trainer.py
=====================================
Self-Adaptive loss weights + Fourier PINN trainer (v2).

Self-Adaptive Weighting
-----------------------
Instead of fixed scalars W_PDE / W_BC / W_END, we learn three positive
scalar parameters λ_pde, λ_bc, λ_end via a softplus constraint:

    λ = softplus(w) = log(1 + exp(w))    (always positive)

These parameters are updated simultaneously with the network weights but
with a *lower* learning rate (lr_lambda = lr_model / 10) to ensure the loss
landscape converges before the weights shift significantly.

The IC loss is intentionally omitted here because FourierPINN uses the
IC-consistent output (θ_ic + τ·δθ), which satisfies the IC exactly at τ=0
by construction — the IC residual is identically zero.

Training schedule (same as trainer.py):
    Phase 1: Adam with cosine annealing
    Phase 2: L-BFGS with strong Wolfe line search

Level 9 insight (thesis-pinn-fem/level9_sa_fourier):
    Learned weights converge to λ_bc ≈ 9.8, λ_ic ≈ 9.1 — boundary and IC
    dominate (physically correct for transient quenching). In our v2 setup
    where IC is exact, λ_bc and λ_end will absorb this weight naturally.

References:
    [1] Anagnostopoulos et al. (2024). Residual-based attention in physics-
        informed neural networks. arXiv:2302.06428.
    [2] Mortensen et al. (2026). DOI: 10.1007/s00170-026-17515-w
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..network.fourier_pinn import FourierPINN

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants
# ──────────────────────────────────────────────────────────────────────────────
K_THERM = 150.0    # W/(m·K)  — A356 thermal conductivity
H_CONV  = 5000.0   # W/(m²·K) — quench heat transfer coefficient
T_WATER = 20.0     # °C
DELTA_T = 520.0    # °C  (T_init − T_water)

# ──────────────────────────────────────────────────────────────────────────────
# SA initial values (match trainer.py fixed weights for fair comparison)
# ──────────────────────────────────────────────────────────────────────────────
SA_INIT_PDE = 1.0
SA_INIT_BC  = 5.0
SA_INIT_END = 10.0


def _inv_softplus(x: float) -> float:
    """Inverse of softplus: w such that softplus(w) = x."""
    return math.log(math.exp(x) - 1.0)


class SelfAdaptiveWeights(nn.Module):
    """
    Three learnable positive loss-weight scalars via softplus.

    Initialised so that softplus(w) equals the given initial values,
    providing a warm-start that matches the fixed weights of trainer.py.
    """

    def __init__(self,
                 init_pde: float = SA_INIT_PDE,
                 init_bc:  float = SA_INIT_BC,
                 init_end: float = SA_INIT_END):
        super().__init__()
        self.w_pde = nn.Parameter(torch.tensor(_inv_softplus(init_pde)))
        self.w_bc  = nn.Parameter(torch.tensor(_inv_softplus(init_bc)))
        self.w_end = nn.Parameter(torch.tensor(_inv_softplus(init_end)))

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (λ_pde, λ_bc, λ_end) — all positive."""
        return (
            F.softplus(self.w_pde),
            F.softplus(self.w_bc),
            F.softplus(self.w_end),
        )

    def values(self) -> dict[str, float]:
        lam = self.forward()
        return {k: float(v.detach()) for k, v in
                zip(["lam_pde", "lam_bc", "lam_end"], lam)}


# ──────────────────────────────────────────────────────────────────────────────
# PDE and BC residuals (identical numerics to trainer.py)
# ──────────────────────────────────────────────────────────────────────────────

def _pde_residual(model: FourierPINN,
                  coords: torch.Tensor,
                  tau: torch.Tensor,
                  theta_ic: torch.Tensor,
                  dt_window: float,
                  alpha: float) -> torch.Tensor:
    """PDE residual in normalised coordinates (same formulation as trainer.py)."""
    theta = model(coords, tau, theta_ic)

    dtheta_dtau = torch.autograd.grad(
        theta, tau, grad_outputs=torch.ones_like(theta),
        create_graph=True, retain_graph=True
    )[0]

    grads = torch.autograd.grad(
        theta, coords, grad_outputs=torch.ones_like(theta),
        create_graph=True, retain_graph=True
    )[0]

    laplacian_norm = torch.zeros_like(theta)
    for i in range(coords.shape[1]):
        d2 = torch.autograd.grad(
            grads[:, i:i+1], coords,
            grad_outputs=torch.ones_like(grads[:, i:i+1]),
            create_graph=True, retain_graph=True
        )[0][:, i:i+1]
        laplacian_norm = laplacian_norm + d2

    alpha_eff = alpha * dt_window
    return (dtheta_dtau - alpha_eff * laplacian_norm).pow(2).mean()


def _robin_bc_loss(model: FourierPINN,
                   bc_coords: torch.Tensor,
                   bc_normals: torch.Tensor,
                   tau_bc: torch.Tensor,
                   theta_ic_bc: torch.Tensor,
                   h: float) -> torch.Tensor:
    """Robin BC residual (same formulation as trainer.py)."""
    theta = model(bc_coords, tau_bc, theta_ic_bc)
    grads = torch.autograd.grad(
        theta, bc_coords, grad_outputs=torch.ones_like(theta),
        create_graph=True, retain_graph=True
    )[0]

    dtheta_dn = (grads * bc_normals).sum(dim=1, keepdim=True)
    bi_approx = max(h / K_THERM, 1.0)
    return ((dtheta_dn + (h / K_THERM) * theta) / bi_approx).pow(2).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_coords(coords: np.ndarray, domain) -> np.ndarray:
    """Normalise spatial coordinates to [0,1]^dim (mirrors trainer.py)."""
    dim = coords.shape[1]
    hi  = np.ones(dim)
    if dim == 2:
        hi[0] = getattr(domain, "Lx", getattr(domain, "R_out", 1.0))
        hi[1] = getattr(domain, "Ly", getattr(domain, "R_out", 1.0))
    else:
        hi[0] = getattr(domain, "Lx", 1.0)
        hi[1] = getattr(domain, "Ly", 1.0)
        hi[2] = getattr(domain, "Lz", getattr(domain, "Hz", 1.0))
    span = hi.copy()
    span[span < 1e-8] = 1.0
    return (coords / span).astype(np.float32)


def train_window_sa(model: FourierPINN,
                    sa_weights: SelfAdaptiveWeights,
                    domain,
                    t_start: float,
                    t_end: float,
                    theta_ic_fn: Callable,
                    theta_end_fn: Callable | None = None,
                    n_domain: int = 1500,
                    n_bc: int = 300,
                    n_epochs: int = 800,
                    lr: float = 1e-3,
                    lr_lambda: float = 1e-4,
                    lr_min: float = 1e-5,
                    lbfgs_iters: int = 50,
                    use_end_supervision: bool = True,
                    device: torch.device | None = None,
                    rng_seed: int = 0) -> dict:
    """
    Train FourierPINN with self-adaptive loss weights on one time window.

    Two separate Adam optimisers:
        - opt_model  : model parameters,   lr = lr
        - opt_lambda : SA weight parameters, lr = lr_lambda (= lr/10)

    Both share the same cosine annealing schedule (T_max = n_epochs).

    Returns
    -------
    dict with keys: loss_history, final_weights, runtime_s
    """
    device    = device or next(model.parameters()).device
    alpha     = getattr(domain, "alpha", 6.25e-5)
    h         = getattr(domain, "h", H_CONV)
    dim       = model.dim
    dt_window = t_end - t_start
    rng       = np.random.default_rng(rng_seed)

    # ── Sample collocation points ────────────────────────────────────────────
    if dim == 2:
        xi, yi = domain.sample_interior(n_domain, rng=rng)
        coords_np = np.stack([xi, yi], axis=1)
        bx, by, nx_, ny_ = domain.sample_boundary(n_bc, rng=rng)
        bc_np    = np.stack([bx, by], axis=1)
        norms_np = np.stack([nx_, ny_], axis=1)
    else:
        xi, yi, zi = domain.sample_interior(n_domain, rng=rng)
        coords_np  = np.stack([xi, yi, zi], axis=1)
        bx, by, bz, nx_, ny_, nz_ = domain.sample_boundary(n_bc, rng=rng)
        bc_np    = np.stack([bx, by, bz], axis=1)
        norms_np = np.stack([nx_, ny_, nz_], axis=1)

    # ── Tensors ──────────────────────────────────────────────────────────────
    def to_t(a): return torch.tensor(a, dtype=torch.float32, device=device)

    coords_norm = _normalise_coords(coords_np, domain)
    bc_norm     = _normalise_coords(bc_np, domain)

    tau_np  = rng.uniform(0, 1, (n_domain, 1)).astype(np.float32)
    tau_bc  = rng.uniform(0, 1, (n_bc,     1)).astype(np.float32)

    theta_ic_vals = theta_ic_fn(coords_np).reshape(-1, 1).astype(np.float32)
    theta_ic_bc   = theta_ic_fn(bc_np).reshape(-1, 1).astype(np.float32)

    coords_t      = to_t(coords_norm).requires_grad_(True)
    tau_t         = to_t(tau_np).requires_grad_(True)
    theta_ic_t    = to_t(theta_ic_vals)
    bc_t          = to_t(bc_norm).requires_grad_(True)
    tau_bc_t      = to_t(tau_bc).requires_grad_(True)
    theta_ic_bc_t = to_t(theta_ic_bc)
    norms_t       = to_t(norms_np)

    theta_end_t   = None
    tau_end_t     = None
    if use_end_supervision and theta_end_fn is not None:
        theta_end_t = to_t(theta_end_fn(coords_np).reshape(-1, 1).astype(np.float32))
        tau_end_t   = torch.ones(n_domain, 1, device=device)

    # ── Optimisers — dual learning rates ─────────────────────────────────────
    opt_model  = optim.Adam(model.parameters(), lr=lr)
    opt_lambda = optim.Adam(sa_weights.parameters(), lr=lr_lambda)
    sched_model  = optim.lr_scheduler.CosineAnnealingLR(opt_model,  T_max=n_epochs, eta_min=lr_min)
    sched_lambda = optim.lr_scheduler.CosineAnnealingLR(opt_lambda, T_max=n_epochs, eta_min=lr_lambda * 0.1)

    loss_history = []
    t0 = time.time()
    model.train()
    sa_weights.train()

    # ── Phase 1: Adam ────────────────────────────────────────────────────────
    for epoch in range(n_epochs):
        opt_model.zero_grad()
        opt_lambda.zero_grad()

        lam_pde, lam_bc, lam_end = sa_weights()

        loss_pde = _pde_residual(model, coords_t, tau_t, theta_ic_t,
                                 dt_window, alpha)
        loss_bc  = _robin_bc_loss(model, bc_t, norms_t, tau_bc_t,
                                  theta_ic_bc_t, h)

        loss = lam_pde * loss_pde + lam_bc * loss_bc

        if theta_end_t is not None:
            pred_end = model(coords_t, tau_end_t, theta_ic_t)
            loss_end = ((pred_end - theta_end_t) ** 2).mean()
            loss     = loss + lam_end * loss_end

        loss.backward()
        opt_model.step()
        opt_lambda.step()
        sched_model.step()
        sched_lambda.step()
        loss_history.append(float(loss.detach()))

    # ── Phase 2: L-BFGS on model params only (fixed λ) ───────────────────────
    if lbfgs_iters > 0:
        # Freeze SA weights during L-BFGS for stability
        lam_pde_f, lam_bc_f, lam_end_f = (float(v.detach()) for v in sa_weights())

        lbfgs = optim.LBFGS(model.parameters(),
                             max_iter=lbfgs_iters,
                             line_search_fn="strong_wolfe",
                             tolerance_grad=1e-7,
                             tolerance_change=1e-9)

        def closure():
            lbfgs.zero_grad()
            lp = _pde_residual(model, coords_t, tau_t, theta_ic_t,
                               dt_window, alpha)
            lb = _robin_bc_loss(model, bc_t, norms_t, tau_bc_t,
                                theta_ic_bc_t, h)
            total = lam_pde_f * lp + lam_bc_f * lb
            if theta_end_t is not None:
                le = ((model(coords_t, tau_end_t, theta_ic_t) - theta_end_t) ** 2).mean()
                total = total + lam_end_f * le
            total.backward()
            return total

        try:
            lbfgs.step(closure)
        except RuntimeError:
            pass

    return {
        "loss_history":   loss_history,
        "final_weights":  sa_weights.values(),
        "runtime_s":      time.time() - t0,
    }


def evaluate_window_sa(model: FourierPINN,
                        domain,
                        t_start: float,
                        t_end: float,
                        theta_ic_fn: Callable,
                        n_eval: int = 500,
                        device: torch.device | None = None) -> dict:
    """Evaluate FourierPINN on a window (same API as trainer.evaluate_window)."""
    device = device or next(model.parameters()).device
    dim    = model.dim
    rng    = np.random.default_rng(42)

    if dim == 2:
        xi, yi = domain.sample_interior(n_eval, rng=rng)
        coords_np = np.stack([xi, yi], axis=1)
        T_ref = domain.T(xi, yi, t_end)
    else:
        xi, yi, zi = domain.sample_interior(n_eval, rng=rng)
        coords_np  = np.stack([xi, yi, zi], axis=1)
        T_ref = domain.T(xi, yi, zi, t_end)

    coords_norm = _normalise_coords(coords_np, domain)
    theta_ic    = theta_ic_fn(coords_np).reshape(-1, 1).astype(np.float32)
    tau_one     = np.ones((n_eval, 1), dtype=np.float32)

    def to_t(a): return torch.tensor(a, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        theta_pred = model(to_t(coords_norm),
                           to_t(tau_one),
                           to_t(theta_ic)).cpu().numpy().flatten()

    T_pred = T_WATER + DELTA_T * theta_pred
    valid  = np.isfinite(T_ref)
    mae    = float(np.mean(np.abs(T_pred[valid] - T_ref[valid])))
    l2     = float(np.linalg.norm(T_pred[valid] - T_ref[valid]) /
                   (np.linalg.norm(T_ref[valid] - T_WATER) + 1e-10))

    return {"mae_C": mae, "l2_rel": l2,
            "T_pred": T_pred, "T_ref": T_ref, "coords": coords_np}
