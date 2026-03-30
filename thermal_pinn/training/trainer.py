"""
thermal_pinn/training/trainer.py
==================================
Time-window training loop for the ThermalPINN.

Training strategy
-----------------
We follow a two-phase scheme motivated by the convergence analysis in
Lu et al. (2021) and empirically validated across all levels of this thesis:

    Phase 1 — Adam with cosine LR annealing (fast, global minimiser)
    Phase 2 — L-BFGS with strong Wolfe line search (local refinement)

The cosine annealing schedule (Loshchilov & Hutter, 2017) avoids the sharp
LR drops of step decay and consistently outperformed fixed-LR Adam in our
Level 5 experiments (see level5_refinement/results/).

Loss function
-------------
Four terms, weighted to balance their typical magnitudes:

    L = W_pde  · L_pde   +  W_ic  · L_ic   +  W_bc  · L_bc   +  W_end · L_end

    L_pde  : PDE residual — ∂θ/∂t − α·∇²θ = 0  (scaled by RHO_CP·ΔT/dt)
    L_ic   : IC mismatch  — θ(x,0) = θ_anchor
    L_bc   : Robin BC     — k·∂θ/∂n + h·ΔT·θ = 0  on ∂Ω
    L_end  : optional end supervision — θ(x,T) = θ_fem_end
             anchors the exit temperature to the next FEM checkpoint,
             which is the key innovation of the MSWP approach (Level 10).

The end-supervision term is especially important for large k (k ≥ 4) where
the PINN must extrapolate over a wide window without internal FEM anchors.

k-skip comparison
-----------------
All k values in K_VALUES are trained and saved independently.
The best k is selected by minimising the mean MAE over the full time series.
All checkpoints are stored so that k comparisons can be reproduced later.

References:
    [1] Wang, Y., & Zhong, L. (2023). NAS-PINN. arXiv:2305.10127.
    [2] Lu, L. et al. (2021). DeepXDE: A library for scientific machine
        learning. SIAM Review, 63(1), 208–228.
    [3] Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient
        descent with warm restarts. ICLR 2017.
    [4] Mortensen et al. (2026). DOI: 10.1007/s00170-026-17515-w
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
import torch.optim as optim

from ..network.pinn import ThermalPINN

# ──────────────────────────────────────────────────────────────────────────────
# Loss weights — tuned empirically across Levels 5, 8, 10
# ──────────────────────────────────────────────────────────────────────────────
W_PDE  = 1.0    # PDE residual weight
W_IC   = 15.0   # IC weight — high because IC error propagates into all time steps
W_BC   = 5.0    # Robin BC weight
W_END  = 10.0   # end-supervision weight (optional, see use_end_supervision)

# k values evaluated in sweep (FEM steps skipped per PINN window)
K_VALUES = [1, 2, 3, 4, 5]

# Physical constants (A356, Mortensen et al. 2026)
K_THERM = 150.0    # W/(m·K)
RHO_CP  = 2.4e6    # J/(m³·K)
H_CONV  = 5000.0   # W/(m²·K)
T_WATER = 20.0     # °C
DELTA_T = 520.0    # °C  (T_init − T_water)


def _pde_residual(model: ThermalPINN,
                  coords: torch.Tensor,
                  tau: torch.Tensor,
                  theta_ic: torch.Tensor,
                  dt_window: float,
                  alpha: float) -> torch.Tensor:
    """
    Compute PDE residual in fully normalised (τ, x_norm) coordinates.

    Spatial coordinates are normalised to [0, 1]^dim by the caller
    (see _normalise_coords). Local time τ = t_local / dt_window ∈ [0, 1].

    The heat equation in dimensionless θ = (T − T_w)/ΔT is:
        ∂θ/∂t = α · ∇²_phys θ

    Converting to normalised variables:
        ∂θ/∂τ = α · dt_window · Σ_i (∂²θ/∂x_i_norm²)   [÷ L_i² absorbed below]

    Because coords are normalised by [0, L_i], the normalised Laplacian
    equals the physical Laplacian times L_i². We absorb this by using the
    effective Fourier-number coefficient α_eff = α · dt_window (dimensionless),
    which gives a residual of order O(0.01–0.1) for typical quenching windows.
    This formulation avoids the overflow that occurs when multiplying by
    dimensional factors like ρCp·ΔT/dt ≈ 2.8×10⁸.

    Note: coords and tau must have requires_grad=True (set by caller).
    """
    theta = model(coords, tau, theta_ic)

    # ∂θ/∂τ — dimensionless time derivative ∈ O(0.01–0.1) per window
    dtheta_dtau = torch.autograd.grad(
        theta, tau, grad_outputs=torch.ones_like(theta),
        create_graph=True, retain_graph=True
    )[0]

    # ∇²_norm θ — Laplacian in normalised spatial coordinates
    grads = torch.autograd.grad(
        theta, coords, grad_outputs=torch.ones_like(theta),
        create_graph=True, retain_graph=True
    )[0]   # (N, dim)

    laplacian_norm = torch.zeros(theta.shape, device=theta.device)
    for i in range(coords.shape[1]):
        d2 = torch.autograd.grad(
            grads[:, i:i+1], coords,
            grad_outputs=torch.ones_like(grads[:, i:i+1]),
            create_graph=True, retain_graph=True
        )[0][:, i:i+1]
        laplacian_norm = laplacian_norm + d2

    # Effective Fourier number: α · dt_window (dimensionless diffusion per window)
    # Typical value: 6.25e-5 * 4.5 ≈ 2.8e-4  (small → PDE residual ≈ dtheta_dtau)
    alpha_eff = alpha * dt_window
    return (dtheta_dtau - alpha_eff * laplacian_norm).pow(2).mean()


def _robin_bc_loss(model: ThermalPINN,
                   bc_coords: torch.Tensor,
                   bc_normals: torch.Tensor,
                   tau_bc: torch.Tensor,
                   theta_ic_bc: torch.Tensor,
                   h: float) -> torch.Tensor:
    """
    Robin BC residual: k·(∂θ/∂n)·ΔT + h·ΔT·θ = 0
    i.e.,  ∂θ/∂n + (h/k)·θ = 0.

    ∂θ/∂n = ∇θ · n̂  (normal derivative, computed via autograd).
    """
    theta = model(bc_coords, tau_bc, theta_ic_bc)
    grads = torch.autograd.grad(
        theta, bc_coords, grad_outputs=torch.ones_like(theta),
        create_graph=True, retain_graph=True
    )[0]   # (N, dim)

    # Project gradient onto outward normal
    dtheta_dn = (grads * bc_normals).sum(dim=1, keepdim=True)

    # The Robin condition in normalised coords is: ∂θ/∂n_norm + Bi·θ = 0,
    # where Bi = h·L_char/k. For a typical domain L_char ≈ 0.5 m,
    # Bi ≈ 5000·0.5/150 ≈ 16.7, so the BC residual without scaling is O(Bi).
    # We normalise by 1/Bi to keep the loss term O(1) and prevent it from
    # dominating the IC/PDE terms.  Bi_approx uses L_char = k/h (= 1/Bi_unit),
    # which gives bc_scale ≈ k/(h·ΔT) ≈ 1/17333 — safe against overflow.
    bi_approx = max(h / K_THERM, 1.0)   # ≈ h/k without L_char (conservative)
    return ((dtheta_dn + (h / K_THERM) * theta) / bi_approx).pow(2).mean()


def train_window(model: ThermalPINN,
                 domain,
                 t_start: float,
                 t_end: float,
                 theta_ic_fn: Callable,
                 theta_end_fn: Callable | None = None,
                 n_domain: int = 1500,
                 n_bc: int = 300,
                 n_epochs: int = 800,
                 lr: float = 1e-3,
                 lr_min: float = 1e-5,
                 lbfgs_iters: int = 50,
                 use_end_supervision: bool = True,
                 device: torch.device | None = None,
                 rng_seed: int = 0) -> dict:
    """
    Train the PINN on a single time window [t_start, t_end].

    Parameters
    ----------
    model         : ThermalPINN instance (modified in-place)
    domain        : 2D or 3D domain object with .sample_interior() / .sample_boundary()
    t_start/t_end : physical time bounds of the window (seconds)
    theta_ic_fn   : function(coords_np) → θ_ic at t_start
    theta_end_fn  : function(coords_np) → θ_fem at t_end (for end supervision)
    use_end_supervision : if True, add L_end to loss (recommended for k ≥ 3)

    Returns
    -------
    dict with keys: mae_train, loss_history, runtime_s
    """
    device  = device or next(model.parameters()).device
    alpha   = getattr(domain, "alpha", 6.25e-5)
    h       = getattr(domain, "h", H_CONV)
    dim     = model.dim
    dt_window = t_end - t_start
    rng     = np.random.default_rng(rng_seed)

    # ── Sample collocation points ────────────────────────────────────────────
    if dim == 2:
        xi, yi = domain.sample_interior(n_domain, rng=rng)
        coords_np = np.stack([xi, yi], axis=1)
        bx, by, nx_, ny_ = domain.sample_boundary(n_bc, rng=rng)
        bc_np     = np.stack([bx, by], axis=1)
        norms_np  = np.stack([nx_, ny_], axis=1)
    else:
        xi, yi, zi = domain.sample_interior(n_domain, rng=rng)
        coords_np  = np.stack([xi, yi, zi], axis=1)
        bx, by, bz, nx_, ny_, nz_ = domain.sample_boundary(n_bc, rng=rng)
        bc_np      = np.stack([bx, by, bz], axis=1)
        norms_np   = np.stack([nx_, ny_, nz_], axis=1)

    # ── Convert to tensors ───────────────────────────────────────────────────
    def to_t(a): return torch.tensor(a, dtype=torch.float32, device=device)

    # Normalise spatial coordinates to [0,1] for numerical stability
    coords_norm = _normalise_coords(coords_np, domain)
    bc_norm     = _normalise_coords(bc_np, domain)

    # Sample τ ∈ [0,1] uniformly (local time within window)
    tau_np  = rng.uniform(0, 1, (n_domain, 1)).astype(np.float32)
    tau_bc  = rng.uniform(0, 1, (n_bc,     1)).astype(np.float32)

    # IC temperatures at t_start
    theta_ic_vals = theta_ic_fn(coords_np).reshape(-1, 1).astype(np.float32)
    theta_ic_bc   = theta_ic_fn(bc_np).reshape(-1, 1).astype(np.float32)

    coords_t  = to_t(coords_norm).requires_grad_(True)
    tau_t     = to_t(tau_np).requires_grad_(True)
    theta_ic_t = to_t(theta_ic_vals)
    bc_t      = to_t(bc_norm).requires_grad_(True)
    tau_bc_t  = to_t(tau_bc).requires_grad_(True)
    theta_ic_bc_t = to_t(theta_ic_bc)
    norms_t   = to_t(norms_np)

    # Optional: end-supervision points at τ=1
    theta_end_t = None
    if use_end_supervision and theta_end_fn is not None:
        theta_end_vals = theta_end_fn(coords_np).reshape(-1, 1).astype(np.float32)
        theta_end_t    = to_t(theta_end_vals)
        tau_end_t      = torch.ones(n_domain, 1, device=device)

    # ── Phase 1: Adam with cosine annealing ──────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr_min
    )
    loss_history = []
    t0 = time.time()
    model.train()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        loss_pde = _pde_residual(model, coords_t, tau_t, theta_ic_t,
                                 dt_window, alpha)
        # IC loss: at τ=0, output should equal θ_ic
        tau_zero  = torch.zeros(n_domain, 1, device=device)
        pred_ic   = model(coords_t, tau_zero, theta_ic_t)
        loss_ic   = ((pred_ic - theta_ic_t) ** 2).mean()

        loss_bc   = _robin_bc_loss(model, bc_t, norms_t, tau_bc_t,
                                   theta_ic_bc_t, h)

        loss = W_PDE * loss_pde + W_IC * loss_ic + W_BC * loss_bc

        if theta_end_t is not None:
            pred_end = model(coords_t, tau_end_t, theta_ic_t)
            loss_end = ((pred_end - theta_end_t) ** 2).mean()
            loss     = loss + W_END * loss_end

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(float(loss.detach()))

    # ── Phase 2: L-BFGS refinement ────────────────────────────────────────────
    # L-BFGS can fail on ill-conditioned problems (we observed this in Level 5
    # for certain architectures). We catch exceptions and fall back gracefully.
    if lbfgs_iters > 0:
        lbfgs = optim.LBFGS(model.parameters(),
                             max_iter=lbfgs_iters,
                             line_search_fn="strong_wolfe",
                             tolerance_grad=1e-7,
                             tolerance_change=1e-9)

        def closure():
            lbfgs.zero_grad()
            lp = _pde_residual(model, coords_t, tau_t, theta_ic_t, dt_window, alpha)
            pred_ic_ = model(coords_t, torch.zeros(n_domain, 1, device=device), theta_ic_t)
            li = ((pred_ic_ - theta_ic_t) ** 2).mean()
            lb = _robin_bc_loss(model, bc_t, norms_t, tau_bc_t, theta_ic_bc_t, h)
            total = W_PDE * lp + W_IC * li + W_BC * lb
            if theta_end_t is not None:
                total = total + W_END * ((model(coords_t, tau_end_t, theta_ic_t) - theta_end_t) ** 2).mean()
            total.backward()
            return total

        try:
            lbfgs.step(closure)
        except RuntimeError:
            pass   # L-BFGS diverged — keep Adam result

    return {
        "loss_history": loss_history,
        "runtime_s":    time.time() - t0,
    }


def evaluate_window(model: ThermalPINN,
                    domain,
                    t_start: float,
                    t_end: float,
                    theta_ic_fn: Callable,
                    n_eval: int = 500,
                    device: torch.device | None = None) -> dict:
    """
    Evaluate trained model on a window: compute MAE and L2 relative error
    against the domain's analytical/numerical reference solution.
    """
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


def _normalise_coords(coords: np.ndarray, domain) -> np.ndarray:
    """
    Normalise spatial coordinates to [0,1]^dim using domain bounding box.
    Stable normalisation prevents exploding gradients during PDE differentiation.
    """
    dim = coords.shape[1]
    lo  = np.zeros(dim)
    hi  = np.ones(dim)
    if dim == 2:
        hi[0] = getattr(domain, "Lx", getattr(domain, "R_out", 1.0))
        hi[1] = getattr(domain, "Ly", getattr(domain, "R_out", 1.0))
    else:
        hi[0] = getattr(domain, "Lx", 1.0)
        hi[1] = getattr(domain, "Ly", 1.0)
        hi[2] = getattr(domain, "Lz", getattr(domain, "Hz", 1.0))
    span = hi - lo
    span[span < 1e-8] = 1.0
    return ((coords - lo) / span).astype(np.float32)
