"""
poisson_aux_loss.py — Quasi-Static Poisson Auxiliary Loss
==========================================================
Physics background (Fizik temeli):

  Quenching heat equation:
      ρCp · ∂T/∂t  =  k · ΔT(x,y,t)

  At any fixed time slice t = t₀, rearranged as a POISSON equation:
      ΔT(x,y,t₀)  =  (ρCp / k) · ∂T/∂t|_{t₀}
                      ─────────────────────────
                      quasi-static source  f(x,y,t₀)

  Residual (should be zero):
      R_poisson(x,y,t₀)  =  ρCp·∂T/∂t  -  k·ΔT   [W/m³]

  Normalization:
      PDE_SCALE = ρCp · ΔT_ref / t_end ≈ 4.16e7 W/m³
      Same as used in quenching make_loss_fn() → L_poisson is O(1)
      and directly comparable to L_heat.

Why does this help beyond the standard heat-equation loss?
  The standard PDE loss uses RANDOM (x,y,t) collocation points, distributing
  gradient signal uniformly across time.  The Poisson auxiliary loss samples
  a DENSE SPATIAL GRID at K_SLICES fixed critical time points, giving the
  network a concentrated spatial correction signal at those moments.
"""

import torch
from typing import List

from src.config import (
    DEVICE, K_THERMAL, RHO_CP, X_MAX, Y_MAX,
    T_INIT, T_WATER_INIT, T_END,
)

# Normalization scale — must match quenching make_loss_fn() PDE_SCALE
# PDE_SCALE = RHO_CP * (T_init - T_water) / T_end ≈ 4.16e7  [W/m³]
_PDE_SCALE = RHO_CP * (T_INIT - T_WATER_INIT) / T_END
# Legacy ratio kept for reference only
_RHO_CP_OVER_K = RHO_CP / K_THERMAL   # ≈ 16000  [s/m²]

# Default critical time slices (seconds) chosen to cover all cooling regimes:
#   0.5s → film boiling / fast early cooling
#   2.0s → nucleate boiling peak
#   5.0s → nucleate→convection transition
#  10.0s → forced convection plateau
#  20.0s → late slow cooling
DEFAULT_T_SLICES = [0.5, 2.0, 5.0, 10.0, 20.0]
N_SPATIAL        = 2000   # collocation points per time slice


def _sample_spatial(n: int) -> torch.Tensor:
    """
    Uniform random spatial sample in [0, X_MAX] × [0, Y_MAX].
    Returns [n, 2] tensor (x, y).
    """
    x = torch.rand(n, 1, device=DEVICE) * X_MAX
    y = torch.rand(n, 1, device=DEVICE) * Y_MAX
    return torch.cat([x, y], dim=1)   # [n, 2]


def poisson_slice_residual(model: torch.nn.Module,
                            t_val: float,
                            n_spatial: int = N_SPATIAL
                            ) -> torch.Tensor:
    """
    Compute normalized Poisson residual at one fixed time slice t = t_val.

    R(x,y) = (ρCp·∂T/∂t  -  k·ΔT) / PDE_SCALE    [dimensionless, O(1)]
           = heat_equation_residual / PDE_SCALE

    Same formula as the standard heat-equation loss in quenching.py,
    but evaluated at a FIXED time with DENSE spatial sampling.

    Returns:  residual tensor  [n_spatial, 1]

    Input ordering: quenching model takes  (t, x, y)  — coords[:, 0] = t.
    """
    xy    = _sample_spatial(n_spatial)                           # [n, 2]
    t_col = torch.full((n_spatial, 1), t_val, device=DEVICE)    # [n, 1]

    # Model input order:  (t, x, y)
    coords = torch.cat([t_col, xy], dim=1).requires_grad_(True) # [n, 3]

    T = model(coords)   # [n, 1]

    # ── First-order gradients: index 0=t, 1=x, 2=y ───────────────────────────
    grads = torch.autograd.grad(
        T, coords,
        grad_outputs=torch.ones_like(T),
        create_graph=True,
    )[0]                     # [n, 3]

    dT_dt = grads[:, 0:1]   # ∂T/∂t  [n, 1]
    dT_dx = grads[:, 1:2]   # ∂T/∂x  [n, 1]
    dT_dy = grads[:, 2:3]   # ∂T/∂y  [n, 1]

    # ── Second-order spatial gradients ∂²T/∂x² and ∂²T/∂y² ───────────────────
    dT_dx2 = torch.autograd.grad(
        dT_dx, coords,
        grad_outputs=torch.ones_like(dT_dx),
        create_graph=True,
    )[0][:, 1:2]             # gradient of (∂T/∂x) w.r.t. x  [n, 1]

    dT_dy2 = torch.autograd.grad(
        dT_dy, coords,
        grad_outputs=torch.ones_like(dT_dy),
        create_graph=True,
    )[0][:, 2:3]             # gradient of (∂T/∂y) w.r.t. y  [n, 1]

    # ── Heat-equation residual form: ρCp·∂T/∂t - k·ΔT  [W/m³] ─────────────────
    # Same formula as src/physics_model.py → heat_equation_residual()
    # Normalized by PDE_SCALE so L_poisson is O(1), comparable to L_heat.
    laplacian = dT_dx2 + dT_dy2
    r_raw     = RHO_CP * dT_dt - K_THERMAL * laplacian  # [W/m³]
    residual  = r_raw / _PDE_SCALE                       # dimensionless O(1)

    return residual


def poisson_auxiliary_loss(model: torch.nn.Module,
                            t_slices:  List[float] = DEFAULT_T_SLICES,
                            n_spatial: int         = N_SPATIAL,
                            ) -> torch.Tensor:
    """
    Normalized Poisson auxiliary loss aggregated over all time slices.

    L_poisson = (1/K) Σ_k  mean( R_normalized(x,y,t_k)² )

    L_poisson is O(1), same scale as L_heat → λ_p=0.1 adds 10% Poisson penalty.
    """
    total = torch.tensor(0.0, device=DEVICE, requires_grad=False)
    for t_k in t_slices:
        r = poisson_slice_residual(model, t_k, n_spatial=n_spatial)
        total = total + (r ** 2).mean()
    return total / len(t_slices)
