"""
batch_sampler.py — Deterministic Fixed-Batch Sampler for L-BFGS
================================================================
L-BFGS requires a FIXED batch — the closure must return the same loss
for the same weights. This module generates one frozen batch at startup
and reuses it for every closure evaluation.

Parameters validated across:
  NAS-PINNS1 (cfg.py): col=2048, ic=512, bc=512
  FEM-PINNS (Poisson):  col=1000, bc=250
  NAS-PINNS2 (quench):  col=1024, ic=256, bc=256
→ Level 5 uses NAS-PINNS1 values as the largest, best-validated set.
"""

import torch
from typing import Dict

from src.config import DEVICE, X_MAX, Y_MAX, T_END, T_INIT


# ── Default batch sizes (from NAS-PINNS1 cfg.py) ─────────────────────────────
COL_POINTS = 2048   # collocation / physics residual
IC_POINTS  = 512    # initial condition  (t=0)
BC_POINTS  = 512    # boundary condition (4 sides, 128 each)


def sample_fixed_batch(
    col_points: int = COL_POINTS,
    ic_points:  int = IC_POINTS,
    bc_points:  int = BC_POINTS,
    seed:       int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Generate one deterministic batch and freeze it.

    Returns dict with keys:
      "domain"   — (col_points, 3)  [t, x, y]  requires_grad=True
      "boundary" — (bc_points,  3)  [t, x, y]  requires_grad=True
      "normals"  — (bc_points,  2)  [nx, ny]   outward unit normals
      "initial"  — (ic_points,  3)  [0, x, y]
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    def _rand(n, lo, hi):
        return (torch.rand(n, 1, generator=gen) * (hi - lo) + lo).to(DEVICE)

    # ── Domain (collocation) ──────────────────────────────────────────────────
    t_d = _rand(col_points, 0.0, T_END)
    x_d = _rand(col_points, 0.0, X_MAX)
    y_d = _rand(col_points, 0.0, Y_MAX)
    domain = torch.cat([t_d, x_d, y_d], dim=1).requires_grad_(True)

    # ── Boundary (4 sides, bc_points//4 each) ────────────────────────────────
    n = bc_points // 4
    t_b = _rand(bc_points, 0.0, T_END)

    x_left  = torch.zeros(n, 1, device=DEVICE)
    y_left  = _rand(n, 0.0, Y_MAX)
    n_left  = torch.tensor([[-1.0, 0.0]]).expand(n, -1).to(DEVICE)

    x_right = torch.full((n, 1), X_MAX, device=DEVICE)
    y_right = _rand(n, 0.0, Y_MAX)
    n_right = torch.tensor([[+1.0, 0.0]]).expand(n, -1).to(DEVICE)

    x_bot   = _rand(n, 0.0, X_MAX)
    y_bot   = torch.zeros(n, 1, device=DEVICE)
    n_bot   = torch.tensor([[0.0, -1.0]]).expand(n, -1).to(DEVICE)

    x_top   = _rand(n, 0.0, X_MAX)
    y_top   = torch.full((n, 1), Y_MAX, device=DEVICE)
    n_top   = torch.tensor([[0.0, +1.0]]).expand(n, -1).to(DEVICE)

    x_b = torch.cat([x_left, x_right, x_bot, x_top], dim=0)
    y_b = torch.cat([y_left, y_right, y_bot, y_top], dim=0)
    normals  = torch.cat([n_left, n_right, n_bot, n_top], dim=0)
    boundary = torch.cat([t_b, x_b, y_b], dim=1).requires_grad_(True)

    # ── Initial condition (t=0) ───────────────────────────────────────────────
    t_i = torch.zeros(ic_points, 1, device=DEVICE)
    x_i = _rand(ic_points, 0.0, X_MAX)
    y_i = _rand(ic_points, 0.0, Y_MAX)
    initial = torch.cat([t_i, x_i, y_i], dim=1)

    return {
        "domain":   domain,
        "boundary": boundary,
        "normals":  normals,
        "initial":  initial,
    }
