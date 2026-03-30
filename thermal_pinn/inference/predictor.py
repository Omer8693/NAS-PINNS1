"""
thermal_pinn/inference/predictor.py
=====================================
Model loading and field prediction utilities.

Loads a trained ThermalPINN checkpoint by (domain, arch, k, dim) and
evaluates it on an arbitrary grid of spatial coordinates at a given time
within the trained window.

The checkpoint stores only the model weights; the domain reference solution
is re-instantiated from thermal_pinn.physics to generate comparison fields.

Usage
-----
    from thermal_pinn.inference.predictor import load_model, predict_field
    model, meta = load_model("rectangle", "bayesian", k=1, dim=2)
    T_pred, T_ref = predict_field(model, meta, t=15.0, n_grid=60)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..network.pinn import ThermalPINN, ARCH_CONFIGS
from ..physics.domains_2d import make_domain, DOMAINS_2D
from ..physics.domains_3d import make_domain_3d, DOMAINS_3D
from ..training.trainer import _normalise_coords

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

T_WATER = 20.0
DELTA_T = 520.0


def _make_eval_grid_2d(domain, n_grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a plotting grid aligned with the domain's native coordinates."""
    if hasattr(domain, "R_out"):                      # Annulus
        lim = domain.R_out
        x_lin = np.linspace(-lim, lim, n_grid)
        y_lin = np.linspace(-lim, lim, n_grid)
    elif hasattr(domain, "R") and not hasattr(domain, "Lx"):   # Circle
        lim = domain.R
        x_lin = np.linspace(-lim, lim, n_grid)
        y_lin = np.linspace(-lim, lim, n_grid)
    else:
        x_lin = np.linspace(0.0, domain.Lx, n_grid)
        y_lin = np.linspace(0.0, domain.Ly, n_grid)
    return np.meshgrid(x_lin, y_lin, indexing="ij")


def _make_eval_grid_3d_mid(domain, n_grid: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a mid-plane slice grid aligned with the domain's geometry."""
    if hasattr(domain, "R") and hasattr(domain, "Hz"):         # Cylinder
        x_lin = np.linspace(-domain.R, domain.R, n_grid)
        y_lin = np.linspace(-domain.R, domain.R, n_grid)
        z_mid = domain.Hz / 2.0
    else:
        x_lin = np.linspace(0.0, getattr(domain, "Lx", 1.0), n_grid)
        y_lin = np.linspace(0.0, getattr(domain, "Ly", 1.0), n_grid)
        z_mid = getattr(domain, "Lz", 1.0) / 2.0
    XX, YY = np.meshgrid(x_lin, y_lin, indexing="ij")
    return XX, YY, z_mid


def _valid_mask(domain, coords_np: np.ndarray) -> np.ndarray:
    """Return an in-domain mask for 2D or 3D evaluation grids."""
    x = coords_np[:, 0]
    y = coords_np[:, 1]

    if coords_np.shape[1] == 3:
        z = coords_np[:, 2]
        if hasattr(domain, "mask"):
            return np.asarray(domain.mask(x, y, z), dtype=bool)
        return np.ones(len(coords_np), dtype=bool)

    if hasattr(domain, "_mask"):
        return np.asarray(domain._mask(x, y), dtype=bool)
    if hasattr(domain, "R_out"):
        r = np.sqrt(x**2 + y**2)
        return (r >= domain.R_in - 1e-9) & (r <= domain.R_out + 1e-9)
    if hasattr(domain, "R") and not hasattr(domain, "Lx"):
        r = np.sqrt(x**2 + y**2)
        return r <= domain.R + 1e-9
    return np.ones(len(coords_np), dtype=bool)


def load_model(domain: str, arch: str, k: int, dim: int,
               device: torch.device | None = None) -> tuple[ThermalPINN, dict]:
    """
    Load a trained ThermalPINN checkpoint from disk.

    Parameters
    ----------
    domain : str  e.g. "rectangle", "cylinder"
    arch   : str  "bayesian" | "nsga2" | "nsga3"
    k      : int  k-skip value used during training
    dim    : int  2 or 3

    Returns
    -------
    model : ThermalPINN  (eval mode, on device)
    meta  : dict  {domain, arch, k, dim, mean_mae, windows, …}
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ThermalPINN(dim=dim, arch=arch).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    meta = {k_: ckpt.get(k_) for k_ in
            ("arch", "dim", "domain", "k", "mean_mae", "windows")}
    return model, meta


def load_best_model(domain: str, arch: str, dim: int,
                    device: torch.device | None = None) -> tuple[ThermalPINN, dict]:
    """Load the best-k checkpoint for (domain, arch, dim) from registry.json."""
    reg_path = CKPT_DIR / "registry.json"
    with open(reg_path) as f:
        registry = json.load(f)

    entry = next((r for r in registry
                  if r["domain"] == domain and r["arch"] == arch
                  and r["dim"] == dim), None)
    if entry is None:
        raise KeyError(f"No registry entry for ({domain}, {arch}, dim={dim})")

    return load_model(domain, arch, entry["best_k"], dim, device)


def predict_field(model: ThermalPINN,
                  domain_name: str,
                  t_query: float,
                  dt_fem: float = 1.5,
                  n_grid: int = 60,
                  device: torch.device | None = None
                  ) -> dict:
    """
    Evaluate trained PINN on a dense grid at time t_query.

    The function identifies which training window covers t_query, retrieves
    the IC temperature at the window start from the domain's analytical/FD
    reference, then evaluates the PINN at τ = (t_query − t_start) / dt_window.

    Parameters
    ----------
    model       : loaded ThermalPINN (from load_model)
    domain_name : str  domain identifier
    t_query     : float  physical time in seconds
    dt_fem      : float  FEM time step (default 1.5 s)
    n_grid      : int    grid resolution per spatial axis

    Returns
    -------
    dict with keys:
        coords    : (N, dim) float32 — spatial coordinates
        T_pred    : (N,) float32     — PINN prediction (°C)
        T_ref     : (N,) float32     — analytical/FD reference (°C)
        mae_C     : float            — mean absolute error
        t_query   : float
        domain    : str
    """
    device = device or next(model.parameters()).device
    dim    = model.dim

    with open(CKPT_DIR / "registry.json") as f:
        registry = json.load(f)

    entry = next(
        (
            r for r in registry
            if r["domain"] == domain_name and r["arch"] == model.arch and r["dim"] == dim
        ),
        None,
    )
    if entry is None:
        raise KeyError(
            f"No registry entry for ({domain_name}, {model.arch}, dim={dim}). "
            "Use predict_field_k(...) with an explicit k value."
        )

    return predict_field_k(
        model,
        domain_name,
        entry["best_k"],
        t_query,
        dt_fem=dt_fem,
        n_grid=n_grid,
        device=device,
    )


def predict_field_k(model: ThermalPINN,
                    domain_name: str,
                    k: int,
                    t_query: float,
                    dt_fem: float = 1.5,
                    n_grid: int = 60,
                    device: torch.device | None = None) -> dict:
    """
    Evaluate trained PINN on a dense grid at time t_query.

    Parameters
    ----------
    model       : ThermalPINN  (loaded from checkpoint)
    domain_name : str
    k           : int  k-skip value used during training
    t_query     : float  physical time in seconds (must be ≤ 30 s)
    dt_fem      : float  nominal FEM time step (1.5 s)
    n_grid      : int    grid resolution per spatial dimension
    """
    device = device or next(model.parameters()).device
    dim    = model.dim

    # Instantiate domain for reference solution
    if dim == 2:
        domain = make_domain(domain_name)
    else:
        domain = make_domain_3d(domain_name)

    dt_window = k * dt_fem
    # Find window [t_start, t_end] that contains t_query
    t_start = (t_query // dt_window) * dt_window
    t_start = min(t_start, 30.0 - dt_window)
    t_end   = min(t_start + dt_window, 30.0)

    tau_val = float((t_query - t_start) / (t_end - t_start))
    tau_val = float(np.clip(tau_val, 0.0, 1.0))

    # Build dense evaluation grid
    if dim == 2:
        XX, YY = _make_eval_grid_2d(domain, n_grid)
        xi, yi = XX.ravel(), YY.ravel()
        coords_np = np.stack([xi, yi], axis=1).astype(np.float32)
        T_ic  = domain.T(xi, yi, t_start)
        T_ref = domain.T(xi, yi, t_query)
    else:
        XX, YY, z_mid = _make_eval_grid_3d_mid(domain, n_grid)
        xi, yi = XX.ravel(), YY.ravel()
        zi     = np.full_like(xi, z_mid)
        coords_np = np.stack([xi, yi, zi], axis=1).astype(np.float32)
        T_ic  = domain.T(xi, yi, zi, t_start)
        T_ref = domain.T(xi, yi, zi, t_query)

    valid = _valid_mask(domain, coords_np)
    T_ic  = np.where(valid, T_ic, T_WATER)
    T_ref = np.where(valid & np.isfinite(T_ref), T_ref, np.nan)

    # Normalise and convert to tensors
    coords_norm = _normalise_coords(coords_np, domain)
    theta_ic_np = np.clip(
        ((T_ic - T_WATER) / DELTA_T), 0.0, 1.2
    ).reshape(-1, 1).astype(np.float32)
    tau_np = np.full((len(coords_np), 1), tau_val, dtype=np.float32)

    def to_t(a): return torch.tensor(a, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        theta_pred = model(to_t(coords_norm),
                           to_t(tau_np),
                           to_t(theta_ic_np)).cpu().numpy().flatten()

    T_pred = T_WATER + DELTA_T * theta_pred
    T_pred = np.where(valid, T_pred, np.nan)
    valid  = np.isfinite(T_ref)
    mae    = float(np.mean(np.abs(T_pred[valid] - T_ref[valid]))) if np.any(valid) else np.nan

    return {
        "coords":   coords_np,
        "T_pred":   T_pred.astype(np.float32),
        "T_ref":    np.where(np.isfinite(T_ref), T_ref, np.nan).astype(np.float32),
        "mae_C":    mae,
        "t_query":  t_query,
        "t_start":  t_start,
        "t_end":    t_end,
        "tau":      tau_val,
        "domain":   domain_name,
        "grid_shape": (n_grid, n_grid),
    }
