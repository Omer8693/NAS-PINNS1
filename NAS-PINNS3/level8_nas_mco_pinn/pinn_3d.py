"""
pinn_3d.py — 3D MCO-PINN Time-Step Skip Operator
==================================================
3D extension of 2D mco_timestepper.py.
5 inputs: (x_norm, y_norm, z_norm, tau, theta_prev)
1 output: theta_next

Supported domains: Rectangular3D, Cylinder3D, StackedCubes3D

Includes loss recording system (returns loss_history).
"""

import os, sys, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.config import DEVICE
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .domains_3d import (
    Rectangular3D, Cylinder3D, StackedCubes3D, LShape3D, LPrism3D,
    T_INIT, T_WATER, DELTA_T, K, RHO_CP, H_CONV, ALPHA,
)
from .mco_timestepper import MCOLoss

# ──────────────────────────────────────────────────────────────
# NAS Architectures (same as 2D)
# ──────────────────────────────────────────────────────────────
ARCH_CONFIGS = {
    "bayesian": {"n_layers": 5, "neurons": [151]*5, "activation": "relu"},
    "nsga2":    {"n_layers": 3, "neurons": [153]*3, "activation": "tanh"},
    "nsga3":    {"n_layers": 3, "neurons": [75]*3,  "activation": "tanh"},
}
ARCH_LABEL = {
    "bayesian": "Bayesian (TPE)",
    "nsga2":    "NSGA-II",
    "nsga3":    "NSGA-III",
}

# ──────────────────────────────────────────────────────────────
# Network structure
# ──────────────────────────────────────────────────────────────

class ResidualNet3D(nn.Module):
    """
    IC-consistent residual network: output = clamp(θ_ic + τ · net(x,y,z,τ,θ_ic), 0, 1)
    At τ=0 the output automatically equals θ_ic → IC loss is zero.
    The network only learns the CHANGE from IC to target.
    """
    def __init__(self, config: dict):
        super().__init__()
        act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}
        act_cls = act_map.get(config.get("activation", "tanh"), nn.Tanh)
        layers, in_dim = [], 5
        for w in config["neurons"]:
            layers += [nn.Linear(in_dim, w), act_cls()]
            in_dim = w
        layers.append(nn.Linear(in_dim, 1))   # raw correction, no activation
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau      = x[:, 3:4]        # 4th column: τ ∈ [0,1]
        theta_ic = x[:, 4:5]        # 5th column: θ_ic
        corr     = self.net(x)      # learned correction
        return torch.clamp(theta_ic + tau * corr, 0.0, 1.0)


def build_net_3d(config: dict) -> ResidualNet3D:
    """Build ResidualNet3D and move to DEVICE."""
    return ResidualNet3D(config).to(DEVICE)


# ──────────────────────────────────────────────────────────────
# Sampling functions
# ──────────────────────────────────────────────────────────────

def _sample_rect(n: int, Lx: float, Ly: float, Lz: float):
    """Random point inside rectangular domain."""
    x = torch.rand(n, 1) * Lx
    y = torch.rand(n, 1) * Ly
    z = torch.rand(n, 1) * Lz
    return x, y, z


def _sample_cyl(n: int, R: float, H: float):
    """Random point inside cylinder (rejection sampling)."""
    pts = []
    while len(pts) < n:
        x = (torch.rand(n*3, 1) * 2 - 1) * R
        y = (torch.rand(n*3, 1) * 2 - 1) * R
        z = torch.rand(n*3, 1) * H
        inside = (x**2 + y**2 <= R**2).squeeze()
        x, y, z = x[inside], y[inside], z[inside]
        pts.append((x, y, z))
    x_all = torch.cat([p[0] for p in pts])[:n]
    y_all = torch.cat([p[1] for p in pts])[:n]
    z_all = torch.cat([p[2] for p in pts])[:n]
    return x_all, y_all, z_all


def _get_dims(domain):
    """Return domain dimensions."""
    if isinstance(domain, (Rectangular3D, LPrism3D, LShape3D)):
        return domain.Lx, domain.Ly, domain.Lz
    elif isinstance(domain, StackedCubes3D):
        return domain.L_cube, domain.L_cube, domain.Lz
    elif isinstance(domain, Cylinder3D):
        return domain.R * 2, domain.R * 2, domain.H
    return 1.3, 0.6, 0.4


def _sample_lshape(n: int, domain):
    """Rejection sampling inside L-shape domain."""
    Lx, Ly, Lz = domain.Lx, domain.Ly, domain.Lz
    pts = []
    while sum(p[0].shape[0] for p in pts) < n:
        x = torch.rand(n * 2, 1) * Lx
        y = torch.rand(n * 2, 1) * Ly
        z = torch.rand(n * 2, 1) * Lz
        x_np = x.numpy().flatten()
        y_np = y.numpy().flatten()
        z_np = z.numpy().flatten()
        inside = domain.mask(x_np, y_np, z_np)
        pts.append((x[inside], y[inside], z[inside]))
    x_all = torch.cat([p[0] for p in pts])[:n]
    y_all = torch.cat([p[1] for p in pts])[:n]
    z_all = torch.cat([p[2] for p in pts])[:n]
    return x_all, y_all, z_all


def sample_interior(domain, n: int):
    """Interior sampling with domain type detection."""
    # LShape3D check MUST come before LPrism3D since LPrism3D = LShape3D (alias)
    if isinstance(domain, LShape3D):
        return _sample_lshape(n, domain)
    elif isinstance(domain, (Rectangular3D, StackedCubes3D)):
        Lx, Ly, Lz = _get_dims(domain)
        return _sample_rect(n, Lx, Ly, Lz)
    elif isinstance(domain, Cylinder3D):
        return _sample_cyl(n, domain.R, domain.H)
    else:
        raise ValueError(f"Unknown domain type: {type(domain)}")


def sample_bc_rect(n_per_face: int, domain):
    """BC points on 6 faces for rectangular domain."""
    Lx, Ly, Lz = _get_dims(domain)
    n = n_per_face
    faces = []
    # x=0, x=Lx
    for xv in [0.0, Lx]:
        x = torch.full((n, 1), xv)
        y = torch.rand(n, 1) * Ly
        z = torch.rand(n, 1) * Lz
        nx = torch.full((n, 1), -1.0 if xv == 0 else 1.0)
        ny = torch.zeros(n, 1)
        nz = torch.zeros(n, 1)
        faces.append((x, y, z, nx, ny, nz))
    # y=0, y=Ly
    for yv in [0.0, Ly]:
        x = torch.rand(n, 1) * Lx
        y = torch.full((n, 1), yv)
        z = torch.rand(n, 1) * Lz
        nx = torch.zeros(n, 1)
        ny = torch.full((n, 1), -1.0 if yv == 0 else 1.0)
        nz = torch.zeros(n, 1)
        faces.append((x, y, z, nx, ny, nz))
    # z=0, z=Lz
    for zv in [0.0, Lz]:
        x = torch.rand(n, 1) * Lx
        y = torch.rand(n, 1) * Ly
        z = torch.full((n, 1), zv)
        nx = torch.zeros(n, 1)
        ny = torch.zeros(n, 1)
        nz = torch.full((n, 1), -1.0 if zv == 0 else 1.0)
        faces.append((x, y, z, nx, ny, nz))
    return faces   # list of (x,y,z,nx,ny,nz) tuples


# ──────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────

def pde_loss_3d(net, domain, x, y, z, tau, theta_ic, dt):
    """
    3D heat equation residual (in normalized form).
    ∂θ/∂τ = Fo_x ∂²θ/∂ξ² + Fo_y ∂²θ/∂η² + Fo_z ∂²θ/∂ζ²
    ξ=x/Lx ∈[0,1], Fo_x = α*dt / Lx²
    """
    Lx, Ly, Lz = _get_dims(domain)
    Fo_x = ALPHA * dt / Lx**2
    Fo_y = ALPHA * dt / Ly**2
    Fo_z = ALPHA * dt / Lz**2

    x_ = (x / Lx).to(DEVICE).requires_grad_(True)
    y_ = (y / Ly).to(DEVICE).requires_grad_(True)
    z_ = (z / Lz).to(DEVICE).requires_grad_(True)
    tau_ = tau.to(DEVICE).requires_grad_(True)

    inp = torch.cat([x_, y_, z_, tau_, theta_ic.to(DEVICE)], dim=1)
    theta = net(inp)

    ones = torch.ones_like(theta)
    dth_dt  = torch.autograd.grad(theta, tau_, ones, create_graph=True)[0]
    dth_dx  = torch.autograd.grad(theta, x_,   ones, create_graph=True)[0]
    dth_dy  = torch.autograd.grad(theta, y_,   ones, create_graph=True)[0]
    dth_dz  = torch.autograd.grad(theta, z_,   ones, create_graph=True)[0]
    d2th_dx2 = torch.autograd.grad(dth_dx, x_, ones, create_graph=True)[0]
    d2th_dy2 = torch.autograd.grad(dth_dy, y_, ones, create_graph=True)[0]
    d2th_dz2 = torch.autograd.grad(dth_dz, z_, ones, create_graph=True)[0]

    residual = dth_dt - Fo_x * d2th_dx2 - Fo_y * d2th_dy2 - Fo_z * d2th_dz2
    return (residual**2).mean()


def _get_theta(domain, x, y, z, t):
    """x,y,z tensors → θ numpy tensor (analytical)."""
    x_np = x.detach().cpu().numpy().flatten()
    y_np = y.detach().cpu().numpy().flatten()
    z_np = z.detach().cpu().numpy().flatten()
    if isinstance(domain, Cylinder3D):
        T_ref = domain.T_xyz(x_np, y_np, z_np, t)
    else:
        T_ref = domain.T(x_np, y_np, z_np, t)
    theta = np.clip((T_ref - T_WATER) / DELTA_T, 0.0, 1.0)
    return torch.tensor(theta, dtype=torch.float32).unsqueeze(1)


def supervision_loss(net, domain, x, y, z, theta_ic, tau_val, theta_target):
    """
    θ_pred(x,y,z,τ=tau_val, θ_ic) = θ_target  supervision loss.
    Used for both IC (tau_val=0) and target (tau_val=1).
    """
    Lx, Ly, Lz = _get_dims(domain)
    x_ = (x / Lx).to(DEVICE)
    y_ = (y / Ly).to(DEVICE)
    z_ = (z / Lz).to(DEVICE)
    tau_ = torch.full((x.shape[0], 1), tau_val).to(DEVICE)
    th_ic = theta_ic.to(DEVICE)

    inp   = torch.cat([x_, y_, z_, tau_, th_ic], dim=1)
    pred  = net(inp)
    target= theta_target.to(DEVICE)
    return ((pred - target) ** 2).mean()


def ic_loss_3d(net, domain, x, y, z, theta_ref):
    """IC loss: θ(x,y,z,τ=0) = θ_ref(x,y,z)."""
    return supervision_loss(net, domain, x, y, z, theta_ref, 0.0, theta_ref)


# ──────────────────────────────────────────────────────────────
# Training window
# ──────────────────────────────────────────────────────────────

def train_window_3d(net, mco, domain, t_start, t_end,
                    n_col=2000, n_bc=200,
                    n_adam=800, lr=1e-3,
                    verbose=False):
    """
    Train a time window [t_start, t_end] with fixed-weight supervision.
    MCO removed: L_total = L_ic + L_target + W_PDE * L_pde
    For t_start=0 uniform θ=1 is used instead of proper IC series.
    Returns:
        net           : trained network
        loss_history  : dict
    """
    dt = t_end - t_start
    W_PDE = 0.05

    # IC: prevent Gibbs oscillation at t=0 → uniform θ=1
    def theta_ic_fn(x, y, z):
        n = x.shape[0]
        if t_start < 1e-9:
            # Uniform initial condition: T=T_init everywhere
            return torch.ones(n, 1, dtype=torch.float32)
        x_np = x.detach().cpu().numpy().flatten()
        y_np = y.detach().cpu().numpy().flatten()
        z_np = z.detach().cpu().numpy().flatten()
        if isinstance(domain, Cylinder3D):
            T_ref = domain.T_xyz(x_np, y_np, z_np, t_start)
        else:
            T_ref = domain.T(x_np, y_np, z_np, t_start)
        theta = np.clip((T_ref - T_WATER) / DELTA_T, 0.0, 1.0)
        return torch.tensor(theta, dtype=torch.float32).unsqueeze(1)

    # Collocation points
    x_col, y_col, z_col = sample_interior(domain, n_col)
    tau_col = torch.rand(n_col, 1)
    theta_ic_col = theta_ic_fn(x_col, y_col, z_col)

    # Target: analytical θ at t_end (full supervision)
    theta_target_col = _get_theta(domain, x_col, y_col, z_col, t_end)

    net = net.to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_adam, eta_min=lr*0.01)

    history = {"L_phys": [], "L_bc": [], "L_ic": [], "L_total": [],
               "w_phys": [], "w_bc": [], "w_ic": []}
    log_every = max(1, n_adam // 20)

    for epoch in range(n_adam):
        optimizer.zero_grad()

        # L_target: θ(τ=1) = θ_end  [target supervision]
        # IC loss is now zero (automatically enforced by ResidualNet3D)
        Lt = supervision_loss(net, domain, x_col, y_col, z_col,
                              theta_ic_col, 1.0, theta_target_col)

        # L_pde: physics regularizer (light)
        Lp = pde_loss_3d(net, domain, x_col, y_col, z_col, tau_col,
                         theta_ic_col, dt)

        L_total = Lt + W_PDE * Lp
        L_total.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Record
        if epoch % log_every == 0 or epoch == n_adam - 1:
            history["L_phys"].append(float(Lp.item()))
            history["L_bc"].append(float(Lt.item()))
            history["L_ic"].append(0.0)
            history["L_total"].append(float(L_total.item()))
            history["w_phys"].append(W_PDE)
            history["w_bc"].append(1.0)
            history["w_ic"].append(1.0)
            if verbose:
                print(f"    ep={epoch:4d}  Lt={Lt.item():.4f}  "
                      f"Lp={Lp.item():.4f}  Ltot={L_total.item():.4f}")

    return net, history


# ──────────────────────────────────────────────────────────────
# Field evaluation
# ──────────────────────────────────────────────────────────────

def predict_field_3d(net, domain, t_start, t_end, nx=25, ny=15, nz=10):
    """
    Evaluate T(x,y,z,t_end) field with trained network.
    Returns: T_pred (ny×nx×nz array), xx, yy (meshgrid for z_mid slice)
    """
    Lx, Ly, Lz = _get_dims(domain)
    if isinstance(domain, Cylinder3D):
        Lx, Ly = domain.R, domain.R   # half-width for display

    xi = np.linspace(0 if not isinstance(domain, Cylinder3D) else -Lx,
                     Lx, nx)
    yi = np.linspace(0 if not isinstance(domain, Cylinder3D) else -Ly,
                     Ly, ny)
    zi = np.linspace(0, Lz, nz)

    T_vol = np.full((ny, nx, nz), np.nan)

    theta_ic_grid = None

    for k, z_val in enumerate(zi):
        xx, yy = np.meshgrid(xi, yi)
        x_flat = xx.ravel()
        y_flat = yy.ravel()
        z_flat = np.full_like(x_flat, z_val)

        if isinstance(domain, Cylinder3D):
            T_ic = domain.T_xyz(x_flat, y_flat, z_flat, t_start)
        else:
            T_ic = domain.T(x_flat, y_flat, z_flat, t_start)
        theta_ic = (T_ic - T_WATER) / DELTA_T

        # Geometry mask
        if isinstance(domain, Cylinder3D):
            r_flat = np.sqrt(x_flat**2 + y_flat**2)
            valid  = r_flat <= domain.R
        elif isinstance(domain, (LPrism3D, LShape3D)):
            valid = domain.mask(x_flat, y_flat, z_flat)
        else:
            valid = np.ones(len(x_flat), dtype=bool)

        if not valid.any():
            continue

        x_v = torch.tensor(x_flat[valid], dtype=torch.float32).unsqueeze(1)
        y_v = torch.tensor(y_flat[valid], dtype=torch.float32).unsqueeze(1)
        z_v = torch.tensor(z_flat[valid], dtype=torch.float32).unsqueeze(1)
        th_v= torch.tensor(theta_ic[valid], dtype=torch.float32).unsqueeze(1)

        Lx2, Ly2, Lz2 = _get_dims(domain)

        x_n = x_v / Lx2
        y_n = y_v / Ly2
        z_n = z_v / Lz2
        tau_ = torch.ones_like(x_n)

        inp = torch.cat([x_n, y_n, z_n, tau_, th_v], dim=1).to(DEVICE)
        with torch.no_grad():
            theta_pred = net(inp).cpu().numpy().flatten()

        T_pred_v = T_WATER + DELTA_T * theta_pred
        T_slice   = np.full(len(x_flat), np.nan)
        T_slice[valid] = T_pred_v
        T_vol[:, :, k] = T_slice.reshape(ny, nx)

    xx2d, yy2d = np.meshgrid(xi, yi)
    return T_vol, xx2d, yy2d, xi, yi, zi


# ──────────────────────────────────────────────────────────────
# Skip operator — full run
# ──────────────────────────────────────────────────────────────

def run_skip_3d(domain, arch_name, skip_values=None,
                n_col=2000, n_bc=200, n_adam=800,
                nx=25, ny=15, nz=10,
                verbose=True):
    """
    Run MCO-PINN for all skip values on the given domain.

    Returns dict: {
        skip: {
            "mae_C": float,
            "loss_history": list of window dicts,
            "T_fields": {window_idx: T_vol ndarray},
            "T_fem":    {window_idx: T_vol ndarray},
            "runtime_s": float,
        }
    }
    """
    if skip_values is None:
        skip_values = [1, 2, 4]

    # FEM time steps: t = 0, 1.5, 3, ..., 30s (21 steps)
    t_fem    = np.arange(0, 30 + 1e-9, 1.5)  # 0..30 every 1.5s = 21 points
    n_steps  = len(t_fem) - 1                  # 20 transitions

    config = ARCH_CONFIGS[arch_name]
    results = {}

    for skip in skip_values:
        t0 = time.time()
        if verbose:
            print(f"\n  [{arch_name}] skip={skip}  domain={domain.name}")

        # Window time points
        indices   = list(range(0, len(t_fem), skip))
        if indices[-1] != len(t_fem) - 1:
            indices.append(len(t_fem) - 1)
        windows   = list(zip(indices[:-1], indices[1:]))

        mae_windows   = []
        all_histories = []
        T_fields      = {}
        T_fem_fields  = {}

        for wi, (i_start, i_end) in enumerate(windows):
            t_start = float(t_fem[i_start])
            t_end   = float(t_fem[i_end])

            net = build_net_3d(config)

            trained_net, history = train_window_3d(
                net, None, domain,
                t_start=t_start, t_end=t_end,
                n_col=n_col, n_bc=n_bc, n_adam=n_adam,
                verbose=verbose,
            )
            all_histories.append(history)

            # Field evaluation
            T_vol, xx2d, yy2d, xi, yi, zi = predict_field_3d(
                trained_net, domain, t_start, t_end, nx, ny, nz)

            # FEM reference (analytical)
            T_fem_vol = np.full((ny, nx, nz), np.nan)
            for k, zv in enumerate(zi):
                zv_arr = np.full_like(xi, float(zv))
                if isinstance(domain, Cylinder3D):
                    T_slice = np.array([
                        domain.T_xyz(xi, np.full_like(xi, yv), zv_arr, t_end)
                        for yv in yi
                    ])
                    r2d = np.sqrt(xx2d**2 + yy2d**2)
                    T_slice[r2d > domain.R] = np.nan
                else:
                    T_slice = np.array([
                        domain.T(xi, np.full_like(xi, yv), zv_arr, t_end)
                        for yv in yi
                    ])
                T_fem_vol[:, :, k] = T_slice

            # MAE
            valid  = ~np.isnan(T_vol) & ~np.isnan(T_fem_vol)
            mae_w  = np.nanmean(np.abs(T_vol[valid] - T_fem_vol[valid]))
            mae_windows.append(mae_w)
            T_fields[wi]     = T_vol
            T_fem_fields[wi] = T_fem_vol

            if verbose:
                print(f"    Window {wi+1}/{len(windows)}: "
                      f"t={t_start:.1f}→{t_end:.1f}s  MAE={mae_w:.2f}°C")

        results[skip] = {
            "mae_C":        float(np.mean(mae_windows)),
            "mae_windows":  mae_windows,
            "loss_history": all_histories,
            "T_fields":     T_fields,
            "T_fem":        T_fem_fields,
            "runtime_s":    time.time() - t0,
            "grid":         {"xi": xi, "yi": yi, "zi": zi,
                             "xx2d": xx2d, "yy2d": yy2d},
        }

        if verbose:
            print(f"  → skip={skip}: MAE={results[skip]['mae_C']:.2f}°C  "
                  f"({time.time()-t0:.0f}s)")

    return results
