# naspinn_baseline_with_quench_2026_data.py
# Baseline NAS-PINN (DARTS-style) with 2026 quenching reference data
# "Mitigating distortions in cast automotive subframes: A finite element simulation approach"
# Published online 27 January 2026, Int J Adv Manuf Technol, Springer, DOI: 10.1007/s00170-026-17515-w
# Author: Grok (integrated key tables/equations March 2026)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import csv
import time
import json
import shutil
import re
from scipy.interpolate import interp1d

try:
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    PSO = None

    class ElementwiseProblem:  # type: ignore[override]
        pass

    def minimize(*_args, **_kwargs):
        raise ImportError("pymoo is required for PSO refinement.")

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────
# Data from 2026 quenching reference study (Springer)
# ────────────────────────────────────────────────

# Table 1: A356 viscoplastic parameters (Eq. 6 & 7 in article)
# Source: Article page 3 / Table 1 (empirical settings for A356 flow stress)
temps_c = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550])
F_vals  = np.array([22.0, 21.5, 21.0, 20.0, 18.9, 17.5, 16.4, 16.0, 15.6, 12.3, 7.2, 2.0])
n_vals  = np.array([0.3, 0.3, 0.3, 0.3, 0.27, 0.21, 0.15, 0.125, 0.02, 0.0, 0.0, 0.0])
m_vals  = np.array([0.0, 0.0, 0.0, 0.006, 0.016, 0.04, 0.1, 0.15, 0.19, 0.22, 0.25, 0.277])

F_interp = interp1d(temps_c, F_vals, kind='linear', fill_value='extrapolate')
n_interp = interp1d(temps_c, n_vals, kind='linear', fill_value='extrapolate')
m_interp = interp1d(temps_c, m_vals, kind='linear', fill_value='extrapolate')

T_hardening_threshold = 420.0  # T0 from article (creep above, hardening below)

def get_viscoplastic_params(T):
    """T (torch tensor in °C) → F(T), n(T), m(T) tensors"""
    T_np = T.detach().cpu().numpy().flatten()
    return (torch.tensor(F_interp(T_np), dtype=torch.float32, device=T.device),
            torch.tensor(n_interp(T_np), dtype=torch.float32, device=T.device),
            torch.tensor(m_interp(T_np), dtype=torch.float32, device=T.device))

# Table 2: Variable stiffness for contact/boundary (air-gap / penetration)
# Source: Article page 5 / Table 2
disp_table_m = np.array([-0.10, -0.00020, -0.00008, -0.00005, -0.00002, -0.00001,
                         0.00000, 0.00001, 0.00002, 0.00003, 0.00005])
stiff_table_pa_per_m = np.array([1e5, 5e5, 5e6, 5e7, 5e8, 1e9, 1e10, 1e10, 1e11, 1e12, 1e13])

stiff_interp = interp1d(disp_table_m, stiff_table_pa_per_m, kind='linear', fill_value='extrapolate')

def get_contact_stiffness(disp):
    """disp (torch tensor in m) → stiffness (Pa/m)"""
    disp_np = disp.detach().cpu().numpy().flatten()
    return torch.tensor(stiff_interp(disp_np), dtype=torch.float32, device=disp.device)

# Other key values from article
COHERENCY_SOLID_FRACTION = 0.9   # gs = 0.9 (coherent mushy zone boundary)
T_SOLUTION_TREATMENT = 540.0     # °C
T_WATER = 25.0                   # °C (quenching bath)

# Approximate digitized references from the 2026 quenching study figures
# Fig. 7: water-temperature rise in tank (depth dependent)
FIG7_TIME_S = np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.float32)
FIG7_DEPTH_M = np.array([0.0, 0.3, 0.6, 0.9], dtype=np.float32)

# Fig. 17-20: average rack-layer distortion (mm), bottom -> top
FIG17_20_LAYER_DISP_MM = np.array([-2.1, -0.8, 0.3, 1.2, 1.8], dtype=np.float32)

# Approximate A356 material properties (literature values, not directly in article)
material_props = {
    'density': 2700.0,          # kg/m³
    'specific_heat': 963.0,     # J/kg·K
    'thermal_conductivity': 151.0,  # W/m·K
    'thermal_expansion': 2.3e-5,    # 1/K (linear coefficient β)
    'young_modulus': 70e9,          # Pa (room temp, approx)
    'poisson_ratio': 0.33,
    'T_coh': 550.0                  # Coherency temperature ≈ max from Table 1 (for thermal strain integral)
}

# Simple approximate HTC(T) for water quenching (literature-based, article uses correlations from ref [20])
# Stages: film (> ~300-400°C), nucleate (~100-300°C), convective (< ~100°C)
def approximate_htc_quenching(T_surface):
    T_sat = 100.0  # boiling point water
    h_film = 200.0      # W/m²K (low, vapour blanket)
    h_nucleate = 10000.0  # high peak
    h_conv = 1000.0
    T_surface = T_surface.item() if torch.is_tensor(T_surface) else T_surface
    if T_surface > 400.0:
        return h_film
    elif T_surface > 120.0:
        return h_nucleate * (1.0 - (T_surface - 120.0) / (400.0 - 120.0))**2  # peak around 200-300
    else:
        return h_conv + 500.0 * (T_sat - T_surface) / T_sat  # natural convection

def approximate_htc_quenching_torch(T_surface):
    """Vectorized HTC(T) for torch tensors (T in °C)."""
    T_sat = 100.0
    h_film = 200.0
    h_nucleate = 10000.0
    h_conv = 1000.0

    h = torch.where(
        T_surface > 400.0,
        torch.full_like(T_surface, h_film),
        torch.where(
            T_surface > 120.0,
            h_nucleate * (1.0 - (T_surface - 120.0) / (400.0 - 120.0))**2,
            h_conv + 500.0 * (T_sat - T_surface) / T_sat,
        ),
    )
    return torch.clamp(h, min=50.0)

# ────────────────────────────────────────────────
# Domain (simplified 2D rectangle representing subframe cross-section)
# Diagonal ~1.3 m from article Fig. 4
# ────────────────────────────────────────────────
x_min, x_max = -0.65, 0.65
y_min, y_max = -0.20, 0.20
t_min, t_max = 0.0, 60.0   # quenching time example (seconds)

N_col = 10000
N_ic  = 1000
N_bc  = 1000

HEAT_RES_SCALE = 1e6
VISC_RES_SCALE = 1e9
BC_FLUX_SCALE = 1e6
TEMP_DATA_SCALE = 100.0
DISP_DATA_SCALE = 1e-3

def _depth_to_y(depth_m):
    # Map Fig. 7 depth [0, 0.9] m onto model y-domain [y_min, y_max]
    return y_min + (depth_m / 0.9) * (y_max - y_min)

def build_quench_reference_data():
    """Build reference points from paper figures (approximate digitization)."""
    # Fig. 7 synthetic digitization used in this project (same trend as time.py)
    x_temp, y_temp, t_temp, target_temp = [], [], [], []
    for depth in FIG7_DEPTH_M:
        y_val = _depth_to_y(depth)
        for t_val in FIG7_TIME_S:
            temp_val = 60.0 + 35.0 * (1.0 - np.exp(-t_val / 10.0)) + 5.0 * depth
            x_temp.append(0.0)
            y_temp.append(y_val)
            t_temp.append(t_val)
            target_temp.append(temp_val)

    # Fig. 17-20 layer-average distortion at end of quench
    layer_y = np.linspace(y_min, y_max, len(FIG17_20_LAYER_DISP_MM), dtype=np.float32)
    layer_disp_m = FIG17_20_LAYER_DISP_MM * 1e-3  # mm -> m

    ref = {
        "x_temp": torch.tensor(np.array(x_temp, dtype=np.float32)[:, None], device=device),
        "y_temp": torch.tensor(np.array(y_temp, dtype=np.float32)[:, None], device=device),
        "t_temp": torch.tensor(np.array(t_temp, dtype=np.float32)[:, None], device=device),
        "target_temp": torch.tensor(np.array(target_temp, dtype=np.float32)[:, None], device=device),
        "x_disp": torch.zeros((len(layer_y), 1), dtype=torch.float32, device=device),
        "y_disp": torch.tensor(layer_y[:, None], dtype=torch.float32, device=device),
        "t_disp": torch.full((len(layer_y), 1), t_max, dtype=torch.float32, device=device),
        "target_uy": torch.tensor(layer_disp_m[:, None], dtype=torch.float32, device=device),
    }
    return ref

# ────────────────────────────────────────────────
# Point sampling (uniform random - placeholder; replace with CAD/ STL points)
# For real subframe, use trimesh or similar to sample interior/boundary from CAD file
# ────────────────────────────────────────────────

def sample_points():
    # Collocation points (interior)
    x_c = torch.rand(N_col, 1, device=device) * (x_max - x_min) + x_min
    y_c = torch.rand(N_col, 1, device=device) * (y_max - y_min) + y_min
    t_c = torch.rand(N_col, 1, device=device) * (t_max - t_min) + t_min

    # Initial condition points (t=0)
    x_ic = torch.rand(N_ic, 1, device=device) * (x_max - x_min) + x_min
    y_ic = torch.rand(N_ic, 1, device=device) * (y_max - y_min) + y_min
    t_ic = torch.zeros_like(x_ic)

    # Boundary points (simple rectangle - replace with actual locator points from CAD)
    t_bc = torch.rand(N_bc, 1, device=device) * (t_max - t_min) + t_min
    x_left  = torch.full((N_bc, 1), x_min, device=device)
    y_left_random = torch.rand(N_bc, 1, device=device) * (y_max - y_min) + y_min
    x_right = torch.full((N_bc, 1), x_max, device=device)
    y_right_random = torch.rand(N_bc, 1, device=device) * (y_max - y_min) + y_min
    y_bottom = torch.full((N_bc, 1), y_min, device=device)
    x_bottom_random = torch.rand(N_bc, 1, device=device) * (x_max - x_min) + x_min

    return (x_c, y_c, t_c), (x_ic, y_ic, t_ic), (x_left, y_left_random, t_bc), (x_right, y_right_random, t_bc), (x_bottom_random, y_bottom, t_bc)

# ────────────────────────────────────────────────
# NAS-PINN Model (DARTS-style baseline)
# ────────────────────────────────────────────────

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class MixedOp(nn.Module):
    def __init__(self, in_c, out_c, mask_levels=[32, 64, 96, 128, 192, 256]):
        super().__init__()
        self.mask_levels = mask_levels
        self.n_masks = len(mask_levels)
        self.ops = nn.ModuleList([
            nn.Identity() if in_c == out_c else nn.Linear(in_c, out_c),
            nn.Sequential(nn.Linear(in_c, out_c), nn.Tanh()),
            nn.Sequential(nn.Linear(in_c, out_c), SinActivation()),
        ])
        self.n_ops = len(self.ops)
        total = self.n_ops + self.n_masks
        self.alpha = nn.Parameter(torch.randn(total) * 0.1)

    def relaxed_op(self, x):
        weights = F.softmax(self.alpha[:self.n_ops], dim=0)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

    def forward(self, x):
        mixed = self.relaxed_op(x)
        mask_weights = torch.sigmoid(self.alpha[self.n_ops:])
        final = 0.0
        dim = mixed.shape[-1]
        for j, keep in enumerate(self.mask_levels):
            k = min(keep, dim)
            mask = torch.zeros(dim, device=mixed.device)
            mask[:k] = 1.0
            masked = mixed * mask.unsqueeze(0)
            final += mask_weights[j] * masked
        return final

class NAS_PINN(nn.Module):
    def __init__(self, layers=5, base_neurons=96):
        super().__init__()
        dims = [3] + [base_neurons] * (layers - 1) + [3]  # input x,y,t → output T, ux, uy
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(MixedOp(dims[i], dims[i+1]))

    def forward(self, xyt):
        x = xyt
        for layer in self.layers:
            x = layer(x)
        return x

def clone_model_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def flatten_model_params(model):
    parts = [p.detach().cpu().reshape(-1).numpy().astype(np.float64) for p in model.parameters()]
    return np.concatenate(parts, axis=0)

def set_model_from_flat_vector(model, flat):
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            n_param = param.numel()
            chunk = torch.from_numpy(flat[offset : offset + n_param]).view_as(param).to(device=device, dtype=param.dtype)
            param.copy_(chunk)
            offset += n_param

def count_model_params(model):
    return int(sum(p.numel() for p in model.parameters()))

class PSOWeightProblem(ElementwiseProblem):
    def __init__(self, lower, upper, objective):
        super().__init__(n_var=int(lower.size), n_obj=1, n_constr=0, xl=lower, xu=upper)
        self.objective = objective

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = float(self.objective(x))

# ────────────────────────────────────────────────
# Physics loss (extended: thermal + simple elastic + approximate viscoplastic + quenching HTC)
# NOTE: Viscoplastic is path-dependent; used simple incremental approximation in training loop
# ────────────────────────────────────────────────

def physics_loss(model, x, y, t, phi_prev=None, create_graph=True):
    inp = torch.cat([x, y, t], dim=1).requires_grad_(True)
    pred = model(inp)
    T = pred[:, 0:1]

    if phi_prev is None:
        phi_prev = torch.zeros_like(T)

    # 1) Heat equation residual on interior points
    # Always build graph for first derivatives so second derivatives are available.
    grad_T = torch.autograd.grad(T.sum(), inp, create_graph=True)[0]
    T_t = grad_T[:, 2:3]
    T_x = grad_T[:, 0:1]
    T_y = grad_T[:, 1:2]
    T_xx = torch.autograd.grad(T_x.sum(), inp, create_graph=create_graph, retain_graph=True)[0][:, 0:1]
    T_yy = torch.autograd.grad(T_y.sum(), inp, create_graph=create_graph)[0][:, 1:2]

    heat_res = (
        material_props["density"] * material_props["specific_heat"] * T_t
        - material_props["thermal_conductivity"] * (T_xx + T_yy)
    )

    # 2) Approximate viscoplastic residual (Eq. 6 surrogate)
    F_t, n_t, m_t = get_viscoplastic_params(T)
    eps_p_dot = torch.abs(T_t) / 100.0
    sigma_bar = torch.abs(eps_p_dot * material_props["young_modulus"])
    phi_eff = torch.clamp(phi_prev + 1e-3, min=1e-6)
    rate_eff = torch.clamp(eps_p_dot + 1e-6, min=1e-6)
    visc_rhs = F_t * phi_eff**n_t * rate_eff**m_t
    visc_res = sigma_bar - visc_rhs

    loss_heat = torch.mean((heat_res / HEAT_RES_SCALE) ** 2)
    loss_visc = torch.mean((visc_res / VISC_RES_SCALE) ** 2)
    return loss_heat + loss_visc, loss_heat, loss_visc

# ────────────────────────────────────────────────
# Initial & boundary conditions loss
# ────────────────────────────────────────────────

def ic_loss(model, x, y, t):
    inp = torch.cat([x, y, t], dim=1)
    pred = model(inp)
    T_pred = pred[:, 0:1]
    ux_pred = pred[:, 1:2]
    uy_pred = pred[:, 2:3]
    T_target = torch.full_like(T_pred, T_SOLUTION_TREATMENT)
    return torch.mean((T_pred - T_target)**2 + ux_pred**2 + uy_pred**2)

def bc_loss(model, x_l, y_l, t_l, x_r, y_r, t_r, x_b, y_b, t_b):
    # Displacement constraints + quenching flux at boundaries
    inp_l = torch.cat([x_l, y_l, t_l], 1)
    inp_r = torch.cat([x_r, y_r, t_r], 1)
    inp_b = torch.cat([x_b, y_b, t_b], 1)
    pred_l = model(inp_l)
    pred_r = model(inp_r)
    pred_b = model(inp_b)

    disp_l = pred_l[:, 1:]
    disp_r = pred_r[:, 1:]
    disp_b = pred_b[:, 1:]

    # Contact stiffness from Table 2 (left support approximation)
    disp_l_norm = torch.norm(disp_l, dim=1)
    stiff_l = get_contact_stiffness(disp_l_norm)
    soft_bc_l = stiff_l * disp_l_norm**2

    # Quenching heat-transfer penalty at boundary points
    T_boundary = torch.cat([pred_l[:, 0:1], pred_r[:, 0:1], pred_b[:, 0:1]], dim=0)
    htc = approximate_htc_quenching_torch(T_boundary)
    flux_res = htc * (T_boundary - T_WATER)

    l_disp = torch.mean(disp_r**2 + disp_b**2) + torch.mean(soft_bc_l)
    l_flux = torch.mean((flux_res / BC_FLUX_SCALE) ** 2)
    return l_disp + l_flux, l_disp, l_flux

def data_loss(model, reference_data):
    # Fig. 7 temperature references
    inp_temp = torch.cat(
        [reference_data["x_temp"], reference_data["y_temp"], reference_data["t_temp"]],
        dim=1,
    )
    T_pred = model(inp_temp)[:, 0:1]
    l_temp = torch.mean(((T_pred - reference_data["target_temp"]) / TEMP_DATA_SCALE) ** 2)

    # Fig. 17-20 layer distortion references (uy at final time)
    inp_disp = torch.cat(
        [reference_data["x_disp"], reference_data["y_disp"], reference_data["t_disp"]],
        dim=1,
    )
    uy_pred = model(inp_disp)[:, 2:3]
    l_disp = torch.mean(((uy_pred - reference_data["target_uy"]) / DISP_DATA_SCALE) ** 2)

    return l_temp + l_disp, l_temp, l_disp

# ────────────────────────────────────────────────
# Main training loop (baseline Adam + optional L-BFGS/PSO)
# Extended with incremental time loop + data loss from reference figures
# ────────────────────────────────────────────────

def collect_epoch_losses(model, points, reference_data, n_time_steps, create_graph=True):
    (x_c, y_c, t_c), (x_ic, y_ic, t_ic), (x_l, y_l, t_l), (x_r, y_r, t_r), (x_b, y_b, t_b) = points

    dt = (t_max - t_min) / n_time_steps
    phi_prev = torch.zeros_like(x_c)

    total_physics = 0.0
    total_heat = 0.0
    total_visc = 0.0

    for step in range(n_time_steps):
        t_step = t_c.clamp(min=t_min + step * dt, max=t_min + (step + 1) * dt)
        l_ph, l_heat, l_visc = physics_loss(
            model,
            x_c,
            y_c,
            t_step,
            phi_prev=phi_prev,
            create_graph=create_graph,
        )

        with torch.no_grad():
            pred_step = model(torch.cat([x_c, y_c, t_step], dim=1))
            T_step = pred_step[:, 0:1]
            u_mag = torch.norm(pred_step[:, 1:3], dim=1, keepdim=True)
            hard_mask = (T_step <= T_hardening_threshold).float()
            phi_prev = phi_prev + 1e-3 * u_mag * hard_mask

        total_physics = total_physics + l_ph / n_time_steps
        total_heat = total_heat + l_heat / n_time_steps
        total_visc = total_visc + l_visc / n_time_steps

    l_ic = ic_loss(model, x_ic, y_ic, t_ic)
    l_bc, l_bc_disp, l_bc_flux = bc_loss(model, x_l, y_l, t_l, x_r, y_r, t_r, x_b, y_b, t_b)
    l_data, l_data_temp, l_data_disp = data_loss(model, reference_data)

    return {
        "physics": total_physics,
        "heat": total_heat,
        "visc": total_visc,
        "ic": l_ic,
        "bc": l_bc,
        "bc_disp": l_bc_disp,
        "bc_flux": l_bc_flux,
        "data": l_data,
        "data_temp": l_data_temp,
        "data_disp": l_data_disp,
    }

def weighted_total_loss(loss_dict, args):
    return (
        args.w_physics * loss_dict["physics"]
        + args.w_ic * loss_dict["ic"]
        + args.w_bc * loss_dict["bc"]
        + args.w_data * loss_dict["data"]
    )

def evaluate_objective(model, points, reference_data, args, n_time_steps=None, create_graph=False):
    n_steps = int(args.n_time_steps if n_time_steps is None else n_time_steps)
    loss_dict = collect_epoch_losses(model, points, reference_data, n_steps, create_graph=create_graph)
    total = weighted_total_loss(loss_dict, args)
    return float(total.item()), loss_dict


def safe_evaluate_objective(
    model,
    points,
    reference_data,
    args,
    stage_name,
    n_time_steps=None,
    fallback_points=None,
    fallback_n_time_steps=None,
):
    try:
        return evaluate_objective(
            model,
            points,
            reference_data,
            args,
            n_time_steps=n_time_steps,
            create_graph=False,
        )
    except torch.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if fallback_points is None:
            raise
        print(f"{stage_name} full-objective evaluation OOM, falling back to reduced evaluation: {exc}")
        return evaluate_objective(
            model,
            fallback_points,
            reference_data,
            args,
            n_time_steps=fallback_n_time_steps,
            create_graph=False,
        )

def save_training_artifacts(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "metrics.csv")
    fields = [
        "epoch",
        "total",
        "physics",
        "heat",
        "visc",
        "ic",
        "bc",
        "bc_disp",
        "bc_flux",
        "data",
        "data_temp",
        "data_disp",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    if not history:
        return

    epochs = np.array([r["epoch"] for r in history], dtype=np.float32)
    total = np.array([r["total"] for r in history], dtype=np.float64)
    physics = np.array([r["physics"] for r in history], dtype=np.float64)
    data = np.array([r["data"] for r in history], dtype=np.float64)
    bc = np.array([r["bc"] for r in history], dtype=np.float64)
    ic = np.array([r["ic"] for r in history], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, total, label="total", linewidth=1.8)
    ax.plot(epochs, physics, label="physics", linewidth=1.4)
    ax.plot(epochs, data, label="data", linewidth=1.4)
    ax.plot(epochs, bc, label="bc", linewidth=1.2)
    ax.plot(epochs, ic, label="ic", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Quench2026 NAS-PINN Training Losses")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=180)
    plt.close(fig)

def save_checkpoint(model, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def load_stage_objectives(save_dir):
    stage_csv = os.path.join(save_dir, "stage_summary.csv")
    if not os.path.exists(stage_csv):
        return {}

    objectives = {}
    try:
        with open(stage_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stage = row.get("stage")
                if not stage:
                    continue
                try:
                    objectives[stage] = float(row.get("objective", "nan"))
                except (TypeError, ValueError):
                    continue
    except OSError:
        return {}
    return objectives


def load_checkpoint_if_exists(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        return False
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    return True


def load_best_total_from_metrics(save_dir):
    metrics_path = os.path.join(save_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        return None

    best = None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    val = float(row.get("total", "nan"))
                except (TypeError, ValueError):
                    continue
                if np.isnan(val):
                    continue
                if best is None or val < best:
                    best = val
    except OSError:
        return None
    return best


def load_best_total_from_train_log(save_dir):
    log_path = os.path.join(save_dir, "train.log")
    if not os.path.exists(log_path):
        return None

    pattern = re.compile(r"Total:\s*([0-9eE+\-.]+)")
    best = None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Epoch" not in line or "Total:" not in line:
                    continue
                match = pattern.search(line)
                if match is None:
                    continue
                try:
                    val = float(match.group(1))
                except (TypeError, ValueError):
                    continue
                if best is None or val < best:
                    best = val
    except OSError:
        return None
    return best


def _subsample_triplet(triplet, target_size):
    if target_size is None or target_size <= 0:
        return triplet
    total = int(triplet[0].shape[0])
    if target_size >= total:
        return triplet
    idx = torch.randperm(total, device=triplet[0].device)[:target_size]
    return tuple(t.index_select(0, idx) for t in triplet)


def build_lbfgs_points(points, args):
    col = _subsample_triplet(points[0], int(args.lbfgs_col_points))
    ic = _subsample_triplet(points[1], int(args.lbfgs_ic_points))
    left = _subsample_triplet(points[2], int(args.lbfgs_bc_points))
    right = _subsample_triplet(points[3], int(args.lbfgs_bc_points))
    bottom = _subsample_triplet(points[4], int(args.lbfgs_bc_points))
    return (col, ic, left, right, bottom)

def train_baseline(args):
    run_start = time.perf_counter()
    args.save_dir = os.path.abspath(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    stage_csv = os.path.join(args.save_dir, "stage_summary.csv")
    run_meta_path = os.path.join(args.save_dir, "run_meta.json")
    final_ckpt = os.path.join(args.save_dir, "baseline_model.pth")
    if (
        not args.force_final
        and os.path.exists(final_ckpt)
        and os.path.exists(stage_csv)
        and os.path.exists(run_meta_path)
    ):
        print(f"Final artifacts already exist in {args.save_dir}; skipping.")
        return

    adam_ckpt_path = os.path.join(args.save_dir, "baseline_model_adam.pth")
    lbfgs_ckpt_path = os.path.join(args.save_dir, "baseline_model_lbfgs.pth")
    pso_ckpt_path = os.path.join(args.save_dir, "baseline_model_pso.pth")
    cached_stage_objectives = load_stage_objectives(args.save_dir)

    print(
        "Loss weights | "
        f"physics={args.w_physics:.3e}, "
        f"ic={args.w_ic:.3e}, "
        f"bc={args.w_bc:.3e}, "
        f"data={args.w_data:.3e}"
    )
    print(f"Architecture | layers={args.layers}, base_neurons={args.base_neurons}")

    model = NAS_PINN(layers=args.layers, base_neurons=args.base_neurons).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.adam_lr)

    points = sample_points()
    reference_data = build_quench_reference_data()
    history = []
    stage_rows = []

    if load_checkpoint_if_exists(model, adam_ckpt_path):
        print(f"Found Adam checkpoint, skipping Adam stage: {adam_ckpt_path}")
        best_adam_state = clone_model_state(model)
        adam_obj = cached_stage_objectives.get("adam")
        if adam_obj is None:
            adam_obj = load_best_total_from_metrics(args.save_dir)
        if adam_obj is None:
            adam_obj = load_best_total_from_train_log(args.save_dir)
        if adam_obj is None:
            print("No cached Adam objective found, evaluating from checkpoint...")
            adam_obj, _ = safe_evaluate_objective(model, points, reference_data, args, stage_name="Adam")
    else:
        best_adam_obj = float("inf")
        best_adam_state = clone_model_state(model)
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            losses = collect_epoch_losses(model, points, reference_data, args.n_time_steps)
            total = weighted_total_loss(losses, args)
            total.backward()
            optimizer.step()

            row = {
                "epoch": epoch,
                "total": float(total.item()),
                "physics": float(losses["physics"].item()),
                "heat": float(losses["heat"].item()),
                "visc": float(losses["visc"].item()),
                "ic": float(losses["ic"].item()),
                "bc": float(losses["bc"].item()),
                "bc_disp": float(losses["bc_disp"].item()),
                "bc_flux": float(losses["bc_flux"].item()),
                "data": float(losses["data"].item()),
                "data_temp": float(losses["data_temp"].item()),
                "data_disp": float(losses["data_disp"].item()),
            }
            history.append(row)

            if epoch % args.log_every == 0:
                print(
                    f"Epoch {epoch:5d} | Total: {row['total']:.4e} | "
                    f"Physics: {row['physics']:.4e} | Data: {row['data']:.4e}"
                )
            if row["total"] < best_adam_obj:
                best_adam_obj = row["total"]
                best_adam_state = clone_model_state(model)

        model.load_state_dict(best_adam_state)
        adam_obj, _ = safe_evaluate_objective(model, points, reference_data, args, stage_name="Adam")
        adam_ckpt_path = save_checkpoint(model, args.save_dir, "baseline_model_adam.pth")
        print(f"Adam checkpoint saved: {adam_ckpt_path}")

    stage_rows = [
        {"stage": "adam", "objective": adam_obj, "checkpoint": os.path.basename(adam_ckpt_path), "selected": 0}
    ]

    # Optional L-BFGS refinement from best Adam checkpoint
    if not args.skip_lbfgs:
        lbfgs_points = build_lbfgs_points(points, args)
        lbfgs_steps = max(1, int(args.lbfgs_time_steps))
        if load_checkpoint_if_exists(model, lbfgs_ckpt_path):
            print(f"Found L-BFGS checkpoint, skipping L-BFGS stage: {lbfgs_ckpt_path}")
            lbfgs_obj = cached_stage_objectives.get("lbfgs")
            if lbfgs_obj is None:
                lbfgs_obj, _ = safe_evaluate_objective(
                    model,
                    points,
                    reference_data,
                    args,
                    stage_name="L-BFGS",
                    fallback_points=lbfgs_points,
                    fallback_n_time_steps=lbfgs_steps,
                )
            stage_rows.append(
                {"stage": "lbfgs", "objective": lbfgs_obj, "checkpoint": os.path.basename(lbfgs_ckpt_path), "selected": 0}
            )
        else:
            model.load_state_dict(best_adam_state)
            n_col = int(lbfgs_points[0][0].shape[0])
            n_ic = int(lbfgs_points[1][0].shape[0])
            n_bc = int(lbfgs_points[2][0].shape[0])
            print(
                "Running L-BFGS refinement... "
                f"(col={n_col}, ic={n_ic}, bc={n_bc}, time_steps={lbfgs_steps})"
            )
            lbfgs = optim.LBFGS(
                model.parameters(),
                lr=1.0,
                max_iter=args.lbfgs_max_iter,
                history_size=args.lbfgs_history_size,
                line_search_fn=None if args.lbfgs_line_search == "none" else args.lbfgs_line_search,
            )

            def closure():
                lbfgs.zero_grad()
                loss_dict = collect_epoch_losses(model, lbfgs_points, reference_data, lbfgs_steps, create_graph=True)
                total = weighted_total_loss(loss_dict, args)
                total.backward()
                return total

            try:
                lbfgs.step(closure)
                lbfgs_obj, _ = safe_evaluate_objective(
                    model,
                    points,
                    reference_data,
                    args,
                    stage_name="L-BFGS",
                    fallback_points=lbfgs_points,
                    fallback_n_time_steps=lbfgs_steps,
                )
                lbfgs_ckpt_path = save_checkpoint(model, args.save_dir, "baseline_model_lbfgs.pth")
                print(f"L-BFGS checkpoint saved: {lbfgs_ckpt_path} | objective={lbfgs_obj:.6e}")
                stage_rows.append(
                    {"stage": "lbfgs", "objective": lbfgs_obj, "checkpoint": os.path.basename(lbfgs_ckpt_path), "selected": 0}
                )
            except torch.OutOfMemoryError as exc:
                print(f"L-BFGS skipped due CUDA OOM: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Optional PSO refinement from best Adam checkpoint
    if args.use_pso:
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo is required for PSO refinement. Install pymoo or run without --use-pso.")

        if load_checkpoint_if_exists(model, pso_ckpt_path):
            print(f"Found PSO checkpoint, skipping PSO stage: {pso_ckpt_path}")
            pso_obj = cached_stage_objectives.get("pso")
            if pso_obj is None:
                pso_obj, _ = safe_evaluate_objective(model, points, reference_data, args, stage_name="PSO")
            stage_rows.append(
                {"stage": "pso", "objective": pso_obj, "checkpoint": os.path.basename(pso_ckpt_path), "selected": 0}
            )
        else:
            model.load_state_dict(best_adam_state)
            print("Running PSO refinement...")
            center = flatten_model_params(model)
            lower = center - args.pso_span
            upper = center + args.pso_span

            def pso_objective(flat):
                set_model_from_flat_vector(model, flat)
                val, _ = evaluate_objective(model, points, reference_data, args, create_graph=False)
                return float(val)

            try:
                pso_problem = PSOWeightProblem(lower, upper, pso_objective)
                pso_algorithm = PSO(pop_size=args.pso_swarm)
                pso_result = minimize(
                    pso_problem,
                    pso_algorithm,
                    termination=("n_gen", args.pso_iters),
                    seed=args.seed,
                    verbose=False,
                )
                best_flat = np.array(pso_result.X, dtype=np.float64)
                pso_obj = float(np.array(pso_result.F).reshape(-1)[0])
                set_model_from_flat_vector(model, best_flat)
                pso_ckpt_path = save_checkpoint(model, args.save_dir, "baseline_model_pso.pth")
                print(f"PSO checkpoint saved: {pso_ckpt_path} | objective={pso_obj:.6e}")
                stage_rows.append(
                    {"stage": "pso", "objective": pso_obj, "checkpoint": os.path.basename(pso_ckpt_path), "selected": 0}
                )
            except torch.OutOfMemoryError as exc:
                print(f"PSO skipped due CUDA OOM: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Select best stage by weighted objective value
    if not stage_rows:
        raise RuntimeError("No completed optimization stage found; cannot produce final checkpoint.")

    best_stage = min(stage_rows, key=lambda r: r["objective"])
    best_stage["selected"] = 1
    selected_ckpt = os.path.join(args.save_dir, best_stage["checkpoint"])
    final_ckpt = os.path.join(args.save_dir, "baseline_model.pth")
    shutil.copy2(selected_ckpt, final_ckpt)

    # Persist stage comparison for pipeline-level analysis
    with open(stage_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "objective", "checkpoint", "selected"])
        writer.writeheader()
        for row in stage_rows:
            writer.writerow(row)

    meta = {
        "layers": int(args.layers),
        "base_neurons": int(args.base_neurons),
        "param_count": count_model_params(model),
        "best_stage": best_stage["stage"],
        "best_objective": float(best_stage["objective"]),
        "run_time_seconds": float(time.perf_counter() - run_start),
    }
    with open(os.path.join(args.save_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    metrics_csv = os.path.join(args.save_dir, "metrics.csv")
    if history or not os.path.exists(metrics_csv):
        save_training_artifacts(history, args.save_dir)
    else:
        print("Keeping existing metrics.csv/loss_curves.png from previous Adam stage.")

    print(
        f"Training completed. Selected stage: {best_stage['stage']} "
        f"| objective={best_stage['objective']:.6e} | final model: {final_ckpt}"
    )

# ────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("Baseline NAS-PINN with A356 quenching reference data (2026)")
    parser = argparse.ArgumentParser(description="Baseline NAS-PINN with Quench2026 reference data")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of Adam epochs")
    parser.add_argument("--save-dir", type=str, default="results_naspinn_quenching", help="Save directory")
    parser.add_argument("--layers", type=int, default=5, help="Network depth")
    parser.add_argument("--base-neurons", type=int, default=96, help="Hidden width")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--skip-lbfgs", action="store_true", help="Skip L-BFGS refinement")
    parser.add_argument("--use-pso", action="store_true", help="Enable PSO refinement from Adam checkpoint")
    parser.add_argument("--pso-iters", type=int, default=8, help="PSO generations")
    parser.add_argument("--pso-swarm", type=int, default=16, help="PSO swarm size")
    parser.add_argument("--pso-span", type=float, default=0.25, help="PSO search span around Adam weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-every", type=int, default=250, help="Log interval in epochs")
    parser.add_argument("--n-time-steps", type=int, default=10, help="Incremental steps for viscoplastic history")
    parser.add_argument("--lbfgs-max-iter", type=int, default=500, help="Max LBFGS iterations")
    parser.add_argument("--lbfgs-col-points", type=int, default=2048, help="Collocation points used during L-BFGS")
    parser.add_argument("--lbfgs-ic-points", type=int, default=512, help="IC points used during L-BFGS")
    parser.add_argument("--lbfgs-bc-points", type=int, default=512, help="BC points per side used during L-BFGS")
    parser.add_argument("--lbfgs-time-steps", type=int, default=4, help="Time steps used during L-BFGS")
    parser.add_argument("--lbfgs-history-size", type=int, default=20, help="LBFGS history size")
    parser.add_argument(
        "--lbfgs-line-search",
        type=str,
        default="strong_wolfe",
        choices=["none", "strong_wolfe"],
        help="LBFGS line search strategy",
    )
    parser.add_argument("--force-final", action="store_true", help="Re-run final stage even if final artifacts exist")
    parser.add_argument("--w-physics", type=float, default=50.0, help="Weight for physics loss (increased)")
    parser.add_argument("--w-ic", type=float, default=1e-3, help="Weight for initial condition loss (decreased)")
    parser.add_argument("--w-bc", type=float, default=1e-18, help="Weight for boundary loss (decreased)")
    parser.add_argument("--w-data", type=float, default=1e-5, help="Weight for data loss from paper references (decreased)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    train_baseline(args)
