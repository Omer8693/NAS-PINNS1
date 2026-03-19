"""
plot_fem_vs_pinn_steps.py — FEM Steps vs PINN Comparison
==========================================================
Thesis visualization:
  FEM  : t=0 → t=1.5s → ... → t=30s  (20 sequential steps, each solved separately)
  PINN : single pass over [0,30s], queryable at any t

Layout:
  Top panel  : T(t) at 4 monitoring points — FEM step dots vs PINN curve
  Bottom panel: |T_FEM - T_PINN| error at each FEM step

Usage:
    cd /home/coder/NAS-PINNS1/NAS-PINNS3
    python level1_single_shot/plot_fem_vs_pinn_steps.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from scipy.integrate import solve_ivp

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.config import (
    T_INIT, T_WATER_INIT, T_END, X_MAX, Y_MAX,
    K_THERMAL, RHO_CP, N_TIME_STEPS,
)
from src.pinn_network import PINNNet

DEVICE  = torch.device("cpu")
OUT_DIR = _ROOT / "level1_single_shot" / "results" / "plots"

# FEM-PINNS stil sabitleri
_FIG_BG  = "#f7f1e3"
_AX_BG   = "#fffdf8"
_SPINE_C = "#d8cfbf"
_TXT_C   = "#2f2a24"
_TICK_C  = "#4c463f"
_FEM_C   = "#1565C0"    # FEM rengi
_PINN_C  = "#C62828"    # PINN rengi
_ERR_C   = "#E65100"    # Hata rengi


def _style(ax):
    ax.set_facecolor(_AX_BG)
    for sp in ax.spines.values():
        sp.set_color(_SPINE_C); sp.set_linewidth(0.8)
    ax.tick_params(colors=_TICK_C, labelsize=8)
    ax.xaxis.label.set_color(_TICK_C)
    ax.yaxis.label.set_color(_TICK_C)


# ── FEM simulation: HTC-based cooling ODE ────────────────────────────────────

def _htc_numpy(T_surf: float, T_w: float = T_WATER_INIT) -> float:
    """
    Surface-temperature-dependent heat transfer coefficient (W/m²K).
    Same formula as Physics_model.WaterQuenchHTC, numpy version.
    """
    T_sat, T_mfb = 100.0, 300.0
    h_conv = 5000.0
    dT = max(T_surf - T_w, 0.1)
    if T_surf < T_sat:
        return h_conv
    T_chf = T_sat + 80.0
    h_chf = 3e6 / (T_chf - T_w + 1.0)
    if T_surf < T_chf:
        return h_conv * (dT / 10.0) ** 2.33
    if T_surf < T_mfb:
        alpha = (T_surf - T_chf) / (T_mfb - T_chf + 1e-6)
        return h_chf * (1 - alpha) + 200.0 * alpha
    T_s_K = T_surf + 273.15
    T_w_K = T_w + 273.15
    return min(200.0 + 5e-9 * (T_s_K**4 - T_w_K**4) / (T_surf - T_w + 1.0), 1e5)


def fem_cooling_ode(t, T, sv_ratio: float) -> list:
    """
    Lumped-parameter heat balance:
      ρCp · dT/dt = -h(T) · (S/V) · (T - T_water)

    S/V = surface-to-volume ratio (higher for points near the boundary)
    """
    h = _htc_numpy(float(T[0]))
    dTdt = -h * sv_ratio * (T[0] - T_WATER_INIT) / RHO_CP
    return [dTdt]


def simulate_fem(sv_ratio: float, n_steps: int = N_TIME_STEPS) -> tuple:
    """
    Solve cooling ODE for a given S/V ratio.
    Each FEM step continues from the previous step's solution.
    Returns: (t_steps[N], T_steps[N]) — value at the start of each step.
    """
    t_steps = np.linspace(0, T_END, n_steps + 1)   # 0..30s, 21 points
    T_fem   = np.zeros(n_steps + 1)
    T_fem[0] = T_INIT

    for i in range(n_steps):
        sol = solve_ivp(
            fem_cooling_ode,
            [t_steps[i], t_steps[i + 1]],
            [T_fem[i]],
            args=(sv_ratio,),
            method="RK45", dense_output=False, rtol=1e-6,
        )
        T_fem[i + 1] = sol.y[0, -1]

    return t_steps, T_fem


# ── Load PINN model ───────────────────────────────────────────────────────────

def load_pinn() -> PINNNet:
    arch_path = _ROOT / "level1_single_shot" / "results" / "bayesian" / "best_arch.json"
    wt_path   = _ROOT / "level1_single_shot" / "results" / "bayesian" / "model.pt"

    with open(arch_path) as f:
        arch = json.load(f)

    net = PINNNet(
        n_input      = arch["n_input"],
        n_output     = arch["n_output"],
        hidden_sizes = arch["neurons"],   # already a list
        activation   = arch["activation"],
        residual     = arch.get("residual", False),
    )
    net.load_state_dict(torch.load(wt_path, map_location="cpu", weights_only=False))
    net.eval()
    return net


def pinn_predict(net: PINNNet, x: float, y: float,
                 t_arr: np.ndarray) -> np.ndarray:
    """Evaluate PINN at a fixed (x,y) point for an array of times t_arr.
    Model input : [t(s), x(m), y(m)] — raw units (t first)
    Model output: temperature directly in °C
    """
    pts = torch.tensor(
        [[ti, x, y] for ti in t_arr],
        dtype=torch.float32,
    )
    with torch.no_grad():
        T_pred = net(pts).squeeze().numpy()
    return T_pred


# ── Monitoring points ─────────────────────────────────────────────────────────

# (x_m, y_m, label, S/V m⁻¹)
# S/V = Σ(1/d_i) — sum of reciprocal distances to boundaries
# Near-surface points: thermal penetration depth ~4 cm (in 30s)
MONITOR_POINTS = [
    (0.02,  0.02,  "Bottom-Left Corner Surface\n(x=0.02m, y=0.02m)",  100.0),  # 2 surfaces
    (0.65,  0.02,  "Bottom Surface Mid\n(x=0.65m, y=0.02m)",           50.0),  # 1 surface
    (0.65,  0.58,  "Top Surface Mid\n(x=0.65m, y=0.58m)",              50.0),  # 1 surface
    (1.28,  0.02,  "Bottom-Right Corner Surface\n(x=1.28m, y=0.02m)", 100.0),  # 2 surfaces
]


# ── Main plot function ────────────────────────────────────────────────────────

def plot_fem_vs_pinn():
    net = load_pinn()
    n_pts   = len(MONITOR_POINTS)
    t_steps = np.linspace(0, T_END, N_TIME_STEPS + 1)   # 21 points (0..30s)
    t_dense = np.linspace(0, T_END, 300)                  # dense curve for PINN

    fig = plt.figure(figsize=(18, 12), facecolor=_FIG_BG)
    fig.suptitle(
        "FEM (Step-by-Step) vs NAS-PINN (Single Pass) — A356 Water Quenching  [0 → 30 s]\n"
        "FEM: ρC_p · dT/dt = −h(T)·(S/V)·(T−T_w)  |  "
        "PINN: 5×151 relu, trained simultaneously over entire domain",
        fontsize=12, color=_TXT_C, y=0.99,
    )

    gs = gridspec.GridSpec(
        2, n_pts,
        height_ratios=[2.5, 1.0],
        hspace=0.42, wspace=0.32,
        left=0.06, right=0.97, top=0.93, bottom=0.06,
    )

    all_fem_T  = []   # [n_pts, n_steps+1]
    all_pinn_T = []   # [n_pts, n_steps+1]
    all_err    = []   # [n_pts, n_steps+1]

    # ── Row 0: T(t) comparison curves ────────────────────────────────────────
    for col, (xp, yp, label, sv) in enumerate(MONITOR_POINTS):
        ax = fig.add_subplot(gs[0, col])
        _style(ax)

        # FEM: ODE solution (20 steps)
        t_fem, T_fem = simulate_fem(sv, N_TIME_STEPS)
        all_fem_T.append(T_fem)

        # PINN: at step points and as a dense curve
        T_pinn_steps  = pinn_predict(net, xp, yp, t_steps)
        T_pinn_dense  = pinn_predict(net, xp, yp, t_dense)
        all_pinn_T.append(T_pinn_steps)
        all_err.append(np.abs(T_fem - T_pinn_steps))

        # PINN continuous curve
        ax.plot(t_dense, T_pinn_dense,
                color=_PINN_C, linewidth=2.0,
                label="PINN (single pass)", zorder=3)

        # FEM step dots
        ax.scatter(t_fem, T_fem,
                   color=_FEM_C, s=55, zorder=5, marker="o",
                   edgecolors="white", linewidths=0.8,
                   label=f"FEM ({N_TIME_STEPS} steps)")

        # FEM step vertical dashed lines
        for ti in t_fem[1:]:
            ax.axvline(ti, color=_FEM_C, linewidth=0.4, linestyle=":", alpha=0.35, zorder=1)

        ax.set_xlim(-0.5, T_END + 0.5)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Temperature (°C)", fontsize=9) if col == 0 else None
        ax.set_title(label, fontsize=10, color=_TXT_C, pad=6)

        # MAE anotasyonu
        mae = float(np.mean(np.abs(T_fem - T_pinn_steps)))
        ax.text(0.97, 0.97,
                f"MAE = {mae:.1f} °C\n"
                f"(x={xp:.2f}, y={yp:.2f})",
                transform=ax.transAxes, va="top", ha="right",
                fontsize=8.5, color=_TXT_C,
                bbox={"facecolor": "#fef6e4", "edgecolor": "#e8c77a",
                      "boxstyle": "round,pad=0.3"})

        if col == 0:
            ax.legend(fontsize=8, loc="upper right",
                      facecolor=_AX_BG, edgecolor=_SPINE_C)

    # ── Row 1: |error| bar at each FEM step ──────────────────────────────────
    for col in range(n_pts):
        ax_e = fig.add_subplot(gs[1, col])
        _style(ax_e)

        err = all_err[col]   # shape (n_steps+1,)
        ax_e.bar(t_steps, err, width=T_END / N_TIME_STEPS * 0.7,
                 color=_ERR_C, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax_e.axhline(np.mean(err), color="#6D4C41", linestyle="--",
                     linewidth=1.2, label=f"Mean {np.mean(err):.1f}°C")
        ax_e.set_xlim(-0.5, T_END + 0.5)
        ax_e.set_xlabel("FEM Step (s)", fontsize=8.5)
        ax_e.set_ylabel("|Δ T| (°C)", fontsize=8.5) if col == 0 else None
        ax_e.set_title("|FEM − PINN| at each step", fontsize=9, color=_TXT_C, pad=4)
        ax_e.legend(fontsize=8, facecolor=_AX_BG, edgecolor=_SPINE_C)

    # ── Bottom corner: summary text box ──────────────────────────────────────
    all_fem_arr  = np.array(all_fem_T)   # [n_pts, n_steps+1]
    all_pinn_arr = np.array(all_pinn_T)
    global_mae   = float(np.mean(np.abs(all_fem_arr - all_pinn_arr)))

    fig.text(
        0.50, 0.005,
        f"FEM: {N_TIME_STEPS} sequential solves  |  "
        f"PINN: 1 pass, entire time domain simultaneously  |  "
        f"Global MAE = {global_mae:.1f} °C  |  "
        f"T₀ = {T_INIT:.0f}°C → T_su = {T_WATER_INIT:.0f}°C  |  "
        f"Alan: {X_MAX}×{Y_MAX} m",
        ha="center", va="bottom", fontsize=9, color=_TXT_C,
        bbox={"facecolor": "#fff7dc", "edgecolor": "#ddc98b",
              "boxstyle": "round,pad=0.3", "alpha": 0.85},
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fem_vs_pinn_steps.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    out = plot_fem_vs_pinn()
    print(f"\nDone → {out}")
