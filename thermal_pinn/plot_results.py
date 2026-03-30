"""
thermal_pinn/plot_results.py
==============================
Publication-quality result figures — Level 8 visual style.

  - plasma / Reds colormaps, fixed global Normalize(20, 540)
  - make_axes_locatable tight colorbars per panel
  - White isoline contours on every T field
  - Red dashed FEM contours overlaid on every PINN panel
  - Cream figure background  (#f7f1e3)
  - True matplotlib 3D surface (FEM colored, PINN wireframe)
  - Per-domain 2×(1+n_arch+1) layout: Reference | Bay | NSGA-II | NSGA-III | Best-err
  - For 3D domains: 2×2 grid per arch  (FEM / PINN / Error / 3D surface)

Reference note
--------------
  - Rectangle, Circle, Annulus, Cylinder, Rectangular, Stacked:
      Analytical eigenfunction series (Carslaw & Jaeger, 1959)
  - LShape 2D/3D:
      Explicit finite-difference solver (cell-centred grid, Robin BC)

Usage
-----
    python thermal_pinn/plot_results.py           # all figures
    python thermal_pinn/plot_results.py --fig 2   # single figure
    python thermal_pinn/plot_results.py --fig 2 --domain rectangle
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent
CKPT_DIR   = ROOT / "checkpoints"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)
(RESULT_DIR / "per_k").mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(ROOT.parent))

from thermal_pinn.network.pinn       import ThermalPINN, ARCH_CONFIGS
from thermal_pinn.physics.domains_2d import make_domain, DOMAINS_2D
from thermal_pinn.physics.domains_3d import make_domain_3d, DOMAINS_3D
from thermal_pinn.training.trainer   import _normalise_coords

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_TOTAL = 30.0
DT_FEM  = 1.5
T_WATER = 20.0
T_INIT  = 540.0
DELTA_T = T_INIT - T_WATER    # 520

# ── Global color / style (Level 8 standard) ──────────────────────────────────
CMAP_T   = "plasma"
CMAP_E   = "Reds"
_FIG_BG  = "#f7f1e3"
_AX_BG   = "#fffdf8"
_SPINE_C = "#d8cfbf"
_TXT_C   = "#2f2a24"
_TICK_C  = "#4c463f"

norm_T_global = Normalize(vmin=T_WATER, vmax=T_INIT)   # for inter-domain comparisons

_ARCH_COLORS = {"bayesian": "#1565C0", "nsga2": "#2E7D32", "nsga3": "#E65100"}
_ARCH_LABELS = {"bayesian": "Bayesian (TPE)", "nsga2": "NSGA-II", "nsga3": "NSGA-III"}
_ARCH_ORDER  = ["bayesian", "nsga2", "nsga3"]

_REF_LABEL = {
    "rectangle":   "Analytical series",
    "circle":      "Analytical Bessel",
    "annulus":     "Analytical Bessel (cross-product)",
    "lshape":      "Explicit FD solver",
    "rectangular": "Analytical product",
    "cylinder":    "Analytical Bessel × plane-wall",
    "stacked":     "Analytical product",
    "lshape3d":    "Explicit FD solver",
}


# ════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ════════════════════════════════════════════════════════════════════════════

def _cb(fig, ax, im, label: str, fs: int = 7):
    """Tight per-panel colorbar via make_axes_locatable."""
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.04)
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=fs, color=_TXT_C)
    cb.ax.tick_params(labelsize=fs - 1, colors=_TICK_C)
    cb.outline.set_edgecolor(_SPINE_C)


def _style(ax):
    ax.set_facecolor(_AX_BG)
    for sp in ax.spines.values():
        sp.set_color(_SPINE_C); sp.set_linewidth(0.7)
    ax.tick_params(colors=_TICK_C, labelsize=7)
    ax.xaxis.label.set_color(_TICK_C)
    ax.yaxis.label.set_color(_TICK_C)


def _adaptive_norm(T_ref, T_pred_list=None, pct_lo=2, pct_hi=98):
    """
    Build adaptive Normalize from the actual data range.
    Uses percentiles to avoid outlier saturation.
    Ensures vmin < vmax.
    """
    vals = T_ref[np.isfinite(T_ref)].ravel()
    if T_pred_list:
        for T_p in T_pred_list:
            vals = np.concatenate([vals, T_p[np.isfinite(T_p)].ravel()])
    if len(vals) == 0:
        return Normalize(vmin=T_WATER, vmax=T_INIT)
    vmin = float(np.percentile(vals, pct_lo))
    vmax = float(np.percentile(vals, pct_hi))
    if vmax - vmin < 5:
        vmin = max(T_WATER, vmin - 10)
        vmax = min(T_INIT,  vmax + 10)
    return Normalize(vmin=vmin, vmax=vmax)


def _adaptive_norm_many(arrays, pct_lo=2, pct_hi=98):
    """Adaptive shared Normalize across many fields."""
    finite = []
    for arr in arrays:
        arr_np = np.asarray(arr, dtype=float)
        vals = arr_np[np.isfinite(arr_np)]
        if vals.size:
            finite.append(vals)
    if not finite:
        return Normalize(vmin=T_WATER, vmax=T_INIT)
    vals = np.concatenate(finite)
    vmin = float(np.percentile(vals, pct_lo))
    vmax = float(np.percentile(vals, pct_hi))
    if vmax - vmin < 5:
        vmin = max(T_WATER, vmin - 10)
        vmax = min(T_INIT, vmax + 10)
    return Normalize(vmin=vmin, vmax=vmax)


def _draw_T(ax, XX, YY, T, norm=None, title: str = "", c_title: str = _TXT_C,
            add_fem_overlay: np.ndarray | None = None, cmap: str | None = None,
            aspect: str = "equal"):
    """
    Draw temperature heatmap with white isoline contours.
    Optionally overlay red dashed FEM contours (for PINN panels).
    norm: pre-computed Normalize (adaptive per-figure).
    """
    if norm is None:
        norm = norm_T_global
    valid = np.isfinite(T)
    T_plot = np.where(valid, T, np.nan)
    im = ax.pcolormesh(XX, YY, T_plot, cmap=cmap or CMAP_T, norm=norm,
                       shading="auto", rasterized=True)
    if valid.any():
        try:
            ax.contour(XX, YY, T_plot, levels=7,
                       colors="white", linewidths=0.4, alpha=0.75)
        except Exception:
            pass
        if add_fem_overlay is not None:
            fem_v = np.where(np.isfinite(add_fem_overlay), add_fem_overlay, np.nan)
            try:
                ax.contour(XX, YY, fem_v, levels=7,
                           colors="#FF4444", linewidths=0.85,
                           alpha=0.9, linestyles="dashed")
            except Exception:
                pass
    ax.set_aspect(aspect)
    _style(ax)
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold",
                     color=c_title, pad=3)
    return im


def _draw_err(ax, XX, YY, err, title: str = "", norm=None, aspect: str = "equal"):
    """Draw absolute-error heatmap with black contours."""
    err_plot = np.where(np.isfinite(err), err, np.nan)
    fin = err_plot[np.isfinite(err_plot)]
    if norm is None:
        err_max = max(float(np.percentile(fin, 97)) if len(fin) > 0 else 1.0, 0.5)
        norm = Normalize(vmin=0, vmax=err_max)
    im = ax.pcolormesh(XX, YY, err_plot, cmap=CMAP_E, norm=norm,
                       shading="auto", rasterized=True)
    if len(fin) > 0:
        try:
            ax.contour(XX, YY, err_plot, levels=4,
                       colors="#333333", linewidths=0.35, alpha=0.6)
        except Exception:
            pass
    ax.set_aspect(aspect)
    _style(ax)
    err_peak = float(np.nanmax(fin)) if len(fin) > 0 else np.nan
    default_title = "|Error|" if np.isnan(err_peak) else f"|Error|  max={err_peak:.1f}°C"
    ax.set_title(title or default_title,
                 fontsize=8, fontweight="bold", color="#B71C1C", pad=3)
    return im, norm


# ════════════════════════════════════════════════════════════════════════════
# Registry / checkpoint loaders
# ════════════════════════════════════════════════════════════════════════════

def load_registry() -> list[dict]:
    with open(CKPT_DIR / "registry.json") as f:
        return json.load(f)


def load_metrics(domain: str, arch: str, k: int, dim: int) -> dict | None:
    p = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}_metrics.json"
    return json.load(open(p)) if p.exists() else None


def load_model(domain: str, arch: str, k: int, dim: int) -> ThermalPINN:
    ckpt_path = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}.pt"
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = ThermalPINN(dim=dim, arch=arch).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _k_skip_schedule(kv: int) -> dict:
    """Return the actual training/evaluation schedule implied by k."""
    dt_window = kv * DT_FEM
    starts = np.arange(0.0, T_TOTAL - dt_window / 2, dt_window)
    windows = [(float(t0), float(min(t0 + dt_window, T_TOTAL))) for t0 in starts]
    ref_times = sorted({
        0.0, T_TOTAL,
        *[w[0] for w in windows],
        *[w[1] for w in windows],
    })
    return {
        "k": kv,
        "dt_window": dt_window,
        "windows": windows,
        "ref_times": ref_times,
        "n_windows": len(windows),
        "n_ref_times": len(ref_times),
    }


def best_k(registry: list[dict], domain: str, arch: str, dim: int) -> int:
    e = next((r for r in registry
              if r["domain"] == domain and r["arch"] == arch
              and r["dim"] == dim), None)
    return e["best_k"] if e else 1


# ════════════════════════════════════════════════════════════════════════════
# Grid evaluation helpers
# ════════════════════════════════════════════════════════════════════════════

def _is_centered_2d(domain) -> bool:
    """True for circle / annulus (centered at origin in 2D)."""
    return hasattr(domain, "R_out") or (hasattr(domain, "R") and not hasattr(domain, "Lx"))


def _is_cylinder_3d(domain) -> bool:
    """True for Cylinder3D (has R + Hz, centered at origin in x/y)."""
    return hasattr(domain, "R") and (hasattr(domain, "Hz") or hasattr(domain, "Lz"))


def _make_2d_grid(domain, n_grid: int):
    if hasattr(domain, "R_out"):                              # Annulus
        L = domain.R_out
    elif hasattr(domain, "R") and not hasattr(domain, "Lx"): # Circle
        L = domain.R
    else:
        L = None
    if L is not None:
        x_lin = np.linspace(-L, L, n_grid)
        y_lin = np.linspace(-L, L, n_grid)
    else:
        x_lin = np.linspace(0, domain.Lx, n_grid)
        y_lin = np.linspace(0, domain.Ly, n_grid)
    return np.meshgrid(x_lin, y_lin, indexing="ij")


def _make_3d_midgrid(domain, n_grid: int):
    """Return XX, YY, z_mid for a z=mid-plane slice of the 3D domain."""
    if _is_cylinder_3d(domain):
        R = domain.R
        x_lin = np.linspace(-R, R, n_grid)
        y_lin = np.linspace(-R, R, n_grid)
    else:
        x_lin = np.linspace(0, getattr(domain, "Lx", 1.0), n_grid)
        y_lin = np.linspace(0, getattr(domain, "Ly", 1.0), n_grid)
    Lz    = getattr(domain, "Lz", getattr(domain, "Hz", 1.0))
    z_mid = Lz / 2.0
    XX, YY = np.meshgrid(x_lin, y_lin, indexing="ij")
    return XX, YY, z_mid


def _mask_T(T_vals, domain, xi, yi, zi=None):
    """
    Replace out-of-domain values with NaN.
    - Cylinder3D: r > R → NaN
    - Annulus2D:  r < R_in or r > R_out → NaN (already NaN from series)
    - Circle2D:   T_water outside (series clamps to T_water, keep as is)
    - Others:     trust the domain.T() method
    """
    T = np.array(T_vals, dtype=float)
    if zi is not None and _is_cylinder_3d(domain):
        # Cylinder is centred at (0,0) in x/y
        r = np.sqrt(xi**2 + yi**2)
        T = np.where(r > domain.R + 1e-9, np.nan, T)
    return T


def _tau_params(t_query: float, k: int, dt_fem: float = 1.5):
    """Return (t_start, t_end, tau_val) for t_query with k-skip windows."""
    dt_window = k * dt_fem
    t_start   = ((t_query - 1e-9) // dt_window) * dt_window
    t_start   = float(np.clip(t_start, 0.0, 30.0 - dt_window))
    t_end     = min(t_start + dt_window, 30.0)
    tau_val   = float(np.clip((t_query - t_start) / (t_end - t_start), 0.0, 1.0))
    return t_start, t_end, tau_val


def _pinn_forward(model, coords_np, tau_val, T_ic_np):
    coords_norm   = _normalise_coords(coords_np, model._domain_ref
                                       if hasattr(model, "_domain_ref") else None)
    # _normalise_coords needs domain — pass from outside
    raise NotImplementedError  # use _pinn_forward_full below


def _pinn_forward_full(model, coords_np, tau_val, T_ic_np, domain):
    coords_norm = _normalise_coords(coords_np, domain)
    theta_ic    = np.clip((T_ic_np - T_WATER) / DELTA_T, 0.0, 1.2)
    theta_ic    = theta_ic.reshape(-1, 1).astype(np.float32)
    tau_arr     = np.full((len(coords_np), 1), tau_val, dtype=np.float32)
    def _t(a): return torch.tensor(a, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        theta_pred = model(_t(coords_norm), _t(tau_arr), _t(theta_ic)
                           ).cpu().numpy().flatten()
    return T_WATER + DELTA_T * theta_pred


def _clean_T(T_raw, domain=None, xi=None, yi=None, zi=None):
    """
    Sanitise raw domain.T() output:
      - Cylinder3D: mask r > R with NaN (eigenfunction series diverges outside)
      - All: replace physically impossible values (<-50 or >600°C) with NaN
    """
    T = np.array(T_raw, dtype=float)
    if domain is not None and zi is not None and _is_cylinder_3d(domain):
        r = np.sqrt(xi**2 + yi**2)
        T = np.where(r > domain.R + 1e-9, np.nan, T)
    # Physical plausibility guard
    T = np.where((T < T_WATER - 5) | (T > T_INIT + 10), np.nan, T)
    return T


def eval_grid_2d(model: ThermalPINN, domain, k: int,
                 t_query: float, n_grid: int = 80) -> dict:
    """Evaluate PINN + reference on uniform 2D grid at t_query."""
    t_start, t_end, tau_val = _tau_params(t_query, k)
    XX, YY = _make_2d_grid(domain, n_grid)
    xi, yi = XX.ravel(), YY.ravel()
    coords = np.stack([xi, yi], axis=1).astype(np.float32)

    T_ic  = _clean_T(domain.T(xi, yi, t_start))
    T_ref = _clean_T(domain.T(xi, yi, t_query))
    T_ic  = np.where(np.isfinite(T_ic), T_ic, T_WATER)

    T_pred = _pinn_forward_full(model, coords, tau_val, T_ic, domain)
    T_pred = np.where(np.isfinite(T_ref), T_pred, np.nan)

    valid = np.isfinite(T_ref)
    mae = float(np.mean(np.abs(T_pred[valid] - T_ref[valid]))) if valid.any() else np.nan
    l2  = (float(np.linalg.norm(T_pred[valid] - T_ref[valid]) /
                 (np.linalg.norm(T_ref[valid] - T_WATER) + 1e-10))
           if valid.any() else np.nan)
    return {
        "xx": XX, "yy": YY,
        "T_pred": T_pred.reshape(n_grid, n_grid),
        "T_ref":  T_ref.reshape(n_grid, n_grid),
        "err":    np.abs(T_pred - T_ref).reshape(n_grid, n_grid),
        "mae": mae, "l2": l2, "t_query": t_query,
    }


def eval_grid_3d_mid(model: ThermalPINN, domain, k: int,
                     t_query: float, n_grid: int = 50) -> dict:
    """Evaluate PINN + reference on z=mid plane of 3D domain."""
    t_start, t_end, tau_val = _tau_params(t_query, k)
    XX, YY, z_mid = _make_3d_midgrid(domain, n_grid)
    xi, yi = XX.ravel(), YY.ravel()
    zi     = np.full_like(xi, z_mid)
    coords = np.stack([xi, yi, zi], axis=1).astype(np.float32)

    T_ic  = _clean_T(domain.T(xi, yi, zi, t_start), domain, xi, yi, zi)
    T_ref = _clean_T(domain.T(xi, yi, zi, t_query), domain, xi, yi, zi)
    T_ic  = np.where(np.isfinite(T_ic), T_ic, T_WATER)

    T_pred = _pinn_forward_full(model, coords, tau_val, T_ic, domain)
    T_pred = np.where(np.isfinite(T_ref), T_pred, np.nan)

    valid = np.isfinite(T_ref)
    mae = float(np.mean(np.abs(T_pred[valid] - T_ref[valid]))) if valid.any() else np.nan
    l2  = (float(np.linalg.norm(T_pred[valid] - T_ref[valid]) /
                 (np.linalg.norm(T_ref[valid] - T_WATER) + 1e-10))
           if valid.any() else np.nan)
    return {
        "xx": XX, "yy": YY, "z_mid": z_mid,
        "T_pred": T_pred.reshape(n_grid, n_grid),
        "T_ref":  T_ref.reshape(n_grid, n_grid),
        "err":    np.abs(T_pred - T_ref).reshape(n_grid, n_grid),
        "mae": mae, "l2": l2, "t_query": t_query,
        "n_grid": n_grid,
    }


def eval_grid_3d_slices(model: ThermalPINN, domain, k: int,
                        t_query: float, n_grid: int = 40) -> dict:
    """Evaluate PINN on three orthogonal mid-plane slices (xy/xz/yz)."""
    t_start, t_end, tau_val = _tau_params(t_query, k)

    if _is_cylinder_3d(domain):                         # Cylinder: centred at (0,0)
        R  = domain.R
        Lx = Ly = 2 * R; x0 = y0 = -R
    else:
        Lx = getattr(domain, "Lx", 1.0); Ly = getattr(domain, "Ly", 1.0)
        x0 = y0 = 0.0
    Lz = getattr(domain, "Lz", getattr(domain, "Hz", 1.0))

    x_lin = np.linspace(x0,  x0 + Lx, n_grid)
    y_lin = np.linspace(y0,  y0 + Ly, n_grid)
    z_lin = np.linspace(0.0, Lz,      n_grid)
    xm = x_lin[n_grid // 2]; ym = y_lin[n_grid // 2]; zm = z_lin[n_grid // 2]

    slices_def = {
        "xy": (x_lin, y_lin, zm, 2, "x [m]", "y [m]", f"z={zm:.2f}m"),
        "xz": (x_lin, z_lin, ym, 1, "x [m]", "z [m]", f"y={ym:.2f}m"),
        "yz": (y_lin, z_lin, xm, 0, "y [m]", "z [m]", f"x={xm:.2f}m"),
    }

    out = {}
    for sname, (v1, v2, fix, fix_ax, l1, l2_lab, fix_str) in slices_def.items():
        V1, V2 = np.meshgrid(v1, v2, indexing="ij")
        v1f, v2f = V1.ravel(), V2.ravel()
        vfix = np.full_like(v1f, fix)
        if   fix_ax == 0: xi, yi, zi = vfix, v1f, v2f
        elif fix_ax == 1: xi, yi, zi = v1f, vfix, v2f
        else:             xi, yi, zi = v1f, v2f, vfix

        coords = np.stack([xi, yi, zi], axis=1).astype(np.float32)
        T_ic  = _clean_T(domain.T(xi, yi, zi, t_start), domain, xi, yi, zi)
        T_ref = _clean_T(domain.T(xi, yi, zi, t_query), domain, xi, yi, zi)
        T_ic  = np.where(np.isfinite(T_ic), T_ic, T_WATER)
        T_pred = _pinn_forward_full(model, coords, tau_val, T_ic, domain)
        T_pred = np.where(np.isfinite(T_ref), T_pred, np.nan)

        out[sname] = {
            "V1": V1, "V2": V2,
            "T_pred": T_pred.reshape(n_grid, n_grid),
            "T_ref":  T_ref.reshape(n_grid, n_grid),
            "err":    np.abs(T_pred - T_ref).reshape(n_grid, n_grid),
            "xlabel": l1, "ylabel": l2_lab, "title_suffix": fix_str,
        }
    return out


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — k-sweep MAE comparison
# ════════════════════════════════════════════════════════════════════════════

def fig1_k_sweep(registry: list[dict]):
    """Bar-chart comparing mean MAE across k values for each (domain, arch)."""
    print("  Building Fig 1 — k-sweep MAE …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(_FIG_BG)
    fig.suptitle("k-Skip MAE Comparison — 2D Domains (left) | 3D Domains (right)",
                 fontsize=12, fontweight="bold", color=_TXT_C)

    for col, (dim, dom_list) in enumerate([(2, list(DOMAINS_2D)), (3, list(DOMAINS_3D))]):
        ax = axes[col]
        ax.set_facecolor(_AX_BG)
        for sp in ax.spines.values():
            sp.set_color(_SPINE_C)
        ax.tick_params(colors=_TICK_C)

        all_k = sorted({1, 2, 3, 4, 5})
        x_ticks, x_labels = [], []
        x_pos = 0.0

        for dname in dom_list:
            for ai, arch in enumerate(_ARCH_ORDER):
                maes = []
                for k in all_k:
                    mp = CKPT_DIR / f"{dname}_{arch}_k{k}_dim{dim}_metrics.json"
                    if mp.exists():
                        m = json.load(open(mp))
                        maes.append((k, m["mean_mae"]))
                if not maes:
                    continue
                xs = [x_pos + i * 0.18 for i, _ in enumerate(maes)]
                ys = [m for _, m in maes]
                ks = [k for k, _ in maes]
                bars = ax.bar(xs, ys, width=0.15,
                              color=_ARCH_COLORS[arch], alpha=0.82,
                              label=_ARCH_LABELS[arch] if dname == dom_list[0] else "_")
                for x, y, kv in zip(xs, ys, ks):
                    ax.text(x, y + 0.3, f"k{kv}", ha="center",
                            fontsize=5.5, color=_TICK_C)
                x_pos += len(maes) * 0.18 + 0.1

            x_ticks.append(x_pos - 0.9 * (len(_ARCH_ORDER)) * 5 * 0.18 / 2 - 0.15)
            x_labels.append(dname)
            x_pos += 0.35

        ax.set_ylabel("Mean MAE [°C]", fontsize=9, color=_TXT_C)
        ax.set_title(f"{'2D' if dim == 2 else '3D'} Domains", fontsize=10,
                     color=_TXT_C, fontweight="bold")
        if x_ticks:
            ax.set_xticks([])
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen and not l.startswith("_"):
                seen[l] = h
        if seen:
            ax.legend(seen.values(), seen.keys(), fontsize=8, framealpha=0.85)
        ax.tick_params(colors=_TICK_C, labelsize=8)
        ax.grid(axis="y", alpha=0.3, color=_SPINE_C)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    p = RESULT_DIR / "fig1_k_sweep.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close(fig)
    print(f"    → {p.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — 2D domain heatmaps (Level 8 style)
# ════════════════════════════════════════════════════════════════════════════

def _plot_domain_2d_figure(domain_name: str, registry: list[dict],
                           t_query: float = 29.0, n_grid: int = 80):
    """
    One figure per 2D domain.
    Layout: cols = [Ref | Bayesian | NSGA-II | NSGA-III | Best-error]
    Row 0: T fields  (plasma, white isolines, red FEM overlay on PINN cols)
    Row 1: |Error|   (Reds)
    Plus L2 ranking bar in a narrow top strip.
    """
    n_arch = len(_ARCH_ORDER)
    n_cols = 1 + n_arch + 1   # Ref + 3 archs + best-error
    fig_w  = 3.5 * n_cols
    fig_h  = 8.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(_FIG_BG)
    gs  = gridspec.GridSpec(3, n_cols, figure=fig,
                            height_ratios=[0.18, 1, 1],
                            hspace=0.42, wspace=0.35)

    domain = make_domain(domain_name)
    ref_label = _REF_LABEL.get(domain_name, "Analytical / FD reference")

    # Collect predictions
    grids = {}
    for arch in _ARCH_ORDER:
        k   = best_k(registry, domain_name, arch, dim=2)
        ckp = CKPT_DIR / f"{domain_name}_{arch}_k{k}_dim2.pt"
        if not ckp.exists():
            continue
        model = load_model(domain_name, arch, k, dim=2)
        grids[arch] = eval_grid_2d(model, domain, k, t_query, n_grid)

    if not grids:
        plt.close(fig)
        return

    # Reference grid (use any arch's xx/yy)
    first = next(iter(grids.values()))
    XX, YY = first["xx"], first["yy"]
    T_ref  = first["T_ref"]

    # ── Adaptive norm (shared across all T panels in this figure) ────────
    norm_fig = _adaptive_norm(T_ref, [grids[a]["T_pred"] for a in grids])

    # ── Top strip: L2 ranking bar ─────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, :])
    ax_bar.set_facecolor(_FIG_BG)
    for sp in ax_bar.spines.values():
        sp.set_visible(False)
    l2s = {a: grids[a]["l2"] for a in grids}
    l2_sorted = sorted(l2s.items(), key=lambda x: x[1])
    colors = [_ARCH_COLORS[a] for a, _ in l2_sorted]
    labels = [f"{_ARCH_LABELS[a]}  L2={v:.4f}" for a, v in l2_sorted]
    ax_bar.barh(range(len(l2_sorted)), [v for _, v in l2_sorted],
                color=colors, height=0.55)
    ax_bar.set_yticks(range(len(l2_sorted)))
    ax_bar.set_yticklabels(labels, fontsize=8, color=_TXT_C)
    ax_bar.set_xlabel("Relative L2 error", fontsize=8, color=_TXT_C)
    ax_bar.tick_params(colors=_TICK_C, labelsize=7)
    ax_bar.set_title(
        f"{domain_name.capitalize()} 2D  —  t = {t_query:.0f} s   ({ref_label})\n"
        f"T range: {norm_fig.vmin:.0f}–{norm_fig.vmax:.0f}°C",
        fontsize=10, fontweight="bold", color=_TXT_C, pad=4)

    # ── Row 1: T fields ───────────────────────────────────────────────────
    # Col 0: Reference
    ax_r = fig.add_subplot(gs[1, 0])
    im_r = _draw_T(ax_r, XX, YY, T_ref, norm_fig,
                   title=f"Reference\nT̄={np.nanmean(T_ref):.0f}°C")
    _cb(fig, ax_r, im_r, "T [°C]")
    ax_r.set_xlabel("x [m]", fontsize=7, color=_TICK_C)
    ax_r.set_ylabel("y [m]", fontsize=7, color=_TICK_C)

    # Cols 1–3: arch predictions
    for ci, arch in enumerate(_ARCH_ORDER):
        if arch not in grids:
            continue
        g  = grids[arch]
        ax = fig.add_subplot(gs[1, ci + 1])
        im = _draw_T(ax, XX, YY, g["T_pred"], norm_fig,
                     title=(f"{_ARCH_LABELS[arch]}\n"
                            f"MAE={g['mae']:.2f}°C  L2={g['l2']:.4f}"),
                     c_title=_ARCH_COLORS[arch],
                     add_fem_overlay=T_ref)
        _cb(fig, ax, im, "T [°C]")
        ax.set_xlabel("x [m]", fontsize=7, color=_TICK_C)

    # Col 4: best-error
    best_arch = min(grids, key=lambda a: grids[a]["mae"])
    g_best    = grids[best_arch]
    ax_e      = fig.add_subplot(gs[1, n_arch + 1])
    im_e, _   = _draw_err(ax_e, XX, YY, g_best["err"],
                           title=f"Best |Error|\n({_ARCH_LABELS[best_arch]})")
    _cb(fig, ax_e, im_e, "|ΔT| [°C]")
    ax_e.set_xlabel("x [m]", fontsize=7, color=_TICK_C)

    # ── Row 2: error maps for all archs ──────────────────────────────────
    ax_re = fig.add_subplot(gs[2, 0])
    ax_re.set_visible(False)

    for ci, arch in enumerate(_ARCH_ORDER):
        if arch not in grids:
            continue
        g  = grids[arch]
        ax = fig.add_subplot(gs[2, ci + 1])
        im_err, norm_e = _draw_err(ax, XX, YY, g["err"])
        _cb(fig, ax, im_err, "|ΔT| [°C]")
        ax.set_xlabel("x [m]", fontsize=7, color=_TICK_C)

    # Aggregate error ax
    ax_ea = fig.add_subplot(gs[2, n_arch + 1])
    # Show relative error (%) of best arch
    rel_err = 100 * g_best["err"] / (np.abs(T_ref) + 1e-6)
    rel_err = np.where(np.isfinite(rel_err), rel_err, np.nan)
    re_max  = max(float(np.nanpercentile(rel_err[np.isfinite(rel_err)], 97))
                  if np.any(np.isfinite(rel_err)) else 1.0, 0.1)
    from matplotlib.colors import Normalize as N_
    im_re   = ax_ea.pcolormesh(XX, YY, rel_err,
                                cmap=CMAP_E, norm=N_(0, re_max),
                                shading="auto", rasterized=True)
    ax_ea.set_aspect("equal")
    _style(ax_ea)
    ax_ea.set_title(f"Relative error [%]\n({_ARCH_LABELS[best_arch]})",
                    fontsize=8, color="#B71C1C", fontweight="bold")
    _cb(fig, ax_ea, im_re, "err [%]")
    ax_ea.set_xlabel("x [m]", fontsize=7, color=_TICK_C)

    # Legend line for red dashed FEM contours
    from matplotlib.lines import Line2D
    fig.legend(
        handles=[
            Line2D([0],[0], color="white",  lw=0.5, label="— white: T isolines"),
            Line2D([0],[0], color="#FF4444", lw=0.9, linestyle="--",
                   label="— — red: reference contours (PINN panels)"),
        ],
        loc="lower center", ncol=2, fontsize=8,
        framealpha=0.9, facecolor=_FIG_BG,
        bbox_to_anchor=(0.5, -0.02),
    )

    out = RESULT_DIR / f"fig2_{domain_name}_2d_t{t_query:.0f}s.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close(fig)
    print(f"    → {out.name}")


def fig2_domains_2d(registry: list[dict], t_query: float = 29.0):
    print("  Building Fig 2 — 2D domain heatmaps …")
    for dname in DOMAINS_2D:
        _plot_domain_2d_figure(dname, registry, t_query)


def fig2_domains_2d_compare(registry: list[dict], t_query: float = 15.0,
                            arch: str = "bayesian", n_grid: int = 80):
    """
    Level-8-style 2D domain comparison:
    rows = domains, cols = [Reference | PINN | |Error|]
    """
    print(f"  Building Fig 2 — 2D domain compare ({arch}) …")

    rows = []
    for dname in DOMAINS_2D:
        k = best_k(registry, dname, arch, dim=2)
        ckp = CKPT_DIR / f"{dname}_{arch}_k{k}_dim2.pt"
        if not ckp.exists():
            continue
        domain = make_domain(dname)
        model  = load_model(dname, arch, k, dim=2)
        rows.append({
            "name": dname,
            "label": dname.capitalize(),
            "color": getattr(domain, "color", _TXT_C),
            "k": k,
            "grid": eval_grid_2d(model, domain, k, t_query, n_grid=n_grid),
        })

    if not rows:
        print("    no 2D checkpoints found.")
        return

    norm_T = _adaptive_norm_many(
        [row["grid"]["T_ref"] for row in rows] +
        [row["grid"]["T_pred"] for row in rows]
    )
    err_vals = np.concatenate([row["grid"]["err"].ravel() for row in rows])
    err_vals = err_vals[np.isfinite(err_vals)]
    norm_e   = Normalize(vmin=0, vmax=max(float(np.nanpercentile(err_vals, 97)), 1.0))

    fig, axes = plt.subplots(len(rows), 3, figsize=(11.8, 2.9 * len(rows)))
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor(_FIG_BG)
    fig.suptitle(
        f"2D Domain Comparison — {_ARCH_LABELS[arch]}  |  t = {t_query:.0f} s\n"
        "Reference | PINN prediction | absolute error",
        fontsize=11, fontweight="bold", color=_TXT_C,
    )

    col_titles = ["Reference", "PINN prediction", "|Reference − PINN|"]
    for ci, title in enumerate(col_titles):
        axes[0, ci].set_title(title, fontsize=9, fontweight="bold", color=_TXT_C, pad=4)

    im_T = None
    im_E = None
    for ri, row in enumerate(rows):
        g = row["grid"]
        XX, YY = g["xx"], g["yy"]
        T_ref, T_pred, err = g["T_ref"], g["T_pred"], g["err"]

        ax_f = axes[ri, 0]
        ax_p = axes[ri, 1]
        ax_e = axes[ri, 2]

        im_T = _draw_T(ax_f, XX, YY, T_ref, norm_T, cmap="YlOrRd")
        ax_f.set_ylabel(row["label"], fontsize=8.5, fontweight="bold", color=row["color"])

        im_T = _draw_T(
            ax_p, XX, YY, T_pred, norm_T,
            title=f"k={row['k']}  MAE={g['mae']:.2f}°C",
            c_title=_ARCH_COLORS[arch],
            add_fem_overlay=T_ref,
            cmap="YlOrRd",
        )
        im_E, _ = _draw_err(ax_e, XX, YY, err, norm=norm_e)

        for ax in (ax_f, ax_p, ax_e):
            ax.set_xlabel("x [m]", fontsize=7, color=_TICK_C)
            ax.tick_params(labelsize=6)
        ax_f.set_ylabel(row["label"], fontsize=8.5, fontweight="bold", color=row["color"])
        ax_p.set_ylabel("y [m]", fontsize=7, color=_TICK_C)

    cbar_T = fig.colorbar(im_T, ax=axes[:, :2].ravel().tolist(), shrink=0.97, pad=0.01)
    cbar_T.set_label("T [°C]", color=_TXT_C)
    cbar_T.ax.tick_params(labelsize=8, colors=_TICK_C)
    cbar_E = fig.colorbar(im_E, ax=axes[:, 2].ravel().tolist(), shrink=0.97, pad=0.01)
    cbar_E.set_label("|ΔT| [°C]", color=_TXT_C)
    cbar_E.ax.tick_params(labelsize=8, colors=_TICK_C)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULT_DIR / f"fig2_2d_domain_compare_{arch}_t{t_query:.0f}s.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close(fig)
    print(f"    → {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — 3D domain figures (Level 8 style — 2×2 grid per arch + 3D surface)
# ════════════════════════════════════════════════════════════════════════════

def _plot_domain_3d_figure(domain_name: str, registry: list[dict],
                           t_query: float = 29.0, n_grid: int = 50,
                           archs: list[str] | None = None):
    """
    One figure per 3D domain, one page per architecture.
    Each page is a 2×2 grid:
      [FEM z-mid]  [PINN z-mid + FEM overlay]
      [|Error|]    [3D matplotlib surface: FEM colored + PINN wireframe]
    """
    domain = make_domain_3d(domain_name)

    arch_list = archs or list(_ARCH_ORDER)

    for arch in arch_list:
        k   = best_k(registry, domain_name, arch, dim=3)
        ckp = CKPT_DIR / f"{domain_name}_{arch}_k{k}_dim3.pt"
        if not ckp.exists():
            continue

        model = load_model(domain_name, arch, k, dim=3)
        g     = eval_grid_3d_mid(model, domain, k, t_query, n_grid)
        XX, YY, T_ref, T_pred = g["xx"], g["yy"], g["T_ref"], g["T_pred"]
        err   = g["err"]
        z_mid = g["z_mid"]

        fig = plt.figure(figsize=(14, 11))
        fig.patch.set_facecolor(_FIG_BG)
        gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.35)
        fig.suptitle(
            f"{domain_name.capitalize()} 3D  —  {_ARCH_LABELS[arch]}"
            f"   k={k}  |  t = {t_query:.0f} s\n"
            f"MAE = {g['mae']:.2f}°C   L2 = {g['l2']:.4f}"
            f"   (z = {z_mid:.2f} m slice)",
            fontsize=11, fontweight="bold", color=_TXT_C,
        )

        norm_fig = _adaptive_norm(T_ref, [T_pred])

        # ── Panel A: FEM (reference) ──────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = _draw_T(ax1, XX, YY, T_ref, norm_fig,
                      title=f"Reference (analytical)  T̄={np.nanmean(T_ref):.0f}°C")
        _cb(fig, ax1, im1, "T [°C]")
        ax1.set_xlabel("x [m]", fontsize=8, color=_TICK_C)
        ax1.set_ylabel("y [m]", fontsize=8, color=_TICK_C)

        # ── Panel B: PINN + red dashed FEM overlay ────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = _draw_T(ax2, XX, YY, T_pred, norm_fig,
                      title=(f"PINN  T̄={np.nanmean(T_pred):.0f}°C   "
                             f"MAE={g['mae']:.2f}°C"),
                      c_title=_ARCH_COLORS[arch],
                      add_fem_overlay=T_ref)
        _cb(fig, ax2, im2, "T [°C]")
        ax2.set_xlabel("x [m]", fontsize=8, color=_TICK_C)
        ax2.set_ylabel("y [m]", fontsize=8, color=_TICK_C)

        # ── Panel C: |Error| ──────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        im3, _ = _draw_err(ax3, XX, YY, err)
        _cb(fig, ax3, im3, "|ΔT| [°C]")
        ax3.set_xlabel("x [m]", fontsize=8, color=_TICK_C)
        ax3.set_ylabel("y [m]", fontsize=8, color=_TICK_C)

        # ── Panel D: 3D matplotlib surface ───────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1], projection="3d")
        ax4.set_facecolor(_AX_BG)
        valid = np.isfinite(T_ref)
        if valid.any():
            T_fem_plot  = np.where(valid, T_ref,  float(np.nanmean(T_ref)))
            T_pred_plot = np.where(valid, T_pred, float(np.nanmean(T_pred)))

            ax4.plot_surface(
                XX, YY, T_fem_plot,
                facecolors=plt.cm.plasma(norm_fig(T_fem_plot)),
                alpha=0.88, linewidth=0, antialiased=True,
            )
            ax4.plot_wireframe(
                XX, YY, T_pred_plot,
                color=_ARCH_COLORS[arch],
                alpha=0.35, linewidth=0.5,
                rstride=max(1, n_grid // 12),
                cstride=max(1, n_grid // 12),
            )

        ax4.set_title("3D Surface: Reference (color) + PINN (wire)",
                      fontsize=8.5, fontweight="bold", color=_TXT_C)
        ax4.set_xlabel("x [m]", fontsize=7, color=_TICK_C)
        ax4.set_ylabel("y [m]", fontsize=7, color=_TICK_C)
        ax4.set_zlabel("T [°C]",fontsize=7, color=_TICK_C)
        ax4.tick_params(labelsize=6)
        ax4.view_init(elev=30, azim=-60)
        ax4.set_zlim(T_WATER - 20, T_INIT + 20)

        from matplotlib.lines import Line2D
        fig.legend(
            handles=[
                Line2D([0],[0], color="white",  lw=0.5, label="— white: T isolines"),
                Line2D([0],[0], color="#FF4444", lw=0.9, linestyle="--",
                       label="— — red: reference contours on PINN panel"),
                Line2D([0],[0], color=_ARCH_COLORS[arch], lw=0.8,
                       label=f"PINN wireframe ({_ARCH_LABELS[arch]})"),
            ],
            loc="lower center", ncol=3, fontsize=8,
            framealpha=0.9, facecolor=_FIG_BG,
            bbox_to_anchor=(0.5, -0.02),
        )

        out = RESULT_DIR / f"fig3_{domain_name}_3d_{arch}_t{t_query:.0f}s.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
        plt.close(fig)
        print(f"    → {out.name}")


def fig3_domains_3d(registry: list[dict], t_query: float = 29.0):
    print("  Building Fig 3 — 3D domain figures …")
    for dname in DOMAINS_3D:
        _plot_domain_3d_figure(dname, registry, t_query)


def fig3_domains_3d_single_arch(registry: list[dict], t_query: float = 15.0,
                                arch: str = "bayesian"):
    """True 3D surface figures for one architecture only."""
    print(f"  Building Fig 3 — 3D surface figures ({arch}) …")
    for dname in DOMAINS_3D:
        _plot_domain_3d_figure(dname, registry, t_query, archs=[arch])


# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — Summary heatmap table (best-k MAE)
# ════════════════════════════════════════════════════════════════════════════

def fig4_summary_table(registry: list[dict]):
    """Heatmap table: rows=domains, cols=archs, cell=best-k MAE."""
    print("  Building Fig 4 — summary table …")

    all_domains = list(DOMAINS_2D) + list(DOMAINS_3D)
    all_archs   = _ARCH_ORDER

    mae_2d = np.full((len(DOMAINS_2D), 3), np.nan)
    mae_3d = np.full((len(DOMAINS_3D), 3), np.nan)
    bk_2d  = np.full_like(mae_2d, 0, dtype=int)
    bk_3d  = np.full_like(mae_3d, 0, dtype=int)

    for ri, dname in enumerate(DOMAINS_2D):
        for ci, arch in enumerate(all_archs):
            e = next((r for r in registry
                      if r["domain"] == dname and r["arch"] == arch
                      and r["dim"] == 2), None)
            if e:
                mae_2d[ri, ci] = e["best_mae"]
                bk_2d[ri, ci]  = e["best_k"]

    for ri, dname in enumerate(DOMAINS_3D):
        for ci, arch in enumerate(all_archs):
            e = next((r for r in registry
                      if r["domain"] == dname and r["arch"] == arch
                      and r["dim"] == 3), None)
            if e:
                mae_3d[ri, ci] = e["best_mae"]
                bk_3d[ri, ci]  = e["best_k"]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(all_domains) * 0.55 + 2)))
    fig.patch.set_facecolor(_FIG_BG)
    fig.suptitle("Best-k Mean MAE Summary  [°C]",
                 fontsize=12, fontweight="bold", color=_TXT_C)

    for ax, data, bk_data, dom_list, dim_label in [
        (axes[0], mae_2d, bk_2d, list(DOMAINS_2D), "2D"),
        (axes[1], mae_3d, bk_3d, list(DOMAINS_3D), "3D"),
    ]:
        ax.set_facecolor(_AX_BG)
        vmin = np.nanmin(data) if np.any(np.isfinite(data)) else 0
        vmax = np.nanmax(data) if np.any(np.isfinite(data)) else 1
        im = ax.imshow(data, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
        _cb(fig, ax, im, "MAE [°C]")
        ax.set_xticks(range(3))
        ax.set_xticklabels([_ARCH_LABELS[a] for a in all_archs],
                           fontsize=9, color=_TXT_C, rotation=15, ha="right")
        ax.set_yticks(range(len(dom_list)))
        ax.set_yticklabels(dom_list, fontsize=9, color=_TXT_C)
        ax.set_title(f"{dim_label} Domains", fontsize=10, fontweight="bold",
                     color=_TXT_C)
        for ri in range(data.shape[0]):
            for ci in range(data.shape[1]):
                v = data[ri, ci]
                k_v = bk_data[ri, ci]
                if np.isfinite(v):
                    txt = f"{v:.2f}\n(k={k_v})"
                    ax.text(ci, ri, txt, ha="center", va="center",
                            fontsize=7.5, color="white" if v > (vmax * 0.6) else _TXT_C)

        for sp in ax.spines.values():
            sp.set_color(_SPINE_C)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = RESULT_DIR / "fig4_summary_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close(fig)
    print(f"    → {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 5 — Time evolution (rectangle, multiple t snapshots)
# ════════════════════════════════════════════════════════════════════════════

def fig5_time_evolution(registry: list[dict],
                        domain_name: str = "rectangle",
                        t_list: list[float] | None = None):
    """Level-7-style temporal heatmaps for one representative 2D domain."""
    if t_list is None:
        t_list = [1.0, 5.0, 15.0, 30.0]
    print(f"  Building Fig 5 — time evolution ({domain_name}) …")

    domain = make_domain(domain_name)
    n_col  = len(t_list)
    row_specs = [("reference", "Reference", _TXT_C)] + [
        (arch, _ARCH_LABELS[arch], _ARCH_COLORS[arch]) for arch in _ARCH_ORDER
    ]

    cols = []
    all_fields = []
    for t_q in t_list:
        XX, YY = _make_2d_grid(domain, 80)
        xi, yi = XX.ravel(), YY.ravel()
        T_ref  = _clean_T(domain.T(xi, yi, t_q)).reshape(80, 80)
        row = {"t": t_q, "xx": XX, "yy": YY, "T_ref": T_ref, "preds": {}}
        all_fields.append(T_ref)

        for arch in _ARCH_ORDER:
            k   = best_k(registry, domain_name, arch, dim=2)
            ckp = CKPT_DIR / f"{domain_name}_{arch}_k{k}_dim2.pt"
            if not ckp.exists():
                continue
            model = load_model(domain_name, arch, k, dim=2)
            g     = eval_grid_2d(model, domain, k, t_q)
            row["preds"][arch] = {"k": k, "grid": g}
            all_fields.append(g["T_pred"])
        cols.append(row)

    norm_fig = _adaptive_norm_many(all_fields)
    fig, axes = plt.subplots(
        len(row_specs), n_col,
        figsize=(4.0 * n_col + 0.8, 2.7 * len(row_specs) + 0.7),
        facecolor=_FIG_BG,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor(_FIG_BG)
    fig.suptitle(
        f"Quenching Temperature Field — {domain_name.capitalize()} 2D\n"
        "Columns = time slices  |  Rows = reference and PINN optimizers",
        fontsize=11, fontweight="bold", color=_TXT_C,
    )

    im_last = None
    for ci, col in enumerate(cols):
        XX, YY, T_ref = col["xx"], col["yy"], col["T_ref"]
        for ri, (key, label, color) in enumerate(row_specs):
            ax = axes[ri, ci]
            if key == "reference":
                im_last = _draw_T(ax, XX, YY, T_ref, norm_fig, cmap="YlOrRd")
            else:
                if key not in col["preds"]:
                    ax.set_visible(False)
                    continue
                pred = col["preds"][key]
                grid = pred["grid"]
                im_last = _draw_T(
                    ax, XX, YY, grid["T_pred"], norm_fig,
                    title=f"MAE={grid['mae']:.1f}°C",
                    c_title=color,
                    add_fem_overlay=T_ref,
                    cmap="YlOrRd",
                )

            if ri == 0:
                ax.set_title(f"t = {col['t']:.0f} s", fontsize=10,
                             fontweight="bold", color=_TXT_C, pad=5)
            if ci == 0:
                ax.set_ylabel(f"{label}\ny [m]", fontsize=8.5,
                              fontweight="bold", color=color)
            if ri == len(row_specs) - 1:
                ax.set_xlabel("x [m]", fontsize=7, color=_TICK_C)
            ax.tick_params(labelsize=6)

    cbar = fig.colorbar(im_last, ax=axes.ravel().tolist(), shrink=0.94, pad=0.02)
    cbar.set_label("T [°C]", color=_TXT_C)
    cbar.ax.tick_params(labelsize=8, colors=_TICK_C)

    out = RESULT_DIR / f"fig5_time_evolution_{domain_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close(fig)
    print(f"    → {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 6 — Per-k heatmaps (each k × each domain, Level 8 style)
# ════════════════════════════════════════════════════════════════════════════

def fig6_per_k_heatmaps(registry: list[dict], t_query: float = 29.0):
    """
    For every domain × every k value: Level 8 style figure
      cols = [Ref | Bayesian | NSGA-II | NSGA-III | Best-error]
      rows = [T field | error]
    """
    print("  Building Fig 6 — per-k heatmaps …")
    k_values = [1, 2, 3, 4, 5]

    loop_specs = [
        (2, list(DOMAINS_2D), make_domain, eval_grid_2d),
        (3, list(DOMAINS_3D), make_domain_3d, eval_grid_3d_mid),
    ]

    for dim, dom_names, make_fn, eval_fn in loop_specs:
        for dname in dom_names:
            domain = make_fn(dname)

            for kv in k_values:
                n_arch = len(_ARCH_ORDER)
                n_cols = 1 + n_arch + 1
                fig    = plt.figure(figsize=(3.4 * n_cols, 7.5))
                fig.patch.set_facecolor(_FIG_BG)
                gs = gridspec.GridSpec(2, n_cols, hspace=0.4, wspace=0.32)
                fig.suptitle(
                    f"{dname.capitalize()} {'2D' if dim==2 else '3D'}  —  k={kv}  "
                    f"(Δt = {kv*1.5:.1f} s)   |   t = {t_query:.0f} s",
                    fontsize=11, fontweight="bold", color=_TXT_C,
                )

                grids = {}
                for arch in _ARCH_ORDER:
                    ckp = CKPT_DIR / f"{dname}_{arch}_k{kv}_dim{dim}.pt"
                    if not ckp.exists():
                        continue
                    model = load_model(dname, arch, kv, dim)
                    grids[arch] = eval_fn(model, domain, kv, t_query)

                if not grids:
                    plt.close(fig)
                    continue

                first = next(iter(grids.values()))
                XX, YY = first["xx"], first["yy"]
                T_ref  = first["T_ref"]

                # Row 0: T fields
                ax_r = fig.add_subplot(gs[0, 0])
                norm_fig = _adaptive_norm(T_ref, [grids[a]["T_pred"] for a in grids])

                im_r = _draw_T(ax_r, XX, YY, T_ref, norm_fig,
                               title=f"Reference\nT̄={np.nanmean(T_ref):.0f}°C")
                _cb(fig, ax_r, im_r, "T [°C]")
                ax_r.set_xlabel("x [m]", fontsize=7, color=_TICK_C)
                ax_r.set_ylabel("y [m]", fontsize=7, color=_TICK_C)

                for ci, arch in enumerate(_ARCH_ORDER):
                    if arch not in grids:
                        continue
                    g   = grids[arch]
                    ax  = fig.add_subplot(gs[0, ci + 1])
                    im  = _draw_T(ax, XX, YY, g["T_pred"], norm_fig,
                                  title=(f"{_ARCH_LABELS[arch]}\n"
                                         f"MAE={g['mae']:.2f}°C"),
                                  c_title=_ARCH_COLORS[arch],
                                  add_fem_overlay=T_ref)
                    _cb(fig, ax, im, "T [°C]")
                    ax.set_xlabel("x [m]", fontsize=7, color=_TICK_C)

                best_arch = min(grids, key=lambda a: grids[a]["mae"])
                ax_be = fig.add_subplot(gs[0, n_arch + 1])
                im_be, _ = _draw_err(ax_be, XX, YY, grids[best_arch]["err"],
                                      title=f"Best |Error|\n({_ARCH_LABELS[best_arch]})")
                _cb(fig, ax_be, im_be, "|ΔT| [°C]")
                ax_be.set_xlabel("x [m]", fontsize=7, color=_TICK_C)

                # Row 1: all error maps
                ax_re = fig.add_subplot(gs[1, 0])
                ax_re.set_visible(False)
                for ci, arch in enumerate(_ARCH_ORDER):
                    if arch not in grids:
                        continue
                    g  = grids[arch]
                    ax = fig.add_subplot(gs[1, ci + 1])
                    im_e, _ = _draw_err(ax, XX, YY, g["err"])
                    _cb(fig, ax, im_e, "|ΔT| [°C]")
                    ax.set_xlabel("x [m]", fontsize=7, color=_TICK_C)

                # L2 mini-bar in last col row 1
                ax_bar = fig.add_subplot(gs[1, n_arch + 1])
                ax_bar.set_facecolor(_AX_BG)
                for sp in ax_bar.spines.values():
                    sp.set_color(_SPINE_C)
                l2_items = [(a, grids[a]["l2"]) for a in _ARCH_ORDER if a in grids]
                l2_items.sort(key=lambda x: x[1])
                ax_bar.barh(range(len(l2_items)), [v for _, v in l2_items],
                            color=[_ARCH_COLORS[a] for a, _ in l2_items], height=0.55)
                ax_bar.set_yticks(range(len(l2_items)))
                ax_bar.set_yticklabels([f"{_ARCH_LABELS[a]}\n{v:.4f}"
                                         for a, v in l2_items],
                                        fontsize=7, color=_TXT_C)
                ax_bar.set_xlabel("Relative L2", fontsize=7, color=_TXT_C)
                ax_bar.tick_params(colors=_TICK_C, labelsize=6)
                ax_bar.set_title("L2 Ranking", fontsize=8, color=_TXT_C, fontweight="bold")
                ax_bar.grid(axis="x", alpha=0.3, color=_SPINE_C)

                out = RESULT_DIR / "per_k" / f"{dname}_dim{dim}_k{kv}_t{t_query:.0f}s.png"
                fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
                plt.close(fig)
                print(f"    → per_k/{out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 7 — Per-k summary tables
# ════════════════════════════════════════════════════════════════════════════

def fig7_per_k_tables(registry: list[dict]):
    """Professional per-k summary tables with explicit baseline/reference column."""
    print("  Building Fig 7 — per-k summary tables …")

    pretty_name = {
        (2, "rectangle"): "Rectangle",
        (2, "circle"): "Circle",
        (2, "annulus"): "Annulus",
        (2, "lshape"): "L-Shape",
        (3, "rectangular"): "Rectangular Prism",
        (3, "cylinder"): "Cylinder",
        (3, "stacked"): "Stacked Cubes",
        (3, "lshape"): "L-Shape 3D",
    }

    def _fmt(v: float) -> str:
        return f"{v:.2f}°C" if np.isfinite(v) else "—"

    col_labels = [
        "Domain",
        "Baseline\n(FEM / FD ref.)",
        "Bayesian",
        "NSGA-II",
        "NSGA-III",
        "Best",
    ]

    for kv in [1, 2, 3, 4, 5]:
        sched = _k_skip_schedule(kv)
        fig, axes = plt.subplots(
            2, 1, figsize=(14.5, 8.6), facecolor=_FIG_BG,
            gridspec_kw={"hspace": 0.30}
        )
        fig.patch.set_facecolor(_FIG_BG)
        fig.suptitle(
            f"Per-k Comparison Table — k = {kv}   (Δt = {kv * 1.5:.1f} s)\n"
            f"Baseline = analytical / FD reference, MAE = 0.00°C by definition   |   "
            f"Reference times used = {sched['n_ref_times']}/21   |   PINN windows = {sched['n_windows']}",
            fontsize=12, fontweight="bold", color=_TXT_C, y=0.98,
        )

        section_specs = [
            (axes[0], list(DOMAINS_2D), 2, "2D Domains"),
            (axes[1], list(DOMAINS_3D), 3, "3D Domains"),
        ]

        for ax, dom_list, dim, section_title in section_specs:
            ax.set_facecolor(_FIG_BG)
            ax.axis("off")
            ax.text(
                0.0, 1.03, section_title,
                transform=ax.transAxes,
                fontsize=10.5, fontweight="bold", color=_TXT_C, va="bottom"
            )

            rows = []
            best_cells: list[tuple[int, int]] = []
            for row_idx, dname in enumerate(dom_list, start=1):
                maes = []
                for arch in _ARCH_ORDER:
                    met = load_metrics(dname, arch, kv, dim)
                    maes.append(float(met["mean_mae"]) if met else np.nan)

                finite_idx = [i for i, v in enumerate(maes) if np.isfinite(v)]
                if finite_idx:
                    best_i = min(finite_idx, key=lambda i: maes[i])
                    best_arch = _ARCH_LABELS[_ARCH_ORDER[best_i]].split()[0]
                    best_txt = f"{best_arch}\n{maes[best_i]:.2f}°C"
                    best_cells.append((row_idx, 2 + best_i))
                else:
                    best_txt = "—"

                rows.append([
                    pretty_name.get((dim, dname), dname),
                    "0.00°C",
                    _fmt(maes[0]),
                    _fmt(maes[1]),
                    _fmt(maes[2]),
                    best_txt,
                ])

            table = ax.table(
                cellText=rows,
                colLabels=col_labels,
                cellLoc="center",
                loc="center",
                bbox=[0.0, 0.0, 1.0, 0.90],
                colWidths=[0.22, 0.16, 0.13, 0.13, 0.13, 0.16],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8.8)
            table.scale(1.0, 1.65)

            for j in range(len(col_labels)):
                header = table[0, j]
                header.set_facecolor("#1565C0")
                header.set_text_props(color="white", fontweight="bold")
                header.set_edgecolor(_SPINE_C)
                header.set_linewidth(0.8)

            for i in range(1, len(rows) + 1):
                row_bg = "#fffdf8" if i % 2 else "#fbf6eb"
                for j in range(len(col_labels)):
                    cell = table[i, j]
                    cell.set_edgecolor(_SPINE_C)
                    cell.set_linewidth(0.7)
                    cell.set_facecolor(row_bg)
                    cell.set_text_props(color=_TXT_C)

                table[i, 0].set_facecolor("#F3EBDD")
                table[i, 0].set_text_props(fontweight="bold", color=_TXT_C)
                table[i, 1].set_facecolor("#FFECB3")
                table[i, 1].set_text_props(color="#8A5A00", fontweight="bold")
                table[i, 5].set_facecolor("#E8F5E9")
                table[i, 5].set_text_props(color="#1B5E20", fontweight="bold")

            for ri, ci in best_cells:
                table[ri, ci].set_facecolor("#C8E6C9")
                table[ri, ci].set_text_props(color="#1B5E20", fontweight="bold")

        out = RESULT_DIR / f"fig7_k{kv}_summary_table.png"
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=_FIG_BG)
        plt.close(fig)
        print(f"    → {out.name}")


def fig7_all_k_table(registry: list[dict]):
    """Single professional summary table covering all k values and all optimizers."""
    print("  Building Fig 7 — all-k summary table …")
    k_values = [1, 2, 3, 4, 5]
    col_pairs = [(arch, kv) for arch in _ARCH_ORDER for kv in k_values]
    col_labels = [
        f"{'Bay' if arch == 'bayesian' else arch.upper()}\nk={kv}"
        for arch, kv in col_pairs
    ]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(16, 0.62 * (len(DOMAINS_2D) + len(DOMAINS_3D)) + 6.5),
        facecolor=_FIG_BG,
        constrained_layout=True,
    )
    fig.patch.set_facecolor(_FIG_BG)
    fig.suptitle(
        "All k Values — Mean MAE Summary  [°C]\n"
        "Columns grouped by optimizer, then k = 1…5",
        fontsize=12, fontweight="bold", color=_TXT_C,
    )

    for ax, dom_list, dim, title in [
        (axes[0], list(DOMAINS_2D), 2, "2D Domains"),
        (axes[1], list(DOMAINS_3D), 3, "3D Domains"),
    ]:
        data = np.full((len(dom_list), len(col_pairs)), np.nan)
        for ri, dname in enumerate(dom_list):
            for ci, (arch, kv) in enumerate(col_pairs):
                mp = CKPT_DIR / f"{dname}_{arch}_k{kv}_dim{dim}_metrics.json"
                if mp.exists():
                    data[ri, ci] = json.load(open(mp))["mean_mae"]

        ax.set_facecolor(_AX_BG)
        vmin = np.nanmin(data) if np.any(np.isfinite(data)) else 0.0
        vmax = np.nanmax(data) if np.any(np.isfinite(data)) else 1.0
        im = ax.imshow(data, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10, fontweight="bold", color=_TXT_C, pad=8)
        ax.set_xticks(range(len(col_pairs)))
        ax.set_xticklabels(col_labels, fontsize=8, color=_TXT_C)
        ax.set_yticks(range(len(dom_list)))
        ax.set_yticklabels(dom_list, fontsize=9, color=_TXT_C)
        ax.tick_params(length=0)

        for split in (4.5, 9.5):
            ax.axvline(split, color="#8a7f6f", lw=1.2, alpha=0.9)

        for ri in range(data.shape[0]):
            for ci in range(data.shape[1]):
                v = data[ri, ci]
                if np.isfinite(v):
                    ax.text(
                        ci, ri, f"{v:.2f}",
                        ha="center", va="center", fontsize=7.2,
                        color="white" if v > (vmax * 0.62) else _TXT_C,
                    )

        for sp in ax.spines.values():
            sp.set_color(_SPINE_C)

        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
        cb.set_label("MAE [°C]", fontsize=8, color=_TXT_C)
        cb.ax.tick_params(labelsize=7, colors=_TICK_C)
        cb.outline.set_edgecolor(_SPINE_C)

    out = RESULT_DIR / "fig7_all_k_table.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close(fig)
    print(f"    → {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 11 — k-skip timeline explainer
# ════════════════════════════════════════════════════════════════════════════

def fig11_k_timeline():
    """Conceptual timeline showing how each k spans the FEM time grid."""
    print("  Building Fig 11 — k timeline explainer …")
    k_values = [1, 2, 3, 4, 5]
    fem_times = np.arange(0.0, T_TOTAL + 1e-9, DT_FEM)
    row_colors = ["#1565C0", "#2E7D32", "#EF6C00", "#8E24AA", "#C62828"]

    fig, ax = plt.subplots(figsize=(14.2, 5.8), facecolor=_FIG_BG)
    fig.patch.set_facecolor(_FIG_BG)
    fig.subplots_adjust(left=0.28, right=0.98, top=0.84, bottom=0.15)
    ax.set_facecolor(_AX_BG)
    ax.set_axisbelow(True)

    ax.set_xticks(np.arange(0.0, T_TOTAL + 1e-9, 3.0))
    ax.set_xticks(fem_times, minor=True)
    ax.grid(which="major", axis="x", color="#d0c6b4", linewidth=0.9, alpha=0.8)
    ax.grid(which="minor", axis="x", color="#e9e1d4", linewidth=0.6, alpha=0.9)

    y_positions = np.arange(len(k_values), 0, -1)
    y_labels = []
    for idx, kv in enumerate(k_values):
        sched = _k_skip_schedule(kv)
        y = y_positions[idx]
        spans = [(t0, t1 - t0) for t0, t1 in sched["windows"]]
        ax.broken_barh(
            spans, (y - 0.23, 0.46),
            facecolors=row_colors[idx], edgecolors="white",
            linewidth=1.2, alpha=0.92, zorder=3,
        )
        ax.scatter(
            sched["ref_times"], [y + 0.26] * len(sched["ref_times"]),
            s=18, color="#1f1f1f", edgecolor="white", linewidth=0.45, zorder=4,
        )
        y_labels.append(
            f"k={kv}   Δt={sched['dt_window']:.1f}s   "
            f"{sched['n_windows']} window   {sched['n_ref_times']} ref"
        )

    ax.set_xlim(0.0, T_TOTAL)
    ax.set_ylim(0.45, len(k_values) + 0.75)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8.8, color=_TXT_C)
    ax.tick_params(axis="y", length=0)
    ax.set_xticklabels([f"{int(t)}" for t in np.arange(0.0, T_TOTAL + 1e-9, 3.0)],
                       fontsize=8.4, color=_TICK_C)
    ax.set_xlabel("Physical time [s]  on the original FEM grid (Δt = 1.5 s)",
                  fontsize=10, color=_TXT_C, labelpad=10)
    ax.set_title(
        "k-Skip Timeline\nBars = one PINN prediction window   |   Dots = reference times used",
        fontsize=12, fontweight="bold", color=_TXT_C, pad=10,
    )
    for sp in ax.spines.values():
        sp.set_color(_SPINE_C)
        sp.set_linewidth(0.9)

    legend = [
        Patch(facecolor=row_colors[2], edgecolor="white", label="PINN prediction window"),
        Line2D([0], [0], marker="o", color="w", label="Reference / FEM time used",
               markerfacecolor="#1f1f1f", markeredgecolor="white", markersize=7),
        Line2D([0], [0], color="#d7cfbf", lw=1.2, label="FEM base grid (1.5 s)"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=8.8,
              facecolor="white", edgecolor=_SPINE_C, framealpha=0.95)

    out = RESULT_DIR / "fig11_k_timeline.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close(fig)
    print(f"    → {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 12 — k-skip trade-off summary
# ════════════════════════════════════════════════════════════════════════════

def fig12_k_tradeoff(registry: list[dict]):
    """Accuracy-efficiency view that explains what changing k really does."""
    del registry
    print("  Building Fig 12 — k trade-off summary …")

    k_values = [1, 2, 3, 4, 5]
    all_domains = [(2, d) for d in DOMAINS_2D] + [(3, d) for d in DOMAINS_3D]
    schedules = [_k_skip_schedule(kv) for kv in k_values]

    fig, axes = plt.subplots(
        1, 2, figsize=(13.8, 5.2), facecolor=_FIG_BG, constrained_layout=True
    )
    fig.patch.set_facecolor(_FIG_BG)
    fig.suptitle(
        "k-Skip Trade-off Summary",
        fontsize=12.5, fontweight="bold", color=_TXT_C,
    )

    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor(_AX_BG)
        for sp in ax.spines.values():
            sp.set_color(_SPINE_C)
        ax.grid(alpha=0.22, color=_SPINE_C, linewidth=0.8)
        ax.tick_params(colors=_TICK_C, labelsize=8.5)

    for arch in _ARCH_ORDER:
        mean_maes = []
        for kv in k_values:
            vals = []
            for dim, dname in all_domains:
                met = load_metrics(dname, arch, kv, dim)
                if met:
                    vals.append(float(met["mean_mae"]))
            mean_maes.append(float(np.mean(vals)) if vals else np.nan)

        ax1.plot(
            k_values, mean_maes,
            marker="o", markersize=6.5, linewidth=2.4,
            color=_ARCH_COLORS[arch], label=_ARCH_LABELS[arch],
        )

    ax1.set_title("A. Mean Error vs k", fontsize=10.5,
                  fontweight="bold", color=_TXT_C)
    ax1.set_xlabel("k value  and  PINN time window", fontsize=9.5, color=_TXT_C)
    ax1.set_ylabel("Mean MAE across all 8 domains [°C]", fontsize=9.5, color=_TXT_C)
    ax1.set_xticks(k_values)
    ax1.set_xticklabels([f"k={kv}\nΔt={kv * DT_FEM:.1f}s" for kv in k_values],
                        fontsize=8.5, color=_TXT_C)
    ax1.legend(loc="upper left", fontsize=8.5, facecolor="white",
               edgecolor=_SPINE_C, framealpha=0.95)

    xpos = np.arange(len(k_values))
    n_windows = [s["n_windows"] for s in schedules]
    n_ref = [s["n_ref_times"] for s in schedules]
    bars1 = ax2.bar(xpos - 0.18, n_windows, width=0.36, color="#F9A825",
                    edgecolor="white", linewidth=1.0, label="PINN windows over 30 s")
    bars2 = ax2.bar(xpos + 0.18, n_ref, width=0.36, color="#1565C0",
                    edgecolor="white", linewidth=1.0, label="Reference times used")
    ax2.axhline(21, color="#8d8d8d", linestyle="--", linewidth=1.2, alpha=0.9,
                label="Full FEM grid = 21 time points")
    ax2.set_title("B. Time Usage Over 0–30 s", fontsize=10.5,
                  fontweight="bold", color=_TXT_C)
    ax2.set_xlabel("k value", fontsize=9.5, color=_TXT_C)
    ax2.set_ylabel("Count", fontsize=9.5, color=_TXT_C)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels([f"k={kv}\nΔt={kv * DT_FEM:.1f}s" for kv in k_values],
                        fontsize=8.5, color=_TXT_C)
    ax2.set_ylim(0, 23)

    for bars in (bars1, bars2):
        for b in bars:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.35,
                     f"{int(round(b.get_height()))}", ha="center", va="bottom",
                     fontsize=7.8, color=_TXT_C)

    ax2.legend(loc="upper right", fontsize=8.2,
               facecolor="white", edgecolor=_SPINE_C, framealpha=0.95)

    out = RESULT_DIR / "fig12_k_tradeoff.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close(fig)
    print(f"    → {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 8 — 3D Orthogonal slice panorama (Level 8 fig_field_slices style)
# ════════════════════════════════════════════════════════════════════════════

def fig8_3d_slices(registry: list[dict], t_query: float = 29.0,
                   domain_name: str = "cylinder", arch: str = "bayesian"):
    """
    3 slices (xy/xz/yz) × 3 cols (Reference | PINN | |Error|)
    Exactly as in Level 8 fig_field_slices.
    """
    print(f"  Building Fig 8 — 3D slices ({domain_name}/{arch}) …")
    k      = best_k(registry, domain_name, arch, dim=3)
    ckp    = CKPT_DIR / f"{domain_name}_{arch}_k{k}_dim3.pt"
    if not ckp.exists():
        print(f"    checkpoint not found: {ckp.name}")
        return

    domain = make_domain_3d(domain_name)
    model  = load_model(domain_name, arch, k, dim=3)
    slices = eval_grid_3d_slices(model, domain, k, t_query, n_grid=40)

    slice_names = ["xy", "xz", "yz"]
    row_labels  = [slices[s]["title_suffix"] for s in slice_names]

    # Adaptive norm from all slice data
    all_T   = np.concatenate([slices[s]["T_ref"].ravel() for s in slice_names])
    all_Tp  = np.concatenate([slices[s]["T_pred"].ravel() for s in slice_names])
    norm_T  = _adaptive_norm(all_T, [all_Tp])   # local variable (not global)
    all_errs = np.concatenate([slices[s]["err"].ravel() for s in slice_names])
    fin_e    = all_errs[np.isfinite(all_errs)]
    err_max  = max(float(np.nanpercentile(fin_e, 97)) if len(fin_e) > 0 else 1.0, 1.0)
    norm_e   = Normalize(vmin=0, vmax=err_max)

    fig, axes = plt.subplots(3, 3, figsize=(11.8, 8.8))
    fig.patch.set_facecolor(_FIG_BG)
    fig.suptitle(
        f"3D Thermal Field Slices  —  {domain_name.capitalize()}\n"
        f"{_ARCH_LABELS[arch]}  |  k={k}  |  t = {t_query:.0f} s",
        fontsize=11, fontweight="bold", color=_TXT_C,
    )
    col_titles = ["Reference", "PINN prediction", "|Reference − PINN|"]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=9, fontweight="bold", color=_TXT_C, pad=4)

    note_box = {
        "facecolor": "#fef6e4",
        "edgecolor": "#e8c77a",
        "boxstyle": "round,pad=0.18",
    }

    for ri, sname in enumerate(slice_names):
        sl    = slices[sname]
        XX, YY = sl["V1"], sl["V2"]
        T_r, T_p, err = sl["T_ref"], sl["T_pred"], sl["err"]
        xl, yl = sl["xlabel"], sl["ylabel"]

        ax_f = axes[ri, 0]
        ax_p = axes[ri, 1]
        ax_e = axes[ri, 2]

        # Reference
        im_f = ax_f.pcolormesh(XX, YY, np.where(np.isfinite(T_r), T_r, np.nan),
                                cmap=CMAP_T, norm=norm_T, shading="auto", rasterized=True)
        if np.any(np.isfinite(T_r)):
            try: ax_f.contour(XX, YY, np.where(np.isfinite(T_r), T_r, np.nan),
                              levels=6, colors="white", linewidths=0.4, alpha=0.75)
            except Exception: pass
        ax_f.set_aspect("auto"); _style(ax_f)
        ax_f.set_ylabel(f"{row_labels[ri]}\n{yl}", fontsize=7.5, fontweight="bold", color=_TXT_C)
        if ri == 2:
            ax_f.set_xlabel(xl, fontsize=7, color=_TICK_C)

        # PINN + FEM overlay
        im_p = ax_p.pcolormesh(XX, YY, np.where(np.isfinite(T_p), T_p, np.nan),
                                cmap=CMAP_T, norm=norm_T, shading="auto", rasterized=True)
        if np.any(np.isfinite(T_p)):
            try: ax_p.contour(XX, YY, np.where(np.isfinite(T_p), T_p, np.nan),
                              levels=6, colors="white", linewidths=0.4, alpha=0.65)
            except Exception: pass
        if np.any(np.isfinite(T_r)):
            try: ax_p.contour(XX, YY, np.where(np.isfinite(T_r), T_r, np.nan),
                              levels=6, colors="#FF4444", linewidths=0.85,
                              alpha=0.9, linestyles="dashed")
            except Exception: pass
        ax_p.set_aspect("auto"); _style(ax_p)
        mae_s = float(np.nanmean(err)) if np.any(np.isfinite(err)) else np.nan
        ax_p.text(0.03, 0.96, f"MAE={mae_s:.1f}°C",
                  transform=ax_p.transAxes, va="top",
                  fontsize=7.2, color=_ARCH_COLORS[arch], bbox=note_box)
        if ri == 2:
            ax_p.set_xlabel(xl, fontsize=7, color=_TICK_C)

        # Error
        err_v = np.where(np.isfinite(err), err, np.nan)
        im_e  = ax_e.pcolormesh(XX, YY, err_v, cmap=CMAP_E, norm=norm_e,
                                  shading="auto", rasterized=True)
        if np.any(np.isfinite(err_v)):
            try: ax_e.contour(XX, YY, err_v, levels=4,
                              colors="#333333", linewidths=0.35, alpha=0.6)
            except Exception: pass
        ax_e.set_aspect("auto"); _style(ax_e)
        ax_e.text(0.03, 0.96, f"max={np.nanmax(err_v):.1f}°C",
                  transform=ax_e.transAxes, va="top",
                  fontsize=7.2, color="#B71C1C", bbox=note_box)
        if ri == 2:
            ax_e.set_xlabel(xl, fontsize=7, color=_TICK_C)

        if ri == 0:
            _cb(fig, ax_f, im_f, "T [°C]")
            _cb(fig, ax_p, im_p, "T [°C]")
            _cb(fig, ax_e, im_e, "|ΔT| [°C]")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULT_DIR / f"fig8_{domain_name}_3d_slices_{arch}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close(fig)
    print(f"    → {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 9 — 3D Domain comparison (Level 8 fig_domain_compare style)
# ════════════════════════════════════════════════════════════════════════════

def fig9_3d_domain_compare(registry: list[dict], t_query: float = 15.0,
                           arch: str = "bayesian", n_grid: int = 50):
    """Compare all 3D domains at a shared z-mid slice for one architecture."""
    print(f"  Building Fig 9 — 3D domain compare ({arch}) …")

    rows = []
    for dname in DOMAINS_3D:
        k = best_k(registry, dname, arch, dim=3)
        ckp = CKPT_DIR / f"{dname}_{arch}_k{k}_dim3.pt"
        if not ckp.exists():
            continue
        domain = make_domain_3d(dname)
        model  = load_model(dname, arch, k, dim=3)
        rows.append({
            "name": dname,
            "label": {
                "rectangular": "Rectangular\nPrism",
                "cylinder": "Cylinder",
                "stacked": "Stacked\nCubes",
                "lshape": "L-Shape\n3D",
            }.get(dname, dname.capitalize()),
            "color": getattr(domain, "color", _TXT_C),
            "k": k,
            "grid": eval_grid_3d_mid(model, domain, k, t_query, n_grid),
        })

    if not rows:
        print("    no 3D checkpoints found.")
        return

    norm_T = _adaptive_norm_many(
        [row["grid"]["T_ref"] for row in rows] +
        [row["grid"]["T_pred"] for row in rows]
    )
    err_vals = np.concatenate([row["grid"]["err"].ravel() for row in rows])
    err_vals = err_vals[np.isfinite(err_vals)]
    err_max  = max(float(np.nanpercentile(err_vals, 97)) if err_vals.size else 1.0, 1.0)
    norm_e   = Normalize(vmin=0, vmax=err_max)

    fig, axes = plt.subplots(len(rows), 3, figsize=(11.8, 3.1 * len(rows)))
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor(_FIG_BG)
    fig.suptitle(
        f"3D Domain Comparison — {_ARCH_LABELS[arch]}  |  t = {t_query:.0f} s\n"
        "Rows = domains  |  Columns = reference, prediction, error",
        fontsize=11, fontweight="bold", color=_TXT_C,
    )

    col_titles = ["Reference", "PINN prediction", "|Reference − PINN|"]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=9, fontweight="bold", color=_TXT_C, pad=4)

    note_box = {
        "facecolor": "#fef6e4",
        "edgecolor": "#e8c77a",
        "boxstyle": "round,pad=0.18",
    }

    for ri, row in enumerate(rows):
        g = row["grid"]
        XX, YY = g["xx"], g["yy"]
        T_ref, T_pred, err = g["T_ref"], g["T_pred"], g["err"]

        ax_f = axes[ri, 0]
        ax_p = axes[ri, 1]
        ax_e = axes[ri, 2]

        im_f = _draw_T(ax_f, XX, YY, T_ref, norm_T, aspect="auto")
        ax_f.set_ylabel(row["label"], fontsize=8.5, fontweight="bold", color=row["color"])
        ax_f.set_xlabel("x [m]", fontsize=7, color=_TICK_C)

        im_p = _draw_T(
            ax_p, XX, YY, T_pred, norm_T,
            c_title=_ARCH_COLORS[arch],
            add_fem_overlay=T_ref,
            aspect="auto",
        )
        ax_p.text(0.03, 0.96, f"k={row['k']}  MAE={g['mae']:.2f}°C",
                  transform=ax_p.transAxes, va="top",
                  fontsize=7.2, color=_ARCH_COLORS[arch], bbox=note_box)
        ax_p.set_xlabel("x [m]", fontsize=7, color=_TICK_C)
        ax_p.set_ylabel("y [m]", fontsize=7, color=_TICK_C)

        im_e, _ = _draw_err(
            ax_e, XX, YY, err,
            norm=norm_e,
            aspect="auto",
        )
        ax_e.text(0.03, 0.96, f"max={np.nanmax(err):.1f}°C",
                  transform=ax_e.transAxes, va="top",
                  fontsize=7.2, color="#B71C1C", bbox=note_box)
        ax_e.set_xlabel("x [m]", fontsize=7, color=_TICK_C)

    cbar_T = fig.colorbar(im_p, ax=axes[:, :2].ravel().tolist(), shrink=0.97, pad=0.01)
    cbar_T.set_label("T [°C]", color=_TXT_C)
    cbar_T.ax.tick_params(labelsize=8, colors=_TICK_C)
    cbar_E = fig.colorbar(im_e, ax=axes[:, 2].ravel().tolist(), shrink=0.97, pad=0.01)
    cbar_E.set_label("|ΔT| [°C]", color=_TXT_C)
    cbar_E.ax.tick_params(labelsize=8, colors=_TICK_C)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULT_DIR / f"fig9_3d_domain_compare_{arch}_t{t_query:.0f}s.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close(fig)
    print(f"    → {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 10 — TRUE 3D Volumetric Renders (Level 8 fig1 style)
# ════════════════════════════════════════════════════════════════════════════

def _surf(ax, X, Y, Z, T, cmap, norm, alpha=0.93):
    """Plot one surface face colored by T."""
    fc = cmap(norm(np.clip(T, T_WATER - 1, T_INIT + 1)))
    ax.plot_surface(X, Y, Z, facecolors=fc, alpha=alpha,
                    rstride=1, cstride=1, linewidth=0.25,
                    edgecolor=(0.55, 0.55, 0.55, 0.25),
                    antialiased=True, shade=False)


def _eval_3d_field(domain, t, X, Y, Z, field_fn=None):
    """Evaluate either the reference field or a supplied PINN field."""
    if field_fn is None:
        return domain.T(X, Y, Z, t)
    return field_fn(X, Y, Z, t)


def _valid_surface_mask(domain, coords_np: np.ndarray) -> np.ndarray:
    """Return an in-domain mask for arbitrary 3D surface coordinates."""
    x = coords_np[:, 0]
    y = coords_np[:, 1]
    z = coords_np[:, 2]
    if hasattr(domain, "mask"):
        return np.asarray(domain.mask(x, y, z), dtype=bool)
    if _is_cylinder_3d(domain):
        return (x**2 + y**2) <= (domain.R**2 + 1e-9)
    return np.ones(len(coords_np), dtype=bool)


def _predict_surface_field(model: ThermalPINN, domain, k: int, t_query: float,
                           X, Y, Z) -> np.ndarray:
    """Predict temperature on arbitrary 3D surface coordinates."""
    x = np.asarray(X, dtype=np.float32).ravel()
    y = np.asarray(Y, dtype=np.float32).ravel()
    z = np.asarray(Z, dtype=np.float32).ravel()
    coords = np.stack([x, y, z], axis=1).astype(np.float32)
    t_start, _, tau_val = _tau_params(t_query, k)

    valid = _valid_surface_mask(domain, coords)
    T_ic = _clean_T(domain.T(x, y, z, t_start), domain, x, y, z)
    T_ic = np.where(valid & np.isfinite(T_ic), T_ic, T_WATER)

    T_pred = _pinn_forward_full(model, coords, tau_val, T_ic, domain)
    T_pred = _clean_T(T_pred, domain, x, y, z)
    T_pred = np.where(valid, T_pred, np.nan)
    return T_pred.reshape(np.asarray(X).shape)


def _surface_meshes_3d(domain, n=14, n_phi=36, n_z=16):
    """Return surface meshes used for 3D volumetric comparisons."""
    meshes = []
    if hasattr(domain, "R") and hasattr(domain, "Hz"):   # Cylinder
        phi = np.linspace(0, 2 * np.pi, n_phi)
        z_l = np.linspace(0, domain.Hz, n_z)
        PHI, ZL = np.meshgrid(phi, z_l, indexing='ij')
        XL = domain.R * np.cos(PHI)
        YL = domain.R * np.sin(PHI)
        meshes.append((XL, YL, ZL))

        r_c = np.linspace(0, domain.R, max(8, n_z // 2))
        PHI_c, RC = np.meshgrid(phi, r_c, indexing='ij')
        XC = RC * np.cos(PHI_c)
        YC = RC * np.sin(PHI_c)
        for z_cap in [0.0, domain.Hz]:
            meshes.append((XC, YC, np.full_like(XC, z_cap)))
        return meshes

    if hasattr(domain, "cut_x") and hasattr(domain, "cut_y"):  # L-shape
        Lx, Ly, Lz = domain.Lx, domain.Ly, domain.Lz
        cx, cy = domain.cut_x, domain.cut_y
        z_lin = np.linspace(0, Lz, n)

        def add_xz(x_vals, y_const):
            XV, ZV = np.meshgrid(x_vals, z_lin, indexing='ij')
            YV = np.full_like(XV, y_const)
            meshes.append((XV, YV, ZV))

        def add_yz(x_const, y_vals):
            YV, ZV = np.meshgrid(y_vals, z_lin, indexing='ij')
            XV = np.full_like(YV, x_const)
            meshes.append((XV, YV, ZV))

        add_xz(np.linspace(0, Lx, n), 0.0)
        add_yz(Lx, np.linspace(0, cy, n))
        add_xz(np.linspace(cx, Lx, n), cy)
        add_yz(cx, np.linspace(cy, Ly, n))
        add_xz(np.linspace(0, cx, n), Ly)
        add_yz(0.0, np.linspace(0, Ly, n))

        xa = np.linspace(0, Lx, n)
        ya = np.linspace(0, cy, n)
        VA, VC = np.meshgrid(xa, ya, indexing='ij')
        xb = np.linspace(0, cx, n)
        yb = np.linspace(cy, Ly, n)
        VB, VD = np.meshgrid(xb, yb, indexing='ij')
        for z_face in [0.0, Lz]:
            meshes.append((VA, VC, np.full_like(VA, z_face)))
            meshes.append((VB, VD, np.full_like(VB, z_face)))
        return meshes

    # Rectangular / stacked boxes
    Lx, Ly, Lz = domain.Lx, domain.Ly, domain.Lz
    v1 = np.linspace(0, Ly, n); v2 = np.linspace(0, Lz, n)
    V1, V2 = np.meshgrid(v1, v2, indexing='ij')
    meshes.extend([
        (np.zeros_like(V1), V1, V2),
        (np.full_like(V1, Lx), V1, V2),
    ])

    v1 = np.linspace(0, Lx, n); v2 = np.linspace(0, Lz, n)
    V1, V2 = np.meshgrid(v1, v2, indexing='ij')
    meshes.extend([
        (V1, np.zeros_like(V1), V2),
        (V1, np.full_like(V1, Ly), V2),
    ])

    v1 = np.linspace(0, Lx, n); v2 = np.linspace(0, Ly, n)
    V1, V2 = np.meshgrid(v1, v2, indexing='ij')
    meshes.extend([
        (V1, V2, np.zeros_like(V1)),
        (V1, V2, np.full_like(V1, Lz)),
    ])
    return meshes


def _surface_mae_3d(model: ThermalPINN, domain, k: int, t_query: float, n=14) -> float:
    """Surface-only MAE consistent with the displayed volumetric panels."""
    errs = []
    for X, Y, Z in _surface_meshes_3d(domain, n=n):
        T_ref = _clean_T(domain.T(X.ravel(), Y.ravel(), Z.ravel(), t_query),
                         domain, X.ravel(), Y.ravel(), Z.ravel()).reshape(X.shape)
        T_pred = _predict_surface_field(model, domain, k, t_query, X, Y, Z)
        valid = np.isfinite(T_ref) & np.isfinite(T_pred)
        if np.any(valid):
            errs.append(np.abs(T_pred[valid] - T_ref[valid]))
    if not errs:
        return np.nan
    return float(np.mean(np.concatenate(errs)))


def _render_box_3d(ax, domain, t, cmap, norm, n=18, field_fn=None):
    """All 6 faces of a rectangular box (Rectangular3D or StackedCubes3D)."""
    Lx = domain.Lx; Ly = domain.Ly; Lz = domain.Lz

    # x=0 and x=Lx faces (y-z plane)
    v1 = np.linspace(0, Ly, n); v2 = np.linspace(0, Lz, n)
    V1, V2 = np.meshgrid(v1, v2, indexing='ij')
    X0 = np.zeros_like(V1)
    XL = np.full_like(V1, Lx)
    _surf(ax, X0, V1, V2, _eval_3d_field(domain, t, X0, V1, V2, field_fn), cmap, norm)
    _surf(ax, XL, V1, V2, _eval_3d_field(domain, t, XL, V1, V2, field_fn), cmap, norm)

    # y=0 and y=Ly faces (x-z plane)
    v1 = np.linspace(0, Lx, n); v2 = np.linspace(0, Lz, n)
    V1, V2 = np.meshgrid(v1, v2, indexing='ij')
    Y0 = np.zeros_like(V1)
    YL = np.full_like(V1, Ly)
    _surf(ax, V1, Y0, V2, _eval_3d_field(domain, t, V1, Y0, V2, field_fn), cmap, norm)
    _surf(ax, V1, YL, V2, _eval_3d_field(domain, t, V1, YL, V2, field_fn), cmap, norm)

    # z=0 and z=Lz faces (x-y plane)
    v1 = np.linspace(0, Lx, n); v2 = np.linspace(0, Ly, n)
    V1, V2 = np.meshgrid(v1, v2, indexing='ij')
    Z0 = np.zeros_like(V1)
    ZL = np.full_like(V1, Lz)
    _surf(ax, V1, V2, Z0, _eval_3d_field(domain, t, V1, V2, Z0, field_fn), cmap, norm)
    _surf(ax, V1, V2, ZL, _eval_3d_field(domain, t, V1, V2, ZL, field_fn), cmap, norm)

    # Stacked cubes: highlight interface lines
    if hasattr(domain, "interface_z"):
        for iz in domain.interface_z:
            xs = [0, Lx, Lx, 0, 0]; ys = [0, 0, Ly, Ly, 0]
            ax.plot(xs, ys, [iz]*5, color="#222222", lw=1.0, zorder=10)


def _render_cylinder_3d(ax, domain, t, cmap, norm, n_phi=48, n_z=20, field_fn=None):
    """Lateral surface + top/bottom caps of a cylinder."""
    R = domain.R; Hz = domain.Hz

    # Lateral surface
    phi = np.linspace(0, 2*np.pi, n_phi)
    z_l = np.linspace(0, Hz, n_z)
    PHI, ZL = np.meshgrid(phi, z_l, indexing='ij')
    XL = R * np.cos(PHI); YL = R * np.sin(PHI)
    _surf(ax, XL, YL, ZL, _eval_3d_field(domain, t, XL, YL, ZL, field_fn), cmap, norm)

    # Top and bottom caps
    r_c = np.linspace(0, R, n_z // 2)
    PHI_c, RC = np.meshgrid(phi, r_c, indexing='ij')
    XC = RC * np.cos(PHI_c); YC = RC * np.sin(PHI_c)
    for z_cap in [0.0, Hz]:
        ZC = np.full_like(XC, z_cap)
        _surf(ax, XC, YC, ZC, _eval_3d_field(domain, t, XC, YC, ZC, field_fn), cmap, norm)


def _render_lshape_3d(ax, domain, t, cmap, norm, n=18, field_fn=None):
    """All outer faces of an L-shaped prism."""
    Lx, Ly, Lz = domain.Lx, domain.Ly, domain.Lz
    cx, cy = domain.cut_x, domain.cut_y
    z_lin = np.linspace(0, Lz, n)

    def wall(x_vals, y_vals):
        XV, ZV = np.meshgrid(x_vals, z_lin, indexing='ij')
        YV = np.full_like(XV, y_vals) if np.ndim(y_vals) == 0 else \
             np.column_stack([y_vals]*len(z_lin)).reshape(len(x_vals), len(z_lin))
        # Fallback: constant y
        if np.ndim(y_vals) == 0:
            _surf(ax, XV, YV, ZV, _eval_3d_field(domain, t, XV, YV, ZV, field_fn), cmap, norm)
        else:
            V1 = np.tile(np.array(y_vals).reshape(-1,1), (1, len(z_lin)))
            # already shaped
            _surf(ax, XV, V1, ZV, _eval_3d_field(domain, t, XV, V1, ZV, field_fn), cmap, norm)

    def wall_xz(x_vals, y_const):
        XV, ZV = np.meshgrid(x_vals, z_lin, indexing='ij')
        YV = np.full_like(XV, y_const)
        _surf(ax, XV, YV, ZV, _eval_3d_field(domain, t, XV, YV, ZV, field_fn), cmap, norm)

    def wall_yz(x_const, y_vals):
        YV, ZV = np.meshgrid(y_vals, z_lin, indexing='ij')
        XV = np.full_like(YV, x_const)
        _surf(ax, XV, YV, ZV, _eval_3d_field(domain, t, XV, YV, ZV, field_fn), cmap, norm)

    # 6 wall segments of the L cross-section
    wall_xz(np.linspace(0, Lx, n), 0.0)          # bottom edge: y=0
    wall_yz(Lx,  np.linspace(0, cy, n))           # right short: x=Lx, y=0..cut_y
    wall_xz(np.linspace(cx, Lx, n), cy)           # inner step horiz: y=cut_y, x=cut_x..Lx
    wall_yz(cx,  np.linspace(cy, Ly, n))           # inner step vert: x=cut_x, y=cut_y..Ly
    wall_xz(np.linspace(0, cx, n), Ly)            # top edge: y=Ly, x=0..cut_x
    wall_yz(0.0, np.linspace(0, Ly, n))            # left edge: x=0

    # Top and bottom L-shaped faces — decomposed into 2 rectangles
    # Rect A: x in [0, Lx], y in [0, cy]
    # Rect B: x in [0, cx], y in [cy, Ly]
    for z_face in [0.0, Lz]:
        xa = np.linspace(0, Lx, n); ya = np.linspace(0, cy, n)
        VA, VC = np.meshgrid(xa, ya, indexing='ij')
        ZA = np.full_like(VA, z_face)
        _surf(ax, VA, VC, ZA, _eval_3d_field(domain, t, VA, VC, ZA, field_fn), cmap, norm)

        xb = np.linspace(0, cx, n); yb = np.linspace(cy, Ly, n)
        VB, VD = np.meshgrid(xb, yb, indexing='ij')
        ZB = np.full_like(VB, z_face)
        _surf(ax, VB, VD, ZB, _eval_3d_field(domain, t, VB, VD, ZB, field_fn), cmap, norm)


def _style_3d_ax(ax, domain):
    """Apply clean minimal isometric style to 3D axis with correct aspect ratio."""
    ax.set_facecolor("white")
    # Transparent panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#e0e0e0")
    ax.yaxis.pane.set_edgecolor("#e0e0e0")
    ax.zaxis.pane.set_edgecolor("#e0e0e0")
    ax.grid(False)
    # Hide tick labels and axis labels for minimal look
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(axis='both', length=0, pad=0)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

    # Set correct proportional aspect ratio per domain geometry
    if hasattr(domain, "R"):                    # Cylinder
        D = 2 * domain.R; Hz = domain.Hz
        ax.set_box_aspect([D, D, Hz])
    else:
        Lx = getattr(domain, "Lx", 1.0)
        Ly = getattr(domain, "Ly", 1.0)
        Lz = getattr(domain, "Lz", getattr(domain, "Hz", 1.0))
        ax.set_box_aspect([Lx, Ly, Lz])


def fig10_3d_volumetric(t_list: list[float] | None = None):
    """
    TRUE 3D volumetric render of all 3D domains at 4 time snapshots.

    Layout: 4 rows (time) × 4 cols (domain) + colorbar per row.
    Matches Level 8 fig1_thermal_fields.png exactly.
    """
    if t_list is None:
        t_list = [3.0, 10.0, 20.0, 30.0]
    print("  Building Fig 10 — TRUE 3D volumetric renders …")

    _DOM_NAMES  = list(DOMAINS_3D)   # rectangular, cylinder, stacked, lshape
    _DOM_LABELS = {
        "rectangular": "Rectangular Prism\n(1.3 × 0.6 × 0.4 m)",
        "cylinder":    "Cylinder\n(R=0.15 m, H=0.4 m)",
        "stacked":     "Stacked Cubes\n(2 × 0.5 m)",
        "lshape":      "L-Shape 3D\n(0.8 × 0.8 × 0.4 m)",
    }
    _DOM_COLORS = {
        "rectangular": "#1565C0",
        "cylinder":    "#2E7D32",
        "stacked":     "#E65100",
        "lshape":      "#6A1B9A",
    }

    print("    Instantiating 3D domains …")
    domains = {n: make_domain_3d(n) for n in _DOM_NAMES}

    n_t = len(t_list); n_d = len(_DOM_NAMES)
    fig = plt.figure(figsize=(5.8 * n_d + 1.4, 5.6 * n_t + 0.8))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Baseline / Reference Temperature Field — Actual 3D Geometry  "
        "(color = T [°C], z = height [m])\n"
        "Rows: time steps  ·  Columns: domain geometry  ·  Colorbar per row",
        fontsize=13, fontweight="bold", color="#1a1a1a", y=0.995,
    )

    gs = gridspec.GridSpec(n_t, n_d + 1,
                           width_ratios=[1.0] * n_d + [0.035],
                           hspace=0.08, wspace=0.06)

    cmap = plt.cm.rainbow
    norm_global = Normalize(vmin=T_WATER, vmax=T_INIT)

    for ri, t_q in enumerate(t_list):
        print(f"    t = {t_q:.0f} s …", flush=True)

        for ci, dname in enumerate(_DOM_NAMES):
            ax = fig.add_subplot(gs[ri, ci], projection='3d')
            domain = domains[dname]

            if dname in ("rectangular", "stacked"):
                _render_box_3d(ax, domain, t_q, cmap, norm_global)
            elif dname == "cylinder":
                _render_cylinder_3d(ax, domain, t_q, cmap, norm_global)
            elif dname == "lshape":
                _render_lshape_3d(ax, domain, t_q, cmap, norm_global)

            _style_3d_ax(ax, domain)
            ax.view_init(elev=25, azim=-60)

            # Column header (time row 0 only)
            if ri == 0:
                ax.set_title(_DOM_LABELS.get(dname, dname),
                             fontsize=9.5, fontweight="bold",
                             color=_DOM_COLORS.get(dname, "#333"),
                             pad=4)

            # Row label (leftmost column only)
            if ci == 0:
                ax.text2D(-0.12, 0.48, f"t = {t_q:.0f} s",
                          transform=ax.transAxes,
                          fontsize=11, fontweight="bold",
                          color="#1a1a1a", va="center",
                          rotation=90, ha="center")

        # Colorbar column
        ax_cb = fig.add_subplot(gs[ri, n_d])
        sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norm_global)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=ax_cb)
        cb.set_label("T [°C]", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_edgecolor("#aaaaaa")

    out = RESULT_DIR / "fig10_3d_volumetric.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    → {out.name}")


def fig10_3d_volumetric_pinn(registry: list[dict], arch: str,
                             domain_name: str | None = None,
                             t_list: list[float] | None = None):
    """
    TRUE 3D volumetric PINN render for one optimizer.

    If domain_name is None:
      rows = time snapshots, cols = all 3D domains.
    Else:
      rows = time snapshots, col = selected domain only.
    """
    if t_list is None:
        t_list = [3.0, 10.0, 20.0, 30.0]
    dom_names = [domain_name] if domain_name else list(DOMAINS_3D)
    print(f"  Building Fig 10 — 3D PINN volumetric renders ({arch}) …")

    dom_labels = {
        "rectangular": "Rectangular Prism",
        "cylinder": "Cylinder",
        "stacked": "Stacked Cubes",
        "lshape": "L-Shape 3D",
    }
    dom_colors = {
        "rectangular": "#1565C0",
        "cylinder": "#2E7D32",
        "stacked": "#E65100",
        "lshape": "#6A1B9A",
    }

    entries = []
    for dname in dom_names:
        k = best_k(registry, dname, arch, dim=3)
        ckpt = CKPT_DIR / f"{dname}_{arch}_k{k}_dim3.pt"
        if not ckpt.exists():
            print(f"    checkpoint not found: {ckpt.name}")
            continue
        domain = make_domain_3d(dname)
        model = load_model(dname, arch, k, dim=3)
        met = load_metrics(dname, arch, k, dim=3)
        entries.append({
            "name": dname,
            "domain": domain,
            "model": model,
            "k": k,
            "mae": float(met["mean_mae"]) if met else np.nan,
        })

    if not entries:
        print("    no matching 3D checkpoints found.")
        return

    n_t = len(t_list)
    n_d = len(entries)
    fig = plt.figure(figsize=(5.8 * n_d + 1.4, 5.6 * n_t + 0.8))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"PINN Temperature Field — {_ARCH_LABELS[arch]}  "
        "(color = T [°C], z = height [m])\n"
        "Rows: time steps  ·  Columns: domain geometry  ·  best k chosen per domain",
        fontsize=13, fontweight="bold", color="#1a1a1a", y=0.995,
    )

    gs = gridspec.GridSpec(
        n_t, n_d + 1,
        width_ratios=[1.0] * n_d + [0.035],
        hspace=0.08, wspace=0.06,
    )
    cmap = plt.cm.rainbow
    norm_global = Normalize(vmin=T_WATER, vmax=T_INIT)

    for ri, t_q in enumerate(t_list):
        print(f"    t = {t_q:.0f} s …", flush=True)
        for ci, info in enumerate(entries):
            ax = fig.add_subplot(gs[ri, ci], projection='3d')
            dname = info["name"]
            domain = info["domain"]
            model = info["model"]
            k = info["k"]
            field_fn = lambda X, Y, Z, t, _m=model, _d=domain, _k=k: _predict_surface_field(
                _m, _d, _k, t, X, Y, Z
            )

            if dname in ("rectangular", "stacked"):
                _render_box_3d(ax, domain, t_q, cmap, norm_global, field_fn=field_fn)
            elif dname == "cylinder":
                _render_cylinder_3d(ax, domain, t_q, cmap, norm_global, field_fn=field_fn)
            elif dname == "lshape":
                _render_lshape_3d(ax, domain, t_q, cmap, norm_global, field_fn=field_fn)

            _style_3d_ax(ax, domain)
            ax.view_init(elev=25, azim=-60)

            if ri == 0:
                mae_txt = "—" if not np.isfinite(info["mae"]) else f"{info['mae']:.1f}°C"
                ax.set_title(
                    f"{dom_labels.get(dname, dname)}\n"
                    f"best k={k}  |  global MAE={mae_txt}",
                    fontsize=9.0, fontweight="bold",
                    color=dom_colors.get(dname, "#333"), pad=4,
                )

            if ci == 0:
                ax.text2D(-0.12, 0.48, f"t = {t_q:.0f} s",
                          transform=ax.transAxes,
                          fontsize=11, fontweight="bold",
                          color="#1a1a1a", va="center",
                          rotation=90, ha="center")

        ax_cb = fig.add_subplot(gs[ri, n_d])
        sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norm_global)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=ax_cb)
        cb.set_label("T [°C]", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_edgecolor("#aaaaaa")

    if domain_name:
        out = RESULT_DIR / f"fig10_{domain_name}_3d_volumetric_{arch}.png"
    else:
        out = RESULT_DIR / f"fig10_3d_volumetric_{arch}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    → {out.name}")


def fig13_k_volumetric_domain(registry: list[dict], domain_name: str,
                              arch: str = "bayesian",
                              t_list: list[float] | None = None):
    """
    3D volumetric comparison for one domain:
    columns = Reference | k=1 | k=2 | k=3 | k=4 | k=5
    rows    = time snapshots
    """
    if t_list is None:
        t_list = [3.0, 7.5, 15.0, 30.0]
    print(f"  Building Fig 13 — k volumetric ({domain_name}/{arch}) …")

    domain = make_domain_3d(domain_name)
    label_map = {
        "rectangular": "Rectangular Prism",
        "cylinder": "Cylinder",
        "stacked": "Stacked Cubes",
        "lshape": "L-Shape 3D",
    }
    color_map = {
        "rectangular": "#1565C0",
        "cylinder": "#2E7D32",
        "stacked": "#E65100",
        "lshape": "#6A1B9A",
    }

    entries = [{
        "kind": "ref",
        "label": "Reference",
        "k": None,
        "shown_mae": 0.0,
        "model": None,
    }]

    for k in [1, 2, 3, 4, 5]:
        ckpt = CKPT_DIR / f"{domain_name}_{arch}_k{k}_dim3.pt"
        if not ckpt.exists():
            continue
        met = load_metrics(domain_name, arch, k, dim=3)
        model = load_model(domain_name, arch, k, dim=3)
        shown_maes = [_surface_mae_3d(model, domain, k, t_q, n=12) for t_q in t_list]
        shown_mae = float(np.nanmean(shown_maes)) if np.any(np.isfinite(shown_maes)) else np.nan
        entries.append({
            "kind": "pinn",
            "label": f"k={k}",
            "k": k,
            "global_mae": float(met["mean_mae"]) if met else np.nan,
            "shown_mae": shown_mae,
            "model": model,
        })

    pinn_entries = [e for e in entries if e["kind"] == "pinn" and np.isfinite(e["shown_mae"])]
    best_k_val = min(pinn_entries, key=lambda e: e["shown_mae"])["k"] if pinn_entries else None

    n_t = len(t_list)
    n_c = len(entries)
    fig = plt.figure(figsize=(4.2 * n_c + 1.0, 4.7 * n_t + 0.7))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"k-Volumetric Comparison — {label_map.get(domain_name, domain_name)}  |  {_ARCH_LABELS[arch]}\n"
        "Rows = common physical times on FEM grid  |  Column MAE = mean surface MAE over shown rows",
        fontsize=13, fontweight="bold", color="#1a1a1a", y=0.992,
    )

    gs = gridspec.GridSpec(
        n_t, n_c + 1,
        width_ratios=[1.0] * n_c + [0.04],
        hspace=0.08, wspace=0.06,
    )
    cmap = plt.cm.rainbow
    norm_global = Normalize(vmin=T_WATER, vmax=T_INIT)

    for ri, t_q in enumerate(t_list):
        print(f"    t = {t_q:.0f} s …", flush=True)
        for ci, info in enumerate(entries):
            ax = fig.add_subplot(gs[ri, ci], projection='3d')

            if info["kind"] == "ref":
                field_fn = None
            else:
                model = info["model"]
                k = info["k"]
                field_fn = lambda X, Y, Z, t, _m=model, _d=domain, _k=k: _predict_surface_field(
                    _m, _d, _k, t, X, Y, Z
                )

            if domain_name in ("rectangular", "stacked"):
                _render_box_3d(ax, domain, t_q, cmap, norm_global, field_fn=field_fn)
            elif domain_name == "cylinder":
                _render_cylinder_3d(ax, domain, t_q, cmap, norm_global, field_fn=field_fn)
            elif domain_name == "lshape":
                _render_lshape_3d(ax, domain, t_q, cmap, norm_global, field_fn=field_fn)

            _style_3d_ax(ax, domain)
            ax.view_init(elev=25, azim=-60)

            if ri == 0:
                if info["kind"] == "ref":
                    title = "Reference"
                    t_color = color_map.get(domain_name, "#333333")
                else:
                    mae_txt = "—" if not np.isfinite(info["shown_mae"]) else f"{info['shown_mae']:.1f}°C"
                    title = f"k={info['k']}\navg={mae_txt}"
                    t_color = "#1B5E20" if info["k"] == best_k_val else "#333333"
                ax.set_title(title, fontsize=9.2, fontweight="bold", color=t_color, pad=4)

            if ci == 0:
                ax.text2D(-0.12, 0.48, f"t = {t_q:.1f} s",
                          transform=ax.transAxes,
                          fontsize=11, fontweight="bold",
                          color="#1a1a1a", va="center",
                          rotation=90, ha="center")

        ax_cb = fig.add_subplot(gs[ri, n_c])
        sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norm_global)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=ax_cb)
        cb.set_label("T [°C]", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_edgecolor("#aaaaaa")

    out = RESULT_DIR / f"fig13_{domain_name}_k_volumetric_{arch}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    → {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 16 — PINN 3D Volumetric (per arch × per k)
# ════════════════════════════════════════════════════════════════════════════

def _make_pinn_field_fn(model: ThermalPINN, domain, k: int):
    """Return a field_fn(X, Y, Z, t) that calls PINN forward pass."""
    def field_fn(X, Y, Z, t):
        return _predict_surface_field(model, domain, k, t, X, Y, Z)
    return field_fn


def fig16_pinn_volumetric(registry: list[dict], arch: str, k: int,
                          t_list: list[float] | None = None):
    """
    PINN 3D volumetric render for one optimizer × one k value.
    Layout: rows=time snapshots × cols=3D domains, colorbar per row.
    Matches fig10 baseline style but colored by PINN prediction.
    """
    if t_list is None:
        t_list = [3.0, 10.0, 20.0, 30.0]

    _DOM_NAMES  = list(DOMAINS_3D)
    _DOM_LABELS = {
        "rectangular": f"Rectangular Prism\n(1.3 × 0.6 × 0.4 m)",
        "cylinder":    f"Cylinder\n(R=0.15 m, H=0.4 m)",
        "stacked":     f"Stacked Cubes\n(2 × 0.5 m)",
        "lshape":      f"L-Shape 3D\n(0.8 × 0.8 × 0.4 m)",
    }
    _DOM_COLORS = {
        "rectangular": "#1565C0", "cylinder": "#2E7D32",
        "stacked": "#E65100",     "lshape":   "#6A1B9A",
    }

    # Load one domain instance + model per domain
    domains = {}; models = {}; field_fns = {}
    for dname in _DOM_NAMES:
        ckp = CKPT_DIR / f"{dname}_{arch}_k{k}_dim3.pt"
        dom = make_domain_3d(dname)
        domains[dname] = dom
        if ckp.exists():
            m = load_model(dname, arch, k, dim=3)
            models[dname] = m
            field_fns[dname] = _make_pinn_field_fn(m, dom, k)
        else:
            models[dname] = None
            field_fns[dname] = None

    n_t = len(t_list); n_d = len(_DOM_NAMES)
    fig = plt.figure(figsize=(5.8 * n_d + 1.4, 5.6 * n_t + 0.8))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"PINN Temperature Field — 3D Volumetric\n"
        f"Optimizer: {_ARCH_LABELS[arch]}  |  k = {k}  "
        f"(Δt = {k * 1.5:.1f} s/window)  |  Rows: time  ·  Cols: domain",
        fontsize=13, fontweight="bold", color="#1a1a1a", y=0.997,
    )

    gs = gridspec.GridSpec(n_t, n_d + 1,
                           width_ratios=[1.0] * n_d + [0.035],
                           hspace=0.08, wspace=0.06)

    cmap = plt.cm.rainbow
    norm_global = Normalize(vmin=T_WATER, vmax=T_INIT)

    for ri, t_q in enumerate(t_list):
        print(f"    [{arch} k={k}] t = {t_q:.0f} s …", flush=True)

        for ci, dname in enumerate(_DOM_NAMES):
            ax = fig.add_subplot(gs[ri, ci], projection='3d')
            domain = domains[dname]
            fn = field_fns[dname]

            if fn is None:
                ax.text2D(0.5, 0.5, "no ckpt", transform=ax.transAxes,
                          ha="center", va="center", fontsize=8, color="#888")
            else:
                if dname in ("rectangular", "stacked"):
                    _render_box_3d(ax, domain, t_q, cmap, norm_global, field_fn=fn)
                elif dname == "cylinder":
                    _render_cylinder_3d(ax, domain, t_q, cmap, norm_global, field_fn=fn)
                elif dname == "lshape":
                    _render_lshape_3d(ax, domain, t_q, cmap, norm_global, field_fn=fn)

                # MAE annotation
                mae_v = _surface_mae_3d(models[dname], domain, k, t_q, n=10)
                ax.text2D(0.5, -0.02, f"MAE={mae_v:.1f}°C",
                          transform=ax.transAxes, ha="center", va="top",
                          fontsize=7.5, color=_ARCH_COLORS[arch], fontweight="bold")

            _style_3d_ax(ax, domain)
            ax.view_init(elev=25, azim=-60)

            if ri == 0:
                ax.set_title(_DOM_LABELS.get(dname, dname),
                             fontsize=9, fontweight="bold",
                             color=_DOM_COLORS.get(dname, "#333"), pad=4)
            if ci == 0:
                ax.text2D(-0.12, 0.48, f"t = {t_q:.0f} s",
                          transform=ax.transAxes, fontsize=11, fontweight="bold",
                          color="#1a1a1a", va="center", rotation=90, ha="center")

        ax_cb = fig.add_subplot(gs[ri, n_d])
        sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norm_global)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=ax_cb)
        cb.set_label("T [°C]", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_edgecolor("#aaaaaa")

    out = RESULT_DIR / f"fig16_pinn_volumetric_{arch}_k{k}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    → {out.name}")


def fig16_all(registry: list[dict], t_list: list[float] | None = None):
    """Generate fig16 for all arch × k combinations (15 figures).

    Optimisation: LShape3D FD solver runs ONCE, shared across all calls.
    """
    if t_list is None:
        t_list = [3.0, 10.0, 20.0, 30.0]
    print("  Building Fig 16 — all PINN volumetric figures (3 archs × 5 k values) …")

    # Pre-instantiate domains once (LShape3D FD solver runs here)
    print("    Instantiating 3D domains (LShape3D FD solver runs once) …")
    shared_domains = {n: make_domain_3d(n) for n in DOMAINS_3D}

    _DOM_LABELS = {
        "rectangular": "Rectangular Prism\n(1.3 × 0.6 × 0.4 m)",
        "cylinder":    "Cylinder\n(R=0.15 m, H=0.4 m)",
        "stacked":     "Stacked Cubes\n(2 × 0.5 m)",
        "lshape":      "L-Shape 3D\n(0.8 × 0.8 × 0.4 m)",
    }
    _DOM_COLORS = {
        "rectangular": "#1565C0", "cylinder": "#2E7D32",
        "stacked": "#E65100",     "lshape":   "#6A1B9A",
    }

    cmap = plt.cm.rainbow
    norm_global = Normalize(vmin=T_WATER, vmax=T_INIT)
    n_t = len(t_list); n_d = len(shared_domains)

    for arch in _ARCH_ORDER:
        for k in [1, 2, 3, 4, 5]:
            print(f"  [{arch} k={k}] rendering …")

            # Load models + field functions using shared domains
            models_k = {}; field_fns_k = {}
            for dname, domain in shared_domains.items():
                ckp = CKPT_DIR / f"{dname}_{arch}_k{k}_dim3.pt"
                if ckp.exists():
                    m = load_model(dname, arch, k, dim=3)
                    models_k[dname] = m
                    field_fns_k[dname] = _make_pinn_field_fn(m, domain, k)
                else:
                    models_k[dname] = None
                    field_fns_k[dname] = None

            fig = plt.figure(figsize=(5.8 * n_d + 1.4, 5.6 * n_t + 0.8))
            fig.patch.set_facecolor("white")
            fig.suptitle(
                f"PINN Temperature Field — 3D Volumetric\n"
                f"Optimizer: {_ARCH_LABELS[arch]}  |  k = {k}  "
                f"(Δt = {k * 1.5:.1f} s/window)  |  Rows: time  ·  Cols: domain",
                fontsize=13, fontweight="bold", color="#1a1a1a", y=0.997,
            )
            gs = gridspec.GridSpec(n_t, n_d + 1,
                                   width_ratios=[1.0]*n_d + [0.035],
                                   hspace=0.08, wspace=0.06)

            for ri, t_q in enumerate(t_list):
                print(f"    t = {t_q:.0f} s …", flush=True)
                for ci, (dname, domain) in enumerate(shared_domains.items()):
                    ax = fig.add_subplot(gs[ri, ci], projection='3d')
                    fn = field_fns_k[dname]

                    if fn is None:
                        ax.text2D(0.5, 0.5, "no ckpt", transform=ax.transAxes,
                                  ha="center", va="center", fontsize=9, color="#888")
                    else:
                        if dname in ("rectangular", "stacked"):
                            _render_box_3d(ax, domain, t_q, cmap, norm_global, field_fn=fn)
                        elif dname == "cylinder":
                            _render_cylinder_3d(ax, domain, t_q, cmap, norm_global, field_fn=fn)
                        elif dname == "lshape":
                            _render_lshape_3d(ax, domain, t_q, cmap, norm_global, field_fn=fn)
                        mae_v = _surface_mae_3d(models_k[dname], domain, k, t_q, n=10)
                        ax.text2D(0.5, -0.02, f"MAE = {mae_v:.1f}°C",
                                  transform=ax.transAxes, ha="center", va="top",
                                  fontsize=7.5, color=_ARCH_COLORS[arch], fontweight="bold")

                    _style_3d_ax(ax, domain)
                    ax.view_init(elev=25, azim=-60)
                    if ri == 0:
                        ax.set_title(_DOM_LABELS.get(dname, dname), fontsize=9,
                                     fontweight="bold", color=_DOM_COLORS.get(dname,"#333"),
                                     pad=4)
                    if ci == 0:
                        ax.text2D(-0.12, 0.48, f"t = {t_q:.0f} s",
                                  transform=ax.transAxes, fontsize=11, fontweight="bold",
                                  color="#1a1a1a", va="center", rotation=90, ha="center")

                ax_cb = fig.add_subplot(gs[ri, n_d])
                sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norm_global)
                sm.set_array([])
                cb = fig.colorbar(sm, cax=ax_cb)
                cb.set_label("T [°C]", fontsize=8)
                cb.ax.tick_params(labelsize=7)
                cb.outline.set_edgecolor("#aaaaaa")

            out = RESULT_DIR / f"fig16_pinn_volumetric_{arch}_k{k}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"    → {out.name}")

    print("  Fig 16 complete — 15 figures saved.")


# ════════════════════════════════════════════════════════════════════════════
# Figure 15 — Exact FEM vs PINN window-by-window comparison
# ════════════════════════════════════════════════════════════════════════════

def fig15_exact_comparison(registry: list[dict],
                           domain_name: str = "rectangle",
                           dim: int = 2,
                           k: int = 2,
                           n_grid: int = 60):
    """
    Per-window FEM vs PINN comparison with exact temperature values.

    Layout:
      Top panel  — cooling curve: FEM spatial-mean T̄(t) + PINN T̄ ± MAE bands
      Bottom grid — 4 key snapshots × 3 cols (FEM | PINN | |Error|)
    Numbers annotated directly on each panel.
    """
    print(f"  Building Fig 15 — exact comparison ({domain_name} dim={dim} k={k}) …")

    archs = _ARCH_ORDER
    domain = make_domain(domain_name) if dim == 2 else make_domain_3d(domain_name)

    # Window time endpoints matching k-skip: k×1.5s windows
    dt_w = k * 1.5
    t_ends = np.arange(dt_w, 30.0 + 1e-6, dt_w)

    # ── FEM spatial-mean T̄ at each window endpoint ──────────────────────────
    if dim == 2:
        XX, YY = _make_2d_grid(domain, n_grid)
        xi, yi = XX.ravel(), YY.ravel()
        def T_fem(t):
            T = domain.T(xi, yi, t)
            T = _clean_T(T, domain, xi, yi)
            return float(np.nanmean(T)), float(np.nanmin(T[np.isfinite(T)])), \
                   float(np.nanmax(T[np.isfinite(T)]))
    else:
        XX, YY, z_mid = _make_3d_midgrid(domain, n_grid)
        xi, yi = XX.ravel(), YY.ravel()
        zi = np.full_like(xi, z_mid)
        def T_fem(t):
            T = _clean_T(domain.T(xi, yi, zi, t), domain, xi, yi, zi)
            return float(np.nanmean(T)), float(np.nanmin(T[np.isfinite(T)])), \
                   float(np.nanmax(T[np.isfinite(T)]))

    t_plot  = np.concatenate([[0.0], t_ends])
    fem_mean = []; fem_min = []; fem_max = []
    for t_q in t_plot:
        mu, lo, hi = T_fem(t_q)
        fem_mean.append(mu); fem_min.append(lo); fem_max.append(hi)
    fem_mean = np.array(fem_mean); fem_min = np.array(fem_min); fem_max = np.array(fem_max)

    # ── Per-arch MAE from metrics JSON ──────────────────────────────────────
    arch_maes = {}
    for arch in archs:
        mp = CKPT_DIR / f"{domain_name}_{arch}_k{k}_dim{dim}_metrics.json"
        if not mp.exists():
            continue
        m = json.load(open(mp))
        # Window MAE at each window endpoint
        maes = [w["mae_C"] for w in m["windows"]]
        arch_maes[arch] = np.array(maes)

    if not arch_maes:
        print(f"    No metrics found for k={k} dim={dim}")
        return

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"FEM vs PINN — Exact Window-by-Window Comparison\n"
        f"Domain: {domain_name.capitalize()}  {'2D' if dim==2 else '3D'}  |  "
        f"k={k}  (Δt = {dt_w:.1f} s/window)",
        fontsize=13, fontweight="bold", color="#1a1a1a",
    )

    gs_top = gridspec.GridSpec(1, 1, top=0.88, bottom=0.52, left=0.07, right=0.97)
    gs_bot = gridspec.GridSpec(len(archs), 3, top=0.47, bottom=0.03,
                               left=0.07, right=0.97, hspace=0.45, wspace=0.28)

    # ── TOP: cooling curve ────────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs_top[0, 0])
    ax_c.set_facecolor("#fafafa")
    ax_c.plot(t_plot, fem_mean, color="#1565C0", lw=2.5, zorder=5,
              label="FEM  (spatial mean T̄)", marker="o", ms=5, markerfacecolor="white",
              markeredgewidth=1.5)
    ax_c.fill_between(t_plot, fem_min, fem_max, color="#1565C0", alpha=0.10,
                      label="FEM  (min–max range)")

    # Annotate FEM values at key points
    for idx, (t_q, mu) in enumerate(zip(t_plot, fem_mean)):
        if idx % max(1, len(t_plot)//6) == 0 or idx == len(t_plot)-1:
            ax_c.annotate(f"{mu:.0f}°C", (t_q, mu),
                          textcoords="offset points", xytext=(0, 8),
                          fontsize=7, color="#1565C0", ha="center", fontweight="bold")

    # Per-arch PINN ± MAE
    pinn_t = t_ends   # window endpoints where MAE is measured
    for arch, maes in arch_maes.items():
        color = _ARCH_COLORS[arch]
        # PINN mean ≈ FEM mean (the error is MAE, not bias — approximate)
        fem_at_end = np.interp(pinn_t, t_plot, fem_mean)
        ax_c.errorbar(pinn_t, fem_at_end, yerr=maes,
                      fmt="D", ms=5, color=color, lw=1.5, capsize=5,
                      label=f"{_ARCH_LABELS[arch]}  MAE ± {maes.mean():.1f}°C avg",
                      zorder=4)
        # Annotate MAE values
        for t_q, mae in zip(pinn_t, maes):
            ax_c.annotate(f"±{mae:.1f}°C", (t_q, np.interp(t_q, t_plot, fem_mean) - mae - 15),
                          fontsize=6, color=color, ha="center", alpha=0.85)

    ax_c.set_xlabel("Time [s]", fontsize=10)
    ax_c.set_ylabel("Temperature T̄ [°C]", fontsize=10)
    ax_c.set_title("Cooling Curve: FEM Reference vs PINN Prediction Error",
                   fontsize=10, fontweight="bold", color="#1a1a1a")
    ax_c.set_xlim(-0.5, 31)
    ax_c.set_ylim(T_WATER - 20, T_INIT + 30)
    ax_c.set_xticks(t_plot)
    ax_c.set_xticklabels([f"{t:.0f}s" for t in t_plot], fontsize=7, rotation=45)
    ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°C"))
    ax_c.tick_params(labelsize=8)
    ax_c.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax_c.grid(True, alpha=0.25, color="#cccccc")

    # Vertical window lines
    for t_q in pinn_t:
        ax_c.axvline(t_q, color="#aaaaaa", lw=0.6, ls="--", alpha=0.6)

    # ── BOTTOM: FEM | PINN | Error heatmaps at 3 snapshots ──────────────────
    t_snaps = [t_ends[0], t_ends[len(t_ends)//2], t_ends[-1]]  # early, mid, late

    for ri, arch in enumerate(archs):
        ckp = CKPT_DIR / f"{domain_name}_{arch}_k{k}_dim{dim}.pt"
        if not ckp.exists():
            for ci in range(3): fig.add_subplot(gs_bot[ri, ci]).set_visible(False)
            continue
        model = load_model(domain_name, arch, k, dim)

        for ci, t_q in enumerate(t_snaps):
            ax = fig.add_subplot(gs_bot[ri, ci])
            if dim == 2:
                g = eval_grid_2d(model, domain, k, t_q, n_grid=50)
            else:
                g = eval_grid_3d_mid(model, domain, k, t_q, n_grid=40)

            norm_fig = _adaptive_norm(g["T_ref"], [g["T_pred"]])
            cmap_r = plt.cm.rainbow

            T_r = np.where(np.isfinite(g["T_ref"]),  g["T_ref"],  np.nan)
            T_p = np.where(np.isfinite(g["T_pred"]), g["T_pred"], np.nan)

            # Overlay: FEM field in background, PINN as semi-transparent
            im = ax.pcolormesh(g["xx"], g["yy"], T_r, cmap=cmap_r,
                               norm=norm_fig, shading="auto", rasterized=True)
            ax.contour(g["xx"], g["yy"], T_p, levels=7,
                       colors="#FF3300", linewidths=0.9, linestyles="--", alpha=0.85)
            ax.contour(g["xx"], g["yy"], T_r, levels=7,
                       colors="white", linewidths=0.5, alpha=0.7)
            ax.set_aspect("equal")
            _style(ax)

            fem_mu  = float(np.nanmean(T_r))
            pinn_mu = float(np.nanmean(T_p))
            mae_v   = g["mae"]
            ax.set_title(
                f"t = {t_q:.0f} s  |  "
                f"FEM T̄ = {fem_mu:.1f}°C\n"
                f"PINN T̄ = {pinn_mu:.1f}°C  |  MAE = {mae_v:.2f}°C",
                fontsize=7.5, fontweight="bold",
                color=_ARCH_COLORS[arch], pad=3,
            )
            if ci == 0:
                ax.set_ylabel(f"{_ARCH_LABELS[arch]}", fontsize=8,
                              fontweight="bold", color=_ARCH_COLORS[arch])
            if ri == len(archs) - 1:
                ax.set_xlabel("x [m]", fontsize=7)

            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="5%", pad=0.04)
            cb  = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=6)

    out = RESULT_DIR / f"fig15_exact_{domain_name}_dim{dim}_k{k}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    → {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate NAS-PINN thermal-quenching figures."
    )
    parser.add_argument("--fig",    type=int, default=0,
                        help="Single figure number (0 = all)")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--arch",   default="bayesian")
    parser.add_argument("--t",      type=float, default=15.0,
                        help="Query time for heatmap figures [s]")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    reg = load_registry()

    run_all = (args.fig == 0)
    thesis_heatmaps_only = run_all

    if (run_all and not thesis_heatmaps_only) or args.fig == 1:
        fig1_k_sweep(reg)
    if run_all or args.fig == 2:
        if args.domain:
            _plot_domain_2d_figure(args.domain, reg, args.t)
        else:
            fig2_domains_2d_compare(reg, args.t, args.arch)
    if run_all or args.fig == 3:
        if args.domain:
            _plot_domain_3d_figure(args.domain, reg, args.t, archs=[args.arch])
        else:
            fig3_domains_3d_single_arch(reg, args.t, args.arch)
    if (run_all and not thesis_heatmaps_only) or args.fig == 4:
        fig4_summary_table(reg)
    if run_all or args.fig == 5:
        fig5_time_evolution(reg, args.domain or "rectangle")
    if (run_all and not thesis_heatmaps_only) or args.fig == 6:
        fig6_per_k_heatmaps(reg, args.t)
    if run_all or args.fig == 7:
        fig7_per_k_tables(reg)
    if run_all or args.fig == 8:
        if args.domain:
            fig8_3d_slices(reg, args.t, args.domain, args.arch)
        else:
            for d3 in DOMAINS_3D:
                fig8_3d_slices(reg, args.t, d3, args.arch)
    if run_all or args.fig == 9:
        fig9_3d_domain_compare(reg, args.t, args.arch)
    if run_all or args.fig == 10:
        fig10_3d_volumetric()
    if run_all or args.fig == 11:
        fig11_k_timeline()
    if run_all or args.fig == 12:
        fig12_k_tradeoff(reg)
    if args.fig == 13:
        archs = _ARCH_ORDER if args.arch == "all" else [args.arch]
        for arch in archs:
            if args.domain:
                fig10_3d_volumetric_pinn(reg, arch, args.domain)
            else:
                fig10_3d_volumetric_pinn(reg, arch)
    if args.fig == 14:
        if args.domain:
            fig13_k_volumetric_domain(reg, args.domain, args.arch)
        else:
            for d3 in DOMAINS_3D:
                fig13_k_volumetric_domain(reg, d3, args.arch)
    if args.fig == 15:
        dom = args.domain or "rectangle"
        dim = 2 if dom in DOMAINS_2D else 3
        k_v = int(args.t) if args.t != 15.0 else 2
        fig15_exact_comparison(reg, dom, dim, k=k_v)
    if args.fig == 16:
        # --arch  → single arch  (default bayesian)
        # --t     → single k value  (default: all k)
        archs = _ARCH_ORDER if args.arch == "all" else [args.arch]
        k_vals = [1,2,3,4,5] if args.t == 15.0 else [int(args.t)]
        for arch in archs:
            for kv in k_vals:
                fig16_pinn_volumetric(reg, arch, kv)

    print("\nAll figures complete.")


if __name__ == "__main__":
    main()
