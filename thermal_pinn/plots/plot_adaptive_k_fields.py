"""
thermal_pinn/plots/plot_adaptive_k_fields.py
=============================================
Exact vs Predicted temperature fields at k-change events for all adaptive-k runs.

Layout (2D)
-----------
    Columns : t=0  |  k-change moments  |  t=30
    Row 0   : Reference (FEM)
    Row 1   : PINN Prediction
    Row 2   : |Error|  (with MAE + L2 in title)

Layout (3D)
-----------
    Columns : same time steps
    Row 0   : FEM Reference (volumetric surface)
    Row 1   : PINN Prediction (volumetric surface)
    MAE text box below each panel

Note: the saved adaptive checkpoint is the model's state after ALL windows have
been trained.  Predictions at early t are the final model evaluated there —
useful for showing temporal generalization, not window-specific accuracy.

Usage
-----
    python -m thermal_pinn.plots.plot_adaptive_k_fields          # all runs
    python -m thermal_pinn.plots.plot_adaptive_k_fields --domain rectangle --arch nsga2 --dim 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

_ROOT      = Path(__file__).resolve().parent.parent
CKPT_DIR   = _ROOT / "checkpoints"
RESULT_DIR = _ROOT / "results" / "adaptive_k_fields"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(_ROOT.parent))

from thermal_pinn.plots.plot_results import (
    _render_box_3d, _render_cylinder_3d, _render_lshape_3d,
    _style_3d_ax, _is_cylinder_3d,
    _make_2d_grid, _normalise_coords, _clean_T, _surface_mean_T,
    T_WATER, T_INIT, DELTA_T, DEVICE,
    _valid_surface_mask, _pinn_forward_full,
)
from thermal_pinn.plots.plot_thesis import (
    _draw_heatmap, _draw_error_map, _add_colorbar,
    CMAP_T, CMAP_E, _WHITE, _FIG_DPI,
)
from thermal_pinn.network.pinn import ThermalPINN

NORM_T = Normalize(vmin=T_WATER, vmax=T_INIT)


def _make_adaptive_field_fn(model: ThermalPINN, domain, t_start: float, t_end: float):
    """
    Build a field_fn(X,Y,Z,t) that uses the CORRECT tau for this adaptive window.
    Unlike _make_pinn_field_fn, tau is computed from [t_start, t_end] directly,
    not from fixed k-spaced boundaries.
    """
    def field_fn(X, Y, Z, t):
        x  = np.asarray(X, dtype=np.float32).ravel()
        y  = np.asarray(Y, dtype=np.float32).ravel()
        z  = np.asarray(Z, dtype=np.float32).ravel()
        coords = np.stack([x, y, z], axis=1)

        dt = t_end - t_start
        tau_val = float(np.clip((t - t_start) / dt, 0.0, 1.0)) if dt > 1e-9 else 1.0

        valid = _valid_surface_mask(domain, coords)
        T_ic  = _clean_T(domain.T(x, y, z, t_start), domain, x, y, z)
        T_ic  = np.where(valid & np.isfinite(T_ic), T_ic, T_WATER)

        T_pred = _pinn_forward_full(model, coords, tau_val, T_ic, domain)
        T_pred = _clean_T(T_pred, domain, x, y, z)
        T_pred = np.where(valid, T_pred, np.nan)
        return T_pred.reshape(np.asarray(X).shape)
    return field_fn

C_UP   = "#1b5e20"   # green  — k increased
C_DOWN = "#b71c1c"   # red    — k decreased
C_BASE = "#1a237e"   # navy   — anchor (t=0 / t=30)

ARCH_LABELS   = {"bayesian": "Bayesian (TPE)", "nsga2": "NSGA-II", "nsga3": "NSGA-III"}
DOMAIN_LABELS = {
    "rectangle":  "Rectangle 2D",  "circle":      "Circle 2D",
    "lshape":     "L-Shape",       "rectangular": "Rectangular 3D",
    "cylinder":   "Cylinder 3D",   "stacked":     "Stacked 3D",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_domain(name: str, dim: int):
    if dim == 2:
        from thermal_pinn.physics.domains_2d import make_domain
        return make_domain(name)
    from thermal_pinn.physics.domains_3d import make_domain_3d
    return make_domain_3d(name)


def _load_model(domain_name: str, arch: str, dim: int) -> ThermalPINN | None:
    tag  = f"{domain_name}_{arch}_adaptive_dim{dim}"
    path = CKPT_DIR / f"{tag}.pt"
    if not path.exists():
        return None
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    model = ThermalPINN(dim=dim, arch=arch).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _eval_window_2d(model: ThermalPINN, domain, t_start: float, t_end: float,
                    n_grid: int = 80) -> dict:
    """
    Evaluate one adaptive window [t_start → t_end].
    Returns FEM IC (t_start), FEM reference (t_end), PINN prediction (t_end), error.
    """
    XX, YY = _make_2d_grid(domain, n_grid)
    xi, yi = XX.ravel(), YY.ravel()
    coords  = np.stack([xi, yi], axis=1).astype(np.float32)

    T_ic  = _clean_T(domain.T(xi, yi, t_start))   # FEM at t_start → PINN IC
    T_ref = _clean_T(domain.T(xi, yi, t_end))      # FEM at t_end   → ground truth
    T_ic_plot = np.where(np.isfinite(T_ic), T_ic, np.nan)
    T_ic  = np.where(np.isfinite(T_ic), T_ic, T_WATER)

    # tau=1.0 → predict at t_end
    theta_ic = np.clip((T_ic - T_WATER) / DELTA_T, 0.0, 1.2).reshape(-1, 1).astype(np.float32)
    tau_arr  = np.ones((len(coords), 1), dtype=np.float32)
    coords_n = _normalise_coords(coords, domain)

    def _t(a): return torch.tensor(a, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        theta_pred = model(_t(coords_n), _t(tau_arr), _t(theta_ic)).cpu().numpy().flatten()
    T_pred = T_WATER + DELTA_T * theta_pred
    T_pred = np.where(np.isfinite(T_ref), T_pred, np.nan)

    valid = np.isfinite(T_ref)
    mae = float(np.mean(np.abs(T_pred[valid] - T_ref[valid]))) if valid.any() else np.nan
    l2  = float(np.linalg.norm(T_pred[valid] - T_ref[valid]) /
                (np.linalg.norm(T_ref[valid] - T_WATER) + 1e-10)) if valid.any() else np.nan

    sh = (n_grid, n_grid)
    return {
        "xx": XX, "yy": YY,
        "T_ic":   T_ic_plot.reshape(sh),   # FEM IC at t_start
        "T_ref":  T_ref.reshape(sh),        # FEM reference at t_end
        "T_pred": T_pred.reshape(sh),       # PINN prediction at t_end
        "err":    np.abs(T_pred - T_ref).reshape(sh),
        "mae": mae, "l2": l2,
    }


def _k_change_events(data: dict) -> list[dict]:
    """t=0 anchor, every k-change window start, t=30 anchor."""
    windows  = data["windows"]
    schedule = data["k_schedule"]

    events = [{"t_start": 0.0, "t_end": windows[0]["t_end"],
                "k": schedule[0], "change": False,
                "label": f"t = 0–{windows[0]['t_end']:.1f} s\nk = {schedule[0]}"}]

    for i in range(1, len(schedule)):
        if schedule[i] != schedule[i - 1]:
            w     = windows[i]
            arrow = "↑" if schedule[i] > schedule[i - 1] else "↓"
            events.append({
                "t_start": w["t_start"], "t_end": w["t_end"],
                "k": schedule[i], "change": True,
                "label": (f"t = {w['t_start']:.1f}–{w['t_end']:.1f} s\n"
                          f"k {arrow} {schedule[i-1]}→{schedule[i]}"),
            })

    last = windows[-1]
    events.append({"t_start": last["t_start"], "t_end": last["t_end"],
                   "k": schedule[-1], "change": False,
                   "label": f"t = {last['t_start']:.1f}–{last['t_end']:.1f} s\nk = {schedule[-1]}"})
    return events


def _event_color(ev: dict) -> str:
    if not ev["change"]:
        return C_BASE
    return C_UP if ev["label"].count("↑") else C_DOWN


def _pick_render_fn(domain):
    name = type(domain).__name__.lower()
    if _is_cylinder_3d(domain):
        return _render_cylinder_3d
    elif "lshape" in name:
        return _render_lshape_3d
    return _render_box_3d


# ─────────────────────────────────────────────────────────────────────────────
# 2D figure  —  3 rows × n_events cols
#
# Row 0: FEM at t_end    (Exact / ground truth)   ← başa taşındı
# Row 1: PINN Prediction at t_end                 (Pred  / MAE in title)
# Row 2: |Error| = |PINN − FEM|                   (L2 in title)
#
# Column header shows pipeline + IC time.
# ─────────────────────────────────────────────────────────────────────────────
def _plot_2d(data: dict, domain, model: ThermalPINN | None,
             events: list[dict], save: bool):
    domain_name = data["domain"]
    arch        = data["arch"]
    arch_lbl    = ARCH_LABELS.get(arch, arch)
    domain_lbl  = DOMAIN_LABELS.get(domain_name, domain_name)
    if domain_name == "lshape":
        domain_lbl = "L-Shape 2D"

    n = len(events)
    fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 8.8), constrained_layout=True)
    fig.patch.set_facecolor(_WHITE)

    grids = []
    for ev in events:
        g = _eval_window_2d(model, domain, ev["t_start"], ev["t_end"]) if model else None
        grids.append(g)

    # Shared error norm across all windows
    all_err = np.concatenate([
        g["err"][np.isfinite(g["err"])].ravel() for g in grids if g is not None
    ]) if any(g is not None for g in grids) else np.array([1.0])
    norm_E = Normalize(vmin=0,
                       vmax=max(float(np.percentile(all_err, 97)) if all_err.size else 1.0, 0.5))

    for col, (ev, g) in enumerate(zip(events, grids)):
        color = _event_color(ev)

        # Column header: pipeline + IC info
        col_title = (f"FEM @ {ev['t_start']:.1f}s → PINN (k={ev['k']}) → "
                     f"FEM @ {ev['t_end']:.1f}s")
        if ev["change"]:
            arrow = "↑" if ev["label"].count("↑") else "↓"
            col_title += f"\n  k {arrow} next window"

        # Shared T norm using ref + pred
        if g is not None:
            vals = np.concatenate([
                g["T_ref"][np.isfinite(g["T_ref"])].ravel(),
                g["T_pred"][np.isfinite(g["T_pred"])].ravel(),
            ])
            vmin = float(np.percentile(vals, 2))  if vals.size else T_WATER
            vmax = float(np.percentile(vals, 98)) if vals.size else T_INIT
            norm_col = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm_col = NORM_T

        # Row 0 — Exact: FEM reference at t_end
        ax0 = axes[0, col]
        if g is not None:
            im0 = _draw_heatmap(ax0, g["xx"], g["yy"], g["T_ref"],
                                norm_col, CMAP_T, title="")
            ax0.set_title(f"t = {ev['t_end']:.1f} s", fontsize=7.5,
                          color=C_BASE, pad=3)
        ax0.set_title(col_title, fontsize=7, color=color, fontweight="bold", pad=3)

        # Row 1 — Pred: PINN prediction at t_end
        ax1 = axes[1, col]
        if g is not None:
            mae_str = f"MAE = {g['mae']:.2f}°C" if np.isfinite(g["mae"]) else "MAE = n/a"
            im1 = _draw_heatmap(ax1, g["xx"], g["yy"], g["T_pred"],
                                norm_col, CMAP_T, title="")
            ax1.set_title(mae_str, fontsize=7.5, color=color, fontweight="bold", pad=3)

        # Row 2 — |Error|
        ax2 = axes[2, col]
        if g is not None:
            l2_str = f"L2 = {g['l2']:.3f}" if np.isfinite(g["l2"]) else "L2 = n/a"
            im2 = _draw_error_map(ax2, g["xx"], g["yy"], g["err"],
                                  norm_E, title="")
            ax2.set_title(l2_str, fontsize=7.5, color=C_DOWN, fontweight="bold", pad=3)

        # Colorbars on last column
        if col == n - 1 and g is not None:
            _add_colorbar(fig, ax0, im0, "T [°C]")
            _add_colorbar(fig, ax1, im1, "T [°C]")
            _add_colorbar(fig, ax2, im2, "|ΔT| [°C]")

    # Row labels (left side)
    row_labels = [
        "① Exact\n    FEM @ t_end",
        "② Pred\n    PINN @ t_end",
        "③ |Error|\n    PINN − FEM",
    ]
    for ri, lbl in enumerate(row_labels):
        axes[ri, 0].set_ylabel(lbl, fontsize=7.5, labelpad=6, fontweight="bold")

    fig.suptitle(
        f"{domain_lbl} / {arch_lbl}  —  k-Skip MSWP pipeline at k-change events\n"
        f"mean MAE = {data['mean_mae']:.2f}°C   "
        f"FEM savings = {data['fem_savings_pct']:.0f}%   "
        f"k schedule: {data['k_schedule']}",
        fontsize=9.5, color=C_BASE, fontweight="bold",
    )
    plt.close(fig)
    if save:
        path = RESULT_DIR / f"{domain_name}_{arch}_adaptive_dim2_fields.png"
        fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight", facecolor=_WHITE)
        print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3D figure  —  2 rows × n_events cols  (Ref | Pred, volumetric)
# ─────────────────────────────────────────────────────────────────────────────
def _set_3d_limits(ax, domain):
    """Set tight axis limits so geometry fills the subplot (no extra padding)."""
    if hasattr(domain, "R"):
        R = domain.R; Hz = domain.Hz
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(0, Hz)
        ax.set_box_aspect([2*R, 2*R, Hz])
    else:
        Lx = getattr(domain, "Lx", 1.0)
        Ly = getattr(domain, "Ly", 1.0)
        Lz = getattr(domain, "Lz", getattr(domain, "Hz", 1.0))
        ax.set_xlim(0, Lx); ax.set_ylim(0, Ly); ax.set_zlim(0, Lz)
        dims = np.array([Lx, Ly, Lz], dtype=float)
        ax.set_box_aspect(dims / dims.max())


def _plot_3d(data: dict, domain, model: ThermalPINN | None,
             events: list[dict], save: bool):
    domain_name = data["domain"]
    arch        = data["arch"]
    arch_lbl    = ARCH_LABELS.get(arch, arch)
    domain_lbl  = DOMAIN_LABELS.get(domain_name, domain_name)
    if domain_name == "lshape":
        domain_lbl = "L-Shape 3D"

    n         = len(events)
    render_fn = _pick_render_fn(domain)
    cmap_obj  = plt.get_cmap("rainbow")
    norm      = Normalize(vmin=T_WATER, vmax=T_INIT)
    n_rows    = 2 if model is not None else 1

    fig = plt.figure(figsize=(3.8 * n, 4.8 * n_rows), constrained_layout=True)
    fig.patch.set_facecolor(_WHITE)

    for col, ev in enumerate(events):
        color  = _event_color(ev)
        t_end  = ev["t_end"]
        k_used = ev["k"]

        # Row 0 — Reference
        ax0 = fig.add_subplot(n_rows, n, col + 1, projection="3d")
        render_fn(ax0, domain, t_end, cmap_obj, norm, field_fn=None)
        _style_3d_ax(ax0, domain)
        _set_3d_limits(ax0, domain)
        ax0.view_init(elev=25, azim=-60)
        ax0.set_title(ev["label"], fontsize=8, color=color, fontweight="bold", pad=5)
        if col == 0:
            ax0.text2D(-0.15, 0.5, "Reference\n(FEM)", transform=ax0.transAxes,
                       fontsize=8, fontweight="bold", va="center", rotation=90, color=C_BASE)

        # Row 1 — PINN Prediction (correct tau from adaptive window boundaries)
        if model is not None and n_rows == 2:
            ax1 = fig.add_subplot(n_rows, n, n + col + 1, projection="3d")
            t_start = ev["t_start"]
            field_fn = _make_adaptive_field_fn(model, domain, t_start, t_end)
            render_fn(ax1, domain, t_end, cmap_obj, norm, field_fn=field_fn)
            _style_3d_ax(ax1, domain)
            _set_3d_limits(ax1, domain)
            ax1.view_init(elev=25, azim=-60)

            # MAE from stored metrics + mean surface T from FEM reference
            mae_val = next(
                (w["mae_C"] for w in data["windows"] if abs(w["t_end"] - t_end) < 0.8),
                None
            )
            t_mean  = _surface_mean_T(domain, t_end)
            mae_txt = (f"MAE = {mae_val:.1f} °C" if mae_val is not None else "")
            if np.isfinite(t_mean):
                mae_txt += f"  T̄ = {t_mean:.0f} °C"
            ax1.text2D(0.5, -0.05, mae_txt,
                       transform=ax1.transAxes, ha="center", va="top",
                       fontsize=9, color=color, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=_WHITE,
                                 edgecolor=color, alpha=0.95, linewidth=1.3))
            if col == 0:
                ax1.text2D(-0.15, 0.5, "PINN\nPrediction", transform=ax1.transAxes,
                           fontsize=8, fontweight="bold", va="center", rotation=90, color=C_BASE)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=fig.axes, shrink=0.6, pad=0.02, aspect=30)
    cb.set_label("T  (°C)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{domain_lbl} / {arch_lbl}  —  FEM Reference vs PINN Prediction at k-change events\n"
        f"mean MAE = {data['mean_mae']:.2f}°C   "
        f"FEM savings = {data['fem_savings_pct']:.0f}%   "
        f"k schedule: {data['k_schedule']}",
        fontsize=9.5, color=C_BASE, fontweight="bold",
    )
    plt.close(fig)
    if save:
        path = RESULT_DIR / f"{domain_name}_{arch}_adaptive_dim3_fields.png"
        fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight", facecolor=_WHITE)
        print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry
# ─────────────────────────────────────────────────────────────────────────────
def plot_fields_one(data: dict, save: bool = True) -> None:
    dim    = data["dim"]
    domain = _load_domain(data["domain"], dim)
    model  = _load_model(data["domain"], data["arch"], dim)
    events = _k_change_events(data)

    if model is None:
        print(f"  [WARN] No checkpoint for {data['domain']}/{data['arch']}/dim{dim} — skipping pred rows")

    if dim == 2:
        _plot_2d(data, domain, model, events, save)
    else:
        _plot_3d(data, domain, model, events, save)


def find_all_adaptive() -> list[dict]:
    results = []
    for p in sorted(CKPT_DIR.glob("*_adaptive_dim*_metrics.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default=None)
    parser.add_argument("--arch",   default=None)
    parser.add_argument("--dim",    type=int, default=None)
    args = parser.parse_args()

    if args.domain:
        dim  = args.dim or (3 if args.domain in {"rectangular", "cylinder", "stacked"} else 2)
        arch = args.arch or "nsga2"
        tag  = f"{args.domain}_{arch}_adaptive_dim{dim}"
        path = CKPT_DIR / f"{tag}_metrics.json"
        if not path.exists():
            print(f"Not found: {path}"); return
        with open(path) as f:
            data = json.load(f)
        plot_fields_one(data)
    else:
        all_data = find_all_adaptive()
        print(f"Found {len(all_data)} adaptive run(s).")
        for d in all_data:
            plot_fields_one(d)


if __name__ == "__main__":
    main()
