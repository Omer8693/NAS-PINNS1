"""
thermal_pinn/plots/plot_ws_heatmaps.py
========================================
Side-by-side heatmaps: Cold-start vs Warm-start (ws_500ep).

For each (domain, arch) best-k combination:
  Row 0: Reference (FEM)
  Row 1: Cold-start PINN
  Row 2: Warm-start PINN (ws_500ep)
  Row 3: |Error| cold
  Row 4: |Error| warm

Four time snapshots (t = 3, 10, 20, 30 s).

Output: results/ws_heatmaps/{arch}_{domain}_{dim}d_k{k}.png

Usage:
    python -m thermal_pinn.plots.plot_ws_heatmaps
    python -m thermal_pinn.plots.plot_ws_heatmaps --dim 2
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from thermal_pinn.plots.plot_results import (
    CKPT_DIR, RESULT_DIR, DEVICE,
    eval_grid_2d, eval_grid_3d_mid, T_WATER, T_INIT, _ARCH_LABELS,
    _render_box_3d, _render_cylinder_3d, _render_lshape_3d,
    _style_3d_ax, _make_pinn_field_fn, _surface_mae_3d,
    _is_cylinder_3d,
)
from thermal_pinn.plots.plot_thesis import (
    _draw_heatmap, _draw_error_map, _add_colorbar,
    CMAP_T, CMAP_E, _WHITE, _FIG_DPI, _HEATMAP_TIMES,
    DOMAINS_2D_LIST, DOMAINS_3D_LIST,
)
from thermal_pinn.physics.domains_2d import make_domain
from thermal_pinn.physics.domains_3d import make_domain_3d
from thermal_pinn.network.pinn import ThermalPINN
from matplotlib.colors import Normalize
import matplotlib.cm as _mcm

ARCHS   = ["bayesian", "nsga2", "nsga3"]
OUT_DIR = RESULT_DIR / "07_warmstart_fields"


def _load_model(domain, arch, k, dim, suffix=""):
    """Load checkpoint — suffix e.g. '' for cold, '_ws_500ep' for warm."""
    path = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}{suffix}.pt"
    if not path.exists():
        return None
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    model = ThermalPINN(dim=dim, arch=arch).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _best_k(domain, arch, dim):
    best_k, best_mae = None, float("inf")
    for k in range(1, 6):
        p = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}_metrics.json"
        if not p.exists():
            continue
        m = json.load(open(p))
        if m["mean_mae"] < best_mae:
            best_mae, best_k = m["mean_mae"], k
    return best_k


def make_2d_ws(domain_name, arch, k, out_path):
    domain     = make_domain(domain_name)
    model_cold = _load_model(domain_name, arch, k, dim=2, suffix="")
    model_warm = _load_model(domain_name, arch, k, dim=2, suffix="_ws_500ep")

    if model_cold is None or model_warm is None:
        print(f"    SKIP {domain_name}/{arch}/k={k} — checkpoint missing")
        return

    times = _HEATMAP_TIMES   # [3, 10, 20, 30]

    grids_c = [eval_grid_2d(model_cold, domain, k, t, n_grid=80) for t in times]
    grids_w = [eval_grid_2d(model_warm, domain, k, t, n_grid=80) for t in times]
    XX, YY  = grids_c[0]["xx"], grids_c[0]["yy"]

    # Shared temperature normalisation per time step
    col_norms = []
    for gc, gw in zip(grids_c, grids_w):
        vals = np.concatenate([
            gc["T_ref"][np.isfinite(gc["T_ref"])].ravel(),
            gc["T_pred"][np.isfinite(gc["T_pred"])].ravel(),
            gw["T_pred"][np.isfinite(gw["T_pred"])].ravel(),
        ])
        vmin = float(np.percentile(vals, 2))  if vals.size else T_WATER
        vmax = float(np.percentile(vals, 98)) if vals.size else T_INIT
        if vmax - vmin < 0.5: vmin -= 0.5; vmax += 0.5
        col_norms.append(Normalize(vmin=vmin, vmax=vmax))

    all_err = np.concatenate([
        g["err"][np.isfinite(g["err"])].ravel()
        for gs in [grids_c, grids_w] for g in gs
    ])
    norm_E = Normalize(vmin=0,
                       vmax=max(float(np.percentile(all_err, 97)) if all_err.size else 1.0, 0.5))

    arch_label = _ARCH_LABELS.get(arch, arch)
    n_rows, n_cols = 5, len(times)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 14))
    fig.patch.set_facecolor(_WHITE)
    fig.suptitle(
        f"Cold-Start vs Warm-Start (500 ep) — {domain_name.capitalize()} 2D\n"
        f"Arch: {arch_label}  |  k={k} (Δt={k*1.5:.1f}s)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    row_labels = ["Reference (FEM)", f"PINN cold (800 ep)", f"PINN warm (500 ep)",
                  "|Error| cold [°C]", "|Error| warm [°C]"]

    for col, (t_q, gc, gw, n_T) in enumerate(zip(times, grids_c, grids_w, col_norms)):
        l2c = f"L2={gc['l2']:.3f}" if np.isfinite(gc['l2']) else "n/a"
        l2w = f"L2={gw['l2']:.3f}" if np.isfinite(gw['l2']) else "n/a"

        im0 = _draw_heatmap(axes[0, col], XX, YY, gc["T_ref"],  n_T, CMAP_T, title=f"t = {t_q:.0f} s")
        im1 = _draw_heatmap(axes[1, col], XX, YY, gc["T_pred"], n_T, CMAP_T, title=f"{l2c}")
        im2 = _draw_heatmap(axes[2, col], XX, YY, gw["T_pred"], n_T, CMAP_T, title=f"{l2w}")
        im3 = _draw_error_map(axes[3, col], XX, YY, gc["err"], norm_E,
                              title=f"MAE={np.nanmean(gc['err']):.2f}°C")
        im4 = _draw_error_map(axes[4, col], XX, YY, gw["err"], norm_E,
                              title=f"MAE={np.nanmean(gw['err']):.2f}°C")

        if col == n_cols - 1:
            _add_colorbar(fig, axes[0, col], im0, "T [°C]")
            _add_colorbar(fig, axes[1, col], im1, "T [°C]")
            _add_colorbar(fig, axes[2, col], im2, "T [°C]")
            _add_colorbar(fig, axes[3, col], im3, "|ΔT| [°C]")
            _add_colorbar(fig, axes[4, col], im4, "|ΔT| [°C]")

    for ri, lbl in enumerate(row_labels):
        axes[ri, 0].set_ylabel(lbl, fontsize=8, labelpad=4)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight", facecolor=_WHITE)
    plt.close(fig)
    print(f"    → {out_path.name}")


def _render_fn(domain):
    """Pick the right volumetric renderer for the domain type."""
    name = type(domain).__name__.lower()
    if _is_cylinder_3d(domain):
        return _render_cylinder_3d
    elif "lshape" in name:
        return _render_lshape_3d
    else:
        return _render_box_3d


def make_3d_ws(domain_name, arch, k, out_path):
    """Volumetric 3D warm-start comparison — same style as existing best_results 3D plots."""
    domain     = make_domain_3d(domain_name)
    model_cold = _load_model(domain_name, arch, k, dim=3, suffix="")
    model_warm = _load_model(domain_name, arch, k, dim=3, suffix="_ws_500ep")

    if model_cold is None or model_warm is None:
        print(f"    SKIP {domain_name}/{arch}/k={k} — checkpoint missing")
        return

    times      = _HEATMAP_TIMES   # [3, 10, 20, 30]
    render_fn  = _render_fn(domain)
    arch_label = _ARCH_LABELS.get(arch, arch)
    cmap_obj   = _mcm.get_cmap(CMAP_T)   # convert string → callable colormap

    # Fixed physical range — same approach as fig16_pinn_volumetric which works for all domains
    norm = Normalize(vmin=T_WATER, vmax=T_INIT)

    # 3 rows × 4 cols: row0=Reference, row1=Cold, row2=Warm
    n_cols = len(times)
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(_WHITE)
    fig.suptitle(
        f"3D Volumetric Temperature — {domain_name.capitalize()}\n"
        f"Optimizer: {arch_label}  |  k={k} (Δt={k*1.5:.1f}s/window)\n"
        f"Row 1: FEM Reference   Row 2: PINN Cold (800 ep)   Row 3: PINN Warm (500 ep)",
        fontsize=11, fontweight="bold",
    )

    row_labels = ["Reference\n(FEM)", f"PINN cold\n(800 ep)", f"PINN warm\n(500 ep)"]

    mae_rows = []
    for row, (label, model) in enumerate(
        [("ref", None), ("cold", model_cold), ("warm", model_warm)]
    ):
        maes = []
        for col, t in enumerate(times):
            ax = fig.add_subplot(3, n_cols, row * n_cols + col + 1, projection="3d")
            field_fn = None if model is None else _make_pinn_field_fn(model, domain, k)
            render_fn(ax, domain, t, cmap_obj, norm, field_fn=field_fn)
            _style_3d_ax(ax, domain)
            ax.view_init(elev=25, azim=-60)   # consistent angle (same as fig16)
            # Set explicit axis limits
            if hasattr(domain, "R"):
                ax.set_xlim(-domain.R, domain.R)
                ax.set_ylim(-domain.R, domain.R)
                ax.set_zlim(0, domain.Hz)
            else:
                Lx = getattr(domain, "Lx", 1.0)
                Ly = getattr(domain, "Ly", 1.0)
                Lz = getattr(domain, "Lz", getattr(domain, "Hz", 1.0))
                ax.set_xlim(0, Lx); ax.set_ylim(0, Ly); ax.set_zlim(0, Lz)
            # Normalise box aspect so all domains fill the subplot equally
            if hasattr(domain, "R"):
                D = 2 * domain.R; Hz = domain.Hz
                dims = np.array([D, D, Hz], dtype=float)
            else:
                dims = np.array([
                    getattr(domain, "Lx", 1.0),
                    getattr(domain, "Ly", 1.0),
                    getattr(domain, "Lz", getattr(domain, "Hz", 1.0)),
                ], dtype=float)
            ax.set_box_aspect(dims / dims.max())
            if row == 0:
                ax.set_title(f"t = {t:.0f} s", fontsize=9, fontweight="bold", pad=2)
            if col == 0:
                ax.text2D(-0.18, 0.5, label, transform=ax.transAxes,
                          fontsize=9, fontweight="bold", va="center", rotation=90)
            if model is not None:
                try:
                    g   = eval_grid_3d_mid(model, domain, k, t, n_grid=20)
                    mae = float(g["mae"]) if np.isfinite(g["mae"]) else float("nan")
                except Exception:
                    mae = float("nan")
                maes.append(mae)
                lbl = f"MAE={mae:.1f}°C" if np.isfinite(mae) else "MAE=n/a"
                # text2D stays in front of the 3D geometry (xlabel gets hidden behind it)
                ax.text2D(0.5, -0.04, lbl, transform=ax.transAxes,
                          ha="center", va="top", fontsize=7.5, fontweight="bold",
                          color="#333333")
        mae_rows.append(maes)

    # Shared colorbar on the right
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, orientation="vertical",
                        fraction=0.012, pad=0.02, aspect=35)
    cbar.set_label("T [°C]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 0.97, 0.92])
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight", facecolor=_WHITE)
    plt.close(fig)
    print(f"    → {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=0, choices=[0, 2, 3])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dims = [2, 3] if args.dim == 0 else [args.dim]

    for dim in dims:
        domains = DOMAINS_2D_LIST if dim == 2 else DOMAINS_3D_LIST
        make_fn  = make_2d_ws    if dim == 2 else make_3d_ws
        print(f"\n{'='*55}")
        print(f"  {dim}D Warm-Start Heatmaps → results/ws_heatmaps/")
        print(f"{'='*55}")

        for arch in ARCHS:
            for domain in domains:
                for k in range(1, 6):
                    cold_ckpt = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}.pt"
                    warm_ckpt = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}_ws_500ep.pt"
                    if not cold_ckpt.exists() and not warm_ckpt.exists():
                        continue
                    out = OUT_DIR / f"{arch}_{domain}_{dim}d_k{k}_ws.png"
                    print(f"  {domain}/{arch} k={k}", flush=True)
                    make_fn(domain, arch, k, out)

    print(f"\nDone → {OUT_DIR}/")
