"""
plot_retrain_heatmaps.py
========================
Generates heatmaps for the best-k of each (domain, arch, dim) combination.
All archs included (bayesian, nsga2, nsga3).

Output:
  results/best_results/{arch}_{domain}_2d.png
  results/best_results/{arch}_{domain}_3d.png

Usage:
  python thermal_pinn/plot_retrain_heatmaps.py
  python thermal_pinn/plot_retrain_heatmaps.py --dim 2
  python thermal_pinn/plot_retrain_heatmaps.py --dim 3
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
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from thermal_pinn.plots.plot_results import (
    load_registry, load_model, CKPT_DIR, RESULT_DIR,
    eval_grid_2d, eval_grid_3d_mid,
    _render_box_3d, _render_cylinder_3d, _render_lshape_3d,
    _style_3d_ax, _make_pinn_field_fn, _surface_mae_3d,
    T_WATER, T_INIT, DEVICE,
    _ARCH_LABELS,
)
from thermal_pinn.plots.plot_thesis import (
    _draw_heatmap, _draw_error_map, _add_colorbar,
    CMAP_T, CMAP_E, _WHITE, _FIG_DPI, _HEATMAP_TIMES,
    DOMAINS_2D_LIST, DOMAINS_3D_LIST,
)
from thermal_pinn.physics.domains_2d import make_domain
from thermal_pinn.physics.domains_3d import make_domain_3d
from matplotlib.colors import Normalize

ARCHS       = ["bayesian", "nsga2", "nsga3"]
OUT_DIR     = RESULT_DIR / "best_results"


def _best_k(domain: str, arch: str, dim: int) -> int | None:
    best_k, best_mae = None, float("inf")
    for k in range(1, 6):
        mp = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}_metrics.json"
        if not mp.exists():
            continue
        m = json.load(open(mp))
        if m["mean_mae"] < best_mae:
            best_mae = m["mean_mae"]
            best_k   = k
    return best_k


def make_2d(domain_name: str, arch: str, k: int, out_path: Path):
    domain = make_domain(domain_name)
    model  = load_model(domain_name, arch, k, dim=2)
    times  = _HEATMAP_TIMES

    grids = [eval_grid_2d(model, domain, k, t, n_grid=80) for t in times]
    XX, YY = grids[0]["xx"], grids[0]["yy"]

    col_norms = []
    for g in grids:
        vals = np.concatenate([g["T_ref"][np.isfinite(g["T_ref"])].ravel(),
                               g["T_pred"][np.isfinite(g["T_pred"])].ravel()])
        vmin = float(np.percentile(vals, 2)) if vals.size else T_WATER
        vmax = float(np.percentile(vals, 98)) if vals.size else T_INIT
        if vmax - vmin < 0.5: vmin -= 0.5; vmax += 0.5
        col_norms.append(Normalize(vmin=vmin, vmax=vmax))

    all_err = np.concatenate([g["err"][np.isfinite(g["err"])].ravel() for g in grids])
    norm_E  = Normalize(vmin=0, vmax=max(float(np.percentile(all_err, 97)) if all_err.size else 1.0, 0.5))

    arch_label = _ARCH_LABELS.get(arch, arch)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.patch.set_facecolor(_WHITE)
    fig.suptitle(
        f"2D Quenching — Reference vs Prediction vs Error\n"
        f"Domain: {domain_name.capitalize()}  |  Optimizer: {arch_label}  |  k={k} (Δt={k*1.5:.1f}s)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    for col, (t_q, g, n_T) in enumerate(zip(times, grids, col_norms)):
        l2_str  = f"L2={g['l2']:.4f}" if np.isfinite(g['l2']) else "L2=n/a"
        fin_err = g["err"][np.isfinite(g["err"])]
        mae_str = f"MAE={np.mean(fin_err):.2f}°C" if fin_err.size else ""

        im0 = _draw_heatmap(axes[0, col], XX, YY, g["T_ref"],  n_T, CMAP_T, title=f"t = {t_q:.0f} s")
        im1 = _draw_heatmap(axes[1, col], XX, YY, g["T_pred"], n_T, CMAP_T, title=f"t = {t_q:.0f} s  [{l2_str}]")
        im2 = _draw_error_map(axes[2, col], XX, YY, g["err"], norm_E, title=f"t = {t_q:.0f} s  [{mae_str}]")
        _add_colorbar(fig, axes[0, col], im0, "T [°C]")
        _add_colorbar(fig, axes[1, col], im1, "T [°C]")
        _add_colorbar(fig, axes[2, col], im2, "|ΔT| [°C]")

    for row_i, lbl in enumerate(["Reference (FEM)", f"PINN ({arch_label})", "|Error| (°C)"]):
        axes[row_i, 0].set_ylabel(f"{lbl}\ny [m]", fontsize=7, labelpad=4)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight", facecolor=_WHITE)
    plt.close(fig)
    print(f"    → {out_path.name}")


def make_3d(domain_name: str, arch: str, k: int, out_path: Path):
    from thermal_pinn.plots.plot_thesis import fig_th1_3d_heatmaps
    from thermal_pinn.plots.plot_results import load_registry as _lr
    # Re-use existing 3D function, but save to our out_dir with custom name
    # We call it normally then move the file
    tmp_dir = RESULT_DIR / arch / f"{domain_name}_3d"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"fig_th1_3d_{domain_name}_{arch}_k{k}.png"
    if tmp_path.exists():
        tmp_path.unlink()

    reg = _lr()
    fig_th1_3d_heatmaps(reg, k, arch, domain_name, RESULT_DIR)

    if tmp_path.exists():
        out_path.write_bytes(tmp_path.read_bytes())
        print(f"    → {out_path.name}")
    else:
        print(f"    WARN: 3D figure not generated for {domain_name}/{arch}/k={k}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=0, choices=[0, 2, 3],
                   help="0 = both 2D and 3D (default)")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dims = [2, 3] if args.dim == 0 else [args.dim]

    for dim in dims:
        domains = DOMAINS_2D_LIST if dim == 2 else DOMAINS_3D_LIST
        print(f"\n{'='*55}")
        print(f"  {dim}D Best-k Heatmaps → results/best_results/")
        print(f"{'='*55}")

        for arch in ARCHS:
            for domain in domains:
                k = _best_k(domain, arch, dim)
                if k is None:
                    print(f"  SKIP {domain}/{arch} — no metrics found")
                    continue
                out_path = OUT_DIR / f"{arch}_{domain}_{dim}d_k{k}.png"
                print(f"  {domain}/{arch} → best k={k}", flush=True)
                if dim == 2:
                    make_2d(domain, arch, k, out_path)
                else:
                    make_3d(domain, arch, k, out_path)

    print(f"\nDone. Output: {OUT_DIR}")
