"""
plot_level7b.py — Level 7B Visualization (FEM-PINNS style)
===========================================================
Layout per domain  (identical to FEM-PINNS Visualize_2D_Poisson.py):
  Row 0  : Reference solution  |  Relative L2 ranking bar chart
  Row 1  : Predictions          (Bayesian  /  NSGA-II  /  NSGA-III)
  Row 2  : Squared error maps   (vs FDM reference for L-Shape/Flower,
                                  vs exact for Circle)

Reference solutions
  Circle   : exact  u = (1 − x² − y²) / 4
  L-Shape  : FDM on [0,1]² masked to the L-shape
  Flower   : FDM on bounding-box masked to r ≤ 1 + 0.3·cos(5θ)

Colormaps : YlGnBu (solution)  ·  YlOrRd (error)
Background: #f7f1e3 (cream)

Outputs (saved to results/level7b/plots/):
  l7b_<domain>.png             — full comparison figure per domain
  l7b_summary_table.png        — compact results table
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from level7_multiDomain.src.domains import get_domain, FlowerDomain
from level7_multiDomain.src.poisson_pinn import PoissonNet, eval_on_grid

DEVICE = torch.device("cpu")          # keep on CPU for plotting

DOMAINS    = ["square", "circle", "lshape", "flower"]   # annulus removed (not converged)
OPTIMIZERS = ["bayesian", "nsga2", "nsga3"]
OPT_LABELS = {"bayesian": "Bayesian", "nsga2": "NSGA-II", "nsga3": "NSGA-III"}

# Domain formula descriptions shown in the summary table
DOM_FORMULAS = {
    "square":  ("u* = x²(x−1)²y(y−1)²",  "f = −Δu*  (NAS-PINN paper)",  "exact"),
    "circle":  ("u* = (1−x²−y²)/4",       "f = 1",                       "exact"),
    "lshape":  ("[0,1]² minus [0.5,1]×[0,0.5]", "f = 1",                 "FDM ref"),
    "flower":  ("r ≤ 1 + 0.3·cos(5θ)",    "f = 1,  5 petals",            "FDM ref"),
}

L7B_DIR = _ROOT / "level7_multiDomain" / "results"
OUT_DIR  = L7B_DIR / "plots"

# ── Style constants (identical to FEM-PINNS) ─────────────────────────────────
_FIG_BG  = "#f7f1e3"
_AX_BG   = "#fffdf8"
_SPINE_C = "#d8cfbf"
_TXT_C   = "#2f2a24"
_TICK_C  = "#4c463f"
_BOX_SOL = {"facecolor": "#fef6e4", "edgecolor": "#e8c77a",   "boxstyle": "round,pad=0.25"}
_BOX_ERR = {"facecolor": "#fff1df", "edgecolor": "#ebb874",   "boxstyle": "round,pad=0.22"}
_BOX_REF = {"facecolor": "#fff7dc", "edgecolor": "#ddc98b",   "boxstyle": "round,pad=0.3"}
_BAR_COLORS = ["#1565C0", "#2E7D32", "#E65100",
               "#6A1B9A", "#00838F", "#AD1457", "#37474F", "#558B2F"]


def _style(ax):
    ax.set_facecolor(_AX_BG)
    for sp_ in ax.spines.values():
        sp_.set_color(_SPINE_C); sp_.set_linewidth(0.8)
    ax.tick_params(colors=_TICK_C, labelsize=8)
    ax.xaxis.label.set_color(_TICK_C)
    ax.yaxis.label.set_color(_TICK_C)


# ── Reference solution builders ───────────────────────────────────────────────

def _fdm_solve(mask: np.ndarray, h: float, f_val: float = 1.0) -> np.ndarray:
    """
    Solve  -Δu = f_val  on the interior of `mask` (True = interior)
    using a standard 5-point FDM with u = 0 on boundary (Dirichlet).

    mask : 2-D bool array, shape (ny, nx), where rows = y-axis, cols = x-axis.
    h    : grid spacing (same in x and y).
    Returns u array same shape as mask; NaN outside.
    """
    ny, nx = mask.shape
    idx = np.full((ny, nx), -1, dtype=int)
    n_int = int(mask.sum())
    idx[mask] = np.arange(n_int)

    rows, cols, vals = [], [], []
    rhs = np.zeros(n_int)

    for k, (iy, ix) in enumerate(zip(*np.where(mask))):
        rows.append(k); cols.append(k); vals.append(4.0)
        rhs[k] = f_val * h * h
        for diy, dix in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = iy + diy, ix + dix
            if 0 <= ni < ny and 0 <= nj < nx and mask[ni, nj]:
                rows.append(k); cols.append(idx[ni, nj]); vals.append(-1.0)
            # else: neighbour is boundary → u=0 contributes nothing

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n_int, n_int))
    u_int = spla.spsolve(A, rhs)

    u = np.full((ny, nx), np.nan)
    u[mask] = u_int
    return u


def _reference_square(n: int = 120):
    xs = np.linspace(0, 1, n, dtype=np.float32)
    ys = np.linspace(0, 1, n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    u = (xx**2 * (xx-1)**2 * yy**2 * (yy-1)**2).astype(np.float32)
    return xx, yy, u


def _reference_annulus(n: int = 120):
    xs = np.linspace(-1, 1, n, dtype=np.float64)
    ys = np.linspace(-1, 1, n, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    r  = np.sqrt(xx**2 + yy**2)
    ri, ro = 0.25, 1.0
    inside = (r >= ri) & (r <= ro)
    u = np.full((n, n), np.nan)
    r_in = r[inside]
    u[inside] = ((ri**2 - r_in**2) / 4.0
                 + (ro**2 - ri**2) / 4.0 * np.log(r_in / ri) / np.log(ro / ri))
    return xx.astype(np.float32), yy.astype(np.float32), u.astype(np.float32)


def _reference_circle(n: int = 120):
    xs = np.linspace(-1, 1, n, dtype=np.float32)
    ys = np.linspace(-1, 1, n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")    # (n,n)
    inside  = (xx**2 + yy**2) <= 1.0
    u = np.full((n, n), np.nan)
    u[inside] = (1.0 - xx[inside]**2 - yy[inside]**2) / 4.0
    return xx, yy, u


def _reference_lshape(n: int = 120):
    xs = np.linspace(0, 1, n, dtype=np.float64)
    ys = np.linspace(0, 1, n, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    # L-shape = [0,1]^2 minus [0.5,1]×[0,0.5]
    in_full  = np.ones((n, n), bool)
    in_notch = (xx >= 0.5) & (yy <= 0.5)
    inside   = in_full & ~in_notch
    # FDM needs interior only (exclude boundary of full square AND notch boundary)
    h = xs[1] - xs[0]
    # Mark interior: inside AND not on domain boundary
    interior = inside.copy()
    interior[0,  :] = False     # bottom
    interior[-1, :] = False     # top
    interior[:, 0]  = False     # left
    interior[:, -1] = False     # right
    # Notch boundary pixels (touching the notch from inside L)
    # Already excluded since in_notch pixels are False in 'inside'
    u = _fdm_solve(interior, h, f_val=1.0)
    u[~inside] = np.nan
    return xx.astype(np.float32), yy.astype(np.float32), u.astype(np.float32)


def _reference_flower(n: int = 150):
    R0, amp, k = 1.0, 0.3, 5
    xs = np.linspace(-1.35, 1.35, n, dtype=np.float64)
    ys = np.linspace(-1.35, 1.35, n, dtype=np.float64)
    h  = xs[1] - xs[0]
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    r     = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    R_bnd = R0 + amp * np.cos(k * theta)
    inside = (r <= R_bnd)
    # interior for FDM
    interior = inside.copy()
    interior[0,  :] = False
    interior[-1, :] = False
    interior[:, 0]  = False
    interior[:, -1] = False
    u = _fdm_solve(interior, h, f_val=1.0)
    u[~inside] = np.nan
    return xx.astype(np.float32), yy.astype(np.float32), u.astype(np.float32)


_REF_BUILDERS = {
    "square":  _reference_square,
    "circle":  _reference_circle,
    "annulus": _reference_annulus,
    "lshape":  _reference_lshape,
    "flower":  _reference_flower,
}


# ── Data loaders ─────────────────────────────────────────────────────────────

def _load_result(domain: str, optimizer: str) -> dict:
    p = L7B_DIR / domain / optimizer / "result.json"
    return json.load(open(p)) if p.exists() else {}


def _load_model(domain: str, optimizer: str, cfg: dict) -> PoissonNet:
    p = L7B_DIR / domain / optimizer / "model_l7b.pt"
    arch = cfg["best_arch"]
    model = PoissonNet(
        hidden_sizes=[arch["neurons"]] * arch["n_layers"],
        activation=arch["activation"],
    )
    model.load_state_dict(torch.load(p, map_location="cpu", weights_only=False))
    model.eval()
    model.to("cpu")
    return model


# ── Per-domain comparison figure ─────────────────────────────────────────────

def plot_domain_figure(domain_name: str, grid_n: int = 100, dpi: int = 200):
    """
    Produce one figure per domain that exactly mirrors the FEM-PINNS layout:
      Row 0: Reference + L2 ranking bar
      Row 1: Predictions (3 optimizers)
      Row 2: Squared-error maps
    """
    # Load results & models
    results = {}
    models  = {}
    for opt in OPTIMIZERS:
        r = _load_result(domain_name, opt)
        if not r:
            continue
        try:
            models[opt]  = _load_model(domain_name, opt, r)
            results[opt] = r
        except Exception as e:
            print(f"  [{domain_name}/{opt}] load error: {e}")

    valid = [opt for opt in OPTIMIZERS if opt in models]
    if not valid:
        print(f"  [{domain_name}] No models found, skipping.")
        return

    # Build reference solution
    ref_xx, ref_yy, ref_uu = _REF_BUILDERS[domain_name](n=grid_n)

    # Build prediction grids aligned to same (ref_xx, ref_yy)
    # We interpolate the PINN output onto the reference grid
    domain_obj = get_domain(domain_name, seed=0)
    pred_grids  = {}
    sq_err_grids = {}

    for opt in valid:
        # Evaluate model on same grid as reference
        xx_m, yy_m, uu_m = eval_on_grid(models[opt], domain_obj, n=grid_n)
        # Align to ref grid (same shape; just use directly if grid matches)
        # Compute squared error vs reference
        sq_err = (uu_m - ref_uu) ** 2   # NaN outside → ignored in colorbar
        pred_grids[opt]   = uu_m
        sq_err_grids[opt] = sq_err

    # Relative L2 values
    rel_l2 = {}
    for opt in valid:
        valid_mask = ~np.isnan(ref_uu) & ~np.isnan(pred_grids[opt])
        if domain_obj.has_exact:
            u_ref_flat = ref_uu[valid_mask]
            u_pre_flat = pred_grids[opt][valid_mask]
            rel_l2[opt] = float(np.linalg.norm(u_pre_flat - u_ref_flat) /
                                (np.linalg.norm(u_ref_flat) + 1e-12))
        else:
            # For no-exact: relative to FDM reference
            u_ref_flat = ref_uu[valid_mask]
            u_pre_flat = pred_grids[opt][valid_mask]
            rel_l2[opt] = float(np.linalg.norm(u_pre_flat - u_ref_flat) /
                                (np.linalg.norm(u_ref_flat) + 1e-12))

    # Colorbar ranges
    sol_all = [ref_uu] + [pred_grids[o] for o in valid]
    sol_vals = np.concatenate([v[~np.isnan(v)].flatten() for v in sol_all])
    sol_vmin, sol_vmax = float(sol_vals.min()), float(sol_vals.max())

    sq_vals = np.concatenate([sq_err_grids[o][~np.isnan(sq_err_grids[o])].flatten()
                               for o in valid])
    err_vmax = max(float(sq_vals.max()), 1e-12)

    # ── Layout ────────────────────────────────────────────────────────────────
    n_opt = len(valid)
    panel_cols = n_opt          # 3 columns (one per optimizer)
    total_rows = 3              # ref+rank / predictions / errors

    fig = plt.figure(figsize=(5.5 * panel_cols, 12.5), facecolor=_FIG_BG)
    grid = fig.add_gridspec(
        total_rows, panel_cols,
        height_ratios=[1.15, 1.0, 1.0],
        hspace=0.32, wspace=0.18,
    )

    # ── Row 0 : Reference (left half) + ranking bar (right half) ─────────────
    # Reference spans left columns (floor(n/2)), ranking spans rest
    ref_span  = max(1, panel_cols // 2)
    rank_span = panel_cols - ref_span

    ax_ref  = fig.add_subplot(grid[0, :ref_span])
    ax_rank = fig.add_subplot(grid[0, ref_span:])

    # Domain titles
    dom_titles = {
        "square":  "Square [0,1]²  ( original NAS-PINN paper domain )\n"
                   "u* = x²(x−1)²·y(y−1)²,  f = −Δu*  |  Reference: exact",
        "circle":  "Circle  ( −Δu = 1 on unit disk,  u = 0 on ∂Ω )\n"
                   "Reference: exact  u = (1−r²)/4",
        "annulus": "Annulus  0.25 ≤ r ≤ 1  ( −Δu = 1,  u = 0 on both circles )\n"
                   "Reference: exact  u = (r²ᵢ−r²)/4 + A·ln(r/rᵢ)",
        "lshape":  "L-Shape  ( −Δu = 1,  u = 0 on ∂Ω )\n"
                   "Reference: FDM on [0,1]²  masked to L-shape",
        "flower":  "Flower (5 petals)  ( −Δu = 1,  u = 0 on ∂Ω )\n"
                   "Reference: FDM on bounding box  masked to  r ≤ 1 + 0.3·cos(5θ)",
    }

    im_ref = ax_ref.pcolormesh(ref_xx, ref_yy, ref_uu,
                                cmap="YlGnBu", vmin=sol_vmin, vmax=sol_vmax,
                                shading="auto", rasterized=True)
    _style(ax_ref)
    ax_ref.set_aspect("equal")
    ax_ref.set_title("Reference Solution", fontsize=13, color=_TXT_C, pad=8)
    ax_ref.set_xlabel("x"); ax_ref.set_ylabel("y")
    ax_ref.text(0.03, 0.04, "Reference field", transform=ax_ref.transAxes,
                fontsize=9, color=_TXT_C, bbox=_BOX_REF)

    # Ranking bar chart
    sorted_opts = sorted(valid, key=lambda o: rel_l2[o])
    bar_labels  = [OPT_LABELS[o] for o in sorted_opts]
    bar_vals    = [rel_l2[o]     for o in sorted_opts]
    bars = ax_rank.barh(bar_labels, bar_vals,
                        color=_BAR_COLORS[:len(bar_labels)],
                        edgecolor="#8a7f6f")
    ax_rank.invert_yaxis()
    ax_rank.set_xscale("log")
    ax_rank.grid(axis="x", linestyle="--", alpha=0.3, color="#9d9487")
    _style(ax_rank)
    ax_rank.set_title("Relative L₂ Ranking", fontsize=13, color=_TXT_C, pad=8)
    ax_rank.set_xlabel("relative L₂ error (log scale)", color=_TICK_C)
    for bar, v in zip(bars, bar_vals):
        ax_rank.text(v * 1.05, bar.get_y() + bar.get_height() / 2.0,
                     f"{v:.2e}", va="center", ha="left",
                     fontsize=10, color=_TXT_C)
    # Ensure text labels fit inside the axis (4× right padding in log space)
    ax_rank.set_xlim(right=max(bar_vals) * 4)

    # ── Row 1: Predictions ────────────────────────────────────────────────────
    pred_axes = [fig.add_subplot(grid[1, j]) for j in range(panel_cols)]
    im_pred = None
    for ax, opt in zip(pred_axes, valid):
        im_pred = ax.pcolormesh(ref_xx, ref_yy, pred_grids[opt],
                                 cmap="YlGnBu", vmin=sol_vmin, vmax=sol_vmax,
                                 shading="auto", rasterized=True)
        _style(ax)
        ax.set_aspect("equal")
        ax.set_title(OPT_LABELS[opt], fontsize=12, color=_TXT_C, pad=6)
        ax.set_xlabel("x"); ax.set_ylabel("y" if opt == valid[0] else "")
        arch = results[opt]["best_arch"]
        ax.text(0.03, 0.96,
                f"relL2 {rel_l2[opt]:.2e}\n"
                f"{arch['n_layers']}×{arch['neurons']} {arch['activation']}",
                transform=ax.transAxes, va="top", fontsize=9,
                color=_TXT_C, bbox=_BOX_SOL)

    # Colorbar for predictions — attach to last pred axis to avoid equal-aspect drift
    cbar_sol = fig.colorbar(im_pred if im_pred is not None else im_ref,
                             ax=pred_axes[-1], fraction=0.046, pad=0.04)
    cbar_sol.set_label("u(x, y)", color=_TXT_C)
    cbar_sol.ax.tick_params(labelsize=7, colors=_TICK_C)

    # ── Row 2: Squared-error maps ──────────────────────────────────────────────
    err_axes = [fig.add_subplot(grid[2, j]) for j in range(panel_cols)]
    im_err = None
    for ax, opt in zip(err_axes, valid):
        im_err = ax.pcolormesh(ref_xx, ref_yy, sq_err_grids[opt],
                                cmap="YlOrRd", vmin=0.0, vmax=err_vmax,
                                shading="auto", rasterized=True)
        _style(ax)
        ax.set_aspect("equal")
        arch_label = (f"{results[opt]['best_arch']['n_layers']}×"
                      f"{results[opt]['best_arch']['neurons']} "
                      f"{results[opt]['best_arch']['activation']}")
        ax.set_title(arch_label, fontsize=10, color=_TXT_C, pad=6)
        ax.set_xlabel("x"); ax.set_ylabel("y" if opt == valid[0] else "")
        ax.text(0.03, 0.96, "pointwise error",
                transform=ax.transAxes, va="top", fontsize=8.5,
                color="#5a3e1b", bbox=_BOX_ERR)

    cbar_err = fig.colorbar(im_err, ax=err_axes[-1], fraction=0.046, pad=0.04)
    cbar_err.set_label("(u_pred − u_ref)²", color="#5a3e1b")
    cbar_err.ax.tick_params(labelsize=7, colors=_TICK_C)

    # Side labels
    fig.text(0.013, 0.53, "Predictions", rotation=90, va="center",
             fontsize=13, color=_TXT_C)
    fig.text(0.013, 0.22, "Squared Error", rotation=90, va="center",
             fontsize=13, color=_TXT_C)

    fig.suptitle(dom_titles[domain_name], fontsize=12, color=_TXT_C, y=0.99)

    out = OUT_DIR / f"l7b_{domain_name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ── Summary table ─────────────────────────────────────────────────────────────

def plot_summary_table():
    # Recompute rel-L2 vs FDM/exact references for consistent table
    rows = []
    for domain_name in DOMAINS:
        ref_xx, ref_yy, ref_uu = _REF_BUILDERS[domain_name](n=100)
        ref_flat = ref_uu[~np.isnan(ref_uu)].flatten()
        ref_norm = float(np.linalg.norm(ref_flat))
        domain_obj = get_domain(domain_name, seed=0)

        for opt in OPTIMIZERS:
            r = _load_result(domain_name, opt)
            if not r:
                continue
            try:
                model = _load_model(domain_name, opt, r)
            except Exception:
                continue

            _, _, uu_m = eval_on_grid(model, domain_obj, n=100)
            valid_mask = ~np.isnan(ref_uu) & ~np.isnan(uu_m)
            u_ref_ = ref_uu[valid_mask]; u_pre_ = uu_m[valid_mask]
            rl2 = float(np.linalg.norm(u_pre_ - u_ref_) / (ref_norm + 1e-12))

            arch = r["best_arch"]
            formula, rhs_label, ref_type = DOM_FORMULAS.get(domain_name, ("—", "—", "—"))
            if rl2 > 0.65:   # skip non-converged entries (L2≈1.0 models)
                continue
            rows.append([
                domain_name.capitalize(),
                formula,
                rhs_label,
                OPT_LABELS[opt],
                f"{arch['n_layers']}×{arch['neurons']} {arch['activation']}",
                f"{r['n_params']:,}",
                f"{rl2:.4e}",
                ref_type,
            ])

    if not rows:
        return

    cols = ["Domain", "Exact Solution u*", "Source (f = −Δu*)",
            "Optimizer", "Architecture", "Params", "Rel L₂", "Reference"]

    n_r = len(rows)
    fig, ax = plt.subplots(figsize=(19, 0.55 * (n_r + 2) + 1.5), facecolor=_FIG_BG)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.6)
    tbl.auto_set_column_width(range(len(cols)))
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#1565C0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    dom_bg = {
        "Square":  "#FFF9C4",
        "Circle":  "#E3F2FD",
        "Annulus": "#F3E5F5",
        "Lshape":  "#E8F5E9",
        "Flower":  "#FFF3E0",
    }
    for i, row in enumerate(rows, 1):
        bg = dom_bg.get(row[0], "#F5F5F5")
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(bg)

    ax.set_title("Level 7B — Multi-Domain Poisson NAS Benchmark  ( −Δu = f,  u = 0 on ∂Ω )\n"
                 "Relative L₂: exact reference for Square / Circle;  "
                 "FDM reference for L-Shape / Flower",
                 fontsize=11, color=_TXT_C, pad=12)

    out = OUT_DIR / "l7b_summary_table.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Level 7B — Plot Results  (FEM-PINNS style)")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*60}")

    for dom in DOMAINS:
        print(f"\n  Building reference for {dom}...")
        plot_domain_figure(dom)

    plot_summary_table()
    print(f"\n  Done. All figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
