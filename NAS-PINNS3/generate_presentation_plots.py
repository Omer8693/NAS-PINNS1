"""
generate_presentation_plots.py
==============================
Creates clean, standardised plots for the NAS-PINN presentation.

All outputs are saved to:   results/pres_plots/

Run this script BEFORE make_presentation.py.

Usage:
    python generate_presentation_plots.py
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "pres_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global style ───────────────────────────────────────────────────────────────
_FIG_BG  = "#f7f1e3"
_AX_BG   = "#fffdf8"
_SPINE   = "#d8cfbf"
_TXT     = "#2f2a24"
_TICK    = "#4c463f"

C_BAY = "#1565C0"
C_N2  = "#C62828"
C_N3  = "#2E7D32"
C_FEM = "#5D4037"
C_MEAS= "#E65100"
C_GRAY= "#9E9E9E"

OPT_COLORS  = {"bayesian": C_BAY, "nsga2": C_N2, "nsga3": C_N3}
OPT_LABELS  = {"bayesian": "Bayesian\n5×151 relu", "nsga2": "NSGA-II\n3×153 tanh", "nsga3": "NSGA-III\n3×75 tanh"}
OPT_SHORT   = {"bayesian": "Bayesian", "nsga2": "NSGA-II", "nsga3": "NSGA-III"}


def _style(ax):
    ax.set_facecolor(_AX_BG)
    for sp in ax.spines.values():
        sp.set_color(_SPINE); sp.set_linewidth(0.8)
    ax.tick_params(colors=_TICK, labelsize=9)
    ax.xaxis.label.set_color(_TICK)
    ax.yaxis.label.set_color(_TICK)
    ax.title.set_color(_TXT)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DOMAIN SHAPES — 5 Poisson geometries with reference solutions
# ══════════════════════════════════════════════════════════════════════════════

def _fdm_ref(mask, h):
    """Solve -Δu = 1 on interior of mask (Dirichlet u=0) using sparse FDM."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    ny, nx = mask.shape
    idx    = np.full((ny, nx), -1, dtype=int)
    n_int  = int(mask.sum())
    idx[mask] = np.arange(n_int)

    rows, cols, vals = [], [], []
    rhs = np.ones(n_int)

    for iy in range(ny):
        for ix in range(nx):
            if not mask[iy, ix]:
                continue
            r = idx[iy, ix]
            rows.append(r); cols.append(r); vals.append(4.0 / h**2)
            for diy, dix in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny2, nx2 = iy+diy, ix+dix
                if 0 <= ny2 < ny and 0 <= nx2 < nx and mask[ny2, nx2]:
                    rows.append(r); cols.append(idx[ny2,nx2]); vals.append(-1.0/h**2)

    A  = sp.csr_matrix((vals, (rows, cols)), shape=(n_int, n_int))
    u0 = spla.spsolve(A, rhs)

    U = np.full((ny, nx), np.nan)
    U[mask] = u0
    return U


def plot_domain_shapes():
    domains = [
        ("Square [0,1]²",   "u* = x²(x−1)²y²(y−1)²"),
        ("Circle x²+y²≤1",  "u* = (1−r²)/4"),
        ("Annulus\n0.25≤r≤1","u* = (r²−r_i²)/4"),
        ("L-Shape",          "−∇²u = 1, u=0 (FDM)"),
        ("Flower\nr≤1+0.3cos5θ","−∇²u = 1, u=0 (FDM)"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), facecolor=_FIG_BG)
    fig.suptitle("Level 7B — Poisson Domains: Five Different Geometries",
                 fontsize=13, color=_TXT, fontweight="bold", y=1.02)

    N = 120

    # 1. Square
    ax = axes[0]
    x  = np.linspace(0, 1, N)
    y  = np.linspace(0, 1, N)
    XX, YY = np.meshgrid(x, y)
    U  = XX**2 * (XX-1)**2 * YY**2 * (YY-1)**2
    im = ax.contourf(XX, YY, U, levels=30, cmap="YlGnBu")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)

    # 2. Circle
    ax = axes[1]
    x  = np.linspace(-1, 1, N)
    y  = np.linspace(-1, 1, N)
    XX, YY = np.meshgrid(x, y)
    R2 = XX**2 + YY**2
    U  = np.where(R2 <= 1, (1-R2)/4, np.nan)
    im = ax.contourf(XX, YY, U, levels=30, cmap="YlGnBu")
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k-", lw=1.5)
    ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.1,1.1)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)

    # 3. Annulus
    ax = axes[2]
    x  = np.linspace(-1.1, 1.1, N)
    y  = np.linspace(-1.1, 1.1, N)
    XX, YY = np.meshgrid(x, y)
    R2 = XX**2 + YY**2
    R  = np.sqrt(R2)
    ri = 0.25
    mask = (R >= ri) & (R <= 1)
    U    = np.where(mask, (R2 - ri**2)/4, np.nan)
    im   = ax.contourf(XX, YY, U, levels=30, cmap="YlGnBu")
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k-", lw=1.5)
    ax.plot(ri*np.cos(theta), ri*np.sin(theta), "k-", lw=1.5)
    ax.set_xlim(-1.15,1.15); ax.set_ylim(-1.15,1.15)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)

    # 4. L-Shape (FDM)
    ax = axes[3]
    h  = 1.0 / (N-1)
    x  = np.linspace(0, 1, N)
    y  = np.linspace(0, 1, N)
    XX, YY = np.meshgrid(x, y)
    mask_l = ~((XX > 0.5) & (YY < 0.5))   # L-shape: full square minus bottom-right
    try:
        U = _fdm_ref(mask_l, h)
        im = ax.contourf(XX, YY, U, levels=30, cmap="YlGnBu")
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    except Exception:
        ax.contourf(XX, YY, np.where(mask_l, 0.5, np.nan), levels=5, cmap="YlGnBu")
    # Correct L-shape outline: [0,1]² minus lower-right [0.5,1]×[0,0.5]
    outline_x = [0, 0.5, 0.5, 1,   1, 0, 0]
    outline_y = [0, 0,   0.5, 0.5, 1, 1, 0]
    ax.plot(outline_x, outline_y, "k-", lw=1.5)
    ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.05,1.05)
    ax.set_aspect("equal")

    # 5. Flower
    ax = axes[4]
    h  = 2.2 / (N-1)
    gx = np.linspace(-1.35, 1.35, N)
    gy = np.linspace(-1.35, 1.35, N)
    XX, YY = np.meshgrid(gx, gy)
    R_grid   = np.sqrt(XX**2 + YY**2)
    TH_grid  = np.arctan2(YY, XX)
    R_bnd    = 1.0 + 0.3 * np.cos(5*TH_grid)
    mask_fl  = R_grid <= R_bnd
    try:
        U = _fdm_ref(mask_fl, h)
        im = ax.contourf(XX, YY, U, levels=30, cmap="YlGnBu")
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    except Exception:
        ax.contourf(XX, YY, np.where(mask_fl, 0.5, np.nan), levels=5, cmap="YlGnBu")
    theta_f = np.linspace(0, 2*np.pi, 1000)
    r_f     = 1.0 + 0.3 * np.cos(5*theta_f)
    ax.plot(r_f*np.cos(theta_f), r_f*np.sin(theta_f), "k-", lw=1.5)
    ax.set_xlim(-1.45,1.45); ax.set_ylim(-1.45,1.45)
    ax.set_aspect("equal")

    for i, (ax, (title, formula)) in enumerate(zip(axes, domains)):
        _style(ax)
        ax.set_title(f"{title}\n{formula}", fontsize=9.5, color=_TXT, pad=4)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    out = OUT_DIR / "pres_domain_shapes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close()
    print(f"  [OK] {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LEVEL 1 — Architecture comparison: L2 and MAE bars
# ══════════════════════════════════════════════════════════════════════════════

def plot_l1_arch_comparison():
    opts   = ["bayesian", "nsga2", "nsga3"]
    labels = ["Bayesian\n5×151 relu", "NSGA-II\n3×153 tanh", "NSGA-III\n3×75 tanh"]
    l2s    = [0.076, 0.252, 0.513]
    maes   = [39.1,  132.6, 270.3]
    params = [92564, 47890, 11776]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=_FIG_BG)
    fig.suptitle("Level 1 — NAS Architecture Search Results (20 000 Adam epochs)",
                 fontsize=12, color=_TXT, fontweight="bold")

    x = np.arange(3)
    w = 0.5
    colors = [C_BAY, C_N2, C_N3]

    # L2 bar
    bars = ax1.bar(x, l2s, width=w, color=colors, edgecolor="white", linewidth=1.2)
    ax1.axhline(0.10, color=C_MEAS, ls="--", lw=1.5, label="Target L2 = 0.10")
    ax1.set_title("Relative L2 Error", fontsize=11, color=_TXT)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("L2 Relative Error", fontsize=10)
    ax1.legend(fontsize=9)
    for bar, val in zip(bars, l2s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                 color=_TXT, fontweight="bold")
    for bar, npar in zip(bars, params):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                 f"{npar//1000}k\nparams", ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold")
    _style(ax1)

    # MAE bar
    bars2 = ax2.bar(x, maes, width=w, color=colors, edgecolor="white", linewidth=1.2)
    ax2.axhline(50, color=C_MEAS, ls="--", lw=1.5, label="Target MAE = 50°C")
    ax2.set_title("Mean Absolute Error (°C)", fontsize=11, color=_TXT)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("MAE (°C)", fontsize=10)
    ax2.legend(fontsize=9)
    for bar, val in zip(bars2, maes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{val:.1f}°C", ha="center", va="bottom", fontsize=10,
                 color=_TXT, fontweight="bold")
    _style(ax2)

    # Paper FEM reference line
    ax2.axhline(0, color=C_FEM, ls=":", lw=1.2, alpha=0.5)

    plt.tight_layout()
    out = OUT_DIR / "pres_l1_arch_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close()
    print(f"  [OK] {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LEVEL 2 — Skip factor vs MAE (500 epochs + 2000 epochs)
# ══════════════════════════════════════════════════════════════════════════════

def plot_l2_skip_comparison():
    skip_path = ROOT / "level2_timestepper/results/skip_table.json"
    sk5k_path = ROOT / "level2_timestepper/results/skip_table_5k.json"
    if not skip_path.exists():
        print("  [SKIP] skip_table.json not found"); return

    with open(skip_path) as f:  l2 = json.load(f)
    sk5k = json.load(open(sk5k_path)) if sk5k_path.exists() else {}

    opts = ["bayesian", "nsga2", "nsga3"]
    CONV = 1.5

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_FIG_BG)
    fig.suptitle("Level 2 — Temporal Skip Operator: MAE vs Skip Factor",
                 fontsize=12, color=_TXT, fontweight="bold")

    # Left: MAE vs skip (500 epochs, all 3 optimizers)
    ax = axes[0]
    ax.set_title("500 Adam Epochs — MAE vs Skip Factor", fontsize=11, color=_TXT)
    markers = {"bayesian": "o", "nsga2": "s", "nsga3": "^"}
    ref_maes = {}
    for opt in opts:
        rows = {r["skip"]: r for r in l2.get(opt, [])}
        ref_maes[opt] = rows.get(1, {}).get("mae_C", 999)
        skips = sorted(rows.keys())
        xs = skips
        ys = [rows[s]["mae_C"] for s in skips]
        ax.plot(xs, ys, color=OPT_COLORS[opt], marker=markers[opt],
                lw=2, ms=8, label=OPT_SHORT[opt])
        for s, y in zip(xs, ys):
            ref = ref_maes[opt]
            r   = y / (ref + 1e-9)
            c   = "✓" if r < CONV else "✗"
            ax.annotate(f"{c} {y:.0f}", (s, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7.5, color=OPT_COLORS[opt])

    # Draw threshold lines (individual per optimizer)
    for opt in opts:
        thr = ref_maes[opt] * CONV
        ax.axhline(thr, color=OPT_COLORS[opt], ls=":", lw=1, alpha=0.5)

    ax.set_xlabel("Skip Factor k", fontsize=10)
    ax.set_ylabel("MAE (°C)", fontsize=10)
    ax.set_xticks([1, 2, 4, 6])
    ax.set_xticklabels(["skip=1\n(21 FEM)", "skip=2\n(11 FEM)", "skip=4\n(6 FEM)", "skip=6\n(4 FEM)"])
    ax.legend(fontsize=10)
    ax.set_ylim(0, 500)
    ax.grid(True, alpha=0.3)
    _style(ax)

    # Right: skip=2 MAE comparison: 500ep vs 2000ep
    ax2 = axes[1]
    ax2.set_title("Skip=2: 500 vs 2000 Adam Epochs", fontsize=11, color=_TXT)

    OPTS = ["nsga2", "nsga3"]
    x    = np.arange(len(OPTS))
    w    = 0.35

    bars_500  = []
    bars_2k   = []
    for opt in OPTS:
        rows_500 = {r["skip"]: r for r in l2.get(opt, [])}
        mae_500  = rows_500.get(2, {}).get("mae_C", 0)
        rows_2k  = {r["skip"]: r for r in sk5k.get(opt, [])}
        mae_2k   = rows_2k.get(2, {}).get("mae_C", 0)
        bars_500.append(mae_500)
        bars_2k.append(mae_2k)

    bp1 = ax2.bar(x - w/2, bars_500, width=w, color=[OPT_COLORS[o] for o in OPTS],
                  alpha=0.5, edgecolor="white", label="500 epochs")
    bp2 = ax2.bar(x + w/2, bars_2k,  width=w, color=[OPT_COLORS[o] for o in OPTS],
                  alpha=1.0, edgecolor="white", label="2000 epochs")

    # Threshold lines
    for i, opt in enumerate(OPTS):
        rows = {r["skip"]: r for r in l2.get(opt, [])}
        thr  = rows.get(1, {}).get("mae_C", 999) * CONV
        ax2.hlines(thr, i - 0.5, i + 0.5, colors=OPT_COLORS[opt], ls="--", lw=1.5, alpha=0.7)

    for bars in [bp1, bp2]:
        for bar in bars:
            v = bar.get_height()
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                         f"{v:.0f}°C", ha="center", va="bottom", fontsize=9,
                         color=_TXT, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([OPT_SHORT[o] for o in OPTS], fontsize=10)
    ax2.set_ylabel("MAE at skip=2 (°C)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.96, "Dashed = 1.5× threshold per optimizer",
             transform=ax2.transAxes, fontsize=8, color=C_GRAY, va="top")
    _style(ax2)

    plt.tight_layout()
    out = OUT_DIR / "pres_l2_skip_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close()
    print(f"  [OK] {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LEVEL 4 — CMM distortion grouped bar chart vs Paper baseline
# ══════════════════════════════════════════════════════════════════════════════

def plot_l4_distortion():
    p = ROOT / "level4_distortion/results/all_results_summary.json"
    if not p.exists():
        print("  [SKIP] level4 all_results_summary.json not found"); return

    with open(p) as f:
        d = json.load(f)

    paper_meas = d["paper_reference"]["measured_mean_abs_mm"]   # 0.781
    paper_fem  = d["paper_reference"]["fem_mean_abs_mm"]        # 0.748
    bay_mean   = d["level4_distortion"]["bayesian"]["mean_delta_mm"]
    n2_mean    = d["level4_distortion"]["nsga2"]["mean_delta_mm"]
    n3_mean    = d["level4_distortion"]["nsga3"]["mean_delta_mm"]

    names  = ["Paper\nMeasured", "Paper\nFEM", "Bayesian\nPINN", "NSGA-II\nPINN", "NSGA-III\nPINN"]
    values = [paper_meas, paper_fem, bay_mean, n2_mean, n3_mean]
    colors = [C_MEAS, C_FEM, C_BAY, C_N2, C_N3]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_FIG_BG)
    fig.suptitle("Level 4 — CMM Distortion: PINN vs Paper Baseline",
                 fontsize=12, color=_TXT, fontweight="bold")

    bars = ax.bar(np.arange(5), values, color=colors, edgecolor="white",
                  linewidth=1.2, width=0.6)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.2f} mm", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=_TXT)

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Mean |δ| (mm)", fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.axhline(paper_meas, color=C_MEAS, ls="--", lw=1.5, alpha=0.5,
               label=f"Measured baseline = {paper_meas:.2f} mm")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate paper vs pinn gap
    ax.annotate("", xy=(2, bay_mean), xytext=(1, paper_fem),
                arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.5))
    ax.text(1.5, (bay_mean + paper_fem)/2, "PINN\nvs FEM", ha="center",
            fontsize=8, color=C_GRAY)

    _style(ax)
    plt.tight_layout()
    out = OUT_DIR / "pres_l4_distortion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close()
    print(f"  [OK] {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  LEVEL 5 — Before/After L2 improvement bars
# ══════════════════════════════════════════════════════════════════════════════

def plot_l5_improvement():
    p = ROOT / "level5_refinement/results/level5_summary.json"
    if not p.exists():
        print("  [SKIP] level5_summary.json not found"); return

    with open(p) as f:
        d = json.load(f)

    opts   = ["bayesian", "nsga2", "nsga3"]
    before = []
    after  = []
    for opt in opts:
        before.append(d[opt]["level1_adam_only"]["L2_rel"])
        after.append(d[opt]["level5_adam_lbfgs"]["L2_rel"])

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_FIG_BG)
    fig.suptitle("Level 5 — Extended Training: L2 Before vs After",
                 fontsize=12, color=_TXT, fontweight="bold")

    x  = np.arange(3)
    w  = 0.35
    b1 = ax.bar(x - w/2, before, width=w, color=[C_BAY, C_N2, C_N3],
                alpha=0.45, edgecolor="white", label="L1 Adam (20k ep)")
    b2 = ax.bar(x + w/2, after,  width=w, color=[C_BAY, C_N2, C_N3],
                alpha=1.0, edgecolor="white", label="L5 Extended Adam")

    ax.axhline(0.10, color=C_MEAS, ls="--", lw=1.5, label="Target L2 = 0.10")

    for bar, v in zip(b1, before):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9.5,
                color=_TXT, fontweight="bold")
    for bar, v, bf in zip(b2, after, before):
        impr = bf / (v + 1e-9)
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                f"{v:.3f}\n({impr:.1f}×↓)", ha="center", va="bottom",
                fontsize=9, color=_TXT, fontweight="bold")

    lbls = [OPT_LABELS[o].replace("\n", "\n") for o in opts]
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, fontsize=9)
    ax.set_ylabel("L2 Relative Error", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 0.6)
    ax.grid(True, alpha=0.3, axis="y")
    _style(ax)

    plt.tight_layout()
    out = OUT_DIR / "pres_l5_improvement.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close()
    print(f"  [OK] {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  LEVEL 7B — Multi-domain L2 grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_l7b_results():
    p = ROOT / "level7_multiDomain/results/level7b_summary.json"
    if not p.exists():
        print("  [SKIP] level7b_summary.json not found"); return

    with open(p) as f:
        d = json.load(f)

    domains = ["square", "circle", "lshape", "flower"]
    opts    = ["bayesian", "nsga2", "nsga3"]
    dom_labels = {"square":"Square", "circle":"Circle", "lshape":"L-Shape", "flower":"Flower"}

    # Gather L2 values
    data = {}
    for dom in domains:
        data[dom] = {}
        for opt in opts:
            key = f"{dom}/{opt}"
            if key in d:
                data[dom][opt] = d[key]["final_l2"]
            else:
                data[dom][opt] = None

    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=_FIG_BG)
    fig.suptitle("Level 7B — Multi-Domain Poisson: NAS Results per Geometry",
                 fontsize=12, color=_TXT, fontweight="bold")

    n_dom = len(domains)
    n_opt = len(opts)
    w     = 0.22
    x     = np.arange(n_dom)

    for j, opt in enumerate(opts):
        vals = []
        for dom in domains:
            v = data[dom].get(opt)
            vals.append(v if v is not None else 0)

        bars = ax.bar(x + (j - 1) * w, vals, width=w, color=OPT_COLORS[opt],
                      edgecolor="white", linewidth=1, label=OPT_SHORT[opt])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.003,
                        f"{v:.4f}" if v < 0.01 else f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.5,
                        color=_TXT, fontweight="bold", rotation=55)

    ax.set_xticks(x)
    ax.set_xticklabels([dom_labels[d] for d in domains], fontsize=11)
    ax.set_ylabel("L2 Relative Error", fontsize=11)
    ax.axhline(0.05, color=C_MEAS, ls="--", lw=1.5, label="Target L2 = 0.05", alpha=0.7)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 0.22)
    ax.grid(True, alpha=0.3, axis="y")
    _style(ax)

    plt.tight_layout()
    out = OUT_DIR / "pres_l7b_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close()
    print(f"  [OK] {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CROSS-LEVEL SUMMARY — L2 progression across all levels
# ══════════════════════════════════════════════════════════════════════════════

def plot_cross_level_progression():
    """Stacked line plot showing L2 improvement from L1 through L5 for each optimizer."""
    # L1 Adam only
    l1 = {"bayesian": 0.0764, "nsga2": 0.2517, "nsga3": 0.5129}
    # L5 Extended Adam
    l5 = {"bayesian": 0.0296, "nsga2": 0.0554, "nsga3": 0.2586}
    # L6 after fine-tuning
    l6 = {"bayesian": 0.0311, "nsga2": 0.0577, "nsga3": 0.2672}

    levels = [1, 5, 6]
    labels = ["L1\nAdam 20k", "L5\nExt. Adam", "L6\nFine-tune"]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_FIG_BG)
    fig.suptitle("Cross-Level L2 Improvement — All Optimizers",
                 fontsize=12, color=_TXT, fontweight="bold")

    data_by_opt = {
        "bayesian": [l1["bayesian"], l5["bayesian"], l6["bayesian"]],
        "nsga2":    [l1["nsga2"],    l5["nsga2"],    l6["nsga2"]],
        "nsga3":    [l1["nsga3"],    l5["nsga3"],    l6["nsga3"]],
    }

    for opt, vals in data_by_opt.items():
        ax.plot(labels, vals, color=OPT_COLORS[opt], marker="o", lw=2.5,
                ms=9, label=OPT_SHORT[opt])
        for lbl, v in zip(labels, vals):
            ax.annotate(f"{v:.3f}", (lbl, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9,
                        color=OPT_COLORS[opt], fontweight="bold")

    ax.axhline(0.10, color=C_MEAS, ls="--", lw=1.5, label="Target L2 = 0.10")
    ax.set_ylabel("L2 Relative Error", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 0.60)
    ax.grid(True, alpha=0.3)
    _style(ax)

    plt.tight_layout()
    out = OUT_DIR / "pres_cross_level_l2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close()
    print(f"  [OK] {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  NAS SEARCH SPACE overview figure
# ══════════════════════════════════════════════════════════════════════════════

def plot_nas_search_space():
    """Show the NAS search space as a visual grid with found architectures marked."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=_FIG_BG)
    fig.suptitle("NAS Search Space — Three Optimizers, Same Space, Different Results",
                 fontsize=12, color=_TXT, fontweight="bold")

    layer_opts  = [2, 3, 4, 5, 6]
    neuron_opts = [32, 64, 96, 128, 160, 192, 256]
    act_opts    = ["tanh", "relu", "swish"]

    found = {
        "bayesian": {"layers": 5, "neurons": 151, "act": "relu"},
        "nsga2":    {"layers": 3, "neurons": 153, "act": "tanh"},
        "nsga3":    {"layers": 3, "neurons": 75,  "act": "tanh"},
    }
    found_act_map = {"relu": 1, "tanh": 0, "swish": 2}

    for ax, (opt, cfg) in zip(axes, found.items()):
        # draw scatter in (neurons, layers) space
        for nl in layer_opts:
            for nn in [32, 64, 96, 128, 160, 192, 256]:
                ax.scatter(nn, nl, s=25, color=_SPINE, alpha=0.5, zorder=1)

        # Mark found architecture
        ax.scatter(cfg["neurons"], cfg["layers"], s=350,
                   color=OPT_COLORS[opt], zorder=5,
                   edgecolors="white", linewidths=2)
        ax.annotate(f"  {cfg['layers']}×{cfg['neurons']}\n  {cfg['act']}",
                    (cfg["neurons"], cfg["layers"]),
                    fontsize=10, color=OPT_COLORS[opt], fontweight="bold")

        ax.set_title(f"{OPT_SHORT[opt]}\nFound: {cfg['layers']}×{cfg['neurons']} {cfg['act']}",
                     fontsize=11, color=_TXT)
        ax.set_xlabel("Neurons per layer", fontsize=10)
        ax.set_ylabel("Num. layers", fontsize=10)
        ax.set_yticks(layer_opts)
        ax.set_ylim(1.5, 6.5)
        ax.grid(True, alpha=0.3)
        _style(ax)

    plt.tight_layout()
    out = OUT_DIR / "pres_nas_search_space.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close()
    print(f"  [OK] {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  LEVEL 3 — Skip rate summary bar chart (80% / 95%)
# ══════════════════════════════════════════════════════════════════════════════

def plot_l3_skip_rates():
    """Bar chart showing FEM vs PINN step counts for both threshold configs."""
    configs = {
        "thr=0.1\nmax_skip=4":  {"fem": 4,  "pinn": 16, "total": 20},
        "thr=0.1\nmax_skip=20": {"fem": 1,  "pinn": 19, "total": 20},
    }

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_FIG_BG)
    fig.suptitle("Level 3 — Hybrid FEM+PINN: Adaptive Step Distribution (Bayesian)",
                 fontsize=12, color=_TXT, fontweight="bold")

    x = np.arange(len(configs))
    labels = list(configs.keys())

    fem_counts  = [v["fem"]  for v in configs.values()]
    pinn_counts = [v["pinn"] for v in configs.values()]

    ax.bar(x, fem_counts,  label="FEM steps",  color=C_FEM,  alpha=0.85, width=0.45, edgecolor="white")
    ax.bar(x, pinn_counts, label="PINN steps", color=C_BAY,  alpha=0.85, width=0.45,
           bottom=fem_counts, edgecolor="white")

    for i, (fm, pn) in enumerate(zip(fem_counts, pinn_counts)):
        skip_pct = pn / (fm + pn) * 100
        ax.text(i, fm + pn + 0.3, f"{skip_pct:.0f}% PINN\n({pn}/{fm+pn} steps)",
                ha="center", va="bottom", fontsize=11, fontweight="bold", color=_TXT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Number of time steps", fontsize=11)
    ax.set_ylim(0, 24)
    ax.axhline(20, color=C_GRAY, ls=":", lw=1.2, alpha=0.5, label="Total steps = 20")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    _style(ax)

    plt.tight_layout()
    out = OUT_DIR / "pres_l3_skip_rates.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close()
    print(f"  [OK] {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 10.  DOMAIN SHAPES + PINN PREDICTION + ERROR  (3-row comparison)
# ══════════════════════════════════════════════════════════════════════════════

def plot_domain_pred_error():
    """
    Combine the 4 existing per-domain plots from plot_level7b.py into a
    single 2×2 grid image.  Uses the already-generated high-quality PNGs:
      level7_multiDomain/results/plots/l7b_{square,circle,lshape,flower}.png
    """
    from matplotlib.image import imread

    L7B_PLOTS = ROOT / "level7_multiDomain/results/plots"
    domains = [
        ("square", "Square"),
        ("circle", "Circle"),
        ("lshape", "L-Shape"),
        ("flower", "Flower"),
    ]

    # Check all source files exist
    missing = [d for d, _ in domains if not (L7B_PLOTS / f"l7b_{d}.png").exists()]
    if missing:
        print(f"  [SKIP] missing source plots: {missing}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), facecolor=_FIG_BG)
    fig.suptitle(
        "Level 7B — Per-Domain: Reference  |  PINN Predictions  |  Error Maps  (Bayesian NAS)",
        fontsize=14, color=_TXT, fontweight="bold", y=1.01
    )

    for ax, (dom, label) in zip(axes.ravel(), domains):
        src = L7B_PLOTS / f"l7b_{dom}.png"
        im  = imread(str(src))
        ax.imshow(im)
        ax.set_title(label, fontsize=13, color=_TXT, fontweight="bold", pad=6)
        ax.axis("off")

    plt.tight_layout(pad=1.0)
    out = OUT_DIR / "pres_domain_pred_error.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close()
    print(f"  [OK] {out.name}")


def plot_l8_3d_results():
    """L8: 3D MCO-PINN results summary figure."""
    json_path = ROOT / "results" / "results_3d.json"
    if not json_path.exists():
        print("  [skip] results_3d.json not found"); return

    with open(json_path) as f:
        data = json.load(f)

    domains = list(data.keys())
    archs   = ["bayesian", "nsga2", "nsga3"]
    skips   = [2, 4]
    arch_short = {"bayesian": "Bayesian", "nsga2": "NSGA-II", "nsga3": "NSGA-III"}
    dom_label  = {"rectangular": "Rectangular\n(1.3×0.6×0.4 m)",
                  "cylinder":    "Cylinder\n(R=0.25 m, H=0.6 m)",
                  "stacked":     "Stacked Cubes\n(2×0.5 m)"}
    dom_color  = {"rectangular": C_BAY, "cylinder": C_N3, "stacked": C_N2}
    skip_alpha = {2: 0.90, 4: 0.50}
    skip_hatch = {2: "",   4: "///"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    fig.patch.set_facecolor(_FIG_BG)
    fig.suptitle("Level 8 — 3D MCO-PINN Skip Operator\n"
                 "3 Domains · 3 Architectures · skip=2/4 | A356 Aluminum Quenching",
                 fontsize=13, fontweight="bold", color=_TXT)

    x = np.arange(len(archs))
    w = 0.32

    for col, dom in enumerate(domains):
        ax = axes[col]
        ax.set_facecolor(_AX_BG)
        for sp in ax.spines.values():
            sp.set_color(_SPINE); sp.set_linewidth(0.8)

        for si, skip in enumerate(skips):
            maes = [data[dom][a][str(skip)]["mae_C"] for a in archs]
            bars = ax.bar(x + (si - 0.5) * w, maes, width=w - 0.04,
                          color=dom_color[dom], alpha=skip_alpha[skip],
                          hatch=skip_hatch[skip], edgecolor="white",
                          label=f"skip={skip}  ({50 if skip==2 else 71:.0f}% FEM↓)")
            for b, v in zip(bars, maes):
                ax.text(b.get_x() + b.get_width()/2,
                        b.get_height() + 0.3,
                        f"{v:.1f}", ha="center", fontsize=8.5,
                        fontweight="bold", color=_TXT)

        ax.axhline(10, color=C_MEAS, lw=1.2, ls="--", alpha=0.6,
                   label="10°C threshold")
        ax.set_title(dom_label[dom], fontsize=10.5, color=_TXT, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([arch_short[a] for a in archs], fontsize=9)
        ax.tick_params(colors=_TICK)
        ax.set_ylim(0, 28)
        if col == 0:
            ax.set_ylabel("Mean MAE [°C]", fontsize=10, color=_TICK)
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")
        ax.legend(fontsize=8, loc="upper right")

    # Bottom caption
    fig.text(0.5, -0.02,
             "skip=2: 11 of 21 FEM steps computed (10 PINN predictions, 52% savings)  |  "
             "skip=4: 6 FEM steps (71% savings)",
             ha="center", fontsize=9, color=_TXT, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "pres_l8_3d_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close(fig)
    print(f"  → {out.name}")


def plot_l8_fem_skip_summary():
    """L8: 2D MCO-PINN FEM step skip summary — presentation quality."""
    mae_data = {
        "bayesian": {1: 1.0, 2: 2.0, 4: 5.2, 6: 7.3},
        "nsga2":    {1: 1.5, 2: 2.9, 4: 6.5, 6: 5.6},
        "nsga3":    {1: 8.9, 2: 3.0, 4: 4.3, 6: 5.8},
    }
    l2_ref = {
        "bayesian": {1: 43.6, 2: 33.2, 4: 57.5, 6: 93.6},
        "nsga2":    {1: 38.1, 2: 56.0, 4: 154.8, 6: 354.6},
        "nsga3":    {1: 44.4, 2: 75.6, 4: 202.1, 6: 428.6},
    }
    fem_count = {1: 21, 2: 11, 4: 6, 6: 4}

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(_FIG_BG)
    fig.suptitle("Level 8: MCO-PINN Skip Operator — FEM Step Savings\n"
                 "2D Analysis | All Architectures | A356 Aluminum",
                 fontsize=13, fontweight="bold", color=_TXT)

    # Sol: MAE vs skip
    for arch, color in OPT_COLORS.items():
        skips = [1, 2, 4, 6]
        maes  = [mae_data[arch][s] for s in skips]
        ax_l.plot(skips, maes, "o-", color=color, lw=2.2,
                  label=OPT_SHORT[arch], ms=8)
        for s, m in zip(skips, maes):
            ax_l.annotate(f"{m:.1f}°C",
                          xy=(s, m), xytext=(4, 6),
                          textcoords="offset points", fontsize=8.5,
                          color=color, fontweight="bold")

    ax_l.axhline(5.0, color=C_MEAS, lw=1.5, ls="--", alpha=0.7,
                 label="Acceptance threshold: 5°C")
    ax_l.set_xlabel("Skip value", fontsize=10, color=_TICK)
    ax_l.set_ylabel("Mean MAE [°C]", fontsize=10, color=_TICK)
    ax_l.set_xticks([1, 2, 4, 6])
    ax_l.set_xticklabels(["skip=1\n(21 FEM)", "skip=2\n(11 FEM)",
                           "skip=4\n(6 FEM)", "skip=6\n(4 FEM)"])
    ax_l.set_ylim(0, 12)
    ax_l.legend(fontsize=9)
    ax_l.grid(True, alpha=0.3, linestyle=":")
    ax_l.set_facecolor(_AX_BG)
    for sp in ax_l.spines.values():
        sp.set_color(_SPINE)
    ax_l.set_title("MCO-PINN MAE vs Skip Value", fontsize=11, color=_TXT)

    # Right: L2 baseline vs MCO-PINN (bayesian)
    arch = "bayesian"
    skips = [1, 2, 4, 6]
    l2_vals  = [l2_ref[arch][s] for s in skips]
    mco_vals = [mae_data[arch][s] for s in skips]
    xpos = np.arange(len(skips))
    w = 0.35

    b1 = ax_r.bar(xpos - w/2, l2_vals, width=w, color=C_GRAY, alpha=0.7,
                  label="L2 Baseline (fixed weights)")
    b2 = ax_r.bar(xpos + w/2, mco_vals, width=w, color=C_BAY, alpha=0.9,
                  label="MCO-PINN (Shen 2023)")

    for b, v in zip(b1, l2_vals):
        ax_r.text(b.get_x()+b.get_width()/2, v+1, f"{v:.0f}",
                  ha="center", fontsize=8, color=C_GRAY, fontweight="bold")
    for b, v in zip(b2, mco_vals):
        ax_r.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}",
                  ha="center", fontsize=8, color=C_BAY, fontweight="bold")

    ax_r.set_xticks(xpos)
    ax_r.set_xticklabels(["skip=1\n(21 FEM)", "skip=2\n(11 FEM)",
                           "skip=4\n(6 FEM)", "skip=6\n(4 FEM)"])
    ax_r.set_ylabel("Mean MAE [°C]", fontsize=10, color=_TICK)
    ax_r.set_ylim(0, 110)
    ax_r.legend(fontsize=9)
    ax_r.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax_r.set_facecolor(_AX_BG)
    for sp in ax_r.spines.values():
        sp.set_color(_SPINE)
    ax_r.set_title("L2 Baseline vs MCO-PINN (Bayesian)", fontsize=11, color=_TXT)

    # Add improvement arrows
    for xi, (l2v, mcov) in enumerate(zip(l2_vals, mco_vals)):
        pct = (1 - mcov/l2v) * 100
        ax_r.annotate(f"▼{pct:.0f}%",
                      xy=(xi + w/2, mcov + 3),
                      ha="center", fontsize=9, color=C_N3,
                      fontweight="bold")

    plt.tight_layout()
    out = OUT_DIR / "pres_l8_fem_skip_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close(fig)
    print(f"  → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating presentation plots …")
    print(f"Output: {OUT_DIR}\n")

    plot_domain_shapes()
    plot_domain_pred_error()
    plot_l1_arch_comparison()
    plot_l2_skip_comparison()
    plot_l4_distortion()
    plot_l5_improvement()
    plot_l7b_results()
    plot_cross_level_progression()
    plot_nas_search_space()
    plot_l3_skip_rates()
    plot_l8_3d_results()
    plot_l8_fem_skip_summary()

    print(f"\nDone. All plots saved to {OUT_DIR}")
