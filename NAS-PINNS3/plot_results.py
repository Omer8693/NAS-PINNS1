"""
plot_results.py — Publication-quality result figures for NAS-PINN thesis
=========================================================================
  fig1 — fig1_thermal_fields.png : 3D temperature field per optimizer (Cylinder domain)
  fig2 — fig2_mae_per_arch.png   : MAE per architecture, 3D results
  fig3 — fig3_2d_per_arch.png    : 2D training results per architecture
  fig4 — fig4_summary_table.png  : Baseline / 2D / 3D comparison table
"""

import os, sys, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from level8_nas_mco_pinn.domains_3d import (
    Rectangular3D, Cylinder3D, StackedCubes3D, LShape3D,
    T_INIT, T_WATER,
)

# ── Paths ──────────────────────────────────────────────────────
RESULTS = os.path.join(os.path.dirname(__file__), "level8_nas_mco_pinn", "results")
V2_DIR  = os.path.join(RESULTS, "v2")
os.makedirs(RESULTS, exist_ok=True)

# ── Professional color palette ─────────────────────────────────
# Temperature colormap: full rainbow, vivid, no black or white
CMAP_T = "turbo"

# Architecture colors — vivid, high-contrast
C_BAY  = "#1976D2"   # vivid blue   — Bayesian
C_N2   = "#2E7D32"   # vivid green  — NSGA-II
C_N3   = "#D32F2F"   # vivid red    — NSGA-III
C_GRAY = "#546E7A"   # blue-gray    — baseline

# Skip colors
C_SK2  = "#1565C0"   # strong blue  — skip=2
C_SK4  = "#E65100"   # deep orange  — skip=4

ARCH_LABEL = {"bayesian": "Bayesian (TPE)", "nsga2": "NSGA-II", "nsga3": "NSGA-III"}
ARCH_COLOR = {"bayesian": C_BAY, "nsga2": C_N2, "nsga3": C_N3}
ARCHS      = ["bayesian", "nsga2", "nsga3"]

DOM_LABEL  = {
    "rectangular": "Rectangular Prism\n(1.3 × 0.6 × 0.4 m)",
    "cylinder":    "Cylinder\n(R = 0.25 m, H = 0.6 m)",
    "stacked":     "Stacked Cubes\n(2 × 0.5 m)",
    "lshape":      "L-Shape 3D\n(0.8 × 0.8 × 0.4 m)",
}
DOM_SHORT = {"rectangular": "Rectangular", "cylinder": "Cylinder",
             "stacked": "Stacked", "lshape": "L-Shape"}
DOMAINS   = ["rectangular", "cylinder", "stacked", "lshape"]

norm_T = Normalize(vmin=T_WATER, vmax=T_INIT)

# ── Global style ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8.5,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── Data ───────────────────────────────────────────────────────
with open(os.path.join(RESULTS, "results_3d.json")) as f:
    data_3d = json.load(f)

with open(os.path.join(RESULTS, "level8_skip_results.json")) as f:
    data_2d = json.load(f)

baseline = data_2d["level2_ref"]
mco_2d   = data_2d["level8_mco"]

_v2_json = os.path.join(V2_DIR, "results_3d_v2.json")
data_v2  = json.load(open(_v2_json)) if os.path.exists(_v2_json) else {}


# ──── 3D geometry drawing helpers ─────────────────────────────────────────────

def _draw_box3d(ax, T_fn, t, Lx, Ly, Lz, cmap_obj, norm, n=18, oz=0.):
    """Draw all 6 faces of a box from (0,0,oz) to (Lx,Ly,oz+Lz) colored by T."""
    kw = dict(shade=False, alpha=0.88, linewidth=0, antialiased=True)
    xi = np.linspace(0, Lx, n); yi = np.linspace(0, Ly, n)
    zi = np.linspace(oz, oz + Lz, n)
    XX, YY = np.meshgrid(xi, yi)
    for z0 in (oz, oz + Lz):
        ZZ = np.full_like(XX, z0)
        T  = T_fn(XX.ravel(), YY.ravel(), ZZ.ravel(), t).reshape(n, n)
        ax.plot_surface(XX, YY, ZZ, facecolors=cmap_obj(norm(T)), **kw)
    XX2, ZZ2 = np.meshgrid(xi, zi)
    for y0 in (0., Ly):
        YY2 = np.full_like(XX2, y0)
        T   = T_fn(XX2.ravel(), YY2.ravel(), ZZ2.ravel(), t).reshape(n, n)
        ax.plot_surface(XX2, YY2, ZZ2, facecolors=cmap_obj(norm(T)), **kw)
    YY3, ZZ3 = np.meshgrid(yi, zi)
    for x0 in (0., Lx):
        XX3 = np.full_like(YY3, x0)
        T   = T_fn(XX3.ravel(), YY3.ravel(), ZZ3.ravel(), t).reshape(n, n)
        ax.plot_surface(XX3, YY3, ZZ3, facecolors=cmap_obj(norm(T)), **kw)


def _draw_cyl3d(ax, dom, t, cmap_obj, norm, n_th=44, n_z=20, n_r=12):
    """Draw cylinder: lateral surface + top & bottom caps."""
    R, H = dom.R, dom.H
    kw = dict(shade=False, alpha=0.90, linewidth=0, antialiased=True)
    th = np.linspace(0, 2 * np.pi, n_th)
    z  = np.linspace(0, H, n_z)
    TH, ZZ = np.meshgrid(th, z)
    XX = R * np.cos(TH); YY = R * np.sin(TH)
    T  = dom.T_xyz(XX.ravel(), YY.ravel(), ZZ.ravel(), t).reshape(n_z, n_th)
    ax.plot_surface(XX, YY, ZZ, facecolors=cmap_obj(norm(T)), **kw)
    r_a = np.linspace(0, R, n_r)
    RR, TH2 = np.meshgrid(r_a, th)
    Xc = RR * np.cos(TH2); Yc = RR * np.sin(TH2)
    for z0 in (0., H):
        Zc = np.full_like(Xc, z0)
        Tc = dom.T_xyz(Xc.ravel(), Yc.ravel(), Zc.ravel(), t).reshape(n_th, n_r)
        ax.plot_surface(Xc, Yc, Zc, facecolors=cmap_obj(norm(Tc)), **kw)


def _wireframe_box3d(ax, x1, y1, z1, x0=0., y0=0., z0=0., **kw):
    """Draw 12 edges of a box."""
    for zi in (z0, z1):
        ax.plot([x0,x1],[y0,y0],[zi,zi], **kw)
        ax.plot([x1,x1],[y0,y1],[zi,zi], **kw)
        ax.plot([x1,x0],[y1,y1],[zi,zi], **kw)
        ax.plot([x0,x0],[y1,y0],[zi,zi], **kw)
    for xi, yi in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]:
        ax.plot([xi,xi],[yi,yi],[z0,z1], **kw)


def _wireframe_cyl3d(ax, R, H, n_th=60, **kw):
    """Draw cylinder outline."""
    th = np.linspace(0, 2 * np.pi, n_th)
    for z0 in (0., H):
        ax.plot(R * np.cos(th), R * np.sin(th), [z0] * n_th, **kw)
    for t0 in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
        ax.plot([R*np.cos(t0)]*2, [R*np.sin(t0)]*2, [0., H], **kw)


def _draw_lshape3d(ax, dom, t, cmap_obj, norm, n=16):
    """
    Draw L-shaped 3D domain: 8 exposed outer faces + inner corner faces.
    L-shape: (x <= cut_x) OR (y <= cut_y), extruded along z.
    Lx=0.8, Ly=0.8, Lz=0.4, cut_x=0.3, cut_y=0.3
    """
    kw = dict(shade=False, alpha=0.85, linewidth=0, antialiased=True)
    Lx, Ly, Lz = dom.Lx, dom.Ly, dom.Lz
    cx, cy = dom.cut_x, dom.cut_y   # 0.3, 0.3

    def surf(xx, yy, zz):
        T = dom.T(xx.ravel(), yy.ravel(), zz.ravel(), t)
        T = np.where(np.isnan(T), T_WATER, T)   # replace NaN with cold color
        return cmap_obj(norm(T.reshape(xx.shape)))

    xi_h = np.linspace(0, Lx, n)    # full x range
    xi_v = np.linspace(0, cx, n)    # vertical arm x range
    yi_h = np.linspace(0, cy, n)    # horizontal arm y range
    yi_v = np.linspace(0, Ly, n)    # full y range
    zi   = np.linspace(0, Lz, n)

    # ── Bottom face (z=0) and Top face (z=Lz) — L-shaped ──
    for z0 in (0., Lz):
        # Horizontal arm patch: x∈[0,Lx], y∈[0,cy]
        XX, YY = np.meshgrid(xi_h, yi_h)
        ZZ = np.full_like(XX, z0)
        ax.plot_surface(XX, YY, ZZ, facecolors=surf(XX, YY, ZZ), **kw)
        # Vertical arm extension: x∈[0,cx], y∈[cy,Ly]
        XX2, YY2 = np.meshgrid(xi_v, np.linspace(cy, Ly, n))
        ZZ2 = np.full_like(XX2, z0)
        ax.plot_surface(XX2, YY2, ZZ2, facecolors=surf(XX2, YY2, ZZ2), **kw)

    # ── Outer faces ──
    # x=0 (left wall): y∈[0,Ly]
    YY, ZZ = np.meshgrid(yi_v, zi)
    XX = np.zeros_like(YY)
    ax.plot_surface(XX, YY, ZZ, facecolors=surf(XX, YY, ZZ), **kw)
    # y=0 (front wall): x∈[0,Lx]
    XX2, ZZ2 = np.meshgrid(xi_h, zi)
    YY2 = np.zeros_like(XX2)
    ax.plot_surface(XX2, YY2, ZZ2, facecolors=surf(XX2, YY2, ZZ2), **kw)
    # x=Lx (right wall, only for y∈[0,cy])
    YY3, ZZ3 = np.meshgrid(yi_h, zi)
    XX3 = np.full_like(YY3, Lx)
    ax.plot_surface(XX3, YY3, ZZ3, facecolors=surf(XX3, YY3, ZZ3), **kw)
    # y=Ly (back wall, only for x∈[0,cx])
    XX4, ZZ4 = np.meshgrid(xi_v, zi)
    YY4 = np.full_like(XX4, Ly)
    ax.plot_surface(XX4, YY4, ZZ4, facecolors=surf(XX4, YY4, ZZ4), **kw)
    # ── Inner corner faces (concave surfaces) ──
    # x=cx (inner vertical face): y∈[cy,Ly]
    YY5, ZZ5 = np.meshgrid(np.linspace(cy, Ly, n), zi)
    XX5 = np.full_like(YY5, cx)
    ax.plot_surface(XX5, YY5, ZZ5, facecolors=surf(XX5, YY5, ZZ5), **kw)
    # y=cy (inner horizontal face): x∈[cx,Lx]
    XX6, ZZ6 = np.meshgrid(np.linspace(cx, Lx, n), zi)
    YY6 = np.full_like(XX6, cy)
    ax.plot_surface(XX6, YY6, ZZ6, facecolors=surf(XX6, YY6, ZZ6), **kw)


def _wireframe_lshape3d(ax, Lx, Ly, Lz, cx, cy, **kw):
    """Draw L-shape 3D wireframe edges."""
    # Bottom and top L-shape contours
    for z0 in (0., Lz):
        pts = [(0,0), (Lx,0), (Lx,cy), (cx,cy), (cx,Ly), (0,Ly), (0,0)]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, [z0]*len(xs), **kw)
    # Vertical edges
    corners = [(0,0), (Lx,0), (Lx,cy), (cx,cy), (cx,Ly), (0,Ly)]
    for xc, yc in corners:
        ax.plot([xc,xc], [yc,yc], [0.,Lz], **kw)


# ══════════════════════════════════════════════════════════════
# Figure 1: 3D Temperature Field — all 3 domains, multiple times
# ══════════════════════════════════════════════════════════════

def fig_thermal_fields():
    """
    4 rows (t=3,10,20,30s) × 3 domain cols + per-row colorbar.
    Shows ACTUAL 3D geometry shapes colored by FEM temperature.
    z-axis = real geometry height, color = temperature.
    """
    import matplotlib.gridspec as gridspec

    dom_dict = {
        "rectangular": Rectangular3D(),
        "cylinder":    Cylinder3D(),
        "stacked":     StackedCubes3D(),
        "lshape":      LShape3D(),
    }
    T_VALS  = [3, 10, 20, 30]
    n_rows  = len(T_VALS)
    n_doms  = len(dom_dict)
    cmap_obj = plt.cm.turbo

    fig = plt.figure(figsize=(18, 4.5 * n_rows))
    fig.suptitle(
        "FEM Temperature Field — Actual 3D Geometry  (color = T [°C], z = height [m])\n"
        "Rows: time steps  ·  Columns: domain geometry  ·  Colorbar per row",
        fontsize=11, fontweight="bold",
    )
    gs = gridspec.GridSpec(n_rows, n_doms + 1,
                           width_ratios=[1, 1, 1, 1, 0.06],
                           hspace=0.15, wspace=0.08, figure=fig)

    dom_names = list(dom_dict.keys())
    dom_list  = list(dom_dict.values())
    col_color = {"rectangular": C_BAY, "cylinder": C_N3,
                 "stacked": C_N2, "lshape": "#6A1B9A"}

    for ri, t_val in enumerate(T_VALS):
        # Row norm: center temperature = max, T_WATER = min
        dom_r = dom_dict["rectangular"]
        T_max = float(dom_r.T(
            np.array([dom_r.Lx / 2]), np.array([dom_r.Ly / 2]),
            np.array([dom_r.Lz / 2]), t_val)[0])
        row_norm = Normalize(vmin=T_WATER, vmax=max(T_max, T_WATER + 10))

        for ci, (dname, dom) in enumerate(dom_dict.items()):
            ax = fig.add_subplot(gs[ri, ci], projection="3d")

            if isinstance(dom, Cylinder3D):
                _draw_cyl3d(ax, dom, t_val, cmap_obj, row_norm)
                ax.set_xlim(-dom.R, dom.R)
                ax.set_ylim(-dom.R, dom.R)
                ax.set_zlim(0, dom.H)
                asp = [2*dom.R, 2*dom.R, dom.H]
            elif isinstance(dom, StackedCubes3D):
                L = dom.L_cube
                # Bottom cube: z = 0 → L
                _draw_box3d(ax, dom.T, t_val, L, L, L, cmap_obj, row_norm, n=18, oz=0.)
                # Top cube: z = L → 2L  (oz=L shifts all faces up)
                _draw_box3d(ax, dom.T, t_val, L, L, L, cmap_obj, row_norm, n=18, oz=L)
                # Joint plane at z=L (inner cross-section — makes stacking visible)
                n_j = 22
                xi_j = np.linspace(0, L, n_j); yi_j = np.linspace(0, L, n_j)
                XX_j, YY_j = np.meshgrid(xi_j, yi_j)
                T_j = dom.T(XX_j.ravel(), YY_j.ravel(),
                            np.full(n_j*n_j, L), t_val).reshape(n_j, n_j)
                ax.plot_surface(XX_j, YY_j, np.full_like(XX_j, L),
                                facecolors=cmap_obj(row_norm(T_j)),
                                shade=False, alpha=0.97, linewidth=0, antialiased=True)
                # Bold joint edge
                ax.plot([0,L,L,0,0],[0,0,L,L,0],[L]*5, 'k-', lw=2.0, alpha=0.8)
                ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, dom.Lz)
                asp = [L, L, dom.Lz]
            elif isinstance(dom, LShape3D):
                _draw_lshape3d(ax, dom, t_val, cmap_obj, row_norm)
                ax.set_xlim(0, dom.Lx); ax.set_ylim(0, dom.Ly); ax.set_zlim(0, dom.Lz)
                asp = [dom.Lx, dom.Ly, dom.Lz]
            else:  # Rectangular
                _draw_box3d(ax, dom.T, t_val,
                            dom.Lx, dom.Ly, dom.Lz,
                            cmap_obj, row_norm)
                ax.set_xlim(0, dom.Lx)
                ax.set_ylim(0, dom.Ly)
                ax.set_zlim(0, dom.Lz)
                asp = [dom.Lx, dom.Ly, dom.Lz]

            # Normalise aspect ratios
            asp_max = max(asp)
            ax.set_box_aspect([a / asp_max for a in asp])

            if ri == 0:
                ax.set_title(DOM_LABEL[dname], fontsize=9, fontweight="bold",
                             color=col_color[dname], pad=6)
            if ci == 0:
                ax.text2D(-0.14, 0.50, f"t = {t_val} s",
                          transform=ax.transAxes,
                          fontsize=9.5, fontweight="bold", color="#1F2937",
                          va="center", rotation=90)

            ax.set_xlabel("x [m]", fontsize=6, labelpad=0)
            ax.set_ylabel("y [m]", fontsize=6, labelpad=0)
            ax.set_zlabel("z [m]",  fontsize=6, labelpad=0)
            ax.tick_params(labelsize=5, pad=0)
            ax.view_init(elev=26, azim=-48)

        # Per-row colorbar
        cax = fig.add_subplot(gs[ri, n_doms])
        sm  = plt.cm.ScalarMappable(cmap="turbo", norm=row_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("T [°C]", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    fig.savefig(os.path.join(RESULTS, "fig1_thermal_fields.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig1_thermal_fields.png")


# ══════════════════════════════════════════════════════════════
# Figure 2: 3D MAE per Architecture
# ══════════════════════════════════════════════════════════════

def fig_mae_per_arch():
    """
    3 subplots — one per architecture.
    X-axis: 4 domains (v2 data), bars grouped by skip value.
    """
    skips      = [2, 4]
    skip_color = {2: C_SK2,  4: C_SK4}
    skip_alpha = {2: 0.88,   4: 0.60}
    skip_hatch = {2: "",     4: "///"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(
        "3D MCO-PINN Skip Operator — Mean Absolute Error by Architecture  (v2: 2000 epochs)",
        fontsize=11, fontweight="bold",
    )

    x = np.arange(len(DOMAINS))
    w = 0.30

    for col, arch in enumerate(ARCHS):
        ax = axes[col]
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.4, color="#D1D5DB")
        ax.set_axisbelow(True)
        c  = ARCH_COLOR[arch]
        ax.set_title(f"({chr(97+col)})  {ARCH_LABEL[arch]}",
                     fontsize=10, fontweight="bold", color=c)

        for si, skip in enumerate(skips):
            maes   = np.array([data_v2.get(d, {}).get(arch, {}).get(str(skip), {}).get("mae_C", np.nan)
                               for d in DOMAINS])
            offset = (si - 0.5) * w
            x_ok   = x[~np.isnan(maes)]
            m_ok   = [v for v in maes if not np.isnan(v)]
            bars   = ax.bar(x_ok + offset, m_ok, width=w,
                            color=skip_color[skip], alpha=skip_alpha[skip],
                            hatch=skip_hatch[skip], edgecolor="white", linewidth=0.6,
                            label=f"skip = {skip}" if col == 0 else "_")
            for b, v in zip(bars, m_ok):
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + 0.35,
                        f"{v:.1f}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold",
                        color=skip_color[skip])

        ax.axhline(10.0, color="#374151", lw=0.9, ls="--", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([DOM_SHORT[d] for d in DOMAINS], fontsize=9)
        ax.set_ylim(0, 28)
        if col == 0:
            ax.set_ylabel("Mean MAE [°C]", fontsize=10)
        ax.tick_params(axis="y", length=3)

    axes[2].text(len(DOMAINS) - 0.38, 10.6, "10 °C", fontsize=7.5, color="#374151", alpha=0.45)

    handles = [
        plt.Rectangle((0, 0), 1, 1,
                       color=skip_color[s], alpha=skip_alpha[s],
                       hatch=skip_hatch[s], ec="white",
                       label=f"skip = {s}  ({'52' if s==2 else '71'}% FEM steps saved)")
        for s in skips
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(os.path.join(RESULTS, "fig2_mae_per_arch.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig2_mae_per_arch.png")


# ══════════════════════════════════════════════════════════════
# Figure 3: 2D Results per Architecture
# ══════════════════════════════════════════════════════════════

def fig_2d_per_arch():
    """
    3 subplots — one per architecture.
    Shows MCO-PINN MAE vs skip (primary y) and L2 Baseline (secondary y).
    """
    skips     = [1, 2, 4, 6]
    fem_count = {1: 21, 2: 11, 4: 6, 6: 4}
    x_labels  = [f"skip = {s}\n({fem_count[s]} FEM steps)" for s in skips]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "2D MCO-PINN Skip Operator — Training Results by Architecture",
        fontsize=11, fontweight="bold",
    )

    for col, arch in enumerate(ARCHS):
        ax = axes[col]
        c  = ARCH_COLOR[arch]

        maes = [mco_2d[arch][str(s)]["mae_C"] for s in skips]
        bls  = [baseline[arch][str(s)]        for s in skips]

        ax.plot(range(len(skips)), bls, "s--",
                color=C_GRAY, lw=1.5, ms=7, alpha=0.7,
                label="L2 Baseline (fixed weights)", zorder=2)
        ax.plot(range(len(skips)), maes, "o-",
                color=c, lw=2.2, ms=8,
                label="MCO-PINN (adaptive)", zorder=3)

        for xi_i, (m, b) in enumerate(zip(maes, bls)):
            ax.annotate(f"{m:.1f}",
                        xy=(xi_i, m), xytext=(5, 7),
                        textcoords="offset points",
                        fontsize=8.5, color=c, fontweight="bold")
            ax.annotate(f"{b:.0f}",
                        xy=(xi_i, b), xytext=(5, -13),
                        textcoords="offset points",
                        fontsize=7.5, color=C_GRAY)

        ax.axhline(5.0, color="#374151", lw=0.9, ls=":", alpha=0.35)
        ax.text(3.1, 5.6, "5 °C", fontsize=7.5, color="#374151", alpha=0.4)
        ax.set_xticks(range(len(skips)))
        ax.set_xticklabels(x_labels, fontsize=8.5)
        ax.set_title(f"({chr(97+col)})  {ARCH_LABEL[arch]}",
                     fontsize=10, fontweight="bold", color=c)
        ax.set_ylabel("Mean MAE [°C]", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(linestyle=":", alpha=0.35)
        ax.set_axisbelow(True)

        ax2 = ax.twinx()
        ax2.set_ylim(0, max(bls) * 1.35)
        ax2.set_ylabel("L2 Baseline MAE [°C]", fontsize=8, color=C_GRAY)
        ax2.tick_params(colors=C_GRAY, labelsize=7)
        ax2.spines["top"].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig3_2d_per_arch.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig3_2d_per_arch.png")


# ══════════════════════════════════════════════════════════════
# Figure 4: Summary Comparison Table
# ══════════════════════════════════════════════════════════════

def fig_summary_table():
    """
    Matplotlib table: skip=2 (left) and skip=4 (right).
    Rows: 3 architectures.
    Columns: Baseline | 2D MCO | 3D-Rect | 3D-Cyl | 3D-Stack.
    """
    skips    = [2, 4]
    fem_used = {2: 11, 4: 6}
    fem_total= 21

    col_labels = [
        "Architecture",
        "Baseline\n(L2, 2D)",
        "MCO-PINN\n(2D)",
        "3D\nRectangular",
        "3D\nCylinder",
        "3D\nStacked",
    ]

    def _cell_bg(val, lo=1.0, hi=50.0):
        """Light green (low/good) → light orange → light red (high/bad)."""
        t = np.clip((val - lo) / (hi - lo), 0, 1)
        if t < 0.5:
            # green → yellow
            r = 0.55 + 0.90 * (t * 2)
            g = 0.90 - 0.10 * (t * 2)
            b = 0.55 - 0.40 * (t * 2)
        else:
            # yellow → red
            r = 1.00
            g = 0.80 - 0.65 * ((t - 0.5) * 2)
            b = 0.15 - 0.10 * ((t - 0.5) * 2)
        return (np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1), 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 3.8))
    fig.suptitle(
        "Skip Operator — Summary: Mean MAE [°C]\n"
        "Columns: Baseline (L2 fixed weights, 2D)  ·  2D MCO-PINN  ·  3D MCO-PINN per domain",
        fontsize=10.5, fontweight="bold",
    )

    for ax_idx, skip in enumerate(skips):
        ax = axes[ax_idx]
        ax.axis("off")

        rows       = []
        cell_colors= []

        for arch in ARCHS:
            bl  = baseline[arch][str(skip)]
            m2d = mco_2d[arch][str(skip)]["mae_C"]
            m3d = {d: data_3d[d][arch][str(skip)]["mae_C"] for d in DOMAINS if d in data_3d}

            rows.append([
                ARCH_LABEL[arch],
                f"{bl:.1f} °C",
                f"{m2d:.2f} °C",
                f"{m3d['rectangular']:.2f} °C",
                f"{m3d['cylinder']:.2f} °C",
                f"{m3d['stacked']:.2f} °C",
            ])

            vals = [bl, m2d, m3d["rectangular"], m3d["cylinder"], m3d["stacked"]]
            cell_colors.append(
                ["#EBEBEB"] + [_cell_bg(v) for v in vals]
            )

        tbl = ax.table(
            cellText    = rows,
            colLabels   = col_labels,
            cellColours = cell_colors,
            loc         = "center",
            cellLoc     = "center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1.0, 2.4)

        # Header row
        for j in range(len(col_labels)):
            cell = tbl[0, j]
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)

        # Architecture column
        for i, arch in enumerate(ARCHS, start=1):
            cell = tbl[i, 0]
            cell.set_facecolor("#F1F5F9")
            cell.set_text_props(fontweight="bold", color=ARCH_COLOR[arch])

        fem_s   = fem_used[skip]
        savings = round((fem_total - fem_s) / fem_total * 100)
        ax.set_title(
            f"skip = {skip}  —  {fem_s}/{fem_total} FEM steps  ({savings}% savings)",
            fontsize=10, fontweight="bold", pad=14,
            color=C_SK2 if skip == 2 else C_SK4,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(RESULTS, "fig4_summary_table.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig4_summary_table.png")


# ══════════════════════════════════════════════════════════════
# Figure 5: Per-Timestep Skip Analysis — FEM vs PINN timeline
# ══════════════════════════════════════════════════════════════

def fig_skip_timeline():
    """
    For each skip value (2 and 4), shows ALL 21 time steps:
      - Which steps FEM computed  (●  solid circle)
      - Which steps PINN predicted (▲  triangle with ±MAE error bar)
    Top panel  : T_surface and T_center curves + markers
    Middle panel: MAE at each PINN-predicted window (bar, green→red)
    Bottom     : Step-by-step table (all 21 steps)
    Architecture: bayesian (best 2D result). Domain: Rectangular (2D).
    """
    from level8_nas_mco_pinn.domains_3d import Rectangular3D

    dom   = Rectangular3D()
    arch  = "bayesian"
    skips = [2, 4]

    # All FEM time points (21 steps, every 1.5 s)
    t_fem   = np.arange(0, 30 + 1e-9, 1.5)   # 0, 1.5, 3, ..., 30

    # Analytical T at surface (x=0) and center (x=Lx/2)
    # Use y=Ly/2, z=Lz/2 (mid-plane)
    x_surf  = np.array([0.0])
    x_cen   = np.array([dom.Lx / 2])
    y_mid   = np.array([dom.Ly / 2])
    z_mid   = np.array([dom.Lz / 2])

    T_surf_exact = np.array([dom.T(x_surf, y_mid, z_mid, t)[0] for t in t_fem])
    T_cen_exact  = np.array([dom.T(x_cen,  y_mid, z_mid, t)[0] for t in t_fem])

    fig, axes = plt.subplots(len(skips), 3,
                             figsize=(20, 6 * len(skips)),
                             gridspec_kw={"width_ratios": [2.2, 1.0, 1.8]})
    fig.suptitle(
        "Skip Operator — Per-Timestep Analysis  (2D Rectangular Domain, Bayesian Architecture)\n"
        "Which steps FEM computes vs. PINN predicts, temperature values and errors at each step",
        fontsize=11, fontweight="bold",
    )

    for row_idx, skip in enumerate(skips):
        ax_tl  = axes[row_idx, 0]   # timeline / temperature
        ax_bar = axes[row_idx, 1]   # MAE bar chart
        ax_tbl = axes[row_idx, 2]   # table

        # FEM anchor indices
        fem_idx = list(range(0, len(t_fem), skip))
        if fem_idx[-1] != len(t_fem) - 1:
            fem_idx.append(len(t_fem) - 1)
        fem_set = set(fem_idx)

        # PINN windows: (i_start, i_end) pairs
        windows = list(zip(fem_idx[:-1], fem_idx[1:]))
        n_win   = len(windows)

        # Per-window MAE (from stored results)
        mae_per_win = mco_2d[arch][str(skip)]["mae_per_window"]

        # ── Timeline panel ─────────────────────────────────────
        ax_tl.plot(t_fem, T_surf_exact, "-", color="#1565C0",
                   lw=1.8, alpha=0.5, label="Surface T (exact)", zorder=1)
        ax_tl.plot(t_fem, T_cen_exact,  "-", color="#D32F2F",
                   lw=1.8, alpha=0.5, label="Center T (exact)", zorder=1)

        for i, t in enumerate(t_fem):
            if i in fem_set:
                # FEM anchor: solid circles
                ax_tl.plot(t, T_surf_exact[i], "o",
                           color="#1565C0", ms=9, zorder=3,
                           markeredgecolor="white", markeredgewidth=0.8)
                ax_tl.plot(t, T_cen_exact[i],  "o",
                           color="#D32F2F",  ms=9, zorder=3,
                           markeredgecolor="white", markeredgewidth=0.8)
            else:
                # PINN predicted: find which window this step belongs to
                win_i = None
                for wi, (i_s, i_e) in enumerate(windows):
                    if i_s < i < i_e or (i == i_e and i not in fem_set):
                        win_i = wi
                        break
                mae = mae_per_win[win_i] if win_i is not None and win_i < len(mae_per_win) else 0

                ax_tl.errorbar(t, T_surf_exact[i], yerr=mae,
                               fmt="^", color="#E65100", ms=8, zorder=4,
                               ecolor="#E65100", elinewidth=1.5, capsize=4,
                               markeredgecolor="white", markeredgewidth=0.6)
                ax_tl.errorbar(t, T_cen_exact[i],  yerr=mae,
                               fmt="^", color="#880E4F", ms=8, zorder=4,
                               ecolor="#880E4F", elinewidth=1.5, capsize=4,
                               markeredgecolor="white", markeredgewidth=0.6)

        # Shade FEM vs PINN regions
        for wi, (i_s, i_e) in enumerate(windows):
            ts, te = t_fem[i_s], t_fem[i_e]
            for mid_i in range(i_s + 1, i_e):
                ax_tl.axvspan(t_fem[mid_i] - 0.75, t_fem[mid_i] + 0.75,
                              alpha=0.07, color="#E65100", zorder=0)

        ax_tl.set_xlabel("Time [s]", fontsize=9)
        ax_tl.set_ylabel("Temperature [°C]", fontsize=9)
        skip_color = C_SK2 if skip == 2 else C_SK4
        fem_n  = len(fem_set)
        pinn_n = len(t_fem) - fem_n
        ax_tl.set_title(
            f"skip = {skip}  |  ● FEM computed ({fem_n} steps)"
            f"   ▲ PINN predicted ({pinn_n} steps, ±MAE error bar)",
            fontsize=9.5, fontweight="bold", color=skip_color,
        )
        # Legend entries
        from matplotlib.lines import Line2D
        leg_els = [
            Line2D([0],[0], marker="o", color="#1565C0", ms=8, lw=1.5,
                   label="Surface T — FEM computed"),
            Line2D([0],[0], marker="o", color="#D32F2F",  ms=8, lw=1.5,
                   label="Center T  — FEM computed"),
            Line2D([0],[0], marker="^", color="#E65100", ms=8, lw=0,
                   label="Surface T — PINN predicted (±MAE)"),
            Line2D([0],[0], marker="^", color="#880E4F", ms=8, lw=0,
                   label="Center T  — PINN predicted (±MAE)"),
        ]
        ax_tl.legend(handles=leg_els, fontsize=7.5, loc="lower left")
        ax_tl.grid(linestyle=":", alpha=0.35)
        ax_tl.set_xlim(-0.5, 31)
        ax_tl.set_ylim(T_WATER - 20, T_INIT + 20)
        ax_tl.spines[["top", "right"]].set_visible(False)

        # ── MAE bar chart per window ────────────────────────────
        x_bars  = np.arange(n_win)
        t_mids  = [(t_fem[i_s] + t_fem[i_e]) / 2 for i_s, i_e in windows]
        bar_labels = [f"{t_fem[i_s]:.0f}→{t_fem[i_e]:.0f}" for i_s, i_e in windows]

        cmap_bar = plt.cm.RdYlGn_r
        mae_max  = max(mae_per_win) if mae_per_win else 10.0
        for bi, (mae, t_mid) in enumerate(zip(mae_per_win, t_mids)):
            clr = cmap_bar(np.clip(mae / 10.0, 0, 1))
            ax_bar.bar(bi, mae, color=clr, edgecolor="white", linewidth=0.5)
            ax_bar.text(bi, mae + 0.05, f"{mae:.1f}", ha="center",
                        fontsize=7, fontweight="bold")

        ax_bar.axhline(5.0, color="#374151", lw=1.0, ls="--", alpha=0.5)
        ax_bar.text(n_win - 0.5, 5.2, "5 °C", fontsize=7.5, color="#374151", ha="right")
        ax_bar.set_xticks(x_bars)
        ax_bar.set_xticklabels(bar_labels, fontsize=6.5, rotation=45, ha="right")
        ax_bar.set_ylabel("MAE [°C]", fontsize=9)
        ax_bar.set_title("MAE per\nPINN window", fontsize=9, fontweight="bold")
        ax_bar.set_ylim(0, max(mae_per_win) * 1.25 + 1)
        ax_bar.spines[["top", "right"]].set_visible(False)
        ax_bar.grid(axis="y", linestyle=":", alpha=0.35)

        # ── Step-by-step table ──────────────────────────────────
        ax_tbl.axis("off")

        col_hdrs = ["Step", "t [s]", "Type", "T_surf [°C]", "T_cen [°C]", "MAE [°C]"]
        rows_tbl = []
        row_clrs = []

        win_ptr = 0
        for i, t in enumerate(t_fem):
            step_no  = i + 1
            t_s      = f"{t:.1f}"
            t_surf_v = f"{T_surf_exact[i]:.1f}"
            t_cen_v  = f"{T_cen_exact[i]:.1f}"

            if i in fem_set:
                row_type = "FEM ●"
                mae_str  = "—"
                bg       = "#DBEAFE"   # light blue
            else:
                # find window
                win_i = None
                for wi2, (i_s2, i_e2) in enumerate(windows):
                    if i_s2 < i < i_e2 or i == i_e2:
                        win_i = wi2; break
                mae_v   = mae_per_win[win_i] if win_i is not None and win_i < len(mae_per_win) else 0
                mae_str = f"{mae_v:.2f}"
                row_type = "PINN ▲"
                # color by quality
                if mae_v < 3:
                    bg = "#DCFCE7"   # green
                elif mae_v < 7:
                    bg = "#FEF9C3"   # yellow
                else:
                    bg = "#FEE2E2"   # red

            rows_tbl.append([str(step_no), t_s, row_type, t_surf_v, t_cen_v, mae_str])
            row_clrs.append([bg] * 6)

        tbl = ax_tbl.table(
            cellText    = rows_tbl,
            colLabels   = col_hdrs,
            cellColours = row_clrs,
            loc         = "center",
            cellLoc     = "center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.2)
        tbl.scale(1.0, 0.97)

        for j in range(len(col_hdrs)):
            cell = tbl[0, j]
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", fontweight="bold", fontsize=7.5)

        ax_tbl.set_title(
            "All 21 FEM time steps\n"
            "● FEM computed  ▲ PINN predicted\n"
            "Green < 3°C  Yellow < 7°C  Red ≥ 7°C",
            fontsize=8.5, fontweight="bold", pad=8,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(RESULTS, "fig5_skip_timeline.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig5_skip_timeline.png")


# ══════════════════════════════════════════════════════════════
# Figure 6: v1 vs v2 Comparison + Loss Curves + Heat Maps
# ══════════════════════════════════════════════════════════════

def fig_v2_comparison():
    """
    Layout: 3 rows (arch) × 5 cols
      Col 0 : MAE bar — v1 vs v2, skip=2 and skip=4, all 3 domains
      Col 1 : Loss curves — all windows, rectangular, skip=2
      Col 2-4: 3D surface (T_pred) — Rectangular / Cylinder / Stacked
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    if not data_v2:
        print("  [skip] v2 results not found — run run_3d_v2.py first")
        return

    C_V1   = "#90A4AE"
    C_V2s2 = "#1565C0"
    C_V2s4 = "#E65100"

    fig = plt.figure(figsize=(36, 17))
    fig.suptitle(
        "3D MCO-PINN — v2 (2000 ep)  ·  MAE Comparison | Training Loss | 3D Temperature Surface per Domain",
        fontsize=12, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(3, 6, figure=fig,
                           width_ratios=[1.5, 1.3, 1, 1, 1, 1],
                           hspace=0.50, wspace=0.30)

    col_labels = ["Rectangular", "Cylinder", "Stacked Cubes", "L-Shape"]

    for row, arch in enumerate(ARCHS):
        c = ARCH_COLOR[arch]

        # ── Col 0: MAE bar ─────────────────────────────────────────
        ax_bar = fig.add_subplot(gs[row, 0])
        x  = np.arange(len(DOMAINS))
        w  = 0.18
        bar_cfg = [
            (-1.5*w, "v1 s=2", C_V1,   0.55, "",    2, data_3d),
            (-0.5*w, "v2 s=2", C_V2s2, 0.90, "",    2, data_v2),
            ( 0.5*w, "v1 s=4", C_V1,   0.55, "///", 4, data_3d),
            ( 1.5*w, "v2 s=4", C_V2s4, 0.90, "///", 4, data_v2),
        ]
        for off, lbl, clr, alp, hatch, sk, src in bar_cfg:
            maes = [src.get(d, {}).get(arch, {}).get(str(sk), {}).get("mae_C", np.nan)
                    for d in DOMAINS]
            valid_mask = [not np.isnan(v) for v in maes]
            x_plot = x[valid_mask]
            maes_plot = [v for v, ok in zip(maes, valid_mask) if ok]
            off_arr = np.zeros(len(x_plot)) + off
            bars = ax_bar.bar(x_plot + off_arr, maes_plot, width=w, color=clr, alpha=alp,
                              hatch=hatch, edgecolor="white", lw=0.5,
                              label=lbl if row == 0 else "_")
            for b, v in zip(bars, maes_plot):
                ax_bar.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                            f"{v:.1f}", ha="center", fontsize=6.5,
                            fontweight="bold", color=clr)

        ax_bar.axhline(10.0, color="#374151", lw=0.8, ls="--", alpha=0.35)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([DOM_SHORT[d] for d in DOMAINS], fontsize=9)
        ax_bar.set_ylim(0, 28)
        ax_bar.set_ylabel("Mean MAE [°C]", fontsize=9)
        ax_bar.set_title(f"({chr(97+row*6)})  {ARCH_LABEL[arch]} — MAE",
                         fontsize=9.5, fontweight="bold", color=c)
        ax_bar.spines[["top","right"]].set_visible(False)
        ax_bar.grid(axis="y", linestyle=":", alpha=0.30)
        ax_bar.set_axisbelow(True)
        if row == 0:
            ax_bar.legend(fontsize=7, ncol=2, loc="upper right",
                          framealpha=0.85)

        # ── Col 1: Loss curves (rectangular, skip=2) ───────────────
        ax_loss = fig.add_subplot(gs[row, 1])
        loss_path = os.path.join(V2_DIR, f"rectangular_{arch}_skip2_loss.json")
        if os.path.exists(loss_path):
            with open(loss_path) as f:
                loss_data = json.load(f)
            cmap_w = plt.cm.turbo(np.linspace(0.1, 0.9, len(loss_data)))
            for wi, (hist, col_w) in enumerate(zip(loss_data, cmap_w)):
                ep = np.linspace(0, 1, len(hist["L_total"]))
                ax_loss.semilogy(ep, hist["L_total"], "-",
                                 color=col_w, lw=1.3, alpha=0.85,
                                 label=f"W{wi+1}" if wi < 6 else "_")
            ax_loss.set_xlabel("Training progress (norm.)", fontsize=8)
            ax_loss.set_ylabel("Total Loss (log)", fontsize=8)
            ax_loss.set_title(
                f"({chr(97+row*6+1)})  {ARCH_LABEL[arch]} — Loss\n"
                "(Rectangular · skip=2 · all windows)",
                fontsize=9, fontweight="bold", color=c)
            ax_loss.legend(fontsize=7, loc="upper right", ncol=3)
            ax_loss.grid(True, linestyle=":", alpha=0.3, which="both")
            ax_loss.spines[["top","right"]].set_visible(False)
        else:
            ax_loss.text(0.5, 0.5, "No loss data", ha="center", va="center",
                         transform=ax_loss.transAxes)
            ax_loss.axis("off")

        # ── Col 2-5: actual 3D geometry + PINN z-mid slice ──────────
        dom_objs = {
            "rectangular": Rectangular3D(),
            "cylinder":    Cylinder3D(),
            "stacked":     StackedCubes3D(),
            "lshape":      LShape3D(),
        }
        wf_kw   = dict(color="#374151", lw=0.6, alpha=0.45)
        norm_s  = plt.Normalize(vmin=T_WATER, vmax=T_INIT)
        cmap_obj = plt.cm.turbo

        for di, dom_name in enumerate(DOMAINS):
            ax3 = fig.add_subplot(gs[row, 2 + di], projection="3d")
            d   = dom_objs[dom_name]

            # Draw geometry wireframe to give 3D shape context
            if dom_name == "cylinder":
                _wireframe_cyl3d(ax3, d.R, d.H, **wf_kw)
                ax3.set_xlim(-d.R, d.R); ax3.set_ylim(-d.R, d.R)
                ax3.set_zlim(0, d.H)
                asp = [2*d.R, 2*d.R, d.H]
            elif dom_name == "stacked":
                _wireframe_box3d(ax3, d.L_cube, d.L_cube, d.Lz, **wf_kw)
                ax3.plot([0,d.L_cube,d.L_cube,0,0],[0,0,d.L_cube,d.L_cube,0],
                         [d.L_cube]*5, **wf_kw)   # joint line
                ax3.set_xlim(0, d.L_cube); ax3.set_ylim(0, d.L_cube)
                ax3.set_zlim(0, d.Lz)
                asp = [d.L_cube, d.L_cube, d.Lz]
            elif dom_name == "lshape":
                _wireframe_lshape3d(ax3, d.Lx, d.Ly, d.Lz, d.cut_x, d.cut_y, **wf_kw)
                ax3.set_xlim(0, d.Lx); ax3.set_ylim(0, d.Ly)
                ax3.set_zlim(0, d.Lz)
                asp = [d.Lx, d.Ly, d.Lz]
            else:  # rectangular
                _wireframe_box3d(ax3, d.Lx, d.Ly, d.Lz, **wf_kw)
                ax3.set_xlim(0, d.Lx); ax3.set_ylim(0, d.Ly)
                ax3.set_zlim(0, d.Lz)
                asp = [d.Lx, d.Ly, d.Lz]

            asp_max = max(asp)
            ax3.set_box_aspect([a / asp_max for a in asp])

            slice_path = os.path.join(V2_DIR, f"{dom_name}_{arch}_skip2_slice.json")
            if os.path.exists(slice_path):
                with open(slice_path) as f:
                    sd = json.load(f)
                xi   = np.array(sd["xi"]); yi = np.array(sd["yi"])
                z_val = float(sd["z_val"])
                wins  = sd["windows"]
                mid_w = str(len(wins) // 2)
                T_pred = np.array(wins[mid_w]["T_pred"], dtype=float)
                T_fem  = np.array(wins[mid_w]["T_fem"],  dtype=float)
                XX, YY = np.meshgrid(xi, yi)
                ZZ_flat = np.full_like(XX, z_val)
                T_ma    = np.ma.masked_invalid(T_pred)
                fc      = cmap_obj(norm_s(T_ma.filled(T_WATER)))
                ax3.plot_surface(XX, YY, ZZ_flat,
                                 facecolors=fc, shade=False,
                                 alpha=0.93, linewidth=0, antialiased=True)
                # FEM contour on same plane (white)
                T_fem_ma = np.ma.masked_invalid(T_fem)
                ax3.contour(XX, YY, T_fem_ma, levels=6,
                            zdir="z", offset=z_val,
                            colors="white", linewidths=0.6, alpha=0.7)

                mae_w = float(np.nanmean(np.abs(T_pred - T_fem)))
                lbl_chr = chr(97 + row * 6 + 2 + di)
                ax3.set_title(
                    f"({lbl_chr})  {ARCH_LABEL[arch]}\n"
                    f"{col_labels[di]}  MAE={mae_w:.1f}°C",
                    fontsize=8.5, fontweight="bold", color=c, pad=3)
            else:
                ax3.set_title(f"{col_labels[di]}\n(no data)", fontsize=8)

            ax3.set_xlabel("x [m]", fontsize=6, labelpad=0)
            ax3.set_ylabel("y [m]", fontsize=6, labelpad=0)
            ax3.set_zlabel("z [m]", fontsize=6, labelpad=0)
            ax3.tick_params(labelsize=5, pad=0)
            ax3.view_init(elev=28, azim=-50)

    fig.savefig(os.path.join(RESULTS, "fig6_v2_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig6_v2_comparison.png")


def fig_v2_summary_table():
    """Extended summary table: v1 vs v2 side by side."""
    if not data_v2:
        print("  [skip] v2 results not found"); return

    skips    = [2, 4]
    fem_used = {2: 11, 4: 6}
    fem_total= 21

    col_labels = [
        "Architecture",
        "2D MCO\n(ref.)",
        "3D v1\nRect.", "3D v1\nCyl.", "3D v1\nStack",
        "3D v2\nRect.", "3D v2\nCyl.", "3D v2\nStack", "3D v2\nL-Shape",
    ]

    def _bg(val, lo=1.0, hi=25.0):
        # green (0.2,0.8,0.2) → amber (0.95,0.75,0.1) → red (0.9,0.1,0.1)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return (0.9, 0.9, 0.9, 1.0)   # light grey for missing data
        t = float(np.clip((val - lo) / (hi - lo), 0, 1))
        if t < 0.5:
            s = t * 2
            return (np.clip(0.2 + 0.75*s, 0, 1), np.clip(0.8 - 0.05*s, 0, 1), np.clip(0.2 - 0.1*s, 0, 1), 1.0)
        s = (t - 0.5) * 2
        return (np.clip(0.95 - 0.05*s, 0, 1), np.clip(0.75 - 0.65*s, 0, 1), np.clip(0.1 - 0.0*s, 0, 1), 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 4.2))
    fig.suptitle(
        "3D MCO-PINN — Full Comparison Table  (v1: 800 epochs  ·  v2: 2000 epochs)\n"
        "Mean MAE [°C]  ·  Green = good  ·  Red = high error",
        fontsize=11, fontweight="bold",
    )

    for ax_idx, skip in enumerate(skips):
        ax = axes[ax_idx]
        ax.axis("off")
        rows, cell_colors = [], []

        for arch in ARCHS:
            m2d  = mco_2d[arch][str(skip)]["mae_C"]
            v1   = {d: data_3d.get(d, {}).get(arch, {}).get(str(skip), {}).get("mae_C", np.nan)
                    for d in DOMAINS}
            v2   = {d: data_v2.get(d, {}).get(arch, {}).get(str(skip), {}).get("mae_C", np.nan)
                    for d in DOMAINS}

            def _fmt(v): return f"{v:.2f}°C" if not np.isnan(v) else "—"
            rows.append([
                ARCH_LABEL[arch],
                f"{m2d:.2f}°C",
                _fmt(v1['rectangular']),
                _fmt(v1['cylinder']),
                _fmt(v1['stacked']),
                _fmt(v2['rectangular']),
                _fmt(v2['cylinder']),
                _fmt(v2['stacked']),
                _fmt(v2.get('lshape', np.nan)),
            ])
            vals = [m2d,
                    v1["rectangular"], v1["cylinder"], v1["stacked"],
                    v2["rectangular"], v2["cylinder"], v2["stacked"],
                    v2.get("lshape", np.nan)]
            cell_colors.append(["#EBEBEB"] + [_bg(v) for v in vals])

        tbl = ax.table(cellText=rows, colLabels=col_labels,
                       cellColours=cell_colors, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1.0, 2.5)

        for j in range(len(col_labels)):
            cell = tbl[0, j]
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)

        # v2 column headers in blue (cols 5-8: Rect, Cyl, Stack, L-Shape)
        for j in range(5, 9):
            tbl[0, j].set_facecolor("#1565C0")

        for i, arch in enumerate(ARCHS, start=1):
            tbl[i, 0].set_text_props(fontweight="bold", color=ARCH_COLOR[arch])
            tbl[i, 0].set_facecolor("#F1F5F9")

        fem_s = fem_used[skip]
        savings = round((fem_total - fem_s) / fem_total * 100)
        ax.set_title(
            f"skip = {skip}  —  {fem_s}/{fem_total} FEM steps  ({savings}% savings)\n"
            f"Blue columns = v2 (2000 epochs, improved)",
            fontsize=10, fontweight="bold", pad=16,
            color="#1565C0" if skip == 2 else C_SK4,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(RESULTS, "fig7_v2_table.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig7_v2_table.png")


# ══════════════════════════════════════════════════════════════
# Figure 8: 3D PINN Feasibility — Can PINN replace FEM steps?
# ══════════════════════════════════════════════════════════════

def fig_3d_feasibility():
    """
    Research question: Can PINN replace FEM time steps for complex 3D domains?

    Layout: 3 rows (domain) × 3 cols
      Col 0 : Temperature trajectory — FEM mean T(t) vs PINN predictions
      Col 1 : MAE per window for each available skip value
      Col 2 : Feasibility summary — heatmap of MAE[domain × skip]

    Uses best arch (lowest mean MAE) per domain from v2 data.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D

    if not data_v2:
        print("  [skip] fig8 — v2 results not found"); return

    # Only include domains that have v2 data
    _dom_candidates = {
        "rectangular": Rectangular3D,
        "cylinder":    Cylinder3D,
        "stacked":     StackedCubes3D,
        "lshape":      LShape3D,
    }
    dom_objs = {k: _dom_candidates[k]()
                for k in _dom_candidates if k in data_v2}
    n_dom_rows = len(dom_objs)
    if n_dom_rows == 0:
        print("  [skip] fig8 — no domain data found"); return

    # FEM time grid: t = 0, 1.5, 3, …, 30 s  (21 points)
    t_fem = np.arange(0, 30 + 1e-9, 1.5)

    SKIP_COLORS = {"1": "#7B1FA2", "2": "#1565C0", "4": "#E65100", "6": "#2E7D32"}
    SKIP_LABELS = {"1": "skip=1  (0% saved)", "2": "skip=2  (52% saved)",
                   "4": "skip=4  (71% saved)", "6": "skip=6  (81% saved)"}
    MAE_OK   = 10.0   # °C  — engineering threshold
    MAE_WARN = 15.0   # °C

    fig = plt.figure(figsize=(22, 5.5 * n_dom_rows))
    fig.suptitle(
        "Research Q: Can PINN Replace FEM Steps in Complex 3D Domains?\n"
        "Left: Temperature trajectory  ·  Centre: MAE per time window  ·  Right: Feasibility map",
        fontsize=12, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(n_dom_rows, 3, figure=fig,
                           width_ratios=[1.6, 1.4, 1.0],
                           hspace=0.45, wspace=0.32)

    # Collect all skip values present in v2 data
    all_skips_int = sorted({int(s)
                            for dom in data_v2.values()
                            for arch in dom.values()
                            for s in arch})
    all_skips = [str(s) for s in all_skips_int]

    # Feasibility matrix: rows=domains, cols=skips  (best arch MAE)
    feas_mat = np.full((n_dom_rows, len(all_skips)), np.nan)

    dom_colors = [C_BAY, C_N3, C_N2, "#6A1B9A"]

    for ri, dom_name in enumerate(dom_objs):
        dom      = dom_objs[dom_name]
        ax_traj  = fig.add_subplot(gs[ri, 0])
        ax_mae   = fig.add_subplot(gs[ri, 1])
        c_dom    = dom_colors[ri % len(dom_colors)]

        # ── Best arch for this domain (v2, skip=2) ──────────────────
        best_arch = min(ARCHS,
                        key=lambda a: data_v2[dom_name][a].get("2", {}).get("mae_C", 999))

        # ── Col 0: Temperature trajectory ───────────────────────────
        # FEM analytical mean temperature over the domain at each t
        t_plot   = t_fem[1:]    # skip t=0 (IC, same everywhere)
        T_fem_mean = []
        for t in t_plot:
            # sample ~300 interior points
            np.random.seed(42)
            n_s = 300
            if dom_name == "cylinder":
                r_s = np.random.uniform(0, dom.R, n_s)
                th_s = np.random.uniform(0, 2*np.pi, n_s)
                xs = r_s*np.cos(th_s); ys = r_s*np.sin(th_s)
                zs = np.random.uniform(0, dom.H, n_s)
                T_s = dom.T_xyz(xs, ys, zs, t)
            elif dom_name == "stacked":
                xs = np.random.uniform(0, dom.L_cube, n_s)
                ys = np.random.uniform(0, dom.L_cube, n_s)
                zs = np.random.uniform(0, dom.Lz, n_s)
                T_s = dom.T(xs, ys, zs, t)
            elif dom_name == "lshape":
                # Rejection sampling inside L-shape
                pts_x, pts_y, pts_z = [], [], []
                while len(pts_x) < n_s:
                    xi = np.random.uniform(0, dom.Lx, n_s * 2)
                    yi = np.random.uniform(0, dom.Ly, n_s * 2)
                    zi = np.random.uniform(0, dom.Lz, n_s * 2)
                    m = dom.mask(xi, yi, zi)
                    pts_x.extend(xi[m]); pts_y.extend(yi[m]); pts_z.extend(zi[m])
                xs = np.array(pts_x[:n_s])
                ys = np.array(pts_y[:n_s])
                zs = np.array(pts_z[:n_s])
                T_s = dom.T(xs, ys, zs, t)
                T_s = T_s[~np.isnan(T_s)]
            else:
                xs = np.random.uniform(0, dom.Lx, n_s)
                ys = np.random.uniform(0, dom.Ly, n_s)
                zs = np.random.uniform(0, dom.Lz, n_s)
                T_s = dom.T(xs, ys, zs, t)
            T_fem_mean.append(float(np.nanmean(T_s)))

        ax_traj.plot(t_plot, T_fem_mean, '-', color="#1F2937",
                     lw=2.0, label="FEM (reference)", zorder=5)

        # PINN trajectory from slice JSON for each skip
        for skip in all_skips:
            sp = os.path.join(V2_DIR,
                 f"{dom_name}_{best_arch}_skip{skip}_slice.json")
            if not os.path.exists(sp):
                continue
            with open(sp) as f:
                sd = json.load(f)
            wins = sd["windows"]
            # Each window end time: window wi covers t_fem[wi*skip .. (wi+1)*skip]
            skip_i = int(skip)
            t_pinn, T_pinn_mean, T_fem_pinn = [], [], []
            for wi in range(len(wins)):
                t_end_idx = min((wi + 1) * skip_i, len(t_fem) - 1)
                t_end     = t_fem[t_end_idx]
                T_p  = np.array(wins[str(wi)]["T_pred"], dtype=float)
                T_f  = np.array(wins[str(wi)]["T_fem"],  dtype=float)
                t_pinn.append(t_end)
                T_pinn_mean.append(float(np.nanmean(T_p)))
                T_fem_pinn.append(float(np.nanmean(T_f)))

            clr = SKIP_COLORS.get(skip, "#888")
            ax_traj.plot(t_pinn, T_pinn_mean, 'o--',
                         color=clr, lw=1.2, ms=5,
                         label=f"PINN {SKIP_LABELS[skip]}",
                         zorder=4)

        ax_traj.set_xlabel("Time [s]", fontsize=9)
        ax_traj.set_ylabel("Mean T [°C]", fontsize=9)
        ax_traj.set_title(f"({chr(97+ri*3)})  {DOM_LABEL[dom_name]}\n"
                          f"Temperature trajectory  (arch: {ARCH_LABEL[best_arch]})",
                          fontsize=9.5, fontweight="bold", color=c_dom)
        ax_traj.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
        ax_traj.grid(linestyle=":", alpha=0.3)
        ax_traj.spines[["top","right"]].set_visible(False)

        # ── Col 1: MAE per window ────────────────────────────────────
        for skip in all_skips:
            if skip not in data_v2[dom_name][best_arch]:
                continue
            mae_wins = data_v2[dom_name][best_arch][skip].get("mae_windows", [])
            if not mae_wins:
                continue
            skip_i = int(skip)
            t_ends = [t_fem[min((wi+1)*skip_i, len(t_fem)-1)]
                      for wi in range(len(mae_wins))]
            clr = SKIP_COLORS.get(skip, "#888")
            ax_mae.plot(t_ends, mae_wins, 'o-',
                        color=clr, lw=1.5, ms=5,
                        label=f"skip={skip}", zorder=4)

        ax_mae.axhline(MAE_OK,   color="#16a34a", lw=1.0, ls="--", alpha=0.7,
                       label=f"{MAE_OK}°C  ✓ OK")
        ax_mae.axhline(MAE_WARN, color="#dc2626", lw=1.0, ls=":",  alpha=0.7,
                       label=f"{MAE_WARN}°C  ✗ High")
        ax_mae.set_xlabel("Time [s]", fontsize=9)
        ax_mae.set_ylabel("MAE [°C]", fontsize=9)
        ax_mae.set_title(f"({chr(98+ri*3)})  {DOM_SHORT[dom_name]} — MAE per Window",
                         fontsize=9.5, fontweight="bold", color=c_dom)
        ax_mae.legend(fontsize=7.5, loc="upper left", framealpha=0.85, ncol=2)
        ax_mae.grid(linestyle=":", alpha=0.3)
        ax_mae.spines[["top","right"]].set_visible(False)
        ax_mae.set_ylim(bottom=0)

        # Fill feasibility matrix (best arch)
        for si, skip in enumerate(all_skips):
            if skip in data_v2[dom_name][best_arch]:
                feas_mat[ri, si] = data_v2[dom_name][best_arch][skip]["mae_C"]

    # ── Col 2: Feasibility heatmap (spanning all 3 rows) ────────────
    ax_feas = fig.add_subplot(gs[:, 2])
    masked  = np.ma.masked_invalid(feas_mat)
    cmap_f  = plt.cm.RdYlGn_r
    norm_f  = plt.Normalize(vmin=0, vmax=25)
    im      = ax_feas.imshow(masked, cmap=cmap_f, norm=norm_f,
                              aspect="auto")

    active_doms = list(dom_objs.keys())
    ax_feas.set_xticks(range(len(all_skips)))
    ax_feas.set_xticklabels([f"skip={s}" for s in all_skips], fontsize=9)
    ax_feas.set_yticks(range(n_dom_rows))
    ax_feas.set_yticklabels([DOM_SHORT[d] for d in active_doms], fontsize=9.5,
                             fontweight="bold")
    ax_feas.set_title("(i)  Feasibility Map\nMAE [°C] — best arch per domain",
                      fontsize=10, fontweight="bold")

    for ri2 in range(n_dom_rows):
        for si, skip in enumerate(all_skips):
            val = feas_mat[ri2, si]
            if not np.isnan(val):
                verdict = "✓" if val < MAE_OK else ("~" if val < MAE_WARN else "✗")
                ax_feas.text(si, ri2, f"{val:.1f}°C\n{verdict}",
                             ha="center", va="center",
                             fontsize=9.5, fontweight="bold",
                             color="white" if val > 13 else "#1a1a1a")
            else:
                ax_feas.text(si, ri2, "⏳", ha="center", va="center",
                             fontsize=14)

    cbar = plt.colorbar(im, ax=ax_feas, fraction=0.046, pad=0.04)
    cbar.set_label("MAE [°C]", fontsize=9)

    # Verdict text box
    n_ok    = int(np.sum(feas_mat < MAE_OK))
    n_total = int(np.sum(~np.isnan(feas_mat)))
    verdict_txt = (f"Results: {n_ok}/{n_total} configs < {MAE_OK}°C\n"
                   f"skip=2 most feasible for 3D\n"
                   f"52% FEM step reduction achievable")
    ax_feas.text(0.5, -0.18, verdict_txt,
                 transform=ax_feas.transAxes,
                 ha="center", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.5",
                           facecolor="#d1fae5", edgecolor="#16a34a", alpha=0.85))

    fig.savefig(os.path.join(RESULTS, "fig8_3d_feasibility.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig8_3d_feasibility.png")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import shutil
    WEB_IMG = os.path.join(os.path.dirname(__file__), "web", "static", "img")

    print("Generating figures ...\n")
    fig_thermal_fields()
    fig_mae_per_arch()
    fig_2d_per_arch()
    fig_summary_table()
    fig_skip_timeline()
    fig_v2_comparison()
    fig_v2_summary_table()
    fig_3d_feasibility()

    # Auto-copy to web static
    if os.path.isdir(WEB_IMG):
        for fn in os.listdir(RESULTS):
            if fn.endswith(".png"):
                shutil.copy2(os.path.join(RESULTS, fn), WEB_IMG)
        print(f"  [OK] copied figures → {WEB_IMG}")

    print("\nDone — figures saved to level8_nas_mco_pinn/results/")
