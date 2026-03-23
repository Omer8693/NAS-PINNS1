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
    Rectangular3D, Cylinder3D, StackedCubes3D,
    T_INIT, T_WATER,
)

# ── Paths ──────────────────────────────────────────────────────
RESULTS = os.path.join(os.path.dirname(__file__), "level8_nas_mco_pinn", "results")
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
}
DOM_SHORT = {"rectangular": "Rectangular", "cylinder": "Cylinder", "stacked": "Stacked"}
DOMAINS   = ["rectangular", "cylinder", "stacked"]

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


# ══════════════════════════════════════════════════════════════
# Figure 1: 3D Temperature Field — all 3 domains, multiple times
# ══════════════════════════════════════════════════════════════

def fig_thermal_fields():
    """
    4 rows (t = 3, 10, 20, 30 s) × 3 domain columns + per-row colorbar.
    Domains: Rectangular, Cylinder, Stacked Cubes.
    Colormap: turbo (full rainbow, no black/white).
    Each row has its own colorbar scaled to that time step's T range.
    """
    dom_dict = {
        "rectangular": Rectangular3D(),
        "cylinder":    Cylinder3D(),
        "stacked":     StackedCubes3D(),
    }
    T_VALS = [3, 10, 20, 30]
    N_GRID = 34

    # Pre-build grids per domain
    grids = {}
    for dname, dom in dom_dict.items():
        if isinstance(dom, Cylinder3D):
            R = dom.R
            xi = np.linspace(-R, R, N_GRID)
            yi = np.linspace(-R, R, N_GRID)
            Lz = dom.H
        elif isinstance(dom, StackedCubes3D):
            xi = np.linspace(0, dom.L_cube, N_GRID)
            yi = np.linspace(0, dom.L_cube, N_GRID)
            Lz = dom.L_cube * 2
        else:
            xi = np.linspace(0, dom.Lx, N_GRID)
            yi = np.linspace(0, dom.Ly, N_GRID)
            Lz = dom.Lz
        z_mid = Lz / 2.0
        xx, yy = np.meshgrid(xi, yi)
        xf, yf = xx.ravel(), yy.ravel()
        zf = np.full_like(xf, z_mid)
        if isinstance(dom, Cylinder3D):
            mask = np.sqrt(xx**2 + yy**2) > dom.R
        else:
            mask = np.zeros(xx.shape, dtype=bool)
        grids[dname] = dict(dom=dom, xx=xx, yy=yy, xf=xf, yf=yf, zf=zf,
                            mask=mask, xi=xi, yi=yi, Lz=Lz)

    # Pre-compute T fields
    T_fields = {dname: {} for dname in dom_dict}
    for dname, g in grids.items():
        dom = g["dom"]
        for t in T_VALS:
            if isinstance(dom, Cylinder3D):
                T2d = dom.T_xyz(g["xf"], g["yf"], g["zf"], t).reshape(g["xx"].shape)
            else:
                T2d = dom.T(g["xf"], g["yf"], g["zf"], t).reshape(g["xx"].shape)
            T2d[g["mask"]] = np.nan
            T_fields[dname][t] = T2d

    # Layout: 4 rows × (3 domain axes + 1 colorbar axis)
    n_rows  = len(T_VALS)
    n_doms  = len(dom_dict)
    fig     = plt.figure(figsize=(15, 4.0 * n_rows))
    fig.suptitle(
        "Exact Temperature Field  T(x, y, z_mid) — Three Domain Geometries\n"
        "Rows: time steps  ·  Columns: domain shape  ·  Colorbar per row",
        fontsize=11, fontweight="bold",
    )

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(n_rows, n_doms + 1,
                           width_ratios=[1, 1, 1, 0.06],
                           hspace=0.10, wspace=0.05,
                           figure=fig)

    dom_names = list(dom_dict.keys())

    for ri, t_val in enumerate(T_VALS):
        # Collect all T values in this row to set common norm
        row_vals = []
        for dname in dom_names:
            T2d = T_fields[dname][t_val]
            row_vals.extend(T2d[~np.isnan(T2d)].ravel())
        T_min = np.percentile(row_vals, 1)
        T_max = np.percentile(row_vals, 99)
        row_norm = Normalize(vmin=T_min, vmax=T_max)

        for ci, dname in enumerate(dom_names):
            g    = grids[dname]
            T2d  = T_fields[dname][t_val]
            mask = g["mask"]
            T_plot = np.where(mask, np.nanmean(T2d[~mask]) if (~mask).any() else T_min, T2d)

            fc = plt.cm.turbo(row_norm(T_plot))
            fc[mask] = [0.96, 0.96, 0.96, 1.0]   # light gray outside geometry

            ax = fig.add_subplot(gs[ri, ci], projection="3d")
            ax.plot_surface(g["xx"], g["yy"], T_plot,
                            facecolors=fc, shade=False,
                            alpha=0.92, linewidth=0, antialiased=True)

            # Column title (first row only)
            if ri == 0:
                ax.set_title(DOM_LABEL[dname],
                             fontsize=9, fontweight="bold",
                             color={"rectangular": C_BAY,
                                    "cylinder":    C_N3,
                                    "stacked":     C_N2}[dname],
                             pad=6)

            # Row label (first column only)
            if ci == 0:
                ax.text2D(-0.14, 0.50, f"t = {t_val} s",
                          transform=ax.transAxes,
                          fontsize=9.5, fontweight="bold", color="#1F2937",
                          va="center", rotation=90)

            ax.set_xlabel("x [m]", fontsize=6, labelpad=0)
            ax.set_ylabel("y [m]", fontsize=6, labelpad=0)
            ax.set_zlabel("T [°C]", fontsize=6, labelpad=0)
            ax.tick_params(labelsize=5, pad=0)
            ax.set_zlim(T_WATER - 10, T_INIT + 10)
            ax.view_init(elev=26, azim=-52)
            ax.set_box_aspect([1, 1, 0.6])

        # Per-row colorbar in the 4th column
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
    X-axis: 3 domains, bars grouped by skip value.
    """
    skips      = [2, 4]
    skip_color = {2: C_SK2,  4: C_SK4}
    skip_alpha = {2: 0.88,   4: 0.60}
    skip_hatch = {2: "",     4: "///"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(
        "3D MCO-PINN Skip Operator — Mean Absolute Error by Architecture",
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
            maes   = [data_3d[d][arch][str(skip)]["mae_C"] for d in DOMAINS]
            offset = (si - 0.5) * w
            bars   = ax.bar(x + offset, maes, width=w,
                            color=skip_color[skip], alpha=skip_alpha[skip],
                            hatch=skip_hatch[skip], edgecolor="white", linewidth=0.6,
                            label=f"skip = {skip}" if col == 0 else "_")
            for b, v in zip(bars, maes):
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

    axes[2].text(2.62, 10.6, "10 °C", fontsize=7.5, color="#374151", alpha=0.45)

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
            m3d = {d: data_3d[d][arch][str(skip)]["mae_C"] for d in DOMAINS}

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

V2_DIR = os.path.join(RESULTS, "v2")

def fig_v2_comparison():
    """
    3 subplots per architecture (3 rows):
      Left  : MAE bar — v1 vs v2, grouped by domain × skip
      Middle : Loss curves (L_total over epochs, all windows, skip=2)
      Right  : Heat map — z-mid slice, FEM vs PINN, best window
    Only runs if v2 results exist.
    """
    v2_json = os.path.join(V2_DIR, "results_3d_v2.json")
    if not os.path.exists(v2_json):
        print("  [skip] v2 results not found — run run_3d_v2.py first")
        return

    with open(v2_json) as f:
        data_v2 = json.load(f)

    skips      = [2, 4]
    C_V1       = "#90A4AE"   # gray  — v1
    C_V2       = "#1565C0"   # blue  — v2 skip=2
    C_V2s4     = "#E65100"   # orange — v2 skip=4

    fig, axes = plt.subplots(3, 3, figsize=(19, 15))
    fig.suptitle(
        "3D MCO-PINN — v1 (800 ep) vs v2 (2000 ep) Comparison\n"
        "Left: MAE improvement  ·  Middle: Training loss curves  ·  Right: Heat map (z-mid)",
        fontsize=11, fontweight="bold",
    )

    for row, arch in enumerate(ARCHS):
        ax_bar  = axes[row, 0]
        ax_loss = axes[row, 1]
        ax_heat = axes[row, 2]

        c = ARCH_COLOR[arch]

        # ── Left: MAE bar v1 vs v2 ──────────────────────────────
        x    = np.arange(len(DOMAINS))
        w    = 0.18
        offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
        labels  = ["v1 s=2", "v2 s=2", "v1 s=4", "v2 s=4"]
        colors  = [C_V1, C_V2, C_V1, C_V2s4]
        alphas  = [0.6, 0.9, 0.6, 0.9]
        hatches = ["", "", "///", "///"]

        for i, (off, lbl, clr, alp, hatch) in enumerate(
                zip(offsets, labels, colors, alphas, hatches)):
            skip = 2 if i < 2 else 4
            src  = data_3d if i % 2 == 0 else data_v2
            maes = [src[d][arch][str(skip)]["mae_C"] for d in DOMAINS]
            bars = ax_bar.bar(x + off, maes, width=w,
                              color=clr, alpha=alp,
                              hatch=hatch, edgecolor="white", lw=0.6,
                              label=lbl if row == 0 else "_")
            for b, v in zip(bars, maes):
                ax_bar.text(b.get_x()+b.get_width()/2,
                            b.get_height()+0.3,
                            f"{v:.1f}", ha="center", fontsize=7,
                            fontweight="bold", color=clr)

        ax_bar.axhline(10.0, color="#374151", lw=0.8, ls="--", alpha=0.3)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([DOM_SHORT[d] for d in DOMAINS], fontsize=9)
        ax_bar.set_ylim(0, 28)
        ax_bar.set_ylabel("Mean MAE [°C]", fontsize=9)
        ax_bar.set_title(f"({chr(97+row*3)})  {ARCH_LABEL[arch]} — MAE",
                         fontsize=9.5, fontweight="bold", color=c)
        ax_bar.spines[["top","right"]].set_visible(False)
        ax_bar.grid(axis="y", linestyle=":", alpha=0.3)
        ax_bar.set_axisbelow(True)
        if row == 0:
            ax_bar.legend(fontsize=7.5, ncol=2, loc="upper right")

        # ── Middle: Loss curves (skip=2, rectangular, v2) ────────
        loss_path = os.path.join(V2_DIR,
            f"rectangular_{arch}_skip2_loss.json")
        if os.path.exists(loss_path):
            with open(loss_path) as f:
                loss_data = json.load(f)

            cmap_w = plt.cm.turbo(np.linspace(0.1, 0.9, len(loss_data)))
            for wi, (hist, col_w) in enumerate(zip(loss_data, cmap_w)):
                ep = np.linspace(0, 1, len(hist["L_total"]))
                ax_loss.semilogy(ep, hist["L_total"], "-",
                                 color=col_w, lw=1.2, alpha=0.8,
                                 label=f"Win {wi+1}" if wi < 5 else "_")

            ax_loss.set_xlabel("Training progress (normalized)", fontsize=8.5)
            ax_loss.set_ylabel("Total Loss (log)", fontsize=8.5)
            ax_loss.set_title(f"({chr(98+row*3)})  {ARCH_LABEL[arch]} — Loss Curves\n"
                              f"(Rectangular, skip=2, all windows, color=window)",
                              fontsize=9, fontweight="bold", color=c)
            ax_loss.legend(fontsize=7, loc="upper right", ncol=2)
            ax_loss.grid(True, linestyle=":", alpha=0.3, which="both")
            ax_loss.spines[["top","right"]].set_visible(False)
        else:
            ax_loss.text(0.5, 0.5, "Loss data not found",
                         ha="center", va="center", transform=ax_loss.transAxes)
            ax_loss.axis("off")

        # ── Right: Heat map — best domain, skip=2, best window ───
        best_dom = min(DOMAINS,
                       key=lambda d: data_v2[d][arch]["2"]["mae_C"])
        slice_path = os.path.join(V2_DIR,
            f"{best_dom}_{arch}_skip2_slice.json")

        if os.path.exists(slice_path):
            with open(slice_path) as f:
                sd = json.load(f)

            xi   = np.array(sd["xi"])
            yi   = np.array(sd["yi"])
            wins = sd["windows"]
            # pick middle window
            mid_w = str(len(wins)//2)
            T_pred = np.array(wins[mid_w]["T_pred"])
            T_fem  = np.array(wins[mid_w]["T_fem"])
            err    = np.abs(T_pred - T_fem)
            xx, yy = np.meshgrid(xi, yi)

            norm_t = plt.Normalize(vmin=T_WATER, vmax=T_INIT)
            im = ax_heat.pcolormesh(xx, yy, T_pred, cmap="turbo",
                                    norm=norm_t, shading="auto")
            cs_fem  = ax_heat.contour(xx, yy, T_fem,  levels=7,
                                      colors="white", linewidths=0.5,
                                      linestyles="solid",  alpha=0.7)
            cs_pinn = ax_heat.contour(xx, yy, T_pred, levels=7,
                                      colors="#FF4444", linewidths=0.8,
                                      linestyles="dashed", alpha=0.9)

            mae_w  = float(np.nanmean(err))
            T_pmean= float(np.nanmean(T_pred))
            T_fmean= float(np.nanmean(T_fem))
            ax_heat.set_title(
                f"({chr(99+row*3)})  {ARCH_LABEL[arch]}  [{DOM_SHORT[best_dom]}]\n"
                f"FEM T̄={T_fmean:.0f}°C  PINN T̄={T_pmean:.0f}°C  MAE={mae_w:.1f}°C",
                fontsize=9, fontweight="bold", color=c)
            ax_heat.set_xlabel("x [m]", fontsize=8.5)
            ax_heat.set_ylabel("y [m]", fontsize=8.5)
            ax_heat.tick_params(labelsize=7)

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            div = make_axes_locatable(ax_heat)
            cax = div.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax).set_label("T [°C]", fontsize=8)

            from matplotlib.lines import Line2D
            leg_els = [
                Line2D([0],[0], color="white",   lw=1.0, label="PINN isotherms"),
                Line2D([0],[0], color="#FF4444", lw=1.0, ls="--", label="FEM isotherms"),
            ]
            ax_heat.legend(handles=leg_els, fontsize=7.5, loc="lower right")
        else:
            ax_heat.text(0.5, 0.5, "Slice data not found",
                         ha="center", va="center", transform=ax_heat.transAxes)
            ax_heat.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(RESULTS, "fig6_v2_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig6_v2_comparison.png")


def fig_v2_summary_table():
    """Extended summary table: v1 vs v2 side by side."""
    v2_json = os.path.join(V2_DIR, "results_3d_v2.json")
    if not os.path.exists(v2_json):
        print("  [skip] v2 results not found"); return

    with open(v2_json) as f:
        data_v2 = json.load(f)

    skips    = [2, 4]
    fem_used = {2: 11, 4: 6}
    fem_total= 21

    col_labels = [
        "Architecture",
        "2D MCO\n(ref.)",
        "3D v1\nRect.", "3D v1\nCyl.", "3D v1\nStack",
        "3D v2\nRect.", "3D v2\nCyl.", "3D v2\nStack",
    ]

    def _bg(val, lo=1.0, hi=25.0):
        # green (0.2,0.8,0.2) → amber (0.95,0.75,0.1) → red (0.9,0.1,0.1)
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
            v1   = {d: data_3d[d][arch][str(skip)]["mae_C"]  for d in DOMAINS}
            v2   = {d: data_v2[d][arch][str(skip)]["mae_C"]  for d in DOMAINS}

            rows.append([
                ARCH_LABEL[arch],
                f"{m2d:.2f}°C",
                f"{v1['rectangular']:.2f}°C",
                f"{v1['cylinder']:.2f}°C",
                f"{v1['stacked']:.2f}°C",
                f"{v2['rectangular']:.2f}°C",
                f"{v2['cylinder']:.2f}°C",
                f"{v2['stacked']:.2f}°C",
            ])
            vals = [m2d,
                    v1["rectangular"], v1["cylinder"], v1["stacked"],
                    v2["rectangular"], v2["cylinder"], v2["stacked"]]
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

        # v2 column headers in blue
        for j in range(5, 8):
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
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating figures ...\n")
    fig_thermal_fields()
    fig_mae_per_arch()
    fig_2d_per_arch()
    fig_summary_table()
    fig_skip_timeline()
    print("\nDone — figures saved to level8_nas_mco_pinn/results/")
