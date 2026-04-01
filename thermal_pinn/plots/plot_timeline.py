"""
thermal_pinn/plots/plot_timeline.py
====================================
FEM–PINN–FEM–PINN timeline chart.

One figure per domain (7 total), showing the alternating
FEM call → PINN window → FEM call → ... pattern.

Layout:
  - 5 subplot rows  (k = 1 … 5)
  - 3 horizontal bars per row  (Bayesian / NSGA-II / NSGA-III)
  - Each coloured block = one PINN window
      · colour = L2 error level (green → yellow → orange → red)
      · text inside = L2 %
  - Thin grey vertical lines = FEM call points

Output: results/timeline/fig_timeline_{domain}_{dim}d.png

Usage:
    python -m thermal_pinn.plots.plot_timeline
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

warnings.filterwarnings("ignore")

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
from thermal_pinn.plots.plot_results import CKPT_DIR, RESULT_DIR

OUT_DIR = RESULT_DIR / "timeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DOMAINS_2D = ["rectangle", "circle", "lshape"]
DOMAINS_3D = ["rectangular", "cylinder", "stacked", "lshape"]
ARCHS      = ["bayesian", "nsga2", "nsga3"]

ARCH_SHORT = {"bayesian": "Bayesian", "nsga2": "NSGA-II", "nsga3": "NSGA-III"}
ARCH_COLOR = {"bayesian": "#1f77b4", "nsga2": "#2ca02c", "nsga3": "#ff7f0e"}
DOM_LABEL  = {
    "rectangle":  "Rectangle",  "circle":     "Circle",
    "lshape":     "L-shape",    "rectangular": "Rectangular",
    "cylinder":   "Cylinder",   "stacked":    "Stacked",
}

T_TOTAL   = 30.0
DT_FEM    = 1.5

# L2 colour map: 0 % → green,  5 % → yellow,  10 % → orange,  ≥ 15 % → red
CMAP = LinearSegmentedColormap.from_list(
    "l2", [(0, "#2ca02c"), (0.33, "#a6d96a"),
           (0.66, "#fdae61"), (1.0,  "#d7191c")])
L2_MAX = 15.0   # clamp for colormap


def _l2_color(l2_pct):
    return CMAP(min(l2_pct, L2_MAX) / L2_MAX)


def _load(domain, arch, k, dim):
    p = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}_metrics.json"
    return json.load(open(p)) if p.exists() else None


# ── Main figure builder ────────────────────────────────────────────────────────

def _timeline_figure(domain, dim):
    ks  = list(range(1, 6))
    n_k = len(ks)

    fig, axes = plt.subplots(n_k, 1,
                              figsize=(14, n_k * 2.2 + 1.2),
                              sharex=True)
    dl      = DOM_LABEL.get(domain, domain)
    dim_tag = "2D" if dim == 2 else "3D"

    fig.suptitle(
        f"FEM – PINN – FEM – PINN Timeline\n"
        f"{dl} ({dim_tag}) — colour = L2 error level inside each PINN window",
        fontsize=12, fontweight="bold")

    # Leave left space for arch labels, right space for k-labels
    fig.subplots_adjust(left=0.12, right=0.76, top=0.92, bottom=0.08,
                        hspace=0.55)

    BAR_H  = 0.55    # height of one arch bar
    GAP    = 0.10    # gap between arch bars
    BARS_TOP = (len(ARCHS) - 1) * (BAR_H + GAP) + BAR_H  # top of highest bar

    for ki, k in enumerate(ks):
        ax = axes[ki]
        ax.set_facecolor("#F8F9FA")

        dt_win = k * DT_FEM   # window duration

        # ── Draw FEM call markers ─────────────────────────────────────────
        fem_times = np.arange(0, T_TOTAL + dt_win * 0.5, dt_win)
        # Only label FEM lines if there aren't too many (k>=2 → <=10 lines)
        show_fem_labels = (k >= 2)
        for t in fem_times:
            if t <= T_TOTAL:
                ax.axvline(t, color="#374151", lw=1.2, zorder=4, alpha=0.7)
                if show_fem_labels:
                    ax.text(t, BARS_TOP + 0.08, "FEM",
                            ha="center", va="bottom", fontsize=6,
                            color="#374151", fontweight="bold",
                            clip_on=True)

        # ── Draw PINN window blocks for each arch ─────────────────────────
        for ai, arch in enumerate(ARCHS):
            m = _load(domain, arch, k, dim)
            if m is None:
                continue

            y_bot = (len(ARCHS) - 1 - ai) * (BAR_H + GAP)

            for wi, win in enumerate(m["windows"]):
                t0   = win["t_start"]
                t1   = win["t_end"]
                l2   = win["l2_rel"] * 100
                clr  = _l2_color(l2)

                ax.barh(y_bot + BAR_H / 2,
                        t1 - t0,
                        BAR_H,
                        left=t0,
                        color=clr,
                        edgecolor="white",
                        linewidth=0.6,
                        zorder=3)

                # Label inside block (only if wide enough)
                win_width_pts = (t1 - t0) / T_TOTAL * 14 * 72  # approx pts
                if win_width_pts > 28:
                    ax.text((t0 + t1) / 2, y_bot + BAR_H / 2,
                            f"{l2:.1f}%",
                            ha="center", va="center",
                            fontsize=max(5.0, min(7.5, win_width_pts / 9)),
                            color="white" if l2 > 6 else "#1a1a1a",
                            fontweight="bold", zorder=5)

            # Arch label on left (using axes transform so it's outside xlim)
            ax.text(-0.02, y_bot + BAR_H / 2,
                    ARCH_SHORT[arch],
                    ha="right", va="center",
                    fontsize=8.5, color=ARCH_COLOR[arch],
                    fontweight="bold",
                    transform=ax.get_yaxis_transform())

        # ── k-label on right (outside axes, using figure-relative position) ─
        n_wins = len(_load(domain, ARCHS[0], k, dim)["windows"]) if \
                 _load(domain, ARCHS[0], k, dim) else "?"
        # Use axes transform: x > 1 is to the right of the axes
        ax.text(1.015, 0.5,
                f"k = {k}\n({n_wins} PINN\nwindows)",
                ha="left", va="center",
                fontsize=8, color="#374151",
                transform=ax.transAxes)

        # ── Axis styling ──────────────────────────────────────────────────
        ax.set_xlim(0, T_TOTAL)
        ax.set_ylim(-0.15, BARS_TOP + 0.35)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", lw=0.4, alpha=0.4, color="#CBD5E0", zorder=1)

        if ki == 0:
            ax.set_title(
                f"  Δt_FEM = {DT_FEM} s   |   "
                "Each block = 1 PINN window   |   "
                "Vertical lines = FEM call points",
                fontsize=8.5, loc="left", color="#6B7280", pad=4)

    axes[-1].set_xlabel("Simulation time [s]", fontsize=9.5)

    # ── Colorbar — explicit position to avoid overlap with k-labels ──────
    cbar_ax = fig.add_axes([0.885, 0.12, 0.018, 0.75])
    sm = plt.cm.ScalarMappable(cmap=CMAP,
                                norm=plt.Normalize(vmin=0, vmax=L2_MAX))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("L2 error [%]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([0, 2, 5, 10, 15])

    # ── Arch legend at bottom centre ─────────────────────────────────────
    handles = [mpatches.Patch(color=ARCH_COLOR[a], label=ARCH_SHORT[a])
               for a in ARCHS]
    fig.legend(handles=handles,
               loc="lower center", ncol=3,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.44, 0.0))

    dim_tag_f = "2d" if dim == 2 else "3d"
    out = OUT_DIR / f"fig_timeline_{domain}_{dim_tag_f}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[Timeline plots]")
    for domain in DOMAINS_2D:
        _timeline_figure(domain, 2)
    for domain in DOMAINS_3D:
        _timeline_figure(domain, 3)
    print(f"\nDone → {OUT_DIR}/")
