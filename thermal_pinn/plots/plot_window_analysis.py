"""
thermal_pinn/plots/plot_window_analysis.py
==========================================
Two window-level visualisation types:

1. Likert-style stacked bar (per k value, 5 figures):
     For each k=1..5 — one figure showing all domains × archs.
     Each bar = one (domain, arch) combination.
     Stacked segments = fraction of windows in L2 error bands.
     Output: results/window/fig_likert_k{k}.png

2. Step plot (per domain, 7 figures):
     For each domain — one figure showing L2 error at every window
     as a step function over simulation time (0–30 s).
     One line per arch (3 lines per plot).
     Vertical dashed lines mark FEM step boundaries.
     Output: results/window/fig_step_{domain}_{dim}d.png

Usage:
    python -m thermal_pinn.plots.plot_window_analysis
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

warnings.filterwarnings("ignore")

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
from thermal_pinn.plots.plot_results import CKPT_DIR, RESULT_DIR

OUT_DIR = RESULT_DIR / "window"
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

# L2 error bands  (label, upper_bound, color)
BANDS = [
    ("< 2%",    2.0,  "#1a9641"),   # dark green
    ("2 – 5%",  5.0,  "#a6d96a"),   # light green
    ("5 – 10%", 10.0, "#fdae61"),   # amber/orange
    ("> 10%",   999,  "#d7191c"),   # red
]


# ── Data helper ────────────────────────────────────────────────────────────────

def _load(domain, arch, k, dim):
    p = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}_metrics.json"
    return json.load(open(p)) if p.exists() else None


def _band_fractions(windows):
    """Return list of fractions (one per band) for a window list."""
    l2s = [w["l2_rel"] * 100 for w in windows]
    n   = len(l2s)
    fracs = []
    lo = 0.0
    for _, hi, _ in BANDS:
        fracs.append(sum(1 for v in l2s if lo <= v < hi) / n)
        lo = hi
    return fracs


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LIKERT-STYLE STACKED BAR — one figure per k value
# ══════════════════════════════════════════════════════════════════════════════

def _likert_figure(k):
    """
    Horizontal stacked bar chart for a single k value.
    Rows = (domain, arch) combos; segments = L2 error band fractions.
    Style follows the Likert / diverging-bar convention.
    """
    # Build row labels and fraction arrays
    all_domains = [(d, 2) for d in DOMAINS_2D] + [(d, 3) for d in DOMAINS_3D]

    row_labels = []
    row_fracs  = []
    row_colors_bar = []   # colour for each row (arch colour, slightly muted)

    for domain, dim in all_domains:
        for arch in ARCHS:
            m = _load(domain, arch, k, dim)
            if m is None:
                continue
            dl  = DOM_LABEL.get(domain, domain)
            dim_tag = f"{'2D' if dim==2 else '3D'}"
            row_labels.append(f"{dl} {dim_tag} — {ARCH_SHORT[arch]}")
            row_fracs.append(_band_fractions(m["windows"]))
            row_colors_bar.append(ARCH_COLOR[arch])

    n_rows = len(row_labels)
    if n_rows == 0:
        return

    fig, ax = plt.subplots(figsize=(11, max(5, n_rows * 0.45 + 1.5)))
    fig.patch.set_facecolor("white")

    y_pos  = np.arange(n_rows)
    bar_h  = 0.62

    # Draw stacked bars left → right
    lefts = np.zeros(n_rows)
    for bi, (label, _, color) in enumerate(BANDS):
        widths = np.array([row_fracs[i][bi] * 100 for i in range(n_rows)])
        bars = ax.barh(y_pos, widths, bar_h, left=lefts,
                       color=color, edgecolor="white", linewidth=0.5,
                       label=label, zorder=3)
        # value labels (hide very small segments)
        for bar, w in zip(bars, widths):
            if w >= 8:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{w:.0f}",
                        ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        lefts += widths

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Fraction of windows in each L2 error band [%]", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_title(
        f"Window-Level L2 Error Distribution — k = {k}  "
        f"(Δt_window = {k * 1.5:.1f} s,  {20 // k if 20 % k == 0 else 20 // k + 1} windows)",
        fontsize=11, fontweight="bold")
    ax.axvline(50, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=2)
    ax.grid(axis="x", alpha=0.3, lw=0.5, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()   # top → bottom order

    # Separator lines between domain groups
    ri = 0
    for domain, dim in all_domains:
        if ri > 0:
            ax.axhline(ri - 0.5, color="gray", lw=0.6, ls="-", alpha=0.4, zorder=2)
        ri += sum(1 for arch in ARCHS
                  if (_load(domain, arch, k, dim)) is not None)

    # Legend (top, horizontal)
    handles = [mpatches.Patch(color=c, label=l) for l, _, c in BANDS]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, 1.07), ncol=4,
              fontsize=9, framealpha=0.9, edgecolor="#CBD5E0")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / f"fig_likert_k{k}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out.name}")


def fig_likert_all_k():
    for k in range(1, 6):
        _likert_figure(k)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  STEP PLOT — L2 error over time, one figure per domain
# ══════════════════════════════════════════════════════════════════════════════

def _step_figure(domain, dim):
    """
    Step function: L2 error at each window boundary over simulation time.
    One line per arch (3 arches), one subplot per k value.
    """
    ks = list(range(1, 6))
    fig, axes = plt.subplots(len(ks), 1,
                              figsize=(10, len(ks) * 2.0 + 0.8),
                              sharex=True)
    dl       = DOM_LABEL.get(domain, domain)
    dim_tag  = "2D" if dim == 2 else "3D"
    fig.suptitle(
        f"L2 Error per Window — {dl} ({dim_tag})\n"
        "Step function over simulation time  (window boundaries = FEM call points)",
        fontsize=11, fontweight="bold", y=1.01)

    for ki, k in enumerate(ks):
        ax  = axes[ki]
        ax.set_facecolor("#F7FAFC")
        ax.set_ylabel(f"k={k}\nL2 [%]", fontsize=8.5, labelpad=2)

        n_windows_k = int(np.ceil(30.0 / (k * 1.5)))
        any_data = False

        for arch in ARCHS:
            m = _load(domain, arch, k, dim)
            if m is None:
                continue
            wins = m["windows"]
            any_data = True

            # Build step data: t values and L2 values
            # Step starts at t_start and holds until t_end
            t_step = []
            l2_step = []
            for w in wins:
                t_step  += [w["t_start"], w["t_end"]]
                l2_val   = w["l2_rel"] * 100
                l2_step += [l2_val, l2_val]

            clr = ARCH_COLOR[arch]
            ax.step(t_step, l2_step, where="post",
                    color=clr, lw=1.8, label=ARCH_SHORT[arch], zorder=3)
            # mark window ends with dots
            t_ends  = [w["t_end"] for w in wins]
            l2_ends = [w["l2_rel"] * 100 for w in wins]
            ax.plot(t_ends, l2_ends, "o",
                    color=clr, ms=4.5, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.5)

        # FEM step boundary lines (Δt=1.5s)
        for t in np.arange(1.5, 30.1, 1.5):
            ax.axvline(t, color="#CBD5E0", lw=0.5, ls=":", zorder=1, alpha=0.7)

        # Window boundary lines (thicker, coloured by k)
        for t in np.arange(k * 1.5, 30.1, k * 1.5):
            ax.axvline(t, color="#94A3B8", lw=1.0, ls="--", zorder=2, alpha=0.5)

        ax.axhline(5.0, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.text(30.2, 5.1, "5%", fontsize=7, color="gray", va="bottom")
        ax.axhspan(0, 5, alpha=0.06, color="#2ca02c", zorder=0)

        ax.set_xlim(-0.5, 31)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

        if ki == 0:
            ax.legend(fontsize=8, loc="upper right",
                      framealpha=0.9, ncol=3)

    axes[-1].set_xlabel("Simulation time [s]", fontsize=9)

    # Shared annotation
    fig.text(0.5, -0.02,
             "Dots = window end-points (FEM call points)  ·  "
             "Dashed lines = window boundaries  ·  "
             "Dotted lines = individual FEM steps (Δt=1.5 s)  ·  "
             "Green shading = L2 < 5%",
             ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    dim_tag_file = "2d" if dim == 2 else "3d"
    out = OUT_DIR / f"fig_step_{domain}_{dim_tag_file}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out.name}")


def fig_step_all_domains():
    for domain in DOMAINS_2D:
        _step_figure(domain, 2)
    for domain in DOMAINS_3D:
        _step_figure(domain, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("[Window analysis plots]")

    print("  [1/2] Likert stacked bars (k=1..5) …")
    fig_likert_all_k()

    print("  [2/2] Step plots (per domain) …")
    fig_step_all_domains()

    print(f"\nDone → {OUT_DIR}/")
