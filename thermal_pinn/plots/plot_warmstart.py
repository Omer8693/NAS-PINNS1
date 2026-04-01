"""
thermal_pinn/plots/plot_warmstart.py
=====================================
Visualise cold-start vs warm-start (ws_500ep) results.

Figures produced
----------------
1. fig_ws_speedup_heatmap_2d.png / fig_ws_speedup_heatmap_3d.png
   Grid: domain × arch, colour = speedup factor, annotated with value.

2. fig_ws_mae_delta_2d.png / fig_ws_mae_delta_3d.png
   Grid: domain × arch, colour = ΔMAE (warm − cold), per best-k only.

3. fig_ws_tradeoff_2d.png / fig_ws_tradeoff_3d.png
   Scatter: x = speedup, y = relative ΔMAE [%], one point per (domain,arch,k),
   coloured by arch.

4. fig_ws_kline_2d.png / fig_ws_kline_3d.png
   Line plot: mean MAE vs k, cold vs warm, one subplot per domain.

Output: results/warmstart/
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

OUT_DIR = RESULT_DIR / "warmstart"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DOMAINS_2D = ["rectangle", "circle", "lshape"]
DOMAINS_3D = ["rectangular", "cylinder", "stacked", "lshape"]
ARCHS      = ["bayesian", "nsga2", "nsga3"]
K_VALUES   = [1, 2, 3, 4, 5]

ARCH_SHORT = {"bayesian": "Bayesian", "nsga2": "NSGA-II", "nsga3": "NSGA-III"}
ARCH_COLOR = {"bayesian": "#1f77b4",  "nsga2": "#2ca02c",  "nsga3": "#ff7f0e"}
DOM_LABEL  = {
    "rectangle":   "Rect 2D",   "circle":      "Circle 2D",
    "lshape":      "L-shape 2D","rectangular": "Rect 3D",
    "cylinder":    "Cyl 3D",    "stacked":     "Stack 3D",
}


def _load(domain, arch, k, dim, tag=""):
    suffix = f"_{tag}" if tag else ""
    p = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}{suffix}_metrics.json"
    return json.load(open(p)) if p.exists() else None


def _runtime(m):
    return sum(w["runtime_s"] for w in m["windows"])


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Speedup heatmap  (best-k average per domain × arch)
# ══════════════════════════════════════════════════════════════════════════════

def _speedup_heatmap(domains, dim):
    tag = "2d" if dim == 2 else "3d"
    n_d = len(domains)
    n_a = len(ARCHS)

    speedup_grid = np.full((n_d, n_a), np.nan)
    for di, domain in enumerate(domains):
        for ai, arch in enumerate(ARCHS):
            vals = []
            for k in K_VALUES:
                c = _load(domain, arch, k, dim)
                w = _load(domain, arch, k, dim, "ws_500ep")
                if c and w:
                    vals.append(_runtime(c) / _runtime(w))
            if vals:
                speedup_grid[di, ai] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(6, max(3, n_d * 0.7 + 1.2)))
    im = ax.imshow(speedup_grid, cmap="YlGn", vmin=1.0, vmax=4.0, aspect="auto")

    ax.set_xticks(range(n_a))
    ax.set_xticklabels([ARCH_SHORT[a] for a in ARCHS], fontsize=10)
    ax.set_yticks(range(n_d))
    ax.set_yticklabels([DOM_LABEL.get(d, d) for d in domains], fontsize=10)

    for di in range(n_d):
        for ai in range(n_a):
            v = speedup_grid[di, ai]
            if not np.isnan(v):
                ax.text(ai, di, f"{v:.1f}×",
                        ha="center", va="center", fontsize=11,
                        fontweight="bold",
                        color="white" if v > 2.5 else "#1a1a1a")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Speedup factor (cold / warm runtime)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(f"Warm-Start Speedup — {'2D' if dim==2 else '3D'} Domains\n"
                 f"(averaged over k=1..5, warm=500 ep vs cold=800 ep)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / f"fig_ws_speedup_heatmap_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ΔMAE heatmap  (best-k per domain × arch)
# ══════════════════════════════════════════════════════════════════════════════

def _mae_delta_heatmap(domains, dim):
    tag = "2d" if dim == 2 else "3d"
    n_d = len(domains)
    n_a = len(ARCHS)

    delta_grid = np.full((n_d, n_a), np.nan)
    for di, domain in enumerate(domains):
        for ai, arch in enumerate(ARCHS):
            # use best k (min cold MAE)
            best_delta, best_mae = None, 1e9
            for k in K_VALUES:
                c = _load(domain, arch, k, dim)
                w = _load(domain, arch, k, dim, "ws_500ep")
                if c and w and c["mean_mae"] < best_mae:
                    best_mae  = c["mean_mae"]
                    best_delta = w["mean_mae"] - c["mean_mae"]
            if best_delta is not None:
                delta_grid[di, ai] = best_delta

    vabs = max(abs(np.nanmin(delta_grid)), abs(np.nanmax(delta_grid)), 1)
    fig, ax = plt.subplots(figsize=(6, max(3, n_d * 0.7 + 1.2)))
    im = ax.imshow(delta_grid, cmap="RdYlGn_r", vmin=-vabs, vmax=vabs, aspect="auto")

    ax.set_xticks(range(n_a))
    ax.set_xticklabels([ARCH_SHORT[a] for a in ARCHS], fontsize=10)
    ax.set_yticks(range(n_d))
    ax.set_yticklabels([DOM_LABEL.get(d, d) for d in domains], fontsize=10)

    for di in range(n_d):
        for ai in range(n_a):
            v = delta_grid[di, ai]
            if not np.isnan(v):
                sign = "+" if v >= 0 else ""
                ax.text(ai, di, f"{sign}{v:.1f}°C",
                        ha="center", va="center", fontsize=10,
                        fontweight="bold",
                        color="white" if abs(v) > vabs * 0.6 else "#1a1a1a")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ΔMAE = warm − cold [°C]  (green = warm better)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(f"Warm-Start MAE Change — {'2D' if dim==2 else '3D'} Domains\n"
                 f"(at best k per combination, warm=500 ep vs cold=800 ep)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / f"fig_ws_mae_delta_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Speedup vs ΔMAE scatter  (accuracy–speed trade-off)
# ══════════════════════════════════════════════════════════════════════════════

def _tradeoff_scatter(domains, dim):
    tag = "2d" if dim == 2 else "3d"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor("#F8F9FA")

    for arch in ARCHS:
        xs, ys = [], []
        for domain in domains:
            for k in K_VALUES:
                c = _load(domain, arch, k, dim)
                w = _load(domain, arch, k, dim, "ws_500ep")
                if not c or not w:
                    continue
                speedup    = _runtime(c) / _runtime(w)
                rel_delta  = (w["mean_mae"] - c["mean_mae"]) / c["mean_mae"] * 100
                xs.append(speedup)
                ys.append(rel_delta)
        ax.scatter(xs, ys, color=ARCH_COLOR[arch], label=ARCH_SHORT[arch],
                   s=60, alpha=0.75, edgecolors="white", linewidths=0.5, zorder=3)

    ax.axhline(0, color="gray", lw=1.0, ls="--", alpha=0.7)
    ax.axhspan(-100, 0, alpha=0.06, color="#2ca02c")   # green = warm better
    ax.text(1.05, -2, "warm better ↓", fontsize=8, color="#2ca02c")
    ax.text(1.05,  2, "warm worse ↑",  fontsize=8, color="#d7191c")

    ax.set_xlabel("Speedup (cold runtime / warm runtime)", fontsize=10)
    ax.set_ylabel("Relative ΔMAE [%]  (warm − cold) / cold", fontsize=10)
    ax.set_title(f"Accuracy–Speed Trade-off: Warm-Start 500 ep — {'2D' if dim==2 else '3D'}\n"
                 "Each point = one (domain, arch, k) combination",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / f"fig_ws_tradeoff_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MAE vs k line plot  (cold vs warm, one subplot per domain)
# ══════════════════════════════════════════════════════════════════════════════

def _kline_plot(domains, dim):
    tag    = "2d" if dim == 2 else "3d"
    n_d    = len(domains)
    n_cols = min(3, n_d)
    n_rows = (n_d + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 4.2, n_rows * 3.2),
                              sharey=False)
    axes = np.array(axes).flatten()

    for di, domain in enumerate(domains):
        ax = axes[di]
        ax.set_facecolor("#F8F9FA")

        for arch in ARCHS:
            cold_maes, warm_maes = [], []
            for k in K_VALUES:
                c = _load(domain, arch, k, dim)
                w = _load(domain, arch, k, dim, "ws_500ep")
                cold_maes.append(c["mean_mae"] if c else np.nan)
                warm_maes.append(w["mean_mae"] if w else np.nan)

            clr = ARCH_COLOR[arch]
            ax.plot(K_VALUES, cold_maes, color=clr, lw=2.0,
                    marker="o", ms=5, label=f"{ARCH_SHORT[arch]} cold")
            ax.plot(K_VALUES, warm_maes, color=clr, lw=1.5,
                    marker="s", ms=5, ls="--", alpha=0.7,
                    label=f"{ARCH_SHORT[arch]} warm")

        ax.set_title(DOM_LABEL.get(domain, domain), fontsize=10, fontweight="bold")
        ax.set_xlabel("k (FEM steps skipped)", fontsize=8.5)
        ax.set_ylabel("Mean MAE [°C]", fontsize=8.5)
        ax.set_xticks(K_VALUES)
        ax.grid(alpha=0.3, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    # hide unused subplots
    for ax in axes[n_d:]:
        ax.set_visible(False)

    # shared legend
    handles = []
    for arch in ARCHS:
        handles.append(plt.Line2D([0], [0], color=ARCH_COLOR[arch],
                                   lw=2, marker="o", ms=5,
                                   label=f"{ARCH_SHORT[arch]} cold"))
        handles.append(plt.Line2D([0], [0], color=ARCH_COLOR[arch],
                                   lw=1.5, ls="--", marker="s", ms=5,
                                   alpha=0.7,
                                   label=f"{ARCH_SHORT[arch]} warm-500ep"))
    fig.legend(handles=handles, loc="lower center",
               ncol=3, fontsize=8.5, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Mean MAE vs k — Cold-Start vs Warm-Start (500 ep) — {'2D' if dim==2 else '3D'}\n"
        "Solid = cold (800 ep)   Dashed = warm (500 ep)",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    out = OUT_DIR / f"fig_ws_kline_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("[Warm-start plots]")

    for dim, domains in [(2, DOMAINS_2D), (3, DOMAINS_3D)]:
        tag = "2D" if dim == 2 else "3D"
        print(f"\n  [{tag}]")
        _speedup_heatmap(domains, dim)
        _mae_delta_heatmap(domains, dim)
        _tradeoff_scatter(domains, dim)
        _kline_plot(domains, dim)

    print(f"\nDone → {OUT_DIR}/")
