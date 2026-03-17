"""
plot_level7b.py — Level 7B Visualization
==========================================
FEM-PINNS style: cream background, YlOrRd heatmaps, clean layout.

Outputs (saved to results/level7b/plots/):
  l7b_solution_<domain>.png   — solution heatmaps for each domain (3 optimizers)
  l7b_l2_comparison.png       — L2 bar chart: 3 domains × 3 optimizers
  l7b_summary_table.png       — full comparison table
  l7b_convergence_<domain>.png — val-L2 convergence curves
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from level7_multiDomain.src.domains import get_domain
from level7_multiDomain.src.poisson_pinn import PoissonNet, eval_on_grid

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOMAINS    = ["circle", "lshape", "flower"]
OPTIMIZERS = ["bayesian", "nsga2", "nsga3"]
OPT_LABELS = {"bayesian": "Bayesian", "nsga2": "NSGA-II", "nsga3": "NSGA-III"}
OPT_COLORS = {"bayesian": "#1565C0", "nsga2": "#2E7D32", "nsga3": "#E65100"}
DOM_LABELS = {"circle": "Circle  (exact u)", "lshape": "L-Shape", "flower": "Flower (5 petals)"}

L7B_DIR = _ROOT / "results" / "level7b"
OUT_DIR = L7B_DIR / "plots"

# FEM-PINNS style
_FIG_BG  = "#f7f1e3"
_AX_BG   = "#fffdf8"
_SPINE_C = "#d8cfbf"
_TXT_C   = "#2f2a24"
_TICK_C  = "#4c463f"
_BOX_FC  = "#fef6e4"
_BOX_EC  = "#e8c77a"


def _style(ax):
    ax.set_facecolor(_AX_BG)
    for sp in ax.spines.values():
        sp.set_color(_SPINE_C)
        sp.set_linewidth(0.8)
    ax.tick_params(colors=_TICK_C, labelsize=8)
    ax.xaxis.label.set_color(_TICK_C)
    ax.yaxis.label.set_color(_TICK_C)


def _box(ax, text, loc="top"):
    y, va = (0.96, "top") if loc == "top" else (0.04, "bottom")
    ax.text(0.03, y, text, transform=ax.transAxes, va=va, fontsize=8,
            color=_TXT_C,
            bbox={"facecolor": _BOX_FC, "edgecolor": _BOX_EC,
                  "boxstyle": "round,pad=0.25"})


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
    model.load_state_dict(torch.load(p, map_location="cpu"))
    model.eval()
    return model


# ── Figure 1: Solution heatmaps for one domain ───────────────────────────────

def plot_solution_domain(domain_name: str):
    results = {}
    models  = {}
    for opt in OPTIMIZERS:
        r = _load_result(domain_name, opt)
        if r:
            results[opt] = r
            try:
                models[opt] = _load_model(domain_name, opt, r)
            except Exception as e:
                print(f"  [{domain_name}/{opt}] Cannot load model: {e}")

    valid = [opt for opt in OPTIMIZERS if opt in models]
    if not valid:
        print(f"  [{domain_name}] No data, skipping.")
        return

    domain = get_domain(domain_name, seed=0)
    GRID_N = 80

    # Collect all grids for unified colorbar
    grids = {}
    for opt in valid:
        xx, yy, uu = eval_on_grid(models[opt], domain, n=GRID_N)
        grids[opt] = (xx, yy, uu)

    all_vals = np.concatenate([uu[~np.isnan(uu)].flatten()
                                for _, _, uu in grids.values()])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    ncols = len(valid)
    fig, axes = plt.subplots(1, ncols,
                              figsize=(4.5 * ncols + 1.2, 4.5),
                              facecolor=_FIG_BG, constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    im_last = None
    for ax, opt in zip(axes, valid):
        xx, yy, uu = grids[opt]
        r = results[opt]
        im_last = ax.pcolormesh(xx, yy, uu,
                                 cmap="YlOrRd", vmin=vmin, vmax=vmax,
                                 shading="auto", rasterized=True)
        _style(ax)
        ax.set_aspect("equal")
        ax.set_title(OPT_LABELS[opt], fontsize=11, color=_TXT_C, pad=6)
        ax.set_xlabel("x")
        ax.set_ylabel("y" if opt == valid[0] else "")

        arch = r["best_arch"]
        l2   = r["final_l2"]
        info = (f"{arch['n_layers']}×{arch['neurons']} {arch['activation']}\n"
                f"L2 = {l2:.5f}")
        if r.get("has_exact"):
            info += " (exact ref)"
        _box(ax, info)

    cbar = fig.colorbar(im_last, ax=axes[-1] if ncols > 1 else axes[0],
                         shrink=0.85, pad=0.02)
    cbar.set_label("u(x,y)", color=_TXT_C)
    cbar.ax.tick_params(labelsize=8, colors=_TICK_C)

    fig.suptitle(
        f"Poisson NAS-PINN — {DOM_LABELS[domain_name]}\n"
        f"−Δu = 1  on Ω,  u = 0  on ∂Ω",
        fontsize=12, color=_TXT_C,
    )

    out = OUT_DIR / f"l7b_solution_{domain_name}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 2: L2 comparison bar chart ────────────────────────────────────────

def plot_l2_comparison():
    data = {}
    for dom in DOMAINS:
        for opt in OPTIMIZERS:
            r = _load_result(dom, opt)
            if r:
                data[(dom, opt)] = r["final_l2"]

    if not data:
        print("  No data for L2 comparison, skipping.")
        return

    x     = np.arange(len(DOMAINS))
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_FIG_BG, constrained_layout=True)
    _style(ax)

    for i, opt in enumerate(OPTIMIZERS):
        vals = [data.get((dom, opt), np.nan) for dom in DOMAINS]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=OPT_LABELS[opt], color=OPT_COLORS[opt],
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=7,
                        color=_TXT_C)

    ax.set_xticks(x)
    ax.set_xticklabels([DOM_LABELS[d] for d in DOMAINS], fontsize=10)
    ax.set_ylabel("Final L₂ Error", fontsize=10)
    ax.set_title(
        "Multi-Domain Poisson NAS — L₂ Comparison\n"
        "(Circle: relative L₂ vs exact;  L-Shape / Flower: RMS prediction)",
        fontsize=11, color=_TXT_C, pad=8,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    out = OUT_DIR / "l7b_l2_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 3: Summary table ───────────────────────────────────────────────────

def plot_summary_table():
    rows = []
    for dom in DOMAINS:
        for opt in OPTIMIZERS:
            r = _load_result(dom, opt)
            if not r:
                continue
            arch = r["best_arch"]
            arch_str = f"{arch['n_layers']}×{arch['neurons']} {arch['activation']}"
            exact_str = "Yes" if r.get("has_exact") else "No"
            rows.append([
                DOM_LABELS[dom],
                OPT_LABELS[opt],
                arch_str,
                f"{r['n_params']:,}",
                f"{r['final_l2']:.5f}",
                f"{r['final_mae']:.5f}",
                exact_str,
                f"{r['search_s']:.0f}s / {r['train_s']:.0f}s",
            ])

    if not rows:
        print("  No data for summary table, skipping.")
        return

    col_labels = ["Domain", "Optimizer", "Architecture", "Params",
                  "L₂ Error", "MAE", "Exact?", "Search / Train"]

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(15, 0.55 * (n_rows + 2) + 1.5),
                            facecolor=_FIG_BG)
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    tbl.auto_set_column_width(range(len(col_labels)))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1565C0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    dom_colors = {"Circle": "#E3F2FD", "L-Shape": "#E8F5E9", "Flower": "#FFF8E1"}
    for i, row in enumerate(rows, 1):
        dom_key = row[0].split()[0]
        bg = dom_colors.get(dom_key, "#F5F5F5")
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)

    ax.set_title(
        "Level 7B — Multi-Domain Poisson NAS Benchmark\n"
        "NAS Architecture Search: 3 domains × 3 optimizers",
        fontsize=11, color=_TXT_C, pad=12,
    )

    out = OUT_DIR / "l7b_summary_table.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 4: Convergence curves ─────────────────────────────────────────────

def plot_convergence(domain_name: str):
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=_FIG_BG, constrained_layout=True)
    _style(ax)

    has_data = False
    for opt in OPTIMIZERS:
        r = _load_result(domain_name, opt)
        if not r or not r.get("val_l2_history"):
            continue
        epochs = [ep for ep, _ in r["val_l2_history"]]
        vals   = [v  for _, v  in r["val_l2_history"]]
        ax.plot(epochs, vals, color=OPT_COLORS[opt], linewidth=2.0,
                label=f"{OPT_LABELS[opt]}  (final L2={r['final_l2']:.5f})")
        has_data = True

    if not has_data:
        plt.close()
        return

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Validation L₂", fontsize=10)
    ax.set_title(
        f"Convergence — {DOM_LABELS[domain_name]}\n"
        f"Final training ({5000} epochs), 3 optimizers",
        fontsize=11, color=_TXT_C, pad=8,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    out = OUT_DIR / f"l7b_convergence_{domain_name}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 5: Circle exact vs predicted ──────────────────────────────────────

def plot_circle_exact_vs_pred():
    """For the circle domain: show exact solution vs best PINN side-by-side."""
    # Find best optimizer for circle
    best_opt, best_l2 = None, float("inf")
    best_result = None
    for opt in OPTIMIZERS:
        r = _load_result("circle", opt)
        if r and r["final_l2"] < best_l2:
            best_l2  = r["final_l2"]
            best_opt = opt
            best_result = r

    if best_opt is None:
        return

    try:
        model = _load_model("circle", best_opt, best_result)
    except Exception as e:
        print(f"  [circle/{best_opt}] Cannot load model: {e}")
        return

    domain = get_domain("circle", seed=0)
    GRID_N = 80

    xx, yy, uu_pred = eval_on_grid(model, domain, n=GRID_N)

    # Exact solution
    xy_flat = np.stack([xx.flatten(), yy.flatten()], axis=1).astype(np.float32)
    inside  = domain.is_inside(xy_flat)
    u_exact = domain.u_exact(xy_flat)
    u_exact[~inside] = np.nan
    uu_exact = u_exact.reshape(GRID_N, GRID_N)

    # Error
    uu_err = np.abs(uu_pred - uu_exact)

    vmin = float(np.nanmin(uu_exact))
    vmax = float(np.nanmax(uu_exact))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                              facecolor=_FIG_BG, constrained_layout=True)

    panels = [
        (uu_exact, "Exact Solution  u = (1−r²)/4", "YlOrRd", vmin, vmax),
        (uu_pred,  f"NAS-PINN ({OPT_LABELS[best_opt]})", "YlOrRd", vmin, vmax),
        (uu_err,   "Absolute Error |u_pred − u_exact|", "Blues",
         0, float(np.nanmax(uu_err))),
    ]

    for ax, (data, title, cmap, v0, v1) in zip(axes, panels):
        im = ax.pcolormesh(xx, yy, data, cmap=cmap, vmin=v0, vmax=v1,
                            shading="auto", rasterized=True)
        _style(ax)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10, color=_TXT_C, pad=5)
        ax.set_xlabel("x")
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02).ax.tick_params(
            labelsize=7, colors=_TICK_C)

    axes[0].set_ylabel("y")

    arch = best_result["best_arch"]
    fig.suptitle(
        f"Circle Domain — Exact vs NAS-PINN  |  "
        f"Arch: {arch['n_layers']}×{arch['neurons']} {arch['activation']}  |  "
        f"Relative L₂ = {best_l2:.5f}",
        fontsize=11, color=_TXT_C,
    )

    out = OUT_DIR / "l7b_circle_exact_vs_pred.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Level 7B — Plot Results")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*60}\n")

    for dom in DOMAINS:
        plot_solution_domain(dom)

    plot_l2_comparison()
    plot_summary_table()

    for dom in DOMAINS:
        plot_convergence(dom)

    plot_circle_exact_vs_pred()

    print(f"\n  All figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
