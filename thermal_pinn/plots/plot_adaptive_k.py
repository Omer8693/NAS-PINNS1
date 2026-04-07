"""
thermal_pinn/plots/plot_adaptive_k.py
=======================================
Publication-quality figures for the Adaptive-k results.

Figure 1 — k-Schedule + MAE timeline (per domain/arch)
    Top panel:    step plot of k(t) over the simulation
    Bottom panel: MAE(°C) per window, with thresh_up/down bands

Figure 2 — Summary comparison across all available adaptive runs
    Bar chart: Fixed best-k vs Adaptive k  [MAE | FEM savings]

Usage
-----
    python -m thermal_pinn.plots.plot_adaptive_k               # all available runs
    python -m thermal_pinn.plots.plot_adaptive_k --domain rectangle --dim 2
    python -m thermal_pinn.plots.plot_adaptive_k --no_summary   # skip Fig 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_ROOT      = Path(__file__).resolve().parent.parent
CKPT_DIR   = _ROOT / "checkpoints"
RESULT_DIR = _ROOT / "results" / "adaptive_k"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style (matches rest of thesis figures) ─────────────────────────────────
NAVY  = "#1a237e"
BLUE  = "#1565c0"
RED   = "#b71c1c"
GREEN = "#1b5e20"
AMBER = "#e65100"
MUTED = "#555555"

ARCH_LABELS = {"bayesian": "Bayesian (TPE)", "nsga2": "NSGA-II", "nsga3": "NSGA-III"}
DOMAIN_LABELS = {
    "rectangle":  "Rectangle 2D",  "circle":   "Circle 2D",
    "lshape":     "L-Shape 2D",    "rectangular": "Rectangular 3D",
    "cylinder":   "Cylinder 3D",   "stacked":  "Stacked 3D",
    "lshape3d":   "L-Shape 3D",
}

K_COLORS = {1: "#b71c1c", 2: "#e65100", 3: "#1565c0", 4: "#1b5e20", 5: "#4a148c"}
K_LABELS = {1: "k = 1  (Δt = 1.5 s)", 2: "k = 2  (Δt = 3.0 s)",
            3: "k = 3  (Δt = 4.5 s)", 4: "k = 4  (Δt = 6.0 s)",
            5: "k = 5  (Δt = 7.5 s)"}


def load_adaptive(domain: str, arch: str, dim: int) -> dict | None:
    # handle lshape 3D stored as "lshape" with dim=3
    tag  = f"{domain}_{arch}_adaptive_dim{dim}"
    path = CKPT_DIR / f"{tag}_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_registry() -> list[dict]:
    reg = CKPT_DIR / "registry.json"
    if not reg.exists():
        return []
    with open(reg) as f:
        return json.load(f)


def plot_timeline(data: dict, save: bool = True) -> plt.Figure:
    """
    Publication timeline matching the 03_timeline style:
      - barh PINN blocks coloured by k, MAE inside each block
      - solid FEM lines at used checkpoints, dashed at skipped ones
      - MAE bar panel below with threshold bands
    """
    from matplotlib.lines import Line2D

    windows     = data["windows"]
    t_starts    = [w["t_start"] for w in windows]
    t_ends      = [w["t_end"]      for w in windows]
    k_vals      = [w["k_used"]     for w in windows]
    maes        = [w["mae_C"]      for w in windows]
    runtimes    = [w.get("runtime_s", 0.0) for w in windows]
    total_rt    = data.get("total_s", sum(runtimes))
    thresh_up   = data.get("thresh_up",   2.0)
    thresh_down = data.get("thresh_down", 5.0)

    domain_lbl = DOMAIN_LABELS.get(data["domain"], data["domain"])
    if data["domain"] == "lshape":
        domain_lbl = f"L-Shape {'3D' if data['dim'] == 3 else '2D'}"
    arch_lbl = ARCH_LABELS.get(data["arch"], data["arch"])

    DT_FEM    = 1.5
    N_FEM_MAX = 20
    T_TOTAL   = 30.0
    FEM_C     = "#374151"        # dark grey (same as 03_timeline)
    BAR_H     = 0.55             # barh height (matches 03_timeline)

    fem_used    = sorted({0.0} | set(t_ends))          # window boundaries
    fem_all     = [i * DT_FEM for i in range(N_FEM_MAX + 1)]  # all possible
    fem_skipped = [t for t in fem_all if t not in fem_used]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 6.0),
        gridspec_kw={"height_ratios": [1.6, 1.1, 1.1]},
        sharex=True,
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.06, right=0.88, top=0.88, bottom=0.10, hspace=0.45)

    # ── Panel 1: 20 equal 1.5 s slots, coloured by window membership ──────────
    # Each of the 20 FEM steps is drawn as a 1.5 s bar.
    # Steps within the same PINN window share the same k-colour.
    # FEM boundaries (window edges) = solid dark line.
    # Inner step boundaries = thin white divider only.
    ax1.set_facecolor("#F8F9FA")
    y_bot = 0.0

    N_STEPS   = int(round(T_TOTAL / DT_FEM))   # 20
    fem_times = sorted({0.0} | set(t_ends))

    # Map each 1.5 s step to its window
    step_window = {}   # step_i → (k, mae, win_idx)
    for win_i, (w0, w1, k, mae) in enumerate(zip(t_starts, t_ends, k_vals, maes)):
        n_steps = int(round((w1 - w0) / DT_FEM))
        first   = int(round(w0 / DT_FEM))
        for s in range(first, first + n_steps):
            step_window[s] = (k, mae, win_i)

    for step_i in range(N_STEPS):
        t_lo = step_i * DT_FEM
        k_s, mae_s, win_i = step_window.get(step_i, (1, 0, -1))
        clr = K_COLORS.get(k_s, BLUE)

        ax1.barh(y_bot + BAR_H / 2, DT_FEM, BAR_H, left=t_lo,
                 color=clr, edgecolor="white", linewidth=0.5,
                 alpha=0.88, zorder=3)

    # FEM boundary lines (window edges) — solid, dark
    for ft in fem_times:
        ax1.axvline(ft, color=FEM_C, lw=1.6, zorder=6, alpha=0.9)
        ax1.text(ft, BAR_H + 0.05, f"FEM\n{ft:.1f}s",
                 ha="center", va="bottom", fontsize=5.5,
                 color=FEM_C, fontweight="bold", clip_on=True)

    # Window k + MAE + runtime label in centre of each window block
    for t0, t1, k, mae, rt in zip(t_starts, t_ends, k_vals, maes, runtimes):
        w   = t1 - t0
        pts = w / T_TOTAL * 14 * 72
        if pts > 18:
            ax1.text((t0 + t1) / 2, y_bot + BAR_H / 2,
                     f"k={k}\n{mae:.1f}°C\n{rt:.0f}s",
                     ha="center", va="center",
                     fontsize=max(5.0, min(7.0, pts / 11)),
                     color="white", fontweight="bold", zorder=5)

    ax1.set_xlim(0, T_TOTAL)
    ax1.set_ylim(-0.10, BAR_H + 0.52)
    ax1.set_yticks([])
    ax1.set_xticks([i * DT_FEM for i in range(N_STEPS + 1)])
    ax1.set_xticklabels([f"{i*DT_FEM:.1f}" for i in range(N_STEPS + 1)],
                        fontsize=5.5, rotation=45)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.grid(axis="x", lw=0.3, alpha=0.25, color="#CBD5E0", zorder=1)

    ax1.set_title(
        f"Δt_FEM = {DT_FEM} s  |  20 equal FEM steps  |"
        f"  Each step coloured by PINN window k-value  |  Solid lines = FEM called",
        fontsize=8, loc="left", color="#6B7280", pad=4)

    # Build info text, add speedup if k=1 data available
    _k1p = CKPT_DIR / f"{data['domain']}_{data['arch']}_k1_dim{data['dim']}_metrics.json"
    if _k1p.exists():
        import json as _j
        _k1d    = _j.load(open(_k1p))
        _k1_tot = _k1d.get("total_s", 0)
        _spdup  = _k1_tot / total_rt if total_rt > 0 else 1.0
        _rt_txt = (f"runtime: {total_rt:.0f}s\n"
                   f"k=1 ref: {_k1_tot:.0f}s\n"
                   f"speedup: {_spdup:.1f}×")
    else:
        _rt_txt = f"runtime: {total_rt:.0f}s"

    ax1.text(1.01, 0.5,
             f"n = {data['n_windows']} windows\n"
             f"FEM saved: {data['fem_savings_pct']:.0f}%\n"
             f"mean MAE: {data['mean_mae']:.2f}°C\n"
             f"{_rt_txt}",
             ha="left", va="center", fontsize=8, color="#374151",
             transform=ax1.transAxes)

    ks_used  = sorted(set(k_vals))
    patches  = [mpatches.Patch(color=K_COLORS[k], alpha=0.88, label=K_LABELS[k])
                for k in ks_used]
    fem_line = matplotlib.lines.Line2D([0], [0], color=FEM_C, lw=1.5,
                                        label="FEM boundary")
    ax1.legend(handles=patches + [fem_line],
               loc="upper right", bbox_to_anchor=(0.99, 1.05),
               fontsize=7, framealpha=0.9, edgecolor=MUTED,
               ncol=len(ks_used) + 1)

    fig.suptitle(
        f"Adaptive k  —  {domain_lbl} / {arch_lbl}",
        fontsize=11, fontweight="bold", color=NAVY)

    # ── Panel 2: MAE bars on physical time ────────────────────────────────────
    ax2.set_facecolor("#F8F9FA")

    for t0, t1, k, mae in zip(t_starts, t_ends, k_vals, maes):
        w = t1 - t0
        ax2.bar((t0 + t1) / 2, mae, width=w * 0.80,
                color=K_COLORS.get(k, BLUE), alpha=0.82,
                edgecolor="white", linewidth=0.5, zorder=3)
    for ft in fem_times:
        ax2.axvline(ft, color=FEM_C, lw=1.0, alpha=0.4, zorder=2)

    ax2.axhline(thresh_up,        color=GREEN, lw=1.3, ls="--",
                label=f"thresh↑ = {thresh_up}°C")
    ax2.axhline(thresh_down,      color=RED,   lw=1.3, ls="--",
                label=f"thresh↓ = {thresh_down}°C")
    ax2.axhline(data["mean_mae"], color=NAVY,  lw=1.2, ls=":",
                label=f"mean = {data['mean_mae']:.2f}°C")

    ax2.fill_between([0, T_TOTAL], 0,          thresh_up,  color=GREEN, alpha=0.07)
    y_top = max(max(maes) * 1.20, thresh_down * 1.25)
    ax2.fill_between([0, T_TOTAL], thresh_down, y_top,     color=RED,   alpha=0.07)
    ax2.set_ylim(0, y_top)
    ax2.set_xlim(0, T_TOTAL)
    ax2.set_xlabel("Physical simulation time  (s)", fontsize=9)
    ax2.set_ylabel("MAE  (°C)", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(fontsize=7.5, framealpha=0.9, edgecolor=MUTED, ncol=3,
               loc="upper right")
    ax2.tick_params(labelsize=8)
    ax2.grid(axis="x", lw=0.4, alpha=0.4, color="#CBD5E0", zorder=1)

    # ── Panel 3: Runtime per window + k=1 FEM reference ──────────────────────
    ax3.set_facecolor("#F8F9FA")

    mid_times  = [(t0 + t1) / 2 for t0, t1 in zip(t_starts, t_ends)]
    widths     = [t1 - t0        for t0, t1 in zip(t_starts, t_ends)]
    bar_colors = [K_COLORS.get(k, BLUE) for k in k_vals]

    ax3.bar(mid_times, runtimes, width=[w * 0.80 for w in widths],
            color=bar_colors, alpha=0.82, edgecolor="white",
            linewidth=0.5, zorder=3)

    # Value labels on bars
    for mid, rt, w in zip(mid_times, runtimes, widths):
        pts = w / T_TOTAL * 14 * 72
        if pts > 16:
            ax3.text(mid, rt + 0.3, f"{rt:.0f}s",
                     ha="center", va="bottom", fontsize=6,
                     color="#374151", fontweight="bold")

    for ft in fem_times:
        ax3.axvline(ft, color=FEM_C, lw=1.0, alpha=0.4, zorder=2)

    mean_rt = sum(runtimes) / len(runtimes)
    ax3.axhline(mean_rt, color=NAVY, lw=1.2, ls=":",
                label=f"adaptive mean = {mean_rt:.0f}s/window")

    # k=1 baseline reference: load from checkpoints if available
    k1_path = CKPT_DIR / f"{data['domain']}_{data['arch']}_k1_dim{data['dim']}_metrics.json"
    if k1_path.exists():
        import json as _json
        k1_data   = _json.load(open(k1_path))
        k1_total  = k1_data.get("total_s", 0)
        k1_mean   = k1_total / k1_data.get("n_windows", 20)
        speedup   = k1_total / total_rt if total_rt > 0 else 1.0

        ax3.axhline(k1_mean, color=RED, lw=1.2, ls="--",
                    label=f"k=1 mean = {k1_mean:.0f}s/window  (total {k1_total:.0f}s)")

        # Speedup annotation in top-right corner
        ax3.text(0.99, 0.95,
                 f"Total: adaptive {total_rt:.0f}s  vs  k=1 {k1_total:.0f}s  →  {speedup:.1f}× faster",
                 ha="right", va="top", fontsize=7.5,
                 color=GREEN, fontweight="bold",
                 transform=ax3.transAxes)

        y_top = max(max(runtimes), k1_mean) * 1.35
    else:
        y_top = max(runtimes) * 1.30

    ax3.set_xlabel("Physical simulation time  (s)", fontsize=9)
    ax3.set_ylabel("Runtime  (s/window)", fontsize=9)
    ax3.set_xlim(0, T_TOTAL)
    ax3.set_ylim(0, y_top)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.legend(fontsize=7.5, framealpha=0.9, edgecolor=MUTED, loc="upper left")
    ax3.tick_params(labelsize=8)
    ax3.grid(axis="x", lw=0.4, alpha=0.4, color="#CBD5E0", zorder=1)

    fig.subplots_adjust(left=0.06, right=0.88, top=0.91, bottom=0.08, hspace=0.45)

    if save:
        tag  = f"{data['domain']}_{data['arch']}_adaptive_dim{data['dim']}"
        path = RESULT_DIR / f"{tag}_timeline.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)

    return fig


def plot_summary(all_adaptive: list[dict], save: bool = True) -> plt.Figure:
    """
    Three-way bar chart: Fixed k=1 | Fixed best-k (offline) | Adaptive k (online)
    Left panel: MAE (°C)   Right panel: FEM savings (%)

    Fixed k=1 is the naive baseline (no FEM skipping, 0% savings).
    Fixed best-k requires training all k∈{1..5} and picking the winner offline.
    Adaptive k determines k online during inference — single training run.
    """
    if not all_adaptive:
        print("  No adaptive results to summarise.")
        return None

    registry = load_registry()

    DT_FEM  = 1.5
    T_TOTAL = 30.0
    N_FEM   = int(T_TOTAL / DT_FEM)   # 20

    labels       = []
    mae_k1       = []   # fixed k=1 baseline (always 0% savings)
    mae_best     = []   # fixed best-k (offline, 5× training cost)
    mae_adapt    = []   # adaptive k  (online, 1× training cost)
    sav_k1       = []
    sav_best     = []
    sav_adapt    = []

    for d in all_adaptive:
        domain_lbl = DOMAIN_LABELS.get(d["domain"], d["domain"])
        if d["domain"] == "lshape":
            domain_lbl = f"L-Shape {'3D' if d['dim'] == 3 else '2D'}"
        arch_lbl = ARCH_LABELS.get(d["arch"], d["arch"])
        labels.append(f"{domain_lbl}\n{arch_lbl}")

        reg_entry = next(
            (r for r in registry
             if r["domain"] == d["domain"] and r["arch"] == d["arch"]
             and r["dim"] == d["dim"]),
            None
        )

        if reg_entry:
            # k=1 MAE
            mae_k1.append(reg_entry["all_k"].get("1", None))
            sav_k1.append(0.0)   # k=1 → N_FEM windows → 0% savings

            # best-k
            best_k = reg_entry["best_k"]
            n_fix  = int(np.ceil(T_TOTAL / (best_k * DT_FEM)))
            mae_best.append(reg_entry["best_mae"])
            sav_best.append((1 - n_fix / N_FEM) * 100)
        else:
            mae_k1.append(None);  sav_k1.append(None)
            mae_best.append(None); sav_best.append(None)

        mae_adapt.append(d["mean_mae"])
        sav_adapt.append(d["fem_savings_pct"])

    n = len(labels)
    x = np.arange(n)
    w = 0.24   # three bars per group

    # separate 2D / 3D for cleaner figures
    dims   = [d["dim"] for d in all_adaptive]
    groups = sorted(set(dims))

    figs = []
    for dim in groups:
        idx = [i for i, d in enumerate(all_adaptive) if d["dim"] == dim]
        if not idx:
            continue

        xi  = np.arange(len(idx))
        lbl = [labels[i]    for i in idx]
        mk1 = [mae_k1[i]    for i in idx]
        mb  = [mae_best[i]  for i in idx]
        ma  = [mae_adapt[i] for i in idx]
        sk1 = [sav_k1[i]    for i in idx]
        sb  = [sav_best[i]  for i in idx]
        sa  = [sav_adapt[i] for i in idx]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, len(idx) * 2.8), 5.5))
        fig.patch.set_facecolor("white")

        COLOR_K1   = "#90a4ae"   # grey  — naive baseline
        COLOR_BEST = BLUE        # blue  — offline best-k
        COLOR_ADAP = AMBER       # amber — adaptive (online)

        def _bars(ax, vals_k1, vals_best, vals_adapt, ylabel, ylim=None):
            for i, (v1, vb, va) in enumerate(zip(vals_k1, vals_best, vals_adapt)):
                if v1 is not None:
                    b1 = ax.bar(xi[i] - w, v1, w, color=COLOR_K1,
                                alpha=0.85, label="Fixed k=1 (baseline)" if i == 0 else "")
                    ax.text(xi[i] - w, v1 + 0.3, f"{v1:.1f}",
                            ha="center", va="bottom", fontsize=6.5, color="#37474f")
                if vb is not None:
                    ax.bar(xi[i],     vb, w, color=COLOR_BEST,
                           alpha=0.85, label="Fixed best-k (offline, 5× runs)" if i == 0 else "")
                    ax.text(xi[i], vb + 0.3, f"{vb:.1f}",
                            ha="center", va="bottom", fontsize=6.5, color=NAVY)
                ax.bar(xi[i] + w,  va, w, color=COLOR_ADAP,
                       alpha=0.85, label="Adaptive k (online, 1 run)" if i == 0 else "")
                ax.text(xi[i] + w, va + 0.3, f"{va:.1f}",
                        ha="center", va="bottom", fontsize=6.5, color="#bf360c")

            ax.set_xticks(xi)
            ax.set_xticklabels(lbl, fontsize=7.5)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=7.5, framealpha=0.9)
            ax.tick_params(labelsize=8)
            if ylim:
                ax.set_ylim(0, ylim)
            else:
                ax.set_ylim(bottom=0)

        _bars(ax1, mk1, mb, ma, "Mean MAE  (°C)")
        _bars(ax2, sk1, sb, sa, "FEM savings  (%)", ylim=100)

        ax1.set_title("Accuracy  (lower is better)", fontsize=10, color=NAVY)
        ax2.set_title("FEM call reduction  (higher is better)", fontsize=10, color=NAVY)

        fig.suptitle(
            f"Adaptive k vs Fixed k strategies  —  {dim}D domains\n"
            f"Fixed best-k: offline selection after 5 training runs  |  "
            f"Adaptive k: online, single training run",
            fontsize=9.5, color=NAVY, fontweight="bold"
        )
        plt.tight_layout()
        plt.close(fig)

        if save:
            path = RESULT_DIR / f"adaptive_k_summary_{dim}d.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            print(f"  Saved: {path}")

        figs.append(fig)

    return figs[0] if figs else None


def find_all_adaptive() -> list[dict]:
    results = []
    for p in sorted(CKPT_DIR.glob("*_adaptive_dim*_metrics.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default=None)
    parser.add_argument("--arch",   default=None)
    parser.add_argument("--dim",    type=int, default=None)
    parser.add_argument("--no_summary", action="store_true")
    args = parser.parse_args()

    if args.domain:
        dim  = args.dim or (3 if args.domain in {"rectangular","cylinder","stacked","lshape3d"} else 2)
        arch = args.arch or "nsga2"
        data = load_adaptive(args.domain, arch, dim)
        if data is None:
            print(f"No adaptive metrics found for {args.domain}/{arch}/dim{dim}")
            return
        plot_timeline(data)
        if not args.no_summary:
            plot_summary([data])
    else:
        all_data = find_all_adaptive()
        if not all_data:
            print("No adaptive_k metrics found in checkpoints/")
            return
        print(f"Found {len(all_data)} adaptive run(s).")
        for d in all_data:
            plot_timeline(d)
        if not args.no_summary:
            plot_summary(all_data)


if __name__ == "__main__":
    main()
