"""
plot_level7a.py — Level 7A Visualization
==========================================
FEM-PINNS style: cream background, YlOrRd heatmaps, clean layout.

Outputs:
  l7a_cooling_curves.png     — T_avg(t) curves: 3 optimizers + analytic ref
  l7a_l2_vs_time.png         — L2(t) curve: accuracy over time
  l7a_heatmaps_bayesian.png  — T(x,y) heatmaps at t=1,5,15,30s [BAYESIAN]
  l7a_heatmaps_all.png       — 3 optimizers x 4 time slices grid
  l7a_timeskip_table.png     — FEM vs NAS-PINN comparison table
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.config import T_INIT, T_WATER_INIT, T_END, ALPHA, X_MAX, Y_MAX

L7_DIR  = _ROOT / "level7_temporal" / "results"
OUT_DIR = L7_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OPTIMIZERS    = ["bayesian", "nsga2", "nsga3"]
OPT_LABELS    = {"bayesian": "Bayesian", "nsga2": "NSGA-II", "nsga3": "NSGA-III"}
OPT_COLORS    = {"bayesian": "#1565C0",  "nsga2": "#2E7D32",  "nsga3": "#E65100"}
HEATMAP_TIMES = [1.0, 5.0, 15.0, 30.0]

# FEM-PINNS style constants
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
        sp.set_color(_SPINE_C); sp.set_linewidth(0.8)
    ax.tick_params(colors=_TICK_C, labelsize=8)
    ax.xaxis.label.set_color(_TICK_C)
    ax.yaxis.label.set_color(_TICK_C)


def _box(ax, text, loc="top"):
    y, va = (0.96, "top") if loc == "top" else (0.04, "bottom")
    ax.text(0.03, y, text, transform=ax.transAxes, va=va, fontsize=9,
            color=_TXT_C,
            bbox={"facecolor": _BOX_FC, "edgecolor": _BOX_EC,
                  "boxstyle": "round,pad=0.25"})


def _load(opt: str) -> dict:
    p = L7_DIR / opt / "time_skip_result.json"
    return json.load(open(p)) if p.exists() else {}


def T_ref_curve(t_arr):
    return T_WATER_INIT + (T_INIT - T_WATER_INIT) * np.exp(-ALPHA * np.asarray(t_arr))


# ── Figure 1: Cooling curves ─────────────────────────────────────────────────

def plot_cooling_curves():
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_FIG_BG,
                           constrained_layout=True)
    _style(ax)

    t_fine = np.linspace(0, T_END, 300)
    ax.plot(t_fine, T_ref_curve(t_fine),
            "k--", linewidth=1.8, label="Analytic reference (lumped)", zorder=5)

    for opt in OPTIMIZERS:
        d = _load(opt)
        if not d:
            continue
        t   = np.array(d["t_dense"])
        T   = np.array(d["T_avg_pinn_dense"])
        ax.plot(t, T, color=OPT_COLORS[opt], linewidth=2.0,
                label=f"{OPT_LABELS[opt]}  (relL2={d['mean_l2_dense']:.3f})")

    for t_mark in HEATMAP_TIMES:
        ax.axvline(t_mark, color="#aaa", linestyle=":", linewidth=0.9)

    ax.set_xlabel("Time t [s]", fontsize=10)
    ax.set_ylabel("Mean Temperature T [°C]", fontsize=10)
    ax.set_title(
        "NAS-PINN Cooling Curve — All Optimizers\n"
        "Dotted verticals: direct NAS-PINN evaluation points  "
        "(FEM must solve every intermediate step)",
        fontsize=11, color=_TXT_C, pad=8,
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)

    out = OUT_DIR / "l7a_cooling_curves.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 2: L2(t) curve ────────────────────────────────────────────────────

def plot_l2_vs_time():
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=_FIG_BG,
                           constrained_layout=True)
    _style(ax)

    for opt in OPTIMIZERS:
        d = _load(opt)
        if not d:
            continue
        t  = np.array(d["t_dense"])
        l2 = np.array(d["l2_curve"])
        ax.plot(t, l2, color=OPT_COLORS[opt], linewidth=2.0,
                label=OPT_LABELS[opt])

    for t_mark in HEATMAP_TIMES:
        ax.axvline(t_mark, color="#aaa", linestyle=":", linewidth=0.9)

    ax.set_xlabel("Time t [s]", fontsize=10)
    ax.set_ylabel("Relative L\u2082 Error", fontsize=10)
    ax.set_title(
        "NAS-PINN Accuracy — Relative L\u2082 Error over Time\n"
        "Direct evaluation at each t — no time-marching required",
        fontsize=11, color=_TXT_C, pad=8,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    out = OUT_DIR / "l7a_l2_vs_time.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 3: Heatmaps — single optimizer ────────────────────────────────────

def plot_heatmaps_single(opt: str = "bayesian"):
    d = _load(opt)
    if not d:
        print(f"  [{opt}] No data, skipping.")
        return

    grids = {float(t): np.array(d["heatmap_grids"][str(t)])
             for t in d["heatmap_times"]}

    all_vals = np.concatenate([g.flatten() for g in grids.values()])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    n = len(HEATMAP_TIMES)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n + 1.2, 3.8),
                              facecolor=_FIG_BG, constrained_layout=True)

    im_last = None
    for ax, t_val in zip(axes, HEATMAP_TIMES):
        im_last = ax.imshow(
            grids[t_val].T, origin="lower", aspect="auto",
            extent=[0, X_MAX, 0, Y_MAX],
            cmap="YlOrRd", vmin=vmin, vmax=vmax,
        )
        _style(ax)
        ax.set_title(f"t = {t_val:.0f} s", fontsize=12, color=_TXT_C, pad=6)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]" if t_val == HEATMAP_TIMES[0] else "")

    cbar = fig.colorbar(im_last, ax=list(axes), shrink=0.85, pad=0.02)
    cbar.set_label("T [°C]", color=_TXT_C)
    cbar.ax.tick_params(labelsize=8, colors=_TICK_C)

    l2  = d.get("mean_l2_dense", 0)
    fwd = d.get("fwd_pass_ms", 0)
    fig.suptitle(
        f"Quenching Temperature Field — {OPT_LABELS[opt]}  |  relL2 = {l2:.3f}\n"
        f"NAS-PINN evaluated directly at t = {{1, 5, 15, 30}} s  "
        f"(FEM: 3000 sequential steps  |  PINN: {fwd:.1f} ms/point)",
        fontsize=11, color=_TXT_C,
    )

    out = OUT_DIR / f"l7a_heatmaps_{opt}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 4: All optimizers x 4 time slices ─────────────────────────────────

def plot_heatmaps_all():
    data  = {opt: _load(opt) for opt in OPTIMIZERS}
    valid = [opt for opt in OPTIMIZERS if data[opt]]
    if not valid:
        return

    all_vals = []
    for opt in valid:
        for t in HEATMAP_TIMES:
            all_vals.extend(np.array(data[opt]["heatmap_grids"][str(t)]).flatten())
    vmin, vmax = float(min(all_vals)), float(max(all_vals))

    nrow = len(valid)
    ncol = len(HEATMAP_TIMES)
    fig, axes = plt.subplots(nrow, ncol,
                              figsize=(4.5 * ncol + 1.0, 3.5 * nrow),
                              facecolor=_FIG_BG, constrained_layout=True)
    axes = np.atleast_2d(axes)

    im_last = None
    for row, opt in enumerate(valid):
        d     = data[opt]
        grids = {float(t): np.array(d["heatmap_grids"][str(t)])
                 for t in d["heatmap_times"]}
        l2    = d.get("mean_l2_dense", 0)

        for col, t_val in enumerate(HEATMAP_TIMES):
            ax = axes[row, col]
            im_last = ax.imshow(
                grids[t_val].T, origin="lower", aspect="auto",
                extent=[0, X_MAX, 0, Y_MAX],
                cmap="YlOrRd", vmin=vmin, vmax=vmax,
            )
            _style(ax)
            if col == 0:
                ax.set_ylabel(f"{OPT_LABELS[opt]}\n(relL2={l2:.3f})\ny [m]",
                              fontsize=9, color=_TXT_C)
            if row == 0:
                ax.set_title(f"t = {t_val:.0f} s", fontsize=11,
                             color=_TXT_C, pad=5)
            if row == nrow - 1:
                ax.set_xlabel("x [m]")

    cbar = fig.colorbar(im_last, ax=axes[:, -1].tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("T [°C]", color=_TXT_C)
    cbar.ax.tick_params(labelsize=8, colors=_TICK_C)

    fig.suptitle(
        "Quenching Temperature Field T(x,y,t) — All Optimizers\n"
        "NAS-PINN: direct evaluation at t = {1, 5, 15, 30} s  "
        "(FEM: sequential global solve required at every step)",
        fontsize=12, color=_TXT_C,
    )

    out = OUT_DIR / "l7a_heatmaps_all.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 5: Comparison table ───────────────────────────────────────────────

def plot_comparison_table():
    summary_path = L7_DIR / "level7a_summary.json"
    if not summary_path.exists():
        return
    s = json.load(open(summary_path))

    rows = []
    rows.append(["FEM (fine,   \u0394t=0.01 s)", "3000", "Sequential \u2717", "\u2014", "\u2014"])
    rows.append(["FEM (coarse, \u0394t=0.30 s)",  " 100", "Sequential \u2717", "\u2014", "\u2014"])

    for opt in OPTIMIZERS:
        if opt not in s["results_per_optimizer"]:
            continue
        r        = s["results_per_optimizer"][opt]
        sc_vals  = list(r["scenarios"].values())
        first_l2 = sc_vals[0]["mean_l2"]
        last_l2  = sc_vals[-1]["mean_l2"]
        fwd      = r["pinn_forward_ms"]
        rows.append([
            f"NAS-PINN {OPT_LABELS[opt]}",
            "3 \u2192 61",
            "Direct \u2713",
            f"{first_l2:.4f} \u2192 {last_l2:.4f}",
            f"{fwd:.2f} ms",
        ])

    col_labels = ["Method", "Time Steps", "Sequential?", "relL2 (3\u219261 pts)", "Single eval"]

    fig, ax = plt.subplots(figsize=(13, 0.6 * (len(rows) + 2) + 1.5),
                            facecolor=_FIG_BG)
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.7)
    tbl.auto_set_column_width(range(len(col_labels)))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1565C0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(rows, 1):
        is_pinn = "NAS-PINN" in row[0]
        bg = "#C8E6C9" if is_pinn else "#FFECB3"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)
        if is_pinn:
            tbl[i, 2].set_text_props(color="#1B5E20", fontweight="bold")

    ax.set_title(
        "Level 7A — FEM vs NAS-PINN Time-Skip Comparison\n"
        "Green = NAS-PINN (direct evaluation)  |  Yellow = FEM (sequential marching)",
        fontsize=11, color=_TXT_C, pad=12,
    )

    out = OUT_DIR / "l7a_timeskip_table.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Level 7A — Plot Results")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*60}\n")

    plot_cooling_curves()
    plot_l2_vs_time()
    for opt in OPTIMIZERS:
        plot_heatmaps_single(opt)
    plot_heatmaps_all()
    plot_comparison_table()

    print(f"\n  All figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
