"""
plot_skip_l5.py — Level 5 Skip Evaluation Plot
================================================
Compares Level 2 (window TimeStepperPINN) vs Level 5 (global L-BFGS PINN)
skip behavior for all 3 optimizers.

Key insight: Level 5 global PINNs are skip-agnostic (error does not grow
with skip because the model predicts T(t,x,y) without needing FEM anchors).

Usage:
    python level5_refinement/plot_skip_l5.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

_ROOT    = Path(__file__).resolve().parent.parent
_FIG_BG  = "#f7f1e3"
_AX_BG   = "#fffdf8"
_SPINE_C = "#d8cfbf"
_TXT_C   = "#2f2a24"
_TICK_C  = "#4c463f"

CONV_FACTOR = 1.5

OPT_STYLE = {
    "bayesian": {"color": "#1565C0", "marker": "o", "ls": "-",
                 "label": "Bayesian NAS\n[5×151 relu]"},
    "nsga2":    {"color": "#C62828", "marker": "s", "ls": "--",
                 "label": "NSGA-II\n[3×153 tanh]"},
    "nsga3":    {"color": "#2E7D32", "marker": "^", "ls": ":",
                 "label": "NSGA-III\n[3×75 tanh]"},
}


def _style(ax):
    ax.set_facecolor(_AX_BG)
    for sp in ax.spines.values():
        sp.set_color(_SPINE_C); sp.set_linewidth(0.8)
    ax.tick_params(colors=_TICK_C, labelsize=8)
    ax.xaxis.label.set_color(_TICK_C)
    ax.yaxis.label.set_color(_TICK_C)


def load_data():
    p2 = _ROOT / "level2_timestepper" / "results" / "skip_table.json"
    with open(p2) as f:
        l2_data = json.load(f)

    p2l = _ROOT / "level2_timestepper" / "results" / "skip_table_lbfgs.json"
    with open(p2l) as f:
        l2l_data = json.load(f)

    p5 = _ROOT / "level5_refinement" / "results" / "skip_table_l5.json"
    with open(p5) as f:
        l5_data = json.load(f)

    return l2_data, l2l_data, l5_data


def plot_skip_comparison(l2_data, l2l_data, l5_data, out_dir: Path):
    """
    Three-series comparison per optimizer:
      L2_orig : Level 2 Adam-only (500 ep, n=1000)
      L2_lbfgs: Level 2 Adam+L-BFGS per window (300 ep + 30 L-BFGS, n=500)
      L5_global: Level 5 global L-BFGS PINN (pre-trained, no retraining)

    Row 0: MAE vs skip (3 optimizers)
    Row 1: L2_rel vs skip (3 optimizers)
    Row 2: Summary table (all three approaches × all skip values)
    """
    skip_vals = [1, 2, 4, 6]
    opts      = ["bayesian", "nsga2", "nsga3"]

    SERIES = [
        {"data": l2_data,  "label": "L2 Adam\n(500 ep)",       "ls": "-",   "marker": "o",  "alpha": 1.0},
        {"data": l2l_data, "label": "L2 Adam+L-BFGS\n(300+30)","ls": "--",  "marker": "s",  "alpha": 0.85},
        {"data": l5_data,  "label": "L5 Global PINN\n(pre-trained)", "ls": "-.", "marker": "D", "alpha": 0.7},
    ]

    fig = plt.figure(figsize=(18, 13), facecolor=_FIG_BG)
    fig.suptitle(
        "Temporal Skip Capability: Three Approaches Compared\n"
        "Level 2 Adam-only  |  Level 2 Adam+L-BFGS per window  |  Level 5 Global PINN",
        fontsize=12, color=_TXT_C, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(3, 3, hspace=0.48, wspace=0.35,
                           left=0.07, right=0.97, top=0.93, bottom=0.03)

    # ── Rows 0 & 1: MAE and L2 curves per optimizer ───────────────────────────
    for col, opt in enumerate(opts):
        style = OPT_STYLE[opt]

        for row_idx, metric_key, ylabel in [(0, "mae_C", "MAE (°C)"),
                                             (1, "l2_rel", "L2 Relative Error")]:
            ax = fig.add_subplot(gs[row_idx, col])
            _style(ax)

            base_mae = None
            for ser in SERIES:
                rows_d = {r["skip"]: r for r in ser["data"].get(opt, [])}
                xs = [s for s in skip_vals if s in rows_d]
                ys = [rows_d[s][metric_key] for s in xs]
                if not xs:
                    continue
                if base_mae is None and metric_key == "mae_C":
                    base_mae = rows_d.get(1, {}).get("mae_C", None)
                ax.plot(xs, ys, marker=ser["marker"], color=style["color"],
                        ls=ser["ls"], lw=2, ms=6, alpha=ser["alpha"],
                        label=ser["label"], zorder=5)

            # Convergence threshold line (based on Level 2 Adam skip=1)
            if metric_key == "mae_C" and base_mae:
                thr = {r["skip"]: r for r in l2_data.get(opt, [])}.get(1, {}).get("mae_C")
                if thr:
                    ax.axhline(thr * CONV_FACTOR, color="#E65100", ls=":", lw=1.3,
                               alpha=0.6, label=f"Threshold ({CONV_FACTOR}×)")
            if metric_key == "l2_rel":
                ax.axhline(0.10, color="#E65100", ls=":", lw=1.3, alpha=0.6,
                           label="Target L2=0.10")

            if row_idx == 0:
                ax.set_title(style["label"], fontsize=9, color=_TXT_C, pad=5)
            ax.set_xlabel("Skip (k)", fontsize=9)
            ax.set_xticks(skip_vals)
            ax.set_xticklabels([f"skip={s}" for s in skip_vals], fontsize=7.5)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9, color=_TICK_C)
            ax.legend(fontsize=6.5, loc="upper left",
                      facecolor=_AX_BG, edgecolor=_SPINE_C)

    # ── Row 2: Summary table ──────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis("off")

    col_labels = ["Optimizer", "Approach",
                  "skip=1\nMAE", "skip=2\nMAE", "skip=2\n✓/✗",
                  "skip=4\nMAE", "skip=4\n✓/✗",
                  "skip=6\nMAE", "skip=6\n✓/✗"]

    row_colors_all = []
    rows_all       = []
    base_cols      = {"bayesian": "#DCEEFA", "nsga2": "#FDDEDE", "nsga3": "#DFF3E0"}
    lite_cols      = {"bayesian": "#EAF4FF", "nsga2": "#FFF0F0", "nsga3": "#F0FAF1"}
    glob_cols      = {"bayesian": "#F0F4FF", "nsga2": "#FFF5F5", "nsga3": "#F5FFF5"}
    ser_labels_short = ["L2 Adam-only", "L2 Adam+LBFGS", "L5 Global"]
    ser_datasets     = [l2_data, l2l_data, l5_data]
    ser_colors       = [base_cols, lite_cols, glob_cols]

    for opt in opts:
        opt_label = OPT_STYLE[opt]["label"].replace("\n", " ")
        # Reference MAE for convergence: Level 2 Adam skip=1
        ref_mae = {r["skip"]: r for r in l2_data.get(opt, [])}.get(1, {}).get("mae_C", 1e9)

        for ser_label, ser_data, ser_col in zip(ser_labels_short, ser_datasets, ser_colors):
            rows_d = {r["skip"]: r for r in ser_data.get(opt, [])}
            base   = rows_d.get(1, {}).get("mae_C", float("nan"))
            row    = [opt_label, ser_label, f"{base:.1f}°C"]
            for sk in [2, 4, 6]:
                r    = rows_d.get(sk, {})
                mae  = r.get("mae_C", float("nan"))
                # Always judge convergence against Level 2 Adam skip=1 baseline
                ratio = mae / (ref_mae + 1e-9)
                conv  = "✓" if ratio < CONV_FACTOR else "✗"
                row.append(f"{mae:.1f}°C")
                row.append(f"{conv} {ratio:.2f}×")
            rows_all.append(row)
            row_colors_all.append(ser_col[opt])

    tbl = ax_tbl.table(cellText=rows_all, colLabels=col_labels,
                       cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.9)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#37474F")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i, rc in enumerate(row_colors_all):
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(rc)
        for j_conv in [4, 6, 8]:
            txt = rows_all[i][j_conv]
            if txt.startswith("✓"):
                tbl[i + 1, j_conv].set_facecolor("#C8E6C9")
                tbl[i + 1, j_conv].set_text_props(color="#1B5E20", fontweight="bold")
            elif txt.startswith("✗"):
                tbl[i + 1, j_conv].set_facecolor("#FFCDD2")
                tbl[i + 1, j_conv].set_text_props(color="#B71C1C", fontweight="bold")

    fig.text(
        0.5, 0.002,
        "Convergence criterion: MAE ratio vs Level 2 Adam skip=1 baseline < 1.5×  |  "
        "L2 Adam+L-BFGS trained with fewer epochs (300+30) vs Adam-only (500)  |  "
        "L5 Global PINN skip-agnostic: no window retraining needed",
        ha="center", va="bottom", fontsize=7.5, color=_TXT_C, style="italic",
        bbox={"facecolor": "#fff7dc", "edgecolor": "#ddc98b",
              "boxstyle": "round,pad=0.3", "alpha": 0.85},
    )

    out_path = out_dir / "level5_skip_comparison.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    l2_data, l2l_data, l5_data = load_data()
    out_dir = _ROOT / "level5_refinement" / "results"
    plot_skip_comparison(l2_data, l2l_data, l5_data, out_dir)
    print("Done.")
