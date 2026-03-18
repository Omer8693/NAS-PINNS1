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
    # Level 2 (window TimeStepperPINN)
    p2 = _ROOT / "level2_timestepper" / "results" / "skip_table.json"
    with open(p2) as f:
        l2_data = json.load(f)

    # Level 5 (global L-BFGS PINN)
    p5 = _ROOT / "results" / "level5_refinement" / "skip_table_l5.json"
    with open(p5) as f:
        l5_data = json.load(f)

    return l2_data, l5_data


def plot_skip_comparison(l2_data, l5_data, out_dir: Path):
    """
    Main comparison figure:
      Row 0: MAE vs skip — Level 2 vs Level 5, all optimizers
      Row 1: L2_rel vs skip — Level 2 vs Level 5, all optimizers
      Row 2: Summary table
    """
    skip_vals = [1, 2, 4, 6]

    fig = plt.figure(figsize=(16, 12), facecolor=_FIG_BG)
    fig.suptitle(
        "Level 2 (Window PINN) vs Level 5 (Global L-BFGS PINN) — Temporal Skip Capability\n"
        "Skip = k → FEM at every k-th step, PINN predicts skipped steps",
        fontsize=12, color=_TXT_C, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.38,
                           left=0.07, right=0.97, top=0.92, bottom=0.04)

    opts = ["bayesian", "nsga2", "nsga3"]

    # ── Row 0: MAE vs skip per optimizer ──────────────────────────────────────
    for col, opt in enumerate(opts):
        ax = fig.add_subplot(gs[0, col])
        _style(ax)
        style = OPT_STYLE[opt]

        # Level 2
        skips_l2 = [r["skip"] for r in l2_data.get(opt, [])]
        maes_l2  = [r["mae_C"] for r in l2_data.get(opt, [])]
        if skips_l2:
            ax.plot(skips_l2, maes_l2, marker=style["marker"], color=style["color"],
                    ls=style["ls"], lw=2, ms=7, label="Level 2\n(window PINN)", zorder=5)
            ax.fill_between(skips_l2, maes_l2, alpha=0.08, color=style["color"])

        # Level 5
        skips_l5 = [r["skip"] for r in l5_data.get(opt, [])]
        maes_l5  = [r["mae_C"] for r in l5_data.get(opt, [])]
        if skips_l5:
            ax.plot(skips_l5, maes_l5, marker="D", color=style["color"],
                    ls="-.", lw=2, ms=7, label="Level 5\n(global PINN)", alpha=0.7, zorder=4)

        # Convergence threshold: 1.5× of skip=1 baseline (level 2)
        if maes_l2:
            thr = maes_l2[0] * CONV_FACTOR
            ax.axhline(thr, color=style["color"], ls=":", lw=1.2, alpha=0.5,
                       label=f"Threshold ({CONV_FACTOR}× baseline)")

        ax.set_title(style["label"], fontsize=9, color=_TXT_C, pad=6)
        ax.set_xlabel("Skip (k)", fontsize=9)
        ax.set_ylabel("MAE (°C)" if col == 0 else "", fontsize=9)
        ax.set_xticks(skip_vals)
        ax.set_xticklabels([f"skip={s}" for s in skip_vals], fontsize=7)
        ax.legend(fontsize=7, loc="upper left",
                  facecolor=_AX_BG, edgecolor=_SPINE_C)
        ax.grid(True, alpha=0.3)

        if col == 0:
            ax.set_ylabel("MAE (°C)", fontsize=9, color=_TICK_C)

    # ── Row 1: L2_rel vs skip ─────────────────────────────────────────────────
    for col, opt in enumerate(opts):
        ax = fig.add_subplot(gs[1, col])
        _style(ax)
        style = OPT_STYLE[opt]

        skips_l2 = [r["skip"] for r in l2_data.get(opt, [])]
        l2s_l2   = [r["l2_rel"] for r in l2_data.get(opt, [])]
        skips_l5 = [r["skip"] for r in l5_data.get(opt, [])]
        l2s_l5   = [r["l2_rel"] for r in l5_data.get(opt, [])]

        if skips_l2:
            ax.plot(skips_l2, l2s_l2, marker=style["marker"], color=style["color"],
                    ls=style["ls"], lw=2, ms=7, label="Level 2", zorder=5)
        if skips_l5:
            ax.plot(skips_l5, l2s_l5, marker="D", color=style["color"],
                    ls="-.", lw=2, ms=7, label="Level 5", alpha=0.7, zorder=4)

        # Target L2 line
        ax.axhline(0.10, color="#E65100", ls="--", lw=1.2, alpha=0.7,
                   label="Target L2=0.10")

        ax.set_xlabel("Skip (k)", fontsize=9)
        ax.set_xticks(skip_vals)
        ax.set_xticklabels([f"skip={s}" for s in skip_vals], fontsize=7)
        ax.legend(fontsize=7, loc="upper left",
                  facecolor=_AX_BG, edgecolor=_SPINE_C)
        ax.grid(True, alpha=0.3)

        if col == 0:
            ax.set_ylabel("L2 Relative Error", fontsize=9, color=_TICK_C)

    # ── Row 2: Summary table ──────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis("off")
    ax_tbl.set_facecolor(_AX_BG)

    rows       = []
    col_labels = ["Optimizer", "Level", "skip=1\nMAE (°C)",
                  "skip=2\nMAE (°C)", "skip=2\nConverged?",
                  "skip=4\nMAE (°C)", "skip=4\nConverged?",
                  "skip=6\nMAE (°C)", "skip=6\nConverged?"]

    def get_row(data, opt, level_label):
        rows_d = {r["skip"]: r for r in data.get(opt, [])}
        base   = rows_d.get(1, {}).get("mae_C", float("nan"))
        out    = [OPT_STYLE[opt]["label"].replace("\n", " "), level_label,
                  f"{base:.1f}"]
        for sk in [2, 4, 6]:
            r   = rows_d.get(sk, {})
            mae = r.get("mae_C", float("nan"))
            ratio = mae / (base + 1e-9)
            conv  = "✓" if ratio < CONV_FACTOR else "✗"
            out.append(f"{mae:.1f}")
            out.append(f"{conv} ({ratio:.2f}×)")
        return out

    row_colors = []
    for opt in opts:
        rows.append(get_row(l2_data, opt, "Level 2\n(Window PINN)"))
        row_colors.append({"bayesian": "#DCEEFA", "nsga2": "#FDDEDE", "nsga3": "#DFF3E0"}[opt])
        rows.append(get_row(l5_data, opt, "Level 5\n(Global PINN)"))
        row_colors.append({"bayesian": "#EAF4FF", "nsga2": "#FFF0F0", "nsga3": "#F0FAF1"}[opt])

    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                       cellLoc="center", loc="center",
                       bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 2.0)

    # Header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#37474F")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Row colors + convergence highlight
    for i, (row, rc) in enumerate(zip(rows, row_colors)):
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(rc)
        # Color convergence cells
        for j_conv, sk in zip([4, 6, 8], [2, 4, 6]):
            cell_txt = rows[i][j_conv]
            if "✓" in cell_txt:
                tbl[i + 1, j_conv].set_facecolor("#C8E6C9")
                tbl[i + 1, j_conv].set_text_props(color="#1B5E20", fontweight="bold")
            elif "✗" in cell_txt:
                tbl[i + 1, j_conv].set_facecolor("#FFCDD2")
                tbl[i + 1, j_conv].set_text_props(color="#B71C1C", fontweight="bold")

    # Insight annotation
    fig.text(0.5, 0.005,
             "Key finding: Level 5 global PINNs are skip-agnostic (MAE ratio ≈ 1.0× for all skip values) "
             "because they predict T(t,x,y) without needing FEM anchor steps.\n"
             "Level 2 window PINNs show skip degradation but achieve lower absolute error (trained directly on analytical cooling curve).",
             ha="center", va="bottom", fontsize=8, color=_TXT_C, style="italic",
             bbox={"facecolor": "#fff7dc", "edgecolor": "#ddc98b",
                   "boxstyle": "round,pad=0.3", "alpha": 0.85})

    out_path = out_dir / "level5_skip_comparison.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    l2_data, l5_data = load_data()
    out_dir = _ROOT / "results" / "level5_refinement"
    plot_skip_comparison(l2_data, l5_data, out_dir)
    print("Done.")
