"""
plot_results.py — Level 5 L-BFGS Refinement Visualization
===========================================================
Reads results/level5_refinement/level5_summary.json and generates:

  Fig 1: Bar chart — L2_rel before (Adam) vs after (L-BFGS) for 3 optimizers
  Fig 2: L-BFGS loss convergence curves (log scale)
  Fig 3: Summary table — Level 1 Adam vs Level 5 L-BFGS side by side

Usage:
    python level5_refinement/plot_results.py
    python level5_refinement/plot_results.py --results_dir results/level5_refinement
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

STYLE = {
    "bayesian": {"color": "#2196F3", "label": "Bayesian\n5×151 relu"},
    "nsga2":    {"color": "#F44336", "label": "NSGA-II\n3×153 tanh"},
    "nsga3":    {"color": "#4CAF50", "label": "NSGA-III\n3×75 tanh"},
}
TARGET_L2 = 0.10


# ── Fig 1: L2_rel before vs after ─────────────────────────────────────────────

def plot_l2_comparison(summary: dict, out_dir: str) -> None:
    valid = {k: v for k, v in summary.items() if "error" not in v}
    opts  = [o for o in ["bayesian", "nsga2", "nsga3"] if o in valid]
    x     = np.arange(len(opts))
    bar_w = 0.32

    l2_adam  = [valid[o]["level1_adam_only"]["L2_rel"]   or 0.0 for o in opts]
    l2_lbfgs = [valid[o]["level5_adam_lbfgs"]["L2_rel"]  or 0.0 for o in opts]
    colors   = [STYLE[o]["color"] for o in opts]
    labels   = [STYLE[o]["label"] for o in opts]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Level 5 — L2_rel: Adam (Level 1) vs L-BFGS Refinement (Level 5)",
                 fontsize=12, fontweight="bold")

    bars1 = ax.bar(x - bar_w/2, l2_adam,  bar_w, label="Adam (Level 1)",
                   color=colors, alpha=0.5, edgecolor="white", hatch="///")
    bars2 = ax.bar(x + bar_w/2, l2_lbfgs, bar_w, label="L-BFGS (Level 5)",
                   color=colors, alpha=0.9, edgecolor="white")

    ax.axhline(TARGET_L2, color="black", linestyle="--", linewidth=1.2,
               label=f"Target L2 = {TARGET_L2}")

    for bar, v in zip(bars1, l2_adam):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="gray")
    for bar, v in zip(bars2, l2_lbfgs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Improvement arrows
    for i, (v_a, v_l, opt) in enumerate(zip(l2_adam, l2_lbfgs, opts)):
        if v_l > 0:
            ratio = v_a / v_l
            ax.annotate(f"{ratio:.1f}×↓", xy=(x[i], max(v_a, v_l) + 0.04),
                        ha="center", fontsize=9, color=STYLE[opt]["color"],
                        fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("L2 Relative Error", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(l2_adam) * 1.4)

    plt.tight_layout()
    p = Path(out_dir) / "level5_l2_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


# ── Fig 2: L-BFGS loss convergence ────────────────────────────────────────────

def plot_convergence(results_dir: str, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Level 5 — L-BFGS Loss Convergence (log scale)",
                 fontsize=12, fontweight="bold")

    found = False
    for opt in ["bayesian", "nsga2", "nsga3"]:
        log_path = Path(results_dir) / opt / "refinement_log.json"
        if not log_path.exists():
            continue
        with open(log_path) as f:
            log = json.load(f)
        history = log.get("loss_history", [])
        if not history:
            continue
        style = STYLE[opt]
        ax.semilogy(history, color=style["color"], linewidth=1.5,
                    label=f"{style['label'].replace(chr(10), ' ')} ({len(history)} iters)")
        found = True

    if not found:
        print("  No loss history found — skipping convergence plot")
        plt.close()
        return

    ax.set_xlabel("L-BFGS Closure Evaluation", fontsize=11)
    ax.set_ylabel("Training Loss (log scale)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    p = Path(out_dir) / "level5_convergence.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


# ── Fig 3: Summary table ───────────────────────────────────────────────────────

def plot_summary_table(summary: dict, out_dir: str) -> None:
    valid = {k: v for k, v in summary.items() if "error" not in v}
    opts  = [o for o in ["bayesian", "nsga2", "nsga3"] if o in valid]

    rows = []
    for o in opts:
        res  = valid[o]
        arch = res["architecture"]
        a    = res["level1_adam_only"]
        l    = res["level5_adam_lbfgs"]

        arch_str = f"{arch['n_layers']}×{arch['neurons'][0]} {arch['activation']}"
        l2_a  = a.get("L2_rel")  or float("nan")
        mae_a = a.get("MAE_C")   or float("nan")
        l2_l  = l.get("L2_rel")  or float("nan")
        mae_l = l.get("MAE_C")   or float("nan")
        imp   = l2_a / (l2_l + 1e-12) if l2_l else float("nan")
        t     = l.get("adam_time_s", 0)
        iters = l.get("lbfgs_iters", 0)

        rows.append([
            STYLE[o]["label"].replace("\n", " "),
            arch_str,
            f"{l2_a:.3f}",
            f"{mae_a:.1f}",
            f"{l2_l:.3f}",
            f"{mae_l:.1f}",
            f"{imp:.1f}×",
            f"{iters}",
            f"{t:.0f}s",
        ])

    rows.append([
        "★ Target (acceptable)", "—",
        "< 0.100", "< 50.0",
        "< 0.100", "< 50.0",
        "—", "—", "—",
    ])

    col_labels = [
        "Optimizer", "Architecture",
        "Adam L2↓", "Adam MAE↓",
        "L-BFGS L2↓", "L-BFGS MAE↓",
        "Improvement", "Iters", "Time",
    ]

    fig, ax = plt.subplots(figsize=(17, max(2.5, 0.8 * (len(rows) + 1.5))))
    fig.suptitle("Level 5 — L-BFGS Refinement Summary: Adam vs L-BFGS",
                 fontsize=12, fontweight="bold")
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.0)

    # Header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#37474F")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Adam columns light red, L-BFGS light blue
    for i in range(1, len(rows)):
        for j in [2, 3]:
            tbl[i, j].set_facecolor("#FFEBEE")
        for j in [4, 5]:
            tbl[i, j].set_facecolor("#E3F2FD")

    # Target row green
    for j in range(len(col_labels)):
        tbl[len(rows), j].set_facecolor("#E8F5E9")
        tbl[len(rows), j].set_text_props(color="#2E7D32", fontweight="bold")

    plt.tight_layout()
    p = Path(out_dir) / "level5_summary_table.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Level 5 — Visualize L-BFGS Refinement")
    parser.add_argument("--results_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_dir     = args.out or args.results_dir
    summary_path = Path(args.results_dir) / "level5_summary.json"

    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run main_level5.py first.")
        exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    plot_l2_comparison(summary, out_dir)
    plot_convergence(args.results_dir, out_dir)
    plot_summary_table(summary, out_dir)
    print(f"\nDone — 3 figures saved to {out_dir}")
