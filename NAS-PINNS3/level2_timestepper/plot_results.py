"""
plot_results.py — Level 2 Skip Comparison Charts
=================================================
Reads results/skip_table.json and generates two figures:

  Fig 1: skip vs L2_rel — three-optimizer comparison
  Fig 2: skip vs MAE (°C) — three-optimizer comparison

Usage:
    python plot_results.py
    python plot_results.py --input results/skip_table.json --out results/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Color and marker style per optimizer
STYLE = {
    "bayesian": {"color": "#2196F3", "marker": "o", "linestyle": "-",  "label": "Bayesian [5×151 relu]"},
    "nsga2":    {"color": "#F44336", "marker": "s", "linestyle": "--", "label": "NSGA-II  [3×153 tanh]"},
    "nsga3":    {"color": "#4CAF50", "marker": "^", "linestyle": ":",  "label": "NSGA-III [3×75  tanh]"},
}

# Level 1 reference lines — run2 results
LEVEL1_REF = {
    "bayesian": {"l2": 0.076, "mae": 39.1},
    "nsga2":    {"l2": 0.252, "mae": 132.6},
    "nsga3":    {"l2": 0.513, "mae": 270.3},
}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_skip_comparison(data: dict, out_dir: str) -> None:
    """
    Two side-by-side figures:
      Left : skip vs L2_rel
      Right: skip vs MAE (°C)
    Level 1 reference shown as dash-dot horizontal lines per optimizer.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Level 2 — Time-Stepping PINN: Skip vs Accuracy\n"
                 "(dashed lines = Level 1 single-shot reference)",
                 fontsize=12, fontweight="bold")

    ax_l2, ax_mae = axes

    for opt_name, rows in data.items():
        style = STYLE.get(opt_name, {})
        skips  = [r["skip"]    for r in rows]
        l2s    = [r["l2_rel"]  for r in rows]
        maes   = [r["mae_C"]   for r in rows]

        # Level 2 lines
        ax_l2.plot(skips, l2s,  marker=style["marker"], color=style["color"],
                   linestyle=style["linestyle"], linewidth=2, markersize=8,
                   label=style.get("label", opt_name))
        ax_mae.plot(skips, maes, marker=style["marker"], color=style["color"],
                    linestyle=style["linestyle"], linewidth=2, markersize=8,
                    label=style.get("label", opt_name))

        # Level 1 reference horizontal line
        ref = LEVEL1_REF.get(opt_name)
        if ref:
            ax_l2.axhline(ref["l2"],  color=style["color"], linestyle="-.", alpha=0.4, linewidth=1)
            ax_mae.axhline(ref["mae"], color=style["color"], linestyle="-.", alpha=0.4, linewidth=1)

    # Left panel: L2_rel
    ax_l2.set_xlabel("Skip (every N-th time step)", fontsize=11)
    ax_l2.set_ylabel("L2 Relative Error", fontsize=11)
    ax_l2.set_title("L2_rel vs Skip", fontsize=11)
    ax_l2.set_xticks([1, 2, 4, 6])
    ax_l2.set_xticklabels(["skip=1\n(21/21)", "skip=2\n(11/21)",
                            "skip=4\n(6/21)",  "skip=6\n(4/21)"])
    ax_l2.legend(fontsize=9, loc="upper left")
    ax_l2.grid(True, alpha=0.3)
    ax_l2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Right panel: MAE
    ax_mae.set_xlabel("Skip (every N-th time step)", fontsize=11)
    ax_mae.set_ylabel("MAE (°C)", fontsize=11)
    ax_mae.set_title("MAE vs Skip", fontsize=11)
    ax_mae.set_xticks([1, 2, 4, 6])
    ax_mae.set_xticklabels(["skip=1\n(21/21)", "skip=2\n(11/21)",
                             "skip=4\n(6/21)",  "skip=6\n(4/21)"])
    ax_mae.legend(fontsize=9, loc="upper left")
    ax_mae.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_dir) / "skip_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_runtime_bar(data: dict, out_dir: str) -> None:
    """
    Runtime bar chart for each optimizer × skip combination.
    Shows how much time is saved by skipping more FEM steps.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Level 2 — Runtime vs Skip (seconds)", fontsize=12, fontweight="bold")

    opt_names  = list(data.keys())
    skip_vals  = [r["skip"] for r in next(iter(data.values()))]
    n_opts     = len(opt_names)
    n_skips    = len(skip_vals)
    bar_width  = 0.22
    x          = np.arange(n_skips)

    for i, opt_name in enumerate(opt_names):
        style    = STYLE.get(opt_name, {})
        runtimes = [r["runtime_s"] for r in data[opt_name]]
        offset   = (i - n_opts / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, runtimes, bar_width,
                      label=style.get("label", opt_name),
                      color=style["color"], alpha=0.85, edgecolor="white")
        # Value labels on top of bars
        for bar, val in zip(bars, runtimes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}s", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"skip={s}\n({[r['steps_used'] for r in next(iter(data.values()))][i]}"
                        f"/{[r['steps_total'] for r in next(iter(data.values()))][i]} steps)"
                        for i, s in enumerate(skip_vals)])
    ax.set_ylabel("Runtime (seconds)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_dir) / "skip_runtime.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Level 2 — Visualize skip results")
    parser.add_argument("--input", default="results/skip_table.json",
                        help="skip_table.json dosya yolu")
    parser.add_argument("--out",   default="results/",
                        help="Output directory")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: {args.input} not found. Run main_level2.py first.")
        exit(1)

    data = load_results(args.input)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    plot_skip_comparison(data, args.out)
    plot_runtime_bar(data, args.out)
    print("Done.")
