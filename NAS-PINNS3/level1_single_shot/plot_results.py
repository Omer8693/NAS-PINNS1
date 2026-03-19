"""
plot_results.py — Level 1 Single-Shot NAS-PINN Visualization
=============================================================
Reads results/run2/comparison.json and generates three figures:

  Fig 1: L2_rel and MAE — grouped bar comparison for three optimizers
  Fig 2: NAS + training runtime
  Fig 3: Paper FEM distortion (Fig17/18) vs PINN thermal error summary

Usage:
    python level1_single_shot/plot_results.py
    python level1_single_shot/plot_results.py --results_dir results/run2 --out results/run2/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Optimizer color and style — consistent with other levels
STYLE = {
    "bayesian": {"color": "#2196F3", "label": "Bayesian\n5×151 relu"},
    "nsga2":    {"color": "#F44336", "label": "NSGA-II\n3×153 tanh"},
    "nsga3":    {"color": "#4CAF50", "label": "NSGA-III\n3×75 tanh"},
}


def load_comparison(results_dir: str) -> dict:
    """Read comparison JSON — run2/comparison.json."""
    p = Path(results_dir) / "comparison.json"
    if not p.exists():
        # Fall back to individual results.json files
        data = {}
        for opt in ["bayesian", "nsga2", "nsga3"]:
            rp = Path(results_dir) / "quenching" / opt / "results.json"
            if rp.exists():
                with open(rp) as f:
                    data[opt] = json.load(f)
        return data
    with open(p) as f:
        return json.load(f)


# ── Fig 1: L2_rel and MAE comparison ─────────────────────────────────────────

def plot_accuracy(data: dict, out_dir: str) -> None:
    """
    L2_rel and MAE_C — grouped bar chart for three optimizers.
    """
    opts   = [o for o in ["bayesian", "nsga2", "nsga3"] if o in data]
    x      = np.arange(len(opts))
    bar_w  = 0.35
    colors = [STYLE[o]["color"] for o in opts]
    labels = [STYLE[o]["label"] for o in opts]

    l2_vals  = [data[o]["paper_baseline"]["thermal_metrics"]["L2_rel"]  for o in opts]
    mae_vals = [data[o]["paper_baseline"]["thermal_metrics"]["MAE_C"]   for o in opts]
    max_vals = [data[o]["paper_baseline"]["thermal_metrics"]["MaxErr_C"] for o in opts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Level 1 — Single-Shot NAS-PINN: Thermal Accuracy vs Baseline Paper",
                 fontsize=12, fontweight="bold")

    # L2_rel
    bars = ax1.bar(x, l2_vals, bar_w * 1.4, color=colors, alpha=0.85, edgecolor="white")
    ax1.axhline(0.1, color="black", linestyle="--", linewidth=1.2,
                label="Target L2_rel = 0.1")
    for bar, v in zip(bars, l2_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("L2 Relative Error", fontsize=11)
    ax1.set_title("L2_rel vs Baseline Paper FEM", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_ylim(0, max(l2_vals) * 1.3)

    # MAE_C
    bars2 = ax2.bar(x, mae_vals, bar_w * 1.4, color=colors, alpha=0.85, edgecolor="white")
    ax2.axhline(50, color="black", linestyle="--", linewidth=1.2,
                label="Target MAE = 50 °C")
    # MaxErr error bars
    errs = [m - mae for m, mae in zip(max_vals, mae_vals)]
    ax2.errorbar(x, mae_vals, yerr=[np.zeros(len(opts)), errs],
                 fmt="none", ecolor="gray", capsize=4, linewidth=1.5)
    for bar, v in zip(bars2, mae_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 2,
                 f"{v:.1f}°C", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("MAE (°C)", fontsize=11)
    ax2.set_title("MAE vs Baseline Paper FEM (error bar = MaxErr)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_ylim(0, max(max_vals) * 1.25)

    plt.tight_layout()
    out_path = Path(out_dir) / "level1_accuracy.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Fig 2: Runtime (NAS + train) ─────────────────────────────────────────────

def plot_runtime(data: dict, out_dir: str) -> None:
    """
    Stacked bar: NAS search time + Adam training time.
    """
    opts   = [o for o in ["bayesian", "nsga2", "nsga3"] if o in data]
    x      = np.arange(len(opts))
    bar_w  = 0.45
    colors = [STYLE[o]["color"] for o in opts]
    labels = [STYLE[o]["label"] for o in opts]

    nas_t   = [data[o]["nas_time_s"]                                     for o in opts]
    train_t = [data[o]["train_results"]["adam"]["train_time"]             for o in opts]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Level 1 — NAS + Training Runtime", fontsize=12, fontweight="bold")

    bars1 = ax.bar(x, nas_t,   bar_w, label="NAS search time (s)",  color=colors, alpha=0.85)
    bars2 = ax.bar(x, train_t, bar_w, label="Adam training time (s)", color=colors, alpha=0.45,
                   hatch="///", bottom=nas_t)

    for i in range(len(opts)):
        total = nas_t[i] + train_t[i]
        ax.text(x[i], total + 15, f"{total:.0f}s", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_dir) / "level1_runtime.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Fig 3: Summary table figure ───────────────────────────────────────────────

def plot_summary_table(data: dict, out_dir: str) -> None:
    """
    Summary as a text table figure.
    """
    opts = [o for o in ["bayesian", "nsga2", "nsga3"] if o in data]

    rows = []
    for o in opts:
        cfg = data[o]["best_config"]
        tm  = data[o]["paper_baseline"]["thermal_metrics"]
        arch_str = f"{cfg['n_layers']}×{cfg['neurons'][0]} {cfg['activation']}"
        rows.append([
            STYLE[o]["label"].replace("\n", " "),
            arch_str,
            f"{cfg.get('objectives', {}).get('n_params', sum(cfg['neurons']) * cfg['n_layers'] + cfg['n_layers']):.0f}",
            f"{tm['L2_rel']:.3f}",
            f"{tm['MAE_C']:.1f}",
            f"{tm['MaxErr_C']:.1f}",
            f"{data[o]['nas_time_s']:.0f}",
        ])

    # Target threshold row — acceptable accuracy limits
    rows.append([
        "★ Target (acceptable)",
        "—",
        "—",
        "< 0.100",
        "< 50.0",
        "—",
        "—",
    ])

    col_labels = ["Optimizer", "Architecture", "Params", "L2_rel↓", "MAE (°C)↓", "MaxErr (°C)↓", "NAS time (s)"]

    fig, ax = plt.subplots(figsize=(13, 3.0))
    fig.suptitle("Level 1 — NAS-PINN Results Summary vs Baseline Paper [2026]",
                 fontsize=12, fontweight="bold")
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.0)

    # Header row color
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#37474F")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Bayesian row highlight (best NAS result)
    for j in range(len(col_labels)):
        tbl[1, j].set_facecolor("#E3F2FD")

    # Baseline Paper row highlight
    baseline_row = len(rows)  # last row (1-indexed due to header)
    for j in range(len(col_labels)):
        tbl[baseline_row, j].set_facecolor("#E8F5E9")
        tbl[baseline_row, j].set_text_props(color="#2E7D32", fontweight="bold")

    plt.tight_layout()
    out_path = Path(out_dir) / "level1_summary_table.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Save results as JSON ──────────────────────────────────────────────────────

def save_level1_json(data: dict, out_dir: str) -> None:
    """
    Save Level 1 results in clean JSON format.
    """
    summary = {}
    for opt, res in data.items():
        cfg = res["best_config"]
        tm  = res["paper_baseline"]["thermal_metrics"]
        summary[opt] = {
            "architecture": {
                "n_layers":   cfg["n_layers"],
                "neurons":    cfg["neurons"][0],
                "activation": cfg["activation"],
            },
            "n_params":      cfg.get("objectives", {}).get("n_params", None),
            "nas_time_s":    res["nas_time_s"],
            "train_time_s":  res["train_results"]["adam"]["train_time"],
            "thermal_metrics": {
                "L2_rel":    tm["L2_rel"],
                "MAE_C":     tm["MAE_C"],
                "RMSE_C":    tm["RMSE_C"],
                "MaxErr_C":  tm["MaxErr_C"],
            },
        }

    out_path = Path(out_dir) / "level1_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Level 1 — Visualize NAS-PINN single-shot results"
    )
    parser.add_argument("--results_dir", type=str,
                        default=str(Path(__file__).parent / "results"),
                        help="Directory containing comparison.json (default: level1_single_shot/results)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: results_dir/plots)")
    args = parser.parse_args()

    out_dir = args.out or str(Path(args.results_dir) / "plots")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    data = load_comparison(args.results_dir)
    if not data:
        print(f"ERROR: No results found in {args.results_dir}")
        exit(1)

    print(f"Loaded results for: {list(data.keys())}")
    plot_accuracy(data, out_dir)
    plot_runtime(data, out_dir)
    plot_summary_table(data, out_dir)
    save_level1_json(data, out_dir)
    print(f"\nDone — 3 figures + 1 JSON saved to {out_dir}")
