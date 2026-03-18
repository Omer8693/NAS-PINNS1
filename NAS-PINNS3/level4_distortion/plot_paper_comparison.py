"""
plot_paper_comparison.py — Paper Fig17/18 Style Comparison Charts
==================================================================
Generates signed distortion bar charts as in the baseline paper.
Shows Bayesian PINN predictions alongside Fig17 (measured) and Fig18 (FEM).

Outputs:
  Fig 1: paper_comparison_signed.png  — signed bar chart (paper style)
  Fig 2: paper_comparison_abs.png     — absolute values + error bars
  Fig 3: paper_mae_summary.png        — MAE summary for all optimizers
  all_results_summary.json            — Level 1-4 + paper data in one JSON

Usage:
    python plot_paper_comparison.py
    python plot_paper_comparison.py --out results/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paper data (manually digitized from Fig17/18 bar charts, ±0.05 mm) ───────

CMM_LABELS = [
    "L1206", "R2206", "LS01",  "RS01",  "LS13",  "RS13",
    "LA518", "RA518", "LA525", "RA525", "LA537", "RA537",
    "LA538", "RA538", "LA550", "LA551", "LS35",  "RS35",
    "LS51",  "RS51",  "LS52",  "RS52",  "LA555", "RA555",
]

# Fig17 — Experimental results, Bottom layer [mm]
PAPER_MEAS = np.array([
    -0.60, -0.20, +1.20, +1.10, +0.15, -0.05,
    -0.15, -0.05, +1.35, +1.00, -0.10, -0.10,
    +2.10, -1.00, +0.90, -0.20, -2.05, -2.00,
    +1.15, +1.10, -0.30, -0.20, -1.00, +0.70,
])

# Fig18 — Modelling results, Bottom layer [mm]
PAPER_FEM = np.array([
    -0.10, -0.10, +1.10, +1.10, +0.40, +0.40,
    -0.10, -0.10, +1.10, +1.00, +0.50, +0.50,
    +2.10, -0.50, +0.75, -0.20, -2.00, -2.00,
    +0.95, +0.95, -0.30, -0.25, -0.90, +0.55,
])

# Color scheme — close to the paper
COL_MEAS  = "#2E7D32"   # dark green  — measured
COL_FEM   = "#E65100"   # orange      — FEM modelling
COL_BAYES = "#1565C0"   # blue        — Bayesian PINN
COL_NSGA2 = "#C62828"   # red         — NSGA-II PINN
COL_NSGA3 = "#6A1B9A"   # purple      — NSGA-III PINN


def load_level4(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


# ── Fig 1: Paper-style signed bar chart ──────────────────────────────────────

def plot_paper_style(l4_data: dict, out_dir: str) -> None:
    """
    Fig17 / Fig18 style: signed distortion on y-axis, CMM points on x-axis.
    Shows: Paper Measured | Paper FEM | Bayesian PINN (best)
    Bayesian PINN values are unsigned (|δ|) so they are shown in a separate panel.
    """
    x = np.arange(len(CMM_LABELS))
    bar_w = 0.25

    # Bayesian PINN values — unsigned (direction unknown)
    bayes_pts   = l4_data.get("bayesian", {}).get("per_point", [])
    bayes_delta = np.array([p["delta_mm"] for p in bayes_pts]) if bayes_pts else np.zeros(24)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                    gridspec_kw={"height_ratios": [3, 1.5]})
    fig.suptitle(
        "Distortion at CMM Points — Comparison with Paper (baseline_paper [2026])\n"
        "Bottom Layer | Heat Treatment with Quenching in Water Tank",
        fontsize=13, fontweight="bold"
    )

    # ── Panel 1: Paper Measured + Paper FEM (signed, paper style) ────────────
    ax1.bar(x - bar_w, PAPER_MEAS, bar_w,
            color=COL_MEAS, alpha=0.9, label="Fig17 Measured (Experimental)")
    ax1.bar(x,          PAPER_FEM,  bar_w,
            color=COL_FEM,  alpha=0.9, label="Fig18 Modelling (FEM)")

    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlim(-0.6, len(CMM_LABELS) - 0.4)
    ax1.set_ylim(-2.8, 2.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(CMM_LABELS, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Distortion [mm]", fontsize=11)
    ax1.set_title("Paper Results (signed, from CAD reference)", fontsize=10, style="italic")
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, axis="y", alpha=0.3)

    # Paper FEM vs Measured difference lines
    for i in range(len(CMM_LABELS)):
        ax1.plot([x[i], x[i]], [PAPER_MEAS[i], PAPER_FEM[i]],
                 color="gray", linewidth=0.8, alpha=0.5, zorder=2)

    # ── Panel 2: PINN |δ| — all optimizers ──────────────────────────────────
    for k, (opt, col, label) in enumerate([
        ("bayesian", COL_BAYES, "Bayesian [5×151 relu]"),
        ("nsga2",    COL_NSGA2, "NSGA-II  [3×153 tanh]"),
        ("nsga3",    COL_NSGA3, "NSGA-III [3×75  tanh]"),
    ]):
        pts = l4_data.get(opt, {}).get("per_point", [])
        if not pts:
            continue
        vals   = np.array([p["delta_mm"] for p in pts])
        offset = (k - 1) * (bar_w * 0.8)
        ax2.bar(x + offset, vals, bar_w * 0.8,
                color=col, alpha=0.85, label=label, edgecolor="white")

    # Paper absolute value reference lines
    ax2.plot(x, np.abs(PAPER_MEAS), "k-",  linewidth=1.8, marker="o",
             markersize=4, label="Paper Measured |δ|", zorder=10)
    ax2.plot(x, np.abs(PAPER_FEM),  "k--", linewidth=1.3, marker="s",
             markersize=3, label="Paper FEM |δ|",      zorder=10, alpha=0.7)

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlim(-0.6, len(CMM_LABELS) - 0.4)
    ax2.set_ylim(0, 9.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(CMM_LABELS, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("|δ| [mm]", fontsize=11)
    ax2.set_title("NAS-PINN Results (unsigned magnitude) vs Paper Reference",
                  fontsize=10, style="italic")
    ax2.legend(fontsize=8, ncol=3, loc="upper right")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_dir) / "paper_comparison_signed.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Fig 2: MAE summary — all optimizers ──────────────────────────────────────

def plot_mae_summary(l4_data: dict, out_dir: str) -> None:
    """
    MAE bar chart per optimizer (vs Fig17 Measured and vs Fig18 FEM).
    Paper's internal MAE (FEM vs Measured = 0.179 mm) shown as reference line.
    """
    opts   = []
    maes_m = []
    maes_f = []
    colors = [COL_BAYES, COL_NSGA2, COL_NSGA3]
    labels_opt = ["Bayesian\n5×151 relu", "NSGA-II\n3×153 tanh", "NSGA-III\n3×75 tanh"]

    for opt in ["bayesian", "nsga2", "nsga3"]:
        pts = l4_data.get(opt, {}).get("per_point", [])
        if not pts:
            continue
        vals = np.array([p["delta_mm"] for p in pts])
        opts.append(opt)
        maes_m.append(float(np.mean(np.abs(vals - np.abs(PAPER_MEAS)))))
        maes_f.append(float(np.mean(np.abs(vals - np.abs(PAPER_FEM)))))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Level 4 — MAE vs Paper Reference\n(NAS-PINN |δ| vs Paper distortions)",
                 fontsize=12, fontweight="bold")

    x     = np.arange(len(opts))
    bar_w = 0.32

    bars1 = ax.bar(x - bar_w/2, maes_m, bar_w,
                   color=colors[:len(opts)], alpha=0.85,
                   label="MAE vs Fig17 Measured")
    bars2 = ax.bar(x + bar_w/2, maes_f, bar_w,
                   color=colors[:len(opts)], alpha=0.45,
                   hatch="///", label="MAE vs Fig18 FEM")

    # Paper internal MAE reference
    paper_internal_mae = 0.179
    ax.axhline(paper_internal_mae, color="black", linestyle="--", linewidth=1.8,
               label=f"Paper FEM vs Measured MAE = {paper_internal_mae:.3f} mm")

    # Value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.03,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_opt, fontsize=10)
    ax.set_ylabel("MAE (mm)", fontsize=11)
    ax.set_ylim(0, max(max(maes_m), max(maes_f)) * 1.25 + 0.3)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_dir) / "paper_mae_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Fig 3: Point-wise error scatter (Bayesian) ───────────────────────────────

def plot_pointwise_error(l4_data: dict, out_dir: str) -> None:
    """
    Bayesian PINN |δ| vs Paper Measured signed distortion — scatter + identity line.
    Each point is a CMM measurement; good prediction → near the diagonal.
    """
    pts = l4_data.get("bayesian", {}).get("per_point", [])
    if not pts:
        return

    pinn_vals = np.array([p["delta_mm"] for p in pts])
    pap_abs   = np.abs(PAPER_MEAS)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle("Bayesian PINN |δ| vs Paper Measured |δ|\n(24 CMM Points — Bottom Layer)",
                 fontsize=12, fontweight="bold")

    sc = ax.scatter(pap_abs, pinn_vals, c=np.arange(len(CMM_LABELS)),
                    cmap="tab20", s=90, zorder=5)

    # Ideal line (1:1)
    lim = max(pap_abs.max(), pinn_vals.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.5, label="Ideal (PINN = Paper)", alpha=0.6)
    ax.plot([0, lim], [0, lim * 2], "r:",  linewidth=1.0, label="2× overestimate", alpha=0.4)

    # Labels
    for i, lbl in enumerate(CMM_LABELS):
        ax.annotate(lbl, (pap_abs[i], pinn_vals[i]),
                    fontsize=6, ha="left", va="bottom", alpha=0.7)

    mae   = float(np.mean(np.abs(pinn_vals - pap_abs)))
    r_val = float(np.corrcoef(pap_abs, pinn_vals)[0, 1])

    ax.set_xlabel("Paper Measured |δ| (mm)", fontsize=11)
    ax.set_ylabel("Bayesian PINN |δ| (mm)", fontsize=11)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.92, f"MAE = {mae:.3f} mm\nr = {r_val:.3f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    out_path = Path(out_dir) / "paper_pointwise_error.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Summary JSON combining all results ───────────────────────────────────────

def save_summary_json(l4_data: dict, out_dir: str) -> None:
    """
    Combines Level 1-4 results + paper reference into a single JSON file.
    """
    # Level 1 (run2 fixed values)
    level1 = {
        "description": "Single-shot global PINN — NAS (run2, 20k Adam)",
        "bayesian": {"arch": "5x151_relu", "L2_rel": 0.076, "MAE_C": 39.1,
                     "MaxErr_C": 54.9, "nas_time_s": 150, "train_time_s": 232},
        "nsga2":    {"arch": "3x153_tanh", "L2_rel": 0.252, "MAE_C": 132.6,
                     "MaxErr_C": 137.9, "nas_time_s": 1583, "train_time_s": 199},
        "nsga3":    {"arch": "3x75_tanh",  "L2_rel": 0.513, "MAE_C": 270.3,
                     "MaxErr_C": 278.2, "nas_time_s": 1609, "train_time_s": 214},
    }

    # Level 2 (skip=2 en iyi)
    level2 = {
        "description": "Time-stepping PINN — windowed training, skip=2",
        "bayesian": {"skip": 2, "steps_used": 11, "L2_rel": 0.108, "MAE_C": 33.2,
                     "runtime_s": 90.5, "speedup_vs_skip1": "2.0x"},
        "nsga2":    {"skip": 2, "steps_used": 11, "L2_rel": 0.181, "MAE_C": 56.0,
                     "runtime_s": 79.9, "speedup_vs_skip1": "1.9x"},
        "nsga3":    {"skip": 2, "steps_used": 11, "L2_rel": 0.217, "MAE_C": 75.6,
                     "runtime_s": 74.3, "speedup_vs_skip1": "2.1x"},
    }

    # Level 3 (thr=0.1, skip20 — aggressive)
    level3_skip20 = {
        "description": "Hybrid FEM+PINN — adaptive skip (threshold=0.1, max_skip=20)",
        "bayesian": {"fem_steps": 1, "pinn_steps": 19, "skip_pct": 95.0,
                     "T_final_C": 470.5, "dT_vs_ref_C": -43.1, "wall_time_s": 96.8},
        "nsga2":    {"fem_steps": 0, "pinn_steps": 20, "skip_pct": 100.0,
                     "T_final_C": 398.1, "dT_vs_ref_C": -115.5, "wall_time_s": 80.4},
        "nsga3":    {"fem_steps": 0, "pinn_steps": 20, "skip_pct": 100.0,
                     "T_final_C": 277.6, "dT_vs_ref_C": -236.0, "wall_time_s": 79.4},
    }

    # Level 3 (thr=0.1, skip4 — FEM anchored)
    level3_skip4 = {
        "description": "Hybrid FEM+PINN — adaptive skip (threshold=0.1, max_skip=4)",
        "analytical_ref_C": 513.6,
        "bayesian": {"fem_steps": 4, "pinn_steps": 16, "skip_pct": 80.0,
                     "T_final_C": 469.1, "dT_vs_ref_C": -44.5, "wall_time_s": 111.2},
        "nsga2":    {"fem_steps": 4, "pinn_steps": 16, "skip_pct": 80.0,
                     "T_final_C": 498.5, "dT_vs_ref_C": -15.1, "wall_time_s": 93.4},
        "nsga3":    {"fem_steps": 4, "pinn_steps": 16, "skip_pct": 80.0,
                     "T_final_C": 513.4, "dT_vs_ref_C": -0.2,  "wall_time_s": 91.8},
    }

    # Level 4 distortion
    level4 = {}
    for opt, res in l4_data.items():
        if "error" in res:
            level4[opt] = res
            continue
        pinn_vals = np.array([p["delta_mm"] for p in res["per_point"]])
        level4[opt] = {
            "T_final_C":       res["T_final_mean"],
            "max_delta_mm":    res["delta_max_mm"],
            "mean_delta_mm":   res["delta_mean_mm"],
            "MAE_vs_measured": float(np.mean(np.abs(pinn_vals - np.abs(PAPER_MEAS)))),
            "MAE_vs_FEM":      float(np.mean(np.abs(pinn_vals - np.abs(PAPER_FEM)))),
        }

    # Paper reference
    paper = {
        "source": "baseline_paper [2026], Fig17 (Experimental) and Fig18 (Modelling)",
        "layer": "Bottom",
        "note": "Values digitized from bar charts, precision ±0.05 mm",
        "measured_mean_abs_mm": float(np.abs(PAPER_MEAS).mean()),
        "measured_max_abs_mm":  float(np.abs(PAPER_MEAS).max()),
        "fem_mean_abs_mm":      float(np.abs(PAPER_FEM).mean()),
        "fem_max_abs_mm":       float(np.abs(PAPER_FEM).max()),
        "mae_fem_vs_measured":  float(np.mean(np.abs(PAPER_FEM - PAPER_MEAS))),
        "per_point": [
            {
                "label":        CMM_LABELS[i],
                "measured_mm":  float(PAPER_MEAS[i]),
                "fem_mm":       float(PAPER_FEM[i]),
                "diff_mm":      float(PAPER_MEAS[i] - PAPER_FEM[i]),
            }
            for i in range(len(CMM_LABELS))
        ],
    }

    summary = {
        "project":       "NAS-PINN Quenching — baseline_paper [2026]",
        "date":          "2026-03-15",
        "paper_reference": paper,
        "level1_single_shot":  level1,
        "level2_timestepper":  level2,
        "level3_hybrid_skip20": level3_skip20,
        "level3_hybrid_skip4":  level3_skip4,
        "level4_distortion":   level4,
    }

    out_path = Path(out_dir) / "all_results_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paper-style comparison plots + summary JSON"
    )
    parser.add_argument("--input", type=str,
                        default="results/level4_distortion.json",
                        help="Level 4 distortion JSON")
    parser.add_argument("--out",   type=str, default="results",
                        help="Output directory")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: {args.input} not found. Run main_distortion.py first.")
        exit(1)

    l4_data = load_level4(args.input)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    plot_paper_style(l4_data, args.out)
    plot_mae_summary(l4_data, args.out)
    plot_pointwise_error(l4_data, args.out)
    save_summary_json(l4_data, args.out)
    print("\nDone — 3 figures + 1 summary JSON saved.")
