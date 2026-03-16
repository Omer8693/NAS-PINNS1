"""
plot_results.py — Level 4 Distortion Görselleştirme
=====================================================
level4_distortion.json dosyasını okur, dört grafik üretir:

  Fig 1: CMM nokta distortion bar chart — üç optimizer karşılaştırması
  Fig 2: u_x ve u_y bileşenleri ayrı subplotlar
  Fig 3: T_final vs max_delta scatter — optimizer performansı
  Fig 4: CMM nokta distortion haritası

Kullanım:
    python plot_results.py
    python plot_results.py --input results/level4_distortion.json --out results/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Her optimizer için renk/stil — Level 2/3 ile tutarlı
STYLE = {
    "bayesian": {"color": "#2196F3", "label": "Bayesian [5×151 relu]"},
    "nsga2":    {"color": "#F44336", "label": "NSGA-II  [3×153 tanh]"},
    "nsga3":    {"color": "#4CAF50", "label": "NSGA-III [3×75  tanh]"},
}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Fig 1: CMM nokta distortion bar chart ────────────────────────────────────

def plot_cmm_distortion(data: dict, out_dir: str) -> None:
    """
    Her CMM noktasında üç optimizer'ın |δ| değerleri — yığılmamış bar grubu.
    """
    valid = {k: v for k, v in data.items() if "error" not in v}
    if not valid:
        print("No valid results — skipping plot")
        return

    opt_names = list(valid.keys())
    n_opts    = len(opt_names)

    labels   = [p["label"] for p in valid[opt_names[0]]["per_point"]]
    n_points = len(labels)
    x        = np.arange(n_points)
    bar_w    = 0.8 / n_opts

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle("Level 4 — CMM Point Distortion |δ| (mm) — Optimizer Comparison",
                 fontsize=12, fontweight="bold")

    for k, opt_name in enumerate(opt_names):
        style  = STYLE.get(opt_name, {"color": "gray", "label": opt_name})
        deltas = [p["delta_mm"] for p in valid[opt_name]["per_point"]]
        offset = (k - n_opts / 2.0 + 0.5) * bar_w
        ax.bar(x + offset, deltas, bar_w,
               label=style["label"],
               color=style["color"], alpha=0.85, edgecolor="white")

    # Paper Fig17 (measured) ve Fig18 (FEM modelling) — mutlak değer olarak çiz
    paper_fem  = [abs(p["paper_fem_mm"])  for p in valid[opt_names[0]]["per_point"]]
    paper_meas = [abs(p["paper_meas_mm"]) for p in valid[opt_names[0]]["per_point"]]
    if any(v != 0 for v in paper_meas):
        ax.plot(x, paper_meas, "k-",  linewidth=2.0, marker="o", markersize=4,
                label="Paper Fig17 Measured |δ|", zorder=10)
    if any(v != 0 for v in paper_fem):
        ax.plot(x, paper_fem,  "k--", linewidth=1.5, marker="s", markersize=3,
                label="Paper Fig18 FEM |δ|",      zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Distortion |δ| (mm)", fontsize=11)
    ax.set_xlabel("CMM Point (Fig17/18)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0.0, color="black", linewidth=0.8)

    ax.text(0.01, 0.97,
            "Note: Paper values digitized from Fig17/18 bar charts (±0.05 mm precision). "
            "PINN |δ| = displacement magnitude; Paper = |signed distortion from CAD|.",
            transform=ax.transAxes, fontsize=7.5, color="gray", va="top")

    plt.tight_layout()
    out_path = Path(out_dir) / "cmm_distortion.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Fig 2: u_x ve u_y bileşenleri ────────────────────────────────────────────

def plot_displacement_components(data: dict, out_dir: str) -> None:
    """
    CMM noktalarında u_x (horizontal) ve u_y (vertical) bileşenleri.
    """
    valid = {k: v for k, v in data.items() if "error" not in v}
    if not valid:
        return

    opt_names = list(valid.keys())
    n_opts    = len(opt_names)
    labels    = [p["label"] for p in valid[opt_names[0]]["per_point"]]
    n_points  = len(labels)
    x         = np.arange(n_points)
    bar_w     = 0.8 / n_opts

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle("Level 4 — Displacement Components at CMM Points",
                 fontsize=12, fontweight="bold")

    for k, opt_name in enumerate(opt_names):
        style  = STYLE.get(opt_name, {"color": "gray", "label": opt_name})
        ux_mm  = [p["u_x_mm"] for p in valid[opt_name]["per_point"]]
        uy_mm  = [p["u_y_mm"] for p in valid[opt_name]["per_point"]]
        offset = (k - n_opts / 2.0 + 0.5) * bar_w

        ax1.bar(x + offset, ux_mm, bar_w, label=style["label"],
                color=style["color"], alpha=0.85, edgecolor="white")
        ax2.bar(x + offset, uy_mm, bar_w, label=style["label"],
                color=style["color"], alpha=0.85, edgecolor="white")

    for ax, ylabel in [(ax1, "u_x (mm)"), (ax2, "u_y (mm)")]:
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("CMM Point", fontsize=11)

    plt.tight_layout()
    out_path = Path(out_dir) / "displacement_components.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Fig 3: T_final vs max_delta scatter ──────────────────────────────────────

def plot_temperature_vs_distortion(data: dict, out_dir: str) -> None:
    """
    T_final (ortalama) vs max CMM distortion — optimizer pozisyon haritası.
    """
    valid = {k: v for k, v in data.items() if "error" not in v}
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Level 4 — T_final vs Max Distortion",
                 fontsize=12, fontweight="bold")

    for opt_name, res in valid.items():
        style = STYLE.get(opt_name, {"color": "gray", "label": opt_name})
        ax.scatter(res["T_final_mean"], res["delta_max_mm"],
                   color=style["color"], s=200, zorder=5,
                   label=style["label"], edgecolors="black", linewidths=1.5)
        ax.annotate(
            f"  {opt_name}",
            (res["T_final_mean"], res["delta_max_mm"]),
            fontsize=9, color=style["color"]
        )

    ax.set_xlabel("T_final (°C) — Level 3 mean", fontsize=11)
    ax.set_ylabel("Max CMM Distortion |δ| (mm)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_dir) / "temp_vs_distortion.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Fig 4: CMM nokta haritası ─────────────────────────────────────────────────

def plot_cmm_map(data: dict, out_dir: str) -> None:
    """
    CMM noktalarını domain üzerinde yerleştir,
    renk kodlu |δ| büyüklükleri göster.
    """
    valid = {k: v for k, v in data.items() if "error" not in v}
    if not valid:
        return

    opt_names = list(valid.keys())
    n_opts    = len(opt_names)

    fig, axes = plt.subplots(1, n_opts, figsize=(6 * n_opts, 5))
    if n_opts == 1:
        axes = [axes]
    fig.suptitle("Level 4 — CMM Point Distortion Map",
                 fontsize=12, fontweight="bold")

    # Tüm optimizer'lardaki maksimum δ — renk skalası için
    vmax = max(
        max(p["delta_mm"] for p in v["per_point"])
        for v in valid.values()
    )

    for ax, opt_name in zip(axes, opt_names):
        style  = STYLE.get(opt_name, {"color": "gray", "label": opt_name})
        pts    = valid[opt_name]["per_point"]
        xs     = [p["x_m"]      for p in pts]
        ys     = [p["y_m"]      for p in pts]
        deltas = [p["delta_mm"] for p in pts]

        # Domain dikdörtgeni
        rect = plt.Rectangle((0, 0), 1.3, 0.6, fill=False,
                              edgecolor="gray", linewidth=2, linestyle="--")
        ax.add_patch(rect)

        sc = ax.scatter(xs, ys, c=deltas, cmap="hot_r", s=120,
                        vmin=0, vmax=vmax, zorder=5)
        plt.colorbar(sc, ax=ax, label="|δ| (mm)", shrink=0.8)

        for p in pts:
            ax.annotate(p["label"], (p["x_m"], p["y_m"]),
                        fontsize=5, ha="center", va="bottom", color="black")

        ax.set_xlim(-0.05, 1.35)
        ax.set_ylim(-0.08, 0.68)
        ax.set_title(style["label"], fontsize=10)
        ax.set_xlabel("x [m]", fontsize=9)
        ax.set_ylabel("y [m]", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = Path(out_dir) / "cmm_map.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Giriş noktası ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Level 4 — Distortion Visualization")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to level4_distortion.json")
    parser.add_argument("--out",   type=str, default="results",
                        help="Output directory")
    args = parser.parse_args()

    input_path = args.input or f"{args.out}/level4_distortion.json"

    if not Path(input_path).exists():
        print(f"ERROR: {input_path} not found. Run main_distortion.py first.")
        exit(1)

    data = load_results(input_path)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    plot_cmm_distortion(data, args.out)
    plot_displacement_components(data, args.out)
    plot_temperature_vs_distortion(data, args.out)
    plot_cmm_map(data, args.out)
    print("Done — 4 figures saved.")
