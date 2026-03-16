"""
plot_results.py — Level 3 Hibrit FEM+PINN Grafikleri
======================================================
results/level3_results.json dosyasını okur, dört grafik üretir:

  Fig 1: T_mean vs t — üç optimizer için sıcaklık profili
  Fig 2: FEM / PINN adım dağılımı — bar grafiği
  Fig 3: PDE rezidü vs t — adaptif karar eşiği görünümü
  Fig 4: Distortion vs CMM nokta — Fig17 karşılaştırma taslağı

Kullanım:
    python plot_results.py
    python plot_results.py --input results/level3_results.json --out results/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Her optimizer için renk ve stil — Level 2 ile aynı
STYLE = {
    "bayesian": {"color": "#2196F3", "marker": "o", "linestyle": "-",
                 "label": "Bayesian [5×151 relu]"},
    "nsga2":    {"color": "#F44336", "marker": "s", "linestyle": "--",
                 "label": "NSGA-II  [3×153 tanh]"},
    "nsga3":    {"color": "#4CAF50", "marker": "^", "linestyle": ":",
                 "label": "NSGA-III [3×75  tanh]"},
}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_temperature_profile(data: dict, threshold: float, out_dir: str) -> None:
    """
    Fig 1: Her adımda T_mean — FEM adımları işaretli, PINN adımları boş.
    Analitik referans kesik çizgi olarak eklenir.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Level 3 — Temperature Profile: Hybrid FEM + PINN",
                 fontsize=12, fontweight="bold")

    # Analitik referans
    t_ref = np.linspace(0, 30, 200)
    T_ref = 20 + 520 * np.exp(-1.75e-3 * t_ref)
    ax.plot(t_ref, T_ref, "k--", linewidth=1.5, alpha=0.5, label="Analytical ref (α=1.75e-3)")

    for opt_name, res in data.items():
        style   = STYLE.get(opt_name, {})
        hist    = res["history"]
        times   = np.array(hist["times"])
        T_means = np.array(hist["T_means"])
        sources = hist["sources"]

        # Ana çizgi
        ax.plot(times, T_means, color=style["color"], linestyle=style["linestyle"],
                linewidth=1.8, alpha=0.7, label=style.get("label", opt_name))

        # FEM adımlarını büyük marker ile işaretle
        fem_t = [times[i] for i, s in enumerate(sources) if "fem" in s]
        fem_T = [T_means[i] for i, s in enumerate(sources) if "fem" in s]
        ax.scatter(fem_t, fem_T, color=style["color"], marker="D",
                   s=60, zorder=5, alpha=0.9)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Mean Temperature (°C)", fontsize=11)
    ax.set_xlim(0, 30)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Legend ek açıklama
    diamond = mpatches.Patch(color="gray", label="◆ = FEM anchor step")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [diamond], labels + ["◆ = FEM anchor step"], fontsize=9)

    plt.tight_layout()
    out_path = Path(out_dir) / "temperature_profile.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_step_distribution(data: dict, out_dir: str) -> None:
    """
    Fig 2: Her optimizer için FEM / PINN adım sayısı yığılı bar grafiği.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Level 3 — FEM vs PINN Step Distribution",
                 fontsize=12, fontweight="bold")

    opt_names = list(data.keys())
    x         = np.arange(len(opt_names))
    bar_w     = 0.45

    fem_counts  = [data[o]["run_summary"]["fem_steps"]  for o in opt_names]
    pinn_counts = [data[o]["run_summary"]["pinn_steps"] for o in opt_names]

    bars_fem  = ax.bar(x, fem_counts,  bar_w, label="FEM steps",  color="#E57373", alpha=0.9)
    bars_pinn = ax.bar(x, pinn_counts, bar_w, label="PINN steps", color="#64B5F6", alpha=0.9,
                       bottom=fem_counts)

    # Değerleri bar içine yaz
    for bar, val in zip(bars_fem, fem_counts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val / 2, str(val),
                    ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    for bar, f, p in zip(bars_pinn, fem_counts, pinn_counts):
        if p > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, f + p / 2, str(p),
                    ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    # Skip rate etiketi
    for i, o in enumerate(opt_names):
        skip_pct = data[o]["run_summary"]["skip_rate_pct"]
        total    = fem_counts[i] + pinn_counts[i]
        ax.text(i, total + 0.3, f"skip {skip_pct}%", ha="center", fontsize=10, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([STYLE.get(o, {}).get("label", o) for o in opt_names], fontsize=9)
    ax.set_ylabel("Number of Time Steps", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(fem_counts[i] + pinn_counts[i] for i in range(len(opt_names))) + 3)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_dir) / "step_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_residual_trace(data: dict, threshold: float, out_dir: str) -> None:
    """
    Fig 3: PDE rezidü vs zaman — adaptif atlama eşiği gösterilir.
    Eşiğin üstündeki noktalar FEM, altındakiler PINN.
    """
    fig, axes = plt.subplots(1, len(data), figsize=(14, 4), sharey=True)
    fig.suptitle(f"Level 3 — PDE Residual vs Time  (threshold={threshold})",
                 fontsize=12, fontweight="bold")

    for ax, (opt_name, res) in zip(axes, data.items()):
        style     = STYLE.get(opt_name, {})
        hist      = res["history"]
        times     = np.array(hist["times"][1:])   # t=0 rezidüsü yok
        residuals = np.array(hist["residuals"][1:])
        sources   = hist["sources"][1:]

        # Renk kodlu scatter: kırmızı=FEM, mavi=PINN
        colors = ["#E53935" if "fem" in s else "#1E88E5" for s in sources]
        ax.scatter(times, residuals, c=colors, s=40, zorder=5, alpha=0.8)
        ax.plot(times, residuals, color=style["color"], linewidth=1, alpha=0.4)

        # Eşik çizgisi
        ax.axhline(threshold, color="orange", linestyle="--", linewidth=1.5,
                   label=f"threshold={threshold}")

        ax.set_title(style.get("label", opt_name), fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("PDE Residual (normalized)", fontsize=10)

    # Ortak legend
    fem_patch  = mpatches.Patch(color="#E53935", label="FEM step")
    pinn_patch = mpatches.Patch(color="#1E88E5", label="PINN step")
    thr_line   = plt.Line2D([0], [0], color="orange", linestyle="--", label=f"threshold={threshold}")
    fig.legend(handles=[fem_patch, pinn_patch, thr_line], loc="lower center",
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    out_path = Path(out_dir) / "residual_trace.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_distortion_cmm(data: dict, out_dir: str) -> None:
    """
    Fig 4: CMM nokta distortion — üç optimizer karşılaştırması.
    Paper FEM / measured verisi eklenince güncellenir.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Level 3 — Distortion at CMM Points (Fig17 comparison)",
                 fontsize=12, fontweight="bold")

    opt_names = list(data.keys())
    n_opts    = len(opt_names)
    n_points  = len(data[opt_names[0]]["distortion_cmm"]["per_point"])
    x         = np.arange(n_points)
    bar_w     = 0.25

    for i, opt_name in enumerate(opt_names):
        style   = STYLE.get(opt_name, {})
        points  = data[opt_name]["distortion_cmm"]["per_point"]
        vals    = [p["pinn_mm"] for p in points]
        offset  = (i - n_opts / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals, bar_w, label=style.get("label", opt_name),
               color=style["color"], alpha=0.85, edgecolor="white")

    labels = [p["label"] for p in data[opt_names[0]]["distortion_cmm"]["per_point"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Distortion (mm)", fontsize=11)
    ax.set_xlabel("CMM Point (Fig17)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8)

    # Uyarı notu — placeholder veri
    ax.text(0.01, 0.97,
            "Note: paper FEM/measured data not yet loaded — distortion is first-order estimate only",
            transform=ax.transAxes, fontsize=8, color="gray", va="top")

    plt.tight_layout()
    out_path = Path(out_dir) / "distortion_cmm.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Level 3 — Visualize results")
    parser.add_argument("--input",     default=None,
                        help="level3_results.json yolu (varsayilan: --dir/level3_results.json)")
    parser.add_argument("--dir",       default="results/thr0.1_skip4",
                        help="Sonuc klasoru (hem input hem output icin)")
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    # --input verilmemişse --dir'den türet
    if args.input is None:
        args.input = f"{args.dir}/level3_results.json"
    args.out = args.dir

    if not Path(args.input).exists():
        print(f"ERROR: {args.input} bulunamadi. Once main_level3.py calistir.")
        exit(1)

    data = load_results(args.input)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    plot_temperature_profile(data, args.threshold, args.out)
    plot_step_distribution(data, args.out)
    plot_residual_trace(data, args.threshold, args.out)
    plot_distortion_cmm(data, args.out)
    print("Done — 4 figures saved.")
