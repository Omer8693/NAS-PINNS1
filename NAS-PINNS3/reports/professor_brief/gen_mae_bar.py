"""
Generate a clean, large, readable MAE bar chart for embedding in reports.
Saves: web/static/img/fig_mae_bar_clean.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.join(BASE, "..", "..", "..")
NP3    = os.path.join(ROOT, "NAS-PINNS3")
IMG    = os.path.join(NP3,  "web", "static", "img")
V2_RES = os.path.join(NP3,  "level8_nas_mco_pinn", "results", "v2", "results_3d_v2.json")
OUT    = os.path.join(IMG,   "fig_mae_bar_clean.png")

with open(V2_RES) as f:
    DATA = json.load(f)

DOMAINS  = ["rectangular", "cylinder", "stacked", "lshape"]
ARCHS    = ["bayesian", "nsga2", "nsga3"]
SKIPS    = [1, 2, 4, 6]
DOM_LBL  = {"rectangular": "Rectangular", "cylinder": "Cylinder",
             "stacked": "Stacked", "lshape": "L-Shape"}
FEM_SAVE = {1: "0%", 2: "52%", 4: "71%", 6: "81%"}

dom_colors = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"]

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif"],
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor("white")

x = np.arange(len(SKIPS))
w = 0.19

for di, (dom, col) in enumerate(zip(DOMAINS, dom_colors)):
    best = []
    for sk in SKIPS:
        vals = [DATA[dom][a].get(str(sk), {}).get("mae_C", np.nan) for a in ARCHS]
        vals = [v for v in vals if not np.isnan(v)]
        best.append(min(vals) if vals else np.nan)

    offset = (di - 1.5) * w
    bars = ax.bar(x + offset, best, w,
                  label=DOM_LBL[dom], color=col, alpha=0.85,
                  edgecolor="white", linewidth=0.7)
    for bar, val in zip(bars, best):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{val:.1f}",
                    ha="center", va="bottom",
                    fontsize=9.5, color=col, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(
    [f"skip = {s}\n({FEM_SAVE[s]} FEM saved)" for s in SKIPS],
    fontsize=12)
ax.set_ylabel("MAE (°C)", fontsize=13)
ax.set_title("Best MAE per Domain and Skip Value\n(minimum over Bayesian, NSGA-II, NSGA-III)",
             fontsize=13, pad=12)
ax.set_ylim(0, 33)
ax.axhline(10, color="#888888", linewidth=1.1, linestyle="--")
ax.text(3.62, 10.7, "10 °C threshold", fontsize=10, color="#666666", va="bottom")
ax.legend(fontsize=11, framealpha=0.88, loc="upper left", ncol=2,
          handlelength=1.4, handletextpad=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.7)
ax.tick_params(axis="y", labelsize=11)

fig.tight_layout(pad=1.0)
fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
