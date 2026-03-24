"""
Generate a clean, readable skip-operator timeline figure.
2×2 grid: one panel per domain, 4 lines per panel (skip=1,2,4,6).
Saves: web/static/img/fig_timeline_clean.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.join(BASE, "..", "..", "..")
NP3    = os.path.join(ROOT, "NAS-PINNS3")
IMG    = os.path.join(NP3,  "web", "static", "img")
V2_RES = os.path.join(NP3,  "level8_nas_mco_pinn", "results", "v2", "results_3d_v2.json")
OUT    = os.path.join(IMG,   "fig_timeline_clean.png")

with open(V2_RES) as f:
    DATA = json.load(f)

DOMAINS  = ["rectangular", "cylinder", "stacked", "lshape"]
ARCHS    = ["bayesian", "nsga2", "nsga3"]
SKIPS    = [1, 2, 4, 6]
FEM_SAVE = {1: "0%", 2: "52%", 4: "71%", 6: "81%"}
DOM_LBL  = {
    "rectangular": "Rectangular Prism",
    "cylinder":    "Cylinder",
    "stacked":     "Stacked Cubes",
    "lshape":      "L-Shape",
}
SKIP_COL  = {1: "#1565C0", 2: "#2E7D32", 4: "#E65100", 6: "#6A1B9A"}
SKIP_MRKS = {1: "o", 2: "s", 4: "^", 6: "D"}

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif"],
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.patch.set_facecolor("white")

for ax, dom in zip(axes.flat, DOMAINS):
    for sk in SKIPS:
        # Best arch for this (dom, skip)
        best_arch = None
        best_mae  = float("inf")
        for arch in ARCHS:
            v = DATA[dom][arch].get(str(sk), {}).get("mae_C", float("inf"))
            if v < best_mae:
                best_mae  = v
                best_arch = arch

        if best_arch is None:
            continue

        windows = DATA[dom][best_arch][str(sk)].get("mae_windows", [])
        if not windows:
            continue

        x = np.arange(1, len(windows) + 1)
        label = f"skip={sk}  ({FEM_SAVE[sk]} saved)"
        ax.plot(x, windows,
                color=SKIP_COL[sk],
                marker=SKIP_MRKS[sk],
                markersize=5.5,
                linewidth=1.8,
                label=label)

    ax.set_title(DOM_LBL[dom], fontsize=13, fontweight="bold", pad=7)
    ax.set_xlabel("Time Window Index", fontsize=11)
    ax.set_ylabel("MAE (°C)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.axhline(10, color="#aaaaaa", linewidth=0.9, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#e8e8e8", linewidth=0.6)
    ax.legend(fontsize=9.5, framealpha=0.85, loc="upper right")

fig.suptitle(
    "Skip Operator: MAE per Time Window  (best NAS optimizer per configuration)",
    fontsize=13.5, y=1.02)
fig.tight_layout(pad=1.6)
fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
