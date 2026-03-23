"""
plot_skip_results.py — Level 8 Skip Comparison Figures
=======================================================
Run:
    cd /home/coder/NAS-PINNS1/NAS-PINNS3
    python -m level8_nas_mco_pinn.plot_skip_results
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "level8_skip_results.json")
OUT     = os.path.join(os.path.dirname(__file__), "..", "results")

with open(RESULTS) as f:
    D = json.load(f)

L2 = D["level2_ref"]    # {arch: {skip: mae}}  — only 3 NAS archs
L8 = D["level8_mco"]    # {arch: {skip: {mae_C, mae_per_window, runtime_s}}}

# ── Architectures ──────────────────────────────────────────────
NAS_ARCHS  = ["bayesian", "nsga2", "nsga3"]          # have L2 baseline
MCO_ARCHS  = ["bayesian", "nsga2", "nsga3", "mco_shen"]  # MCO results only

ARCH_LABEL = {
    "bayesian":  "Bayesian (TPE)",
    "nsga2":     "NSGA-II",
    "nsga3":     "NSGA-III",
    "mco_shen":  "Shen-Arch",
}
ARCH_COLOR = {
    "bayesian":  "#1565C0",
    "nsga2":     "#2E7D32",
    "nsga3":     "#E65100",
    "mco_shen":  "#6A1B9A",
}
SKIPS      = [1, 2, 4, 6]
SKIP_STEPS = {1: "21/21", 2: "11/21", 4: "6/21", 6: "4/21"}

# ── Data accessors ──────────────────────────────────────────────
def l2_mae(arch, skip):    return float(L2[arch][str(skip)])
def l8_mae(arch, skip):    return float(L8[arch][str(skip)]["mae_C"])
def l8_win(arch, skip):    return L8[arch][str(skip)]["mae_per_window"]
def l8_rt(arch, skip):     return float(L8[arch][str(skip)]["runtime_s"])


# ══════════════════════════════════════════════════════════════
# Fig 1 — Main bar comparison: L2 vs L8 MCO (NAS archs only)
# ══════════════════════════════════════════════════════════════
def fig_bar_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle(
        "NAS-PINN L2 (fixed weights) vs NAS-MCO-PINN L8 (adaptive weights)\n"
        "A356 Al quenching — Time-step skip operator | MAE [°C]",
        fontsize=13, fontweight="bold"
    )

    ax = axes[0]
    x  = np.arange(len(SKIPS))
    w  = 0.13
    offsets = np.linspace(-(len(NAS_ARCHS)-1)*w, (len(NAS_ARCHS)-1)*w, len(NAS_ARCHS)*2)

    for ai, arch in enumerate(NAS_ARCHS):
        col     = ARCH_COLOR[arch]
        l2_vals = [l2_mae(arch, s) for s in SKIPS]
        l8_vals = [l8_mae(arch, s) for s in SKIPS]

        ax.bar(x + offsets[ai*2],   l2_vals, w*0.9,
               color=col, alpha=0.30, label=f"{ARCH_LABEL[arch]} L2",
               edgecolor=col, linewidth=0.8)
        bars = ax.bar(x + offsets[ai*2+1], l8_vals, w*0.9,
               color=col, alpha=0.92, label=f"{ARCH_LABEL[arch]} MCO",
               edgecolor="white", linewidth=0.5)

        for bar, v in zip(bars, l8_vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+2, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold", color=col)

    ax.set_xticks(x)
    ax.set_xticklabels([f"skip={s}\n({SKIP_STEPS[s]} FEM)" for s in SKIPS], fontsize=9)
    ax.set_ylabel("MAE [°C]", fontsize=11)
    ax.set_title("All NAS optimizers × skip values", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 480)
    ax.axhline(10, color="gray", lw=1.0, ls="--", alpha=0.6, label="10°C threshold")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left", framealpha=0.85)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

    # Right: zoomed MCO view including Shen-Arch
    ax2 = axes[1]
    for arch in MCO_ARCHS:
        col     = ARCH_COLOR[arch]
        l8_vals = [l8_mae(arch, s) for s in SKIPS]
        ls      = "--" if arch == "mco_shen" else "-"
        ax2.plot(SKIPS, l8_vals, f"o{ls}", color=col, lw=2, ms=7,
                 label=ARCH_LABEL[arch])
        for s, v in zip(SKIPS, l8_vals):
            ax2.annotate(f"{v:.1f}°C", (s, v), textcoords="offset points",
                         xytext=(5, 3), fontsize=7, color=col)

    ax2.set_xticks(SKIPS)
    ax2.set_xticklabels([f"skip={s}" for s in SKIPS], fontsize=9)
    ax2.set_ylabel("MAE [°C]", fontsize=11)
    ax2.set_title("MCO zoomed (incl. Shen-Arch)\n0–12°C", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 12)
    ax2.axhline(5, color="#E65100", lw=1.0, ls=":", alpha=0.7, label="5°C target")
    ax2.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.spines[["top","right"]].set_visible(False)

    fig.tight_layout()
    p = os.path.join(OUT, "level8_bar_comparison.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(p)}")


# ══════════════════════════════════════════════════════════════
# Fig 2 — MAE Heatmap: L2 left / L8 MCO right (all 4 archs)
# ══════════════════════════════════════════════════════════════
def fig_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "MAE Heatmap: NAS-PINN L2 (fixed) vs NAS-MCO-PINN L8 (adaptive)\n"
        "Values in °C  |  Column = skip  |  Row = architecture",
        fontsize=12, fontweight="bold"
    )

    # Left: L2 — only 3 NAS archs
    ax = axes[0]
    mat_l2 = np.array([[l2_mae(a, s) for s in SKIPS] for a in NAS_ARCHS])
    im_l2  = ax.imshow(mat_l2, cmap="YlOrRd", vmin=0, vmax=430, aspect="auto")
    for ri, arch in enumerate(NAS_ARCHS):
        for ci, s in enumerate(SKIPS):
            v = mat_l2[ri, ci]
            c = "white" if v > 200 else "black"
            ax.text(ci, ri, f"{v:.1f}°C", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=c)
    ax.set_xticks(range(len(SKIPS)))
    ax.set_xticklabels([f"skip={s}\n{SKIP_STEPS[s]} FEM" for s in SKIPS], fontsize=9)
    ax.set_yticks(range(len(NAS_ARCHS)))
    ax.set_yticklabels([ARCH_LABEL[a] for a in NAS_ARCHS], fontsize=10)
    ax.set_title("NAS-PINN L2  (Level 2 — fixed weights)", fontsize=10, fontweight="bold", pad=8)
    cb = fig.colorbar(im_l2, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("MAE [°C]", fontsize=9); cb.ax.tick_params(labelsize=8)

    # Right: L8 MCO — all 4 archs
    ax = axes[1]
    mat_l8 = np.array([[l8_mae(a, s) for s in SKIPS] for a in MCO_ARCHS])
    im_l8  = ax.imshow(mat_l8, cmap="YlGn", vmin=0, vmax=10, aspect="auto")
    for ri, arch in enumerate(MCO_ARCHS):
        for ci, s in enumerate(SKIPS):
            v = mat_l8[ri, ci]
            c = "white" if v > 7 else "black"
            ax.text(ci, ri, f"{v:.1f}°C", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=c)
    ax.set_xticks(range(len(SKIPS)))
    ax.set_xticklabels([f"skip={s}\n{SKIP_STEPS[s]} FEM" for s in SKIPS], fontsize=9)
    ax.set_yticks(range(len(MCO_ARCHS)))
    ax.set_yticklabels([ARCH_LABEL[a] for a in MCO_ARCHS], fontsize=10)
    ax.set_title("NAS-MCO-PINN L8  (Level 8 — adaptive weights)", fontsize=10, fontweight="bold", pad=8)
    cb = fig.colorbar(im_l8, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("MAE [°C]", fontsize=9); cb.ax.tick_params(labelsize=8)

    fig.tight_layout()
    p = os.path.join(OUT, "level8_heatmap.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(p)}")


# ══════════════════════════════════════════════════════════════
# Fig 3 — MCO Improvement heatmap ΔMAE = L2 − L8
# ══════════════════════════════════════════════════════════════
def fig_improvement_heatmap():
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.suptitle(
        "MCO Improvement: ΔMAE = L2 − L8 MCO  [°C]\n"
        "Green = MCO better  |  Percentage = reduction vs L2 baseline",
        fontsize=12, fontweight="bold"
    )

    mat = np.array([[l2_mae(a,s) - l8_mae(a,s) for s in SKIPS] for a in NAS_ARCHS])
    im  = ax.imshow(mat, cmap="Greens", vmin=0, vmax=mat.max(), aspect="auto")

    for ri, arch in enumerate(NAS_ARCHS):
        for ci, s in enumerate(SKIPS):
            v   = mat[ri, ci]
            pct = v / max(l2_mae(arch, s), 1) * 100
            c   = "white" if v > mat.max()*0.6 else "black"
            ax.text(ci, ri, f"−{v:.0f}°C\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=10, fontweight="bold", color=c)

    ax.set_xticks(range(len(SKIPS)))
    ax.set_xticklabels([f"skip={s}\n{SKIP_STEPS[s]} FEM" for s in SKIPS], fontsize=9)
    ax.set_yticks(range(len(NAS_ARCHS)))
    ax.set_yticklabels([ARCH_LABEL[a] for a in NAS_ARCHS], fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cb.set_label("ΔMAE [°C]", fontsize=9)

    fig.tight_layout()
    p = os.path.join(OUT, "level8_improvement.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(p)}")


# ══════════════════════════════════════════════════════════════
# Fig 4 — Per-window MAE: skip=2 and skip=4 (all 4 archs)
# ══════════════════════════════════════════════════════════════
def fig_window_mae():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(
        "Per-Window MAE — PINN Error in Each Time Window\n"
        "NAS-MCO-PINN L8 | All 4 architectures",
        fontsize=12, fontweight="bold"
    )

    T_STEPS = np.linspace(0, 30, 21)

    for ax, skip in zip(axes, [2, 4]):
        anchor = T_STEPS[::skip]
        t_mids = [(anchor[i-1] + anchor[i])/2 for i in range(1, len(anchor))]

        for arch in MCO_ARCHS:
            wins = l8_win(arch, skip)
            col  = ARCH_COLOR[arch]
            ls   = "--" if arch == "mco_shen" else "-"
            lbl  = ARCH_LABEL[arch]

            # Flag NSGA-III anomaly
            if arch == "nsga3" and skip == 1:
                lbl += " ⚠ spike"

            ax.plot(t_mids, wins, f"o{ls}", color=col, lw=2, ms=6, label=lbl)
            ax.fill_between(t_mids, wins, alpha=0.08, color=col)

            # Annotate spike windows
            if arch == "nsga3":
                for ti, (t, w) in enumerate(zip(t_mids, wins)):
                    if w > 20:
                        ax.annotate(f"⚠ {w:.0f}°C\n(spike)",
                                    xy=(t, w), xytext=(t+0.5, w+2),
                                    fontsize=7, color=col,
                                    arrowprops=dict(arrowstyle="->", color=col, lw=0.8))

        ax.set_xlabel("Time [s]", fontsize=10)
        ax.set_ylabel("MAE [°C]", fontsize=10)
        ax.set_title(f"skip = {skip}  ({SKIP_STEPS[skip]} FEM steps)",
                     fontsize=11, fontweight="bold")
        ax.set_xlim(0, 30)
        ax.set_ylim(0, max(12, min(ax.get_ylim()[1], 25)))
        ax.axhline(5, color="gray", lw=1, ls="--", alpha=0.5, label="5°C threshold")
        ax.legend(fontsize=9, framealpha=0.85)
        ax.grid(alpha=0.25, linestyle="--")
        ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout()
    p = os.path.join(OUT, "level8_window_mae.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(p)}")


# ══════════════════════════════════════════════════════════════
# Fig 5 — Runtime vs MAE trade-off (efficiency frontier)
# ══════════════════════════════════════════════════════════════
def fig_runtime_tradeoff():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Efficiency: Runtime vs Accuracy — NAS-MCO-PINN L8\n"
        "A356 Al quenching | Each point = one (arch, skip) configuration",
        fontsize=13, fontweight="bold"
    )

    # Left: all (arch, skip) points
    ax = axes[0]
    for arch in MCO_ARCHS:
        col = ARCH_COLOR[arch]
        rts  = [l8_rt(arch, s)  for s in SKIPS]
        maes = [l8_mae(arch, s) for s in SKIPS]
        ms   = "D" if arch == "mco_shen" else "o"
        ax.plot(rts, maes, f"{ms}-", color=col, lw=1.5, ms=8,
                label=ARCH_LABEL[arch], alpha=0.9)
        for s, rt, mae in zip(SKIPS, rts, maes):
            ax.annotate(f"skip={s}", (rt, mae),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=7.5, color=col)

    ax.axhline(5,  color="green",  lw=1.2, ls="--", alpha=0.7, label="5°C  target")
    ax.axhline(10, color="orange", lw=1.0, ls=":",  alpha=0.6, label="10°C threshold")
    ax.set_xlabel("Runtime [s]", fontsize=11)
    ax.set_ylabel("MAE [°C]", fontsize=11)
    ax.set_title("Full range (all skips)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

    # Right: zoomed — highlight the "sweet spot" (skip=2)
    ax2 = axes[1]
    sweet_spot_rt, sweet_spot_mae = [], []
    for arch in MCO_ARCHS:
        col = ARCH_COLOR[arch]
        rts  = [l8_rt(arch, s)  for s in SKIPS]
        maes = [l8_mae(arch, s) for s in SKIPS]
        ms   = "D" if arch == "mco_shen" else "o"
        ax2.scatter(rts, maes, c=col, s=[180 if s==2 else 70 for s in SKIPS],
                    marker=ms, zorder=4, edgecolors="white", linewidth=0.8,
                    label=ARCH_LABEL[arch], alpha=0.9)
        # skip=2 points
        rt2  = l8_rt(arch, 2)
        mae2 = l8_mae(arch, 2)
        sweet_spot_rt.append(rt2)
        sweet_spot_mae.append(mae2)
        ax2.annotate(f"{ARCH_LABEL[arch]}\nMAE={mae2:.1f}°C",
                     (rt2, mae2), textcoords="offset points",
                     xytext=(5, 5), fontsize=7.5, color=col, fontweight="bold")

    # Shade sweet-spot region
    ax2.axvspan(60, 120, alpha=0.05, color="green", label="skip=2 zone")
    ax2.axhline(5, color="green", lw=1.2, ls="--", alpha=0.7, label="5°C target")
    ax2.set_xlim(0, 260)
    ax2.set_ylim(0, 12)
    ax2.set_xlabel("Runtime [s]", fontsize=11)
    ax2.set_ylabel("MAE [°C]", fontsize=11)
    ax2.set_title("Zoomed: skip=2 sweet spot (large markers)\n"
                  "11/21 FEM steps → ~2× speedup, <3°C error",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8.5, framealpha=0.85)
    ax2.grid(alpha=0.25, linestyle="--")
    ax2.spines[["top","right"]].set_visible(False)

    fig.tight_layout()
    p = os.path.join(OUT, "level8_runtime_tradeoff.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(p)}")


# ══════════════════════════════════════════════════════════════
# Fig 6 — Efficiency: FEM steps vs MAE (L2 → L8 arrows)
# ══════════════════════════════════════════════════════════════
def fig_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Efficiency Analysis: FEM Step Reduction vs Accuracy\n"
        "L2 (faded) → L8 MCO (solid) | Each arrow = MCO gain",
        fontsize=12, fontweight="bold"
    )

    fem_counts = {1: 21, 2: 11, 4: 6, 6: 4}

    ax = axes[0]
    for arch in NAS_ARCHS:
        col = ARCH_COLOR[arch]
        for skip in SKIPS:
            fc  = fem_counts[skip]
            ax.scatter(fc, l2_mae(arch, skip), color=col, s=60, alpha=0.30, marker="s", zorder=3)
            ax.scatter(fc, l8_mae(arch, skip), color=col, s=80, alpha=0.95, marker="o", zorder=4)
            ax.annotate("", xy=(fc, l8_mae(arch, skip)), xytext=(fc, l2_mae(arch, skip)),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=0.6))
        ax.plot([], [], "o-", color=col, lw=1.5, label=ARCH_LABEL[arch])

    # Shen-Arch MCO only (no L2)
    col = ARCH_COLOR["mco_shen"]
    sh_fcs   = [fem_counts[s] for s in SKIPS]
    sh_maes  = [l8_mae("mco_shen", s) for s in SKIPS]
    ax.scatter(sh_fcs, sh_maes, color=col, s=90, marker="D", zorder=4, alpha=0.9,
               label="Shen-Arch MCO")

    ax.axhline(10, color="gray", lw=1, ls="--", alpha=0.5, label="10°C threshold")
    ax.set_xlabel("FEM Steps Used", fontsize=10)
    ax.set_ylabel("MAE [°C]", fontsize=10)
    ax.set_title("□=L2  ○=L8 MCO  ◆=Shen-Arch\n(arrow = improvement)", fontsize=10, fontweight="bold")
    ax.set_xticks([4, 6, 11, 21])
    ax.set_xticklabels(["4\n(skip=6)", "6\n(skip=4)", "11\n(skip=2)", "21\n(skip=1)"])
    ax.set_ylim(-5, 470)
    ax.legend(fontsize=8.5, framealpha=0.85)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

    # Right: zoomed MCO only (0–12°C)
    ax2 = axes[1]
    for arch in MCO_ARCHS:
        col     = ARCH_COLOR[arch]
        l8_vals = [l8_mae(arch, s) for s in SKIPS]
        fcs     = [fem_counts[s] for s in SKIPS]
        ms      = "D--" if arch == "mco_shen" else "o-"
        ax2.plot(fcs, l8_vals, ms, color=col, lw=2.2, ms=8, label=ARCH_LABEL[arch])
        for fc, v in zip(fcs, l8_vals):
            ax2.annotate(f"{v:.1f}°C", (fc, v),
                         textcoords="offset points", xytext=(5, 3),
                         fontsize=8, color=col)

    ax2.axhline(5, color="#E65100", lw=1.2, ls=":", alpha=0.7, label="5°C target")
    ax2.set_xlabel("FEM Steps Used", fontsize=10)
    ax2.set_ylabel("MAE [°C]", fontsize=10)
    ax2.set_title("L8 MCO zoomed (0–12°C)\nAll 4 architectures", fontsize=10, fontweight="bold")
    ax2.set_xticks([4, 6, 11, 21])
    ax2.set_xticklabels(["4\n(skip=6)", "6\n(skip=4)", "11\n(skip=2)", "21\n(skip=1)"])
    ax2.set_ylim(0, 12)
    ax2.legend(fontsize=8.5, framealpha=0.85)
    ax2.grid(alpha=0.25, linestyle="--")
    ax2.spines[["top","right"]].set_visible(False)

    fig.tight_layout()
    p = os.path.join(OUT, "level8_efficiency.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(p)}")


# ══════════════════════════════════════════════════════════════
# Fig 7 — Summary table (all 4 archs, L2 vs L8)
# ══════════════════════════════════════════════════════════════
def fig_summary_table():
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axis("off")
    fig.suptitle(
        "Summary Table — NAS-PINN L2 vs NAS-MCO-PINN L8\n"
        "A356 Al Quenching  |  T₀=540°C → T_w=20°C  |  t=0–30 s",
        fontsize=13, fontweight="bold", y=0.98
    )

    cols = ["Method", "Architecture", "skip=1\n(21/21 FEM)",
            "skip=2\n(11/21 FEM)", "skip=4\n(6/21 FEM)",
            "skip=6\n(4/21 FEM)", "Best Skip", "Runtime (s)"]

    rows = []
    for arch in NAS_ARCHS:
        lbl = ARCH_LABEL[arch]
        l2_row = ["NAS-PINN  L2", lbl]
        for s in SKIPS:
            l2_row.append(f"{l2_mae(arch,s):.1f}°C")
        best_s = min(SKIPS, key=lambda s: l2_mae(arch,s))
        l2_row.append(f"skip={best_s}")
        l2_row.append("—")
        rows.append(l2_row)

        l8_row = ["NAS-MCO  L8", lbl]
        for s in SKIPS:
            l8_row.append(f"{l8_mae(arch,s):.1f}°C")
        best_s8 = min(SKIPS, key=lambda s: l8_mae(arch,s))
        l8_row.append(f"skip={best_s8}")
        l8_row.append(f"{l8_rt(arch,best_s8):.0f}s")
        rows.append(l8_row)
        rows.append([""] * len(cols))

    # Shen-Arch row
    sh_row = ["NAS-MCO  L8", "Shen-Arch"]
    for s in SKIPS:
        sh_row.append(f"{l8_mae('mco_shen',s):.1f}°C")
    best_sh = min(SKIPS, key=lambda s: l8_mae("mco_shen", s))
    sh_row.append(f"skip={best_sh}")
    sh_row.append(f"{l8_rt('mco_shen',best_sh):.0f}s")
    rows.append(sh_row)
    rows.append(["Shen 2023", "MCO-PINN", "RMSE=0.0041*", "—", "—", "—", "—", "—"])

    table = ax.table(cellText=rows, colLabels=cols,
                     cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    for j in range(len(cols)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(rows):
        is_l8   = "NAS-MCO" in str(row[0])
        is_sep  = row[0] == ""
        is_shen = "Shen 2023" in str(row[0])
        is_sh_arch = row[1] == "Shen-Arch" if len(row) > 1 else False
        for j in range(len(cols)):
            cell = table[i+1, j]
            if is_sep:
                cell.set_facecolor("#F5F5F5"); cell.set_text_props(color="#DDD")
            elif is_shen:
                cell.set_facecolor("#FFF3E0")
            elif is_sh_arch:
                cell.set_facecolor("#F3E5F5")
            elif is_l8:
                cell.set_facecolor("#E8F5E9")
            else:
                cell.set_facecolor("#FFFFFF")

            txt = str(row[j]) if j < len(row) else ""
            if "°C" in txt and not is_sep:
                try:
                    v = float(txt.replace("°C",""))
                    if is_l8 or is_sh_arch:
                        c = "#1B5E20" if v<5 else "#2E7D32" if v<10 else "#388E3C"
                        table[i+1, j].set_text_props(color=c, fontweight="bold")
                    else:
                        c = "#B71C1C" if v>100 else "#E53935" if v>30 else "#757575"
                        table[i+1, j].set_text_props(color=c)
                except:
                    pass

    fig.tight_layout()
    p = os.path.join(OUT, "level8_summary_table.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(p)}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("\nGenerating Level 8 comparison figures...\n")
    fig_bar_comparison()
    fig_heatmap()
    fig_improvement_heatmap()
    fig_window_mae()
    fig_runtime_tradeoff()
    fig_efficiency()
    fig_summary_table()

    # Remove outdated figure
    old = os.path.join(OUT, "level8_cooling_curves.png")
    if os.path.exists(old):
        os.remove(old)
        print(f"  Removed: level8_cooling_curves.png")

    print("\nDone →", OUT)
    for f in sorted(os.listdir(OUT)):
        if f.startswith("level8_") and f.endswith(".png"):
            kb = os.path.getsize(os.path.join(OUT, f)) // 1024
            print(f"  {f}  ({kb} KB)")
