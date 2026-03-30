"""
thermal_pinn/plots/plot_comparison.py
======================================
Clean comparison figures — style follows results_v2/summary/.

Output files (results/summary/):

  k-skip line plots (epoch-separated, 4 files):
    fig_kskip_800ep_2d.png        fig_kskip_800ep_3d.png
    fig_kskip_1500ep_2d.png       fig_kskip_1500ep_3d.png

  Per-domain jump bar charts — all k=1..5, before vs after (7 files):
    fig_jump_{domain}_2d.png  ×3   (rectangle, circle, lshape)
    fig_jump_{domain}_3d.png  ×4   (rectangular, cylinder, stacked, lshape)

  Epoch tables — clean ax.table() style (4 files):
    fig_table_800ep_2d.png    fig_table_800ep_3d.png
    fig_table_1500ep_2d.png   fig_table_1500ep_3d.png

  Full summary tables — both epochs, all k, runtime, FEM reduction (2 files):
    fig_full_summary_2d.png   fig_full_summary_3d.png

Usage:
    python -m thermal_pinn.plots.plot_comparison
"""
from __future__ import annotations

import json
import subprocess
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
from thermal_pinn.plots.plot_results import CKPT_DIR, RESULT_DIR

OUT_DIR = RESULT_DIR / "summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DOMAINS_2D = ["rectangle", "circle", "lshape"]
DOMAINS_3D = ["rectangular", "cylinder", "stacked", "lshape"]
ARCHS      = ["bayesian", "nsga2", "nsga3"]

ARCH_SHORT = {"bayesian": "Bayesian (TPE)", "nsga2": "NSGA-II", "nsga3": "NSGA-III"}
ARCH_COLOR = {"bayesian": "#1f77b4", "nsga2": "#2ca02c", "nsga3": "#ff7f0e"}
DOM_LABEL  = {
    "rectangle":  "Rectangle",  "circle":      "Circle",
    "lshape":     "Lshape",     "rectangular": "Rectangular",
    "cylinder":   "Cylinder",   "stacked":     "Stacked",
}

N_FEM_FULL = 20   # total FEM steps for 30 s quench at Δt=1.5 s
SS_SKIP    = 3


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load_new(domain, arch, k, dim):
    p = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}_metrics.json"
    return json.load(open(p)) if p.exists() else None


def _load_old(domain, arch, k, dim):
    """Pre-retrain metrics from git HEAD; fallback to current (Bayesian unchanged)."""
    p = f"thermal_pinn/checkpoints/{domain}_{arch}_k{k}_dim{dim}_metrics.json"
    try:
        raw = subprocess.check_output(["git", "show", f"HEAD:{p}"],
                                      stderr=subprocess.DEVNULL)
        return json.loads(raw)
    except Exception:
        return _load_new(domain, arch, k, dim)


def _ss_l2(m):
    w = m["windows"][SS_SKIP:] if len(m["windows"]) > SS_SKIP else m["windows"]
    return float(np.mean([x["l2_rel"] for x in w]))


def _ss_mae(m):
    w = m["windows"][SS_SKIP:] if len(m["windows"]) > SS_SKIP else m["windows"]
    return float(np.mean([x["mae_C"] for x in w]))


def _runtime(m):
    return float(np.sum([x["runtime_s"] for x in m["windows"]]))


def _runtime_per_win(m):
    return float(np.mean([x["runtime_s"] for x in m["windows"]]))


def _fem_reduction(m):
    """FEM call reduction = N_FEM_full / n_windows (how many fewer FEM steps needed)."""
    n_windows = len(m["windows"])
    return N_FEM_FULL / max(n_windows, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  k-SKIP LINE PLOTS  — one file per epoch × dimension
# ══════════════════════════════════════════════════════════════════════════════

def _kskip_plot(domains, dim, loader, epoch_label, out_name):
    """
    Line plot: L2 vs k, one subplot per domain.
    Style follows fig_th4_k_progression.
    """
    n = len(domains)
    if n <= 3:
        nrows, ncols = 1, n
    else:
        nrows, ncols = 2, (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4.5, nrows * 3.6))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    dim_label = "2D" if dim == 2 else "3D"
    fig.suptitle(
        f"L2 Error vs k-Skip Window Size — {dim_label} Domains  ({epoch_label})",
        fontsize=12)

    for pi, domain in enumerate(domains):
        ax = axes_flat[pi]
        for arch in ARCHS:
            clr = ARCH_COLOR[arch]
            pts = [(k, _ss_l2(m))
                   for k in range(1, 6)
                   if (m := loader(domain, arch, k, dim)) is not None]
            if not pts:
                continue
            ks, l2s = zip(*pts)
            best_k = ks[list(l2s).index(min(l2s))]
            ax.plot(ks, l2s, color=clr, lw=1.8, marker="o", ms=5,
                    label=ARCH_SHORT[arch])
            ax.plot(best_k, min(l2s), marker="*", ms=13, color=clr, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label="Best k" if pi == 0 and arch == ARCHS[0] else "_")

        ax.axhline(0.05, color="gray", lw=1.0, ls="--", alpha=0.7,
                   label="5% threshold" if pi == 0 else "_")
        ax.set_title(DOM_LABEL.get(domain, domain), fontsize=10, fontweight="bold")
        ax.set_xlabel("k (skip window size)", fontsize=9)
        if pi % ncols == 0:
            ax.set_ylabel("Mean Relative L2", fontsize=9)
        ax.set_xticks(range(1, 6))
        ax.set_xlim(0.7, 5.4)
        ax.grid(True, alpha=0.3, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if pi == 0:
            ax.legend(fontsize=8.5, framealpha=0.9)

    for pi in range(len(domains), len(axes_flat)):
        axes_flat[pi].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def fig_kskip_by_epoch():
    for domains, dim, tag in [(DOMAINS_2D, 2, "2d"), (DOMAINS_3D, 3, "3d")]:
        dl = "2D" if dim == 2 else "3D"
        _kskip_plot(domains, dim, _load_old,
                    "n_epochs = 800  (original training)",
                    f"fig_kskip_800ep_{tag}.png")
        _kskip_plot(domains, dim, _load_new,
                    "n_epochs = 1 500  (NSGA-II/III retrain)",
                    f"fig_kskip_1500ep_{tag}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PER-DOMAIN JUMP BAR CHARTS  — one file per domain
# ══════════════════════════════════════════════════════════════════════════════

def _jump_domain(domain, dim):
    """
    Grouped bar chart for one domain.
    x = k=1..5 groups.
    Bars per group: Bayesian (ref) | NSGA-II 800ep | NSGA-II 1500ep |
                    NSGA-III 800ep | NSGA-III 1500ep.
    """
    ks = list(range(1, 6))
    BW = 0.155
    offsets = np.array([-2.5, -1.3, -0.3, 0.7, 1.7]) * BW

    # (arch, use_old_loader, label)
    specs = [
        ("bayesian", False, "Bayesian (800 ep — ref)"),
        ("nsga2",    True,  "NSGA-II  800 ep"),
        ("nsga2",    False, "NSGA-II  1 500 ep"),
        ("nsga3",    True,  "NSGA-III 800 ep"),
        ("nsga3",    False, "NSGA-III 1 500 ep"),
    ]
    hatches = [None, "///", None, "///", None]
    alphas  = [0.85, 1.0,   0.85, 1.0,   0.85]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    dim_label = "2D" if dim == 2 else "3D"
    fig.suptitle(
        f"{DOM_LABEL.get(domain, domain)} ({dim_label}) — L2 Error per k\n"
        "Hatched = Before retrain (800 ep)   Solid = After retrain (1 500 ep)",
        fontsize=10.5, y=0.97)

    all_vals = []
    bar_data = []  # store for smart label placement
    for bi, ((arch, use_old, lbl), hatch, alpha) in enumerate(
            zip(specs, hatches, alphas)):
        clr    = ARCH_COLOR[arch]
        loader = _load_old if use_old else _load_new

        xs, ys = [], []
        for k in ks:
            m = loader(domain, arch, k, dim)
            if m:
                v = _ss_l2(m) * 100
                x_pos = k + offsets[bi]
                xs.append(x_pos)
                ys.append(v)
                all_vals.append(v)
                bar_data.append((x_pos, v, clr, bi))

        if xs:
            fc = "white" if hatch else clr
            ax.bar(xs, ys, BW * 0.94,
                   color=fc, alpha=alpha,
                   edgecolor=clr, hatch=hatch, lw=0.95, zorder=3)

    # Smart label placement: avoid overlaps
    if all_vals:
        max_val = max(all_vals)
        for x, y, clr, bi in bar_data:
            # Offset increases for taller bars
            if y >= max_val * 0.75:
                offset = 0.18 + (0.20 if y == max_val else 0.08)
            elif y >= max_val * 0.5:
                offset = 0.14
            else:
                offset = 0.10
            
            ax.text(x, y + offset, f"{y:.1f}",
                    ha="center", va="bottom",
                    fontsize=6.2, color=clr,
                    rotation=0, clip_on=False,
                    fontweight="normal")

    ax.axhline(5.0, color="gray", lw=1.1, ls="--", alpha=0.65)
    ax.text(5.60, 5.25, "5%", fontsize=8.2, color="gray", fontweight="bold")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"k = {k}" for k in ks], fontsize=10)
    ax.set_ylabel("SS L2 Error [%]", fontsize=9.5)
    ax.set_xlabel("Window skip factor k", fontsize=9.5)
    ax.set_xlim(0.5, 5.5)
    if all_vals:
        ax.set_ylim(0, max(all_vals) * 1.82)
    ax.grid(axis="y", alpha=0.32, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # legend with better styling
    handles = []
    for (arch, use_old, lbl), hatch, alpha in zip(specs, hatches, alphas):
        clr = ARCH_COLOR[arch]
        if hatch:
            h = plt.Rectangle((0,0),1,1, fc="white", ec=clr, hatch=hatch, lw=1)
        else:
            h = plt.Rectangle((0,0),1,1, fc=clr, alpha=alpha, ec=clr, lw=1)
        handles.append((h, lbl))
    handles.append((plt.Line2D([0],[0], color="gray", lw=1.1, ls="--"), "5% target"))
    ax.legend([h for h,_ in handles], [l for _,l in handles],
              fontsize=7.9, ncol=2, framealpha=0.93, loc="upper right",
              edgecolor="#bbb", fancybox=True, shadow=False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    dim_tag = "2d" if dim == 2 else "3d"
    out = OUT_DIR / f"fig_jump_{domain}_{dim_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def fig_jump_per_domain():
    for domain in DOMAINS_2D:
        _jump_domain(domain, 2)
    for domain in DOMAINS_3D:
        _jump_domain(domain, 3)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  EPOCH TABLES  — ax.table() style, one file per epoch × dimension
# ══════════════════════════════════════════════════════════════════════════════

def _epoch_table(domains, dim, loader, epoch_label, out_name):
    """
    Clean ax.table() — SS L2 for each domain / arch / k.
    Green cell = best k per row.
    """
    col_labels = ["Domain", "Architecture",
                  "k = 1", "k = 2", "k = 3", "k = 4", "k = 5",
                  "Best k", "Total [s]", "Per-win [s]", "FEM calls↓"]

    rows_data  = []
    cell_colors = []

    for di, domain in enumerate(domains):
        row_bg = "#EEF5FF" if di % 2 == 0 else "#FFFFFF"
        for ai, arch in enumerate(ARCHS):
            kd = {}
            for k in range(1, 6):
                m = loader(domain, arch, k, dim)
                if m:
                    kd[k] = (_ss_l2(m), _runtime(m), _fem_reduction(m), _runtime_per_win(m))
            best_k = min(kd, key=lambda k: kd[k][0]) if kd else None

            row = [
                DOM_LABEL.get(domain, domain) if ai == 0 else "",
                ARCH_SHORT[arch],
            ]
            colors = [row_bg, row_bg]

            for k in range(1, 6):
                if k in kd:
                    l2  = kd[k][0]
                    star = "★" if k == best_k else ""
                    row.append(f"{l2*100:.2f}%{star}")
                    if k == best_k:
                        c = "#C6EFCE" if l2 < 0.05 else ("#FFEB9C" if l2 < 0.10 else "#FFC7CE")
                    else:
                        c = row_bg
                    colors.append(c)
                else:
                    row.append("—")
                    colors.append(row_bg)

            if best_k and best_k in kd:
                row += [f"k = {best_k}",
                        f"{kd[best_k][1]:.0f} s",
                        f"{kd[best_k][3]:.1f} s",
                        f"{kd[best_k][2]:.1f}×"]
                colors += [row_bg, row_bg, row_bg, "#DBEAFE"]
            else:
                row    += ["—", "—", "—", "—"]
                colors += [row_bg] * 4

            rows_data.append(row)
            cell_colors.append(colors)

    n_rows = len(rows_data)
    fig_h  = n_rows * 0.44 + 1.5
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    tbl = ax.table(cellText=rows_data, colLabels=col_labels,
                   cellColours=cell_colors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)

    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#2C5282")
        cell.set_text_props(color="white", fontweight="bold")

    dim_label = "2D" if dim == 2 else "3D"
    ax.set_title(
        f"NAS-PINN k-Skip — SS L2 Error  |  {dim_label} Domains  |  {epoch_label}\n"
        f"(★ = best k per row  ·  Green < 5%  ·  Amber 5–10%  ·  Red ≥ 10%  ·  "
        f"FEM calls↓ = {N_FEM_FULL} steps / n_windows  (Δt=1.5 s, T=30 s))",
        fontsize=10, pad=14)

    plt.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def fig_epoch_tables():
    for domains, dim, tag in [(DOMAINS_2D, 2, "2d"), (DOMAINS_3D, 3, "3d")]:
        _epoch_table(domains, dim, _load_old,
                     "n_epochs = 800  (original training)",
                     f"fig_table_800ep_{tag}.png")
        _epoch_table(domains, dim, _load_new,
                     "n_epochs = 1 500  (NSGA-II/III retrain)",
                     f"fig_table_1500ep_{tag}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FULL SUMMARY TABLES  — both epochs, all k, runtime, FEM reduction
# ══════════════════════════════════════════════════════════════════════════════

def _full_summary(domains, dim, out_name):
    """
    Comprehensive ax.table() — both epochs (800ep / 1500ep) per arch,
    all k values, runtime, speedup.
    Bayesian appears once (unchanged).
    """
    col_labels = ["Domain", "Architecture", "Epoch",
                  "k = 1", "k = 2", "k = 3", "k = 4", "k = 5",
                  "Best k", "Total [s]", "Per-win [s]", "FEM calls↓"]

    rows_data   = []
    cell_colors = []

    for di, domain in enumerate(domains):
        base_bg = "#EEF5FF" if di % 2 == 0 else "#FFFFFF"
        old_bg  = "#F0F4FF" if di % 2 == 0 else "#F8F8FF"

        for arch in ARCHS:
            epoch_list = (
                [("800 ep", _load_old)]
                if arch == "bayesian"
                else [("800 ep", _load_old), ("1 500 ep", _load_new)]
            )
            for ei, (ep_label, loader) in enumerate(epoch_list):
                kd = {}
                for k in range(1, 6):
                    m = loader(domain, arch, k, dim)
                    if m:
                        kd[k] = (_ss_l2(m), _runtime(m), _fem_reduction(m), _runtime_per_win(m))
                best_k = min(kd, key=lambda k: kd[k][0]) if kd else None

                dom_txt  = DOM_LABEL.get(domain, domain) if arch == ARCHS[0] and ei == 0 else ""
                arch_txt = ARCH_SHORT[arch] if ei == 0 else ""
                row_bg   = old_bg if ep_label == "800 ep" else base_bg

                row    = [dom_txt, arch_txt, ep_label]
                colors = [row_bg, row_bg,
                          "#DDEEFF" if ep_label == "800 ep" else "#D5F5E3"]

                for k in range(1, 6):
                    if k in kd:
                        l2   = kd[k][0]
                        star = "★" if k == best_k else ""
                        row.append(f"{l2*100:.2f}%{star}")
                        if k == best_k:
                            c = "#C6EFCE" if l2<0.05 else ("#FFEB9C" if l2<0.10 else "#FFC7CE")
                        else:
                            c = row_bg
                        colors.append(c)
                    else:
                        row.append("—")
                        colors.append(row_bg)

                if best_k and best_k in kd:
                    row    += [f"k = {best_k}",
                               f"{kd[best_k][1]:.0f} s",
                               f"{kd[best_k][3]:.1f} s",
                               f"{kd[best_k][2]:.1f}×"]
                    colors += [row_bg, row_bg, row_bg, "#DBEAFE"]
                else:
                    row    += ["—", "—", "—", "—"]
                    colors += [row_bg] * 4

                rows_data.append(row)
                cell_colors.append(colors)

    n_rows = len(rows_data)
    fig_h  = n_rows * 0.44 + 1.8
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")

    tbl = ax.table(cellText=rows_data, colLabels=col_labels,
                   cellColours=cell_colors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)

    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#2C5282")
        cell.set_text_props(color="white", fontweight="bold")

    dim_label = "2D" if dim == 2 else "3D"
    ax.set_title(
        f"NAS-PINN k-Skip — Full Summary  |  {dim_label} Domains\n"
        "Before (800 ep) & After (1 500 ep NSGA retrain)  ·  All k values  ·  "
        f"Runtime  ·  FEM calls↓ = {N_FEM_FULL} steps / n_windows  (Δt=1.5 s, T=30 s)\n"
        "(Blue rows = 800 ep  ·  Green rows = 1 500 ep  ·  "
        "★ = best k  ·  Bayesian shown once — not retrained)",
        fontsize=9.5, pad=16)

    plt.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def fig_full_summary():
    for domains, dim, tag in [(DOMAINS_2D, 2, "2d"), (DOMAINS_3D, 3, "3d")]:
        _full_summary(domains, dim, f"fig_full_summary_{tag}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("[Comparison plots]")

    print("  [1/4] k-skip line plots (800 ep / 1 500 ep) …")
    fig_kskip_by_epoch()

    print("  [2/4] Per-domain jump bar charts …")
    fig_jump_per_domain()

    print("  [3/4] Epoch tables (800 ep / 1 500 ep) …")
    fig_epoch_tables()

    print("  [4/4] Full summary tables …")
    fig_full_summary()

    print(f"\nDone → {OUT_DIR}/")
