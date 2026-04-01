"""
thermal_pinn/plots/plot_warmstart.py
=====================================
Comparison of training strategies: cold-800ep → v2 (Fourier+SA) → warm-500ep.

Figures produced
----------------
A. fig_ws_mae_comparison_2d.png / _3d.png
   MAE vs k line plot — same style as fig_th4_k_progression.
   One panel per domain, 3 archs × 3 variants (cold / v2 / warm).
   Cold=solid, v2=dashed, warm=dotted. Colour=arch.

B. fig_ws_summary_table_2d.png / _3d.png
   Numeric table: domain/arch | cold MAE | v2 MAE | warm MAE |
                  cold time | warm time | speedup | recommendation.

C. fig_ws_recommendation_2d.png / _3d.png
   Colour-coded matrix (domain × arch): green/yellow/red + 1-line verdict.

Output: results/06_warmstart_stats/
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import argparse

warnings.filterwarnings("ignore")

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
from thermal_pinn.plots.plot_results import CKPT_DIR, RESULT_DIR

OUT_DIR = RESULT_DIR / "06_warmstart_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DOMAINS_2D   = ["rectangle", "circle", "lshape"]
DOMAINS_3D   = ["rectangular", "cylinder", "stacked", "lshape"]
DOM_LABELS   = {
    "rectangle":  "Rectangle 2D",
    "circle":     "Circle 2D",
    "lshape":     "L-Shape 2D",
    "rectangular":"Rectangular 3D",
    "cylinder":   "Cylinder 3D",
    "stacked":    "Stacked 3D",
    "lshape":     "L-Shape",
}
ARCHS        = ["bayesian", "nsga2", "nsga3"]
ARCH_LABELS  = {"bayesian": "Bayesian", "nsga2": "NSGA-II", "nsga3": "NSGA-III"}
ARCH_COLORS  = {"bayesian": "#1565C0",  "nsga2": "#2E7D32",  "nsga3": "#E65100"}
K_VALS       = [1, 2, 3, 4, 5]

_WHITE   = "#ffffff"
_FIG_DPI = 150
_HEADER  = "#1C3557"   # dark navy — same as PPTX palette


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load(domain: str, arch: str, k: int, dim: int, suffix: str = "") -> dict | None:
    p = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}{suffix}_metrics.json"
    if not p.exists():
        return None
    return json.load(open(p))


def _mae(m: dict | None) -> float:
    return m["mean_mae"] if m else float("nan")


def _time(m: dict | None) -> float:
    return m.get("total_s", float("nan")) if m else float("nan")


def _best_k(domain: str, arch: str, dim: int) -> int:
    """Best k = lowest cold-start MAE."""
    best, best_k = float("inf"), 1
    for k in K_VALS:
        m = _load(domain, arch, k, dim)
        if m and m["mean_mae"] < best:
            best, best_k = m["mean_mae"], k
    return best_k


# ── Figure A — MAE vs k comparison ────────────────────────────────────────────

def fig_ws_mae_comparison(dim: int) -> None:
    """
    Line plot: MAE vs k for cold / v2 / warm — one panel per domain.
    Style mirrors fig_th4_k_progression from plot_thesis.py.
    """
    domains = DOMAINS_2D if dim == 2 else DOMAINS_3D
    tag     = "2d" if dim == 2 else "3d"
    ncols   = len(domains)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5), sharey=False)
    fig.patch.set_facecolor(_WHITE)

    variants = [
        ("",          "solid",  "o", "Cold (800 ep)"),
        ("_v2",       "dashed", "s", "v2 (Fourier+SA)"),
        ("_ws_500ep", "dotted", "^", "Warm (500 ep)"),
    ]

    for ci, domain in enumerate(domains):
        ax = axes[ci] if ncols > 1 else axes
        ax.set_facecolor(_WHITE)
        any_data = False

        for arch in ARCHS:
            color = ARCH_COLORS[arch]
            for suffix, ls, mk, _ in variants:
                k_list, mae_list = [], []
                for k in K_VALS:
                    m = _load(domain, arch, k, dim, suffix)
                    if m:
                        k_list.append(k)
                        mae_list.append(m["mean_mae"])
                if k_list:
                    any_data = True
                    ax.plot(k_list, mae_list, linestyle=ls, marker=mk,
                            color=color, linewidth=1.5, markersize=5, alpha=0.85)

        dom_label = domain.capitalize().replace("_", " ")
        ax.set_title(f"{dom_label} ({tag.upper()})", fontsize=9, fontweight="bold")
        ax.set_xlabel("k (FEM steps skipped)", fontsize=8)
        if ci == 0:
            ax.set_ylabel("Mean MAE [°C]", fontsize=8)
        ax.set_xticks(K_VALS)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        if not any_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

    # Legend: arch (colour) + variant (line style)
    legend_handles = []
    for arch in ARCHS:
        legend_handles.append(
            Line2D([0], [0], color=ARCH_COLORS[arch], linewidth=2,
                   label=ARCH_LABELS[arch]))
    legend_handles.append(Line2D([], [], color="gray", lw=0))  # spacer
    for suffix, ls, mk, lbl in variants:
        legend_handles.append(
            Line2D([0], [0], color="#555555", linestyle=ls, marker=mk,
                   markersize=5, linewidth=1.5, label=lbl))

    fig.legend(handles=legend_handles, fontsize=8, framealpha=0.9,
               loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.04))

    dim_str = "2D" if dim == 2 else "3D"
    fig.suptitle(
        f"MAE vs k — Cold-Start vs v2 (Fourier+SA) vs Warm-Start (500 ep)"
        f" — {dim_str} Domains",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    out = OUT_DIR / f"fig_ws_mae_comparison_{tag}.png"
    fig.savefig(out, dpi=_FIG_DPI, bbox_inches="tight", facecolor=_WHITE)
    plt.close(fig)
    print(f"    → {out.name}")


# ── Figure B — Summary table ──────────────────────────────────────────────────

def _recommend(cold_mae, v2_mae, warm_mae, cold_t, warm_t) -> tuple[str, str]:
    """
    Returns (verdict_text, color) based on the data.
    Priority: lowest MAE at lowest cost.
    """
    speedup = cold_t / warm_t if warm_t > 0 and np.isfinite(warm_t) else 1.0
    dm_warm = warm_mae - cold_mae   # + = warm worse
    dm_v2   = v2_mae  - cold_mae   # - = v2 better

    if np.isfinite(dm_v2) and dm_v2 < -1.0:
        # v2 meaningfully better
        if np.isfinite(dm_warm) and dm_warm < 1.0 and speedup > 1.3:
            return "Warm + v2", "#C8E6C9"   # green
        return "v2 önerilir", "#C8E6C9"
    if np.isfinite(dm_warm) and dm_warm < 1.0 and speedup > 1.3:
        return "Warm önerilir", "#C8E6C9"   # green
    if np.isfinite(dm_warm) and dm_warm < 2.0 and speedup > 1.0:
        return "Warm kabul.", "#FFF9C4"      # yellow
    return "Cold yeterli", "#FFCDD2"         # red


def fig_ws_summary_table(dim: int) -> None:
    domains  = DOMAINS_2D if dim == 2 else DOMAINS_3D
    tag      = "2d" if dim == 2 else "3d"

    col_headers = [
        "Domain", "Arch", "Best k",
        "Cold MAE\n(°C)", "v2 MAE\n(°C)", "Warm MAE\n(°C)",
        "Cold time\n(s)", "Warm time\n(s)", "Speedup",
        "Öneri",
    ]
    rows = []
    row_colors = []

    for domain in domains:
        for arch in ARCHS:
            bk       = _best_k(domain, arch, dim)
            m_cold   = _load(domain, arch, bk, dim)
            m_v2     = _load(domain, arch, bk, dim, "_v2")
            m_warm   = _load(domain, arch, bk, dim, "_ws_500ep")

            cold_mae = _mae(m_cold)
            v2_mae   = _mae(m_v2)
            warm_mae = _mae(m_warm)
            cold_t   = _time(m_cold)
            warm_t   = _time(m_warm)
            speedup  = cold_t / warm_t if warm_t > 0 and np.isfinite(warm_t) else float("nan")

            verdict, cell_col = _recommend(cold_mae, v2_mae, warm_mae, cold_t, warm_t)

            def fmt(v): return f"{v:.2f}" if np.isfinite(v) else "—"
            rows.append([
                domain.capitalize(), ARCH_LABELS[arch], str(bk),
                fmt(cold_mae), fmt(v2_mae), fmt(warm_mae),
                fmt(cold_t),   fmt(warm_t), fmt(speedup) + "×",
                verdict,
            ])
            row_colors.append(cell_col)

    n_rows = len(rows)
    fig_h  = max(3.5, 0.45 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    fig.patch.set_facecolor(_WHITE)
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(list(range(len(col_headers))))

    # Header style
    for j in range(len(col_headers)):
        cell = tbl[0, j]
        cell.set_facecolor(_HEADER)
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.12)

    # Row colours + alternating stripes
    for i, (rw, rc) in enumerate(zip(rows, row_colors)):
        base = "#F8F8F8" if i % 2 == 0 else _WHITE
        for j in range(len(col_headers)):
            cell = tbl[i + 1, j]
            cell.set_height(0.09)
            # Colour only the "Öneri" column
            if j == len(col_headers) - 1:
                cell.set_facecolor(rc)
            else:
                cell.set_facecolor(base)

    dim_str = "2D" if dim == 2 else "3D"
    fig.suptitle(
        f"Training Strategy Summary — {dim_str} Domains\n"
        f"Best k per (domain, arch) | Cold 800 ep | v2 Fourier+SA | Warm 500 ep",
        fontsize=11, fontweight="bold", y=0.98,
    )

    out = OUT_DIR / f"fig_ws_summary_table_{tag}.png"
    fig.savefig(out, dpi=_FIG_DPI, bbox_inches="tight", facecolor=_WHITE)
    plt.close(fig)
    print(f"    → {out.name}")


# ── Figure C — Recommendation matrix ─────────────────────────────────────────

def fig_ws_recommendation(dim: int) -> None:
    """
    Colour-coded matrix: rows=domains, cols=archs.
    Each cell: background colour + 3-line text (cold/v2/warm MAE + verdict).
    """
    domains = DOMAINS_2D if dim == 2 else DOMAINS_3D
    tag     = "2d" if dim == 2 else "3d"
    n_rows  = len(domains)
    n_cols  = len(ARCHS)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.2 * n_cols, 2.8 * n_rows))
    fig.patch.set_facecolor(_WHITE)

    col_colors = {"bayesian": "#E3F2FD", "nsga2": "#E8F5E9", "nsga3": "#FFF3E0"}

    for ri, domain in enumerate(domains):
        for ci, arch in enumerate(ARCHS):
            ax = axes[ri, ci] if n_rows > 1 else axes[ci]
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis("off")

            bk     = _best_k(domain, arch, dim)
            m_cold = _load(domain, arch, bk, dim)
            m_v2   = _load(domain, arch, bk, dim, "_v2")
            m_warm = _load(domain, arch, bk, dim, "_ws_500ep")

            cold_mae = _mae(m_cold); v2_mae = _mae(m_v2); warm_mae = _mae(m_warm)
            cold_t   = _time(m_cold); warm_t = _time(m_warm)
            speedup  = cold_t / warm_t if warm_t > 0 and np.isfinite(warm_t) else float("nan")
            verdict, bg = _recommend(cold_mae, v2_mae, warm_mae, cold_t, warm_t)

            ax.add_patch(mpatches.FancyBboxPatch(
                (0.04, 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.02",
                facecolor=bg, edgecolor="#BBBBBB", linewidth=0.8,
            ))

            def fmt(v): return f"{v:.1f}°C" if np.isfinite(v) else "—"
            sp_str = f"{speedup:.1f}×" if np.isfinite(speedup) else "—"

            lines = [
                f"Cold: {fmt(cold_mae)}",
                f"v2:   {fmt(v2_mae)}",
                f"Warm: {fmt(warm_mae)}  ({sp_str})",
                "",
                verdict,
            ]
            y_pos = [0.82, 0.66, 0.50, 0.36, 0.22]
            for txt, yp in zip(lines, y_pos):
                weight = "bold" if txt == verdict else "normal"
                size   = 9 if txt == verdict else 8
                ax.text(0.5, yp, txt, ha="center", va="center",
                        fontsize=size, fontweight=weight, transform=ax.transAxes)

            # Row and column labels
            if ri == 0:
                ax.set_title(ARCH_LABELS[arch], fontsize=10, fontweight="bold",
                             color=ARCH_COLORS[arch], pad=6)
            if ci == 0:
                dom_lbl = domain.capitalize().replace("_3d", "").replace("_", " ")
                ax.text(-0.12, 0.5, dom_lbl, transform=ax.transAxes,
                        fontsize=9, fontweight="bold", va="center",
                        ha="right", rotation=0)

    # Legend for colours
    legend_patches = [
        mpatches.Patch(color="#C8E6C9", label="Warm/v2 önerilir"),
        mpatches.Patch(color="#FFF9C4", label="Kabul edilebilir"),
        mpatches.Patch(color="#FFCDD2", label="Cold yeterli"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    dim_str = "2D" if dim == 2 else "3D"
    fig.suptitle(
        f"Training Strategy Recommendation — {dim_str} Domains\n"
        f"At best k per combination  |  Cold 800 ep / v2 Fourier+SA / Warm 500 ep",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    out = OUT_DIR / f"fig_ws_recommendation_{tag}.png"
    fig.savefig(out, dpi=_FIG_DPI, bbox_inches="tight", facecolor=_WHITE)
    plt.close(fig)
    print(f"    → {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=0, choices=[0, 2, 3],
                        help="0=both, 2=2D only, 3=3D only")
    args = parser.parse_args()

    dims = [2, 3] if args.dim == 0 else [args.dim]

    # Delete old figures first
    for old in OUT_DIR.glob("fig_ws_*.png"):
        old.unlink()

    for dim in dims:
        tag = "2D" if dim == 2 else "3D"
        print(f"\n{'='*55}")
        print(f"  Warmstart Comparison — {tag}")
        print(f"{'='*55}")
        fig_ws_mae_comparison(dim)
        fig_ws_summary_table(dim)
        fig_ws_recommendation(dim)

    print(f"\nDone → {OUT_DIR}/")
