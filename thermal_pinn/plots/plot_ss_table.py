"""
plot_ss_table.py
================
Steady-State MAE table for all k values.
Excludes first SS_SKIP windows (transient phase).

Output:
  results/summary/fig_ss_table_2d.png
  results/summary/fig_ss_table_3d.png

Usage:
  python thermal_pinn/plot_ss_table.py
  python thermal_pinn/plot_ss_table.py --skip 5
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

CKPT_DIR   = ROOT / "checkpoints"
RESULT_DIR = ROOT / "results"

DOMAINS_2D = ["rectangle", "circle", "lshape"]
DOMAINS_3D = ["rectangular", "cylinder", "stacked", "lshape"]
ARCHS      = ["bayesian", "nsga2", "nsga3"]
ARCH_LABELS = {"bayesian": "Bayesian", "nsga2": "NSGA-II", "nsga3": "NSGA-III"}

_WHITE  = "#ffffff"
_HEADER = "#1565C0"
_GREEN  = "#C8E6C9"
_AMBER  = "#FFF9C4"
_RED    = "#FFCDD2"
_BEST   = "#A5D6A7"   # darker green for best k per (domain, arch)
_DPI    = 150


def _load_all_k(domain: str, arch: str, dim: int, skip: int) -> list[dict]:
    rows = []
    for k in range(1, 6):
        mp = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}_metrics.json"
        if not mp.exists():
            continue
        m    = json.load(open(mp))
        wins = m["windows"]

        full_mae = m["mean_mae"]
        full_l2  = float(np.mean([w["l2_rel"] for w in wins]))

        ss_wins = wins[skip:] if len(wins) > skip else wins
        ss_mae  = float(np.mean([w["mae_C"]  for w in ss_wins]))
        ss_l2   = float(np.mean([w["l2_rel"] for w in ss_wins]))

        rows.append({
            "domain":   domain,
            "arch":     arch,
            "dim":      dim,
            "k":        k,
            "full_mae": full_mae,
            "full_l2":  full_l2,
            "ss_mae":   ss_mae,
            "ss_l2":    ss_l2,
            "n_wins":   len(wins),
            "n_ss":     len(ss_wins),
        })
    return rows


def _cell_color(l2: float) -> str:
    if l2 < 0.05:  return _GREEN
    if l2 < 0.10:  return _AMBER
    return _RED


def _make_table(all_rows: list[dict], title: str, out_path: Path,
                skip: int, row_bg: str):
    col_headers = [
        "Domain", "Optimizer", "k",
        "Full MAE", "SS MAE", "Improvement",
        "Full L2", "SS L2", "Status",
    ]

    # Find best k per (domain, arch) based on ss_mae
    best_key: dict[tuple, float] = {}
    for r in all_rows:
        key = (r["domain"], r["arch"])
        if key not in best_key or r["ss_mae"] < best_key[key]:
            best_key[key] = r["ss_mae"]

    cell_text   = []
    cell_colors = []
    prev_group  = None

    for r in all_rows:
        gain   = (r["full_mae"] - r["ss_mae"]) / r["full_mae"] * 100
        status = ("Good"   if r["ss_l2"] < 0.05 else
                  "Fair"   if r["ss_l2"] < 0.10 else
                  "Poor")
        is_best = abs(r["ss_mae"] - best_key[(r["domain"], r["arch"])]) < 1e-9

        cell_text.append([
            r["domain"].capitalize(),
            ARCH_LABELS[r["arch"]],
            str(r["k"]),
            f"{r['full_mae']:.2f} °C",
            f"{r['ss_mae']:.2f} °C",
            f"−{gain:.1f} %",
            f"{r['full_l2']:.4f}",
            f"{r['ss_l2']:.4f}",
            f"{'★ ' if is_best else ''}{status}",
        ])

        # Row background: alternate by domain group
        group = (r["domain"], r["arch"])
        if group != prev_group:
            row_bg_cur = row_bg
            prev_group = group
        else:
            row_bg_cur = row_bg

        l2_col = _BEST if is_best else _cell_color(r["ss_l2"])
        base   = _BEST if is_best else row_bg_cur
        cell_colors.append([base]*6 + [base, l2_col, l2_col])

    n_rows = len(cell_text)
    n_cols = len(col_headers)

    fig_h = max(3.0, 0.38 * n_rows + 1.4)
    fig, ax = plt.subplots(figsize=(17, fig_h))
    fig.patch.set_facecolor(_WHITE)
    ax.set_facecolor(_WHITE)
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    for col in range(n_cols):
        cell = tbl[0, col]
        cell.set_facecolor(_HEADER)
        cell.set_text_props(color="white", fontweight="bold")

    col_widths = [0.10, 0.09, 0.04, 0.09, 0.09, 0.09, 0.08, 0.08, 0.09]
    for col, w in enumerate(col_widths):
        for row in range(n_rows + 1):
            tbl[row, col].set_width(w)

    ax.set_title(title, fontsize=11, fontweight="bold",
                 color="#1a1a1a", pad=14, loc="left")

    note = (f"SS MAE = Steady-State MAE  |  "
            f"First {skip} windows excluded (transient phase)  |  "
            f"★ = best k per (domain, optimizer)  |  "
            f"Green < 5%  /  Amber < 10%  /  Red ≥ 10% relative L2")
    fig.text(0.01, 0.005, note, fontsize=7.5, color="#555555", style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight", facecolor=_WHITE)
    plt.close(fig)
    print(f"  → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip", type=int, default=3,
                   help="Number of transient windows to exclude (default: 3)")
    args = p.parse_args()

    summary_dir = RESULT_DIR / "summary"

    for dim, domains, label, bg in [
        (2, DOMAINS_2D, "2D", "#EEF5FB"),
        (3, DOMAINS_3D, "3D", "#FFF8EE"),
    ]:
        all_rows = []
        for domain in domains:
            for arch in ARCHS:
                all_rows.extend(_load_all_k(domain, arch, dim, skip=args.skip))

        title = (f"NAS-PINN k-Skip  —  {label} Domains  |  "
                 f"Steady-State Performance (first {args.skip} windows excluded)")
        out   = summary_dir / f"fig_ss_table_{label.lower()}.png"
        _make_table(all_rows, title, out, skip=args.skip, row_bg=bg)
        print(f"  {label}: {len(all_rows)} rows")

    print("\nDone.")
