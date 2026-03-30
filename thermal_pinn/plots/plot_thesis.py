"""
thermal_pinn/plot_thesis.py
============================
Publication-quality thesis figures for the PINN thermal quenching study.
Styled after the level9_sa_fourier reference plots (YlGnBu / YlOrRd colormaps,
white background, blue-header tables, green-highlighted best values).

Figures produced
----------------
  results/{arch}/{domain}/fig_th1_2d_{domain}_{arch}_k{k}.png  — per-arch×domain×k heatmaps
  results/summary/fig_th2a_all_k_table.png  — all k values table
  results/summary/fig_th2b_best_k_table.png — best k only table
  results/summary/fig_th3_l2_comparison.png — grouped bar chart
  results/summary/fig_th4_k_progression_2d.png
  results/summary/fig_th4_k_progression_3d.png

Usage
-----
    python thermal_pinn/plot_thesis.py              # all figures, all k, all archs
    python thermal_pinn/plot_thesis.py --fig 1 --k 2 --arch bayesian --domain rectangle
    python thermal_pinn/plot_thesis.py --fig 2
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
warnings.filterwarnings("ignore")

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

# ── Shared helpers from plot_results ─────────────────────────────────────────
from thermal_pinn.plots.plot_results import (
    load_registry, load_model, load_metrics, best_k,
    eval_grid_2d, eval_grid_3d_mid, _make_2d_grid, _make_3d_midgrid,
    _clean_T, _adaptive_norm, _pinn_forward_full,
    _render_box_3d, _render_cylinder_3d, _render_lshape_3d,
    _style_3d_ax, _make_pinn_field_fn, _surface_mae_3d,
    CKPT_DIR, RESULT_DIR,
    T_WATER, T_INIT, DELTA_T, DEVICE,
    _ARCH_ORDER, _ARCH_LABELS, _ARCH_COLORS,
)
from thermal_pinn.network.fourier_pinn import FourierPINN


def load_registry_v2() -> list[dict]:
    with open(CKPT_DIR / "registry_v2.json") as f:
        return json.load(f)


def load_model_v2(domain: str, arch: str, k: int, dim: int):
    ckpt_path = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}_v2.pt"
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = FourierPINN(dim=dim, arch=arch).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _load_model_auto(domain: str, arch: str, k: int, dim: int, suffix: str = ""):
    if suffix == "_v2":
        return load_model_v2(domain, arch, k, dim)
    return load_model(domain, arch, k, dim)
from thermal_pinn.physics.domains_2d import make_domain, DOMAINS_2D
from thermal_pinn.physics.domains_3d import make_domain_3d, DOMAINS_3D

# ── Thesis style constants ────────────────────────────────────────────────────
CMAP_T   = "YlGnBu"          # dark navy=cold, light yellow=hot  (reversed)
CMAP_E   = "YlOrRd"          # light=small error, dark red=large
_WHITE   = "#ffffff"
_HEADER  = "#1565C0"          # dark blue table header background
_ROW_2D  = "#EEF5FB"          # light blue tint for 2D rows
_ROW_3D  = "#FFF8EE"          # light amber tint for 3D rows
_BEST_G  = "#C8E6C9"          # green highlight for best L2

_FIG_DPI = 150

DOMAINS_2D_LIST = ["rectangle", "circle", "lshape"]
DOMAINS_3D_LIST = ["rectangular", "cylinder", "stacked", "lshape"]

# Query times (seconds) used for heatmap panels
_HEATMAP_TIMES = [3.0, 9.0, 18.0, 27.0]


# ════════════════════════════════════════════════════════════════════════════
# Low-level drawing helpers
# ════════════════════════════════════════════════════════════════════════════

def _draw_heatmap(ax, XX, YY, T, norm, cmap, title="", fontsize=8):
    """pcolormesh with white isolines and tight axes."""
    T_plot = np.where(np.isfinite(T), T, np.nan)
    im = ax.pcolormesh(XX, YY, T_plot, cmap=cmap, norm=norm,
                       shading="auto", rasterized=True)
    fin = T_plot[np.isfinite(T_plot)]
    if fin.size > 0 and fin.max() - fin.min() > 0.5:
        try:
            ax.contour(XX, YY, T_plot, levels=6,
                       colors="white", linewidths=0.4, alpha=0.6)
        except Exception:
            pass
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=6)
    ax.set_ylabel("y [m]", fontsize=6)
    ax.tick_params(labelsize=6)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    if title:
        ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=3)
    return im


def _draw_error_map(ax, XX, YY, err, norm_e, title="", fontsize=8):
    """pcolormesh for |error| field with YlOrRd colormap."""
    err_plot = np.where(np.isfinite(err), np.abs(err), np.nan)
    im = ax.pcolormesh(XX, YY, err_plot, cmap=CMAP_E, norm=norm_e,
                       shading="auto", rasterized=True)
    fin = err_plot[np.isfinite(err_plot)]
    if fin.size > 0 and fin.max() - fin.min() > 0.1:
        try:
            ax.contour(XX, YY, err_plot, levels=4,
                       colors="#333333", linewidths=0.3, alpha=0.5)
        except Exception:
            pass
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=6)
    ax.set_ylabel("y [m]", fontsize=6)
    ax.tick_params(labelsize=6)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    if title:
        ax.set_title(title, fontsize=fontsize, fontweight="bold",
                     color="#B71C1C", pad=3)
    return im


def _add_colorbar(fig, ax, im, label, fontsize=6):
    """Tight per-axes colorbar via make_axes_locatable."""
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.04)
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=fontsize)
    cb.ax.tick_params(labelsize=fontsize - 1)
    return cb


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — per-arch × domain × k 2D heatmaps
# ════════════════════════════════════════════════════════════════════════════

def fig_th1_heatmaps(registry: list[dict], k: int, arch: str,
                     domain_name: str, out_dir: Path, n_grid: int = 80,
                     suffix: str = ""):
    """
    3 rows × 4 cols figure:
      Row 0 — Reference temperature at each of 4 query times
      Row 1 — PINN prediction for given arch
      Row 2 — |Error| = |pred - ref|

    Each (arch, domain, k) combination produces its own file at:
      out_dir/{arch}/{domain_name}/fig_th1_2d_{domain_name}_{arch}_k{k}.png
    """
    save_dir = out_dir / arch / domain_name
    save_dir.mkdir(parents=True, exist_ok=True)

    out_path = save_dir / f"fig_th1_2d_{domain_name}_{arch}_k{k}{suffix}.png"
    if out_path.exists():
        print(f"  [Fig TH1] {domain_name}/{arch}/k={k} — already exists, skipping")
        return

    ckpt_path = CKPT_DIR / f"{domain_name}_{arch}_k{k}_dim2{suffix}.pt"
    if not ckpt_path.exists():
        print(f"  [Fig TH1] {domain_name}/{arch}/k={k} — checkpoint missing, skip")
        return

    print(f"  [Fig TH1] {domain_name}/{arch}/k={k} — evaluating …")

    domain = make_domain(domain_name)
    model  = _load_model_auto(domain_name, arch, k, dim=2, suffix=suffix)

    times = _HEATMAP_TIMES
    n_t   = len(times)

    # ── Evaluate all grids ────────────────────────────────────────────────
    grids = []
    for t in times:
        g = eval_grid_2d(model, domain, k, t, n_grid=n_grid)
        grids.append(g)

    XX, YY = grids[0]["xx"], grids[0]["yy"]

    # ── Per-column norms (per time step) so spatial gradients are visible ──
    # Using per-time norm ensures t=9s internal gradients are not washed out
    # by the wider range of t=27s. Both ref and pred share the same norm per col.
    col_norm_T = []
    for g in grids:
        all_vals = np.concatenate([
            g["T_ref"][np.isfinite(g["T_ref"])].ravel(),
            g["T_pred"][np.isfinite(g["T_pred"])].ravel(),
        ])
        if all_vals.size > 0:
            v_min = float(np.percentile(all_vals, 2))
            v_max = float(np.percentile(all_vals, 98))
            if v_max - v_min < 0.5:
                v_min -= 0.5
                v_max += 0.5
        else:
            v_min, v_max = T_WATER, T_INIT
        col_norm_T.append(Normalize(vmin=v_min, vmax=v_max))

    # Global error norm across all time steps
    all_err = np.concatenate([g["err"][np.isfinite(g["err"])].ravel()
                               for g in grids])
    err_max = float(np.percentile(all_err, 97)) if all_err.size else 1.0
    err_max = max(err_max, 0.5)
    norm_E  = Normalize(vmin=0, vmax=err_max)

    # ── Layout ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, n_t, figsize=(n_t * 4.0, 3 * 3.5))
    fig.patch.set_facecolor(_WHITE)

    dt_val = k * 1.5
    arch_label = _ARCH_LABELS.get(arch, arch)
    fig.suptitle(
        f"2D Quenching — Reference vs Prediction vs Error\n"
        f"Domain: {domain_name}  |  Optimizer: {arch_label}  |  k={k} (Δt={dt_val:.1f}s)",
        fontsize=11, fontweight="bold", y=1.02,
    )

    row_labels = ["Reference (FEM)", f"PINN ({arch_label})", "|Error|  (°C)"]

    for col, (t_q, g, n_T) in enumerate(zip(times, grids, col_norm_T)):
        l2_val = g["l2"]

        # Row 0 — Reference
        im_ref = _draw_heatmap(axes[0, col], XX, YY, g["T_ref"],
                                n_T, CMAP_T, title=f"t = {t_q:.0f} s")

        # Row 1 — PINN prediction
        l2_str = f"L2={l2_val:.4f}" if np.isfinite(l2_val) else "L2=n/a"
        im_pred = _draw_heatmap(axes[1, col], XX, YY, g["T_pred"],
                                 n_T, CMAP_T,
                                 title=f"t = {t_q:.0f} s  [{l2_str}]")

        # Row 2 — |Error|
        fin_err = g["err"][np.isfinite(g["err"])]
        mae_str = f"MAE={np.mean(fin_err):.2f}°C" if fin_err.size else ""
        im_err = _draw_error_map(axes[2, col], XX, YY, g["err"],
                                  norm_E, title=f"t = {t_q:.0f} s  [{mae_str}]")

        # Colorbars
        _add_colorbar(fig, axes[0, col], im_ref,  "T [°C]")
        _add_colorbar(fig, axes[1, col], im_pred, "T [°C]")
        _add_colorbar(fig, axes[2, col], im_err,  "|ΔT| [°C]")

    # Row labels on left-most column y-axis
    for row_i, rl in enumerate(row_labels):
        axes[row_i, 0].set_ylabel(f"{rl}\ny [m]", fontsize=7, labelpad=4)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight", facecolor=_WHITE)
    plt.close(fig)
    print(f"    → {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — comprehensive L2 + runtime tables
# ════════════════════════════════════════════════════════════════════════════

def _collect_all_k_data(registry: list[dict], suffix: str = "",
                        ss_skip: int = 0) -> list[dict]:
    """
    Returns one row per (domain, dim, arch, k) for ALL k values found.
    Used for the full table (fig_th2a).
    If ss_skip > 0, MAE and L2 are computed over windows[ss_skip:] only.
    """
    rows = []
    all_combinations = (
        [(d, 2) for d in DOMAINS_2D_LIST] +
        [(d, 3) for d in DOMAINS_3D_LIST]
    )

    for domain_name, dim in all_combinations:
        for arch in _ARCH_ORDER:
            for k in range(1, 6):
                mp = CKPT_DIR / f"{domain_name}_{arch}_k{k}_dim{dim}{suffix}_metrics.json"
                if not mp.exists():
                    continue
                m = json.load(open(mp))
                wins = m.get("windows", [])
                if not wins:
                    continue

                ss_wins         = wins[ss_skip:] if len(wins) > ss_skip else wins
                mean_l2         = float(np.mean([w["l2_rel"]    for w in ss_wins]))
                total_runtime   = float(np.sum ([w["runtime_s"] for w in wins]))
                runtime_per_win = float(np.mean([w["runtime_s"] for w in wins]))
                mean_mae        = float(np.mean([w["mae_C"]     for w in ss_wins]))
                n_windows       = m["n_windows"]

                rows.append({
                    "domain":    domain_name,
                    "dim":       dim,
                    "arch":      arch,
                    "k":         k,
                    "mean_l2":   mean_l2,
                    "mean_mae":  mean_mae,
                    "rt_win":    runtime_per_win,
                    "rt_total":  total_runtime,
                    "n_windows": n_windows,
                })

    arch_rank = {a: i for i, a in enumerate(_ARCH_ORDER)}
    rows.sort(key=lambda r: (r["dim"], r["domain"], arch_rank.get(r["arch"], 99), r["k"]))
    return rows


def _collect_best_k_data(registry: list[dict], suffix: str = "",
                         ss_skip: int = 0) -> list[dict]:
    """
    Returns one row per (domain, dim, arch) using the best k (min mean_mae).
    Used for the best-k summary table (fig_th2b).
    If ss_skip > 0, MAE and L2 are computed over windows[ss_skip:] only.
    """
    rows = []
    all_combinations = (
        [(d, 2) for d in DOMAINS_2D_LIST] +
        [(d, 3) for d in DOMAINS_3D_LIST]
    )

    for domain_name, dim in all_combinations:
        for arch in _ARCH_ORDER:
            best_k_val   = None
            best_mae     = float("inf")
            best_metrics = None

            for k in range(1, 6):
                mp = CKPT_DIR / f"{domain_name}_{arch}_k{k}_dim{dim}{suffix}_metrics.json"
                if not mp.exists():
                    continue
                m = json.load(open(mp))
                wins_k  = m.get("windows", [])
                ss_wins = wins_k[ss_skip:] if len(wins_k) > ss_skip else wins_k
                cmp_mae = float(np.mean([w["mae_C"] for w in ss_wins]))
                if cmp_mae < best_mae:
                    best_mae     = cmp_mae
                    best_k_val   = k
                    best_metrics = m

            if best_metrics is None:
                continue

            wins            = best_metrics["windows"]
            ss_wins         = wins[ss_skip:] if len(wins) > ss_skip else wins
            mean_l2         = float(np.mean([w["l2_rel"]    for w in ss_wins]))
            mean_mae_ss     = float(np.mean([w["mae_C"]     for w in ss_wins]))
            total_runtime   = float(np.sum ([w["runtime_s"] for w in wins]))
            runtime_per_win = float(np.mean([w["runtime_s"] for w in wins]))
            n_windows       = best_metrics["n_windows"]

            rows.append({
                "domain":    domain_name,
                "dim":       dim,
                "arch":      arch,
                "best_k":    best_k_val,
                "mean_l2":   mean_l2,
                "mean_mae":  mean_mae_ss,
                "rt_win":    runtime_per_win,
                "rt_total":  total_runtime,
                "n_windows": n_windows,
            })

    arch_rank = {a: i for i, a in enumerate(_ARCH_ORDER)}
    rows.sort(key=lambda r: (r["dim"], r["domain"], arch_rank.get(r["arch"], 99)))
    return rows


def _render_table(ax, col_headers: list[str], cell_text: list[list],
                  cell_color: list[list], title: str):
    """Render a styled matplotlib table on axes ax."""
    n_rows = len(cell_text)
    n_cols = len(col_headers)

    tbl = ax.table(
        cellText=cell_text,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.3)

    for (row_i, col_j), cell in tbl.get_celld().items():
        if row_i < len(cell_color) and col_j < len(cell_color[row_i]):
            cell.set_facecolor(cell_color[row_i][col_j])
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)
        if row_i == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(7.5)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=8, color="#1A237E")


def fig_th2a_matrix_allk(registry: list[dict], out_dir: Path, suffix: str = "",
                         ss_skip: int = 0):
    """
    Compact L2 color matrix:
      Rows    = domain × arch combinations (grouped by domain)
      Columns = k = 1 … 5
      Cell    = L2 value, color-coded green (low) → red (high)

    Two panels stacked: 2D domains (top) and 3D domains (bottom).
    Saved to: out_dir/fig_th2a_l2_matrix.png
    """
    print("  [Fig TH2a] L2 color matrix …")
    rows = _collect_all_k_data(registry, suffix=suffix, ss_skip=ss_skip)
    if not rows:
        print("    WARN: no data — skipping")
        return

    k_cols = [1, 2, 3, 4, 5]

    def _build_matrix(dim, dom_list):
        """Returns row_labels list and 2D array (n_rows × 5) of L2 values."""
        labels = []
        mat    = []
        for domain in dom_list:
            for arch in _ARCH_ORDER:
                labels.append(f"{domain}\n{_ARCH_LABELS.get(arch, arch)}")
                row_vals = []
                for k in k_cols:
                    match = [r for r in rows
                             if r["domain"] == domain and r["arch"] == arch
                             and r["dim"] == dim and r["k"] == k]
                    row_vals.append(match[0]["mean_l2"] if match else np.nan)
                mat.append(row_vals)
        return labels, np.array(mat, dtype=float)

    labels_2d, mat_2d = _build_matrix(2, DOMAINS_2D_LIST)
    labels_3d, mat_3d = _build_matrix(3, DOMAINS_3D_LIST)

    # Shared vmax = 95th percentile of all finite values for readable color scale
    all_vals = np.concatenate([
        mat_2d[np.isfinite(mat_2d)], mat_3d[np.isfinite(mat_3d)]
    ])
    vmax = float(np.percentile(all_vals, 95)) if all_vals.size else 1.0
    vmax = max(vmax, 0.05)

    cmap_mat = plt.cm.RdYlGn_r   # green=low L2 (good), red=high (bad)

    nrows_2d = len(labels_2d)
    nrows_3d = len(labels_3d)

    fig_h = 0.55 * (nrows_2d + nrows_3d) + 3.0
    fig   = plt.figure(figsize=(9, fig_h), facecolor=_WHITE)
    gs    = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45,
                              height_ratios=[nrows_2d, nrows_3d])

    for panel_i, (mat, labels, title_sfx) in enumerate([
        (mat_2d, labels_2d, "2D Domains"),
        (mat_3d, labels_3d, "3D Domains"),
    ]):
        ax = fig.add_subplot(gs[panel_i])
        im = ax.imshow(mat, aspect="auto", cmap=cmap_mat,
                       vmin=0, vmax=vmax, interpolation="nearest")

        # Annotate each cell with the L2 value
        for ri in range(mat.shape[0]):
            for ci in range(mat.shape[1]):
                val = mat[ri, ci]
                if np.isfinite(val):
                    txt_color = "black" if val < vmax * 0.6 else "white"
                    # Star for best k per (domain, arch)
                    row_vals = mat[ri, :]
                    is_best  = np.nanargmin(row_vals) == ci
                    star     = "★" if is_best else ""
                    ax.text(ci, ri, f"{val:.3f}{star}",
                            ha="center", va="center",
                            fontsize=7.5, fontweight="bold" if is_best else "normal",
                            color=txt_color)
                else:
                    ax.text(ci, ri, "—", ha="center", va="center",
                            fontsize=7, color="#aaaaaa")

        ax.set_xticks(range(len(k_cols)))
        ax.set_xticklabels([f"k={k}" for k in k_cols], fontsize=9)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_title(f"L2 Relative Error — {title_sfx}\n"
                     f"(★ = best k per row | green=low, red=high)",
                     fontsize=9, fontweight="bold", color="#1A237E", pad=6)

        # Domain separator lines every 3 rows (one per domain)
        n_arch = len(_ARCH_ORDER)
        for sep in range(n_arch, len(labels), n_arch):
            ax.axhline(sep - 0.5, color="#555555", linewidth=1.2, alpha=0.6)

        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label("L2 Relative Error", fontsize=7)
        cb.ax.tick_params(labelsize=6)

    fig.suptitle("NAS-PINN k-Skip: L2 Matrix per Domain × Optimizer × k",
                 fontsize=11, fontweight="bold", y=1.01)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag      = f"_ss{ss_skip}" if ss_skip > 0 else ""
    out_path = out_dir / f"fig_th2a_l2_matrix{tag}.png"
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight", facecolor=_WHITE)
    plt.close(fig)
    print(f"    → {out_path.name}")


def fig_th2b_table_bestk(registry: list[dict], out_dir: Path, suffix: str = "",
                         ss_skip: int = 0):
    """
    Best-k summary table: one row per (domain, dim, arch) using best k.
    Saved to: out_dir/summary/fig_th2b_best_k_table.png
    """
    print("  [Fig TH2b] Best-k summary table …")
    rows = _collect_best_k_data(registry, suffix=suffix, ss_skip=ss_skip)
    if not rows:
        print("    WARN: no data — skipping")
        return

    # Best L2 per (domain, dim) for green highlight
    best_l2_key: dict[tuple, float] = {}
    for r in rows:
        key = (r["domain"], r["dim"])
        if key not in best_l2_key or r["mean_l2"] < best_l2_key[key]:
            best_l2_key[key] = r["mean_l2"]

    col_headers = [
        "Domain", "Dim", "Optimizer", "Best k",
        "L2 Rel", "MAE [°C]", "RT/win [s]", "Total RT [s]",
        "Windows", "Notes",
    ]
    n_cols = len(col_headers)

    cell_text  = [col_headers]
    cell_color = [[_HEADER] * n_cols]

    for r in rows:
        key     = (r["domain"], r["dim"])
        is_best = abs(r["mean_l2"] - best_l2_key[key]) < 1e-9
        bg_base = _ROW_2D if r["dim"] == 2 else _ROW_3D
        bg      = _BEST_G if is_best else bg_base
        note    = "★ best" if is_best else ""

        cell_text.append([
            r["domain"],
            str(r["dim"]) + "D",
            _ARCH_LABELS.get(r["arch"], r["arch"]),
            str(r["best_k"]),
            f"{r['mean_l2']:.4f}",
            f"{r['mean_mae']:.2f}",
            f"{r['rt_win']:.1f}",
            f"{r['rt_total']:.1f}",
            str(r["n_windows"]),
            note,
        ])
        cell_color.append([bg] * n_cols)

    n_rows = len(cell_text)
    row_h  = 0.42
    fig_h  = max(4.0, (n_rows + 2) * row_h)
    fig, ax = plt.subplots(figsize=(22, fig_h))
    fig.patch.set_facecolor(_WHITE)
    ax.axis("off")

    _render_table(ax, col_headers, cell_text, cell_color,
                  "PINN k-Skip Performance — Best k per Optimizer × Domain\n"
                  "(Green = best L2 per domain·dim | 2D domains top, 3D bottom)")

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag      = f"_ss{ss_skip}" if ss_skip > 0 else ""
    out_path = out_dir / f"fig_th2b_best_k_table{tag}.png"
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight", facecolor=_WHITE)
    plt.close(fig)
    print(f"    → {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 1b — per-arch × 3D domain × k mid-plane heatmaps
# ════════════════════════════════════════════════════════════════════════════

def fig_th1_3d_heatmaps(registry: list[dict], k: int, arch: str,
                         domain_name: str, out_dir: Path, suffix: str = ""):
    """
    True 3D volumetric render per (arch, domain, k).
    2 rows × 4 cols layout:
      Row 0 — FEM reference volumetric solid
      Row 1 — PINN prediction volumetric solid (rainbow colormap)

    MAE annotation below each PINN panel.
    Saves to: out_dir/{arch}/{domain_name}_3d/fig_th1_3d_{domain_name}_{arch}_k{k}.png
    """
    save_dir = out_dir / arch / f"{domain_name}_3d"
    save_dir.mkdir(parents=True, exist_ok=True)

    out_path = save_dir / f"fig_th1_3d_{domain_name}_{arch}_k{k}{suffix}.png"
    if out_path.exists():
        print(f"  [Fig TH1-3D] {domain_name}/{arch}/k={k} — already exists, skipping")
        return

    ckpt_path = CKPT_DIR / f"{domain_name}_{arch}_k{k}_dim3{suffix}.pt"
    if not ckpt_path.exists():
        print(f"  [Fig TH1-3D] {domain_name}/{arch}/k={k} — checkpoint missing, skip")
        return

    print(f"  [Fig TH1-3D] {domain_name}/{arch}/k={k} — volumetric render …")

    domain    = make_domain_3d(domain_name)
    model     = _load_model_auto(domain_name, arch, k, dim=3, suffix=suffix)
    field_fn  = _make_pinn_field_fn(model, domain, k)

    times      = _HEATMAP_TIMES
    n_t        = len(times)
    cmap_vol   = plt.cm.rainbow
    norm_vol   = Normalize(vmin=T_WATER, vmax=T_INIT)
    arch_label = _ARCH_LABELS.get(arch, arch)

    # Dispatch renderer by domain type
    def _render(ax, t, fn=None):
        if domain_name in ("rectangular", "stacked"):
            _render_box_3d(ax, domain, t, cmap_vol, norm_vol, field_fn=fn)
        elif domain_name == "cylinder":
            _render_cylinder_3d(ax, domain, t, cmap_vol, norm_vol, field_fn=fn)
        else:  # lshape
            _render_lshape_3d(ax, domain, t, cmap_vol, norm_vol, field_fn=fn)

    fig = plt.figure(figsize=(n_t * 4.2, 2 * 4.5), facecolor=_WHITE)
    gs  = gridspec.GridSpec(2, n_t + 1,
                            width_ratios=[1.0] * n_t + [0.04],
                            hspace=0.06, wspace=0.04,
                            figure=fig)

    for col, t_q in enumerate(times):
        # Row 0 — FEM reference (no field_fn = uses domain.T)
        ax_ref = fig.add_subplot(gs[0, col], projection="3d")
        _render(ax_ref, t_q, fn=None)
        _style_3d_ax(ax_ref, domain)
        ax_ref.view_init(elev=25, azim=-60)
        ax_ref.set_title(f"t = {t_q:.0f} s", fontsize=8, fontweight="bold", pad=3)
        if col == 0:
            ax_ref.text2D(-0.14, 0.48, "Reference\n(FEM)",
                          transform=ax_ref.transAxes, fontsize=8,
                          fontweight="bold", va="center", ha="center",
                          rotation=90, color="#1A237E")

        # Row 1 — PINN prediction
        ax_pinn = fig.add_subplot(gs[1, col], projection="3d")
        _render(ax_pinn, t_q, fn=field_fn)
        _style_3d_ax(ax_pinn, domain)
        ax_pinn.view_init(elev=25, azim=-60)

        mae_v  = _surface_mae_3d(model, domain, k, t_q, n=10)
        mae_str = f"MAE={mae_v:.1f}°C" if np.isfinite(mae_v) else "MAE=n/a"
        ax_pinn.text2D(0.5, -0.04, mae_str,
                       transform=ax_pinn.transAxes, ha="center", va="top",
                       fontsize=7.5, color=_ARCH_COLORS.get(arch, "#333"),
                       fontweight="bold")
        if col == 0:
            ax_pinn.text2D(-0.14, 0.48, f"PINN\n({arch_label})",
                           transform=ax_pinn.transAxes, fontsize=8,
                           fontweight="bold", va="center", ha="center",
                           rotation=90, color="#B71C1C")

    # Shared colorbar column
    ax_cb = fig.add_subplot(gs[:, n_t])
    sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norm_vol)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.set_label("T [°C]", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    dt_val = k * 1.5
    fig.suptitle(
        f"3D Volumetric Temperature — {domain_name.capitalize()}\n"
        f"Optimizer: {arch_label}  |  k={k} (Δt={dt_val:.1f}s/window)",
        fontsize=10, fontweight="bold", y=1.01,
    )

    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight", facecolor=_WHITE)
    plt.close(fig)
    print(f"    → {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — grouped bar chart, L2 by domain and optimizer
# ════════════════════════════════════════════════════════════════════════════

def fig_th3_l2_bars(registry: list[dict], out_dir: Path, suffix: str = ""):
    """
    Two-panel grouped bar chart:
      Left  — 2D domains
      Right — 3D domains
    Each domain has 3 bars (Bayesian / NSGA-II / NSGA-III) with best-k L2.
    """
    print("  [Fig TH3] L2 bar chart …")

    rows = _collect_best_k_data(registry, suffix=suffix)

    def _get_l2(domain, arch, dim):
        for r in rows:
            if r["domain"] == domain and r["arch"] == arch and r["dim"] == dim:
                return r["mean_l2"]
        return np.nan

    fig, (ax2d, ax3d) = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor(_WHITE)
    fig.suptitle(
        "NAS-PINN k-Skip: L2 Relative Error by Domain and Optimizer",
        fontsize=12, fontweight="bold",
    )

    bar_w = 0.22

    for ax, dim, dom_list, panel_label in [
        (ax2d, 2, DOMAINS_2D_LIST, "2D Domains"),
        (ax3d, 3, DOMAINS_3D_LIST, "3D Domains"),
    ]:
        n_dom  = len(dom_list)
        x_base = np.arange(n_dom)
        n_arch = len(_ARCH_ORDER)
        offsets = np.linspace(-(n_arch - 1) / 2 * bar_w,
                               (n_arch - 1) / 2 * bar_w,
                               n_arch)

        for ai, arch in enumerate(_ARCH_ORDER):
            l2_vals = [_get_l2(d, arch, dim) for d in dom_list]
            color   = _ARCH_COLORS[arch]
            bars = ax.bar(
                x_base + offsets[ai], l2_vals,
                width=bar_w, color=color, alpha=0.85,
                label=_ARCH_LABELS[arch], zorder=3,
            )
            for x_pos, y_val in zip(x_base + offsets[ai], l2_vals):
                if np.isfinite(y_val):
                    ax.text(x_pos, y_val + 0.001, f"{y_val:.3f}",
                            ha="center", va="bottom", fontsize=6.5,
                            color="#333333")

        ax.axhline(y=0.05, color="#888888", linestyle="--",
                   linewidth=1.2, alpha=0.7, label="5% threshold", zorder=2)

        ax.set_xticks(x_base)
        ax.set_xticklabels([d.capitalize() for d in dom_list], fontsize=9)
        ax.set_ylabel("Relative L2 Error", fontsize=9)
        ax.set_title(panel_label, fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_facecolor(_WHITE)
        for sp in ax.spines.values():
            sp.set_linewidth(0.7)
        ax.legend(fontsize=8, framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_th3_l2_comparison.png"
    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight", facecolor=_WHITE)
    plt.close(fig)
    print(f"    → {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — per-k L2 progression (2×2 grid)
# ════════════════════════════════════════════════════════════════════════════

def _load_l2_per_k(domain_name: str, arch: str, dim: int,
                   suffix: str = "") -> dict[int, float]:
    """Load mean L2 for each k value from metrics JSON files."""
    result = {}
    for k in range(1, 6):
        mp = CKPT_DIR / f"{domain_name}_{arch}_k{k}_dim{dim}{suffix}_metrics.json"
        if not mp.exists():
            continue
        m = json.load(open(mp))
        wins = m.get("windows", [])
        if wins:
            result[k] = float(np.mean([w["l2_rel"] for w in wins]))
    return result


def _plot_k_progression_panel(ax, domain_name: str, dim: int,
                               is_first: bool = False, suffix: str = ""):
    """One subplot: L2 vs k for all 3 archs, with star marker at best k."""
    arch_styles = {
        "bayesian": ("solid",  "o"),
        "nsga2":    ("dashed", "s"),
        "nsga3":    ("dotted", "^"),
    }

    any_data = False
    for arch in _ARCH_ORDER:
        l2_map = _load_l2_per_k(domain_name, arch, dim, suffix=suffix)
        if not l2_map:
            continue
        any_data = True
        k_vals = sorted(l2_map)
        l2_vals = [l2_map[k] for k in k_vals]
        best_k_val = k_vals[int(np.argmin(l2_vals))]
        best_l2    = l2_map[best_k_val]

        ls, mk = arch_styles[arch]
        color  = _ARCH_COLORS[arch]
        label  = _ARCH_LABELS[arch] if is_first else "_nolegend_"

        ax.plot(k_vals, l2_vals, linestyle=ls, marker=mk,
                color=color, linewidth=1.5, markersize=5, label=label)
        ax.plot(best_k_val, best_l2, marker="*", color=color,
                markersize=12, zorder=5, linestyle="none")

    ax.axhline(y=0.05, color="#888888", linestyle="--",
               linewidth=1.0, alpha=0.7)

    ax.set_title(domain_name.capitalize(), fontsize=9, fontweight="bold")
    ax.set_xlabel("k (skip window size)", fontsize=8)
    ax.set_ylabel("Mean Relative L2", fontsize=8)
    ax.set_xticks(range(1, 6))
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_facecolor(_WHITE)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)

    if not any_data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=9)
    return any_data


def fig_th4_k_progression(registry: list[dict], out_dir: Path, suffix: str = ""):
    """
    Two separate figures (2D / 3D), each a 2×2 grid of domain subplots.
    """
    print("  [Fig TH4] k-progression charts …")

    for label, dim, dom_list in [
        ("2d", 2, DOMAINS_2D_LIST),
        ("3d", 3, DOMAINS_3D_LIST),
    ]:
        n_dom = len(dom_list)
        ncols = 2
        nrows = (n_dom + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(6.5 * ncols, 4.5 * nrows))
        fig.patch.set_facecolor(_WHITE)
        axes_flat = axes.ravel()

        for i, domain_name in enumerate(dom_list):
            is_first = (i == 0)
            _plot_k_progression_panel(axes_flat[i], domain_name, dim, is_first,
                                      suffix=suffix)

        for j in range(n_dom, len(axes_flat)):
            axes_flat[j].set_visible(False)

        handles, lbls = axes_flat[0].get_legend_handles_labels()
        if handles:
            from matplotlib.lines import Line2D
            star_proxy  = Line2D([0], [0], marker="*", color="#555555",
                                 linestyle="none", markersize=10, label="Best k")
            thresh_proxy = Line2D([0], [0], linestyle="--", color="#888888",
                                   linewidth=1.0, label="5% threshold")
            handles = handles + [star_proxy, thresh_proxy]
            lbls    = lbls    + ["Best k",   "5% threshold"]
            axes_flat[0].legend(handles, lbls,
                                fontsize=8, framealpha=0.85, loc="upper right")

        dim_str = "2D" if dim == 2 else "3D"
        fig.suptitle(
            f"L2 Error vs k-Skip Window Size — {dim_str} Domains",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"fig_th4_k_progression_{label}.png"
        fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight", facecolor=_WHITE)
        plt.close(fig)
        print(f"    → {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# main()
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate thesis figures for thermal PINN study."
    )
    p.add_argument("--fig", type=int, default=0,
                   help="Figure to generate (0=all, 1=2D heatmaps, 13=3D heatmaps, "
                        "2=tables, 3=bars, 4=k-prog)")
    p.add_argument("--k", type=int, default=0,
                   help="k value for Fig 1/13 (0 = all k from 1 to 5)")
    p.add_argument("--arch", type=str, default="all",
                   help="Optimizer: bayesian | nsga2 | nsga3 | all")
    p.add_argument("--domain", type=str, default="all",
                   help="Domain (2D or 3D name) | all")
    p.add_argument("--v2", action="store_true",
                   help="Use v2 (Fourier+SA) models and registry_v2.json")
    p.add_argument("--ss", action="store_true",
                   help="Use Steady-State MAE (exclude first --skip transient windows)")
    p.add_argument("--skip", type=int, default=3,
                   help="Number of transient windows to exclude when --ss is set (default: 3)")
    args = p.parse_args()

    if args.v2:
        reg    = load_registry_v2()
        suffix = "_v2"
        res_dir = RESULT_DIR.parent / "results_v2"
    else:
        reg    = load_registry()
        suffix = ""
        res_dir = RESULT_DIR

    ss_skip = args.skip if args.ss else 0

    # Summary figures go to results[_v2]/summary/
    summary_dir = res_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    run_all = (args.fig == 0)

    # ── Fig 1: per-arch × 2D domain × k heatmaps ─────────────────────────
    if run_all or args.fig == 1:
        k_list      = list(range(1, 6)) if args.k == 0 else [args.k]
        arch_list   = _ARCH_ORDER if args.arch == "all" else [args.arch]
        domain_list = DOMAINS_2D_LIST if args.domain == "all" else [args.domain]

        total = len(k_list) * len(arch_list) * len(domain_list)
        done  = 0
        for arch in arch_list:
            for domain_name in domain_list:
                for kv in k_list:
                    fig_th1_heatmaps(reg, kv, arch, domain_name, res_dir, suffix=suffix)
                    done += 1
                    print(f"  2D Progress: {done}/{total}")

    # ── Fig 13: per-arch × 3D domain × k volumetric heatmaps ─────────────
    if run_all or args.fig == 13:
        k_list      = list(range(1, 6)) if args.k == 0 else [args.k]
        arch_list   = _ARCH_ORDER if args.arch == "all" else [args.arch]
        domain_list = DOMAINS_3D_LIST if args.domain == "all" else [args.domain]

        total = len(k_list) * len(arch_list) * len(domain_list)
        done  = 0
        for arch in arch_list:
            for domain_name in domain_list:
                for kv in k_list:
                    fig_th1_3d_heatmaps(reg, kv, arch, domain_name, res_dir, suffix=suffix)
                    done += 1
                    print(f"  3D Progress: {done}/{total}")

    # ── Fig 2: tables ─────────────────────────────────────────────────────
    if run_all or args.fig == 2:
        fig_th2a_matrix_allk(reg, summary_dir, suffix=suffix, ss_skip=ss_skip)
        fig_th2b_table_bestk(reg, summary_dir, suffix=suffix, ss_skip=ss_skip)

    # ── Fig 3: bar chart ──────────────────────────────────────────────────
    if run_all or args.fig == 3:
        fig_th3_l2_bars(reg, summary_dir, suffix=suffix)

    # ── Fig 4: k-progression ─────────────────────────────────────────────
    if run_all or args.fig == 4:
        fig_th4_k_progression(reg, summary_dir, suffix=suffix)

    print("\nThesis figures done.")
    print(f"  Heatmaps: {res_dir}/{{arch}}/{{domain}}/")
    print(f"  Summary:  {summary_dir}/")
