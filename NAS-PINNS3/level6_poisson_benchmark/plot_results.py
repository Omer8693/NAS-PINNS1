"""
plot_results.py — Level 6 Visualization
=========================================
Compares Level 1 → Level 5 → Level 6 quenching accuracy.

Figures saved to results/level6/plots/:
  l6_l2_progression.png     — L2 improvement chain: L1 → L5 → L6
  l6_improvement.png        — Before/after bar chart (L5 vs L6) per optimizer
  l6_training_curve.png     — Total loss + Poisson aux loss during fine-tuning
  l6_summary_table.png      — Full metrics table across all three levels
  l6_heatmap_t5s.png        — T(x,y) heatmaps at t=5s: L5 vs L6 (all opts)
  l6_heatmap_timeslices.png — T(x,y) at t=1,5,15,30s for best optimizer

Usage:
    cd /home/coder/NAS-PINNS1/NAS-PINNS3
    python level6_poisson_benchmark/plot_results.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

L5_SUMMARY = _ROOT / "level5_refinement" / "results" / "level5_summary.json"
L6_DIR     = _ROOT / "level6_poisson_benchmark" / "results"

OPT_COLORS = {
    "bayesian": "#1565C0",
    "nsga2":    "#2E7D32",
    "nsga3":    "#E65100",
}
TARGET_L2  = 0.100
OPTIMIZERS = ["bayesian", "nsga2", "nsga3"]


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────────

def _load_l5() -> dict:
    if not L5_SUMMARY.exists():
        return {}
    with open(L5_SUMMARY) as f:
        return json.load(f)


def _load_l6(opt: str) -> dict:
    p = L6_DIR / opt / "result.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — L2 progression chain: L1 → L5 → L6
# ──────────────────────────────────────────────────────────────────────────────

def plot_l2_progression(l5: dict, out_dir: Path) -> None:
    levels = ["Level 1\n(Adam)", "Level 5\n(Cosine LR)", "Level 6\n(+Poisson)"]
    fig, ax = plt.subplots(figsize=(9, 5))

    for opt in OPTIMIZERS:
        l5d = l5.get(opt, {})
        l6d = _load_l6(opt)

        l2_l1 = (l5d.get("level1_adam_only") or {}).get("L2_rel")
        l2_l5 = (l5d.get("level5_adam_lbfgs") or {}).get("L2_rel")
        l2_l6 = (l6d.get("level6_after") or {}).get("L2_rel")

        vals = [l2_l1, l2_l5, l2_l6]
        if any(v is None for v in vals):
            continue

        color = OPT_COLORS[opt]
        ax.plot(levels, vals, "o-", color=color, linewidth=2, markersize=8,
                label=f"{opt.upper()}")

        # Annotate final value
        ax.annotate(f"{l2_l6:.4f}", (levels[-1], l2_l6),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=8, color=color)

    ax.axhline(TARGET_L2, color="#B71C1C", linestyle="--", linewidth=1.5,
               label=f"Target (< {TARGET_L2})")
    ax.set_ylabel("Relative L₂ Error", fontsize=11)
    ax.set_title("Quenching PINN Improvement Chain\n"
                 "Level 1 → Level 5 (cosine LR) → Level 6 (+Poisson auxiliary)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = out_dir / "l6_l2_progression.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Before/After improvement bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_improvement(l5: dict, out_dir: Path) -> None:
    opts_valid = [o for o in OPTIMIZERS if _load_l6(o)]
    if not opts_valid:
        print("  [skip] No Level 6 results.")
        return

    x      = np.arange(len(opts_valid))
    width  = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    l2_l5_vals, l2_l6_vals = [], []
    mae_l5_vals, mae_l6_vals = [], []

    for opt in opts_valid:
        l6d = _load_l6(opt)
        # Use level5_before from result.json — same evaluation as level6_after
        l2_l5_vals.append( (l6d.get("level5_before") or {}).get("L2_rel", 0) )
        l2_l6_vals.append( (l6d.get("level6_after")  or {}).get("L2_rel", 0) )
        mae_l5_vals.append((l6d.get("level5_before") or {}).get("MAE_C",  0) )
        mae_l6_vals.append((l6d.get("level6_after")  or {}).get("MAE_C",  0) )

    labels = [o.upper() for o in opts_valid]

    # ── rel L2 ────────────────────────────────────────────────────────────────
    ax = axes[0]
    b1 = ax.bar(x - width/2, l2_l5_vals, width, label="Level 5", color="#64B5F6", edgecolor="white")
    b2 = ax.bar(x + width/2, l2_l6_vals, width, label="Level 6", color="#1565C0", edgecolor="white")
    ax.axhline(TARGET_L2, color="#B71C1C", linestyle="--", linewidth=1.2,
               label=f"Target ({TARGET_L2})")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Relative L₂ Error", fontsize=10)
    ax.set_title("L₂ Error: Level 5 vs Level 6", fontsize=10)
    ax.legend(fontsize=9)
    for bar in [*b1, *b2]:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002, f"{v:.4f}",
                ha="center", fontsize=8)

    # ── MAE ────────────────────────────────────────────────────────────────────
    ax2 = axes[1]
    b3 = ax2.bar(x - width/2, mae_l5_vals, width, label="Level 5", color="#A5D6A7", edgecolor="white")
    b4 = ax2.bar(x + width/2, mae_l6_vals, width, label="Level 6", color="#2E7D32", edgecolor="white")
    ax2.axhline(50.0, color="#B71C1C", linestyle="--", linewidth=1.2, label="Target (50°C)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("MAE (°C)", fontsize=10)
    ax2.set_title("MAE: Level 5 vs Level 6", fontsize=10)
    ax2.legend(fontsize=9)
    for bar in [*b3, *b4]:
        v = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}",
                 ha="center", fontsize=8)

    plt.suptitle("Level 6 Improvement: Poisson Auxiliary Loss", fontsize=12)
    plt.tight_layout()
    out = out_dir / "l6_improvement.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Training curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(out_dir: Path) -> None:
    opts_valid = [o for o in OPTIMIZERS if _load_l6(o)]
    if not opts_valid:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for opt in opts_valid:
        l6d   = _load_l6(opt)
        color = OPT_COLORS[opt]

        lh = l6d.get("loss_history", [])
        ph = l6d.get("poisson_history", [])

        if lh:
            axes[0].plot(lh, color=color, linewidth=1, alpha=0.85, label=opt.upper())
        if ph:
            axes[1].plot(ph, color=color, linewidth=1, alpha=0.85, label=opt.upper())

    axes[0].set_title("Total Loss During Fine-Tuning", fontsize=10)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].set_title("Poisson Auxiliary Loss  L_p(epoch)", fontsize=10)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("L_poisson")
    axes[1].set_yscale("log"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    warmup = 500
    for ax in axes:
        ax.axvline(warmup, color="#555", linestyle=":", linewidth=1,
                   label=f"λ_p warm-up end ({warmup})")

    plt.suptitle("Level 6 Fine-Tuning Training Curves", fontsize=11)
    plt.tight_layout()
    out = out_dir / "l6_training_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — Summary table
# ──────────────────────────────────────────────────────────────────────────────

def plot_summary_table(l5: dict, out_dir: Path) -> None:
    col_labels = ["Optimizer", "Arch", "Level 1 L2", "Level 5 L2",
                  "Level 6 L2 ↓", "Improv.", "MAE L6 (°C)", "Target"]
    rows = []

    for opt in OPTIMIZERS:
        l5d  = l5.get(opt, {})
        l6d  = _load_l6(opt)
        arch = l5d.get("architecture", {})
        n    = arch.get("n_layers", "?")
        w    = (arch.get("neurons") or ["?"])[0]
        a    = arch.get("activation", "?")

        l2_l1  = (l5d.get("level1_adam_only")    or {}).get("L2_rel")
        # Use level5_before from result.json for consistent L5→L6 comparison
        l2_l5  = (l6d.get("level5_before")       or
                  l5d.get("level5_adam_lbfgs")   or {}).get("L2_rel")
        l2_l6  = (l6d.get("level6_after")         or {}).get("L2_rel")
        mae_l6 = (l6d.get("level6_after")         or {}).get("MAE_C")
        impr   = (l6d.get("improvement")          or {}).get("L2_pct")

        met = "✓" if (l2_l6 is not None and l2_l6 < TARGET_L2) else "✗"

        rows.append([
            opt.upper(),
            f"{n}×{w} {a}",
            f"{l2_l1:.4f}" if l2_l1 else "—",
            f"{l2_l5:.4f}" if l2_l5 else "—",
            f"{l2_l6:.4f}" if l2_l6 else "not run",
            f"{impr:+.1f}%" if impr is not None else "—",
            f"{mae_l6:.1f}" if mae_l6 else "—",
            met,
        ])

    fig, ax = plt.subplots(figsize=(14, 0.55 * (len(rows) + 2) + 1.5))
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.55)
    tbl.auto_set_column_width(range(len(col_labels)))

    # Header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1565C0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Row coloring: green if hits target
    for i, row in enumerate(rows, 1):
        hit = row[-1] == "✓"
        bg  = "#C8E6C9" if hit else "#FFECB3"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)
        if hit:
            tbl[i, len(col_labels)-1].set_text_props(color="#1B5E20", fontweight="bold")

    # Target row note
    ax.text(0.5, -0.06,
            f"Target: rel_L2 < {TARGET_L2}  |  Green = target met  |  Yellow = above target",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=8.5, color="#555")

    ax.set_title("Level 6 — Poisson-Assisted Quenching Enhancement\n"
                 "Full Accuracy Summary: Level 1 → Level 5 → Level 6",
                 fontsize=11, pad=10)

    plt.tight_layout()
    out = out_dir / "l6_summary_table.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 & 6 — Temperature heatmaps
# ──────────────────────────────────────────────────────────────────────────────

LEVEL1_DIR = _ROOT / "level1_single_shot" / "results"
LEVEL5_DIR = _ROOT / "level5_refinement" / "results"
LEVEL6_DIR = _ROOT / "level6_poisson_benchmark" / "results"

# Domain bounds (must match src/config.py)
_X_MAX = 1.3
_Y_MAX = 0.6

# ── FEM-PINNS style helpers ───────────────────────────────────────────────────
_FIG_BG   = "#f7f1e3"   # figure background
_AX_BG    = "#fffdf8"   # axes background
_SPINE_C  = "#d8cfbf"   # spine colour
_TXT_C    = "#2f2a24"   # primary text
_TICK_C   = "#4c463f"   # tick/label colour
_BOX_FC   = "#fef6e4"   # annotation box face
_BOX_EC   = "#e8c77a"   # annotation box edge


def _style_ax(ax):
    ax.set_facecolor(_AX_BG)
    for spine in ax.spines.values():
        spine.set_color(_SPINE_C)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=_TICK_C, labelsize=8)
    ax.xaxis.label.set_color(_TICK_C)
    ax.yaxis.label.set_color(_TICK_C)


def _annotate_box(ax, text, loc="top"):
    y = 0.96 if loc == "top" else 0.04
    va = "top"  if loc == "top" else "bottom"
    ax.text(0.03, y, text, transform=ax.transAxes, va=va, fontsize=9,
            color=_TXT_C,
            bbox={"facecolor": _BOX_FC, "edgecolor": _BOX_EC,
                  "boxstyle": "round,pad=0.25"})


def _load_model_for_plot(opt: str, level: int):
    """
    Load a trained PINNNet for visualization.
    level: 5 → results/level5_refinement/{opt}/model_lbfgs.pt
           6 → results/level6/{opt}/model_l6.pt
    Returns model or None if weights not found.
    """
    try:
        import json as _json
        from src.pinn_network import PINNNet
        from src.config import DEVICE

        arch_path = LEVEL1_DIR / opt / "best_arch.json"
        if not arch_path.exists():
            return None
        cfg = _json.load(open(arch_path))
        cfg = cfg.get("config") or cfg.get("best_config") or cfg

        weights = (LEVEL5_DIR / opt / "model_lbfgs.pt") if level == 5 \
                  else (LEVEL6_DIR / opt / "model_l6.pt")
        if not weights.exists():
            return None

        model = PINNNet(
            n_input      = cfg.get("n_input",  3),
            n_output     = cfg.get("n_output", 1),
            hidden_sizes = cfg["neurons"],
            activation   = cfg["activation"],
            residual     = cfg.get("residual", False),
        ).to(DEVICE)
        model.load_state_dict(torch.load(weights, map_location=DEVICE,
                                          weights_only=True))
        model.eval()
        return model
    except Exception as e:
        print(f"  [model load error] {opt} L{level}: {e}")
        return None


def _eval_temperature_grid(model, t_val: float,
                            nx: int = 80, ny: int = 40) -> np.ndarray:
    """
    Evaluate T(x,y) on a uniform grid at time t=t_val.
    Returns [nx, ny] numpy array in °C. Input order: (t, x, y).
    """
    from src.config import DEVICE
    xs = torch.linspace(0.0, _X_MAX, nx, device=DEVICE)
    ys = torch.linspace(0.0, _Y_MAX, ny, device=DEVICE)
    xg, yg = torch.meshgrid(xs, ys, indexing="ij")
    t_grid = torch.full_like(xg, t_val)

    coords = torch.stack([t_grid.reshape(-1),
                          xg.reshape(-1),
                          yg.reshape(-1)], dim=1)
    with torch.no_grad():
        T = model(coords).squeeze(-1).cpu().numpy()
    return T.reshape(nx, ny)


def plot_temperature_heatmaps(l5_data: dict, out_dir: Path) -> None:
    """
    Figure 5 — T(x,y) at t=5s for all Level 6 (+Poisson) models.
    One column per optimizer. FEM-PINNS style.
    """
    T_EVAL = 5.0
    opts   = [o for o in OPTIMIZERS if _load_model_for_plot(o, 6) is not None]
    if not opts:
        print("  [heatmap] No Level 6 models found.")
        return

    # Load all grids first to get shared colour range
    models, grids = {}, {}
    for opt in opts:
        m = _load_model_for_plot(opt, 6)
        if m:
            models[opt] = m
            grids[opt]  = _eval_temperature_grid(m, T_EVAL)

    all_vals = np.concatenate([g.flatten() for g in grids.values()])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    n = len(opts)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n + 1.2, 3.8),
                              facecolor=_FIG_BG,
                              constrained_layout=True)

    im_last = None
    for ax, opt in zip(np.atleast_1d(axes), opts):
        l6d  = _load_l6(opt)
        l2   = (l6d.get("level6_after") or {}).get("L2_rel", 0)
        T    = grids[opt]

        im_last = ax.imshow(
            T.T, origin="lower", aspect="auto",
            extent=[0, _X_MAX, 0, _Y_MAX],
            cmap="YlOrRd", vmin=vmin, vmax=vmax,
        )
        _style_ax(ax)
        ax.set_title(opt.upper(), fontsize=12, color=_TXT_C, pad=6)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]" if opt == opts[0] else "")
        _annotate_box(ax, f"relL2  {l2:.2e}")

    cbar = fig.colorbar(im_last, ax=list(np.atleast_1d(axes)), shrink=0.85, pad=0.02)
    cbar.set_label("T [°C]", color=_TXT_C)
    cbar.ax.tick_params(labelsize=8, colors=_TICK_C)

    fig.suptitle(
        f"Temperature Field T(x,y) at t={T_EVAL:.0f}s\n"
        f"Level 6 — Poisson-Assisted Fine-Tuning",
        fontsize=13, color=_TXT_C,
    )
    out = out_dir / "l6_heatmap_t5s.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")


def plot_timeslice_heatmaps(out_dir: Path) -> None:
    """
    Figure 6 — T(x,y) evolution over time for the best Level 6 model.
    t = 1, 5, 15, 30 s in one row. FEM-PINNS style.
    """
    T_SLICES = [1.0, 5.0, 15.0, 30.0]

    best_opt = next((o for o in OPTIMIZERS if _load_model_for_plot(o, 6)), None)
    if best_opt is None:
        print("  [timeslice] No Level 6 model found.")
        return

    model  = _load_model_for_plot(best_opt, 6)
    grids  = [_eval_temperature_grid(model, t) for t in T_SLICES]
    vmin   = min(g.min() for g in grids)
    vmax   = max(g.max() for g in grids)

    n_slices = len(T_SLICES)
    fig, axes = plt.subplots(1, n_slices,
                              figsize=(4.8 * n_slices + 1.2, 3.8),
                              facecolor=_FIG_BG,
                              constrained_layout=True)

    im_last = None
    for ax, T_grid, t_val in zip(axes, grids, T_SLICES):
        im_last = ax.imshow(
            T_grid.T, origin="lower", aspect="auto",
            extent=[0, _X_MAX, 0, _Y_MAX],
            cmap="YlOrRd", vmin=vmin, vmax=vmax,
        )
        _style_ax(ax)
        ax.set_title(f"t = {t_val:.0f} s", fontsize=12, color=_TXT_C, pad=6)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]" if t_val == T_SLICES[0] else "")

    cbar = fig.colorbar(im_last, ax=list(axes), shrink=0.85, pad=0.02)
    cbar.set_label("T [°C]", color=_TXT_C)
    cbar.ax.tick_params(labelsize=8, colors=_TICK_C)

    l6d = _load_l6(best_opt)
    l2  = (l6d.get("level6_after") or {}).get("L2_rel", 0)
    fig.suptitle(
        f"Quenching Temperature Evolution — Level 6 [{best_opt.upper()}]\n"
        f"A356 aluminium  |  relL2 = {l2:.2e}  |  domain {_X_MAX} m × {_Y_MAX} m",
        fontsize=13, color=_TXT_C,
    )
    out = out_dir / "l6_heatmap_timeslices.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out}")




def main():
    out_dir = L6_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Level 6 — Plot Results")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    l5 = _load_l5()
    if not l5:
        print("  [WARNING] Level 5 summary not found. Run Level 5 first.")
        return

    l6_found = [o for o in OPTIMIZERS if _load_l6(o)]
    if not l6_found:
        print("  [WARNING] No Level 6 results. Run main_level6.py first.")
        return
    print(f"  Level 5 loaded: {list(l5.keys())}")
    print(f"  Level 6 loaded: {l6_found}\n")

    plot_l2_progression(l5, out_dir)
    plot_improvement(l5, out_dir)
    plot_training_curves(out_dir)
    plot_summary_table(l5, out_dir)
    plot_temperature_heatmaps(l5, out_dir)
    plot_timeslice_heatmaps(out_dir)

    print(f"\n  All figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
