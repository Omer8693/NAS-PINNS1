"""
thermal_pinn/plots/plot_adaptive_k_triplet_3d.py
=================================================
Three-way 3D comparison for:
    1. fixed k=1
    2. adaptive k v1
    3. adaptive k v2

Why this script exists
----------------------
`plot_adaptive_k_fields.py` is built around a single adaptive run and k-change
events. For fair qualitative comparison between controllers, it is often easier
to compare all methods at the same physical times.

This script does exactly that for 3D domains:
    - common snapshot times across all methods
    - one FEM reference row
    - three prediction rows (fixed k=1 / adaptive v1 / adaptive v2)
    - summary strip at the bottom with global and final-window metrics

Usage
-----
    python -m thermal_pinn.plots.plot_adaptive_k_triplet_3d \
        --domain cylinder --arch bayesian --v2-tag adaptive_v2_all3d_rerun

    python -m thermal_pinn.plots.plot_adaptive_k_triplet_3d \
        --all --v2-tag adaptive_v2_all3d_rerun
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

from thermal_pinn.network.pinn import ARCH_CONFIGS, ThermalPINN
from thermal_pinn.physics.domains_3d import DOMAINS_3D, make_domain_3d
from thermal_pinn.plots.plot_results import (
    DEVICE,
    DELTA_T,
    T_INIT,
    T_WATER,
    _clean_T,
    _pinn_forward_full,
    _render_box_3d,
    _render_cylinder_3d,
    _render_lshape_3d,
    _style_3d_ax,
    _surface_meshes_3d,
    _valid_surface_mask,
)

_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = _ROOT / "checkpoints"
RESULT_DIR = _ROOT / "results" / "adaptive_k_triplet_3d"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TIMES = [1.5, 4.5, 10.5, 18.0, 30.0]

METHOD_ORDER = ["fixed", "adaptive_v1", "adaptive_v2"]
METHOD_LABELS = {
    "fixed": "Fixed k=1",
    "adaptive_v1": "Adaptive v1",
    "adaptive_v2": "Adaptive v2",
}
METHOD_COLORS = {
    "fixed": "#1565C0",
    "adaptive_v1": "#E65100",
    "adaptive_v2": "#1B5E20",
}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_model(ckpt_path: Path, arch: str, dim: int = 3) -> ThermalPINN:
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = ThermalPINN(dim=dim, arch=arch).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _pick_render_fn(domain):
    if hasattr(domain, "R") and hasattr(domain, "Hz"):
        return _render_cylinder_3d
    if hasattr(domain, "cut_x") and hasattr(domain, "cut_y"):
        return _render_lshape_3d
    return _render_box_3d


def _set_3d_limits(ax, domain):
    if hasattr(domain, "R") and hasattr(domain, "Hz"):
        R = domain.R
        Hz = domain.Hz
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_zlim(0, Hz)
        ax.set_box_aspect([2 * R, 2 * R, Hz])
        return

    Lx = getattr(domain, "Lx", 1.0)
    Ly = getattr(domain, "Ly", 1.0)
    Lz = getattr(domain, "Lz", getattr(domain, "Hz", 1.0))
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    dims = np.array([Lx, Ly, Lz], dtype=float)
    ax.set_box_aspect(dims / dims.max())


def _domain_label(name: str) -> str:
    labels = {
        "rectangular": "Rectangular 3D",
        "cylinder": "Cylinder 3D",
        "stacked": "Stacked 3D",
        "lshape": "L-Shape 3D",
    }
    return labels.get(name, name)


def _window_k(metrics: dict, window: dict) -> int:
    if "k_used" in window:
        return int(window["k_used"])
    if "k" in metrics:
        return int(metrics["k"])
    dt = float(window["t_end"]) - float(window["t_start"])
    return max(1, int(round(dt / 1.5)))


def _find_window(metrics: dict, t_query: float, tol: float = 1e-8) -> dict:
    windows = metrics["windows"]

    exact_end = [w for w in windows if abs(float(w["t_end"]) - t_query) < tol]
    if exact_end:
        return exact_end[0]

    containing = [
        w for w in windows
        if float(w["t_start"]) - tol <= t_query <= float(w["t_end"]) + tol
    ]
    if containing:
        return containing[0]

    return windows[-1]


def _predict_for_coords(
    model: ThermalPINN,
    domain,
    metrics: dict,
    coords_np: np.ndarray,
    t_query: float,
) -> tuple[np.ndarray, dict]:
    window = _find_window(metrics, t_query)
    t_start = float(window["t_start"])
    t_end = float(window["t_end"])
    dt = max(t_end - t_start, 1e-9)
    tau_val = float(np.clip((t_query - t_start) / dt, 0.0, 1.0))

    x = coords_np[:, 0]
    y = coords_np[:, 1]
    z = coords_np[:, 2]
    valid = _valid_surface_mask(domain, coords_np)

    T_ic = _clean_T(domain.T(x, y, z, t_start), domain, x, y, z)
    T_ic = np.where(valid & np.isfinite(T_ic), T_ic, T_WATER)

    T_pred = _pinn_forward_full(model, coords_np.astype(np.float32), tau_val, T_ic, domain)
    T_pred = _clean_T(T_pred, domain, x, y, z)
    T_pred = np.where(valid, T_pred, np.nan)
    return T_pred, window


def _make_metrics_field_fn(model: ThermalPINN, domain, metrics: dict):
    def field_fn(X, Y, Z, t):
        x = np.asarray(X, dtype=np.float32).ravel()
        y = np.asarray(Y, dtype=np.float32).ravel()
        z = np.asarray(Z, dtype=np.float32).ravel()
        coords = np.stack([x, y, z], axis=1)
        T_pred, _ = _predict_for_coords(model, domain, metrics, coords, float(t))
        return T_pred.reshape(np.asarray(X).shape)
    return field_fn


def _surface_snapshot_stats(model: ThermalPINN, domain, metrics: dict, t_query: float, n: int = 16) -> dict:
    abs_errs = []
    diff_all = []
    ref_all = []
    window = _find_window(metrics, t_query)

    for X, Y, Z in _surface_meshes_3d(domain, n=n):
        x = X.ravel()
        y = Y.ravel()
        z = Z.ravel()
        coords = np.stack([x, y, z], axis=1).astype(np.float32)

        T_ref = _clean_T(domain.T(x, y, z, t_query), domain, x, y, z)
        T_pred, _ = _predict_for_coords(model, domain, metrics, coords, t_query)

        valid = np.isfinite(T_ref) & np.isfinite(T_pred)
        if np.any(valid):
            diff = T_pred[valid] - T_ref[valid]
            abs_errs.append(np.abs(diff))
            diff_all.append(diff)
            ref_all.append(T_ref[valid] - T_WATER)

    if not abs_errs:
        return {
            "mae": float("nan"),
            "l2": float("nan"),
            "k_used": _window_k(metrics, window),
            "t_start": float(window["t_start"]),
            "t_end": float(window["t_end"]),
        }

    diff_cat = np.concatenate(diff_all)
    ref_cat = np.concatenate(ref_all)
    return {
        "mae": float(np.mean(np.concatenate(abs_errs))),
        "l2": float(np.linalg.norm(diff_cat) / (np.linalg.norm(ref_cat) + 1e-10)),
        "k_used": _window_k(metrics, window),
        "t_start": float(window["t_start"]),
        "t_end": float(window["t_end"]),
    }


def _format_schedule(metrics: dict) -> str:
    schedule = metrics.get("k_schedule")
    if schedule:
        if len(schedule) <= 10:
            return "-".join(str(v) for v in schedule)
        return "-".join(str(v) for v in schedule[:5]) + "-...-" + "-".join(str(v) for v in schedule[-3:])

    k = metrics.get("k")
    if k is not None:
        return f"{k} constant"
    return "n/a"


def _summary_text(label: str, metrics: dict) -> str:
    final_window_mae = float(metrics["windows"][-1]["mae_C"]) if metrics.get("windows") else float("nan")
    fem_savings = float(metrics.get("fem_savings_pct", 0.0))
    return (
        f"{label}\n"
        f"mean MAE: {metrics['mean_mae']:.2f}C\n"
        f"final window MAE: {final_window_mae:.2f}C\n"
        f"windows: {metrics['n_windows']}\n"
        f"FEM saved: {fem_savings:.0f}%\n"
        f"k schedule: {_format_schedule(metrics)}"
    )


def _discover_v2_metrics(domain: str, arch: str, dim: int, tag: str | None) -> Path | None:
    if tag:
        exact = CKPT_DIR / f"{domain}_{arch}_{tag}_dim{dim}_metrics.json"
        return exact if exact.exists() else None

    candidates = sorted(
        CKPT_DIR.glob(f"{domain}_{arch}_adaptive_v2*_dim{dim}_metrics.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_triplet(domain: str, arch: str, dim: int, fixed_k: int, v2_tag: str | None):
    fixed_metrics_path = CKPT_DIR / f"{domain}_{arch}_k{fixed_k}_dim{dim}_metrics.json"
    fixed_ckpt_path = CKPT_DIR / f"{domain}_{arch}_k{fixed_k}_dim{dim}.pt"
    adaptive_metrics_path = CKPT_DIR / f"{domain}_{arch}_adaptive_dim{dim}_metrics.json"
    adaptive_ckpt_path = CKPT_DIR / f"{domain}_{arch}_adaptive_dim{dim}.pt"
    v2_metrics_path = _discover_v2_metrics(domain, arch, dim, v2_tag)

    if not (fixed_metrics_path.exists() and fixed_ckpt_path.exists()):
        raise FileNotFoundError(f"Missing fixed-k files for {domain}/{arch}/dim{dim}")
    if not (adaptive_metrics_path.exists() and adaptive_ckpt_path.exists()):
        raise FileNotFoundError(f"Missing adaptive v1 files for {domain}/{arch}/dim{dim}")
    if v2_metrics_path is None:
        raise FileNotFoundError(f"Missing adaptive v2 metrics for {domain}/{arch}/dim{dim}")

    v2_ckpt_path = v2_metrics_path.with_name(v2_metrics_path.name.replace("_metrics.json", ".pt"))
    if not v2_ckpt_path.exists():
        raise FileNotFoundError(f"Missing adaptive v2 checkpoint: {v2_ckpt_path}")

    bundles = {
        "fixed": {
            "metrics": _load_json(fixed_metrics_path),
            "model": _load_model(fixed_ckpt_path, arch, dim),
            "ckpt": fixed_ckpt_path,
            "metrics_path": fixed_metrics_path,
        },
        "adaptive_v1": {
            "metrics": _load_json(adaptive_metrics_path),
            "model": _load_model(adaptive_ckpt_path, arch, dim),
            "ckpt": adaptive_ckpt_path,
            "metrics_path": adaptive_metrics_path,
        },
        "adaptive_v2": {
            "metrics": _load_json(v2_metrics_path),
            "model": _load_model(v2_ckpt_path, arch, dim),
            "ckpt": v2_ckpt_path,
            "metrics_path": v2_metrics_path,
        },
    }
    return bundles


def plot_triplet(domain_name: str, arch: str, fixed_k: int = 1,
                 times: list[float] | None = None, v2_tag: str | None = None,
                 save: bool = True) -> Path:
    domain = make_domain_3d(domain_name)
    bundles = _load_triplet(domain_name, arch, 3, fixed_k, v2_tag)
    times = times or list(DEFAULT_TIMES)

    render_fn = _pick_render_fn(domain)
    norm = Normalize(vmin=T_WATER, vmax=T_INIT)
    cmap = plt.get_cmap("rainbow")

    summary = {}
    for key in METHOD_ORDER:
        summary[key] = [_surface_snapshot_stats(bundles[key]["model"], domain, bundles[key]["metrics"], t) for t in times]

    fig = plt.figure(figsize=(3.4 * len(times), 12.2), constrained_layout=True)
    gs = fig.add_gridspec(5, len(times), height_ratios=[1.0, 1.0, 1.0, 1.0, 0.55])
    fig.patch.set_facecolor("white")

    row_labels = {
        0: ("Reference (FEM)", "#1A237E"),
        1: (METHOD_LABELS["fixed"], METHOD_COLORS["fixed"]),
        2: (METHOD_LABELS["adaptive_v1"], METHOD_COLORS["adaptive_v1"]),
        3: (METHOD_LABELS["adaptive_v2"], METHOD_COLORS["adaptive_v2"]),
    }

    for col, t_query in enumerate(times):
        ax_ref = fig.add_subplot(gs[0, col], projection="3d")
        render_fn(ax_ref, domain, t_query, cmap, norm, field_fn=None)
        _style_3d_ax(ax_ref, domain)
        _set_3d_limits(ax_ref, domain)
        ax_ref.view_init(elev=25, azim=-60)
        ax_ref.set_title(f"t = {t_query:.1f} s", fontsize=9, fontweight="bold", color="#1A237E", pad=5)
        if col == 0:
            ax_ref.text2D(-0.18, 0.5, row_labels[0][0], transform=ax_ref.transAxes,
                          fontsize=9, fontweight="bold", va="center", rotation=90,
                          color=row_labels[0][1])

        for row_idx, method_key in enumerate(METHOD_ORDER, start=1):
            bundle = bundles[method_key]
            field_fn = _make_metrics_field_fn(bundle["model"], domain, bundle["metrics"])
            stats = summary[method_key][col]

            ax = fig.add_subplot(gs[row_idx, col], projection="3d")
            render_fn(ax, domain, t_query, cmap, norm, field_fn=field_fn)
            _style_3d_ax(ax, domain)
            _set_3d_limits(ax, domain)
            ax.view_init(elev=25, azim=-60)

            txt = (
                f"MAE={stats['mae']:.1f}C  "
                f"k={stats['k_used']}  "
                f"[{stats['t_start']:.1f}->{stats['t_end']:.1f}s]"
            )
            ax.text2D(
                0.5, -0.05, txt,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8.2, color=METHOD_COLORS[method_key], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          edgecolor=METHOD_COLORS[method_key], alpha=0.95, linewidth=1.2),
            )
            if col == 0:
                ax.text2D(-0.18, 0.5, row_labels[row_idx][0], transform=ax.transAxes,
                          fontsize=9, fontweight="bold", va="center", rotation=90,
                          color=row_labels[row_idx][1])

    ax_summary = fig.add_subplot(gs[4, :])
    ax_summary.axis("off")
    x_positions = [0.17, 0.50, 0.83]
    for x, method_key in zip(x_positions, METHOD_ORDER):
        box = _summary_text(METHOD_LABELS[method_key], bundles[method_key]["metrics"])
        ax_summary.text(
            x, 0.5, box,
            transform=ax_summary.transAxes,
            ha="center", va="center",
            fontsize=8.3, color="#24303F", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=METHOD_COLORS[method_key], linewidth=1.5, alpha=0.97),
        )

    chosen_v2 = bundles["adaptive_v2"]["metrics"].get("tag", "adaptive_v2")
    fig.suptitle(
        f"3D Snapshot Comparison  —  {_domain_label(domain_name)} / {arch}\n"
        f"Reference vs Fixed k={fixed_k} vs Adaptive v1 vs Adaptive v2 ({chosen_v2})\n"
        f"Common snapshot times: {', '.join(f'{t:.1f}s' for t in times)}"
        f"  |  early snapshots use the final trained checkpoint",
        fontsize=11.5, fontweight="bold", color="#1A237E",
    )

    safe_tag = "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in chosen_v2)
    out_path = RESULT_DIR / f"{domain_name}_{arch}_triplet_3d_{safe_tag}.png"
    if save:
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)
    return out_path


def _available_triplets(v2_tag: str | None) -> list[tuple[str, str]]:
    combos = []
    for domain in DOMAINS_3D:
        for arch in ARCH_CONFIGS:
            try:
                _load_triplet(domain, arch, 3, fixed_k=1, v2_tag=v2_tag)
            except FileNotFoundError:
                continue
            combos.append((domain, arch))
    return combos


def main():
    parser = argparse.ArgumentParser(
        description="Three-way 3D comparison: fixed k=1 vs adaptive v1 vs adaptive v2."
    )
    parser.add_argument("--domain", default="cylinder", choices=list(DOMAINS_3D))
    parser.add_argument("--arch", default="bayesian", choices=list(ARCH_CONFIGS))
    parser.add_argument("--fixed-k", type=int, default=1)
    parser.add_argument("--times", nargs="+", type=float, default=DEFAULT_TIMES)
    parser.add_argument("--v2-tag", default=None,
                        help="Adaptive v2 tag, e.g. adaptive_v2_all3d_rerun. If omitted, newest adaptive_v2* file is used.")
    parser.add_argument("--all", action="store_true", help="Render all available 3D triplets.")
    args = parser.parse_args()

    if args.all:
        combos = _available_triplets(args.v2_tag)
        print(f"Found {len(combos)} available 3D triplet(s).")
        for domain, arch in combos:
            plot_triplet(domain, arch, fixed_k=args.fixed_k, times=args.times, v2_tag=args.v2_tag)
        return

    plot_triplet(args.domain, args.arch, fixed_k=args.fixed_k, times=args.times, v2_tag=args.v2_tag)


if __name__ == "__main__":
    main()
