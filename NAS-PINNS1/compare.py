#!/usr/bin/env python3
"""Exact-vs-pred plots and L2 comparison outputs."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from core import ensure_dir, infer_checkpoint, read_run_meta


def finite_range(vals: np.ndarray) -> Tuple[float, float]:
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi <= lo:
        eps = 1e-12 if lo == 0 else abs(lo) * 1e-6
        return lo - eps, hi + eps
    return lo, hi


def write_matrix_csv(
    matrix: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    x_name: str,
    y_name: str,
    value_name: str,
    out_csv: Path,
) -> None:
    rows: List[Dict] = []
    for i, y_val in enumerate(y_axis):
        for j, x_val in enumerate(x_axis):
            rows.append({y_name: float(y_val), x_name: float(x_val), value_name: float(matrix[i, j])})
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def _apply_axis_ticks(ax, x_ticks: np.ndarray, y_ticks: np.ndarray, x_fmt: str, y_fmt: str) -> None:
    if len(x_ticks) == 1:
        ax.set_xticks([0])
        ax.set_xticklabels([format(x_ticks[0], x_fmt)])
    else:
        x_idx = np.linspace(0, len(x_ticks) - 1, min(8, len(x_ticks))).astype(int)
        ax.set_xticks(x_idx)
        ax.set_xticklabels([format(x_ticks[i], x_fmt) for i in x_idx], rotation=20)

    if len(y_ticks) == 1:
        ax.set_yticks([0])
        ax.set_yticklabels([format(y_ticks[0], y_fmt)])
    else:
        y_idx = np.arange(len(y_ticks))
        ax.set_yticks(y_idx)
        ax.set_yticklabels([format(y_ticks[i], y_fmt) for i in y_idx])


def save_side_by_side_heatmaps(
    exact_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    x_ticks: np.ndarray,
    y_ticks: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    out_png: Path,
    cmap: str,
    vmin: float,
    vmax: float,
    cbar_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    axes[0].imshow(exact_matrix, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Exact (Paper)")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    _apply_axis_ticks(axes[0], x_ticks, y_ticks, ".2f", ".3f")

    im1 = axes[1].imshow(pred_matrix, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    axes[1].set_xlabel(x_label)
    _apply_axis_ticks(axes[1], x_ticks, y_ticks, ".2f", ".3f")

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist())
    cbar.set_label(cbar_label)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_temperature_profile_plot(
    t_vals: np.ndarray,
    y_vals: np.ndarray,
    exact_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    out_png: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for i, y in enumerate(y_vals):
        ax.plot(t_vals, exact_matrix[i, :], "--", linewidth=2.0, label=f"Exact y={y:.3f}")
        ax.plot(t_vals, pred_matrix[i, :], "-", linewidth=1.8, label=f"Pred y={y:.3f}")
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("temperature")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_displacement_profile_plot(
    y_layers: np.ndarray,
    exact_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    out_png: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.plot(y_layers, exact_matrix[:, 0], "--o", linewidth=2.0, label="Exact (paper)")
    ax.plot(y_layers, pred_matrix[:, 0], "-s", linewidth=1.8, label="Prediction")
    ax.set_title(title)
    ax.set_xlabel("y layer")
    ax.set_ylabel("uy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def generate_exact_pred_and_l2(
    baseline_mod,
    method_to_run_dir: Dict[str, Path],
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    ref = baseline_mod.build_quench_reference_data()
    rows: List[Dict] = []

    temp_all: List[np.ndarray] = []
    disp_all: List[np.ndarray] = []
    cached = {}

    for method, run_dir in method_to_run_dir.items():
        meta = read_run_meta(run_dir)
        layers = int(meta["layers"])
        neurons = int(meta["base_neurons"])
        best_stage = str(meta.get("best_stage", "")).strip().lower() or None
        ckpt = infer_checkpoint(run_dir, best_stage)

        model = baseline_mod.NAS_PINN(layers=layers, base_neurons=neurons).to(baseline_mod.device)
        state = torch.load(str(ckpt), map_location=baseline_mod.device)
        model.load_state_dict(state)
        model.eval()

        with torch.no_grad():
            t_pred = model(torch.cat([ref["x_temp"], ref["y_temp"], ref["t_temp"]], dim=1))[:, 0:1]
            u_pred = model(torch.cat([ref["x_disp"], ref["y_disp"], ref["t_disp"]], dim=1))[:, 2:3]

        y_np = ref["y_temp"].detach().cpu().numpy().reshape(-1)
        t_np = ref["t_temp"].detach().cpu().numpy().reshape(-1)
        t_ref = ref["target_temp"].detach().cpu().numpy().reshape(-1)
        t_pred_np = t_pred.detach().cpu().numpy().reshape(-1)
        uniq_y = np.unique(y_np)
        uniq_t = np.unique(t_np)
        n_y, n_t = len(uniq_y), len(uniq_t)
        t_ref_m = t_ref.reshape(n_y, n_t)
        t_pred_m = t_pred_np.reshape(n_y, n_t)

        y_layers = ref["y_disp"].detach().cpu().numpy().reshape(-1)
        x_axis = np.array([float(ref["x_disp"][0].item())], dtype=np.float32)
        u_ref = ref["target_uy"].detach().cpu().numpy().reshape(-1, 1)
        u_pred_m = u_pred.detach().cpu().numpy().reshape(-1, 1)

        temp_all.extend([t_ref_m.reshape(-1), t_pred_m.reshape(-1)])
        disp_all.extend([u_ref.reshape(-1), u_pred_m.reshape(-1)])

        cached[method] = {
            "run_dir": run_dir,
            "layers": layers,
            "neurons": neurons,
            "best_stage": best_stage,
            "best_objective": float(meta.get("best_objective", np.nan)),
            "checkpoint": ckpt,
            "t_vals": uniq_t,
            "y_vals": uniq_y,
            "t_ref_m": t_ref_m,
            "t_pred_m": t_pred_m,
            "x_axis": x_axis,
            "y_layers": y_layers,
            "u_ref": u_ref,
            "u_pred_m": u_pred_m,
        }

    temp_vmin, temp_vmax = finite_range(np.concatenate(temp_all))
    disp_vmin, disp_vmax = finite_range(np.concatenate(disp_all))

    for method, out in cached.items():
        method_dir = out_dir / method
        ensure_dir(method_dir)
        t_vals = out["t_vals"]
        y_vals = out["y_vals"]
        t_ref_m = out["t_ref_m"]
        t_pred_m = out["t_pred_m"]
        x_axis = out["x_axis"]
        y_layers = out["y_layers"]
        u_ref = out["u_ref"]
        u_pred_m = out["u_pred_m"]

        write_matrix_csv(t_ref_m, t_vals, y_vals, "time_s", "y_coord", "temp_exact", method_dir / "temp_exact_grid.csv")
        write_matrix_csv(t_pred_m, t_vals, y_vals, "time_s", "y_coord", "temp_pred", method_dir / "temp_pred_grid.csv")
        save_side_by_side_heatmaps(
            exact_matrix=t_ref_m,
            pred_matrix=t_pred_m,
            x_ticks=t_vals,
            y_ticks=y_vals,
            x_label="time (s)",
            y_label="y (model coord)",
            title=f"{method} | Temperature Exact vs Prediction",
            out_png=method_dir / "temp_exact_vs_pred_heatmap.png",
            cmap="magma",
            vmin=temp_vmin,
            vmax=temp_vmax,
            cbar_label="temperature (degC)",
        )
        save_temperature_profile_plot(
            t_vals=t_vals,
            y_vals=y_vals,
            exact_matrix=t_ref_m,
            pred_matrix=t_pred_m,
            out_png=method_dir / "temp_exact_vs_pred_curves.png",
            title=f"{method} | Temperature Curves (Exact vs Pred)",
        )

        write_matrix_csv(u_ref, x_axis, y_layers, "x_coord", "y_layer", "uy_exact", method_dir / "disp_exact_grid.csv")
        write_matrix_csv(u_pred_m, x_axis, y_layers, "x_coord", "y_layer", "uy_pred", method_dir / "disp_pred_grid.csv")
        save_side_by_side_heatmaps(
            exact_matrix=u_ref,
            pred_matrix=u_pred_m,
            x_ticks=x_axis,
            y_ticks=y_layers,
            x_label="x (reference uses x=0 only)",
            y_label="y layer",
            title=f"{method} | Displacement Exact vs Prediction",
            out_png=method_dir / "disp_exact_vs_pred_heatmap.png",
            cmap="viridis",
            vmin=disp_vmin,
            vmax=disp_vmax,
            cbar_label="displacement (m)",
        )
        save_displacement_profile_plot(
            y_layers=y_layers,
            exact_matrix=u_ref,
            pred_matrix=u_pred_m,
            out_png=method_dir / "disp_exact_vs_pred_curve.png",
            title=f"{method} | Displacement Curve (Exact vs Pred)",
        )

        t_num = np.linalg.norm((t_pred_m - t_ref_m).reshape(-1))
        t_den = np.linalg.norm(t_ref_m.reshape(-1)) + 1e-12
        u_num = np.linalg.norm((u_pred_m - u_ref).reshape(-1))
        u_den = np.linalg.norm(u_ref.reshape(-1)) + 1e-12
        temp_rel_l2 = float(t_num / t_den)
        disp_rel_l2 = float(u_num / u_den)

        rows.append(
            {
                "method": method,
                "run_dir": str(Path(out["run_dir"]).resolve()),
                "layers": int(out["layers"]),
                "neurons": int(out["neurons"]),
                "checkpoint": str(Path(out["checkpoint"]).resolve()),
                "best_stage": out["best_stage"],
                "best_objective": float(out["best_objective"]),
                "temp_rel_l2": temp_rel_l2,
                "disp_rel_l2": disp_rel_l2,
            }
        )

    l2_df = pd.DataFrame(rows).sort_values("temp_rel_l2", ascending=True).reset_index(drop=True)
    l2_df.to_csv(out_dir / "l2_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    x = np.arange(len(l2_df))
    ax.bar(x - 0.18, l2_df["temp_rel_l2"], width=0.36, label="temp_rel_l2")
    ax.bar(x + 0.18, l2_df["disp_rel_l2"], width=0.36, label="disp_rel_l2")
    ax.set_xticks(x)
    ax.set_xticklabels(l2_df["method"].tolist(), rotation=15)
    ax.set_yscale("log")
    ax.set_ylabel("relative L2 (log)")
    ax.set_title("L2 Error Comparison (Exact vs Pred)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "l2_error_comparison.png", dpi=180)
    plt.close(fig)
