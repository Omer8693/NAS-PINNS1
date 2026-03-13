#!/usr/bin/env python3
"""Generate fair physical error metrics and heatmaps for quench2026."""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import torch


def load_baseline_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("quench2026_baseline_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_checkpoint(run_dir: Path, best_stage: Optional[str]) -> Path:
    # Prefer explicit best-stage checkpoint when available.
    if best_stage:
        stage_ckpt = run_dir / f"baseline_model_{best_stage}.pth"
        if stage_ckpt.exists():
            return stage_ckpt
    if (run_dir / "baseline_model.pth").exists():
        return run_dir / "baseline_model.pth"
    for name in ("baseline_model_lbfgs.pth", "baseline_model_adam.pth", "baseline_model_pso.pth"):
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def scenario_dirs(results_root: Path) -> Dict[str, Path]:
    return {
        "baseline_lbfgs5000": results_root / "baseline_adam_refine5000" / "lbfgs" / "L5_N96",
        "nsga2_lbfgs5000": results_root / "best_adam_lbfgs5000" / "lbfgs" / "nsga2" / "L6_N132",
        "nsga3_lbfgs5000": results_root / "best_adam_lbfgs5000" / "lbfgs" / "nsga3" / "L6_N141",
        "bayesian_lbfgs5000": results_root / "best_adam_lbfgs5000" / "lbfgs" / "bayesian" / "L5_N121",
        "baseline_original": results_root / "baseline" / "L5_N96",
    }


def build_temperature_heatmap_arrays(
    model,
    ref: Dict[str, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Fair set: paper reference temperature points only (depth-time grid).
    x = ref["x_temp"]
    y = ref["y_temp"]
    t = ref["t_temp"]
    target = ref["target_temp"]
    with torch.no_grad():
        pred = model(torch.cat([x, y, t], dim=1))[:, 0:1]

    y_np = y.detach().cpu().numpy().reshape(-1)
    t_np = t.detach().cpu().numpy().reshape(-1)
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    tgt_np = target.detach().cpu().numpy().reshape(-1)
    err_np = np.abs(pred_np - tgt_np)

    uniq_y = np.unique(y_np)
    uniq_t = np.unique(t_np)
    n_y = len(uniq_y)
    n_t = len(uniq_t)

    # build_quench_reference_data iterates depth outer, time inner.
    pred_m = pred_np.reshape(n_y, n_t)
    tgt_m = tgt_np.reshape(n_y, n_t)
    err_m = err_np.reshape(n_y, n_t)
    return uniq_y, uniq_t, pred_m, tgt_m, err_m


def build_displacement_x0_arrays(
    model,
    ref: Dict[str, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Fair set: paper displacement references only (x=0, t=t_max, layer points).
    x_ref = ref["x_disp"]  # all zeros
    y_ref = ref["y_disp"]
    t_ref = ref["t_disp"]
    uy_ref = ref["target_uy"].detach().cpu().numpy().reshape(-1)

    with torch.no_grad():
        pred = model(torch.cat([x_ref, y_ref, t_ref], dim=1))[:, 2:3]  # uy
    uy_pred = pred.detach().cpu().numpy().reshape(-1)

    y_layers = y_ref.detach().cpu().numpy().reshape(-1)
    x_axis = np.array([float(x_ref[0].item())], dtype=np.float32)
    uy_pred_m = uy_pred.reshape(len(y_layers), 1)
    uy_ref_m = uy_ref.reshape(len(y_layers), 1)
    err_m = np.abs(uy_pred_m - uy_ref_m)
    return x_axis, y_layers, uy_pred_m, uy_ref_m, err_m


def save_heatmap(
    matrix: np.ndarray,
    x_ticks: np.ndarray,
    y_ticks: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    out_png: Path,
    cmap: str = "inferno",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    cbar_label: str = "absolute error",
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    norm = None
    if center is not None and vmin is not None and vmax is not None and (vmax > vmin):
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    if norm is not None:
        im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    else:
        im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if len(x_ticks) == 1:
        ax.set_xticks([0])
        ax.set_xticklabels([f"{x_ticks[0]:.2f}"])
    elif len(x_ticks) > 1:
        x_idx = np.linspace(0, len(x_ticks) - 1, min(8, len(x_ticks))).astype(int)
        ax.set_xticks(x_idx)
        ax.set_xticklabels([f"{x_ticks[i]:.2f}" for i in x_idx], rotation=20)

    if len(y_ticks) == 1:
        ax.set_yticks([0])
        ax.set_yticklabels([f"{y_ticks[0]:.3f}"])
    elif len(y_ticks) > 1:
        y_idx = np.arange(len(y_ticks))
        ax.set_yticks(y_idx)
        ax.set_yticklabels([f"{y_ticks[i]:.3f}" for i in y_idx])

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _apply_axis_ticks(ax, x_ticks: np.ndarray, y_ticks: np.ndarray, x_fmt: str, y_fmt: str) -> None:
    if len(x_ticks) == 1:
        ax.set_xticks([0])
        ax.set_xticklabels([format(x_ticks[0], x_fmt)])
    elif len(x_ticks) > 1:
        x_idx = np.linspace(0, len(x_ticks) - 1, min(8, len(x_ticks))).astype(int)
        ax.set_xticks(x_idx)
        ax.set_xticklabels([format(x_ticks[i], x_fmt) for i in x_idx], rotation=20)

    if len(y_ticks) == 1:
        ax.set_yticks([0])
        ax.set_yticklabels([format(y_ticks[0], y_fmt)])
    elif len(y_ticks) > 1:
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
    cmap: str = "inferno",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_label: str = "value",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)

    im0 = axes[0].imshow(exact_matrix, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
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
        ax.plot(t_vals, exact_matrix[i, :], linestyle="--", linewidth=2.0, label=f"Exact y={y:.3f}")
        ax.plot(t_vals, pred_matrix[i, :], linestyle="-", linewidth=1.8, label=f"Pred y={y:.3f}")
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
    exact_vals = exact_matrix[:, 0]
    pred_vals = pred_matrix[:, 0]
    ax.plot(y_layers, exact_vals, marker="o", linestyle="--", linewidth=2.0, label="Exact (paper)")
    ax.plot(y_layers, pred_vals, marker="s", linestyle="-", linewidth=1.8, label="Prediction")
    ax.set_title(title)
    ax.set_xlabel("y layer")
    ax.set_ylabel("uy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate quench2026 physical error heatmaps")
    parser.add_argument("--results-root", type=str, default="results/quench2026")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/quench2026/report_pack/physical_heatmaps",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="baseline_lbfgs5000,nsga2_lbfgs5000,nsga3_lbfgs5000,bayesian_lbfgs5000,baseline_original",
        help="Comma-separated scenario keys",
    )
    parser.add_argument(
        "--include-baseline-delta",
        action="store_true",
        help="Also emit delta-vs-baseline files for the baseline scenario itself",
    )
    return parser.parse_args()


def finite_range(vals: np.ndarray) -> Tuple[float, float]:
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi <= lo:
        eps = 1e-12 if lo == 0.0 else abs(lo) * 1e-6
        return lo - eps, hi + eps
    return lo, hi


def to_float(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    results_root = (repo_root / args.results_root).resolve()
    out_root = (repo_root / args.out_dir).resolve()
    ensure_dir(out_root)

    baseline_script = repo_root / "naspinn_baseline_with_quench_2026_data.py"
    baseline_mod = load_baseline_module(baseline_script)
    ref = baseline_mod.build_quench_reference_data()

    selected = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
    scenario_map = scenario_dirs(results_root)

    scenario_outputs: Dict[str, Dict[str, object]] = {}
    summary_rows: List[Dict] = []

    for scenario in selected:
        run_dir = scenario_map.get(scenario)
        if run_dir is None or not run_dir.exists():
            print(f"[skip] scenario missing: {scenario}")
            continue

        meta = read_json(run_dir / "run_meta.json")
        if not isinstance(meta, dict):
            print(f"[skip] run_meta missing: {run_dir}")
            continue

        layers = int(meta.get("layers"))
        neurons = int(meta.get("base_neurons"))
        best_stage = str(meta.get("best_stage", "")).strip().lower() or None
        ckpt = infer_checkpoint(run_dir, best_stage)

        model = baseline_mod.NAS_PINN(layers=layers, base_neurons=neurons).to(baseline_mod.device)
        state = torch.load(str(ckpt), map_location=baseline_mod.device)
        model.load_state_dict(state)
        model.eval()

        y_vals, t_vals, t_pred, t_ref, t_err = build_temperature_heatmap_arrays(model, ref)
        x_vals, y_layers, u_pred, u_ref_m, u_err = build_displacement_x0_arrays(model, ref)

        t_mae = float(np.mean(t_err))
        t_rmse = float(np.sqrt(np.mean((t_pred - t_ref) ** 2)))
        u_mae = float(np.mean(u_err))
        u_rmse = float(np.sqrt(np.mean((u_pred - u_ref_m) ** 2)))

        scenario_outputs[scenario] = {
            "run_dir": run_dir,
            "layers": layers,
            "neurons": neurons,
            "best_stage": best_stage,
            "best_objective": to_float(meta.get("best_objective")),
            "checkpoint": ckpt,
            "t_vals": t_vals,
            "y_vals": y_vals,
            "t_err": t_err,
            "t_pred": t_pred,
            "t_ref": t_ref,
            "x_vals": x_vals,
            "y_layers": y_layers,
            "u_err": u_err,
            "u_pred": u_pred,
            "u_ref": u_ref_m,
            "temp_mae": t_mae,
            "temp_rmse": t_rmse,
            "disp_mae": u_mae,
            "disp_rmse": u_rmse,
        }
        print(
            f"[ok] {scenario} | temp_mae={t_mae:.4e} disp_mae={u_mae:.4e} "
            f"(disp scope: x=0 reference only)"
        )

    if not scenario_outputs:
        print("[done] no scenarios were processed.")
        return

    temp_all = np.concatenate(
        [
            np.asarray(v["t_ref"]).reshape(-1)
            for v in scenario_outputs.values()
        ]
        + [
            np.asarray(v["t_pred"]).reshape(-1)
            for v in scenario_outputs.values()
        ]
    )
    disp_all = np.concatenate(
        [
            np.asarray(v["u_ref"]).reshape(-1)
            for v in scenario_outputs.values()
        ]
        + [
            np.asarray(v["u_pred"]).reshape(-1)
            for v in scenario_outputs.values()
        ]
    )
    temp_vmin, temp_vmax = finite_range(temp_all)
    disp_vmin, disp_vmax = finite_range(disp_all)
    baseline_key = "baseline_original" if "baseline_original" in scenario_outputs else next(iter(scenario_outputs))

    for scenario, out in scenario_outputs.items():
        scenario_dir = out_root / scenario
        ensure_dir(scenario_dir)

        t_ref = np.asarray(out["t_ref"])
        t_pred = np.asarray(out["t_pred"])
        t_vals = np.asarray(out["t_vals"])
        y_vals = np.asarray(out["y_vals"])
        write_matrix_csv(
            matrix=t_ref,
            x_axis=t_vals,
            y_axis=y_vals,
            x_name="time_s",
            y_name="y_coord",
            value_name="temp_exact",
            out_csv=scenario_dir / "temp_exact_grid.csv",
        )
        write_matrix_csv(
            matrix=t_pred,
            x_axis=t_vals,
            y_axis=y_vals,
            x_name="time_s",
            y_name="y_coord",
            value_name="temp_pred",
            out_csv=scenario_dir / "temp_pred_grid.csv",
        )
        save_side_by_side_heatmaps(
            exact_matrix=t_ref,
            pred_matrix=t_pred,
            x_ticks=t_vals,
            y_ticks=y_vals,
            x_label="time (s)",
            y_label="y (model coord)",
            title=f"{scenario} | Temperature Exact vs Prediction",
            out_png=scenario_dir / "temp_exact_vs_pred_heatmap.png",
            cmap="magma",
            vmin=temp_vmin,
            vmax=temp_vmax,
            cbar_label="temperature (degC)",
        )
        save_temperature_profile_plot(
            t_vals=t_vals,
            y_vals=y_vals,
            exact_matrix=t_ref,
            pred_matrix=t_pred,
            out_png=scenario_dir / "temp_exact_vs_pred_curves.png",
            title=f"{scenario} | Temperature Curves (Exact vs Pred)",
        )

        u_ref = np.asarray(out["u_ref"])
        u_pred = np.asarray(out["u_pred"])
        x_vals = np.asarray(out["x_vals"])
        y_layers = np.asarray(out["y_layers"])
        write_matrix_csv(
            matrix=u_ref,
            x_axis=x_vals,
            y_axis=y_layers,
            x_name="x_coord",
            y_name="y_layer",
            value_name="uy_exact",
            out_csv=scenario_dir / "disp_exact_grid.csv",
        )
        write_matrix_csv(
            matrix=u_pred,
            x_axis=x_vals,
            y_axis=y_layers,
            x_name="x_coord",
            y_name="y_layer",
            value_name="uy_pred",
            out_csv=scenario_dir / "disp_pred_grid.csv",
        )
        save_side_by_side_heatmaps(
            exact_matrix=u_ref,
            pred_matrix=u_pred,
            x_ticks=x_vals,
            y_ticks=y_layers,
            x_label="x (reference uses x=0 only)",
            y_label="y layer",
            title=f"{scenario} | Displacement Exact vs Prediction",
            out_png=scenario_dir / "disp_exact_vs_pred_heatmap.png",
            cmap="viridis",
            vmin=disp_vmin,
            vmax=disp_vmax,
            cbar_label="displacement (m)",
        )
        save_displacement_profile_plot(
            y_layers=y_layers,
            exact_matrix=u_ref,
            pred_matrix=u_pred,
            out_png=scenario_dir / "disp_exact_vs_pred_curve.png",
            title=f"{scenario} | Displacement Curve (Exact vs Pred)",
        )

        t_err = np.abs(t_pred - t_ref)
        u_err = np.abs(u_pred - u_ref)
        t_mae = float(np.mean(t_err))
        t_rmse = float(np.sqrt(np.mean((t_pred - t_ref) ** 2)))
        u_mae = float(np.mean(u_err))
        u_rmse = float(np.sqrt(np.mean((u_pred - u_ref) ** 2)))
        summary_rows.append(
            {
                "scenario": scenario,
                "run_dir": str(Path(out["run_dir"]).resolve()),
                "layers": int(out["layers"]),
                "neurons": int(out["neurons"]),
                "checkpoint": str(Path(out["checkpoint"]).resolve()),
                "best_stage": out["best_stage"],
                "best_objective": float(out["best_objective"]),
                "temp_mae": t_mae,
                "temp_rmse": t_rmse,
                "disp_mae": u_mae,
                "disp_rmse": u_rmse,
                "temp_metric_scope": "paper_temp_reference_grid",
                "disp_metric_scope": "paper_disp_reference_x0_only",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    baseline_row = summary_df[summary_df["scenario"] == baseline_key]
    if not baseline_row.empty:
        base_temp_mae = float(baseline_row.iloc[0]["temp_mae"])
        base_disp_mae = float(baseline_row.iloc[0]["disp_mae"])
        summary_df["temp_vs_baseline_pct"] = (summary_df["temp_mae"] / base_temp_mae * 100.0).round(3)
        summary_df["disp_vs_baseline_pct"] = (summary_df["disp_mae"] / base_disp_mae * 100.0).round(3)

    temp_best = float(summary_df["temp_mae"].min())
    disp_best = float(summary_df["disp_mae"].min())
    summary_df["temp_score_pct"] = (temp_best / summary_df["temp_mae"] * 100.0).round(3)
    summary_df["disp_score_pct"] = (disp_best / summary_df["disp_mae"] * 100.0).round(3)
    summary_df["combined_score_pct"] = (
        0.5 * summary_df["temp_score_pct"] + 0.5 * summary_df["disp_score_pct"]
    ).round(3)
    summary_df = summary_df.sort_values(["combined_score_pct", "disp_mae"], ascending=[False, True]).reset_index(drop=True)
    summary_df.to_csv(out_root / "physical_accuracy_summary.csv", index=False)

    print(f"Exact-vs-pred heatmaps/plots written: {out_root}")


if __name__ == "__main__":
    main()
