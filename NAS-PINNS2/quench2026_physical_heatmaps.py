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

    temp_all = np.concatenate([np.asarray(v["t_err"]).reshape(-1) for v in scenario_outputs.values()])
    disp_all = np.concatenate([np.asarray(v["u_err"]).reshape(-1) for v in scenario_outputs.values()])
    temp_vmin, temp_vmax = finite_range(temp_all)
    disp_vmin, disp_vmax = finite_range(disp_all)

    baseline_key = "baseline_original" if "baseline_original" in scenario_outputs else next(iter(scenario_outputs))
    base_t_err = np.asarray(scenario_outputs[baseline_key]["t_err"])
    base_u_err = np.asarray(scenario_outputs[baseline_key]["u_err"])

    temp_delta_max = 0.0
    disp_delta_max = 0.0
    for out in scenario_outputs.values():
        t_delta = np.asarray(out["t_err"]) - base_t_err
        u_delta = np.asarray(out["u_err"]) - base_u_err
        temp_delta_max = max(temp_delta_max, float(np.max(np.abs(t_delta))))
        disp_delta_max = max(disp_delta_max, float(np.max(np.abs(u_delta))))
    temp_delta_max = max(temp_delta_max, 1e-12)
    disp_delta_max = max(disp_delta_max, 1e-12)

    for scenario, out in scenario_outputs.items():
        scenario_dir = out_root / scenario
        ensure_dir(scenario_dir)

        t_err = np.asarray(out["t_err"])
        t_vals = np.asarray(out["t_vals"])
        y_vals = np.asarray(out["y_vals"])
        write_matrix_csv(
            matrix=t_err,
            x_axis=t_vals,
            y_axis=y_vals,
            x_name="time_s",
            y_name="y_coord",
            value_name="abs_temp_error",
            out_csv=scenario_dir / "temp_error_grid.csv",
        )
        save_heatmap(
            matrix=t_err,
            x_ticks=t_vals,
            y_ticks=y_vals,
            x_label="time (s)",
            y_label="y (model coord)",
            title=f"{scenario} | |T_pred - T_ref| (shared scale)",
            out_png=scenario_dir / "temp_error_heatmap.png",
            cmap="magma",
            vmin=temp_vmin,
            vmax=temp_vmax,
            cbar_label="absolute temperature error (degC)",
        )

        u_err = np.asarray(out["u_err"])
        x_vals = np.asarray(out["x_vals"])
        y_layers = np.asarray(out["y_layers"])
        write_matrix_csv(
            matrix=u_err,
            x_axis=x_vals,
            y_axis=y_layers,
            x_name="x_coord",
            y_name="y_layer",
            value_name="abs_uy_error",
            out_csv=scenario_dir / "disp_error_grid.csv",
        )
        save_heatmap(
            matrix=u_err,
            x_ticks=x_vals,
            y_ticks=y_layers,
            x_label="x (reference uses x=0 only)",
            y_label="y layer",
            title=f"{scenario} | |uy_pred - uy_ref| at x=0 (shared scale)",
            out_png=scenario_dir / "disp_error_heatmap.png",
            cmap="viridis",
            vmin=disp_vmin,
            vmax=disp_vmax,
            cbar_label="absolute displacement error (m)",
        )

        t_delta = t_err - base_t_err
        write_matrix_csv(
            matrix=t_delta,
            x_axis=t_vals,
            y_axis=y_vals,
            x_name="time_s",
            y_name="y_coord",
            value_name="delta_temp_error_vs_baseline",
            out_csv=scenario_dir / "temp_error_delta_vs_baseline_grid.csv",
        )
        save_heatmap(
            matrix=t_delta,
            x_ticks=t_vals,
            y_ticks=y_vals,
            x_label="time (s)",
            y_label="y (model coord)",
            title=f"{scenario} - {baseline_key} | delta temp abs-error",
            out_png=scenario_dir / "temp_error_delta_vs_baseline_heatmap.png",
            cmap="coolwarm",
            vmin=-temp_delta_max,
            vmax=temp_delta_max,
            center=0.0,
            cbar_label="delta abs error vs baseline (degC)",
        )

        u_delta = u_err - base_u_err
        write_matrix_csv(
            matrix=u_delta,
            x_axis=x_vals,
            y_axis=y_layers,
            x_name="x_coord",
            y_name="y_layer",
            value_name="delta_disp_error_vs_baseline",
            out_csv=scenario_dir / "disp_error_delta_vs_baseline_grid.csv",
        )
        save_heatmap(
            matrix=u_delta,
            x_ticks=x_vals,
            y_ticks=y_layers,
            x_label="x (reference uses x=0 only)",
            y_label="y layer",
            title=f"{scenario} - {baseline_key} | delta disp abs-error",
            out_png=scenario_dir / "disp_error_delta_vs_baseline_heatmap.png",
            cmap="coolwarm",
            vmin=-disp_delta_max,
            vmax=disp_delta_max,
            center=0.0,
            cbar_label="delta abs error vs baseline (m)",
        )

        summary_rows.append(
            {
                "scenario": scenario,
                "run_dir": str(Path(out["run_dir"]).resolve()),
                "layers": int(out["layers"]),
                "neurons": int(out["neurons"]),
                "checkpoint": str(Path(out["checkpoint"]).resolve()),
                "best_stage": out["best_stage"],
                "best_objective": float(out["best_objective"]),
                "temp_mae": float(out["temp_mae"]),
                "temp_rmse": float(out["temp_rmse"]),
                "disp_mae": float(out["disp_mae"]),
                "disp_rmse": float(out["disp_rmse"]),
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

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(summary_df["scenario"], summary_df["temp_mae"])
    ax.set_yscale("log")
    ax.set_ylabel("temp_mae (log)")
    ax.set_title("Temperature MAE (fair ref grid) by Scenario")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_root / "temp_mae_bar.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(summary_df["scenario"], summary_df["disp_mae"])
    ax.set_yscale("log")
    ax.set_ylabel("disp_mae (log)")
    ax.set_title("Displacement MAE at x=0 (fair ref points) by Scenario")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_root / "disp_mae_bar.png", dpi=180)
    plt.close(fig)

    print(f"Physical heatmaps written: {out_root}")
    print(f"Baseline for delta heatmaps: {baseline_key}")


if __name__ == "__main__":
    main()
