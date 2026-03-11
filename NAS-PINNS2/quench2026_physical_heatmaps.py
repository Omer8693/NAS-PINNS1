#!/usr/bin/env python3
"""Generate physical error heatmaps: |T_pred-T_ref| and |u_pred-u_ref|."""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
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
    if (run_dir / "baseline_model.pth").exists():
        return run_dir / "baseline_model.pth"
    if best_stage:
        stage_ckpt = run_dir / f"baseline_model_{best_stage}.pth"
        if stage_ckpt.exists():
            return stage_ckpt
    # fallback
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
    baseline_mod,
    model,
    ref: Dict[str, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Reference temp points are built as nested loops: depth outer, time inner.
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

    # Data ordering from build_quench_reference_data: for each depth, iterate time.
    pred_m = pred_np.reshape(n_y, n_t)
    tgt_m = tgt_np.reshape(n_y, n_t)
    err_m = err_np.reshape(n_y, n_t)
    return uniq_y, uniq_t, pred_m, tgt_m, err_m


def build_displacement_heatmap_arrays(
    baseline_mod,
    model,
    ref: Dict[str, torch.Tensor],
    x_grid_size: int = 81,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Reference uy is available on layer y points at x=0, t=t_max.
    y_layers = ref["y_disp"].detach().cpu().numpy().reshape(-1)
    uy_ref = ref["target_uy"].detach().cpu().numpy().reshape(-1)

    x_min = float(baseline_mod.x_min)
    x_max = float(baseline_mod.x_max)
    t_max = float(baseline_mod.t_max)
    x_grid = np.linspace(x_min, x_max, x_grid_size, dtype=np.float32)

    X, Y = np.meshgrid(x_grid, y_layers)
    T = np.full_like(X, t_max, dtype=np.float32)
    inp = np.stack([X.reshape(-1), Y.reshape(-1), T.reshape(-1)], axis=1)
    inp_t = torch.tensor(inp, dtype=torch.float32, device=baseline_mod.device)

    with torch.no_grad():
        pred = model(inp_t)[:, 2:3]  # uy
    uy_pred = pred.detach().cpu().numpy().reshape(len(y_layers), len(x_grid))
    uy_ref_m = np.repeat(uy_ref[:, None], len(x_grid), axis=1)
    err_m = np.abs(uy_pred - uy_ref_m)
    return x_grid, y_layers, uy_pred, uy_ref_m, err_m


def save_heatmap(
    matrix: np.ndarray,
    x_ticks: np.ndarray,
    y_ticks: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    out_png: Path,
    cmap: str = "inferno",
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("absolute error")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Sparse ticks for readability.
    if len(x_ticks) > 1:
        x_idx = np.linspace(0, len(x_ticks) - 1, min(8, len(x_ticks))).astype(int)
        ax.set_xticks(x_idx)
        ax.set_xticklabels([f"{x_ticks[i]:.2f}" for i in x_idx], rotation=20)
    if len(y_ticks) > 1:
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
        best_stage = meta.get("best_stage")
        ckpt = infer_checkpoint(run_dir, best_stage)

        model = baseline_mod.NAS_PINN(layers=layers, base_neurons=neurons).to(baseline_mod.device)
        state = torch.load(str(ckpt), map_location=baseline_mod.device)
        model.load_state_dict(state)
        model.eval()

        scenario_dir = out_root / scenario
        ensure_dir(scenario_dir)

        # Temperature error heatmap over (time, y-depth) reference grid.
        y_vals, t_vals, t_pred, t_ref, t_err = build_temperature_heatmap_arrays(baseline_mod, model, ref)
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
            title=f"{scenario} | |T_pred - T_ref|",
            out_png=scenario_dir / "temp_error_heatmap.png",
            cmap="magma",
        )

        # Displacement error heatmap over (x, y-layer) with layer ref broadcast over x.
        x_vals, y_layers, u_pred, u_ref_m, u_err = build_displacement_heatmap_arrays(baseline_mod, model, ref)
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
            x_label="x (model coord)",
            y_label="y layer",
            title=f"{scenario} | |u_pred - u_ref| (uy)",
            out_png=scenario_dir / "disp_error_heatmap.png",
            cmap="viridis",
        )

        # Summary metrics
        t_mae = float(np.mean(t_err))
        t_rmse = float(np.sqrt(np.mean((t_pred - t_ref) ** 2)))
        u_mae = float(np.mean(u_err))
        u_rmse = float(np.sqrt(np.mean((u_pred - u_ref_m) ** 2)))
        summary_rows.append(
            {
                "scenario": scenario,
                "run_dir": str(run_dir.resolve()),
                "layers": layers,
                "neurons": neurons,
                "checkpoint": str(ckpt.resolve()),
                "best_stage": best_stage,
                "best_objective": float(meta.get("best_objective")),
                "temp_mae": t_mae,
                "temp_rmse": t_rmse,
                "disp_mae": u_mae,
                "disp_rmse": u_rmse,
            }
        )
        print(f"[ok] {scenario} | temp_mae={t_mae:.4e} disp_mae={u_mae:.4e}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(["temp_mae", "disp_mae"], ascending=True).reset_index(drop=True)
        # Accuracy-like score (objective-based inverse error score) for ranking in report tables.
        temp_best = float(summary_df["temp_mae"].min())
        disp_best = float(summary_df["disp_mae"].min())
        summary_df["temp_score_pct"] = (temp_best / summary_df["temp_mae"] * 100.0).round(3)
        summary_df["disp_score_pct"] = (disp_best / summary_df["disp_mae"] * 100.0).round(3)
        summary_df["combined_score_pct"] = (
            0.5 * summary_df["temp_score_pct"] + 0.5 * summary_df["disp_score_pct"]
        ).round(3)
        summary_df.to_csv(out_root / "physical_accuracy_summary.csv", index=False)

        # Quick bar charts for summary
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        ax.bar(summary_df["scenario"], summary_df["temp_mae"])
        ax.set_yscale("log")
        ax.set_ylabel("temp_mae (log)")
        ax.set_title("Temperature MAE by Scenario")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out_root / "temp_mae_bar.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        ax.bar(summary_df["scenario"], summary_df["disp_mae"])
        ax.set_yscale("log")
        ax.set_ylabel("disp_mae (log)")
        ax.set_title("Displacement MAE by Scenario")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out_root / "disp_mae_bar.png", dpi=180)
        plt.close(fig)

    print(f"Physical heatmaps written: {out_root}")


if __name__ == "__main__":
    main()
