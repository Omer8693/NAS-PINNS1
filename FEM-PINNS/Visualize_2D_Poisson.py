import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from Compare_2D_Poisson import EXPERIMENT_SPECS, load_method_summary
from Poisson_2D_Common import analytic_sol, load_eval_points, save_json, start_run_logging


METHOD_DISPLAY_NAMES = {
    "fem": "FEM",
    "pinn": "PINN",
    "naspinn": "NAS-PINN",
    "nsga2": "NSGA-II",
    "nsga3": "NSGA-III",
    "nsga2_pymoo": "NSGA-II (local pymoo)",
    "nsga3_pymoo": "NSGA-III (local pymoo)",
    "bayesian": "Bayesian",
}

RESULT_FILE_NAMES = {
    "fem": "FEM_results.json",
    "pinn": "PINNs_results.json",
    "naspinn": "NASPINN_results.json",
    "nsga2": "NSGA2_results.json",
    "nsga3": "NSGA3_results.json",
    "nsga2_pymoo": "NSGA2_Pymoo_results.json",
    "nsga3_pymoo": "NSGA3_Pymoo_results.json",
    "bayesian": "Bayesian_results.json",
}

METHOD_ORDER = [
    "fem",
    "pinn",
    "naspinn",
    "nsga2",
    "nsga3",
    "nsga2_pymoo",
    "nsga3_pymoo",
    "bayesian",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create visual comparisons for 2D Poisson exact/prediction/error results"
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=".",
        help="root directory that contains the 2D-Poisson-* result folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="directory to write figures; default is <results-root>/2D-Poisson-Comparison",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="figure DPI",
    )
    return parser.parse_args()


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_reference_points():
    points = load_eval_points(target_device=torch.device("cpu")).detach().cpu().numpy()
    return np.asarray(points, dtype=np.float64)


def _reshape_on_grid(points, values):
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)

    x_unique = np.unique(points[:, 0])
    y_unique = np.unique(points[:, 1])

    if x_unique.size * y_unique.size != points.shape[0]:
        raise ValueError("Evaluation points do not form a structured grid")

    x_index = np.searchsorted(x_unique, points[:, 0])
    y_index = np.searchsorted(y_unique, points[:, 1])

    grid = np.full((y_unique.size, x_unique.size), np.nan, dtype=np.float64)
    grid[y_index, x_index] = values

    if np.isnan(grid).any():
        raise ValueError("Grid reshape left missing values")

    return x_unique, y_unique, grid


def _load_selected_dataset(method, spec, results_root):
    selected_row, _, detail = load_method_summary(method, spec, results_root)
    if selected_row is None:
        expected = detail.get("expected_file", "unknown")
        raise FileNotFoundError(f"Missing data for {method}: {expected}")

    result_path = results_root / spec["dir_name"] / RESULT_FILE_NAMES[method]
    result_data = _read_json(result_path)

    if spec["kind"] == "fem_sweep":
        points = _get_reference_points()
        pred = np.asarray(result_data["y_results"][selected_row["config_key"]], dtype=np.float64)
    elif spec["kind"] == "pinn_sweep":
        key = selected_row["config_key"]
        points = np.asarray(result_data["domain_pts"][key], dtype=np.float64)
        pred = np.asarray(result_data["y_results"][key], dtype=np.float64)
    else:
        points = np.asarray(result_data["domain_pts"], dtype=np.float64)
        pred = np.asarray(result_data["y_results"], dtype=np.float64)

    exact = np.asarray(analytic_sol(points[:, 0], points[:, 1]), dtype=np.float64).reshape(-1)
    pred = pred.reshape(-1)
    sq_error = (pred - exact) ** 2

    x_unique, y_unique, exact_grid = _reshape_on_grid(points, exact)
    _, _, pred_grid = _reshape_on_grid(points, pred)
    _, _, sq_error_grid = _reshape_on_grid(points, sq_error)

    return {
        "method": method,
        "display_name": METHOD_DISPLAY_NAMES[method],
        "config_key": selected_row["config_key"],
        "config_label": selected_row["config_label"],
        "rel_l2": float(selected_row["rel_l2"]),
        "points": points,
        "x_unique": x_unique,
        "y_unique": y_unique,
        "exact": exact,
        "pred": pred,
        "sq_error": sq_error,
        "exact_grid": exact_grid,
        "pred_grid": pred_grid,
        "sq_error_grid": sq_error_grid,
    }


def _clean_config_label(label):
    cleaned = label.replace("arch=", "").replace("widths=", "").replace("mesh=", "mesh ")
    return cleaned.replace(", ", ",")


def _style_axis(ax):
    ax.set_facecolor("#fffdf8")
    for spine in ax.spines.values():
        spine.set_color("#d8cfbf")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#4c463f", labelsize=8)


def _remove_legacy_outputs(output_dir):
    removed = []
    legacy_names = [
        "poisson_exact_prediction_comparison.png",
        "poisson_l2_error_heatmaps.png",
        "poisson_rel_l2_bar.png",
        "poisson_comparison_single_figure.png",
        "poisson_comparison_single_figure.pdf",
    ]

    for name in legacy_names:
        path = output_dir / name
        if path.exists():
            path.unlink()
            removed.append(str(path))

    return removed


def _plot_single_figure(datasets, png_path, pdf_path, dpi):
    solution_values = [datasets[0]["exact_grid"]]
    solution_values.extend(dataset["pred_grid"] for dataset in datasets)
    solution_vmin = min(float(np.min(values)) for values in solution_values)
    solution_vmax = max(float(np.max(values)) for values in solution_values)

    error_vmax = max(float(np.max(dataset["sq_error_grid"])) for dataset in datasets)
    if error_vmax <= 0.0:
        error_vmax = 1.0

    reference = datasets[0]
    extent = [
        float(reference["x_unique"][0]),
        float(reference["x_unique"][-1]),
        float(reference["y_unique"][0]),
        float(reference["y_unique"][-1]),
    ]

    n_methods = len(datasets)
    panel_cols = max(2, min(4, n_methods))
    pred_rows = math.ceil(n_methods / panel_cols)
    err_rows = math.ceil(n_methods / panel_cols)
    total_rows = 1 + pred_rows + err_rows

    fig = plt.figure(
        figsize=(5.0 * panel_cols, 3.5 * total_rows + 1.0),
        facecolor="#f7f1e3",
    )
    grid = fig.add_gridspec(
        total_rows,
        panel_cols,
        height_ratios=[1.1] + [1.0] * (pred_rows + err_rows),
        hspace=0.30,
        wspace=0.18,
    )

    exact_span = 1 if panel_cols <= 2 else 2
    ax_exact = fig.add_subplot(grid[0, :exact_span])
    ax_rank = fig.add_subplot(grid[0, exact_span:])

    pred_axes = []
    err_axes = []

    pred_row_start = 1
    err_row_start = pred_row_start + pred_rows

    for idx in range(n_methods):
        pred_row = pred_row_start + (idx // panel_cols)
        pred_col = idx % panel_cols
        pred_axes.append(fig.add_subplot(grid[pred_row, pred_col]))

        err_row = err_row_start + (idx // panel_cols)
        err_col = idx % panel_cols
        err_axes.append(fig.add_subplot(grid[err_row, err_col]))

    exact_im = ax_exact.imshow(
        reference["exact_grid"],
        origin="lower",
        extent=extent,
        cmap="YlGnBu",
        vmin=solution_vmin,
        vmax=solution_vmax,
        aspect="equal",
    )
    _style_axis(ax_exact)
    ax_exact.set_title("Exact Solution", fontsize=14, color="#2f2a24", pad=10)
    ax_exact.set_xlabel("x", color="#4c463f")
    ax_exact.set_ylabel("y", color="#4c463f")
    ax_exact.text(
        0.03,
        0.04,
        "Reference field",
        transform=ax_exact.transAxes,
        fontsize=10,
        color="#2f2a24",
        bbox={"facecolor": "#fff7dc", "edgecolor": "#ddc98b", "boxstyle": "round,pad=0.3"},
    )

    ranked = sorted(datasets, key=lambda item: item["rel_l2"])
    labels = [item["display_name"] for item in ranked]
    values = [item["rel_l2"] for item in ranked]
    bar_colors = ["#5b8e7d", "#7ba7b8", "#a4c3b2", "#f2b880", "#de7c5a", "#c06c84", "#6c91bf", "#8d6a9f"]
    if labels:
        bars = ax_rank.barh(labels, values, color=bar_colors[: len(labels)], edgecolor="#8a7f6f")
        ax_rank.invert_yaxis()
        ax_rank.set_xscale("log")
        ax_rank.grid(axis="x", linestyle="--", alpha=0.3, color="#9d9487")
        _style_axis(ax_rank)
        ax_rank.set_title("Relative L2 Ranking", fontsize=14, color="#2f2a24", pad=10)
        ax_rank.set_xlabel("relative L2 error (log scale)", color="#4c463f")
        for bar, value in zip(bars, values):
            ax_rank.text(
                value * 1.08,
                bar.get_y() + bar.get_height() / 2.0,
                f"{value:.2e}",
                va="center",
                ha="left",
                fontsize=10,
                color="#2f2a24",
            )
    else:
        _style_axis(ax_rank)
        ax_rank.set_title("Relative L2 Ranking", fontsize=14, color="#2f2a24", pad=10)

    pred_im = None
    err_im = None
    for ax, dataset in zip(pred_axes, datasets):
        pred_im = ax.imshow(
            dataset["pred_grid"],
            origin="lower",
            extent=extent,
            cmap="YlGnBu",
            vmin=solution_vmin,
            vmax=solution_vmax,
            aspect="equal",
        )
        _style_axis(ax)
        ax.set_title(dataset["display_name"], fontsize=12, color="#2f2a24", pad=6)
        ax.set_xlabel("x", color="#4c463f")
        ax.set_ylabel("y", color="#4c463f")
        ax.text(
            0.03,
            0.96,
            f"relL2 {dataset['rel_l2']:.2e}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            color="#2f2a24",
            bbox={"facecolor": "#fef6e4", "edgecolor": "#e8c77a", "boxstyle": "round,pad=0.25"},
        )

    for ax, dataset in zip(err_axes, datasets):
        err_im = ax.imshow(
            dataset["sq_error_grid"],
            origin="lower",
            extent=extent,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=error_vmax,
            aspect="equal",
        )
        _style_axis(ax)
        ax.set_title(_clean_config_label(dataset["config_label"]), fontsize=10, color="#2f2a24", pad=6)
        ax.set_xlabel("x", color="#4c463f")
        ax.set_ylabel("y", color="#4c463f")
        ax.text(
            0.03,
            0.96,
            "pointwise error",
            transform=ax.transAxes,
            va="top",
            fontsize=8.5,
            color="#5a3e1b",
            bbox={"facecolor": "#fff1df", "edgecolor": "#ebb874", "boxstyle": "round,pad=0.22"},
        )

    prediction_label_y = 1.0 - (pred_row_start + pred_rows / 2.0) / total_rows
    error_label_y = 1.0 - (err_row_start + err_rows / 2.0) / total_rows
    fig.text(0.015, prediction_label_y, "Predictions", rotation=90, va="center", fontsize=13, color="#2f2a24")
    fig.text(0.015, error_label_y, "Squared Error", rotation=90, va="center", fontsize=13, color="#2f2a24")
    fig.suptitle(
        "2D Poisson Comparison: Exact Field, Predictions, and Error Maps",
        fontsize=18,
        color="#221f1b",
        y=0.98,
    )

    solution_axes = [ax_exact] + pred_axes
    solution_cbar = fig.colorbar(pred_im if pred_im is not None else exact_im, ax=solution_axes, fraction=0.025, pad=0.012)
    solution_cbar.set_label("u(x, y)", color="#2f2a24")
    solution_cbar.ax.tick_params(labelsize=8, colors="#4c463f")

    error_cbar = fig.colorbar(err_im, ax=err_axes, fraction=0.025, pad=0.012)
    error_cbar.set_label("(u_pred - u_exact)^2", color="#5a3e1b")
    error_cbar.ax.tick_params(labelsize=8, colors="#4c463f")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_root / "2D-Poisson-Comparison"

    _, stop_logging = start_run_logging(str(output_dir), log_name="visualization.log")
    try:
        print(f"Results root : {results_root}")
        print(f"Output dir   : {output_dir}")

        datasets = []
        missing_methods = []
        for method in METHOD_ORDER:
            spec = EXPERIMENT_SPECS[method]
            try:
                dataset = _load_selected_dataset(method, spec, results_root)
            except FileNotFoundError as exc:
                missing_methods.append({"method": method, "reason": str(exc)})
                print(f"[SKIP] {method:12s} {exc}")
                continue
            datasets.append(dataset)
            print(
                f"[OK] {dataset['display_name']:9s} "
                f"{dataset['config_label']} | relL2={dataset['rel_l2']:.8e}"
            )

        if not datasets:
            print("No available result folders were found for visualization.")
            return

        removed_legacy_files = _remove_legacy_outputs(output_dir)
        single_png_path = output_dir / "poisson_comparison_single_figure.png"
        single_pdf_path = output_dir / "poisson_comparison_single_figure.pdf"
        summary_path = output_dir / "visualization_summary.json"

        _plot_single_figure(datasets, single_png_path, single_pdf_path, args.dpi)

        save_json(
            summary_path,
            {
                "results_root": str(results_root),
                "output_dir": str(output_dir),
                "figures": {
                    "single_figure_png": str(single_png_path),
                    "single_figure_pdf": str(single_pdf_path),
                },
                "removed_legacy_files": removed_legacy_files,
                "missing_methods": missing_methods,
                "selected_methods": [
                    {
                        "method": dataset["method"],
                        "display_name": dataset["display_name"],
                        "config_key": dataset["config_key"],
                        "config_label": dataset["config_label"],
                        "rel_l2": dataset["rel_l2"],
                    }
                    for dataset in datasets
                ],
                "note": "The figure uses pointwise squared error for the heatmaps and relative L2 for the ranking panel.",
            },
            indent=4,
        )

        if removed_legacy_files:
            print("\nRemoved legacy figures:")
            for path in removed_legacy_files:
                print(f"  {path}")

        print("\nSaved figure:")
        print(f"  {single_png_path}")
        print(f"  {single_pdf_path}")
        print(f"\nSaved summary: {summary_path}")
    finally:
        stop_logging()


if __name__ == "__main__":
    main()
