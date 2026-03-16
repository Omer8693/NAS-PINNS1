import argparse
import csv
import json
import os
from pathlib import Path

from Poisson_2D_Common import save_json, start_run_logging


EXPERIMENT_SPECS = {
    "fem": {
        "dir_name": "2D-Poisson-FEM",
        "file_name": "FEM_results.json",
        "kind": "fem_sweep",
    },
    "pinn": {
        "dir_name": "2D-Poisson-PINN",
        "file_name": "PINNs_evaluation.json",
        "kind": "pinn_sweep",
    },
    "naspinn": {
        "dir_name": "2D-Poisson-NASPINN",
        "file_name": "NASPINN_evaluation.json",
        "kind": "single",
    },
    "nsga2": {
        "dir_name": "2D-Poisson-NSGA2",
        "file_name": "NSGA2_evaluation.json",
        "kind": "single",
    },
    "nsga3": {
        "dir_name": "2D-Poisson-NSGA3",
        "file_name": "NSGA3_evaluation.json",
        "kind": "single",
    },
    "bayesian": {
        "dir_name": "2D-Poisson-Bayesian",
        "file_name": "Bayesian_evaluation.json",
        "kind": "single",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare 2D Poisson FEM/PINN/NAS results")
    parser.add_argument(
        "--results-root",
        type=str,
        default=".",
        help="root directory that contains 2D-Poisson-* result folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="directory to write comparison outputs; default is <results-root>/2D-Poisson-Comparison",
    )
    return parser.parse_args()


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_arch(arch):
    if isinstance(arch, list):
        return "[" + ", ".join(str(item) for item in arch) + "]"
    return str(arch)


def _numeric_items(mapping):
    return sorted(mapping.items(), key=lambda item: int(item[0]))


def parse_fem(eval_path, result_dir):
    data = _read_json(eval_path)
    rows = []

    for mesh_key, rel_l2 in _numeric_items(data.get("l2_rel", {})):
        solve_time = float(data["times_solve"][mesh_key])
        eval_time = float(data["times_eval"][mesh_key])
        rows.append(
            {
                "method": "fem",
                "config_key": mesh_key,
                "config_label": f"mesh={mesh_key}",
                "rel_l2": float(rel_l2),
                "times_adam": None,
                "times_lbfgs": None,
                "times_solve": solve_time,
                "times_eval": eval_time,
                "times_total": solve_time + eval_time,
                "notes": "",
                "result_dir": str(result_dir),
                "source_file": str(eval_path),
            }
        )

    best_row = min(rows, key=lambda row: row["rel_l2"])
    return best_row, rows


def parse_pinn(eval_path, result_dir):
    data = _read_json(eval_path)
    rows = []

    for idx_key, rel_l2 in _numeric_items(data.get("l2_rel", {})):
        arch = data["arch"][idx_key]
        rows.append(
            {
                "method": "pinn",
                "config_key": idx_key,
                "config_label": f"arch={_format_arch(arch)}",
                "rel_l2": float(rel_l2),
                "times_adam": float(data["times_adam"][idx_key]),
                "times_lbfgs": float(data["times_lbfgs"][idx_key]),
                "times_solve": None,
                "times_eval": float(data["times_eval"][idx_key]),
                "times_total": float(data["times_total"][idx_key]),
                "notes": f"var={float(data['var'][idx_key]):.6e}",
                "result_dir": str(result_dir),
                "source_file": str(eval_path),
            }
        )

    best_row = min(rows, key=lambda row: row["rel_l2"])
    return best_row, rows


def parse_single(method, eval_path, result_dir):
    data = _read_json(eval_path)

    if "best_widths" in data:
        config_label = f"widths={_format_arch(data['best_widths'])}"
    elif "arch" in data:
        config_label = "learned_arch"
    else:
        config_label = "default"

    row = {
        "method": method,
        "config_key": "best",
        "config_label": config_label,
        "rel_l2": float(data["l2_rel"]),
        "times_adam": float(data.get("times_adam", 0.0)) if data.get("times_adam") is not None else None,
        "times_lbfgs": float(data.get("times_lbfgs", 0.0)) if data.get("times_lbfgs") is not None else None,
        "times_solve": None,
        "times_eval": float(data.get("times_eval", 0.0)) if data.get("times_eval") is not None else None,
        "times_total": float(data.get("times_total", 0.0)) if data.get("times_total") is not None else None,
        "notes": "",
        "result_dir": str(result_dir),
        "source_file": str(eval_path),
    }
    return row, [row], data


def load_method_summary(method, spec, results_root):
    result_dir = results_root / spec["dir_name"]
    eval_path = result_dir / spec["file_name"]

    if not eval_path.exists():
        return None, None, {"status": "missing", "expected_file": str(eval_path)}

    if spec["kind"] == "fem_sweep":
        best_row, all_rows = parse_fem(eval_path, result_dir)
        return best_row, all_rows, {"status": "ok", "selected": best_row, "all_rows": all_rows}

    if spec["kind"] == "pinn_sweep":
        best_row, all_rows = parse_pinn(eval_path, result_dir)
        return best_row, all_rows, {"status": "ok", "selected": best_row, "all_rows": all_rows}

    best_row, all_rows, raw = parse_single(method, eval_path, result_dir)
    return best_row, all_rows, {"status": "ok", "selected": best_row, "raw": raw}


def write_summary_csv(path, rows):
    fieldnames = [
        "rank",
        "method",
        "config_key",
        "config_label",
        "rel_l2",
        "times_adam",
        "times_lbfgs",
        "times_solve",
        "times_eval",
        "times_total",
        "notes",
        "result_dir",
        "source_file",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_root / "2D-Poisson-Comparison"

    _, stop_logging = start_run_logging(str(output_dir), log_name="comparison.log")
    try:
        print(f"Results root : {results_root}")
        print(f"Output dir   : {output_dir}")

        best_rows = []
        details = {}

        for method, spec in EXPERIMENT_SPECS.items():
            best_row, all_rows, detail = load_method_summary(method, spec, results_root)
            details[method] = detail

            if best_row is None:
                print(f"[MISSING] {method}: expected {detail['expected_file']}")
                continue

            best_rows.append(best_row)
            print(
                f"[OK] {method:8s} best={best_row['config_label']} "
                f"rel_l2={best_row['rel_l2']:.8e} total={best_row['times_total']}"
            )

        ranked_rows = sorted(best_rows, key=lambda row: row["rel_l2"])
        for rank, row in enumerate(ranked_rows, start=1):
            row["rank"] = rank

        if not ranked_rows:
            print("No result files found. Nothing to compare.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "comparison_summary.csv"
        json_path = output_dir / "comparison_summary.json"
        details_path = output_dir / "comparison_details.json"

        write_summary_csv(csv_path, ranked_rows)
        save_json(
            json_path,
            {
                "results_root": str(results_root),
                "ranking": ranked_rows,
            },
            indent=4,
        )
        save_json(details_path, details, indent=4)

        print("\nRanking by relative L2:")
        for row in ranked_rows:
            print(
                f"{row['rank']:2d}. {row['method']:8s} "
                f"{row['rel_l2']:.8e} | {row['config_label']}"
            )

        print(f"\nSaved CSV    : {csv_path}")
        print(f"Saved JSON   : {json_path}")
        print(f"Saved detail : {details_path}")
        print(f"Saved log    : {output_dir / 'comparison.log'}")
    finally:
        stop_logging()


if __name__ == "__main__":
    main()
