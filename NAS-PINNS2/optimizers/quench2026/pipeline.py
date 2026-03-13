import argparse
import csv
from pathlib import Path

from .bayesian import run as run_bayesian
from .common import append_log, ensure_dir, load_json, save_json
from .nsga2 import run as run_nsga2
from .nsga3 import run as run_nsga3


METHOD_RUNNERS = {
    "nsga2": run_nsga2,
    "nsga3": run_nsga3,
    "bayesian": run_bayesian,
}


def _to_method_args(base_args, method_save_dir: Path):
    payload = vars(base_args).copy()
    payload["save_dir"] = str(method_save_dir)
    return argparse.Namespace(**payload)


def write_comparison(save_dir: Path, methods):
    rows = []
    stage_rows = []
    for method in methods:
        run_state_path = save_dir / method / "run_state.json"
        if not run_state_path.exists():
            continue
        state = load_json(run_state_path, default={})
        search = state.get("search", {})
        final = state.get("final", {})
        rows.append(
            {
                "method": method,
                "best_layers": search.get("best_layers"),
                "best_neurons": search.get("best_neurons"),
                "search_best_proxy_loss": search.get("best_proxy_loss"),
                "search_best_param_count": search.get("best_param_count"),
                "final_best_stage": final.get("best_stage"),
                "final_best_objective": final.get("best_objective"),
                "final_run_time_seconds": final.get("run_time_seconds"),
            }
        )
        for sr in final.get("stage_rows", []):
            stage_rows.append(
                {
                    "method": method,
                    "stage": sr.get("stage"),
                    "objective": sr.get("objective"),
                    "selected": sr.get("selected"),
                }
            )

    comparison_csv = save_dir / "comparison.csv"
    with open(comparison_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "best_layers",
                "best_neurons",
                "search_best_proxy_loss",
                "search_best_param_count",
                "final_best_stage",
                "final_best_objective",
                "final_run_time_seconds",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    stage_csv = save_dir / "stage_comparison.csv"
    with open(stage_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "stage", "objective", "selected"])
        writer.writeheader()
        for row in stage_rows:
            writer.writerow(row)


def parse_args():
    default_save_dir = str((Path(__file__).resolve().parents[2] / "results" / "quench2026" / "pipeline").resolve())
    parser = argparse.ArgumentParser(description="Quench2026 NAS-PINN full optimization pipeline with resume")
    parser.add_argument("--save-dir", type=str, default=default_save_dir)
    parser.add_argument("--methods", type=str, default="nsga2,nsga3,bayesian")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--proxy-epochs", type=int, default=300)
    parser.add_argument("--skip-final", action="store_true")

    parser.add_argument("--layers-min", type=int, default=3)
    parser.add_argument("--layers-max", type=int, default=6)
    parser.add_argument("--neurons-min", type=int, default=64)
    parser.add_argument("--neurons-max", type=int, default=160)
    parser.add_argument("--proxy-fail-penalty", type=float, default=1e30)

    parser.add_argument("--pop-size", type=int, default=24)
    parser.add_argument("--n-gen", type=int, default=16)
    parser.add_argument("--ref-partitions", type=int, default=10)
    parser.add_argument("--bo-init-points", type=int, default=4)
    parser.add_argument("--bo-iters", type=int, default=12)

    parser.add_argument("--n-time-steps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--lbfgs-max-iter", type=int, default=500)
    parser.add_argument("--lbfgs-col-points", type=int, default=2048)
    parser.add_argument("--lbfgs-ic-points", type=int, default=512)
    parser.add_argument("--lbfgs-bc-points", type=int, default=512)
    parser.add_argument("--lbfgs-time-steps", type=int, default=4)
    parser.add_argument("--lbfgs-history-size", type=int, default=20)
    parser.add_argument("--lbfgs-line-search", type=str, default="strong_wolfe", choices=["none", "strong_wolfe"])
    parser.add_argument("--pso-iters", type=int, default=8)
    parser.add_argument("--pso-swarm", type=int, default=16)
    parser.add_argument("--pso-span", type=float, default=0.25)
    parser.add_argument("--skip-lbfgs-final", action="store_true")
    parser.add_argument("--use-pso-final", action="store_true")
    parser.add_argument("--force-final", action="store_true")

    parser.add_argument("--w-physics", type=float, default=50.0)
    parser.add_argument("--w-ic", type=float, default=1e-3)
    parser.add_argument("--w-bc", type=float, default=1e-18)
    parser.add_argument("--w-data", type=float, default=1e-2)
    parser.add_argument("--temp-ref-t0-mode", type=str, default="align_ic", choices=["align_ic", "drop", "keep"])
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir).resolve()
    ensure_dir(save_dir)
    log_path = save_dir / "logs" / "pipeline.log"
    state_path = save_dir / "pipeline_state.json"
    methods = [m.strip().lower() for m in str(args.methods).split(",") if m.strip()]
    for method in methods:
        if method not in METHOD_RUNNERS:
            raise ValueError(f"Unknown method '{method}'. Valid: {', '.join(METHOD_RUNNERS.keys())}")

    state = load_json(state_path, default={"methods": {}, "completed": []})
    append_log(log_path, f"Pipeline start | methods={methods}")

    for method in methods:
        method_state = state["methods"].get(method, {})
        if method_state.get("completed") and not bool(args.force_final):
            append_log(log_path, f"[{method}] already completed, skipping")
            continue

        method_dir = save_dir / method
        method_args = _to_method_args(args, method_dir)
        append_log(log_path, f"[{method}] starting")
        try:
            METHOD_RUNNERS[method](method_args)
        except Exception as exc:
            method_state["completed"] = False
            method_state["error"] = str(exc)
            state["methods"][method] = method_state
            save_json(state_path, state)
            append_log(log_path, f"[{method}] failed: {exc}")
            raise

        method_state["completed"] = True
        method_state["error"] = ""
        state["methods"][method] = method_state
        if method not in state["completed"]:
            state["completed"].append(method)
        save_json(state_path, state)
        append_log(log_path, f"[{method}] completed")

    write_comparison(save_dir, methods)
    append_log(log_path, "Pipeline completed")


if __name__ == "__main__":
    main()
