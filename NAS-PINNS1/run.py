#!/usr/bin/env python3
"""Main runner for the paper-baseline GPU quench pipeline."""

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from cfg import BASELINE_SCRIPT, PYMOO_DIR, parse_args
from compare import generate_exact_pred_and_l2
from core import (
    append_log,
    ensure_dir,
    load_json,
    load_module,
    require_gpu,
    run_final_training,
    save_json,
)
from opt_bayes import search_bayes
from opt_nsga2 import search_nsga2
from opt_nsga3 import search_nsga3
from optimizer_bridge import load_pymoo_reference_optimizers


def run_method(
    method: str,
    args,
    baseline_mod,
    custom_nsga2,
    custom_nsga3,
    env: Dict[str, str],
    save_dir: Path,
) -> Dict:
    method_dir = save_dir / method
    ensure_dir(method_dir)
    log_path = method_dir / "logs" / f"{method}.log"
    state_path = method_dir / "run_state.json"
    state = load_json(state_path, default={"method": method, "search_done": False, "final_done": False})

    append_log(log_path, f"=== {method} run started ===")
    if not state.get("search_done", False):
        if method == "nsga2":
            search_result = search_nsga2(args, baseline_mod, BASELINE_SCRIPT, custom_nsga2, method_dir, log_path, env)
        elif method == "nsga3":
            search_result = search_nsga3(args, baseline_mod, BASELINE_SCRIPT, custom_nsga3, method_dir, log_path, env)
        elif method == "bayesian":
            search_result = search_bayes(args, baseline_mod, BASELINE_SCRIPT, method_dir, log_path, env)
        else:
            raise ValueError(f"Unknown method: {method}")
        state["search_done"] = True
        state["search"] = search_result
        save_json(state_path, state)
        append_log(log_path, f"search done | best={search_result['best_layers']}x{search_result['best_neurons']}")
    else:
        append_log(log_path, "search already done, using cached state")

    if not args.skip_final and (not state.get("final_done", False) or bool(args.force_final)):
        best_layers = int(state["search"]["best_layers"])
        best_neurons = int(state["search"]["best_neurons"])
        final_result = run_final_training(args, BASELINE_SCRIPT, method_dir, best_layers, best_neurons, log_path, env)
        state["final_done"] = True
        state["final"] = final_result
        save_json(state_path, state)
        append_log(log_path, f"final done | best_stage={final_result.get('best_stage')}")
    elif args.skip_final:
        append_log(log_path, "final stage skipped")
    else:
        append_log(log_path, "final already done")

    append_log(log_path, f"=== {method} run completed ===")
    return state


def write_pipeline_comparison(save_dir: Path, methods: List[str]) -> None:
    rows = []
    stage_rows = []
    for method in methods:
        run_state = load_json(save_dir / method / "run_state.json", default={})
        search = run_state.get("search", {})
        final = run_state.get("final", {})
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

    pd.DataFrame(rows).to_csv(save_dir / "comparison.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(save_dir / "stage_comparison.csv", index=False)


def main():
    args = parse_args()
    require_gpu()
    if not BASELINE_SCRIPT.exists():
        raise FileNotFoundError(f"Missing baseline script: {BASELINE_SCRIPT}")

    save_dir = Path(args.save_dir).resolve()
    ensure_dir(save_dir)
    pipeline_log = save_dir / "logs" / "pipeline.log"
    append_log(pipeline_log, f"Pipeline start | methods={args.methods}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    baseline_mod = load_module("paper_baseline_module", BASELINE_SCRIPT)
    custom_nsga2, custom_nsga3, _custom_pso = load_pymoo_reference_optimizers(PYMOO_DIR)

    methods = [m.strip().lower() for m in str(args.methods).split(",") if m.strip()]
    valid = {"nsga2", "nsga3", "bayesian"}
    for m in methods:
        if m not in valid:
            raise ValueError(f"Unknown method '{m}'. Valid: {sorted(valid)}")

    run_states = {}
    for method in methods:
        append_log(pipeline_log, f"[{method}] start")
        state = run_method(method, args, baseline_mod, custom_nsga2, custom_nsga3, env, save_dir)
        run_states[method] = state
        append_log(pipeline_log, f"[{method}] done")

    write_pipeline_comparison(save_dir, methods)

    if not args.skip_final:
        method_to_run_dir: Dict[str, Path] = {}
        for method, state in run_states.items():
            final_dir = state.get("final", {}).get("final_dir")
            if isinstance(final_dir, str) and final_dir:
                method_to_run_dir[method] = Path(final_dir)
        if method_to_run_dir:
            generate_exact_pred_and_l2(
                baseline_mod=baseline_mod,
                method_to_run_dir=method_to_run_dir,
                out_dir=save_dir / "comparison_plots",
            )
            append_log(pipeline_log, "Exact-vs-pred and L2 comparison plots generated")

    append_log(pipeline_log, "Pipeline completed")
    print(f"Saved in: {save_dir}")


if __name__ == "__main__":
    main()
