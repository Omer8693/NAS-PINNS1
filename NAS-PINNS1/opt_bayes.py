#!/usr/bin/env python3
"""Bayesian optimization search routine."""

from pathlib import Path
from typing import Dict, List

import numpy as np
from bayes_opt import BayesianOptimization

from core import evaluate_architecture, save_json


def search_bayes(args, baseline_mod, baseline_script: Path, method_dir: Path, log_path: Path, env: Dict[str, str]) -> Dict:
    eval_counter = {"k": 0}
    rows: List[Dict] = []
    best = {"layers": None, "neurons": None, "proxy_loss": float("inf"), "param_count": float("nan")}

    def objective(layers, neurons):
        eval_counter["k"] += 1
        l_int = int(np.clip(np.round(float(layers)), args.layers_min, args.layers_max))
        n_int = int(np.clip(np.round(float(neurons)), args.neurons_min, args.neurons_max))
        proxy_loss, param_count, _ = evaluate_architecture(
            args=args,
            baseline_mod=baseline_mod,
            baseline_script=baseline_script,
            method_dir=method_dir,
            layers=l_int,
            neurons=n_int,
            eval_idx=eval_counter["k"],
            log_path=log_path,
            env=env,
        )
        rows.append(
            {
                "eval": eval_counter["k"],
                "layers": l_int,
                "neurons": n_int,
                "proxy_loss": float(proxy_loss),
                "param_count": float(param_count),
            }
        )
        if proxy_loss < best["proxy_loss"]:
            best["layers"] = l_int
            best["neurons"] = n_int
            best["proxy_loss"] = float(proxy_loss)
            best["param_count"] = float(param_count)
        return -float(proxy_loss)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={
            "layers": (args.layers_min, args.layers_max),
            "neurons": (args.neurons_min, args.neurons_max),
        },
        random_state=args.seed,
        verbose=2,
    )
    optimizer.maximize(init_points=args.bo_init_points, n_iter=args.bo_iters)

    with (method_dir / "search_population.csv").open("w", encoding="utf-8") as f:
        f.write("eval,layers,neurons,obj_proxy_loss,obj_param_count\n")
        for row in rows:
            f.write(
                f"{row['eval']},{row['layers']},{row['neurons']},"
                f"{row['proxy_loss']:.8e},{row['param_count']:.1f}\n"
            )

    result = {
        "best_layers": int(best["layers"]),
        "best_neurons": int(best["neurons"]),
        "best_proxy_loss": float(best["proxy_loss"]),
        "best_param_count": float(best["param_count"]),
        "population_size": int(len(rows)),
    }
    save_json(method_dir / "search_result.json", result)
    return result
