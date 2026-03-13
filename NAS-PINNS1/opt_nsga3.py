#!/usr/bin/env python3
"""NSGA3 search routine."""

from pathlib import Path
from typing import Dict

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from core import evaluate_architecture, save_json, write_search_population_csv


def search_nsga3(args, baseline_mod, baseline_script: Path, custom_nsga3, method_dir: Path, log_path: Path, env: Dict[str, str]) -> Dict:
    class Problem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=2,
                n_obj=2,
                n_constr=0,
                xl=np.array([args.layers_min, args.neurons_min], dtype=np.float64),
                xu=np.array([args.layers_max, args.neurons_max], dtype=np.float64),
            )
            self.eval_count = 0

        def _evaluate(self, x, out, *a, **k):
            self.eval_count += 1
            layers = int(np.clip(np.round(float(x[0])), args.layers_min, args.layers_max))
            neurons = int(np.clip(np.round(float(x[1])), args.neurons_min, args.neurons_max))
            loss, params, _ = evaluate_architecture(
                args=args,
                baseline_mod=baseline_mod,
                baseline_script=baseline_script,
                method_dir=method_dir,
                layers=layers,
                neurons=neurons,
                eval_idx=self.eval_count,
                log_path=log_path,
                env=env,
            )
            out["F"] = [loss, params]

    problem = Problem()
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=args.ref_partitions)
    pop_size = max(args.pop_size, len(ref_dirs))
    algorithm = custom_nsga3(ref_dirs=ref_dirs, pop_size=pop_size)
    res = minimize(problem, algorithm, termination=("n_gen", args.n_gen), seed=args.seed, verbose=True)
    X = np.array(res.X, dtype=np.float64)
    F = np.array(res.F, dtype=np.float64)
    best_idx = int(np.argmin(F[:, 0]))
    best_layers = int(np.clip(np.round(float(X[best_idx, 0])), args.layers_min, args.layers_max))
    best_neurons = int(np.clip(np.round(float(X[best_idx, 1])), args.neurons_min, args.neurons_max))

    write_search_population_csv(method_dir / "search_population.csv", X, F)
    result = {
        "best_layers": best_layers,
        "best_neurons": best_neurons,
        "best_proxy_loss": float(F[best_idx, 0]),
        "best_param_count": float(F[best_idx, 1]),
        "population_size": int(F.shape[0]),
    }
    save_json(method_dir / "search_result.json", result)
    return result
