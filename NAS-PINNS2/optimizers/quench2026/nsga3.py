import argparse
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from .common import (
    add_shared_args,
    evaluate_architecture,
    run_method,
    save_json,
    write_search_population_csv,
)


class Quench2026NSGA3Problem(ElementwiseProblem):
    def __init__(self, args, method_dir: Path, log_path: Path):
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=0,
            xl=np.array([args.layers_min, args.neurons_min], dtype=np.float64),
            xu=np.array([args.layers_max, args.neurons_max], dtype=np.float64),
        )
        self.args = args
        self.method_dir = method_dir
        self.log_path = log_path
        self.eval_count = 0

    def _evaluate(self, x, out, *args, **kwargs):
        self.eval_count += 1
        layers = int(np.clip(np.round(float(x[0])), self.args.layers_min, self.args.layers_max))
        neurons = int(np.clip(np.round(float(x[1])), self.args.neurons_min, self.args.neurons_max))
        proxy_loss, param_count, _ = evaluate_architecture(
            self.args,
            self.method_dir,
            layers,
            neurons,
            self.eval_count,
            self.log_path,
        )
        out["F"] = [proxy_loss, param_count]


def run_search(args, method_dir: Path, log_path: Path):
    problem = Quench2026NSGA3Problem(args, method_dir, log_path)
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=args.ref_partitions)
    pop_size = max(args.pop_size, len(ref_dirs))
    algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=pop_size)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Quench2026 NAS-PINN with NSGA-III architecture search")
    add_shared_args(parser, default_save_dir="results/quench2026/nsga3")
    parser.add_argument("--pop-size", type=int, default=24)
    parser.add_argument("--n-gen", type=int, default=16)
    parser.add_argument("--ref-partitions", type=int, default=10)
    return parser.parse_args()


def run(args):
    return run_method("nsga3", args, run_search)


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
