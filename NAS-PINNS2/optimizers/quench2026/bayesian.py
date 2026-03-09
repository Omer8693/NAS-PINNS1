import argparse
from pathlib import Path

import numpy as np

from .common import add_shared_args, evaluate_architecture, run_method, save_json

try:
    from bayes_opt import BayesianOptimization
except ImportError as exc:
    raise ImportError("Missing dependency: bayes_opt. Install with 'pip install bayesian-optimization'.") from exc


def run_search(args, method_dir: Path, log_path: Path):
    eval_counter = {"k": 0}
    best = {"layers": None, "neurons": None, "proxy_loss": float("inf"), "param_count": float("nan")}
    rows = []

    def objective(layers, neurons):
        eval_counter["k"] += 1
        l_int = int(np.clip(np.round(float(layers)), args.layers_min, args.layers_max))
        n_int = int(np.clip(np.round(float(neurons)), args.neurons_min, args.neurons_max))
        proxy_loss, param_count, _ = evaluate_architecture(
            args,
            method_dir,
            l_int,
            n_int,
            eval_counter["k"],
            log_path,
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

    with open(method_dir / "search_population.csv", "w", encoding="utf-8") as f:
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


def parse_args():
    parser = argparse.ArgumentParser(description="Quench2026 NAS-PINN with Bayesian architecture search")
    add_shared_args(parser, default_save_dir="results/quench2026/bayesian")
    parser.add_argument("--bo-init-points", type=int, default=4)
    parser.add_argument("--bo-iters", type=int, default=12)
    return parser.parse_args()


def run(args):
    return run_method("bayesian", args, run_search)


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
