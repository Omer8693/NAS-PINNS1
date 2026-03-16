"""CLI entrypoint for NAS-PINN experiments."""

import argparse
import os

from src.baseline_data import BaselinePaper
from src.experiment_runner import (
    AVAILABLE_PROBLEMS,
    DEFAULT_PROBLEM_NAME,
    TIME_MODES,
    compare_all_optimizers,
    run_experiment,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="NAS-PINN experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--compare_all", action="store_true", help="Run all 3 optimizers and compare")
    parser.add_argument("--baseline_only", action="store_true", help="Only export paper baseline plots")
    parser.add_argument(
        "--problem",
        type=str,
        default=DEFAULT_PROBLEM_NAME,
        choices=list(AVAILABLE_PROBLEMS),
        help="Problem module to run on top of the NAS-PINN core",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="nsga2",
        choices=["nsga2", "nsga3", "bayesian"],
        help="NAS optimizer to use",
    )
    parser.add_argument(
        "--time_mode",
        type=str,
        default="full",
        choices=list(TIME_MODES),
        help="Temporal training mode for quenching: full, fixed, adaptive",
    )
    parser.add_argument(
        "--fixed_skip",
        type=int,
        default=2,
        help="Stride used when --time_mode fixed is selected",
    )
    parser.add_argument("--adam_epochs", type=int, default=10000)
    parser.add_argument("--lbfgs_max_iter", type=int, default=5000)
    parser.add_argument("--lbfgs_history_size", type=int, default=20)
    parser.add_argument("--lbfgs_tolerance_grad", type=float, default=1e-7)
    parser.add_argument("--lbfgs_tolerance_change", type=float, default=1e-9)
    parser.add_argument("--pso_steps", type=int, default=50)
    parser.add_argument("--pso_particles", type=int, default=20)
    parser.add_argument("--run_lbfgs", action="store_true", default=False)
    parser.add_argument("--run_pso", action="store_true", default=False)
    parser.add_argument("--nas_pop", type=int, default=24, help="NSGA population size (NAS-PINNS1: 24)")
    parser.add_argument("--nas_gen", type=int, default=16, help="NSGA number of generations (NAS-PINNS1: 16)")
    parser.add_argument("--nas_calls", type=int, default=16, help="Bayesian total calls: init_points=4 + n_iter=12 (NAS-PINNS1: 16)")
    parser.add_argument("--nas_epochs", type=int, default=300, help="Proxy-train epochs for NAS evaluation (NAS-PINNS1: 300)")
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.baseline_only:
        baseline = BaselinePaper()
        baseline.summary()
        baseline_dir = os.path.join(args.output_dir, "baseline")
        os.makedirs(baseline_dir, exist_ok=True)
        for dataset in ["fig15_ac", "fig17_ht", "fig7_water_temp", "table1_a356", "fig16_creep"]:
            baseline.plot_baseline(dataset, baseline_dir)
        baseline.save_all_to_json(os.path.join(baseline_dir, "baseline_data.json"))
        print(f"\n  Baseline plots saved: {baseline_dir}/")
    elif args.compare_all:
        compare_all_optimizers(
            adam_epochs=args.adam_epochs,
            lbfgs_max_iter=args.lbfgs_max_iter,
            lbfgs_history_size=args.lbfgs_history_size,
            lbfgs_tolerance_grad=args.lbfgs_tolerance_grad,
            lbfgs_tolerance_change=args.lbfgs_tolerance_change,
            pso_steps=args.pso_steps,
            pso_particles=args.pso_particles,
            run_lbfgs=args.run_lbfgs,
            run_pso=args.run_pso,
            nas_pop=args.nas_pop,
            nas_gen=args.nas_gen,
            nas_calls=args.nas_calls,
            nas_epochs=args.nas_epochs,
            output_dir=args.output_dir,
            problem_name=args.problem,
            time_mode=args.time_mode,
            fixed_skip=args.fixed_skip,
        )
    else:
        run_experiment(
            optimizer=args.optimizer,
            problem_name=args.problem,
            time_mode=args.time_mode,
            fixed_skip=args.fixed_skip,
            adam_epochs=args.adam_epochs,
            lbfgs_max_iter=args.lbfgs_max_iter,
            lbfgs_history_size=args.lbfgs_history_size,
            lbfgs_tolerance_grad=args.lbfgs_tolerance_grad,
            lbfgs_tolerance_change=args.lbfgs_tolerance_change,
            pso_steps=args.pso_steps,
            pso_particles=args.pso_particles,
            run_lbfgs=args.run_lbfgs,
            run_pso=args.run_pso,
            nas_pop=args.nas_pop,
            nas_gen=args.nas_gen,
            nas_calls=args.nas_calls,
            nas_epochs=args.nas_epochs,
            output_dir=args.output_dir,
        )
