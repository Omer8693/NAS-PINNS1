import argparse
import os

import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from .naspinn import NASPINNBurgers2D, run_single, sample_points_uniform, train_model
from .profiles import apply_profile, parse_slice_times


class Burgers2DNSGA2Problem(ElementwiseProblem):
    def __init__(self, args):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=np.array([4, 64]), xu=np.array([8, 256]))
        self.args = args
        self.eval_count = 0

    def _evaluate(self, x, out, *args, **kwargs):
        self.eval_count += 1
        layers = int(np.clip(np.round(float(x[0])), 4, 8))
        neurons = int(np.clip(np.round(float(x[1])), 64, 256))

        seed = self.args.seed + self.eval_count
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = NASPINNBurgers2D(layers=layers, base_neurons=neurons).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        points = sample_points_uniform(
            train_nt=self.args.train_nt,
            train_nx=self.args.train_nx,
            train_ny=self.args.train_ny,
        )
        train_info = train_model(
            model,
            points,
            epochs=self.args.proxy_epochs,
            skip_lbfgs=True,
            use_pso=False,
            return_stage_info=True,
        )
        proxy_loss = float(train_info["stage_losses"]["adam"])
        n_params = float(sum(p.numel() for p in model.parameters()))
        out["F"] = [proxy_loss, n_params]


def run_search(args):
    print("Starting NSGA-II architecture search (2D Burgers)")
    problem = Burgers2DNSGA2Problem(args)
    algorithm = NSGA2(pop_size=args.pop_size)
    res = minimize(problem, algorithm, termination=("n_gen", args.n_gen), seed=args.seed, verbose=True)

    best_idx = int(np.argmin(res.F[:, 0]))
    best_layers = int(np.clip(np.round(float(res.X[best_idx, 0])), 4, 8))
    best_neurons = int(np.clip(np.round(float(res.X[best_idx, 1])), 64, 256))
    return best_layers, best_neurons, res


def save_search_summary(save_dir, best_layers, best_neurons, res):
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, "search_summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("best_layers,best_neurons,best_proxy_loss,best_param_count\n")
        best_idx = int(np.argmin(res.F[:, 0]))
        f.write(f"{best_layers},{best_neurons},{float(res.F[best_idx,0]):.8e},{float(res.F[best_idx,1]):.1f}\n")
    print(f"Saved summary: {summary_path}")


def run_single_case(args, seed, out_dir):
    best_layers, best_neurons, res = run_search(args)
    print(f"Best architecture from NSGA-II: layers={best_layers}, neurons={best_neurons}")

    args_local = argparse.Namespace(**vars(args))
    args_local.seed = seed
    args_local.layers = best_layers
    args_local.base_neurons = best_neurons
    args_local.save_dir = out_dir

    rel_l2, run_time = run_single(args_local)
    save_search_summary(out_dir, best_layers, best_neurons, res)
    return rel_l2, run_time


def run_paper_protocol(args):
    rows = []
    for run_id in range(1, args.repeats + 1):
        seed = args.seed + run_id - 1
        out_dir = os.path.join(args.save_dir, f"run_{run_id:02d}")
        rel_l2, run_time = run_single_case(args, seed=seed, out_dir=out_dir)
        rows.append((run_id, rel_l2, run_time))

    out_csv = os.path.join(args.save_dir, "paper_protocol_summary.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("run,rel_l2,run_time_seconds\n")
        for run_id, rel_l2, run_time in rows:
            f.write(f"{run_id},{rel_l2:.8e},{run_time:.6f}\n")
        mean_l2 = float(np.mean([r[1] for r in rows]))
        std_l2 = float(np.std([r[1] for r in rows]))
        f.write(f"mean,{mean_l2:.8e},-\n")
        f.write(f"std,{std_l2:.8e},-\n")
    print(f"Saved summary: {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="2D Burgers NAS-PINN with NSGA-II")
    parser.add_argument("--profile", type=str, choices=["paper_baseline", "ours_fast"], default="ours_fast")
    parser.add_argument("--paper-protocol", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None)

    parser.add_argument("--stage", type=str, choices=["adam", "lbfgs", "pso"], default="lbfgs")
    parser.add_argument("--skip-lbfgs", action="store_true")
    parser.add_argument("--use-pso", action="store_true")
    parser.add_argument("--pso-iters", type=int, default=8)
    parser.add_argument("--pso-swarm", type=int, default=16)
    parser.add_argument("--pso-span", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=12000)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--base-neurons", type=int, default=128)
    parser.add_argument("--train-nt", type=int, default=20)
    parser.add_argument("--train-nx", type=int, default=25)
    parser.add_argument("--train-ny", type=int, default=25)
    parser.add_argument("--test-nt", type=int, default=41)
    parser.add_argument("--test-nx", type=int, default=500)
    parser.add_argument("--test-ny", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--slice-grid", type=int, default=200)
    parser.add_argument("--slice-times", type=str, default="0,1,2")

    parser.add_argument("--proxy-epochs", type=int, default=200)
    parser.add_argument("--pop-size", type=int, default=12)
    parser.add_argument("--n-gen", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    args = apply_profile(args, method_name="nsga2")
    args.slice_times = parse_slice_times(args.slice_times)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.paper_protocol:
        run_paper_protocol(args)
    else:
        run_single_case(args, seed=args.seed, out_dir=args.save_dir)


if __name__ == "__main__":
    main()
