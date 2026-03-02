import argparse
import os

import numpy as np
import torch
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from .naspinn import NASPINNAdvection, run_single, sample_points_uniform, train_model
from .profiles import apply_profile, parse_beta_list


class AdvectionNSGA3Problem(ElementwiseProblem):
    def __init__(self, beta, args):
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=0,
            xl=np.array([args.search_layers_min, args.search_neurons_min]),
            xu=np.array([args.search_layers_max, args.search_neurons_max]),
        )
        self.beta = beta
        self.args = args
        self.eval_count = 0

    def _evaluate(self, x, out, *args, **kwargs):
        self.eval_count += 1
        layers = int(
            np.clip(
                np.round(float(x[0])),
                self.args.search_layers_min,
                self.args.search_layers_max,
            )
        )
        neurons = int(
            np.clip(
                np.round(float(x[1])),
                self.args.search_neurons_min,
                self.args.search_neurons_max,
            )
        )

        seed = self.args.seed + self.eval_count
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = NASPINNAdvection(layers=layers, base_neurons=neurons).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        points = sample_points_uniform(nt=self.args.train_nt, nx=self.args.train_nx)
        train_info = train_model(
            model,
            points,
            beta=self.beta,
            epochs=self.args.proxy_epochs,
            skip_lbfgs=True,
            use_pso=False,
            return_stage_info=True,
        )
        proxy_loss = float(train_info["stage_losses"]["adam"])
        n_params = float(sum(p.numel() for p in model.parameters()))
        out["F"] = [proxy_loss, n_params]


def run_search(beta, args):
    print(f"Starting NSGA-III architecture search: beta={beta:.3f}")
    problem = AdvectionNSGA3Problem(beta, args)
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=args.ref_partitions)
    pop_size = max(args.pop_size, len(ref_dirs))
    algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=pop_size)
    res = minimize(problem, algorithm, termination=("n_gen", args.n_gen), seed=args.seed, verbose=True)

    best_idx = int(np.argmin(res.F[:, 0]))
    best_layers = int(
        np.clip(
            np.round(float(res.X[best_idx, 0])),
            args.search_layers_min,
            args.search_layers_max,
        )
    )
    best_neurons = int(
        np.clip(
            np.round(float(res.X[best_idx, 1])),
            args.search_neurons_min,
            args.search_neurons_max,
        )
    )
    return best_layers, best_neurons, res


def save_search_summary(save_dir, beta, best_layers, best_neurons, res):
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, "search_summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("beta,best_layers,best_neurons,best_proxy_loss,best_param_count\n")
        best_idx = int(np.argmin(res.F[:, 0]))
        f.write(
            f"{beta:.6f},{best_layers},{best_neurons},{float(res.F[best_idx,0]):.8e},{float(res.F[best_idx,1]):.1f}\n"
        )
    print(f"Saved summary: {summary_path}")


def run_single_beta(args, beta, seed, out_dir):
    best_layers, best_neurons, res = run_search(beta, args)
    print(f"Best architecture from NSGA-III: layers={best_layers}, neurons={best_neurons}")

    args_local = argparse.Namespace(**vars(args))
    args_local.beta = beta
    args_local.seed = seed
    args_local.layers = best_layers
    args_local.base_neurons = best_neurons
    args_local.save_dir = out_dir

    rel_l2, run_time = run_single(args_local)
    save_search_summary(out_dir, beta, best_layers, best_neurons, res)
    return rel_l2, run_time


def run_multi_beta(args):
    beta_values = parse_beta_list(args.beta_list)
    summary = []
    os.makedirs(args.save_dir, exist_ok=True)

    for idx, beta_val in enumerate(beta_values):
        seed = args.seed + idx
        case_dir = os.path.join(args.save_dir, f"beta_{beta_val:.3f}")
        rel_l2, run_time = run_single_beta(args, beta_val, seed=seed, out_dir=case_dir)
        summary.append((beta_val, rel_l2, run_time))

    csv_path = os.path.join(args.save_dir, "beta_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("beta,rel_l2,run_time_seconds\n")
        for beta_val, rel_l2, run_time in summary:
            f.write(f"{beta_val:.6f},{rel_l2:.8e},{run_time:.6f}\n")
    print(f"Saved summary: {csv_path}")


def run_paper_protocol(args):
    beta_values = parse_beta_list(args.paper_betas)
    rows = []
    for beta in beta_values:
        run_errors = []
        for run_id in range(1, args.repeats + 1):
            seed = args.seed + run_id
            out_dir = os.path.join(args.save_dir, f"paper_beta_{beta:.3f}", f"run_{run_id:02d}")
            rel_l2, _ = run_single_beta(args, beta, seed=seed, out_dir=out_dir)
            run_errors.append(rel_l2)
        rows.append((beta, float(np.mean(run_errors)), float(np.std(run_errors))))

    out_csv = os.path.join(args.save_dir, "paper_protocol_summary.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("beta,mean_rel_l2,std_rel_l2\n")
        for beta, mean_l2, std_l2 in rows:
            f.write(f"{beta:.6f},{mean_l2:.8e},{std_l2:.8e}\n")
    print(f"Saved summary: {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Advection NAS-PINN with NSGA-III")
    parser.add_argument("--profile", type=str, choices=["paper_baseline", "ours_fast"], default="paper_baseline")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--multi-beta", action="store_true")
    parser.add_argument("--beta-list", type=str, default="1.0,0.5,0.1")
    parser.add_argument("--paper-protocol", action="store_true")
    parser.add_argument("--paper-betas", type=str, default="1.0,0.5,0.1")
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
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--base-neurons", type=int, default=128)
    parser.add_argument("--train-nt", type=int, default=40)
    parser.add_argument("--train-nx", type=int, default=120)
    parser.add_argument("--test-nt", type=int, default=40)
    parser.add_argument("--test-nx", type=int, default=120)
    parser.add_argument("--slice-times", type=str, default="0,0.5,1.0,1.5,2.0")

    parser.add_argument("--proxy-epochs", type=int, default=300)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--n-gen", type=int, default=20)
    parser.add_argument("--ref-partitions", type=int, default=12)
    parser.add_argument("--bo-init-points", type=int, default=4)
    parser.add_argument("--bo-iters", type=int, default=12)
    parser.add_argument("--search-layers-min", type=int, default=3)
    parser.add_argument("--search-layers-max", type=int, default=6)
    parser.add_argument("--search-neurons-min", type=int, default=64)
    parser.add_argument("--search-neurons-max", type=int, default=192)
    return parser.parse_args()


def main():
    args = parse_args()
    args = apply_profile(args, method_name="nsga3")
    args.slice_times = [float(v.strip()) for v in str(args.slice_times).split(",") if v.strip()]
    os.makedirs(args.save_dir, exist_ok=True)

    if args.paper_protocol:
        run_paper_protocol(args)
    elif args.multi_beta:
        run_multi_beta(args)
    else:
        run_single_beta(args, beta=args.beta, seed=args.seed, out_dir=args.save_dir)


if __name__ == "__main__":
    main()
