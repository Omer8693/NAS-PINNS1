import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
except ImportError as exc:
    raise ImportError("Missing dependency: pymoo. Install with 'pip install pymoo'.") from exc

from optimizers.poisson.common import (
    device,
    lambda_pde,
    lambda_bc,
    finalize_plot,
    set_seed,
    sample_points_protocol,
    pde_loss,
    bc_loss,
    predict_on_grid,
    plot_loss_curve,
)
from optimizers.poisson.plots import plot_poisson_results


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class FixedPoissonPINN(nn.Module):
    def __init__(self, hidden_widths, hidden_acts):
        super().__init__()
        layers = []
        in_dim = 2
        for width, act_name in zip(hidden_widths, hidden_acts):
            layers.append(nn.Linear(in_dim, width))
            layers.append(SinActivation() if act_name == "sin" else nn.Tanh())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)


def train_model(model, points, epochs=12000, lr=1e-3, skip_lbfgs=False, track_history=False):
    (x_col, y_col), (x_bc, y_bc) = points
    loss_history = []

    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        lp = pde_loss(model, x_col, y_col)
        lb = bc_loss(model, x_bc, y_bc)
        loss = lambda_pde * lp + lambda_bc * lb
        loss.backward()
        opt.step()
        if track_history:
            loss_history.append(float(loss.detach().cpu().item()))

    if not skip_lbfgs:
        lbfgs = optim.LBFGS(model.parameters(), lr=0.8, max_iter=2000, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            lp = pde_loss(model, x_col, y_col)
            lb = bc_loss(model, x_bc, y_bc)
            loss = lambda_pde * lp + lambda_bc * lb
            loss.backward()
            if track_history:
                loss_history.append(float(loss.detach().cpu().item()))
            return loss

        lbfgs.step(closure)

    return loss_history


class PoissonNSGA2Problem(ElementwiseProblem):
    def __init__(self, train_points, test_nx, test_ny, eval_epochs, max_hidden_layers):
        self.train_points = train_points
        self.test_nx = test_nx
        self.test_ny = test_ny
        self.eval_epochs = eval_epochs
        self.max_hidden_layers = max_hidden_layers

        super().__init__(
            n_var=2 * max_hidden_layers,
            n_obj=2,
            n_ieq_constr=0,
            xl=np.array([32] * max_hidden_layers + [0] * max_hidden_layers, dtype=float),
            xu=np.array([256] * max_hidden_layers + [1] * max_hidden_layers, dtype=float),
        )

    def decode(self, x):
        widths = [int(round(v)) for v in x[: self.max_hidden_layers]]
        acts = ["sin" if v >= 0.5 else "tanh" for v in x[self.max_hidden_layers :]]
        return widths, acts

    def _evaluate(self, x, out, *args, **kwargs):
        widths, acts = self.decode(x)
        model = FixedPoissonPINN(widths, acts).to(device)
        train_model(model, self.train_points, epochs=self.eval_epochs, lr=1e-3, skip_lbfgs=True)

        rel_l2, _, _, _, _ = predict_on_grid(model, test_nx=self.test_nx, test_ny=self.test_ny)
        complexity = float(sum(widths))
        out["F"] = [float(rel_l2), complexity]


def run_nsga2_search(train_points, args):
    problem = PoissonNSGA2Problem(
        train_points=train_points,
        test_nx=args.test_nx,
        test_ny=args.test_ny,
        eval_epochs=args.search_eval_epochs,
        max_hidden_layers=args.max_hidden_layers,
    )

    algorithm = NSGA2(pop_size=args.search_pop)
    termination = get_termination("n_gen", args.search_gen)

    res = minimize(problem, algorithm, termination, seed=args.seed, save_history=False, verbose=True)

    Fs = res.F
    Xs = res.X
    idx_best = int(np.argmin(Fs[:, 0]))
    widths, acts = problem.decode(Xs[idx_best])

    return {
        "pareto_F": Fs,
        "pareto_X": Xs,
        "best_widths": widths,
        "best_acts": acts,
        "best_rel_l2": float(Fs[idx_best, 0]),
    }


def plot_results(model, save_dir):
    rel_l2 = plot_poisson_results(
        model,
        predict_on_grid,
        os.path.join(save_dir, "poisson_nsga2_results.png"),
        pred_title="Predicted (NSGA-II)",
    )
    print(f"Final relative L2 error: {rel_l2:.4e}")
    return float(rel_l2)


def parse_args():
    parser = argparse.ArgumentParser(description="Poisson NAS-PINN with NSGA-II")
    parser.add_argument("--epochs", type=int, default=12000, help="final training epochs")
    parser.add_argument("--skip-lbfgs", action="store_true", help="skip L-BFGS in final training")
    parser.add_argument("--save-dir", type=str, default="results/poisson/nsga2", help="output directory")
    parser.add_argument("--checkpoint", type=str, default="poisson_checkpoint_last_nsga2.pth", help="checkpoint filename")

    parser.add_argument("--search-pop", type=int, default=16)
    parser.add_argument("--search-gen", type=int, default=8)
    parser.add_argument("--search-eval-epochs", type=int, default=1200)
    parser.add_argument("--max-hidden-layers", type=int, default=4)

    parser.add_argument("--train-nx", type=int, default=100)
    parser.add_argument("--train-ny", type=int, default=100)
    parser.add_argument("--boundary-n", type=int, default=200)
    parser.add_argument("--test-nx", type=int, default=150)
    parser.add_argument("--test-ny", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seed", action="store_true", help="run three seed values and save comparison")
    parser.add_argument("--seed-list", type=str, default="42,43,44", help="comma-separated seeds for --multi-seed")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, args.checkpoint)

    if args.multi_seed:
        seeds = [int(v.strip()) for v in args.seed_list.split(",") if v.strip()]
        summary = []
        base_dir = args.save_dir

        for seed_val in seeds:
            run_start = time.perf_counter()
            run_dir = os.path.join(base_dir, f"seed_{seed_val}")
            os.makedirs(run_dir, exist_ok=True)

            args_local = argparse.Namespace(**vars(args))
            args_local.seed = seed_val
            args_local.save_dir = run_dir

            set_seed(seed_val)
            train_points = sample_points_protocol(
                train_nx=args.train_nx,
                train_ny=args.train_ny,
                boundary_n=args.boundary_n,
            )

            search_out = run_nsga2_search(train_points, args_local)
            model = FixedPoissonPINN(search_out["best_widths"], search_out["best_acts"]).to(device)
            loss_history = train_model(
                model,
                train_points,
                epochs=args.epochs,
                lr=1e-3,
                skip_lbfgs=args.skip_lbfgs,
                track_history=True,
            )

            ckpt_local = os.path.join(run_dir, args.checkpoint)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_widths": search_out["best_widths"],
                    "best_acts": search_out["best_acts"],
                    "pareto_F": search_out["pareto_F"],
                    "pareto_X": search_out["pareto_X"],
                },
                ckpt_local,
            )
            plot_loss_curve(
                loss_history,
                os.path.join(run_dir, "poisson_nsga2_loss_curve.png"),
                title=f"Poisson NSGA-II Training Loss (seed={seed_val})",
            )
            rel_l2 = plot_results(model, run_dir)
            run_time = time.perf_counter() - run_start
            with open(os.path.join(run_dir, "run_time.txt"), "w", encoding="utf-8") as f:
                f.write(f"run_time_seconds,{run_time:.6f}\n")
            summary.append((seed_val, rel_l2, run_time))
            print(f"Run time (seed={seed_val}): {run_time:.2f} s")

        out_csv = os.path.join(base_dir, "seed_comparison.csv")
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("seed,rel_l2,run_time_seconds\n")
            for seed_val, rel_l2, run_time in summary:
                f.write(f"{seed_val},{rel_l2:.8e},{run_time:.6f}\n")
        print(f"Saved summary: {out_csv}")

        plt.figure(figsize=(7, 4))
        plot_seeds = []
        plot_l2s = []
        for item in summary:
            if isinstance(item, tuple):
                plot_seeds.append(item[0])
                plot_l2s.append(item[1])
            elif isinstance(item, str):
                parts = item.strip().split(",")
                if len(parts) >= 2:
                    plot_seeds.append(parts[0])
                    plot_l2s.append(float(parts[1]))
        plt.plot(plot_seeds, plot_l2s, marker="o", linewidth=2)
        plt.yscale("log")
        plt.xlabel("Seed")
        plt.ylabel("Relative L2 Error")
        plt.title("Poisson NSGA-II: Seed Comparison")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        finalize_plot(plt, os.path.join(base_dir, "seed_comparison.png"))
        return

    if args.plot_only:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model = FixedPoissonPINN(ckpt["best_widths"], ckpt["best_acts"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        plot_results(model, args.save_dir)
        return

    run_start = time.perf_counter()
    set_seed(args.seed)
    train_points = sample_points_protocol(train_nx=args.train_nx, train_ny=args.train_ny, boundary_n=args.boundary_n)

    search_out = run_nsga2_search(train_points, args)
    print("\nNSGA-II selected architecture")
    print(f"widths: {search_out['best_widths']}")
    print(f"acts:   {search_out['best_acts']}")
    print(f"best relL2 (search): {search_out['best_rel_l2']:.4e}")

    model = FixedPoissonPINN(search_out["best_widths"], search_out["best_acts"]).to(device)
    loss_history = train_model(
        model,
        train_points,
        epochs=args.epochs,
        lr=1e-3,
        skip_lbfgs=args.skip_lbfgs,
        track_history=True,
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "best_widths": search_out["best_widths"],
            "best_acts": search_out["best_acts"],
            "pareto_F": search_out["pareto_F"],
            "pareto_X": search_out["pareto_X"],
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")

    plot_loss_curve(
        loss_history,
        os.path.join(args.save_dir, "poisson_nsga2_loss_curve.png"),
        title="Poisson NSGA-II Training Loss",
    )

    plot_results(model, args.save_dir)
    run_time = time.perf_counter() - run_start
    with open(os.path.join(args.save_dir, "run_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"run_time_seconds,{run_time:.6f}\n")
    print(f"Run time: {run_time:.2f} s")


if __name__ == "__main__":
    main()
