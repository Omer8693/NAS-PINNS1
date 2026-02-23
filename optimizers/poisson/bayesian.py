import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

try:
    from bayes_opt import BayesianOptimization
except ImportError as exc:
    raise ImportError("Missing dependency: bayes_opt. Install with 'pip install bayesian-optimization'.") from exc

from optimizers.poisson.common import (
    device,
    lambda_pde,
    lambda_bc,
    finalize_plot,
    set_seed,
    sample_points,
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


class BayesianPoissonPINN(nn.Module):
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


def decode_architecture(params):
    n_layers = int(round(params["n_layers"]))
    widths_all = [int(round(params["n1"])), int(round(params["n2"])), int(round(params["n3"])), int(round(params["n4"]))]
    acts_all = [
        "sin" if params["a1"] >= 0.5 else "tanh",
        "sin" if params["a2"] >= 0.5 else "tanh",
        "sin" if params["a3"] >= 0.5 else "tanh",
        "sin" if params["a4"] >= 0.5 else "tanh",
    ]
    return widths_all[:n_layers], acts_all[:n_layers], float(params["lr"])


def architecture_signature(widths, acts):
    return " | ".join([f"{a}-{w}" for w, a in zip(widths, acts)])


def run_bayesian_search(train_points, args):
    eval_counter = {"k": 0}

    def objective(n_layers, n1, n2, n3, n4, a1, a2, a3, a4, lr):
        eval_counter["k"] += 1
        params = {
            "n_layers": n_layers,
            "n1": n1,
            "n2": n2,
            "n3": n3,
            "n4": n4,
            "a1": a1,
            "a2": a2,
            "a3": a3,
            "a4": a4,
            "lr": lr,
        }
        widths, acts, lr_decoded = decode_architecture(params)
        set_seed(args.seed + eval_counter["k"])

        model = BayesianPoissonPINN(widths, acts).to(device)
        train_model(model, train_points, epochs=args.bo_epochs, lr=lr_decoded, skip_lbfgs=True)

        rel_l2, _, _, _, _ = predict_on_grid(model, test_nx=args.test_nx, test_ny=args.test_ny)
        print(f"BO eval {eval_counter['k']:02d} | widths={widths} acts={acts} lr={lr_decoded:.2e} | relL2={rel_l2:.4e}")
        return -float(rel_l2)

    pbounds = {
        "n_layers": (2, 4),
        "n1": (32, 256), "n2": (32, 256), "n3": (32, 256), "n4": (32, 256),
        "a1": (0, 1), "a2": (0, 1), "a3": (0, 1), "a4": (0, 1),
        "lr": (5e-4, 2e-3),
    }

    bo = BayesianOptimization(f=objective, pbounds=pbounds, random_state=args.seed, verbose=2)
    bo.maximize(init_points=args.bo_init_points, n_iter=args.bo_iters)

    best_widths, best_acts, best_lr = decode_architecture(bo.max["params"])
    return best_widths, best_acts, best_lr, -bo.max["target"]


def plot_results(model, save_dir):
    rel_l2 = plot_poisson_results(
        model,
        predict_on_grid,
        os.path.join(save_dir, "bayes_poisson_results.png"),
        pred_title="Predicted (Bayesian NAS)",
    )
    print(f"Final relative L2 error: {rel_l2:.4e}")
    return float(rel_l2)


def run_single(args):
    run_start = time.perf_counter()
    set_seed(args.seed)
    train_points = sample_points()

    best_widths, best_acts, best_lr, bo_rel_l2 = run_bayesian_search(train_points, args)
    print("\nBest architecture from BO")
    print(f"hidden widths: {best_widths}")
    print(f"hidden acts  : {best_acts}")
    print(f"lr           : {best_lr:.4e}")
    print(f"BO rel L2    : {bo_rel_l2:.4e}")

    best_model = BayesianPoissonPINN(best_widths, best_acts).to(device)
    loss_history = train_model(
        best_model,
        train_points,
        epochs=args.epochs,
        lr=best_lr,
        skip_lbfgs=args.skip_lbfgs,
        track_history=True,
    )

    ckpt_path = os.path.join(args.save_dir, args.checkpoint)
    torch.save(
        {
            "model_state": best_model.state_dict(),
            "hidden_widths": best_widths,
            "hidden_acts": best_acts,
            "lr": best_lr,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")
    plot_loss_curve(
        loss_history,
        os.path.join(args.save_dir, "bayes_poisson_loss_curve.png"),
        title="Poisson Bayesian Training Loss",
    )
    rel_l2 = plot_results(best_model, args.save_dir)
    run_time = time.perf_counter() - run_start
    with open(os.path.join(args.save_dir, "run_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"run_time_seconds,{run_time:.6f}\n")
    print(f"Run time: {run_time:.2f} s")
    return float(rel_l2), float(run_time)


def run_paper_protocol(args):
    train_points = sample_points_protocol(train_nx=args.train_nx, train_ny=args.train_ny, boundary_n=args.boundary_n)

    run_errors = []
    last_arch = ""
    for run_id in range(1, args.repeats + 1):
        args_local = argparse.Namespace(**vars(args))
        args_local.seed = args.seed + run_id

        best_widths, best_acts, best_lr, _ = run_bayesian_search(train_points, args_local)
        last_arch = architecture_signature(best_widths, best_acts)

        model = BayesianPoissonPINN(best_widths, best_acts).to(device)
        train_model(model, train_points, epochs=args.epochs, lr=best_lr, skip_lbfgs=args.skip_lbfgs)

        rel_l2, _, _, _, _ = predict_on_grid(model, test_nx=args.test_nx, test_ny=args.test_ny)
        run_errors.append(rel_l2)
        print(f"run={run_id} relL2={rel_l2:.4e}")

    mean_l2 = float(np.mean(run_errors))
    std_l2 = float(np.std(run_errors))

    out_csv = os.path.join(args.save_dir, "bayes_poisson_paper_protocol_summary.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("name,architecture,run,rel_l2\n")
        for idx, err in enumerate(run_errors, start=1):
            f.write(f"Bayes-NAS,\"{last_arch}\",{idx},{err:.8e}\n")
        f.write(f"Bayes-NAS,\"{last_arch}\",mean,{mean_l2:.8e}\n")
        f.write(f"Bayes-NAS,\"{last_arch}\",std,{std_l2:.8e}\n")
    print(f"Saved summary: {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Poisson NAS-PINN with Bayesian Optimization")
    parser.add_argument("--epochs", type=int, default=12000, help="final training epochs")
    parser.add_argument("--skip-lbfgs", action="store_true", help="skip L-BFGS in final training")
    parser.add_argument("--save-dir", type=str, default="results/poisson/bayesian", help="output directory")
    parser.add_argument("--checkpoint", type=str, default="bayes_poisson_checkpoint_last.pth", help="checkpoint filename")

    parser.add_argument("--bo-init-points", type=int, default=2, help="BO initial random points")
    parser.add_argument("--bo-iters", type=int, default=8, help="BO guided iterations")
    parser.add_argument("--bo-epochs", type=int, default=1200, help="training epochs per BO evaluation")

    parser.add_argument("--paper-protocol", action="store_true", help="paper-like repeated protocol")
    parser.add_argument("--repeats", type=int, default=5, help="number of repeats")
    parser.add_argument("--train-nx", type=int, default=100)
    parser.add_argument("--train-ny", type=int, default=100)
    parser.add_argument("--boundary-n", type=int, default=200)
    parser.add_argument("--test-nx", type=int, default=150)
    parser.add_argument("--test-ny", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seed", action="store_true", help="run three seed values and save comparison")
    parser.add_argument("--seed-list", type=str, default="42,43,44", help="comma-separated seeds for --multi-seed")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.multi_seed:
        seeds = [int(v.strip()) for v in args.seed_list.split(",") if v.strip()]
        summary = []
        base_dir = args.save_dir

        for seed_val in seeds:
            run_dir = os.path.join(base_dir, f"seed_{seed_val}")
            os.makedirs(run_dir, exist_ok=True)

            args_local = argparse.Namespace(**vars(args))
            args_local.seed = seed_val
            args_local.save_dir = run_dir
            rel_l2, run_time = run_single(args_local)
            summary.append((seed_val, rel_l2, run_time))

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
        plt.title("Poisson Bayesian: Seed Comparison")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        finalize_plot(plt, os.path.join(base_dir, "seed_comparison.png"))
        return

    if args.paper_protocol:
        run_paper_protocol(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
