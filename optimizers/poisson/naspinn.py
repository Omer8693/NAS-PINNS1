import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from optimizers.poisson.common import (
    device,
    lambda_pde,
    lambda_bc,
    finalize_plot,
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


class MixedOp(nn.Module):
    def __init__(self, in_c, out_c, mask_levels=(30, 50, 70, 90, 110)):
        super().__init__()
        self.mask_levels = list(mask_levels)
        self.n_masks = len(self.mask_levels)
        self.ops = nn.ModuleList([
            nn.Identity() if in_c == out_c else nn.Linear(in_c, out_c),
            nn.Sequential(nn.Linear(in_c, out_c), nn.Tanh()),
        ])
        self.n_ops = len(self.ops)
        self.alpha = nn.Parameter(torch.randn(self.n_ops + self.n_masks) * 0.1)

    def relaxed_op(self, x):
        weights = F.softmax(self.alpha[: self.n_ops], dim=0)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

    def forward(self, x):
        mixed = self.relaxed_op(x)
        mask_weights = torch.sigmoid(self.alpha[self.n_ops :])

        final = 0.0
        dim = mixed.shape[-1]
        for j, keep in enumerate(self.mask_levels):
            k = min(keep, dim)
            mask = torch.zeros(dim, device=device)
            mask[:k] = 1.0
            final += mask_weights[j] * (mixed * mask.unsqueeze(0))
        return final


class NAS_PINN(nn.Module):
    def __init__(self, layers=4, base_neurons=110, mask_levels=(30, 50, 70, 90, 110)):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [2] + [base_neurons] * (layers - 1) + [1]
        for i in range(layers):
            self.layers.append(MixedOp(dims[i], dims[i + 1], mask_levels))

    def forward(self, xy):
        out = xy
        for layer in self.layers:
            out = layer(out)
        return out


def save_checkpoint(path, model, opt_inner, opt_outer, epoch, points):
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_inner_state": opt_inner.state_dict() if opt_inner is not None else None,
        "opt_outer_state": opt_outer.state_dict() if opt_outer is not None else None,
        "points": {
            "x_col": points[0].detach().cpu(),
            "y_col": points[1].detach().cpu(),
            "x_bc": points[2].detach().cpu(),
            "y_bc": points[3].detach().cpu(),
        },
    }
    torch.save(payload, path)


def load_checkpoint(path, model, opt_inner=None, opt_outer=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    if opt_inner is not None and ckpt.get("opt_inner_state") is not None:
        opt_inner.load_state_dict(ckpt["opt_inner_state"])
    if opt_outer is not None and ckpt.get("opt_outer_state") is not None:
        opt_outer.load_state_dict(ckpt["opt_outer_state"])

    p = ckpt.get("points")
    if p is not None:
        points = (p["x_col"].to(device), p["y_col"].to(device), p["x_bc"].to(device), p["y_bc"].to(device))
    else:
        (x_col, y_col), (x_bc, y_bc) = sample_points()
        points = (x_col, y_col, x_bc, y_bc)

    return ckpt.get("epoch", 0), points


def train_with_resume(
    model,
    total_epochs=12000,
    inner_lr=1e-3,
    outer_lr=3e-4,
    outer_every=5,
    checkpoint_path="poisson_checkpoint_last.pth",
    checkpoint_every=1000,
    resume=False,
    skip_lbfgs=False,
    fixed_points=None,
):
    opt_inner = optim.Adam(model.parameters(), lr=inner_lr)
    arch_params = [layer.alpha for layer in model.layers]
    opt_outer = optim.Adam(arch_params, lr=outer_lr)

    start_epoch = 0
    if resume and os.path.exists(checkpoint_path):
        start_epoch, points = load_checkpoint(checkpoint_path, model, opt_inner, opt_outer)
        print(f"Resumed from checkpoint: {checkpoint_path} (epoch {start_epoch})")
    else:
        if fixed_points is None:
            (x_col, y_col), (x_bc, y_bc) = sample_points()
        else:
            (x_col, y_col), (x_bc, y_bc) = fixed_points
        points = (x_col, y_col, x_bc, y_bc)

    x_col, y_col, x_bc, y_bc = points
    loss_history = []
    print("Starting Adam + bi-level optimization...")

    # Outer loss: supervised MSE with analytic solution
    def outer_loss(model, x, y):
        x_t = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
        xy = torch.cat([x_t, y_t], dim=1)
        u = model(xy)
        true = torch.cos(np.pi * x_t) * torch.cos(np.pi * y_t)
        return torch.mean((u - true) ** 2)

    x_outer, y_outer = x_col, y_col  # Outer loop için collocation noktaları kullanılıyor

    for epoch in range(start_epoch, total_epochs):
        opt_inner.zero_grad()
        l_pde = pde_loss(model, x_col, y_col)
        l_bc = bc_loss(model, x_bc, y_bc)
        loss_inner = 1.0 * l_pde + 1.0 * l_bc
        loss_inner.backward()
        opt_inner.step()
        loss_history.append(float(loss_inner.detach().cpu().item()))

        if epoch % outer_every == 0:
            opt_outer.zero_grad()
            l_outer = outer_loss(model, x_outer, y_outer)
            l_outer.backward()
            opt_outer.step()

        if epoch % 2000 == 0:
            print(f"[{epoch:5d}] PDE residual: {l_pde:.4e}   BC: {l_bc:.4e}")

        if ((epoch + 1) % checkpoint_every == 0) or (epoch + 1 == total_epochs):
            save_checkpoint(checkpoint_path, model, opt_inner, opt_outer, epoch + 1, (x_col, y_col, x_bc, y_bc))

    if not skip_lbfgs:
        print("\nL-BFGS refinement...")
        lbfgs = optim.LBFGS(model.parameters(), lr=0.8, max_iter=2000, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            lp = pde_loss(model, x_col, y_col)
            lb = bc_loss(model, x_bc, y_bc)
            total = 1.0 * lp + 1.0 * lb
            total.backward()
            return total

        lbfgs.step(closure)
        save_checkpoint(checkpoint_path, model, opt_inner, opt_outer, total_epochs, (x_col, y_col, x_bc, y_bc))

    return loss_history


def architecture_signature(model):
    parts = []
    op_names = ["Identity", "Tanh"]
    for layer in model.layers:
        op_p = F.softmax(layer.alpha[: layer.n_ops], dim=0)
        mask_p = torch.sigmoid(layer.alpha[layer.n_ops :])
        op_idx = torch.argmax(op_p).item()
        mask_idx = torch.argmax(mask_p).item()
        parts.append(f"{op_names[op_idx]}-{layer.mask_levels[mask_idx]}")
    return " | ".join(parts)


def print_discovered_architecture(model):
    print("\nDiscovered architecture (most probable ops & mask):")
    print(architecture_signature(model))


def plot_results(model, save_dir):
    rel_l2 = plot_poisson_results(
        model,
        predict_on_grid,
        os.path.join(save_dir, "poisson_results.png"),
        pred_title="Predicted φ(x,y)",
        suptitle="NAS-PINN Poisson Equation Results",
    )
    print(f"\nRelative L2 error (full grid): {rel_l2:.4e}")
    return float(rel_l2)


def run_protocol(args):
    fixed_points = sample_points_protocol(train_nx=args.train_nx, train_ny=args.train_ny, boundary_n=args.boundary_n)
    summary = []
    last_arch = ""

    print("\nRunning protocol mode for Poisson equation")
    for run_id in range(1, args.repeats + 1):
        torch.manual_seed(args.seed + run_id)
        np.random.seed(args.seed + run_id)

        model = NAS_PINN(layers=3, base_neurons=192).to(device)
        ckpt_path = os.path.join(args.save_dir, f"poisson_protocol_run_{run_id}.pth")
        loss_history = train_with_resume(
            model,
            total_epochs=args.epochs,
            checkpoint_path=ckpt_path,
            resume=False,
            skip_lbfgs=args.skip_lbfgs,
            fixed_points=fixed_points,
        )
        plot_loss_curve(
            loss_history,
            os.path.join(args.save_dir, f"poisson_protocol_run_{run_id}_loss_curve.png"),
            title=f"Poisson NAS-PINN Loss (Run {run_id})",
        )

        rel_l2, _, _, _, _ = predict_on_grid(model, test_nx=args.test_nx, test_ny=args.test_ny)
        summary.append(rel_l2)
        last_arch = architecture_signature(model)
        print(f"run={run_id} | rel L2={rel_l2:.4e}")

    mean_l2 = float(np.mean(summary))
    std_l2 = float(np.std(summary))
    out_csv = os.path.join(args.save_dir, "paper_protocol_summary.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("name,architecture,run,rel_l2\n")
        for idx, err in enumerate(summary, start=1):
            f.write(f"NAS-PINN,\"{last_arch}\",{idx},{err:.8e}\n")
        f.write(f"NAS-PINN,\"{last_arch}\",mean,{mean_l2:.8e}\n")
        f.write(f"NAS-PINN,\"{last_arch}\",std,{std_l2:.8e}\n")
    print(f"Saved summary: {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="NAS-PINN Poisson")
    parser.add_argument("--checkpoint", type=str, default="poisson_checkpoint_last.pth", help="checkpoint filename")
    parser.add_argument("--resume", action="store_true", help="resume training from checkpoint")
    parser.add_argument("--plot-only", action="store_true", help="skip training and only run plots from checkpoint")
    parser.add_argument("--skip-lbfgs", action="store_true", help="skip L-BFGS refinement")
    parser.add_argument("--epochs", type=int, default=12000, help="number of Adam epochs")
    parser.add_argument("--seed", type=int, default=42, help="base random seed")
    parser.add_argument("--multi-seed", action="store_true", help="run three seed values and save comparison")
    parser.add_argument("--seed-list", type=str, default="42,43,44", help="comma-separated seeds for --multi-seed")
    parser.add_argument("--save-dir", type=str, default="results/poisson/naspinn", help="directory to save outputs")
    parser.add_argument("--protocol", action="store_true", help="run repeated fixed-grid protocol")
    parser.add_argument("--paper-protocol", action="store_true", help="alias of --protocol")
    parser.add_argument("--repeats", type=int, default=5, help="number of repeated runs")
    parser.add_argument("--train-nx", type=int, default=100)
    parser.add_argument("--train-ny", type=int, default=100)
    parser.add_argument("--boundary-n", type=int, default=200)
    parser.add_argument("--test-nx", type=int, default=150)
    parser.add_argument("--test-ny", type=int, default=150)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.protocol or args.paper_protocol:
        run_protocol(args)
        return

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
            checkpoint_path = os.path.join(run_dir, args.checkpoint)

            run_start = time.perf_counter()
            torch.manual_seed(seed_val)
            np.random.seed(seed_val)
            model = NAS_PINN(layers=3, base_neurons=192).to(device)

            loss_history = train_with_resume(
                model,
                total_epochs=args.epochs,
                checkpoint_path=checkpoint_path,
                resume=False,
                skip_lbfgs=args.skip_lbfgs,
            )
            plot_loss_curve(
                loss_history,
                os.path.join(run_dir, "poisson_naspinn_loss_curve.png"),
                title=f"Poisson NAS-PINN Training Loss (seed={seed_val})",
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
            for item in summary:
                # Eğer summary elemanı tuple değilse (ör. str), parse et
                if isinstance(item, tuple):
                    seed_val, rel_l2, run_time = item
                elif isinstance(item, str):
                    parts = item.strip().split(",")
                    if len(parts) == 3:
                        seed_val, rel_l2, run_time = parts
                    else:
                        continue
                else:
                    continue
                f.write(f"{seed_val},{float(rel_l2):.8e},{float(run_time):.6f}\n")
        print(f"Saved summary: {out_csv}")

        plt.figure(figsize=(7, 4))
        # summary elemanları (seed, rel_l2, run_time) tuple'ı
        # summary elemanları tuple değilse, parse et
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
        plt.title("Poisson NAS-PINN: Seed Comparison")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        finalize_plot(plt, os.path.join(base_dir, "seed_comparison.png"))
        return

    checkpoint_path = os.path.join(args.save_dir, args.checkpoint)

    model = NAS_PINN(layers=3, base_neurons=192).to(device)
    run_start = time.perf_counter()
    if args.plot_only:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found for plot-only mode: {checkpoint_path}")
        load_checkpoint(checkpoint_path, model)
        loss_history = []
    else:
        loss_history = train_with_resume(
            model,
            total_epochs=args.epochs,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
            skip_lbfgs=args.skip_lbfgs,
        )
        plot_loss_curve(
            loss_history,
            os.path.join(args.save_dir, "poisson_naspinn_loss_curve.png"),
            title="Poisson NAS-PINN Training Loss",
        )

    print_discovered_architecture(model)
    plot_results(model, args.save_dir)
    run_time = time.perf_counter() - run_start
    with open(os.path.join(args.save_dir, "run_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"run_time_seconds,{run_time:.6f}\n")
    print(f"Run time: {run_time:.2f} s")


if __name__ == "__main__":
    main()
