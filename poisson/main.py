import argparse
import csv
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    from pyswarm import pso
except ImportError:
    pso = None

from poisson.domains import annulus, circle, flower, lshape, rectangular


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MASK_LEVELS = [30, 50, 70, 90, 110]
BASE_NEURONS = 110
LAYERS = 5
EPOCHS_ADAM = 12000
INNER_LR = 1e-3
OUTER_LR = 3e-4
OUTER_EVERY = 5
LBFGS_MAX_ITER = 1500
N_COL = 4000
N_BC = 400
TEST_GRID_SIZE = 500
SIMPLE_CMAP = "YlGnBu"
PSO_SPAN = 0.25


DOMAIN_MODULES = {
    "rectangular": rectangular,
    "circle": circle,
    "lshape": lshape,
    "flower": flower,
    "annulus": annulus,
}

DOMAIN_BOUNDS = {
    "rectangular": (-1.0, 1.0, -1.0, 1.0),
    "circle": (-1.0, 1.0, -1.0, 1.0),
    "annulus": (-1.0, 1.0, -1.0, 1.0),
    "flower": (-1.4, 1.4, -1.4, 1.4),
    "lshape": (-1.0, 2.0, -1.0, 2.0),
}


class MixedOp(nn.Module):
    def __init__(self, in_features, max_out_features):
        super().__init__()
        self.max_out = max_out_features
        self.linear = nn.Linear(in_features, max_out_features)
        self.alpha_skip = nn.Parameter(torch.tensor(0.0))
        self.alpha_active = nn.Parameter(torch.tensor(0.0))
        self.alpha_mask = nn.Parameter(0.1 * torch.randn(len(MASK_LEVELS)))

    def forward(self, x):
        skip = x
        if x.shape[-1] < self.max_out:
            skip = F.pad(skip, (0, self.max_out - x.shape[-1]))

        active = torch.tanh(self.linear(x))
        w_skip = torch.sigmoid(self.alpha_skip)
        w_active = torch.sigmoid(self.alpha_active)
        mask_w = F.softmax(self.alpha_mask, dim=0)

        masked_active = 0.0
        for i, lvl in enumerate(MASK_LEVELS):
            mask = torch.zeros(self.max_out, device=device)
            k = min(lvl, self.max_out)
            mask[:k] = 1.0
            masked_active += mask_w[i] * (active * mask)

        out = w_skip * skip + w_active * masked_active
        if self.max_out == 1:
            out = out[:, :1]
        return out


class NAS_PINN(nn.Module):
    def __init__(self, num_layers=LAYERS, base_neurons=BASE_NEURONS):
        super().__init__()
        dims = [2] + [base_neurons] * (num_layers - 1) + [1]
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(MixedOp(dims[i], dims[i + 1]))

    def forward(self, xy):
        x = xy
        for layer in self.layers:
            x = layer(x)
        return x


def pde_loss(model, x, y, domain_mod):
    xy = torch.cat([x, y], dim=1).requires_grad_(True)
    u = model(xy)
    grad = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    ux, uy = grad[:, 0:1], grad[:, 1:2]
    uxx = torch.autograd.grad(ux.sum(), xy, create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(uy.sum(), xy, create_graph=True)[0][:, 1:2]
    residual = uxx + uyy - domain_mod.poisson_rhs(x, y)
    return residual.pow(2).mean()


def bc_loss(model, x_bc, y_bc, domain_mod):
    xy_bc = torch.cat([x_bc, y_bc], dim=1)
    u_pred = model(xy_bc)
    u_true = domain_mod.true_solution(x_bc, y_bc)
    return F.mse_loss(u_pred, u_true)


def supervised_mse(model, x, y, domain_mod):
    xy = torch.cat([x, y], dim=1)
    u_pred = model(xy)
    u_true = domain_mod.true_solution(x, y)
    return F.mse_loss(u_pred, u_true)


def print_discrete_architecture(model):
    print("\n" + "=" * 60)
    print("DISCRETE ARCHITECTURE AFTER TRAINING")
    print("=" * 60)
    for i, layer in enumerate(model.layers):
        w_skip = torch.sigmoid(layer.alpha_skip).item()
        w_active = torch.sigmoid(layer.alpha_active).item()
        mask_probs = F.softmax(layer.alpha_mask, dim=0)
        best_mask_idx = torch.argmax(mask_probs).item()
        best_neurons = MASK_LEVELS[best_mask_idx]
        best_prob = mask_probs[best_mask_idx].item()
        connection = "SKIP" if w_skip > w_active else "ACTIVE"
        print(
            f"Layer {i + 1:2d} -> {connection:6s} | "
            f"neurons: {best_neurons:3d} (mask prob: {best_prob:.4f})"
        )
    print("=" * 60 + "\n")


def clone_model_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def flatten_model_params(model):
    return np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in model.parameters()])


def set_model_from_flat_vector(model, flat):
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            n_param = param.numel()
            chunk = torch.from_numpy(flat[offset : offset + n_param]).view_as(param).to(device)
            param.copy_(chunk)
            offset += n_param


def train_nas_pinn(
    domain_mod,
    epochs=EPOCHS_ADAM,
    inner_lr=INNER_LR,
    outer_lr=OUTER_LR,
    outer_every=OUTER_EVERY,
    skip_lbfgs=False,
    use_pso=False,
    pso_iters=8,
    pso_swarm=16,
    pso_span=PSO_SPAN,
    return_stage_info=False,
):
    model = NAS_PINN().to(device)
    opt_inner = optim.Adam(model.parameters(), lr=inner_lr)

    arch_params = []
    for layer in model.layers:
        arch_params.extend([layer.alpha_skip, layer.alpha_active, layer.alpha_mask])
    opt_outer = optim.Adam(arch_params, lr=outer_lr)

    loss_history = []
    training_start = time.perf_counter()
    print("Starting NAS-PINN training...")

    for epoch in range(epochs):
        (x_col, y_col), (x_bc, y_bc) = domain_mod.sample_points(N_COL, N_BC, device)
        opt_inner.zero_grad()
        l_pde = pde_loss(model, x_col, y_col, domain_mod)
        l_bc = bc_loss(model, x_bc, y_bc, domain_mod)
        loss_inner = l_pde + l_bc
        loss_inner.backward()
        opt_inner.step()

        if epoch % outer_every == 0:
            opt_outer.zero_grad()
            loss_outer = supervised_mse(model, x_col, y_col, domain_mod)
            loss_outer.backward()
            opt_outer.step()
        else:
            loss_outer = torch.tensor(0.0, device=device)

        loss_history.append(float(loss_inner.item()))
        if epoch % 2000 == 0 or epoch == epochs - 1:
            print(
                f"[{epoch:5d}] PDE: {l_pde:.4e}  BC: {l_bc:.4e}  "
                f"Outer MSE: {loss_outer:.4e}"
            )

    adam_state = clone_model_state(model)
    adam_time = time.perf_counter() - training_start

    if not skip_lbfgs:
        print("\nStarting L-BFGS refinement...")
        optimizer_lbfgs = optim.LBFGS(
            model.parameters(),
            lr=0.8,
            max_iter=LBFGS_MAX_ITER,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer_lbfgs.zero_grad()
            lp = pde_loss(model, x_col, y_col, domain_mod)
            lb = bc_loss(model, x_bc, y_bc, domain_mod)
            total = lp + lb
            total.backward()
            return total

        optimizer_lbfgs.step(closure)
        lbfgs_loss = float((pde_loss(model, x_col, y_col, domain_mod) + bc_loss(model, x_bc, y_bc, domain_mod)).item())
    else:
        lbfgs_loss = None

    stage_states = {"adam": adam_state}
    stage_losses = {"adam": float(loss_history[-1]) if loss_history else float("nan")}
    stage_times = {"adam": adam_time}

    if not skip_lbfgs:
        stage_states["lbfgs"] = clone_model_state(model)
        stage_losses["lbfgs"] = float(lbfgs_loss)
        stage_times["lbfgs"] = time.perf_counter() - training_start

    if use_pso:
        if pso is None:
            raise ImportError("Missing dependency: pyswarm. Install with 'pip install pyswarm' to run PSO.")
        if skip_lbfgs:
            print("PSO requested without L-BFGS; running PSO after Adam stage.")
        print("\nStarting PSO refinement...")
        center = flatten_model_params(model)
        lower = center - pso_span
        upper = center + pso_span

        def pso_objective(flat):
            set_model_from_flat_vector(model, flat)
            total = pde_loss(model, x_col, y_col, domain_mod) + bc_loss(model, x_bc, y_bc, domain_mod)
            return float(total.item())

        best_flat, best_loss = pso(
            pso_objective,
            lower,
            upper,
            swarmsize=pso_swarm,
            maxiter=pso_iters,
        )
        set_model_from_flat_vector(model, best_flat)
        stage_states["pso"] = clone_model_state(model)
        stage_losses["pso"] = float(best_loss)
        stage_times["pso"] = time.perf_counter() - training_start
        print(f"PSO best objective: {best_loss:.4e}")

    print_discrete_architecture(model)
    if return_stage_info:
        return {
            "model": model,
            "loss_history": loss_history,
            "stage_states": stage_states,
            "stage_losses": stage_losses,
            "stage_times": stage_times,
        }
    return model, loss_history


def plot_loss_history(loss_history, save_path, title):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="Training Loss (PDE + BC)", color="royalblue", linewidth=1.2)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot: {save_path}")


def evaluate_and_plot(model, domain_mod, save_dir, method_label="NAS-PINN", test_grid=TEST_GRID_SIZE):
    os.makedirs(save_dir, exist_ok=True)

    domain_name = domain_mod.__name__.split(".")[-1]
    x_min, x_max, y_min, y_max = DOMAIN_BOUNDS.get(domain_name, (-1.0, 1.0, -1.0, 1.0))

    nx = ny = test_grid
    x = torch.linspace(x_min, x_max, nx, device=device)
    y = torch.linspace(y_min, y_max, ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    xy_grid = torch.stack([X.flatten(), Y.flatten()], dim=1)

    with torch.no_grad():
        pred = model(xy_grid).reshape(nx, ny).cpu().numpy()
        true = domain_mod.true_solution(X, Y).cpu().numpy()
        err_abs = np.abs(pred - true)

    err_sq = (pred - true) ** 2
    rel_l2 = np.sqrt(np.mean(err_sq)) / (np.sqrt(np.mean(true ** 2)) + 1e-12)
    print(f"\nRelative L2 error ({domain_name}, grid {nx}x{ny}): {rel_l2:.4e}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin = float(np.min(true))
    vmax = float(np.max(true))

    im0 = axes[0].imshow(
        true.T,
        origin="lower",
        cmap=SIMPLE_CMAP,
        extent=[x_min, x_max, y_min, y_max],
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Exact")
    plt.colorbar(im0, ax=axes[0], shrink=0.6)

    im1 = axes[1].imshow(
        pred.T,
        origin="lower",
        cmap=SIMPLE_CMAP,
        extent=[x_min, x_max, y_min, y_max],
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(f"Predicted ({method_label})")
    plt.colorbar(im1, ax=axes[1], shrink=0.6)

    im2 = axes[2].imshow(
        err_abs.T,
        origin="lower",
        cmap=SIMPLE_CMAP,
        extent=[x_min, x_max, y_min, y_max],
    )
    axes[2].set_title("|Pred - Exact|")
    plt.colorbar(im2, ax=axes[2], shrink=0.6)

    plt.suptitle(f"Poisson - {domain_name.capitalize()} Domain\nRel L2 = {rel_l2:.4e}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "result_comparison.png"), dpi=300)
    plt.close()

    csv_path = os.path.join(save_dir, "results_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "exact", "predicted", "abs_error"])
        for i in range(nx):
            for j in range(ny):
                writer.writerow(
                    [
                        float(X[i, j]),
                        float(Y[i, j]),
                        float(true[i, j]),
                        float(pred[i, j]),
                        float(err_abs[i, j]),
                    ]
                )
    print(f"Saved CSV: {csv_path}")
    return float(rel_l2)


def save_run_metrics(save_dir, method, domain, seed, rel_l2, run_time_seconds):
    run_time_path = os.path.join(save_dir, "run_time.txt")
    with open(run_time_path, "w", encoding="utf-8") as f:
        f.write(f"run_time_seconds,{run_time_seconds:.6f}\n")

    metrics_path = os.path.join(save_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "domain", "seed", "rel_l2", "run_time_seconds"])
        writer.writerow([method, domain, seed, f"{rel_l2:.8e}", f"{run_time_seconds:.6f}"])


def build_stage_loss_curve(base_history, stage_name, stage_losses):
    if not base_history:
        return []
    if stage_name == "adam":
        return list(base_history)
    if stage_name in stage_losses:
        return list(base_history) + [float(stage_losses[stage_name])]
    return list(base_history)


def save_stage_outputs(args, domain_name, domain_mod, model, stage_name, stage_dir, loss_history, stage_losses, run_time_seconds):
    os.makedirs(stage_dir, exist_ok=True)
    stage_loss = build_stage_loss_curve(loss_history, stage_name, stage_losses)
    plot_loss_history(
        stage_loss,
        save_path=os.path.join(stage_dir, "poisson_naspinn_loss_curve.png"),
        title=f"Poisson NAS-PINN Loss ({domain_name}, {stage_name.upper()})",
    )
    rel_l2 = evaluate_and_plot(model, domain_mod, save_dir=stage_dir, method_label=f"NAS-PINN {stage_name.upper()}")
    with open(os.path.join(stage_dir, "l2_error.txt"), "w", encoding="utf-8") as f:
        f.write(f"stage,{stage_name}\ndomain,{domain_name}\nrel_l2,{rel_l2:.8e}\n")
    with open(os.path.join(stage_dir, "run_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"run_time_seconds,{run_time_seconds:.6f}\n")
    with open(os.path.join(stage_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "stage", "domain", "seed", "rel_l2", "run_time_seconds"])
        writer.writerow(["naspinn", stage_name, domain_name, args.seed, f"{rel_l2:.8e}", f"{run_time_seconds:.6f}"])
    return float(rel_l2)


def copy_stage_to_root(stage_dir, root_dir):
    copy_map = [
        ("poisson_naspinn_loss_curve.png", "poisson_naspinn_loss_curve.png"),
        ("result_comparison.png", "result_comparison.png"),
        ("results_summary.csv", "results_summary.csv"),
        ("l2_error.txt", "l2_error.txt"),
        ("metrics.csv", "metrics.csv"),
        ("run_time.txt", "run_time.txt"),
    ]
    for src_name, dst_name in copy_map:
        src = os.path.join(stage_dir, src_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(root_dir, dst_name))


def apply_stage_flags(args):
    stage = getattr(args, "stage", None)
    if stage == "adam":
        args.skip_lbfgs = True
        args.use_pso = False
    elif stage == "lbfgs":
        args.skip_lbfgs = False
        args.use_pso = False
    elif stage == "pso":
        args.skip_lbfgs = False
        args.use_pso = True

    if args.skip_lbfgs:
        args.use_pso = False


def run_single(args, domain_name, save_dir):
    start = time.perf_counter()
    os.makedirs(save_dir, exist_ok=True)
    apply_stage_flags(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    domain_mod = DOMAIN_MODULES[domain_name]
    train_info = train_nas_pinn(
        domain_mod,
        epochs=args.epochs,
        skip_lbfgs=args.skip_lbfgs,
        use_pso=args.use_pso,
        pso_iters=args.pso_iters,
        pso_swarm=args.pso_swarm,
        pso_span=args.pso_span,
        return_stage_info=True,
    )

    loss_hist = train_info["loss_history"]
    stage_states = train_info["stage_states"]
    stage_losses = train_info["stage_losses"]
    stage_times = train_info["stage_times"]

    stage_order = ["adam", "lbfgs", "pso"]
    stage_results = []
    for stage_name in stage_order:
        if stage_name not in stage_states:
            continue
        model_stage = NAS_PINN().to(device)
        model_stage.load_state_dict(stage_states[stage_name])
        stage_dir = os.path.join(save_dir, f"stage_{stage_name}")
        rel_l2 = save_stage_outputs(
            args,
            domain_name,
            domain_mod,
            model_stage,
            stage_name,
            stage_dir,
            loss_hist,
            stage_losses,
            float(stage_times.get(stage_name, 0.0)),
        )
        stage_results.append((stage_name, rel_l2, float(stage_times.get(stage_name, 0.0))))

    if not stage_results:
        raise RuntimeError("No stage output was produced.")

    final_stage_name, final_rel_l2, _ = stage_results[-1]
    final_stage_dir = os.path.join(save_dir, f"stage_{final_stage_name}")
    copy_stage_to_root(final_stage_dir, save_dir)

    with open(os.path.join(save_dir, "stage_summary.csv"), "w", encoding="utf-8") as f:
        f.write("stage,domain,seed,rel_l2,run_time_seconds\n")
        for stage_name, rel_l2, stage_time in stage_results:
            f.write(f"{stage_name},{domain_name},{args.seed},{rel_l2:.8e},{stage_time:.6f}\n")

    run_time = time.perf_counter() - start
    with open(os.path.join(save_dir, "run_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"run_time_seconds,{run_time:.6f}\n")
    with open(os.path.join(save_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "stage", "domain", "seed", "rel_l2", "run_time_seconds"])
        writer.writerow(["naspinn", final_stage_name, domain_name, args.seed, f"{final_rel_l2:.8e}", f"{run_time:.6f}"])
    print(f"Run time: {run_time:.2f} s")
    return float(final_rel_l2), float(run_time)


def run_multi_domain(args):
    domains = [d.strip() for d in args.domain_list.split(",") if d.strip()]
    summary = []

    for idx, domain_name in enumerate(domains):
        if domain_name not in DOMAIN_MODULES:
            raise ValueError(f"Unknown domain in --domain-list: {domain_name}")
        args.seed = args.base_seed + idx
        out_dir = os.path.join(args.save_dir, f"domain_{domain_name}")
        rel_l2, run_time = run_single(args, domain_name=domain_name, save_dir=out_dir)
        summary.append((domain_name, rel_l2, run_time))

    csv_path = os.path.join(args.save_dir, "domain_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("domain,rel_l2,run_time_seconds\n")
        for domain_name, rel_l2, run_time in summary:
            f.write(f"{domain_name},{rel_l2:.8e},{run_time:.6f}\n")
    print(f"Saved summary: {csv_path}")

    domains = [row[0] for row in summary]
    errs = [row[1] for row in summary]
    plt.figure(figsize=(7, 4))
    plt.plot(domains, errs, marker="o", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Domain")
    plt.ylabel("Relative L2 Error")
    plt.title("Poisson NAS-PINN: Domain Comparison")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "domain_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="NAS-PINN Poisson Solver")
    parser.add_argument(
        "--domain",
        type=str,
        default="rectangular",
        choices=list(DOMAIN_MODULES.keys()),
        help="single domain type",
    )
    parser.add_argument(
        "--multi-domain",
        action="store_true",
        help="run multiple domains from --domain-list",
    )
    parser.add_argument(
        "--domain-list",
        type=str,
        default="rectangular,circle,lshape,flower,annulus",
        help="comma-separated domains for --multi-domain",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/poisson/naspinn",
        help="directory to save outputs",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS_ADAM, help="Adam epochs")
    parser.add_argument("--seed", type=int, default=42, help="seed for single-domain run")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["adam", "lbfgs", "pso"],
        default=None,
        help="optional stage shortcut: adam | lbfgs | pso",
    )
    parser.add_argument("--skip-lbfgs", action="store_true", help="skip L-BFGS refinement")
    parser.add_argument("--use-pso", action="store_true", help="enable PSO refinement after L-BFGS")
    parser.add_argument("--pso-iters", type=int, default=8, help="PSO max iterations")
    parser.add_argument("--pso-swarm", type=int, default=16, help="PSO swarm size")
    parser.add_argument("--pso-span", type=float, default=PSO_SPAN, help="PSO search span around current weights")
    return parser.parse_args()


def main():
    args = parse_args()
    args.base_seed = args.seed
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Using device: {device}")

    if args.multi_domain:
        run_multi_domain(args)
    else:
        run_single(args, domain_name=args.domain, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
