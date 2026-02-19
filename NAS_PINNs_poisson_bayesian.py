import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

try:
    from bayes_opt import BayesianOptimization
except ImportError as exc:
    raise ImportError(
        "Missing dependency: bayes_opt. Install with 'pip install bayesian-optimization'."
    ) from exc


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
interactive_plots = os.environ.get("DISPLAY") is not None

# Same core Poisson setup as NAS_PINNs_poisson.py
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
N_col = 10000
N_bc = 400
lambda_pde = 1.0
lambda_bc = 100.0


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def finalize_plot(save_path):
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {save_path}")
    if interactive_plots:
        plt.show()
    else:
        plt.close()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def sample_points():
    x_col = torch.rand(N_col, 1, device=device) * (x_max - x_min) + x_min
    y_col = torch.rand(N_col, 1, device=device) * (y_max - y_min) + y_min

    n_per_side = N_bc // 4
    x_bot = torch.rand(n_per_side, 1, device=device) * (x_max - x_min) + x_min
    y_bot = torch.zeros_like(x_bot)

    x_top = torch.rand(n_per_side, 1, device=device) * (x_max - x_min) + x_min
    y_top = torch.ones_like(x_top)

    y_left = torch.rand(n_per_side, 1, device=device) * (y_max - y_min) + y_min
    x_left = torch.zeros_like(y_left)

    y_right = torch.rand(n_per_side, 1, device=device) * (y_max - y_min) + y_min
    x_right = torch.ones_like(y_right)

    x_bc = torch.cat([x_bot, x_top, x_left, x_right], dim=0)
    y_bc = torch.cat([y_bot, y_top, y_left, y_right], dim=0)

    return (x_col, y_col), (x_bc, y_bc)


def sample_points_protocol(train_nx=100, train_ny=100, boundary_n=200):
    x_vals = torch.linspace(x_min, x_max, train_nx, device=device)
    y_vals = torch.linspace(y_min, y_max, train_ny, device=device)

    X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")
    x_col = X.reshape(-1, 1)
    y_col = Y.reshape(-1, 1)

    x_side = torch.linspace(x_min, x_max, boundary_n, device=device).unsqueeze(1)
    y_side = torch.linspace(y_min, y_max, boundary_n, device=device).unsqueeze(1)

    x_bot = x_side
    y_bot = torch.full_like(x_bot, y_min)
    x_top = x_side
    y_top = torch.full_like(x_top, y_max)

    y_left = y_side
    x_left = torch.full_like(y_left, x_min)
    y_right = y_side
    x_right = torch.full_like(y_right, x_max)

    x_bc = torch.cat([x_bot, x_top, x_left, x_right], dim=0)
    y_bc = torch.cat([y_bot, y_top, y_left, y_right], dim=0)

    return (x_col, y_col), (x_bc, y_bc)


class BayesianPoissonPINN(nn.Module):
    def __init__(self, hidden_widths, hidden_acts):
        super().__init__()
        self.hidden_widths = hidden_widths
        self.hidden_acts = hidden_acts

        layers = []
        in_dim = 2
        for width, act_name in zip(hidden_widths, hidden_acts):
            layers.append(nn.Linear(in_dim, width))
            if act_name == "sin":
                layers.append(SinActivation())
            else:
                layers.append(nn.Tanh())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)


def pde_loss(model, x, y):
    xy = torch.cat([x, y], dim=1).requires_grad_(True)
    phi = model(xy)

    grad_phi = torch.autograd.grad(phi.sum(), xy, create_graph=True)[0]
    phi_x = grad_phi[:, 0:1]
    phi_y = grad_phi[:, 1:2]

    phi_xx = torch.autograd.grad(phi_x.sum(), xy, create_graph=True)[0][:, 0:1]
    phi_yy = torch.autograd.grad(phi_y.sum(), xy, create_graph=True)[0][:, 1:2]

    residual = phi_xx + phi_yy + 2 * (np.pi ** 2) * torch.cos(np.pi * x) * torch.cos(np.pi * y)
    return torch.mean(residual ** 2)


def bc_loss(model, x_bc, y_bc):
    xy_bc = torch.cat([x_bc, y_bc], dim=1)
    phi_pred = model(xy_bc)
    phi_exact = torch.cos(np.pi * x_bc) * torch.cos(np.pi * y_bc)
    return torch.mean((phi_pred - phi_exact) ** 2)


def train_model(model, points, epochs=12000, lr=1e-3, skip_lbfgs=False):
    (x_col, y_col), (x_bc, y_bc) = points

    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        lp = pde_loss(model, x_col, y_col)
        lb = bc_loss(model, x_bc, y_bc)
        loss = lambda_pde * lp + lambda_bc * lb
        loss.backward()
        opt.step()

    if not skip_lbfgs:
        lbfgs = optim.LBFGS(model.parameters(), lr=0.8, max_iter=2000, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            lp = pde_loss(model, x_col, y_col)
            lb = bc_loss(model, x_bc, y_bc)
            loss = lambda_pde * lp + lambda_bc * lb
            loss.backward()
            return loss

        lbfgs.step(closure)


def predict_on_grid(model, test_nx=150, test_ny=150):
    x_grid = torch.linspace(x_min, x_max, test_nx, device=device)
    y_grid = torch.linspace(y_min, y_max, test_ny, device=device)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing="ij")
    XY = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        phi_pred = model(XY).cpu().numpy().reshape(test_nx, test_ny)

    x_np = X.cpu().numpy()
    y_np = Y.cpu().numpy()
    phi_exact = np.cos(np.pi * x_np) * np.cos(np.pi * y_np)
    rel_l2 = np.linalg.norm(phi_pred - phi_exact) / (np.linalg.norm(phi_exact) + 1e-12)
    return rel_l2, x_np, y_np, phi_pred, phi_exact


def decode_architecture(params):
    n_layers = int(round(params["n_layers"]))
    widths_all = [
        int(round(params["n1"])),
        int(round(params["n2"])),
        int(round(params["n3"])),
        int(round(params["n4"])),
    ]
    acts_all = [
        "sin" if params["a1"] >= 0.5 else "tanh",
        "sin" if params["a2"] >= 0.5 else "tanh",
        "sin" if params["a3"] >= 0.5 else "tanh",
        "sin" if params["a4"] >= 0.5 else "tanh",
    ]
    widths = widths_all[:n_layers]
    acts = acts_all[:n_layers]
    lr = float(params["lr"])
    return widths, acts, lr


def architecture_signature(widths, acts):
    items = [f"{a}-{w}" for w, a in zip(widths, acts)]
    return " | ".join(items)


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
        print(
            f"BO eval {eval_counter['k']:02d} | widths={widths} acts={acts} "
            f"lr={lr_decoded:.2e} | relL2={rel_l2:.4e}"
        )
        return -float(rel_l2)

    pbounds = {
        "n_layers": (2, 4),
        "n1": (32, 256),
        "n2": (32, 256),
        "n3": (32, 256),
        "n4": (32, 256),
        "a1": (0, 1),
        "a2": (0, 1),
        "a3": (0, 1),
        "a4": (0, 1),
        "lr": (5e-4, 2e-3),
    }

    bo = BayesianOptimization(f=objective, pbounds=pbounds, random_state=args.seed, verbose=2)
    bo.maximize(init_points=args.bo_init_points, n_iter=args.bo_iters)

    best_widths, best_acts, best_lr = decode_architecture(bo.max["params"])
    best_rel_l2 = -bo.max["target"]
    return best_widths, best_acts, best_lr, best_rel_l2


def plot_results(model, save_dir):
    rel_l2, x_np, y_np, phi_pred, phi_exact = predict_on_grid(
        model,
        test_nx=150,
        test_ny=150,
    )
    print(f"Final relative L2 error: {rel_l2:.4e}")

    abs_err = np.abs(phi_pred - phi_exact)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    cs0 = axes[0].contourf(x_np, y_np, phi_exact, levels=50, cmap="viridis")
    axes[0].set_title("Exact φ(x,y)")
    plt.colorbar(cs0, ax=axes[0])

    cs1 = axes[1].contourf(x_np, y_np, phi_pred, levels=50, cmap="viridis")
    axes[1].set_title("Predicted (Bayes-NAS)")
    plt.colorbar(cs1, ax=axes[1])

    cs2 = axes[2].contourf(x_np, y_np, abs_err, levels=50, cmap="magma")
    axes[2].set_title("|Pred-Exact|")
    plt.colorbar(cs2, ax=axes[2])

    finalize_plot(os.path.join(save_dir, "bayes_poisson_results.png"))


def run_single(args):
    set_seed(args.seed)
    train_points = sample_points()

    best_widths, best_acts, best_lr, bo_rel_l2 = run_bayesian_search(train_points, args)
    print("\nBest architecture from BO")
    print(f"hidden widths: {best_widths}")
    print(f"hidden acts  : {best_acts}")
    print(f"lr           : {best_lr:.4e}")
    print(f"BO rel L2    : {bo_rel_l2:.4e}")

    best_model = BayesianPoissonPINN(best_widths, best_acts).to(device)
    train_model(
        best_model,
        train_points,
        epochs=args.epochs,
        lr=best_lr,
        skip_lbfgs=args.skip_lbfgs,
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

    plot_results(best_model, args.save_dir)


def run_paper_protocol(args):
    train_points = sample_points_protocol(
        train_nx=args.train_nx,
        train_ny=args.train_ny,
        boundary_n=args.boundary_n,
    )

    run_errors = []
    last_arch = ""

    print("\nRunning paper-style protocol (Bayesian NAS, Poisson)")
    print(
        f"Train grid: x={args.train_nx}, y={args.train_ny}, boundary={args.boundary_n} | "
        f"Test grid: x={args.test_nx}, y={args.test_ny}"
    )
    print(f"Repeats: {args.repeats}")

    for run_id in range(1, args.repeats + 1):
        args_local = argparse.Namespace(**vars(args))
        args_local.seed = args.seed + run_id

        best_widths, best_acts, best_lr, _ = run_bayesian_search(train_points, args_local)
        last_arch = architecture_signature(best_widths, best_acts)

        model = BayesianPoissonPINN(best_widths, best_acts).to(device)
        train_model(
            model,
            train_points,
            epochs=args.epochs,
            lr=best_lr,
            skip_lbfgs=args.skip_lbfgs,
        )

        rel_l2, x_np, y_np, phi_pred, phi_exact = predict_on_grid(
            model,
            test_nx=args.test_nx,
            test_ny=args.test_ny,
        )
        run_errors.append(rel_l2)
        print(f"run={run_id} relL2={rel_l2:.4e}")

        if run_id == args.repeats:
            abs_err = np.abs(phi_pred - phi_exact)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
            cs0 = axes[0].contourf(x_np, y_np, phi_exact, levels=50, cmap="viridis")
            axes[0].set_title("Exact φ(x,y)")
            plt.colorbar(cs0, ax=axes[0])

            cs1 = axes[1].contourf(x_np, y_np, phi_pred, levels=50, cmap="viridis")
            axes[1].set_title("Predicted φ(x,y)")
            plt.colorbar(cs1, ax=axes[1])

            cs2 = axes[2].contourf(x_np, y_np, abs_err, levels=50, cmap="magma")
            axes[2].set_title("|Pred-Exact|")
            plt.colorbar(cs2, ax=axes[2])

            finalize_plot(os.path.join(args.save_dir, "bayes_poisson_protocol_last_run.png"))

    mean_l2 = float(np.mean(run_errors))
    std_l2 = float(np.std(run_errors))

    print("\nPaper-style summary (5-run average):")
    print("Name      Architecture                         mean_rel_L2      std_rel_L2")
    print(f"Bayes-NAS {last_arch:<35}  {mean_l2:.4e}    {std_l2:.4e}")

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
    parser.add_argument("--save-dir", type=str, default="results_plots_poisson_bayes", help="output directory")
    parser.add_argument("--checkpoint", type=str, default="bayes_poisson_checkpoint_last.pth", help="checkpoint filename")

    parser.add_argument("--bo-init-points", type=int, default=2, help="BO initial random points")
    parser.add_argument("--bo-iters", type=int, default=8, help="BO guided iterations")
    parser.add_argument("--bo-epochs", type=int, default=1200, help="training epochs per BO evaluation")

    parser.add_argument("--paper-protocol", action="store_true", help="paper-like repeated protocol")
    parser.add_argument("--repeats", type=int, default=5, help="number of repeats")
    parser.add_argument("--train-nx", type=int, default=100, help="protocol train x-grid points")
    parser.add_argument("--train-ny", type=int, default=100, help="protocol train y-grid points")
    parser.add_argument("--boundary-n", type=int, default=200, help="protocol boundary points per side")
    parser.add_argument("--test-nx", type=int, default=150, help="protocol test x-grid points")
    parser.add_argument("--test-ny", type=int, default=150, help="protocol test y-grid points")

    parser.add_argument("--seed", type=int, default=42, help="base random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.paper_protocol:
        run_paper_protocol(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
