import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.integrate import solve_ivp
from .plots import (
    plot_burgers_exact_pred_error,
    plot_burgers_heatmap,
    plot_loss_curve,
    plot_burgers_time_slices,
    plot_burgers_time_slices_with_exact,
    should_use_exact_plots,
)

try:
    from bayes_opt import BayesianOptimization
except ImportError as exc:
    raise ImportError("Missing dependency: bayes_opt. Install with 'pip install bayesian-optimization'.") from exc


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
interactive_plots = os.environ.get("DISPLAY") is not None

x_min, x_max = -1.0, 1.0
t_min, t_max = 0.0, 1.0
N_col = 10000
N_ic = 200
N_bc = 200
lambda_pde = 1.0
lambda_ic = 100.0
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
    x_c = torch.rand(N_col, 1, device=device) * (x_max - x_min) + x_min
    t_c = torch.rand(N_col, 1, device=device) * (t_max - t_min) + t_min

    x_ic = torch.rand(N_ic, 1, device=device) * (x_max - x_min) + x_min
    t_ic = torch.zeros_like(x_ic)

    t_bc = torch.rand(N_bc, 1, device=device) * (t_max - t_min) + t_min
    x_left = torch.full((N_bc, 1), x_min, device=device)
    x_right = torch.full((N_bc, 1), x_max, device=device)

    return (x_c, t_c), (x_ic, t_ic), (x_left, t_bc), (x_right, t_bc)


def sample_points_paper(train_nx=250, train_nt=21):
    x_vals = torch.linspace(x_min, x_max, train_nx, device=device)
    t_vals = torch.linspace(t_min, t_max, train_nt, device=device)

    X, T = torch.meshgrid(x_vals, t_vals, indexing="ij")
    x_c = X.reshape(-1, 1)
    t_c = T.reshape(-1, 1)

    x_ic = x_vals.unsqueeze(1)
    t_ic = torch.zeros_like(x_ic)

    t_bc = t_vals.unsqueeze(1)
    x_left = torch.full_like(t_bc, x_min)
    x_right = torch.full_like(t_bc, x_max)

    return (x_c, t_c), (x_ic, t_ic), (x_left, t_bc), (x_right, t_bc)


class BayesianNASPINN(nn.Module):
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

    def forward(self, xt):
        return self.net(xt)


def pde_residual(model, x, t, nu_coef):
    xt = torch.cat([x, t], dim=1).requires_grad_(True)
    u = model(xt)

    grads = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
    u_t = grads[:, 1:2]
    u_x = grads[:, 0:1]
    u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]

    f = u_t + u * u_x - nu_coef * u_xx
    return torch.mean(f ** 2)


def ic_loss(model, x, t):
    xt = torch.cat([x, t], dim=1)
    u_pred = model(xt)
    u_true = -torch.sin(np.pi * x)
    return torch.mean((u_pred - u_true) ** 2)


def bc_loss(model, x_l, t_l, x_r, t_r):
    xt_l = torch.cat([x_l, t_l], dim=1)
    xt_r = torch.cat([x_r, t_r], dim=1)
    return torch.mean(model(xt_l) ** 2) + torch.mean(model(xt_r) ** 2)


def train_model(model, points, nu_coef, epochs=15000, lr=1e-3, skip_lbfgs=False, record_history=False):
    (x_c, t_c), (x_ic, t_ic), (x_l, t_l), (x_r, t_r) = points

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_history = [] if record_history else None
    for _ in range(epochs):
        opt.zero_grad()
        l_pde = pde_residual(model, x_c, t_c, nu_coef)
        l_ic = ic_loss(model, x_ic, t_ic)
        l_bc = bc_loss(model, x_l, t_l, x_r, t_r)
        loss = lambda_pde * l_pde + lambda_ic * l_ic + lambda_bc * l_bc
        loss.backward()
        opt.step()
        if record_history:
            loss_history.append(float(loss.item()))

    if not skip_lbfgs:
        lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=3000, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            l_pde = pde_residual(model, x_c, t_c, nu_coef)
            l_ic = ic_loss(model, x_ic, t_ic)
            l_bc = bc_loss(model, x_l, t_l, x_r, t_r)
            loss = lambda_pde * l_pde + lambda_ic * l_ic + lambda_bc * l_bc
            loss.backward()
            return loss

        lbfgs.step(closure)

    return loss_history


def predict_on_grid(model, x_values, t_values):
    Xg, Tg = torch.meshgrid(x_values, t_values, indexing="ij")
    XT = torch.cat([Xg.reshape(-1, 1), Tg.reshape(-1, 1)], dim=1)
    with torch.no_grad():
        return model(XT).cpu().numpy().reshape(len(x_values), len(t_values))


def reference_solution_fd(nu_coef, x_vals_np, t_vals_np):
    nx = len(x_vals_np)
    dx = x_vals_np[1] - x_vals_np[0]

    u0 = -np.sin(np.pi * x_vals_np)
    u0[0] = 0.0
    u0[-1] = 0.0

    def rhs(_t, u_inner):
        u = np.zeros(nx, dtype=np.float64)
        u[1:-1] = u_inner
        ux = (u[2:] - u[:-2]) / (2.0 * dx)
        uxx = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx ** 2)
        return -u[1:-1] * ux + nu_coef * uxx

    sol = solve_ivp(
        rhs,
        t_span=(float(t_vals_np[0]), float(t_vals_np[-1])),
        y0=u0[1:-1],
        t_eval=t_vals_np,
        method="BDF",
        rtol=1e-5,
        atol=1e-7,
    )
    if not sol.success:
        raise RuntimeError(f"Reference solver failed: {sol.message}")

    U = np.zeros((nx, len(t_vals_np)), dtype=np.float64)
    U[1:-1, :] = sol.y
    return U


def get_reference_solution(nu_coef, x_test, t_test):
    x_np = x_test.detach().cpu().numpy()
    t_np = t_test.detach().cpu().numpy()

    if should_use_exact_plots(nu_coef) and os.path.exists("burgers_shock.mat"):
        data = loadmat("burgers_shock.mat")
        x_exact = data["x"].squeeze()
        t_exact = data["t"].squeeze()
        u_exact = np.real(data["usol"])
        if len(x_exact) == len(x_np) and len(t_exact) == len(t_np):
            return u_exact

    return reference_solution_fd(nu_coef, x_np, t_np)


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


def run_bayesian_search(nu_coef, train_points, x_test, t_test, args):
    exact_u = get_reference_solution(nu_coef, x_test, t_test)
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
        model = BayesianNASPINN(widths, acts).to(device)
        train_model(
            model,
            train_points,
            nu_coef=nu_coef,
            epochs=args.bo_epochs,
            lr=lr_decoded,
            skip_lbfgs=True,
            record_history=False,
        )

        pred_u = predict_on_grid(model, x_test, t_test)
        rel_l2 = np.linalg.norm(pred_u - exact_u) / (np.linalg.norm(exact_u) + 1e-12)
        print(f"BO eval {eval_counter['k']:02d} | widths={widths} acts={acts} lr={lr_decoded:.2e} | relL2={rel_l2:.4e}")
        return -float(rel_l2)

    pbounds = {
        "n_layers": (3, 4),
        "n1": (32, 192), "n2": (32, 192), "n3": (32, 192), "n4": (32, 192),
        "a1": (0, 1), "a2": (0, 1), "a3": (0, 1), "a4": (0, 1),
        "lr": (5e-4, 2e-3),
    }

    bo = BayesianOptimization(f=objective, pbounds=pbounds, random_state=args.seed, verbose=2)
    bo.maximize(init_points=args.bo_init_points, n_iter=args.bo_iters)

    best_widths, best_acts, best_lr = decode_architecture(bo.max["params"])
    return best_widths, best_acts, best_lr, -bo.max["target"]


def plot_results(model, save_dir, nu_coef, x_test, t_test, include_exact_plot=True):
    x_np = x_test.detach().cpu().numpy()
    t_np = t_test.detach().cpu().numpy()
    pred_u = predict_on_grid(model, x_test, t_test)
    exact_u = get_reference_solution(nu_coef, x_test, t_test)

    rel_l2 = np.linalg.norm(pred_u - exact_u) / (np.linalg.norm(exact_u) + 1e-12)
    print(f"Final relative L2 error: {rel_l2:.4e}")

    if include_exact_plot:
        plot_burgers_exact_pred_error(
            x_np,
            t_np,
            exact_u,
            pred_u,
            os.path.join(save_dir, "bayes_burgers_full_exact_vs_pred.png"),
            pred_title="Predicted (Bayes-NAS)",
            use_interactive=interactive_plots,
        )
    return float(rel_l2)


def run_single(args):
    run_start = time.perf_counter()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    train_points = sample_points()
    x_test = torch.linspace(x_min, x_max, args.test_nx, device=device)
    t_test = torch.linspace(t_min, t_max, args.test_nt, device=device)

    best_widths, best_acts, best_lr, bo_rel_l2 = run_bayesian_search(args.nu, train_points, x_test, t_test, args)
    print("\nBest architecture from BO")
    print(f"hidden widths: {best_widths}")
    print(f"hidden acts  : {best_acts}")
    print(f"lr           : {best_lr:.4e}")
    print(f"BO rel L2    : {bo_rel_l2:.4e}")

    best_model = BayesianNASPINN(best_widths, best_acts).to(device)
    loss_history = train_model(
        best_model,
        train_points,
        nu_coef=args.nu,
        epochs=args.epochs,
        lr=best_lr,
        skip_lbfgs=args.skip_lbfgs,
        record_history=True,
    )

    ckpt_name = args.checkpoint
    if not os.path.isabs(ckpt_name):
        ckpt_path = os.path.join(args.save_dir, ckpt_name)
    else:
        ckpt_path = ckpt_name
    torch.save(
        {
            "model_state": best_model.state_dict(),
            "nu": args.nu,
            "hidden_widths": best_widths,
            "hidden_acts": best_acts,
            "lr": best_lr,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")

    plot_loss_curve(
        loss_history,
        os.path.join(args.save_dir, "loss_curve.png"),
        title=f"Burgers Bayesian Loss (nu={args.nu:.3f})",
        use_interactive=interactive_plots,
    )

    plot_burgers_heatmap(
        best_model,
        device,
        x_min,
        x_max,
        t_min,
        t_max,
        os.path.join(args.save_dir, "burgers_heatmap.png"),
        use_interactive=interactive_plots,
    )

    use_exact = should_use_exact_plots(args.nu)
    if use_exact:
        plot_burgers_time_slices(
            best_model,
            device,
            x_min,
            x_max,
            os.path.join(args.save_dir, "burgers_time_slices.png"),
            use_interactive=interactive_plots,
        )
        plot_burgers_time_slices_with_exact(
            best_model,
            device,
            os.path.join(args.save_dir, "burgers_exact_vs_pred_time_slices.png"),
            mat_path="burgers_shock.mat",
            use_interactive=interactive_plots,
        )
    else:
        print("Exact/time-slice comparison is only generated for --nu=0.01. Saving heatmap + rel L2 for this run.")

    rel_l2 = plot_results(best_model, args.save_dir, args.nu, x_test, t_test, include_exact_plot=use_exact)
    with open(os.path.join(args.save_dir, "l2_error.txt"), "w", encoding="utf-8") as f:
        f.write(f"nu,{args.nu:.6f}\nrel_l2,{rel_l2:.8e}\n")
    run_time = time.perf_counter() - run_start
    with open(os.path.join(args.save_dir, "run_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"run_time_seconds,{run_time:.6f}\n")
    with open(os.path.join(args.save_dir, "metrics.csv"), "w", encoding="utf-8") as f:
        f.write("method,nu,seed,rel_l2,run_time_seconds\n")
        f.write(f"bayesian,{args.nu:.6f},{args.seed},{rel_l2:.8e},{run_time:.6f}\n")
    print(f"Run time: {run_time:.2f} s")
    return float(rel_l2), float(run_time)


def run_multi_nu(args):
    nu_values = [float(v.strip()) for v in args.nu_list.split(",") if v.strip()]
    summary = []
    base_dir = args.save_dir
    os.makedirs(base_dir, exist_ok=True)

    for idx, nu_val in enumerate(nu_values):
        args_local = argparse.Namespace(**vars(args))
        args_local.nu = nu_val
        args_local.seed = args.seed + idx
        args_local.save_dir = os.path.join(base_dir, f"nu_{nu_val:.3f}")
        args_local.checkpoint = "bayes_checkpoint_last.pth"
        rel_l2, run_time = run_single(args_local)
        summary.append((nu_val, rel_l2, run_time))

    out_csv = os.path.join(base_dir, "viscosity_comparison.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("nu,rel_l2,run_time_seconds\n")
        for nu_val, rel_l2, run_time in summary:
            f.write(f"{nu_val:.6f},{rel_l2:.8e},{run_time:.6f}\n")
    print(f"Saved summary: {out_csv}")

    nus = [row[0] for row in summary]
    errs = [row[1] for row in summary]
    plt.figure(figsize=(7, 4))
    plt.plot(nus, errs, marker="o", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Viscosity (nu)")
    plt.ylabel("Relative L2 Error")
    plt.title("Burgers Bayesian: Viscosity Comparison")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    finalize_plot(os.path.join(base_dir, "viscosity_comparison.png"))


def run_paper_protocol(args):
    nu_values = [float(v.strip()) for v in args.paper_nus.split(",") if v.strip()]
    train_points = sample_points_paper(train_nx=args.train_nx, train_nt=args.train_nt)
    x_test = torch.linspace(x_min, x_max, args.test_nx, device=device)
    t_test = torch.linspace(t_min, t_max, args.test_nt, device=device)

    rows = []
    for nu_val in nu_values:
        run_errors = []
        for run_id in range(1, args.repeats + 1):
            args_local = argparse.Namespace(**vars(args))
            args_local.seed = args.seed + run_id + int(1000 * nu_val)

            best_widths, best_acts, best_lr, _ = run_bayesian_search(nu_val, train_points, x_test, t_test, args_local)
            model = BayesianNASPINN(best_widths, best_acts).to(device)
            train_model(model, train_points, nu_coef=nu_val, epochs=args.epochs, lr=best_lr, skip_lbfgs=args.skip_lbfgs)

            pred_u = predict_on_grid(model, x_test, t_test)
            exact_u = get_reference_solution(nu_val, x_test, t_test)
            rel_l2 = np.linalg.norm(pred_u - exact_u) / (np.linalg.norm(exact_u) + 1e-12)
            run_errors.append(rel_l2)
            print(f"nu={nu_val:.3f} run={run_id} relL2={rel_l2:.4e}")

        rows.append((nu_val, float(np.mean(run_errors)), float(np.std(run_errors))))

    out_csv = os.path.join(args.save_dir, "bayes_paper_protocol_summary.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("nu,mean_rel_l2,std_rel_l2\n")
        for nu_val, mean_l2, std_l2 in rows:
            f.write(f"{nu_val:.6f},{mean_l2:.8e},{std_l2:.8e}\n")
    print(f"Saved summary: {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Burgers NAS-PINN with Bayesian Optimization")
    parser.add_argument("--nu", type=float, default=0.01 / np.pi, help="viscosity coefficient")
    parser.add_argument("--multi-nu", action="store_true", help="run three viscosity values and save comparison")
    parser.add_argument("--nu-list", type=str, default="0.01,0.04,0.07", help="comma-separated viscosities for --multi-nu")
    parser.add_argument("--epochs", type=int, default=15000, help="final training epochs")
    parser.add_argument("--skip-lbfgs", action="store_true", help="skip L-BFGS in final training")
    parser.add_argument("--save-dir", type=str, default="results/burgers/bayesian", help="output directory")
    parser.add_argument("--checkpoint", type=str, default="bayes_checkpoint_last.pth", help="checkpoint filename")

    parser.add_argument("--bo-init-points", type=int, default=2)
    parser.add_argument("--bo-iters", type=int, default=8)
    parser.add_argument("--bo-epochs", type=int, default=1500)

    parser.add_argument("--paper-protocol", action="store_true")
    parser.add_argument("--paper-nus", type=str, default="0.1,0.07,0.04")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--train-nt", type=int, default=21)
    parser.add_argument("--train-nx", type=int, default=250)
    parser.add_argument("--test-nt", type=int, default=21)
    parser.add_argument("--test-nx", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.paper_protocol:
        run_paper_protocol(args)
    elif args.multi_nu:
        run_multi_nu(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
