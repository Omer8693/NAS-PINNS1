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
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
interactive_plots = os.environ.get("DISPLAY") is not None

x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 2.0

lambda_pde = 1.0
lambda_ic = 100.0
lambda_bc = 10.0
PSO_SPAN = 0.25


def finalize_plot(save_path):
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {save_path}")
    if interactive_plots:
        plt.show()
    else:
        plt.close()


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class MixedOp(nn.Module):
    def __init__(self, in_features, out_features, mask_levels=None):
        super().__init__()
        if mask_levels is None:
            mask_levels = [16, 32, 48, 64, 96, 128]
        self.mask_levels = mask_levels
        self.n_masks = len(mask_levels)

        self.ops = nn.ModuleList(
            [
                nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features),
                nn.Sequential(nn.Linear(in_features, out_features), nn.Tanh()),
                nn.Sequential(nn.Linear(in_features, out_features), SinActivation()),
            ]
        )
        self.n_ops = len(self.ops)
        self.alpha = nn.Parameter(torch.randn(self.n_ops + self.n_masks) * 0.1)

    def relaxed_op(self, x):
        op_weights = F.softmax(self.alpha[: self.n_ops], dim=0)
        return sum(w * op(x) for w, op in zip(op_weights, self.ops))

    def forward(self, x):
        mixed = self.relaxed_op(x)
        mask_weights = torch.sigmoid(self.alpha[self.n_ops :])
        dim = mixed.shape[-1]
        out = 0.0
        for j, keep in enumerate(self.mask_levels):
            k = min(keep, dim)
            mask = torch.zeros(dim, device=device)
            mask[:k] = 1.0
            out += mask_weights[j] * (mixed * mask.unsqueeze(0))
        return out


class NASPINNAdvection(nn.Module):
    def __init__(self, layers=4, base_neurons=128, mask_levels=None):
        super().__init__()
        if mask_levels is None:
            mask_levels = [16, 32, 48, 64, 96, 128]
        dims = [2] + [base_neurons] * (layers - 1) + [1]
        self.layers = nn.ModuleList(
            [MixedOp(dims[i], dims[i + 1], mask_levels=mask_levels) for i in range(layers)]
        )

    def forward(self, xt):
        h = xt
        for layer in self.layers:
            h = layer(h)
        return h


def exact_solution(beta, x, t):
    return 0.8 * torch.sin(4.0 * np.pi * (x - beta * t) + np.pi / 4.0)


def sample_points_uniform(nt=40, nx=120):
    x_vals = torch.linspace(x_min, x_max, nx, device=device)
    t_vals = torch.linspace(t_min, t_max, nt, device=device)

    X, T = torch.meshgrid(x_vals, t_vals, indexing="ij")
    x_c = X.reshape(-1, 1)
    t_c = T.reshape(-1, 1)

    x_ic = x_vals.unsqueeze(1)
    t_ic = torch.zeros_like(x_ic)

    t_bc = t_vals.unsqueeze(1)
    x_l = torch.zeros_like(t_bc)
    x_r = torch.ones_like(t_bc)

    return (x_c, t_c), (x_ic, t_ic), (x_l, t_bc), (x_r, t_bc)


def pde_residual(model, x, t, beta):
    xt = torch.cat([x, t], dim=1).requires_grad_(True)
    u = model(xt)
    grads = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    f = u_t + beta * u_x
    return torch.mean(f.pow(2))


def ic_loss(model, x, t, beta):
    xt = torch.cat([x, t], dim=1)
    u_pred = model(xt)
    u_true = exact_solution(beta, x, t)
    return torch.mean((u_pred - u_true).pow(2))


def bc_loss(model, x_l, t_l, x_r, t_r):
    u_l = model(torch.cat([x_l, t_l], dim=1))
    u_r = model(torch.cat([x_r, t_r], dim=1))
    return torch.mean((u_l - u_r).pow(2))


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


class PSOWeightProblem(ElementwiseProblem):
    def __init__(self, lower, upper, objective):
        super().__init__(n_var=int(lower.size), n_obj=1, n_constr=0, xl=lower, xu=upper)
        self.objective = objective

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = float(self.objective(x))


def train_model(
    model,
    points,
    beta,
    epochs=12000,
    inner_lr=1e-3,
    outer_lr=3e-4,
    outer_every=5,
    skip_lbfgs=False,
    use_pso=False,
    pso_iters=8,
    pso_swarm=16,
    pso_span=PSO_SPAN,
    pso_seed=None,
    return_stage_info=False,
):
    (x_c, t_c), (x_ic, t_ic), (x_l, t_l), (x_r, t_r) = points

    opt_inner = optim.Adam(model.parameters(), lr=inner_lr)
    arch_params = [l.alpha for l in model.layers]
    opt_outer = optim.Adam(arch_params, lr=outer_lr)

    training_start = time.perf_counter()
    loss_history = []
    best_adam_loss = float("inf")
    best_adam_state = clone_model_state(model)

    for ep in range(epochs):
        opt_inner.zero_grad()
        l_pde = pde_residual(model, x_c, t_c, beta)
        l_ic = ic_loss(model, x_ic, t_ic, beta)
        l_bc = bc_loss(model, x_l, t_l, x_r, t_r)
        loss = lambda_pde * l_pde + lambda_ic * l_ic + lambda_bc * l_bc
        loss_value = float(loss.item())
        loss.backward()
        opt_inner.step()
        loss_history.append(loss_value)

        if ep % outer_every == 0:
            opt_outer.zero_grad()
            l_pde_o = pde_residual(model, x_c, t_c, beta)
            l_ic_o = ic_loss(model, x_ic, t_ic, beta)
            l_bc_o = bc_loss(model, x_l, t_l, x_r, t_r)
            loss_o = lambda_pde * l_pde_o + lambda_ic * l_ic_o + lambda_bc * l_bc_o
            loss_o.backward()
            opt_outer.step()

        if loss_value < best_adam_loss:
            best_adam_loss = loss_value
            best_adam_state = clone_model_state(model)

        if ep % 2000 == 0 or ep == epochs - 1:
            print(f"Adam [{ep:5d}] loss: {loss_value:.4e}")

    adam_state = best_adam_state
    adam_time = time.perf_counter() - training_start
    stage_states = {"adam": adam_state}
    stage_losses = {"adam": float(best_adam_loss)}
    stage_times = {"adam": adam_time}

    def refinement_objective():
        return (
            lambda_pde * pde_residual(model, x_c, t_c, beta)
            + lambda_ic * ic_loss(model, x_ic, t_ic, beta)
            + lambda_bc * bc_loss(model, x_l, t_l, x_r, t_r)
        )

    if not skip_lbfgs:
        model.load_state_dict(adam_state)
        lbfgs_start = time.perf_counter()
        print("\nL-BFGS refinement...")
        lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=1500, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            total = refinement_objective()
            total.backward()
            return total

        lbfgs.step(closure)
        lbfgs_loss = float(refinement_objective().item())
        stage_states["lbfgs"] = clone_model_state(model)
        stage_losses["lbfgs"] = lbfgs_loss
        stage_times["lbfgs"] = adam_time + (time.perf_counter() - lbfgs_start)

    if use_pso:
        model.load_state_dict(adam_state)
        pso_start = time.perf_counter()
        print("\nPSO refinement...")
        center = flatten_model_params(model)
        lower = center - pso_span
        upper = center + pso_span

        def pso_objective(flat):
            set_model_from_flat_vector(model, flat)
            return float(refinement_objective().item())

        pso_problem = PSOWeightProblem(lower, upper, pso_objective)
        pso_algorithm = PSO(pop_size=pso_swarm)
        pso_result = minimize(
            pso_problem,
            pso_algorithm,
            termination=("n_gen", pso_iters),
            seed=pso_seed,
            verbose=False,
        )
        best_flat = np.array(pso_result.X, dtype=np.float64)
        best_loss = float(np.array(pso_result.F).reshape(-1)[0])
        set_model_from_flat_vector(model, best_flat)
        stage_states["pso"] = clone_model_state(model)
        stage_losses["pso"] = best_loss
        stage_times["pso"] = adam_time + (time.perf_counter() - pso_start)
        print(f"PSO best objective: {best_loss:.4e}")

    if return_stage_info:
        return {
            "loss_history": loss_history,
            "stage_states": stage_states,
            "stage_losses": stage_losses,
            "stage_times": stage_times,
        }
    return loss_history


def predict_on_grid(model, x_vals, t_vals):
    Xg, Tg = torch.meshgrid(x_vals, t_vals, indexing="ij")
    XT = torch.cat([Xg.reshape(-1, 1), Tg.reshape(-1, 1)], dim=1)
    with torch.no_grad():
        u_pred = model(XT).cpu().numpy().reshape(len(x_vals), len(t_vals))
    return u_pred


def plot_loss_curve(losses, save_path, title):
    plt.figure(figsize=(7, 4))
    plt.plot(losses, linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    finalize_plot(save_path)


def plot_advection_comparison(x_vals_np, t_vals_np, exact_u, pred_u, save_path):
    Xg, Tg = np.meshgrid(x_vals_np, t_vals_np, indexing="ij")
    err = np.abs(pred_u - exact_u)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    cs0 = axes[0].contourf(Xg, Tg, exact_u, levels=60, cmap="YlGnBu")
    axes[0].set_title("Exact")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(cs0, ax=axes[0])

    cs1 = axes[1].contourf(Xg, Tg, pred_u, levels=60, cmap="YlGnBu")
    axes[1].set_title("Predicted")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    fig.colorbar(cs1, ax=axes[1])

    cs2 = axes[2].contourf(Xg, Tg, err, levels=60, cmap="YlGnBu")
    axes[2].set_title("|Pred-Exact|")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")
    fig.colorbar(cs2, ax=axes[2])

    finalize_plot(save_path)


def plot_advection_heatmap(x_vals_np, t_vals_np, pred_u, save_path):
    Xg, Tg = np.meshgrid(x_vals_np, t_vals_np, indexing="ij")
    plt.figure(figsize=(8, 5))
    cs = plt.contourf(Xg, Tg, pred_u, levels=60, cmap="YlGnBu")
    plt.colorbar(cs, label="u(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Advection prediction heatmap")
    plt.tight_layout()
    finalize_plot(save_path)


def _nearest_time_index(t_vals_np, t_query):
    return int(np.argmin(np.abs(t_vals_np - float(t_query))))


def plot_advection_time_slices(x_vals_np, t_vals_np, exact_u, pred_u, slice_times, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(slice_times), 1)))

    for idx, t_sel in enumerate(slice_times):
        t_idx = _nearest_time_index(t_vals_np, t_sel)
        t_used = float(t_vals_np[t_idx])
        c = colors[idx % len(colors)]
        label = f"t={t_used:.2f}"
        axes[0].plot(x_vals_np, exact_u[:, t_idx], color=c, linewidth=1.8, label=label)
        axes[1].plot(x_vals_np, pred_u[:, t_idx], color=c, linewidth=1.8, label=label)

    axes[0].set_title("Exact Time Slices")
    axes[1].set_title("Predicted Time Slices")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle("Advection1D: Exact vs Predicted Time Slices")
    finalize_plot(save_path)


def save_advection_time_slice_table(path, t_vals_np, exact_u, pred_u, slice_times):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_requested", "t_used", "rel_l2_slice", "mae_slice", "max_abs_slice"])
        for t_sel in slice_times:
            t_idx = _nearest_time_index(t_vals_np, t_sel)
            t_used = float(t_vals_np[t_idx])
            exact_slice = exact_u[:, t_idx]
            pred_slice = pred_u[:, t_idx]
            abs_err = np.abs(pred_slice - exact_slice)
            rel_l2 = float(np.linalg.norm(pred_slice - exact_slice) / (np.linalg.norm(exact_slice) + 1e-12))
            mae = float(np.mean(abs_err))
            max_abs = float(np.max(abs_err))
            writer.writerow([float(t_sel), t_used, f"{rel_l2:.8e}", f"{mae:.8e}", f"{max_abs:.8e}"])


def build_stage_loss_curve(base_history, stage_name, stage_losses):
    if not base_history:
        return []
    if stage_name == "adam":
        return list(base_history)
    if stage_name in stage_losses:
        return list(base_history) + [float(stage_losses[stage_name])]
    return list(base_history)


def save_stage_outputs(model, args, stage_name, stage_dir, loss_history, stage_losses, run_time_seconds):
    os.makedirs(stage_dir, exist_ok=True)
    stage_loss = build_stage_loss_curve(loss_history, stage_name, stage_losses)
    plot_loss_curve(
        stage_loss,
        os.path.join(stage_dir, "loss_curve.png"),
        title=f"Advection NAS-PINN Loss ({stage_name.upper()}, beta={args.beta:.3f})",
    )

    x_test = torch.linspace(x_min, x_max, args.test_nx, device=device)
    t_test = torch.linspace(t_min, t_max, args.test_nt, device=device)
    pred_u = predict_on_grid(model, x_test, t_test)
    X, T = torch.meshgrid(x_test, t_test, indexing="ij")
    exact_u = exact_solution(args.beta, X, T).detach().cpu().numpy()
    rel_l2 = np.linalg.norm(pred_u - exact_u) / (np.linalg.norm(exact_u) + 1e-12)

    x_np = x_test.detach().cpu().numpy()
    t_np = t_test.detach().cpu().numpy()
    plot_advection_comparison(x_np, t_np, exact_u, pred_u, os.path.join(stage_dir, "result_comparison.png"))
    plot_advection_heatmap(x_np, t_np, pred_u, os.path.join(stage_dir, "advection_heatmap.png"))
    plot_advection_time_slices(
        x_np,
        t_np,
        exact_u,
        pred_u,
        args.slice_times,
        os.path.join(stage_dir, "advection_time_slices_exact_vs_pred.png"),
    )
    save_advection_time_slice_table(
        os.path.join(stage_dir, "time_slice_comparison.csv"),
        t_np,
        exact_u,
        pred_u,
        args.slice_times,
    )

    with open(os.path.join(stage_dir, "l2_error.txt"), "w", encoding="utf-8") as f:
        f.write(f"stage,{stage_name}\nbeta,{args.beta:.6f}\nrel_l2,{rel_l2:.8e}\n")
    with open(os.path.join(stage_dir, "run_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"run_time_seconds,{run_time_seconds:.6f}\n")
    with open(os.path.join(stage_dir, "metrics.csv"), "w", encoding="utf-8") as f:
        f.write("method,stage,beta,seed,rel_l2,run_time_seconds\n")
        f.write(f"naspinn,{stage_name},{args.beta:.6f},{args.seed},{rel_l2:.8e},{run_time_seconds:.6f}\n")
    return float(rel_l2)


def copy_stage_to_root(stage_dir, root_dir):
    os.makedirs(root_dir, exist_ok=True)
    for name in [
        "loss_curve.png",
        "result_comparison.png",
        "advection_heatmap.png",
        "advection_time_slices_exact_vs_pred.png",
        "time_slice_comparison.csv",
        "l2_error.txt",
        "run_time.txt",
        "metrics.csv",
    ]:
        src = os.path.join(stage_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(root_dir, name))


def apply_stage_flags(args):
    if args.stage == "adam":
        args.skip_lbfgs = True
        args.use_pso = False
    elif args.stage == "lbfgs":
        args.skip_lbfgs = False
        args.use_pso = False
    elif args.stage == "pso":
        args.skip_lbfgs = False
        args.use_pso = True


def run_single(args):
    run_start = time.perf_counter()
    os.makedirs(args.save_dir, exist_ok=True)
    apply_stage_flags(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = NASPINNAdvection(layers=args.layers, base_neurons=args.base_neurons).to(device)
    points = sample_points_uniform(nt=args.train_nt, nx=args.train_nx)
    train_info = train_model(
        model,
        points,
        beta=args.beta,
        epochs=args.epochs,
        skip_lbfgs=args.skip_lbfgs,
        use_pso=args.use_pso,
        pso_iters=args.pso_iters,
        pso_swarm=args.pso_swarm,
        pso_span=args.pso_span,
        pso_seed=args.seed,
        return_stage_info=True,
    )

    loss_history = train_info["loss_history"]
    stage_states = train_info["stage_states"]
    stage_losses = train_info["stage_losses"]
    stage_times = train_info["stage_times"]

    stage_order = ["adam", "lbfgs", "pso"]
    stage_results = []
    for stage_name in stage_order:
        if stage_name not in stage_states:
            continue
        model_stage = NASPINNAdvection(layers=args.layers, base_neurons=args.base_neurons).to(device)
        model_stage.load_state_dict(stage_states[stage_name])
        stage_dir = os.path.join(args.save_dir, f"stage_{stage_name}")
        rel_l2 = save_stage_outputs(
            model_stage,
            args,
            stage_name,
            stage_dir,
            loss_history,
            stage_losses,
            float(stage_times.get(stage_name, 0.0)),
        )
        stage_results.append((stage_name, rel_l2, float(stage_times.get(stage_name, 0.0))))

    if not stage_results:
        raise RuntimeError("No stage output was produced.")

    best_stage_name, best_rel_l2, best_stage_time = min(stage_results, key=lambda x: x[1])
    best_stage_dir = os.path.join(args.save_dir, f"stage_{best_stage_name}")
    copy_stage_to_root(best_stage_dir, args.save_dir)
    stage_best_dir = os.path.join(args.save_dir, "stage_best")
    copy_stage_to_root(best_stage_dir, stage_best_dir)
    with open(os.path.join(stage_best_dir, "selected_stage.txt"), "w", encoding="utf-8") as f:
        f.write(f"{best_stage_name}\n")

    with open(os.path.join(args.save_dir, "stage_summary.csv"), "w", encoding="utf-8") as f:
        f.write("stage,beta,seed,rel_l2,run_time_seconds\n")
        for stage_name, rel_l2, stage_time in stage_results:
            f.write(f"{stage_name},{args.beta:.6f},{args.seed},{rel_l2:.8e},{stage_time:.6f}\n")
        f.write(f"best:{best_stage_name},{args.beta:.6f},{args.seed},{best_rel_l2:.8e},{best_stage_time:.6f}\n")

    run_time = time.perf_counter() - run_start
    with open(os.path.join(args.save_dir, "run_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"run_time_seconds,{run_time:.6f}\n")
    with open(os.path.join(args.save_dir, "metrics.csv"), "w", encoding="utf-8") as f:
        f.write("method,stage,beta,seed,rel_l2,run_time_seconds\n")
        f.write(f"naspinn,best:{best_stage_name},{args.beta:.6f},{args.seed},{best_rel_l2:.8e},{run_time:.6f}\n")

    print(f"Selected best stage: {best_stage_name} (rel_l2={best_rel_l2:.4e})")
    print(f"Run time: {run_time:.2f} s")
    return float(best_rel_l2), float(run_time)


def run_multi_beta(args):
    beta_values = [float(v.strip()) for v in args.beta_list.split(",") if v.strip()]
    summary = []
    base_dir = args.save_dir
    os.makedirs(base_dir, exist_ok=True)

    for idx, beta_val in enumerate(beta_values):
        args_local = argparse.Namespace(**vars(args))
        args_local.beta = beta_val
        args_local.seed = args.seed + idx
        args_local.save_dir = os.path.join(base_dir, f"beta_{beta_val:.3f}")
        rel_l2, run_time = run_single(args_local)
        summary.append((beta_val, rel_l2, run_time))

    csv_path = os.path.join(base_dir, "beta_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("beta,rel_l2,run_time_seconds\n")
        for beta_val, rel_l2, run_time in summary:
            f.write(f"{beta_val:.6f},{rel_l2:.8e},{run_time:.6f}\n")
    print(f"Saved summary: {csv_path}")

    betas = [row[0] for row in summary]
    errs = [row[1] for row in summary]
    plt.figure(figsize=(7, 4))
    plt.plot(betas, errs, marker="o", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Advection speed (beta)")
    plt.ylabel("Relative L2 Error")
    plt.title("Advection NAS-PINN: Beta Comparison")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    finalize_plot(os.path.join(base_dir, "beta_comparison.png"))


def run_paper_protocol(args):
    beta_values = [float(v.strip()) for v in args.paper_betas.split(",") if v.strip()]
    rows = []
    for beta in beta_values:
        run_errors = []
        for run_id in range(1, args.repeats + 1):
            args_local = argparse.Namespace(**vars(args))
            args_local.beta = beta
            args_local.seed = args.seed + run_id
            args_local.save_dir = os.path.join(args.save_dir, f"paper_beta_{beta:.3f}", f"run_{run_id:02d}")
            rel_l2, _ = run_single(args_local)
            run_errors.append(rel_l2)
        rows.append((beta, float(np.mean(run_errors)), float(np.std(run_errors))))

    out_csv = os.path.join(args.save_dir, "paper_protocol_summary.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("beta,mean_rel_l2,std_rel_l2\n")
        for beta, mean_l2, std_l2 in rows:
            f.write(f"{beta:.6f},{mean_l2:.8e},{std_l2:.8e}\n")
    print(f"Saved summary: {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="NAS-PINN Advection solver")
    parser.add_argument("--beta", type=float, default=1.0, help="single-run advection speed")
    parser.add_argument("--multi-beta", action="store_true", help="run multiple beta values from --beta-list")
    parser.add_argument("--beta-list", type=str, default="1.0,0.5,0.1", help="comma-separated beta values")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["adam", "lbfgs", "pso"],
        default="lbfgs",
        help="stage mode: adam | lbfgs | pso",
    )
    parser.add_argument("--skip-lbfgs", action="store_true", help="skip L-BFGS refinement")
    parser.add_argument("--use-pso", action="store_true", help="enable PSO refinement from Adam checkpoint")
    parser.add_argument("--pso-iters", type=int, default=8, help="PSO max iterations")
    parser.add_argument("--pso-swarm", type=int, default=16, help="PSO swarm size")
    parser.add_argument("--pso-span", type=float, default=PSO_SPAN, help="PSO search span around current weights")
    parser.add_argument("--epochs", type=int, default=12000, help="Adam epochs")
    parser.add_argument("--layers", type=int, default=4, help="number of layers in NAS model")
    parser.add_argument("--base-neurons", type=int, default=128, help="base hidden width")
    parser.add_argument("--train-nt", type=int, default=40, help="train grid points along t-axis")
    parser.add_argument("--train-nx", type=int, default=120, help="train grid points along x-axis")
    parser.add_argument("--test-nt", type=int, default=40, help="test grid points along t-axis")
    parser.add_argument("--test-nx", type=int, default=120, help="test grid points along x-axis")
    parser.add_argument(
        "--slice-times",
        type=str,
        default="0,0.5,1.0,1.5,2.0",
        help="comma-separated time slices for exact-vs-pred line comparison",
    )
    parser.add_argument("--paper-protocol", action="store_true", help="run paper-style repeated evaluation")
    parser.add_argument("--paper-betas", type=str, default="1.0,0.5,0.1", help="beta list for paper protocol")
    parser.add_argument("--repeats", type=int, default=5, help="repeat count for paper protocol")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--save-dir", type=str, default="results/advection/naspinn", help="output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    args.slice_times = [float(v.strip()) for v in args.slice_times.split(",") if v.strip()]
    os.makedirs(args.save_dir, exist_ok=True)
    if args.paper_protocol:
        run_paper_protocol(args)
    elif args.multi_beta:
        run_multi_beta(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
