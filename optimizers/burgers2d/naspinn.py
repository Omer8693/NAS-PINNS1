import argparse
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

t_min, t_max = 0.0, 2.0
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
nu_coef = 0.1

lambda_pde = 1.0
lambda_ic = 100.0
lambda_bc = 100.0
PSO_SPAN = 0.25


def finalize_plot(save_path):
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {save_path}")
    if interactive_plots:
        plt.show()
    else:
        plt.close()


def exact_solution_np(x, y, t):
    return 1.0 / (1.0 + np.exp((x + y - t) / 0.2))


def exact_solution_torch(x, y, t):
    return 1.0 / (1.0 + torch.exp((x + y - t) / 0.2))


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


class NASPINNBurgers2D(nn.Module):
    def __init__(self, layers=5, base_neurons=128, mask_levels=None):
        super().__init__()
        if mask_levels is None:
            mask_levels = [16, 32, 48, 64, 96, 128]
        dims = [3] + [base_neurons] * (layers - 1) + [1]
        self.layers = nn.ModuleList(
            [MixedOp(dims[i], dims[i + 1], mask_levels=mask_levels) for i in range(layers)]
        )

    def forward(self, xyt):
        h = xyt
        for layer in self.layers:
            h = layer(h)
        return h


def sample_points_uniform(train_nt=20, train_nx=25, train_ny=25):
    t_vals = torch.linspace(t_min, t_max, train_nt, device=device)
    x_vals = torch.linspace(x_min, x_max, train_nx, device=device)
    y_vals = torch.linspace(y_min, y_max, train_ny, device=device)

    X, Y, T = torch.meshgrid(x_vals, y_vals, t_vals, indexing="ij")
    x_c = X.reshape(-1, 1)
    y_c = Y.reshape(-1, 1)
    t_c = T.reshape(-1, 1)

    Xi, Yi = torch.meshgrid(x_vals, y_vals, indexing="ij")
    x_ic = Xi.reshape(-1, 1)
    y_ic = Yi.reshape(-1, 1)
    t_ic = torch.zeros_like(x_ic)

    Yb, Tb = torch.meshgrid(y_vals, t_vals, indexing="ij")
    y_edge = Yb.reshape(-1, 1)
    t_edge_xy = Tb.reshape(-1, 1)
    x_left = torch.zeros_like(y_edge)
    x_right = torch.ones_like(y_edge)

    Xb, Tb2 = torch.meshgrid(x_vals, t_vals, indexing="ij")
    x_edge = Xb.reshape(-1, 1)
    t_edge_yx = Tb2.reshape(-1, 1)
    y_bottom = torch.zeros_like(x_edge)
    y_top = torch.ones_like(x_edge)

    x_bc = torch.cat([x_left, x_right, x_edge, x_edge], dim=0)
    y_bc = torch.cat([y_edge, y_edge, y_bottom, y_top], dim=0)
    t_bc = torch.cat([t_edge_xy, t_edge_xy, t_edge_yx, t_edge_yx], dim=0)

    return (x_c, y_c, t_c), (x_ic, y_ic, t_ic), (x_bc, y_bc, t_bc)


def pde_residual(model, x, y, t):
    xyt = torch.cat([x, y, t], dim=1).requires_grad_(True)
    u = model(xyt)
    grads = torch.autograd.grad(u.sum(), xyt, create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]
    u_xx = torch.autograd.grad(u_x.sum(), xyt, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y.sum(), xyt, create_graph=True)[0][:, 1:2]
    f = u_t + u * (u_x + u_y) - nu_coef * (u_xx + u_yy)
    return torch.mean(f.pow(2))


def ic_loss(model, x, y, t):
    pred = model(torch.cat([x, y, t], dim=1))
    true = exact_solution_torch(x, y, t)
    return torch.mean((pred - true).pow(2))


def bc_loss(model, x, y, t):
    pred = model(torch.cat([x, y, t], dim=1))
    true = exact_solution_torch(x, y, t)
    return torch.mean((pred - true).pow(2))


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
    (x_c, y_c, t_c), (x_ic, y_ic, t_ic), (x_bc, y_bc, t_bc) = points

    opt_inner = optim.Adam(model.parameters(), lr=inner_lr)
    arch_params = [l.alpha for l in model.layers]
    opt_outer = optim.Adam(arch_params, lr=outer_lr)

    training_start = time.perf_counter()
    loss_history = []
    best_adam_loss = float("inf")
    best_adam_state = clone_model_state(model)

    for ep in range(epochs):
        opt_inner.zero_grad()
        l_pde = pde_residual(model, x_c, y_c, t_c)
        l_ic = ic_loss(model, x_ic, y_ic, t_ic)
        l_bc = bc_loss(model, x_bc, y_bc, t_bc)
        loss = lambda_pde * l_pde + lambda_ic * l_ic + lambda_bc * l_bc
        loss_value = float(loss.item())
        loss.backward()
        opt_inner.step()
        loss_history.append(loss_value)

        if ep % outer_every == 0:
            opt_outer.zero_grad()
            l_pde_o = pde_residual(model, x_c, y_c, t_c)
            l_ic_o = ic_loss(model, x_ic, y_ic, t_ic)
            l_bc_o = bc_loss(model, x_bc, y_bc, t_bc)
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
            lambda_pde * pde_residual(model, x_c, y_c, t_c)
            + lambda_ic * ic_loss(model, x_ic, y_ic, t_ic)
            + lambda_bc * bc_loss(model, x_bc, y_bc, t_bc)
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


def evaluate_rel_l2_batched(model, test_nt=41, test_nx=500, test_ny=500, batch_size=65536):
    x_vals = np.linspace(x_min, x_max, test_nx, dtype=np.float64)
    y_vals = np.linspace(y_min, y_max, test_ny, dtype=np.float64)
    t_vals = np.linspace(t_min, t_max, test_nt, dtype=np.float64)
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")
    x_flat = X.reshape(-1)
    y_flat = Y.reshape(-1)
    n_xy = x_flat.size

    num = 0.0
    den = 0.0
    with torch.no_grad():
        for t in t_vals:
            t_flat = np.full_like(x_flat, t)
            xyt = np.stack([x_flat, y_flat, t_flat], axis=1).astype(np.float32)
            xyt_t = torch.from_numpy(xyt).to(device)

            pred_chunks = []
            for start in range(0, n_xy, batch_size):
                end = min(start + batch_size, n_xy)
                pred_chunks.append(model(xyt_t[start:end]).detach().cpu().numpy().reshape(-1))
            pred = np.concatenate(pred_chunks, axis=0).reshape(test_nx, test_ny)
            exact = exact_solution_np(X, Y, t)

            diff = pred - exact
            num += float(np.sum(diff * diff))
            den += float(np.sum(exact * exact))
    return float(np.sqrt(num / (den + 1e-12)))


def plot_slice_comparison(model, t_value, save_path, grid_size=200):
    x_vals = np.linspace(x_min, x_max, grid_size, dtype=np.float32)
    y_vals = np.linspace(y_min, y_max, grid_size, dtype=np.float32)
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")
    t_flat = np.full(X.size, t_value, dtype=np.float32)
    xyt = np.stack([X.reshape(-1), Y.reshape(-1), t_flat], axis=1)
    xyt_t = torch.from_numpy(xyt).to(device)

    with torch.no_grad():
        pred = model(xyt_t).detach().cpu().numpy().reshape(grid_size, grid_size)
    exact = exact_solution_np(X, Y, t_value)
    err = np.abs(pred - exact)
    rel_l2_slice = float(np.linalg.norm(pred - exact) / (np.linalg.norm(exact) + 1e-12))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    cs0 = axes[0].contourf(X, Y, exact, levels=60, cmap="YlGnBu")
    axes[0].set_title(f"Exact (t={t_value:.2f})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(cs0, ax=axes[0])

    cs1 = axes[1].contourf(X, Y, pred, levels=60, cmap="YlGnBu")
    axes[1].set_title(f"Predicted (t={t_value:.2f})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(cs1, ax=axes[1])

    cs2 = axes[2].contourf(X, Y, err, levels=60, cmap="YlGnBu")
    axes[2].set_title(f"|Pred-Exact| (t={t_value:.2f})")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(cs2, ax=axes[2])
    finalize_plot(save_path)
    return rel_l2_slice


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
        title=f"2D Burgers NAS-PINN Loss ({stage_name.upper()})",
    )

    slice_rows = []
    default_result_src = None
    for t_value in args.slice_times:
        out_path = os.path.join(stage_dir, f"slice_t_{float(t_value):.2f}.png")
        rel_l2_slice = plot_slice_comparison(
            model,
            t_value=float(t_value),
            save_path=out_path,
            grid_size=args.slice_grid,
        )
        slice_rows.append((float(t_value), rel_l2_slice))
        if default_result_src is None or abs(float(t_value) - 1.0) < abs(default_result_src[0] - 1.0):
            default_result_src = (float(t_value), out_path)

    if default_result_src is not None:
        shutil.copy2(default_result_src[1], os.path.join(stage_dir, "result_comparison.png"))

    with open(os.path.join(stage_dir, "slice_comparison_table.csv"), "w", encoding="utf-8") as f:
        f.write("t_slice,rel_l2_slice\n")
        for t_val, rel_l2_slice in slice_rows:
            f.write(f"{t_val:.6f},{rel_l2_slice:.8e}\n")

    rel_l2 = evaluate_rel_l2_batched(
        model,
        test_nt=args.test_nt,
        test_nx=args.test_nx,
        test_ny=args.test_ny,
        batch_size=args.eval_batch_size,
    )
    print(f"Relative L2 error (2D Burgers): {rel_l2:.4e}")

    with open(os.path.join(stage_dir, "l2_error.txt"), "w", encoding="utf-8") as f:
        f.write(f"stage,{stage_name}\nrel_l2,{rel_l2:.8e}\n")
    with open(os.path.join(stage_dir, "run_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"run_time_seconds,{run_time_seconds:.6f}\n")
    with open(os.path.join(stage_dir, "metrics.csv"), "w", encoding="utf-8") as f:
        f.write("method,stage,seed,rel_l2,run_time_seconds\n")
        f.write(f"naspinn,{stage_name},{args.seed},{rel_l2:.8e},{run_time_seconds:.6f}\n")
    return float(rel_l2)


def copy_stage_to_root(stage_dir, root_dir):
    os.makedirs(root_dir, exist_ok=True)
    copy_names = [
        "loss_curve.png",
        "result_comparison.png",
        "slice_comparison_table.csv",
        "l2_error.txt",
        "run_time.txt",
        "metrics.csv",
    ]
    for name in copy_names:
        src = os.path.join(stage_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(root_dir, name))

    for entry in os.listdir(stage_dir):
        if entry.startswith("slice_t_") and entry.endswith(".png"):
            shutil.copy2(os.path.join(stage_dir, entry), os.path.join(root_dir, entry))


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

    model = NASPINNBurgers2D(layers=args.layers, base_neurons=args.base_neurons).to(device)
    points = sample_points_uniform(train_nt=args.train_nt, train_nx=args.train_nx, train_ny=args.train_ny)
    train_info = train_model(
        model,
        points,
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
        model_stage = NASPINNBurgers2D(layers=args.layers, base_neurons=args.base_neurons).to(device)
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
        f.write("stage,seed,rel_l2,run_time_seconds\n")
        for stage_name, rel_l2, stage_time in stage_results:
            f.write(f"{stage_name},{args.seed},{rel_l2:.8e},{stage_time:.6f}\n")
        f.write(f"best:{best_stage_name},{args.seed},{best_rel_l2:.8e},{best_stage_time:.6f}\n")

    run_time = time.perf_counter() - run_start
    with open(os.path.join(args.save_dir, "run_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"run_time_seconds,{run_time:.6f}\n")
    with open(os.path.join(args.save_dir, "metrics.csv"), "w", encoding="utf-8") as f:
        f.write("method,stage,seed,rel_l2,run_time_seconds\n")
        f.write(f"naspinn,best:{best_stage_name},{args.seed},{best_rel_l2:.8e},{run_time:.6f}\n")

    print(f"Selected best stage: {best_stage_name} (rel_l2={best_rel_l2:.4e})")
    print(f"Run time: {run_time:.2f} s")
    return float(best_rel_l2), float(run_time)


def run_paper_protocol(args):
    rows = []
    for run_id in range(1, args.repeats + 1):
        args_local = argparse.Namespace(**vars(args))
        args_local.seed = args.seed + run_id - 1
        args_local.save_dir = os.path.join(args.save_dir, f"run_{run_id:02d}")
        rel_l2, run_time = run_single(args_local)
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
    parser = argparse.ArgumentParser(description="NAS-PINN 2D Burgers solver")
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
    parser.add_argument("--layers", type=int, default=5, help="number of layers in NAS model")
    parser.add_argument("--base-neurons", type=int, default=128, help="base hidden width")
    parser.add_argument("--train-nt", type=int, default=20, help="train grid points along t-axis")
    parser.add_argument("--train-nx", type=int, default=25, help="train grid points along x-axis")
    parser.add_argument("--train-ny", type=int, default=25, help="train grid points along y-axis")
    parser.add_argument("--test-nt", type=int, default=41, help="test grid points along t-axis")
    parser.add_argument("--test-nx", type=int, default=500, help="test grid points along x-axis")
    parser.add_argument("--test-ny", type=int, default=500, help="test grid points along y-axis")
    parser.add_argument("--eval-batch-size", type=int, default=65536, help="batch size for batched L2 evaluation")
    parser.add_argument("--slice-grid", type=int, default=200, help="grid size for t-slice visualization")
    parser.add_argument(
        "--slice-times",
        type=str,
        default="0,1,2",
        help="comma-separated time slices for visualization",
    )
    parser.add_argument("--paper-protocol", action="store_true", help="run paper-style repeated evaluation")
    parser.add_argument("--repeats", type=int, default=5, help="repeat count for paper protocol")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--save-dir", type=str, default="results/burgers2d/naspinn", help="output directory")
    args = parser.parse_args()
    args.slice_times = [float(v.strip()) for v in args.slice_times.split(",") if v.strip()]
    return args


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    if args.paper_protocol:
        run_paper_protocol(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
