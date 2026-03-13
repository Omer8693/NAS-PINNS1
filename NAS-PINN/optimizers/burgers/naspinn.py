import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import time
from scipy.io import loadmat  # referans çözüm için (opsiyonel)
from scipy.integrate import solve_ivp
try:
    from pyswarm import pso
except ImportError:
    pso = None
from .plots import (
    plot_burgers_exact_pred_error,
    plot_burgers_full_exact_vs_pred,
    plot_burgers_heatmap,
    plot_loss_curve,
    plot_burgers_time_slices,
    plot_burgers_time_slices_with_exact,
    should_use_exact_plots,
)

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
interactive_plots = os.environ.get("DISPLAY") is not None


def finalize_plot(save_path):
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {save_path}")
    if interactive_plots:
        plt.show()
    else:
        plt.close()

# ────────────────────────────────────────────────
# Hiperparametreler (makaleye göre tipik Burgers ayarları)
# ────────────────────────────────────────────────
nu = 0.01 / np.pi           # viscosity (makalede ν/π)
N_col = 10000               # collocation points
N_ic  = 200                 # initial condition points
N_bc  = 200                 # boundary points per side
x_min, x_max = -1.0, 1.0
t_min, t_max = 0.0, 1.0

# Loss ağırlıkları (makalede dengelenmiş, pratikte IC/BC yüksek)
lambda_pde = 1.0
lambda_ic  = 100.0
lambda_bc  = 100.0
PSO_SPAN = 0.25

# ────────────────────────────────────────────────
# Mixed Operation – DARTS relaxation + mask (NAS-PINN Eq. 8 & 9)
# ────────────────────────────────────────────────
class MixedOp(nn.Module):
    def __init__(self, in_c, out_c, mask_levels=[20, 40, 64, 96, 128, 192]):
        super().__init__()
        self.mask_levels = mask_levels
        self.n_masks = len(mask_levels)

        # Candidate operations (tanh ve sin Burgers'da etkili)
        self.ops = nn.ModuleList([
            nn.Identity() if in_c == out_c else nn.Linear(in_c, out_c),   # skip / linear
            nn.Sequential(nn.Linear(in_c, out_c), nn.Tanh()),
            nn.Sequential(nn.Linear(in_c, out_c), SinActivation()),              # sin iyi çalışıyor
        ])
        self.n_ops = len(self.ops)

        # α parametreleri (ops + masks)
        total = self.n_ops + self.n_masks
        self.alpha = nn.Parameter(torch.randn(total) * 0.1)

    def relaxed_op(self, x):
        weights = F.softmax(self.alpha[:self.n_ops], dim=0)
        out = sum(w * op(x) for w, op in zip(weights, self.ops))
        return out

    def forward(self, x):
        mixed = self.relaxed_op(x)

        # Mask ile nöron seçimi (Eq. 9 tarzı)
        mask_weights = torch.sigmoid(self.alpha[self.n_ops:])  # NAS-PINN'de genelde softmax ama sigmoid daha stabil
        final = 0.0
        dim = mixed.shape[-1]
        for j, keep in enumerate(self.mask_levels):
            k = min(keep, dim)
            mask = torch.zeros(dim, device=device)
            mask[:k] = 1.0
            masked = mixed * mask.unsqueeze(0)
            final += mask_weights[j] * masked

        return final


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


# ────────────────────────────────────────────────
# NAS-PINN Model (searchable)
# ────────────────────────────────────────────────
class NAS_PINN(nn.Module):
    def __init__(self, layers=4, base_neurons=128, mask_levels=[32,64,96,128,192]):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [2] + [base_neurons] * (layers - 1) + [1]  # input (x,t), output u

        for i in range(layers):
            self.layers.append(MixedOp(dims[i], dims[i+1], mask_levels))

    def forward(self, xt):
        x = xt
        for layer in self.layers:
            x = layer(x)
        return x


# ────────────────────────────────────────────────
# Veri noktaları (Latin Hypercube benzeri uniform)
# ────────────────────────────────────────────────
def sample_points():
    # Collocation
    x_c = torch.rand(N_col, 1, device=device) * (x_max - x_min) + x_min
    t_c = torch.rand(N_col, 1, device=device) * (t_max - t_min) + t_min

    # IC (t=0)
    x_ic = torch.rand(N_ic, 1, device=device) * (x_max - x_min) + x_min
    t_ic = torch.zeros_like(x_ic)

    # BC left & right
    t_bc = torch.rand(N_bc, 1, device=device) * (t_max - t_min) + t_min
    x_left  = torch.full((N_bc, 1), x_min, device=device)
    x_right = torch.full((N_bc, 1), x_max, device=device)

    return (x_c, t_c), (x_ic, t_ic), (x_left, t_bc), (x_right, t_bc)


def sample_points_paper(train_nx=250, train_nt=21):
    x_vals = torch.linspace(x_min, x_max, train_nx, device=device)
    t_vals = torch.linspace(t_min, t_max, train_nt, device=device)

    X, T = torch.meshgrid(x_vals, t_vals, indexing='ij')
    x_c = X.reshape(-1, 1)
    t_c = T.reshape(-1, 1)

    x_ic = x_vals.unsqueeze(1)
    t_ic = torch.zeros_like(x_ic)

    t_bc = t_vals.unsqueeze(1)
    x_left = torch.full_like(t_bc, x_min)
    x_right = torch.full_like(t_bc, x_max)

    return (x_c, t_c), (x_ic, t_ic), (x_left, t_bc), (x_right, t_bc)


# ────────────────────────────────────────────────
# Loss fonksiyonları (makaledeki Eq. 2–5)
# ────────────────────────────────────────────────
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
    xt = torch.cat([x, t], 1)
    u_pred = model(xt)
    u_true = -torch.sin(np.pi * x)
    return torch.mean((u_pred - u_true) ** 2)


def bc_loss(model, x_l, t_l, x_r, t_r):
    xt_l = torch.cat([x_l, t_l], 1)
    xt_r = torch.cat([x_r, t_r], 1)
    u_l = model(xt_l)
    u_r = model(xt_r)
    return torch.mean(u_l ** 2) + torch.mean(u_r ** 2)


def save_checkpoint(path, model, opt_inner, opt_outer, epoch, points):
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_inner_state": opt_inner.state_dict() if opt_inner is not None else None,
        "opt_outer_state": opt_outer.state_dict() if opt_outer is not None else None,
        "points": {
            "x_c": points[0].detach().cpu(),
            "t_c": points[1].detach().cpu(),
            "x_ic": points[2].detach().cpu(),
            "t_ic": points[3].detach().cpu(),
            "x_l": points[4].detach().cpu(),
            "t_l": points[5].detach().cpu(),
            "x_r": points[6].detach().cpu(),
            "t_r": points[7].detach().cpu(),
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
        points = (
            p["x_c"].to(device), p["t_c"].to(device),
            p["x_ic"].to(device), p["t_ic"].to(device),
            p["x_l"].to(device), p["t_l"].to(device),
            p["x_r"].to(device), p["t_r"].to(device),
        )
    else:
        (x_c, t_c), (x_ic, t_ic), (x_l, t_l), (x_r, t_r) = sample_points()
        points = (x_c, t_c, x_ic, t_ic, x_l, t_l, x_r, t_r)

    return ckpt.get("epoch", 0), points


def clone_model_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_model_from_flat_vector(model, flat):
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            n_param = param.numel()
            chunk = torch.from_numpy(flat[offset : offset + n_param]).view_as(param).to(device)
            param.copy_(chunk)
            offset += n_param


def flatten_model_params(model):
    return np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in model.parameters()])


def train_with_resume(model, total_epochs=15000, inner_lr=1e-3, outer_lr=3e-4,
                      outer_every=5, checkpoint_path="checkpoint_last.pth",
                      checkpoint_every=1000, resume=False, skip_lbfgs=False,
                      use_pso=False, pso_iters=8, pso_swarm=16, pso_span=PSO_SPAN,
                      nu_coef=nu, fixed_points=None, return_stage_info=False):
    opt_inner = optim.Adam(model.parameters(), lr=inner_lr)
    arch_params = [l.alpha for l in model.layers]
    opt_outer = optim.Adam(arch_params, lr=outer_lr)

    start_epoch = 0
    if resume and os.path.exists(checkpoint_path):
        start_epoch, points = load_checkpoint(checkpoint_path, model, opt_inner, opt_outer)
        print(f"Resumed from checkpoint: {checkpoint_path} (epoch {start_epoch})")
    else:
        if fixed_points is None:
            (x_c, t_c), (x_ic, t_ic), (x_l, t_l), (x_r, t_r) = sample_points()
        else:
            (x_c, t_c), (x_ic, t_ic), (x_l, t_l), (x_r, t_r) = fixed_points
        points = (x_c, t_c, x_ic, t_ic, x_l, t_l, x_r, t_r)
        if resume:
            print(f"Resume requested but checkpoint not found: {checkpoint_path}")
            print("Starting from scratch...")

    x_c, t_c, x_ic, t_ic, x_l, t_l, x_r, t_r = points

    training_start = time.perf_counter()
    print("Starting Adam + bi-level optimization...")
    loss_history = []
    for epoch in range(start_epoch, total_epochs):
        opt_inner.zero_grad()

        l_pde = pde_residual(model, x_c, t_c, nu_coef)
        l_ic = ic_loss(model, x_ic, t_ic)
        l_bc = bc_loss(model, x_l, t_l, x_r, t_r)
        loss_inner = lambda_pde * l_pde + lambda_ic * l_ic + lambda_bc * l_bc

        loss_inner.backward()
        opt_inner.step()
        loss_history.append(float(loss_inner.item()))

        if epoch % outer_every == 0:
            opt_outer.zero_grad()
            l_pde_outer = pde_residual(model, x_c, t_c, nu_coef)
            l_ic_outer = ic_loss(model, x_ic, t_ic)
            l_bc_outer = bc_loss(model, x_l, t_l, x_r, t_r)
            loss_outer = lambda_pde * l_pde_outer + lambda_ic * l_ic_outer + lambda_bc * l_bc_outer
            loss_outer.backward()
            opt_outer.step()

        if epoch % 2000 == 0:
            print(f"[{epoch:5d}] PDE: {l_pde:.4e}  IC: {l_ic:.4e}  BC: {l_bc:.4e}")

        if ((epoch + 1) % checkpoint_every == 0) or (epoch + 1 == total_epochs):
            save_checkpoint(
                checkpoint_path,
                model,
                opt_inner,
                opt_outer,
                epoch + 1,
                (x_c, t_c, x_ic, t_ic, x_l, t_l, x_r, t_r),
            )

    stage_states = {"adam": clone_model_state(model)}
    stage_losses = {"adam": float(loss_history[-1]) if loss_history else float("nan")}
    stage_times = {"adam": time.perf_counter() - training_start}

    if not skip_lbfgs:
        print("\nL-BFGS refinement...")
        lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=3000, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            lp = pde_residual(model, x_c, t_c, nu_coef)
            li = ic_loss(model, x_ic, t_ic)
            lb = bc_loss(model, x_l, t_l, x_r, t_r)
            total = lambda_pde * lp + lambda_ic * li + lambda_bc * lb
            total.backward()
            return total

        lbfgs.step(closure)
        lbfgs_loss = float(
            (
                lambda_pde * pde_residual(model, x_c, t_c, nu_coef)
                + lambda_ic * ic_loss(model, x_ic, t_ic)
                + lambda_bc * bc_loss(model, x_l, t_l, x_r, t_r)
            ).item()
        )
        stage_states["lbfgs"] = clone_model_state(model)
        stage_losses["lbfgs"] = lbfgs_loss
        stage_times["lbfgs"] = time.perf_counter() - training_start

        save_checkpoint(
            checkpoint_path,
            model,
            opt_inner,
            opt_outer,
            total_epochs,
            (x_c, t_c, x_ic, t_ic, x_l, t_l, x_r, t_r),
        )

    if use_pso:
        if pso is None:
            raise ImportError("Missing dependency: pyswarm. Install with 'pip install pyswarm' to run PSO.")
        print("\nPSO refinement...")
        center = flatten_model_params(model)
        lower = center - pso_span
        upper = center + pso_span

        def pso_objective(flat):
            set_model_from_flat_vector(model, flat)
            total = (
                lambda_pde * pde_residual(model, x_c, t_c, nu_coef)
                + lambda_ic * ic_loss(model, x_ic, t_ic)
                + lambda_bc * bc_loss(model, x_l, t_l, x_r, t_r)
            )
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

    if return_stage_info:
        return {
            "loss_history": loss_history,
            "stage_states": stage_states,
            "stage_losses": stage_losses,
            "stage_times": stage_times,
        }
    return loss_history


def predict_on_grid(model, x_values, t_values):
    Xg, Tg = torch.meshgrid(x_values, t_values, indexing='ij')
    XT = torch.cat([Xg.reshape(-1, 1), Tg.reshape(-1, 1)], dim=1)
    with torch.no_grad():
        u_pred = model(XT).cpu().numpy().reshape(len(x_values), len(t_values))
    return u_pred


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
        method='BDF',
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

    if should_use_exact_plots(nu_coef) and os.path.exists('burgers_shock.mat'):
        data = loadmat('burgers_shock.mat')
        x_exact = data['x'].squeeze()
        t_exact = data['t'].squeeze()
        u_exact = np.real(data['usol'])
        if len(x_exact) == len(x_np) and len(t_exact) == len(t_np):
            return u_exact

    return reference_solution_fd(nu_coef, x_np, t_np)


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
        title=f"Burgers NAS-PINN Loss ({stage_name.upper()}, nu={args.nu:.3f})",
        use_interactive=interactive_plots,
    )

    plot_heatmap(model, stage_dir)

    x_test = torch.linspace(x_min, x_max, args.test_nx, device=device)
    t_test = torch.linspace(t_min, t_max, args.test_nt, device=device)
    u_pred = predict_on_grid(model, x_test, t_test)
    u_exact = get_reference_solution(args.nu, x_test, t_test)
    rel_l2 = np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-12)

    x_np = x_test.detach().cpu().numpy()
    t_np = t_test.detach().cpu().numpy()
    plot_burgers_exact_pred_error(
        x_np,
        t_np,
        u_exact,
        u_pred,
        os.path.join(stage_dir, "result_comparison.png"),
        pred_title=f"Predicted (NAS-PINN, {stage_name.upper()})",
        use_interactive=interactive_plots,
    )

    with open(os.path.join(stage_dir, "l2_error.txt"), "w", encoding="utf-8") as f:
        f.write(f"stage,{stage_name}\nnu,{args.nu:.6f}\nrel_l2,{rel_l2:.8e}\n")
    with open(os.path.join(stage_dir, "run_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"run_time_seconds,{run_time_seconds:.6f}\n")
    with open(os.path.join(stage_dir, "metrics.csv"), "w", encoding="utf-8") as f:
        f.write("method,stage,nu,seed,rel_l2,run_time_seconds\n")
        f.write(f"naspinn,{stage_name},{args.nu:.6f},{args.seed},{rel_l2:.8e},{run_time_seconds:.6f}\n")

    if should_use_exact_plots(args.nu):
        plot_time_slices(model, stage_dir)
        plot_time_slices_with_exact(model, stage_dir)
        plot_full_exact_vs_pred(model, stage_dir, "burgers_shock.mat")

    return float(rel_l2)


def copy_stage_to_root(stage_dir, root_dir):
    copy_map = [
        ("loss_curve.png", "loss_curve.png"),
        ("burgers_heatmap.png", "burgers_heatmap.png"),
        ("result_comparison.png", "result_comparison.png"),
        ("l2_error.txt", "l2_error.txt"),
        ("run_time.txt", "run_time.txt"),
        ("metrics.csv", "metrics.csv"),
    ]
    for src_name, dst_name in copy_map:
        src = os.path.join(stage_dir, src_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(root_dir, dst_name))

    exact_plot_src = os.path.join(stage_dir, "burgers_full_exact_vs_pred.png")
    if os.path.exists(exact_plot_src):
        shutil.copy2(exact_plot_src, os.path.join(root_dir, "burgers_full_exact_vs_pred.png"))


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


def run_paper_protocol(args):
    nu_values = [float(v.strip()) for v in args.paper_nus.split(',') if v.strip()]
    train_points = sample_points_paper(train_nx=args.train_nx, train_nt=args.train_nt)

    x_test = torch.linspace(x_min, x_max, args.test_nx, device=device)
    t_test = torch.linspace(t_min, t_max, args.test_nt, device=device)
    x_test_np = x_test.detach().cpu().numpy()
    t_test_np = t_test.detach().cpu().numpy()

    summary = []
    os.makedirs(args.save_dir, exist_ok=True)

    print("\nRunning paper protocol for Burgers equation")
    print(f"nu values: {nu_values}")
    print(f"Train grid: t={args.train_nt}, x={args.train_nx} | Test grid: t={args.test_nt}, x={args.test_nx}")
    print(f"Repeats: {args.repeats}")

    for nu_val in nu_values:
        exact_u = reference_solution_fd(nu_val, x_test_np, t_test_np)
        run_errors = []

        for run_id in range(1, args.repeats + 1):
            seed_val = args.seed + int(1000 * nu_val) + run_id
            torch.manual_seed(seed_val)
            np.random.seed(seed_val)

            model = NAS_PINN(layers=4, base_neurons=128).to(device)
            ckpt_path = os.path.join(args.save_dir, f"ckpt_nu_{nu_val:.3f}_run_{run_id}.pth")

            train_with_resume(
                model,
                total_epochs=args.epochs,
                checkpoint_path=ckpt_path,
                resume=False,
                skip_lbfgs=args.skip_lbfgs,
                nu_coef=nu_val,
                fixed_points=train_points,
            )

            pred_u = predict_on_grid(model, x_test, t_test)
            rel_l2 = np.linalg.norm(pred_u - exact_u) / (np.linalg.norm(exact_u) + 1e-12)
            run_errors.append(rel_l2)
            print(f"nu={nu_val:.3f} | run={run_id} | rel L2={rel_l2:.4e}")

            if run_id == args.repeats:
                plot_burgers_exact_pred_error(
                    x_test_np,
                    t_test_np,
                    exact_u,
                    pred_u,
                    os.path.join(args.save_dir, f"paper_protocol_nu_{nu_val:.3f}.png"),
                    pred_title=f"Pred (nu={nu_val:.3f})",
                    use_interactive=interactive_plots,
                )

        mean_l2 = float(np.mean(run_errors))
        std_l2 = float(np.std(run_errors))
        summary.append((nu_val, mean_l2, std_l2))

    print("\nPaper-style summary (5-run average):")
    print("nu      mean_rel_L2      std_rel_L2")
    for nu_val, mean_l2, std_l2 in summary:
        print(f"{nu_val:<6.3f}  {mean_l2:.4e}    {std_l2:.4e}")

    out_csv = os.path.join(args.save_dir, "paper_protocol_summary.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("nu,mean_rel_l2,std_rel_l2\n")
        for nu_val, mean_l2, std_l2 in summary:
            f.write(f"{nu_val:.6f},{mean_l2:.8e},{std_l2:.8e}\n")
    print(f"Saved summary: {out_csv}")

# ────────────────────────────────────────────────
# Isı haritası (contourf) – makaledeki gibi
# ────────────────────────────────────────────────
def plot_heatmap(model, save_dir):
    plot_burgers_heatmap(
        model,
        device,
        x_min,
        x_max,
        t_min,
        t_max,
        os.path.join(save_dir, "burgers_heatmap.png"),
        use_interactive=interactive_plots,
    )

def plot_time_slices(model, save_dir, t_values=[0.0, 0.25, 0.5, 0.75, 1.0], nx=400):
    plot_burgers_time_slices(
        model,
        device,
        x_min,
        x_max,
        os.path.join(save_dir, "burgers_time_slices.png"),
        t_values=t_values,
        nx=nx,
        use_interactive=interactive_plots,
    )

def plot_time_slices_with_exact(model, save_dir, mat_path='burgers_shock.mat', t_values=[0.0, 0.25, 0.5, 0.75, 1.0]):
    plot_burgers_time_slices_with_exact(
        model,
        device,
        os.path.join(save_dir, "burgers_exact_vs_pred_time_slices.png"),
        mat_path=mat_path,
        t_values=t_values,
        use_interactive=interactive_plots,
    )

def plot_full_exact_vs_pred(model, save_dir, mat_path='burgers_shock.mat'):
    plot_burgers_full_exact_vs_pred(
        model,
        device,
        os.path.join(save_dir, "burgers_full_exact_vs_pred.png"),
        mat_path=mat_path,
        pred_title="Predicted (NAS-PINN)",
        use_interactive=interactive_plots,
    )


def print_discovered_architecture(model):
    print("\nDiscovered architecture (most probable ops & mask):")
    for idx, layer in enumerate(model.layers):
        op_p = F.softmax(layer.alpha[:layer.n_ops], 0)
        m_p = torch.sigmoid(layer.alpha[layer.n_ops:])
        op_best = torch.argmax(op_p).item()
        m_best = torch.argmax(m_p).item()
        op_names = ["Linear/Identity", "Tanh", "Sin"]
        print(f"Layer {idx+1}: {op_names[op_best]} , neurons ≈ {layer.mask_levels[m_best]}")


def parse_args():
    parser = argparse.ArgumentParser(description="NAS-PINN Burgers training and plotting")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_last.pth", help="checkpoint path")
    parser.add_argument("--resume", action="store_true", help="resume training from checkpoint")
    parser.add_argument("--plot-only", action="store_true", help="skip training and only run plots from checkpoint")
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
    parser.add_argument("--nu", type=float, default=nu, help="single-run viscosity coefficient")
    parser.add_argument("--multi-nu", action="store_true", help="run three viscosity values and save comparison")
    parser.add_argument("--nu-list", type=str, default="0.01,0.04,0.07", help="comma-separated viscosities for --multi-nu")
    parser.add_argument("--epochs", type=int, default=15000, help="number of Adam epochs")
    parser.add_argument("--save-dir", type=str, default="results/burgers/naspinn", help="directory to save plot images")
    parser.add_argument("--paper-protocol", action="store_true", help="run paper-style multi-viscosity protocol")
    parser.add_argument("--paper-nus", type=str, default="0.1,0.07,0.04", help="comma-separated nu values")
    parser.add_argument("--repeats", type=int, default=5, help="number of repeated runs per viscosity")
    parser.add_argument("--train-nt", type=int, default=21, help="paper train t-grid points")
    parser.add_argument("--train-nx", type=int, default=250, help="paper train x-grid points")
    parser.add_argument("--test-nt", type=int, default=21, help="paper test t-grid points")
    parser.add_argument("--test-nx", type=int, default=500, help="paper test x-grid points")
    parser.add_argument("--seed", type=int, default=42, help="base random seed")
    return parser.parse_args()


def run_single(args):
    run_start = time.perf_counter()
    os.makedirs(args.save_dir, exist_ok=True)
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(args.save_dir, args.checkpoint)
    apply_stage_flags(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = NAS_PINN(layers=4, base_neurons=128).to(device)

    if args.plot_only:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found for plot-only mode: {args.checkpoint}")
        load_checkpoint(args.checkpoint, model)
        print(f"Loaded model for plot-only mode from: {args.checkpoint}")
        stage_dir = os.path.join(args.save_dir, "stage_loaded")
        rel_l2 = save_stage_outputs(
            model,
            args,
            stage_name="loaded",
            stage_dir=stage_dir,
            loss_history=[],
            stage_losses={},
            run_time_seconds=time.perf_counter() - run_start,
        )
        copy_stage_to_root(stage_dir, args.save_dir)
        run_time = time.perf_counter() - run_start
        print(f"Run time: {run_time:.2f} s")
        return float(rel_l2), float(run_time)
    else:
        train_info = train_with_resume(
            model,
            total_epochs=args.epochs,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
            skip_lbfgs=args.skip_lbfgs,
            use_pso=args.use_pso,
            pso_iters=args.pso_iters,
            pso_swarm=args.pso_swarm,
            pso_span=args.pso_span,
            nu_coef=args.nu,
            return_stage_info=True,
        )
        loss_history = train_info["loss_history"]
        stage_states = train_info["stage_states"]
        stage_losses = train_info["stage_losses"]
        stage_times = train_info["stage_times"]

    stage_order = ["adam", "lbfgs", "pso"]
    completed_stages = [name for name in stage_order if name in stage_states]
    stage_results = []
    for stage_name in completed_stages:
        model_stage = NAS_PINN(layers=4, base_neurons=128).to(device)
        model_stage.load_state_dict(stage_states[stage_name])
        print_discovered_architecture(model_stage)
        stage_dir = os.path.join(args.save_dir, f"stage_{stage_name}")
        rel_l2 = save_stage_outputs(
            model_stage,
            args,
            stage_name=stage_name,
            stage_dir=stage_dir,
            loss_history=loss_history,
            stage_losses=stage_losses,
            run_time_seconds=float(stage_times.get(stage_name, 0.0)),
        )
        stage_results.append((stage_name, rel_l2, float(stage_times.get(stage_name, 0.0))))

    if not stage_results:
        raise RuntimeError("No stage output was produced.")

    final_stage_name, final_rel_l2, _ = stage_results[-1]
    final_stage_dir = os.path.join(args.save_dir, f"stage_{final_stage_name}")
    copy_stage_to_root(final_stage_dir, args.save_dir)

    with open(os.path.join(args.save_dir, "stage_summary.csv"), "w", encoding="utf-8") as f:
        f.write("stage,nu,seed,rel_l2,run_time_seconds\n")
        for stage_name, rel_l2, stage_time in stage_results:
            f.write(f"{stage_name},{args.nu:.6f},{args.seed},{rel_l2:.8e},{stage_time:.6f}\n")

    run_time = time.perf_counter() - run_start
    with open(os.path.join(args.save_dir, "run_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"run_time_seconds,{run_time:.6f}\n")
    with open(os.path.join(args.save_dir, "metrics.csv"), "w", encoding="utf-8") as f:
        f.write("method,stage,nu,seed,rel_l2,run_time_seconds\n")
        f.write(f"naspinn,{final_stage_name},{args.nu:.6f},{args.seed},{final_rel_l2:.8e},{run_time:.6f}\n")
    print(f"Run time: {run_time:.2f} s")
    return float(final_rel_l2), float(run_time)


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
        args_local.checkpoint = "checkpoint_last.pth"
        rel_l2, run_time = run_single(args_local)
        summary.append((nu_val, rel_l2, run_time))

    csv_path = os.path.join(base_dir, "viscosity_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("nu,rel_l2,run_time_seconds\n")
        for nu_val, rel_l2, run_time in summary:
            f.write(f"{nu_val:.6f},{rel_l2:.8e},{run_time:.6f}\n")
    print(f"Saved summary: {csv_path}")

    nus = [row[0] for row in summary]
    errs = [row[1] for row in summary]
    plt.figure(figsize=(7, 4))
    plt.plot(nus, errs, marker="o", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Viscosity (nu)")
    plt.ylabel("Relative L2 Error")
    plt.title("Burgers NAS-PINN: Viscosity Comparison")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    finalize_plot(os.path.join(base_dir, "viscosity_comparison.png"))


def main():
    args = parse_args()

    if args.paper_protocol:
        os.makedirs(args.save_dir, exist_ok=True)
        run_paper_protocol(args)
        return

    if args.multi_nu:
        run_multi_nu(args)
        return

    run_single(args)


if __name__ == "__main__":
    main()

# Zaman kesitleri (opsiyonel, makalede karşılaştırma için)
# t = 0.0, 0.25, 0.5, 0.75, 1.0 gibi noktalar için çizdirilebilir
