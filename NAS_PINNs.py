import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.io import loadmat  # referans çözüm için (opsiyonel)
from scipy.integrate import solve_ivp

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


def train_with_resume(model, total_epochs=15000, inner_lr=1e-3, outer_lr=3e-4,
                      outer_every=5, checkpoint_path="checkpoint_last.pth",
                      checkpoint_every=1000, resume=False, skip_lbfgs=False,
                      nu_coef=nu, fixed_points=None):
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

    print("Starting Adam + bi-level optimization...")
    for epoch in range(start_epoch, total_epochs):
        opt_inner.zero_grad()

        l_pde = pde_residual(model, x_c, t_c, nu_coef)
        l_ic = ic_loss(model, x_ic, t_ic)
        l_bc = bc_loss(model, x_l, t_l, x_r, t_r)
        loss_inner = lambda_pde * l_pde + lambda_ic * l_ic + lambda_bc * l_bc

        loss_inner.backward()
        opt_inner.step()

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

        save_checkpoint(
            checkpoint_path,
            model,
            opt_inner,
            opt_outer,
            total_epochs,
            (x_c, t_c, x_ic, t_ic, x_l, t_l, x_r, t_r),
        )


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
                fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
                Xp, Tp = np.meshgrid(x_test_np, t_test_np, indexing='ij')
                cs0 = axes[0].contourf(Xp, Tp, exact_u, levels=60, cmap='viridis')
                axes[0].set_title(f'Exact (nu={nu_val:.3f})')
                fig.colorbar(cs0, ax=axes[0])
                cs1 = axes[1].contourf(Xp, Tp, pred_u, levels=60, cmap='viridis')
                axes[1].set_title(f'Pred (nu={nu_val:.3f})')
                fig.colorbar(cs1, ax=axes[1])
                cs2 = axes[2].contourf(Xp, Tp, np.abs(pred_u - exact_u), levels=60, cmap='magma')
                axes[2].set_title('|Pred-Exact|')
                fig.colorbar(cs2, ax=axes[2])
                finalize_plot(os.path.join(args.save_dir, f"paper_protocol_nu_{nu_val:.3f}.png"))

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
    nx, nt = 200, 100
    x_grid = torch.linspace(x_min, x_max, nx, device=device).unsqueeze(1)
    t_grid = torch.linspace(t_min, t_max, nt, device=device).unsqueeze(1)
    X, T = torch.meshgrid(x_grid.squeeze(), t_grid.squeeze(), indexing='ij')
    XT = torch.cat([X.flatten().unsqueeze(1), T.flatten().unsqueeze(1)], dim=1)

    with torch.no_grad():
        U = model(XT).cpu().reshape(nx, nt)

    plt.figure(figsize=(10, 6))
    cs = plt.contourf(X.cpu().numpy(), T.cpu().numpy(), U.numpy(),
                      levels=60, cmap='viridis')
    plt.colorbar(cs, label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Burgers solution learned with NAS-PINN (heatmap)')
    plt.tight_layout()
    finalize_plot(os.path.join(save_dir, "burgers_heatmap.png"))

def plot_time_slices(model, save_dir, t_values=[0.0, 0.25, 0.5, 0.75, 1.0], nx=400):
    x_line = torch.linspace(x_min, x_max, nx, device=device).unsqueeze(1)

    plt.figure(figsize=(10, 6))
    with torch.no_grad():
        for t_val in t_values:
            t_line = torch.full_like(x_line, float(t_val))
            xt = torch.cat([x_line, t_line], dim=1)
            u_line = model(xt).cpu().numpy().squeeze()
            plt.plot(x_line.cpu().numpy().squeeze(), u_line, label=f't={t_val:.2f}')

    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('Time slices of the learned Burgers solution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    finalize_plot(os.path.join(save_dir, "burgers_time_slices.png"))

def plot_time_slices_with_exact(model, save_dir, mat_path='burgers_shock.mat', t_values=[0.0, 0.25, 0.5, 0.75, 1.0]):
    try:
        data = loadmat(mat_path)
    except Exception as exc:
        print(f"Could not load exact solution file '{mat_path}': {exc}")
        return

    x_exact = data['x'].squeeze()
    t_exact = data['t'].squeeze()
    u_exact = np.real(data['usol'])

    x_tensor = torch.tensor(x_exact, dtype=torch.float32, device=device).unsqueeze(1)

    plt.figure(figsize=(10, 6))
    with torch.no_grad():
        for t_val in t_values:
            t_idx = int(np.argmin(np.abs(t_exact - t_val)))
            t_match = float(t_exact[t_idx])

            t_tensor = torch.full_like(x_tensor, t_match)
            xt = torch.cat([x_tensor, t_tensor], dim=1)
            u_pred = model(xt).cpu().numpy().squeeze()
            u_ref = u_exact[:, t_idx].squeeze()

            plt.plot(x_exact, u_ref, '--', linewidth=2, label=f'Exact t={t_match:.2f}')
            plt.plot(x_exact, u_pred, '-', linewidth=2, label=f'Pred t={t_match:.2f}')

    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('Exact vs Predicted time slices (Burgers)')
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    finalize_plot(os.path.join(save_dir, "burgers_exact_vs_pred_time_slices.png"))

def plot_full_exact_vs_pred(model, save_dir, mat_path='burgers_shock.mat'):
    try:
        data = loadmat(mat_path)
    except Exception as exc:
        print(f"Could not load exact solution file '{mat_path}': {exc}")
        return

    x_exact = data['x'].squeeze()
    t_exact = data['t'].squeeze()
    u_exact = np.real(data['usol'])

    Xg, Tg = np.meshgrid(x_exact, t_exact, indexing='ij')
    xt_np = np.stack([Xg.ravel(), Tg.ravel()], axis=1)
    xt = torch.tensor(xt_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        u_pred = model(xt).cpu().numpy().reshape(len(x_exact), len(t_exact))

    rel_l2 = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    print(f"Relative L2 error (full grid): {rel_l2:.4e}")

    abs_err = np.abs(u_pred - u_exact)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    cs0 = axes[0].contourf(Xg, Tg, u_exact, levels=60, cmap='viridis')
    axes[0].set_title('Exact (burgers_shock.mat)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    plt.colorbar(cs0, ax=axes[0])

    cs1 = axes[1].contourf(Xg, Tg, u_pred, levels=60, cmap='viridis')
    axes[1].set_title('Predicted (NAS-PINN)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    plt.colorbar(cs1, ax=axes[1])

    cs2 = axes[2].contourf(Xg, Tg, abs_err, levels=60, cmap='magma')
    axes[2].set_title('|Pred - Exact|')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('t')
    plt.colorbar(cs2, ax=axes[2])

    finalize_plot(os.path.join(save_dir, "burgers_full_exact_vs_pred.png"))


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
    parser.add_argument("--skip-lbfgs", action="store_true", help="skip L-BFGS refinement")
    parser.add_argument("--nu", type=float, default=nu, help="single-run viscosity coefficient")
    parser.add_argument("--epochs", type=int, default=15000, help="number of Adam epochs")
    parser.add_argument("--save-dir", type=str, default="results_plots1", help="directory to save plot images")
    parser.add_argument("--paper-protocol", action="store_true", help="run paper-style multi-viscosity protocol")
    parser.add_argument("--paper-nus", type=str, default="0.1,0.07,0.04", help="comma-separated nu values")
    parser.add_argument("--repeats", type=int, default=5, help="number of repeated runs per viscosity")
    parser.add_argument("--train-nt", type=int, default=21, help="paper train t-grid points")
    parser.add_argument("--train-nx", type=int, default=250, help="paper train x-grid points")
    parser.add_argument("--test-nt", type=int, default=21, help="paper test t-grid points")
    parser.add_argument("--test-nx", type=int, default=500, help="paper test x-grid points")
    parser.add_argument("--seed", type=int, default=42, help="base random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.paper_protocol:
        run_paper_protocol(args)
        return

    model = NAS_PINN(layers=4, base_neurons=128).to(device)

    if args.plot_only:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found for plot-only mode: {args.checkpoint}")
        load_checkpoint(args.checkpoint, model)
        print(f"Loaded model for plot-only mode from: {args.checkpoint}")
    else:
        train_with_resume(
            model,
            total_epochs=args.epochs,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
            skip_lbfgs=args.skip_lbfgs,
            nu_coef=args.nu,
        )

    print_discovered_architecture(model)
    plot_heatmap(model, args.save_dir)
    plot_time_slices(model, args.save_dir)
    if abs(args.nu - (0.01 / np.pi)) < 1e-12:
        plot_time_slices_with_exact(model, args.save_dir)
        plot_full_exact_vs_pred(model, args.save_dir, 'burgers_shock.mat')
    else:
        print("Skipping burgers_shock.mat exact comparison because --nu differs from dataset viscosity 0.01/pi")


if __name__ == "__main__":
    main()

# Zaman kesitleri (opsiyonel, makalede karşılaştırma için)
# t = 0.0, 0.25, 0.5, 0.75, 1.0 gibi noktalar için çizdirilebilir