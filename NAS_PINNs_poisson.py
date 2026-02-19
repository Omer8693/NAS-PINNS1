import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

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
# Hiperparametreler (makaleye göre Poisson ayarları)
# ────────────────────────────────────────────────
N_col = 10000               # interior collocation points
N_bc  = 400                 # boundary points (her kenar için yaklaşık N_bc/4)
domain_x = [0.0, 1.0]
domain_y = [0.0, 1.0]

lambda_pde = 1.0
lambda_bc  = 100.0          # boundary genellikle daha yüksek ağırlık alır

pi = torch.tensor(np.pi, dtype=torch.float32, device=device)


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


# ────────────────────────────────────────────────
# Mixed Operation – NAS-PINN Eq. (8) & (9) tarzı
# ────────────────────────────────────────────────
class MixedOp(nn.Module):
    def __init__(self, in_c, out_c, mask_levels=[32, 64, 96, 128, 192, 256]):
        super().__init__()
        self.mask_levels = mask_levels
        self.n_masks = len(mask_levels)

        self.ops = nn.ModuleList([
            nn.Identity() if in_c == out_c else nn.Linear(in_c, out_c),   # skip/linear
            nn.Sequential(nn.Linear(in_c, out_c), nn.Tanh()),
            nn.Sequential(nn.Linear(in_c, out_c), SinActivation()),
        ])
        self.n_ops = len(self.ops)

        total = self.n_ops + self.n_masks
        self.alpha = nn.Parameter(torch.randn(total) * 0.1)

    def relaxed_op(self, x):
        weights = F.softmax(self.alpha[:self.n_ops], dim=0)
        out = sum(w * op(x) for w, op in zip(weights, self.ops))
        return out

    def forward(self, x):
        mixed = self.relaxed_op(x)

        mask_weights = torch.sigmoid(self.alpha[self.n_ops:])   # NAS-PINN'de stabilite için sigmoid yaygın
        final = 0.0
        dim = mixed.shape[-1]
        for j, keep in enumerate(self.mask_levels):
            k = min(keep, dim)
            mask = torch.zeros(dim, device=device)
            mask[:k] = 1.0
            masked = mixed * mask.unsqueeze(0)
            final += mask_weights[j] * masked

        return final


# ────────────────────────────────────────────────
# NAS-PINN Model (searchable, Poisson için shallow önerilir)
# ────────────────────────────────────────────────
class NAS_PINN(nn.Module):
    def __init__(self, layers=3, base_neurons=192, mask_levels=[64, 96, 128, 192, 256]):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [2] + [base_neurons] * (layers - 1) + [1]   # input (x,y), output φ

        for i in range(layers):
            self.layers.append(MixedOp(dims[i], dims[i+1], mask_levels))

    def forward(self, xy):
        x = xy
        for layer in self.layers:
            x = layer(x)
        return x


# ────────────────────────────────────────────────
# Veri noktaları üretimi
# ────────────────────────────────────────────────
def sample_points():
    # Interior collocation points (Ω içindeki random noktalar)
    x_col = torch.rand(N_col, 1, device=device) * (domain_x[1] - domain_x[0]) + domain_x[0]
    y_col = torch.rand(N_col, 1, device=device) * (domain_y[1] - domain_y[0]) + domain_y[0]

    # Boundary points (4 kenar)
    n_per_side = N_bc // 4
    # bottom: y=0
    x_bot = torch.rand(n_per_side, 1, device=device) * (domain_x[1] - domain_x[0]) + domain_x[0]
    y_bot = torch.zeros_like(x_bot)
    # top: y=1
    x_top = torch.rand(n_per_side, 1, device=device) * (domain_x[1] - domain_x[0]) + domain_x[0]
    y_top = torch.ones_like(x_top)
    # left: x=0
    y_left = torch.rand(n_per_side, 1, device=device) * (domain_y[1] - domain_y[0]) + domain_y[0]
    x_left = torch.zeros_like(y_left)
    # right: x=1
    y_right = torch.rand(n_per_side, 1, device=device) * (domain_y[1] - domain_y[0]) + domain_y[0]
    x_right = torch.ones_like(y_right)

    x_bc = torch.cat([x_bot, x_top, x_left, x_right], dim=0)
    y_bc = torch.cat([y_bot, y_top, y_left, y_right], dim=0)

    return (x_col, y_col), (x_bc, y_bc)


def sample_points_protocol(train_nx=100, train_ny=100, boundary_n=200):
    x_vals = torch.linspace(domain_x[0], domain_x[1], train_nx, device=device)
    y_vals = torch.linspace(domain_y[0], domain_y[1], train_ny, device=device)

    X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
    x_col = X.reshape(-1, 1)
    y_col = Y.reshape(-1, 1)

    x_side = torch.linspace(domain_x[0], domain_x[1], boundary_n, device=device).unsqueeze(1)
    y_side = torch.linspace(domain_y[0], domain_y[1], boundary_n, device=device).unsqueeze(1)

    x_bot = x_side
    y_bot = torch.full_like(x_bot, domain_y[0])
    x_top = x_side
    y_top = torch.full_like(x_top, domain_y[1])

    y_left = y_side
    x_left = torch.full_like(y_left, domain_x[0])
    y_right = y_side
    x_right = torch.full_like(y_right, domain_x[1])

    x_bc = torch.cat([x_bot, x_top, x_left, x_right], dim=0)
    y_bc = torch.cat([y_bot, y_top, y_left, y_right], dim=0)

    return (x_col, y_col), (x_bc, y_bc)


# ────────────────────────────────────────────────
# Loss fonksiyonları
# ────────────────────────────────────────────────
def pde_loss(model, x, y):
    xy = torch.cat([x, y], dim=1).requires_grad_(True)
    phi = model(xy)

    phi_x = torch.autograd.grad(phi.sum(), xy, create_graph=True)[0][:, 0:1]
    phi_y = torch.autograd.grad(phi.sum(), xy, create_graph=True)[0][:, 1:2]

    phi_xx = torch.autograd.grad(phi_x.sum(), xy, create_graph=True)[0][:, 0:1]
    phi_yy = torch.autograd.grad(phi_y.sum(), xy, create_graph=True)[0][:, 1:2]

    residual = phi_xx + phi_yy + 2 * pi**2 * torch.cos(pi * x) * torch.cos(pi * y)
    return torch.mean(residual ** 2)


def bc_loss(model, x_bc, y_bc):
    xy_bc = torch.cat([x_bc, y_bc], dim=1)
    phi_pred = model(xy_bc)
    phi_exact = torch.cos(pi * x_bc) * torch.cos(pi * y_bc)
    return torch.mean((phi_pred - phi_exact) ** 2)


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
        points = (
            p["x_col"].to(device),
            p["y_col"].to(device),
            p["x_bc"].to(device),
            p["y_bc"].to(device),
        )
    else:
        (x_col, y_col), (x_bc, y_bc) = sample_points()
        points = (x_col, y_col, x_bc, y_bc)

    return ckpt.get("epoch", 0), points


def train_with_resume(model, total_epochs=12000, inner_lr=1e-3, outer_lr=3e-4,
                      outer_every=5, checkpoint_path="poisson_checkpoint_last.pth",
                      checkpoint_every=1000, resume=False, skip_lbfgs=False,
                      fixed_points=None):
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
        if resume:
            print(f"Resume requested but checkpoint not found: {checkpoint_path}")
            print("Starting from scratch...")

    x_col, y_col, x_bc, y_bc = points

    print("Starting Adam + bi-level optimization...")
    for epoch in range(start_epoch, total_epochs):
        opt_inner.zero_grad()

        l_pde = pde_loss(model, x_col, y_col)
        l_bc = bc_loss(model, x_bc, y_bc)
        loss_inner = lambda_pde * l_pde + lambda_bc * l_bc

        loss_inner.backward()
        opt_inner.step()

        if epoch % outer_every == 0:
            opt_outer.zero_grad()
            l_pde_o = pde_loss(model, x_col, y_col)
            l_bc_o = bc_loss(model, x_bc, y_bc)
            loss_outer = lambda_pde * l_pde_o + lambda_bc * l_bc_o
            loss_outer.backward()
            opt_outer.step()

        if epoch % 2000 == 0:
            print(f"[{epoch:5d}] PDE residual: {l_pde:.4e}   BC: {l_bc:.4e}")

        if ((epoch + 1) % checkpoint_every == 0) or (epoch + 1 == total_epochs):
            save_checkpoint(
                checkpoint_path,
                model,
                opt_inner,
                opt_outer,
                epoch + 1,
                (x_col, y_col, x_bc, y_bc),
            )

    if not skip_lbfgs:
        print("\nL-BFGS refinement...")
        lbfgs = optim.LBFGS(model.parameters(), lr=0.8, max_iter=2000, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            lp = pde_loss(model, x_col, y_col)
            lb = bc_loss(model, x_bc, y_bc)
            total = lambda_pde * lp + lambda_bc * lb
            total.backward()
            return total

        lbfgs.step(closure)

        save_checkpoint(
            checkpoint_path,
            model,
            opt_inner,
            opt_outer,
            total_epochs,
            (x_col, y_col, x_bc, y_bc),
        )


def print_discovered_architecture(model):
    print("\nDiscovered architecture (most probable ops & mask):")
    for i, layer in enumerate(model.layers):
        op_p = F.softmax(layer.alpha[:layer.n_ops], dim=0)
        mask_p = torch.sigmoid(layer.alpha[layer.n_ops:])
        op_idx = torch.argmax(op_p).item()
        mask_idx = torch.argmax(mask_p).item()
        op_names = ["Linear/Identity", "Tanh", "Sin"]
        print(f"Layer {i+1}: {op_names[op_idx]} , neurons ≈ {layer.mask_levels[mask_idx]}")


def architecture_signature(model):
    parts = []
    op_names = ["Linear/Identity", "Tanh", "Sin"]
    for layer in model.layers:
        op_p = F.softmax(layer.alpha[:layer.n_ops], dim=0)
        mask_p = torch.sigmoid(layer.alpha[layer.n_ops:])
        op_idx = torch.argmax(op_p).item()
        mask_idx = torch.argmax(mask_p).item()
        parts.append(f"{op_names[op_idx]}-{layer.mask_levels[mask_idx]}")
    return " | ".join(parts)


def predict_on_grid(model, test_nx=150, test_ny=150):
    x_grid = torch.linspace(domain_x[0], domain_x[1], test_nx, device=device)
    y_grid = torch.linspace(domain_y[0], domain_y[1], test_ny, device=device)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    XY = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        phi_pred = model(XY).cpu().numpy().reshape(test_nx, test_ny)

    x_np = X.cpu().numpy()
    y_np = Y.cpu().numpy()
    phi_exact = np.cos(np.pi * x_np) * np.cos(np.pi * y_np)
    rel_l2 = np.linalg.norm(phi_pred - phi_exact) / (np.linalg.norm(phi_exact) + 1e-12)
    return rel_l2, x_np, y_np, phi_pred, phi_exact

# ────────────────────────────────────────────────
# Görselleştirme – makaledeki gibi heatmap + error
# ────────────────────────────────────────────────
def plot_results(model, save_dir):
    nx, ny = 150, 150
    x_grid = torch.linspace(0, 1, nx, device=device).unsqueeze(1)
    y_grid = torch.linspace(0, 1, ny, device=device).unsqueeze(1)
    X, Y = torch.meshgrid(x_grid.squeeze(), y_grid.squeeze(), indexing='ij')
    XY = torch.cat([X.flatten().unsqueeze(1), Y.flatten().unsqueeze(1)], dim=1)

    with torch.no_grad():
        phi_pred = model(XY).cpu().reshape(nx, ny).numpy()

    phi_exact_np = np.cos(np.pi * X.cpu().numpy()) * np.cos(np.pi * Y.cpu().numpy())

    abs_err = np.abs(phi_pred - phi_exact_np)
    rel_l2 = np.linalg.norm(phi_pred - phi_exact_np) / np.linalg.norm(phi_exact_np)
    print(f"\nRelative L2 error (full grid): {rel_l2:.4e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # Predicted
    cs0 = axes[0].contourf(X.cpu().numpy(), Y.cpu().numpy(), phi_pred,
                           levels=50, cmap='viridis')
    axes[0].set_title('Predicted φ(x,y)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    fig.colorbar(cs0, ax=axes[0])

    # Exact
    cs1 = axes[1].contourf(X.cpu().numpy(), Y.cpu().numpy(), phi_exact_np,
                           levels=50, cmap='viridis')
    axes[1].set_title('Exact φ(x,y) = cos(πx)cos(πy)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    fig.colorbar(cs1, ax=axes[1])

    # Absolute error
    cs2 = axes[2].contourf(X.cpu().numpy(), Y.cpu().numpy(), abs_err,
                           levels=50, cmap='magma')
    axes[2].set_title('|Predicted - Exact|')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    fig.colorbar(cs2, ax=axes[2])

    plt.suptitle('NAS-PINN Poisson Equation Results')
    finalize_plot(os.path.join(save_dir, "poisson_results.png"))


def run_protocol(args):
    fixed_points = sample_points_protocol(
        train_nx=args.train_nx,
        train_ny=args.train_ny,
        boundary_n=args.boundary_n,
    )
    summary = []
    last_arch = ""

    print("\nRunning protocol mode for Poisson equation")
    print(f"Train grid: x={args.train_nx}, y={args.train_ny}, boundary={args.boundary_n}")
    print(f"Test grid: x={args.test_nx}, y={args.test_ny}")
    print(f"Repeats: {args.repeats}")

    for run_id in range(1, args.repeats + 1):
        seed_val = args.seed + run_id
        torch.manual_seed(seed_val)
        np.random.seed(seed_val)

        model = NAS_PINN(layers=3, base_neurons=192).to(device)
        ckpt_path = os.path.join(args.save_dir, f"poisson_protocol_run_{run_id}.pth")

        train_with_resume(
            model,
            total_epochs=args.epochs,
            checkpoint_path=ckpt_path,
            resume=False,
            skip_lbfgs=args.skip_lbfgs,
            fixed_points=fixed_points,
        )

        rel_l2, x_np, y_np, phi_pred, phi_exact = predict_on_grid(
            model,
            test_nx=args.test_nx,
            test_ny=args.test_ny,
        )
        summary.append(rel_l2)
        last_arch = architecture_signature(model)
        print(f"run={run_id} | rel L2={rel_l2:.4e}")

        if run_id == args.repeats:
            abs_err = np.abs(phi_pred - phi_exact)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
            cs0 = axes[0].contourf(x_np, y_np, phi_exact, levels=50, cmap='viridis')
            axes[0].set_title('Exact φ(x,y)')
            fig.colorbar(cs0, ax=axes[0])
            cs1 = axes[1].contourf(x_np, y_np, phi_pred, levels=50, cmap='viridis')
            axes[1].set_title('Predicted φ(x,y)')
            fig.colorbar(cs1, ax=axes[1])
            cs2 = axes[2].contourf(x_np, y_np, abs_err, levels=50, cmap='magma')
            axes[2].set_title('|Predicted - Exact|')
            fig.colorbar(cs2, ax=axes[2])
            plt.suptitle('NAS-PINN Poisson Protocol Results')
            finalize_plot(os.path.join(args.save_dir, "poisson_protocol_last_run.png"))

    mean_l2 = float(np.mean(summary))
    std_l2 = float(np.std(summary))

    print("\nPaper-style summary (5-run average):")
    print("Name      Architecture                         mean_rel_L2      std_rel_L2")
    print(f"NAS-PINN  {last_arch:<35}  {mean_l2:.4e}    {std_l2:.4e}")

    out_csv = os.path.join(args.save_dir, "paper_protocol_summary.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("name,architecture,run,rel_l2\n")
        for idx, err in enumerate(summary, start=1):
            f.write(f"NAS-PINN,\"{last_arch}\",{idx},{err:.8e}\n")
        f.write(f"NAS-PINN,\"{last_arch}\",mean,{mean_l2:.8e}\n")
        f.write(f"NAS-PINN,\"{last_arch}\",std,{std_l2:.8e}\n")
    print(f"Saved summary: {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="NAS-PINN Poisson training and plotting")
    parser.add_argument("--checkpoint", type=str, default="poisson_checkpoint_last.pth", help="checkpoint path")
    parser.add_argument("--resume", action="store_true", help="resume training from checkpoint")
    parser.add_argument("--plot-only", action="store_true", help="skip training and only run plots from checkpoint")
    parser.add_argument("--skip-lbfgs", action="store_true", help="skip L-BFGS refinement")
    parser.add_argument("--epochs", type=int, default=12000, help="number of Adam epochs")
    parser.add_argument("--seed", type=int, default=42, help="base random seed")
    parser.add_argument("--save-dir", type=str, default="results_plots", help="directory to save plot images")
    parser.add_argument("--protocol", action="store_true", help="run repeated fixed-grid protocol")
    parser.add_argument("--paper-protocol", action="store_true", help="alias of --protocol (paper-style mode)")
    parser.add_argument("--repeats", type=int, default=5, help="number of repeated runs")
    parser.add_argument("--train-nx", type=int, default=100, help="protocol train grid points in x")
    parser.add_argument("--train-ny", type=int, default=100, help="protocol train grid points in y")
    parser.add_argument("--boundary-n", type=int, default=200, help="protocol boundary points per side")
    parser.add_argument("--test-nx", type=int, default=150, help="protocol test grid points in x")
    parser.add_argument("--test-ny", type=int, default=150, help="protocol test grid points in y")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.protocol or args.paper_protocol:
        run_protocol(args)
        return

    model = NAS_PINN(layers=3, base_neurons=192).to(device)

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
        )

    print_discovered_architecture(model)
    plot_results(model, args.save_dir)


if __name__ == "__main__":
    main()