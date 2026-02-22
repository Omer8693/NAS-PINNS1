import os
import numpy as np
import torch
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
interactive_plots = os.environ.get("DISPLAY") is not None

x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
N_col = 10000
N_bc = 400
lambda_pde = 1.0
lambda_bc = 100.0


def finalize_plot(plt, save_path):
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {save_path}")
    if interactive_plots:
        plt.show()
    else:
        plt.close()


def plot_loss_curve(loss_history, save_path, title="Training Loss vs Epoch"):
    if not loss_history:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, linewidth=2)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    finalize_plot(plt, save_path)


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
