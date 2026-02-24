"""
NAS-PINN baseline for Poisson equation on flower-shaped domain
Makale: Wang & Zhong, 2024 (arXiv:2305.10127v2)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Domain definition: flower-shaped r(θ) = 1 + 0.3*sin(5θ)
PETAL_A = 1.0
PETAL_B = 0.3
PETAL_N = 5

# Analytic solution
def analytic_solution(x, y):
    return np.cos(np.pi * x) * np.cos(np.pi * y)

def rhs_pde(x, y):
    return -2 * np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y)

# Sampling functions
# Rejection sampling for collocation points inside flower domain
def sample_collocation_points(n):
    x = []
    y = []
    while len(x) < n:
        x_try = np.random.uniform(-1.3, 1.3, n)
        y_try = np.random.uniform(-1.3, 1.3, n)
        r = np.sqrt(x_try**2 + y_try**2)
        theta = np.arctan2(y_try, x_try)
        r_boundary = PETAL_A + PETAL_B * np.sin(PETAL_N * theta)
        mask = r <= r_boundary
        x.extend(x_try[mask])
        y.extend(y_try[mask])
    return np.array(x[:n]), np.array(y[:n])

# Boundary points: sample along flower edge (polar)
def sample_boundary_points(n):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = PETAL_A + PETAL_B * np.sin(PETAL_N * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# NAS-PINN model (tanh, skip, mask_levels)
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
        weights = torch.softmax(self.alpha[: self.n_ops], dim=0)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

    def forward(self, x):
        mixed = self.relaxed_op(x)
        mask_weights = torch.sigmoid(self.alpha[self.n_ops :])
        final = 0.0
        dim = mixed.shape[-1]
        for j, keep in enumerate(self.mask_levels):
            k = min(keep, dim)
            mask = torch.zeros(dim, device=x.device)
            mask[:k] = 1.0
            final += mask_weights[j] * (mixed * mask.unsqueeze(0))
        return final

class NAS_PINN(nn.Module):
    def __init__(self, layers=6, base_neurons=110, mask_levels=(30, 50, 70, 90, 110)):
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

# Loss functions

def pde_loss(model, x, y):
    x_t = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    xy = torch.cat([x_t, y_t], dim=1).requires_grad_(True)
    u = model(xy)
    grads = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0]
    u_y = grads[:, 1]
    u_xx = torch.autograd.grad(u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(u_y, xy, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1]
    laplacian = u_xx + u_yy
    rhs = torch.tensor(rhs_pde(x, y), dtype=torch.float32)
    residual = laplacian - rhs
    return torch.mean(residual ** 2)

def bc_loss(model, x, y):
    x_t = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    xy = torch.cat([x_t, y_t], dim=1)
    u = model(xy)
    bc = torch.tensor(analytic_solution(x, y), dtype=torch.float32).view(-1, 1)
    return torch.mean((u - bc) ** 2)

def outer_loss(model, x, y):
    x_t = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    xy = torch.cat([x_t, y_t], dim=1)
    u = model(xy)
    true = torch.tensor(analytic_solution(x, y), dtype=torch.float32).view(-1, 1)
    return torch.mean((u - true) ** 2)

# Relative L2 error

def relative_l2_error(model, x, y):
    x_t = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    xy = torch.cat([x_t, y_t], dim=1)
    u_pred = model(xy).detach().cpu().numpy().flatten()
    u_true = analytic_solution(x, y)
    return np.sqrt(np.mean((u_pred - u_true) ** 2)) / np.sqrt(np.mean(u_true ** 2))

# Training loop (NAS-PINN bi-level)
def train_naspinn(layers=6, base_neurons=110, mask_levels=(30,50,70,90,110),
                  collocation_n=6000, boundary_n=1200, outer_n=6000,
                  adam_epochs=15000, inner_lr=1e-3, outer_lr=3e-4,
                  outer_every=5, lbfgs_iter=2000, lbfgs_lr=0.8):
    model = NAS_PINN(layers=layers, base_neurons=base_neurons, mask_levels=mask_levels)
    opt_inner = optim.Adam(model.parameters(), lr=inner_lr)
    arch_params = [layer.alpha for layer in model.layers]
    opt_outer = optim.Adam(arch_params, lr=outer_lr)
    x_col, y_col = sample_collocation_points(collocation_n)
    x_bc, y_bc = sample_boundary_points(boundary_n)
    x_outer, y_outer = sample_collocation_points(outer_n)
    for epoch in range(adam_epochs):
        opt_inner.zero_grad()
        l_pde = pde_loss(model, x_col, y_col)
        l_bc = bc_loss(model, x_bc, y_bc)
        loss_inner = l_pde + l_bc
        loss_inner.backward()
        opt_inner.step()
        if epoch % outer_every == 0:
            opt_outer.zero_grad()
            l_outer = outer_loss(model, x_outer, y_outer)
            l_outer.backward()
            opt_outer.step()
        # Terminalde epoch ilerlemesini göster
        if epoch % 100 == 0 or epoch == adam_epochs - 1:
            print(f"Epoch {epoch}/{adam_epochs} | Inner Loss: {loss_inner.item():.6e}")
    # L-BFGS refinement
    lbfgs = optim.LBFGS(model.parameters(), lr=lbfgs_lr, max_iter=lbfgs_iter, line_search_fn="strong_wolfe")
    def closure():
        lbfgs.zero_grad()
        l_pde = pde_loss(model, x_col, y_col)
        l_bc = bc_loss(model, x_bc, y_bc)
        return l_pde + l_bc
    lbfgs.step(closure)
    return model

# Mimari raporlama
def architecture_signature(model):
    parts = []
    op_names = ["Identity", "Tanh"]
    for layer in model.layers:
        op_p = torch.softmax(layer.alpha[: layer.n_ops], dim=0)
        mask_p = torch.sigmoid(layer.alpha[layer.n_ops :])
        op_idx = torch.argmax(op_p).item()
        mask_idx = torch.argmax(mask_p).item()
        parts.append(f"{op_names[op_idx]}-{layer.mask_levels[mask_idx]}")
    return " | ".join(parts)

if __name__ == "__main__":
    # 5 repeat for reproducibility
    import matplotlib.pyplot as plt
    save_dir = "results/poisson/flower/"
    os.makedirs(save_dir, exist_ok=True)
    l2_errors = []
    for repeat in range(5):
        model = train_naspinn()
        x_test, y_test = sample_collocation_points(1000000)
        rel_l2 = relative_l2_error(model, x_test, y_test)
        print(f"Repeat {repeat+1}: Relative L2 error = {rel_l2:.6e}")
        print("Architecture:", architecture_signature(model))
        l2_errors.append(rel_l2)
        # Solution and error heatmap
        grid_n = 200
        theta = np.linspace(0, 2*np.pi, grid_n)
        r = np.linspace(0, 1.3, grid_n)
        T, R = np.meshgrid(theta, r)
        X = R * np.cos(T)
        Y = R * np.sin(T)
        petal_mask = R <= (PETAL_A + PETAL_B * np.sin(PETAL_N * T))
        xy = np.stack([X.ravel(), Y.ravel()], axis=1)
        u_pred = model(torch.tensor(xy, dtype=torch.float32)).detach().cpu().numpy().reshape(grid_n, grid_n)
        u_pred[~petal_mask] = np.nan
        u_true = analytic_solution(X, Y)
        u_true[~petal_mask] = np.nan
        error = np.abs(u_pred - u_true)
        error[~petal_mask] = np.nan
        plt.figure(figsize=(6,5))
        plt.imshow(u_pred, extent=[-1.3,1.3,-1.3,1.3], origin="lower", cmap="coolwarm")
        plt.colorbar()
        plt.title("Predicted Solution")
        plt.savefig(os.path.join(save_dir, f"solution_repeat{repeat+1}.png"))
        plt.close()
        plt.figure(figsize=(6,5))
        plt.imshow(error, extent=[-1.3,1.3,-1.3,1.3], origin="lower", cmap="magma")
        plt.colorbar()
        plt.title("Absolute Error")
        plt.savefig(os.path.join(save_dir, f"error_repeat{repeat+1}.png"))
        plt.close()
        # Save L2 error
        with open(os.path.join(save_dir, f"l2_error_repeat{repeat+1}.txt"), "w") as f:
            f.write(f"{rel_l2:.8e}\n")
        # Save architecture
        with open(os.path.join(save_dir, f"architecture_repeat{repeat+1}.txt"), "w") as f:
            f.write(architecture_signature(model) + "\n")
    # Save summary
    with open(os.path.join(save_dir, "l2_error_summary.txt"), "w") as f:
        f.write(f"Mean L2 error: {np.mean(l2_errors):.8e}\n")
        f.write(f"Std L2 error: {np.std(l2_errors):.8e}\n")
