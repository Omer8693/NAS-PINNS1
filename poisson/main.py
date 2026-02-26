import argparse
import numpy as np

# Unified main script for Poisson PINN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
from poisson.domains import rectangular, circle, lshape, flower, annulus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- CONSTANTS ---
MASK_LEVELS     = [30, 50, 70, 90, 110]
BASE_NEURONS    = 110
LAYERS          = 5
EPOCHS_ADAM     = 12000
INNER_LR        = 1e-3
OUTER_LR        = 3e-4
OUTER_EVERY     = 5
LBFGS_MAX_ITER  = 1500
N_COL           = 4000
N_BC            = 400
TEST_GRID_SIZE  = 500

# Domain selection
DOMAIN_MODULES = {
    "rectangular": rectangular,
    "circle": circle,
    "lshape": lshape,
    "flower": flower,
    "annulus": annulus,
}


# --- MixedOp (skip + tanh + mask, similar to Eq.8 in the paper) ---
class MixedOp(nn.Module):
    def __init__(self, in_features, max_out_features):
        super().__init__()
        self.max_out = max_out_features
        self.linear = nn.Linear(in_features, max_out_features)
        self.alpha_skip   = nn.Parameter(torch.tensor(0.0))
        self.alpha_active = nn.Parameter(torch.tensor(0.0))
        self.alpha_mask   = nn.Parameter(0.1 * torch.randn(len(MASK_LEVELS)))

    def forward(self, x):
        skip = x
        if x.shape[-1] < self.max_out:
            skip = F.pad(skip, (0, self.max_out - x.shape[-1]))
        active = torch.tanh(self.linear(x))
        w_skip   = torch.sigmoid(self.alpha_skip)
        w_active = torch.sigmoid(self.alpha_active)
        mask_w   = F.softmax(self.alpha_mask, dim=0)
        masked_active = 0.0
        for i, lvl in enumerate(MASK_LEVELS):
            mask = torch.zeros(self.max_out, device=device)
            k = min(lvl, self.max_out)
            mask[:k] = 1.0
            masked_active += mask_w[i] * (active * mask)
        out = w_skip * skip + w_active * masked_active
        # If last layer, output shape should be [batch, 1]
        if self.max_out == 1:
            out = out[:, :1]
        return out


# --- NAS-PINN Model ---
class NAS_PINN(nn.Module):
    def __init__(self, num_layers=LAYERS, base_neurons=BASE_NEURONS):
        super().__init__()
        dims = [2] + [base_neurons] * (num_layers - 1) + [1]
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(MixedOp(dims[i], dims[i+1]))
    def forward(self, xy):
        x = xy
        for layer in self.layers:
            x = layer(x)
        return x


# --- LOSS FUNCTIONS ---
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


# --- PRINT DISCRETE ARCHITECTURE ---
def print_discrete_architecture(model):
    print("\n" + "="*60)
    print("DISCRETE ARCHITECTURE AFTER L-BFGS (most likely selections)")
    print("="*60)
    for i, layer in enumerate(model.layers):
        w_skip   = torch.sigmoid(layer.alpha_skip).item()
        w_active = torch.sigmoid(layer.alpha_active).item()
        mask_probs = F.softmax(layer.alpha_mask, dim=0)
        best_mask_idx = torch.argmax(mask_probs).item()
        best_neurons  = MASK_LEVELS[best_mask_idx]
        best_prob     = mask_probs[best_mask_idx].item()
        connection = "SKIP (identity)" if w_skip > w_active else "ACTIVE (tanh)"
        print(f"Layer {i+1:2d} → {connection:12s}  |  neurons: {best_neurons:3d}  "
              f"(prob: {best_prob:.4f}, skip weight: {w_skip:.4f}, active weight: {w_active:.4f})")
    print("="*60 + "\n")


# --- TRAINING FUNCTION ---
def train_nas_pinn(domain_mod):
    model = NAS_PINN().to(device)
    opt_inner = optim.Adam(model.parameters(), lr=INNER_LR)
    arch_params = []
    for layer in model.layers:
        arch_params.extend([layer.alpha_skip, layer.alpha_active, layer.alpha_mask])
    opt_outer = optim.Adam(arch_params, lr=OUTER_LR)
    loss_history = []
    print("Starting NAS-PINN training...")
    start_time = time.time()
    for epoch in range(EPOCHS_ADAM):
        (x_col, y_col), (x_bc, y_bc) = domain_mod.sample_points(N_COL, N_BC, device)
        opt_inner.zero_grad()
        l_pde = pde_loss(model, x_col, y_col, domain_mod)
        l_bc  = bc_loss(model, x_bc, y_bc, domain_mod)
        loss_inner = l_pde + l_bc
        loss_inner.backward()
        opt_inner.step()
        if epoch % OUTER_EVERY == 0:
            opt_outer.zero_grad()
            loss_outer = supervised_mse(model, x_col, y_col, domain_mod)
            loss_outer.backward()
            opt_outer.step()
        loss_history.append(loss_inner.item())
        if epoch % 2000 == 0 or epoch == EPOCHS_ADAM - 1:
            print(f"[{epoch:5d}] PDE: {l_pde:.4e}  BC: {l_bc:.4e}  Outer MSE: {loss_outer:.4e}")
    # L-BFGS refinement
    print("\nStarting L-BFGS refinement...")
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.8,
                                  max_iter=LBFGS_MAX_ITER,
                                  line_search_fn='strong_wolfe')
    def closure():
        optimizer_lbfgs.zero_grad()
        lp = pde_loss(model, x_col, y_col, domain_mod)
        lb = bc_loss(model, x_bc, y_bc, domain_mod)
        total = lp + lb
        total.backward()
        return total
    optimizer_lbfgs.step(closure)
    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.1f} minutes")
    print_discrete_architecture(model)
    return model, loss_history


# --- PLOT LOSS HISTORY ---
def plot_loss_history(loss_history, save_path="loss_history.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="Inner Loss (PDE + BC)", color='royalblue', linewidth=1.2)
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("NAS-PINN Training Loss History")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Loss plot saved: {save_path}")


# --- EVALUATE AND PLOT RESULTS ---
def evaluate_and_plot(model, domain_mod, save_dir="results_baseline"):
    os.makedirs(save_dir, exist_ok=True)
    nx = ny = TEST_GRID_SIZE
    x = torch.linspace(-1, 1, nx, device=device)
    y = torch.linspace(-1, 1, ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    xy_grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
    with torch.no_grad():
        pred = model(xy_grid).reshape(nx, ny).cpu().numpy()
        true = domain_mod.true_solution(X, Y).cpu().numpy()
        err_abs = np.abs(pred - true)
    err_sq = (pred - true)**2
    rel_l2 = np.sqrt(np.mean(err_sq)) / np.sqrt(np.mean(true**2))
    print(f"\nRelatif L² hata (grid {nx}×{ny}): {rel_l2:.4e}")
    # Plot: Exact, Predicted, Error (user color scheme)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin = np.min(true)
    vmax = np.max(true)
    im0 = axes[0].imshow(true.T, origin='lower', cmap='YlGnBu', extent=[-1,1,-1,1], vmin=vmin, vmax=vmax)
    axes[0].set_title("Exact")
    plt.colorbar(im0, ax=axes[0], shrink=0.6)
    im1 = axes[1].imshow(pred.T, origin='lower', cmap='YlGnBu', extent=[-1,1,-1,1], vmin=vmin, vmax=vmax)
    axes[1].set_title("Predicted (NAS-PINN)")
    plt.colorbar(im1, ax=axes[1], shrink=0.6)
    im2 = axes[2].imshow(err_abs.T, origin='lower', cmap='YlGnBu', extent=[-1,1,-1,1])
    axes[2].set_title("|Pred - Exact|")
    plt.colorbar(im2, ax=axes[2], shrink=0.6)
    plt.suptitle(f"NAS-PINN – {domain_mod.__name__.capitalize()} Domain\nRel L² = {rel_l2:.4e}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "result_comparison.png"), dpi=300)
    plt.close()
    # Save results to CSV
    csv_path = os.path.join(save_dir, "results_summary.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "exact", "predicted", "abs_error"])
        for i in range(nx):
            for j in range(ny):
                writer.writerow([float(X[i,j]), float(Y[i,j]), float(true[i,j]), float(pred[i,j]), float(err_abs[i,j])])
    print(f"Sonuçlar CSV'ye kaydedildi: {csv_path}")
    # GPU memory cleanup
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU belleği temizlendi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAS-PINN Poisson Solver")
    parser.add_argument("--domain", type=str, default="rectangular", choices=list(DOMAIN_MODULES.keys()), help="Domain type")
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    domain_mod = DOMAIN_MODULES[args.domain]
    model, loss_hist = train_nas_pinn(domain_mod)

    results_dir = "results_baseline"
    os.makedirs(results_dir, exist_ok=True)
    plot_loss_history(loss_hist, save_path=os.path.join(results_dir, "loss_history.png"))
    if args.domain == "rectangular":
        evaluate_and_plot(model, domain_mod, save_dir=results_dir)

# 🔥 EKLE
    del model
    del loss_hist
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# (Removed duplicate/legacy code below)
