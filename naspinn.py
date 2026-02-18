import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class ResidualBurgerPINN(nn.Module):
    """
    PINN with optional residual (skip) connections – inspired by NAS-PINN style architectures
    """
    def __init__(self, layers, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.architecture = layers

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        identity = inputs

        for i, layer in enumerate(self.layers[:-1]):
            out = torch.tanh(layer(inputs))
            if self.use_residual and out.shape[-1] == identity.shape[-1]:
                out = out + identity
            inputs = out
            identity = inputs  # dense residual style

        return self.layers[-1](inputs)


def generate_data(n_collocation=15000, n_boundary=200, n_initial=200):
    """
    Generate collocation, boundary and initial condition points
    Slightly more points than usual for better gradient estimation
    """
    # Collocation points (interior)
    x_pde = torch.rand(n_collocation, 1, dtype=torch.float64) * 2 - 1
    t_pde = torch.rand(n_collocation, 1, dtype=torch.float64)

    # Boundary conditions: u(t, -1) = u(t, 1) = 0
    x_bc_left = torch.full((n_boundary, 1), -1.0, dtype=torch.float64)
    x_bc_right = torch.full((n_boundary, 1), 1.0, dtype=torch.float64)
    t_bc = torch.rand(n_boundary, 1, dtype=torch.float64)
    x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
    t_bc = torch.cat([t_bc, t_bc], dim=0)
    u_bc = torch.zeros_like(x_bc)

    # Initial condition: u(0, x) = -sin(πx)
    x_ic = torch.rand(n_initial, 1, dtype=torch.float64) * 2 - 1
    t_ic = torch.zeros_like(x_ic)
    u_ic = -torch.sin(np.pi * x_ic)

    return {
        'x_coll': x_pde,    # Changed from 'x_pde' to 'x_coll'
        't_coll': t_pde,    # Changed from 't_pde' to 't_coll'
        'x_bc': x_bc, 
        't_bc': t_bc, 
        'u_bc': u_bc,
        'x_ic': x_ic, 
        't_ic': t_ic, 
        'u_ic': u_ic
    }


def pde_loss(model, x, t, nu):
    """Burgers PDE residual: u_t + u u_x - (ν/π) u_xx = 0"""
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    u = model(x, t)

    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

    residual = u_t + u * u_x - (nu / np.pi) * u_xx
    return torch.mean(residual ** 2)


def total_loss(model, data, nu, lambda_pde=100.0, lambda_bc=20.0, lambda_ic=5.0):
    """
    Compute total loss: PDE residual + boundary conditions + initial conditions
    Returns: scalar tensor (not tuple)
    """
    x_coll = data['x_coll'].clone().detach().requires_grad_(True)
    t_coll = data['t_coll'].clone().detach().requires_grad_(True)
    x_ic = data['x_ic'].clone().detach()
    x_bc = data['x_bc'].clone().detach()
    t_bc = data['t_bc'].clone().detach().requires_grad_(True)
    
    # PDE residual loss
    u_coll = model(x_coll, t_coll)
    u_t = torch.autograd.grad(u_coll.sum(), t_coll, create_graph=True)[0]
    u_x = torch.autograd.grad(u_coll.sum(), x_coll, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x_coll, create_graph=True)[0]
    
    # Burgers' equation: u_t + u*u_x - nu*u_xx = 0
    pde_residual = u_t + u_coll * u_x - nu * u_xx
    loss_pde = torch.mean(pde_residual ** 2)
    
    # Initial condition loss: u(x, 0) = -sin(πx)
    u_ic = model(x_ic, torch.zeros_like(x_ic))
    u_ic_true = -torch.sin(torch.pi * x_ic)
    loss_ic = torch.mean((u_ic - u_ic_true) ** 2)
    
    # Boundary condition loss: u(-1, t) = u(1, t) = 0
    u_bc_left = model(x_bc[:len(x_bc)//2], t_bc[:len(t_bc)//2])
    u_bc_right = model(x_bc[len(x_bc)//2:], t_bc[len(t_bc)//2:])
    loss_bc = torch.mean(u_bc_left ** 2) + torch.mean(u_bc_right ** 2)
    
    # Total loss (scalar)
    total = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
    
    return total  # Return scalar, not tuple


def train_pinn(model, nu, data,
               epochs=1200,
               lr=3e-4,
               lambda_pde=100.0,
               lambda_bc=20.0,
               lambda_ic=5.0,
               patience=100,
               verbose=True,
               use_lbfgs=False):
    """
    Train with Adam + learning rate scheduler + early stopping
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=50, factor=0.5
    )

    best_loss = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute loss - should be scalar tensor
        loss = total_loss(model, data, nu, lambda_pde, lambda_bc, lambda_ic)
        
        # Ensure loss is a scalar
        if isinstance(loss, tuple):
            loss = loss[0]
        
        loss_val = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        history.append(loss_val)
        
        if loss_val < best_loss:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose and epoch % 100 == 0:
                print(f"Early stopping at epoch {epoch}")
            break
        
        scheduler.step(loss_val)

    return model, history


def compute_mean_l2_error(model, nu, nt=100, nx=300):
    """
    Compute time-averaged relative L² error over [0,1]×[-1,1]
    More robust than evaluating only at t=1
    """
    x = torch.linspace(-1, 1, nx, dtype=torch.float64).unsqueeze(1)
    t_vals = torch.linspace(0, 1, nt)

    errors = []
    for ti in t_vals:
        t = ti * torch.ones_like(x)
        with torch.no_grad():
            u_pred = model(x, t).squeeze().cpu().numpy()

        u_exact = -np.sin(np.pi * x.squeeze().numpy()) * np.exp(-nu * np.pi**2 * ti.item())
        u_exact[0] = u_exact[-1] = 0.0

        norm_diff = np.linalg.norm(u_pred - u_exact)
        norm_exact = np.linalg.norm(u_exact) + 1e-12
        errors.append(norm_diff / norm_exact)

    return np.mean(errors), np.std(errors)


def nas_search(num_generations=12, population_size=24, verbose=True):
    """
    Simple population-based architecture search inspired by NAS-PINN ideas
    - Wider search space
    - Residual connections
    - Time-averaged L2 error as objective
    """
    nu = 0.01
    data = generate_data()

    print("\n" + "=" * 90)
    print("NAS-PINN style architecture search – Burgers equation")
    print(f"  Viscosity ν = {nu:.4f}    Population = {population_size}    Generations = {num_generations}")
    print("=" * 90 + "\n")

    population = []
    best_individual = None
    best_mean_l2 = float('inf')

    for gen in tqdm(range(1, num_generations + 1), desc="Generation"):
        print(f"\nGeneration {gen}/{num_generations}")

        for i in range(population_size):
            # Random architecture sampling (NAS-PINN inspired range)
            n_hidden = np.random.randint(4, 11)                    # 4–10 hidden layers
            neurons = [np.random.randint(48, 257) for _ in range(n_hidden)]
            use_residual = np.random.choice([True, False], p=[0.75, 0.25])

            layers = [2] + neurons + [1]
            print(f"  [{i+1:2d}] layers = {layers}    residual = {use_residual}")

            model = ResidualBurgerPINN(layers, use_residual=use_residual).double()

            # Train (Adam only – fast)
            model, loss_history = train_pinn(
                model,
                nu,
                data,
                epochs=1000,
                lr=np.random.uniform(1e-4, 8e-4),
                lambda_pde=np.random.uniform(80, 130),   # slight variation
                lambda_bc=20.0,
                lambda_ic=5.0,
                patience=120,
                verbose=True,
                use_lbfgs=False
            )

            mean_l2, std_l2 = compute_mean_l2_error(model, nu)

            individual = {
                'layers': layers,
                'use_residual': use_residual,
                'mean_l2': mean_l2,
                'std_l2': std_l2,
                'params': sum(p.numel() for p in model.parameters()),
                'model_state': model.state_dict(),  # save state for later refinement
                'loss_history': loss_history
            }

            population.append(individual)

            if mean_l2 < best_mean_l2:
                best_mean_l2 = mean_l2
                best_individual = individual
                print(f"     → New best: mean L² = {mean_l2:.2e} ± {std_l2:.2e}")

    print("\n" + "=" * 90)
    print("FINAL BEST ARCHITECTURE (according to time-averaged L² error)")
    print(f"  Architecture   : {best_individual['layers']}")
    print(f"  Residual       : {best_individual['use_residual']}")
    print(f"  Mean L² error  : {best_individual['mean_l2']:.2e} ± {best_individual['std_l2']:.2e}")
    print(f"  Parameters     : {best_individual['params']:,}")
    print("=" * 90)

    return best_individual, population


def plot_loss_curve(loss_history, save_path=None):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_solution_heatmap(model, nu, save_path=None, nt=100, nx=300):
    """Plot solution as a heatmap over space-time domain"""
    x = torch.linspace(-1, 1, nx, dtype=torch.float64).unsqueeze(1)
    t_vals = torch.linspace(0, 1, nt)
    
    u_pred = np.zeros((nt, nx))
    for i, ti in enumerate(t_vals):
        t = ti * torch.ones_like(x)
        with torch.no_grad():
            u_pred[i] = model(x, t).squeeze().cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.contourf(x.squeeze().numpy(), t_vals.numpy(), u_pred, levels=50, cmap='RdBu_r')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('t', fontsize=12)
    plt.title(f'Solution Heatmap (ν={nu:.4f})', fontsize=14)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_solution_snapshots(model, nu, save_path=None, nt=5, nx=300):
    """Plot solution snapshots at different time steps"""
    x = torch.linspace(-1, 1, nx, dtype=torch.float64).unsqueeze(1)
    t_vals = torch.linspace(0, 1, nt)
    
    plt.figure(figsize=(14, 8))
    for i, ti in enumerate(t_vals):
        t = ti * torch.ones_like(x)
        with torch.no_grad():
            u_pred = model(x, t).squeeze().cpu().numpy()
        
        u_exact = -np.sin(np.pi * x.squeeze().numpy()) * np.exp(-nu * np.pi**2 * ti.item())
        
        plt.subplot(2, 3, i+1)
        plt.plot(x.squeeze().numpy(), u_pred, 'b-', linewidth=2, label='Pred')
        plt.plot(x.squeeze().numpy(), u_exact, 'r--', linewidth=2, label='Exact')
        plt.xlabel('x', fontsize=10)
        plt.ylabel('u(x,t)', fontsize=10)
        plt.title(f't = {ti:.2f}', fontsize=11)
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison_at_final_time(model, nu, save_path=None, nx=300):
    """Plot prediction vs exact solution at final time t=1"""
    x = torch.linspace(-1, 1, nx, dtype=torch.float64).unsqueeze(1)
    t = torch.ones_like(x)
    
    with torch.no_grad():
        u_pred = model(x, t).squeeze().cpu().numpy()
    
    u_exact = -np.sin(np.pi * x.squeeze().numpy()) * np.exp(-nu * np.pi**2)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x.squeeze().numpy(), u_pred, 'b-', linewidth=2.5, label='PINN Prediction')
    plt.plot(x.squeeze().numpy(), u_exact, 'r--', linewidth=2.5, label='Exact Solution')
    plt.fill_between(x.squeeze().numpy(), u_pred, u_exact, alpha=0.2, color='gray')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x, t=1)', fontsize=12)
    plt.title(f'Solution at Final Time t=1 (ν={nu:.4f})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_error_distribution(model, nu, save_path=None, nt=50, nx=300):
    """Plot pointwise L² error distribution"""
    x = torch.linspace(-1, 1, nx, dtype=torch.float64).unsqueeze(1)
    t_vals = torch.linspace(0, 1, nt)
    
    errors = np.zeros((nt, nx))
    for i, ti in enumerate(t_vals):
        t = ti * torch.ones_like(x)
        with torch.no_grad():
            u_pred = model(x, t).squeeze().cpu().numpy()
        
        u_exact = -np.sin(np.pi * x.squeeze().numpy()) * np.exp(-nu * np.pi**2 * ti.item())
        errors[i] = np.abs(u_pred - u_exact)
    
    plt.figure(figsize=(12, 6))
    plt.contourf(x.squeeze().numpy(), t_vals.numpy(), errors, levels=50, cmap='viridis')
    plt.colorbar(label='|u_pred - u_exact|')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('t', fontsize=12)
    plt.title(f'Pointwise Error Distribution (ν={nu:.4f})', fontsize=14)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    start_time = time.time()
    best, all_pop = nas_search(num_generations=12, population_size=24)
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes  ({elapsed:.0f} seconds)")