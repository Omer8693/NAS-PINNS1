# ============================================================
# Project: Enhancing PINNs with Automated Architecture and Training Strategies
# Description (TR):
# Burgers denklemini PINN ile çözerken Genetik Algoritma (GA) kullanarak 
# en uygun ağ mimarisini seçer. Eğitim sürecinde önce Adam, ardından L-BFGS 
# optimizasyonu uygulanır.
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random, os, time

# ----------------------------
# PINN Model Definition
# ----------------------------
class BurgerPINN(nn.Module):
    def __init__(self, layers, neurons):
        super(BurgerPINN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.Tanh()
        input_dim, output_dim = 2, 1

        prev_dim = input_dim
        for _ in range(layers):
            self.layers.append(nn.Linear(prev_dim, neurons, dtype=torch.double))
            prev_dim = neurons
        self.out_layer = nn.Linear(prev_dim, output_dim, dtype=torch.double)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        out = inputs
        for layer in self.layers:
            out = self.activation(layer(out))
        return self.out_layer(out)

    def f_residual(self, x, t):
        u = self.forward(x, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        f = u_t + u * u_x - (0.01 / np.pi) * u_xx
        return f

# ----------------------------
# Data Generation
# ----------------------------
def generate_data(N_bc=200, N_f=10000):
    x0 = np.random.uniform(-1, 1, (N_bc, 1))
    t0 = np.zeros_like(x0)
    u0 = -np.sin(np.pi * x0)

    tb = np.random.uniform(0, 1, (N_bc, 1))
    xb_left = -np.ones_like(tb)
    xb_right = np.ones_like(tb)
    ub_left = np.zeros_like(tb)
    ub_right = np.zeros_like(tb)

    x_bc = np.vstack((x0, xb_left, xb_right))
    t_bc = np.vstack((t0, tb, tb))
    u_bc = np.vstack((u0, ub_left, ub_right))

    x_f = np.random.uniform(-1, 1, (N_f, 1))
    t_f = np.random.uniform(0, 1, (N_f, 1))

    return (
        torch.tensor(x_bc, dtype=torch.double, requires_grad=True),
        torch.tensor(t_bc, dtype=torch.double, requires_grad=True),
        torch.tensor(u_bc, dtype=torch.double),
        torch.tensor(x_f, dtype=torch.double, requires_grad=True),
        torch.tensor(t_f, dtype=torch.double, requires_grad=True),
    )

# ----------------------------
# Training Function (Adam + L-BFGS)
# ----------------------------
def train_pinn(model, data, epochs=2000, lr=1e-3, device='cpu'):
    x_bc, t_bc, u_bc, x_f, t_f = data
    loss_history = []

    # 1️⃣ Adam optimizer (coarse training)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        u_pred = model(x_bc, t_bc)
        f_pred = model.f_residual(x_f, t_f)

        mse_u = torch.mean((u_pred - u_bc) ** 2)
        mse_f = torch.mean(f_pred ** 2)
        loss = mse_u + mse_f
        loss.backward()
        optimizer.step()

        loss_history.append([epoch, loss.item(), mse_u.item(), mse_f.item()])

        if epoch % 200 == 0:
            print(f"[Adam] Epoch {epoch}: Loss={loss.item():.4e}, MSE_u={mse_u.item():.4e}, MSE_f={mse_f.item():.4e}")

    # 2️⃣ L-BFGS optimizer (fine-tuning)
    print("\n🔧 Starting L-BFGS fine-tuning...")
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=500,
        tolerance_grad=1e-8,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        u_pred = model(x_bc, t_bc)
        f_pred = model.f_residual(x_f, t_f)
        mse_u = torch.mean((u_pred - u_bc) ** 2)
        mse_f = torch.mean(f_pred ** 2)
        loss = mse_u + mse_f
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)
    print("✅ L-BFGS optimization completed.\n")

    return pd.DataFrame(loss_history, columns=['epoch', 'loss', 'mse_u', 'mse_f'])

# ----------------------------
# Genetic Algorithm for Architecture Search
# ----------------------------
def evaluate_architecture(layers, neurons, lr):
    torch.manual_seed(0)
    data = generate_data()
    model = BurgerPINN(layers, neurons).double()
    log = train_pinn(model, data, epochs=1000, lr=lr)
    final_loss = log['loss'].iloc[-1]
    return final_loss, model, log

def ga_search(pop_size=6, generations=4):
    population = [(random.randint(2, 8), random.randint(10, 60), 10**random.uniform(-4, -2)) for _ in range(pop_size)]
    best_model, best_loss, best_log = None, 1e9, None

    for gen in range(generations):
        print(f"\n=== Generation {gen+1}/{generations} ===")
        new_population = []
        for (layers, neurons, lr) in population:
            loss, model, log = evaluate_architecture(layers, neurons, lr)
            print(f"Config (L={layers}, N={neurons}, lr={lr:.1e}) -> Loss={loss:.4e}")
            if loss < best_loss:
                best_loss, best_model, best_log = loss, model, log
            new_population.append((layers, neurons, lr, loss))

        new_population.sort(key=lambda x: x[3])
        elites = new_population[:2]
        population = []
        for _ in range(pop_size):
            p1, p2 = random.sample(elites, 2)
            child_layers = max(2, min(8, random.choice([p1[0], p2[0]]) + random.randint(-1, 1)))
            child_neurons = max(10, min(60, random.choice([p1[1], p2[1]]) + random.randint(-5, 5)))
            child_lr = random.choice([p1[2], p2[2]]) * random.uniform(0.8, 1.2)
            population.append((child_layers, child_neurons, child_lr))

    return best_model, best_log

# ----------------------------
# Visualization
# ----------------------------
def plot_loss(log):
    plt.figure()
    plt.semilogy(log['epoch'], log['loss'], label='Total Loss')
    plt.semilogy(log['epoch'], log['mse_u'], label='MSE_u')
    plt.semilogy(log['epoch'], log['mse_f'], label='MSE_f')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_plot_burgers_1.png', dpi=300)

def plot_snapshots(model):
    x = torch.linspace(-1, 1, 200).reshape(-1, 1).double()
    t_vals = [0.0, 0.25, 0.5, 0.75]
    plt.figure(figsize=(8, 6))
    for i, t in enumerate(t_vals, 1):
        t_tensor = torch.ones_like(x) * t
        u_pred = model(x, t_tensor).detach().numpy()
        plt.subplot(2, 2, i)
        plt.plot(x.detach().numpy(), u_pred)
        plt.title(f"t={t}")
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
    plt.tight_layout()
    plt.savefig('predictions_burgers_1.png', dpi=300)

def plot_heatmap(model, nx=100, nt=100):
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    X_tensor = torch.tensor(X.flatten(), dtype=torch.double).reshape(-1, 1)
    T_tensor = torch.tensor(T.flatten(), dtype=torch.double).reshape(-1, 1)
    with torch.no_grad():
        U = model(X_tensor, T_tensor).reshape(nt, nx).numpy()
    plt.figure(figsize=(8, 5))
    plt.imshow(U, extent=[-1, 1, 0, 1], origin='lower', aspect='auto', cmap='jet')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Spatio-temporal Heatmap of u(x,t)')
    plt.savefig('heatmap_burgers_1.png', dpi=300)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    start = time.time()
    best_model, best_log = ga_search()

    torch.save(best_model.state_dict(), 'best_model_burgers_1.pt')
    best_log.to_csv('train_log_burgers_1.csv', index=False)

    plot_loss(best_log)
    plot_snapshots(best_model)
    plot_heatmap(best_model)

    print(f"\nTraining complete in {time.time()-start:.1f}s. Best model saved as 'best_model_burgers_1.pt'")
