import numpy as np
import torch


RADIUS = 1.0
N_COL = 3000
N_BC = 400


def true_solution(x, y):
    return torch.cos(np.pi * x) * torch.cos(np.pi * y)


def poisson_rhs(x, y):
    return -2.0 * (np.pi**2) * torch.cos(np.pi * x) * torch.cos(np.pi * y)


def sample_points(n_col=N_COL, n_bc=N_BC, device=None, radius=RADIUS):
    r = radius * torch.sqrt(torch.rand(n_col, device=device))
    theta = 2.0 * np.pi * torch.rand(n_col, device=device)

    x_col = (r * torch.cos(theta)).unsqueeze(1)
    y_col = (r * torch.sin(theta)).unsqueeze(1)

    theta_bc = torch.linspace(0.0, 2.0 * np.pi, n_bc, device=device)
    x_bc = radius * torch.cos(theta_bc).unsqueeze(1)
    y_bc = radius * torch.sin(theta_bc).unsqueeze(1)
    return (x_col, y_col), (x_bc, y_bc)
