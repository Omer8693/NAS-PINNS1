import torch
import numpy as np

R_INNER = 0.3
R_OUTER = 1.0
N_COL = 6000
N_BC = 1000
TEST_GRID_SIZE = 500

def true_solution(x, y):
    return torch.cos(np.pi * x) * torch.cos(np.pi * y)

def poisson_rhs(x, y):
    return -2.0 * (np.pi ** 2) * torch.cos(np.pi * x) * torch.cos(np.pi * y)

def sample_points(n_col=N_COL, n_bc=N_BC, r_inner=R_INNER, r_outer=R_OUTER, device=None):
    collected = []
    total = 0
    while total < n_col:
        cand = torch.rand(n_col * 4, 2, device=device) * 2 - 1
        r = torch.norm(cand, dim=1)
        inside = (r >= r_inner) & (r <= r_outer)
        valid = cand[inside]
        if valid.numel() == 0:
            continue
        take = min(n_col - total, valid.shape[0])
        collected.append(valid[:take])
        total += take

    xy_col = torch.cat(collected, dim=0)
    x_col, y_col = xy_col[:, 0:1], xy_col[:, 1:2]

    theta = torch.linspace(0, 2 * np.pi, n_bc // 2, device=device)
    x_inner = r_inner * torch.cos(theta).unsqueeze(1)
    y_inner = r_inner * torch.sin(theta).unsqueeze(1)
    x_outer = r_outer * torch.cos(theta).unsqueeze(1)
    y_outer = r_outer * torch.sin(theta).unsqueeze(1)
    x_bc = torch.cat([x_inner, x_outer])
    y_bc = torch.cat([y_inner, y_outer])
    return (x_col, y_col), (x_bc, y_bc)
