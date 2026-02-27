import torch
import numpy as np

def true_solution(x, y):
    return torch.cos(np.pi * x) * torch.cos(np.pi * y)

def poisson_rhs(x, y):
    return -2.0 * (np.pi ** 2) * torch.cos(np.pi * x) * torch.cos(np.pi * y)

def sample_points(n_col=6000, n_bc=1000, device=None):
    collected = []
    total = 0
    while total < n_col:
        cand = torch.rand(n_col * 4, 2, device=device) * 3.0 - 1.0
        x, y = cand[:, 0], cand[:, 1]
        inside = (
            ((x >= -1.0) & (x <= 2.0) & (y >= -1.0) & (y <= 1.0))
            | ((x >= -1.0) & (x <= 1.0) & (y >= 1.0) & (y <= 2.0))
        )
        valid = cand[inside]
        if valid.numel() == 0:
            continue
        take = min(n_col - total, valid.shape[0])
        collected.append(valid[:take])
        total += take

    xy_col = torch.cat(collected, dim=0)
    x_col, y_col = xy_col[:, 0:1], xy_col[:, 1:2]

    # Boundary segments of the L-shape
    n_seg = max(n_bc // 6, 1)
    rem = n_bc - (n_seg * 6)

    counts = [n_seg] * 6
    for i in range(rem):
        counts[i % 6] += 1

    x_bc_parts = []
    y_bc_parts = []

    # y = -1, x in [-1, 2]
    x_bc_parts.append(torch.rand(counts[0], 1, device=device) * 3.0 - 1.0)
    y_bc_parts.append(torch.full((counts[0], 1), -1.0, device=device))
    # x = 2, y in [-1, 1]
    x_bc_parts.append(torch.full((counts[1], 1), 2.0, device=device))
    y_bc_parts.append(torch.rand(counts[1], 1, device=device) * 2.0 - 1.0)
    # y = 1, x in [1, 2]
    x_bc_parts.append(torch.rand(counts[2], 1, device=device) + 1.0)
    y_bc_parts.append(torch.full((counts[2], 1), 1.0, device=device))
    # x = 1, y in [1, 2]
    x_bc_parts.append(torch.full((counts[3], 1), 1.0, device=device))
    y_bc_parts.append(torch.rand(counts[3], 1, device=device) + 1.0)
    # y = 2, x in [-1, 1]
    x_bc_parts.append(torch.rand(counts[4], 1, device=device) * 2.0 - 1.0)
    y_bc_parts.append(torch.full((counts[4], 1), 2.0, device=device))
    # x = -1, y in [-1, 2]
    x_bc_parts.append(torch.full((counts[5], 1), -1.0, device=device))
    y_bc_parts.append(torch.rand(counts[5], 1, device=device) * 3.0 - 1.0)

    x_bc = torch.cat(x_bc_parts, dim=0)
    y_bc = torch.cat(y_bc_parts, dim=0)
    return (x_col, y_col), (x_bc, y_bc)
