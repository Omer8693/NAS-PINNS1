import torch
import numpy as np

def true_solution(x, y):
    return torch.cos(np.pi * x) * torch.cos(np.pi * y)

def poisson_rhs(x, y):
    return -2.0 * (np.pi ** 2) * torch.cos(np.pi * x) * torch.cos(np.pi * y)

def sample_points(n_col=7000, n_bc=1200, n_petals=6, amp=0.3, device=None):
    pts = []
    while len(pts) < n_col:
        cand = torch.rand(n_col * 6, 2, device=device) * 2.8 - 1.4
        r = torch.norm(cand, dim=1)
        theta = torch.atan2(cand[:,1], cand[:,0])
        r_max = 1.0 + amp * torch.sin(n_petals * theta)
        mask = r <= r_max
        pts.append(cand[mask][:n_col - len(pts)])
    xy_col = torch.cat(pts)
    x_col, y_col = xy_col[:,0:1], xy_col[:,1:2]
    theta_b = torch.linspace(0, 2*np.pi, n_bc, device=device)
    r_b = 1.0 + amp * torch.sin(n_petals * theta_b)
    x_bc = r_b * torch.cos(theta_b).unsqueeze(1)
    y_bc = r_b * torch.sin(theta_b).unsqueeze(1)
    return (x_col, y_col), (x_bc, y_bc)
