import torch
import numpy as np

def true_solution(x, y):
    return torch.cos(np.pi * x) * torch.cos(np.pi * y)

def poisson_rhs(x, y):
    return -2.0 * (np.pi ** 2) * torch.cos(np.pi * x) * torch.cos(np.pi * y)

def sample_points(n_col=6000, n_bc=1000, device=None):
    pts = []
    while len(pts) < n_col:
        cand = torch.rand(n_col * 5, 2, device=device) * 3 - 1  # [-1,2] aralığı
        x, y = cand[:,0], cand[:,1]
        mask = ((x >= -1) & (x <= 2) & (y >= -1) & (y <= 1)) | \
               ((x >= -1) & (x <= 1) & (y >= 1) & (y <= 2))
        pts.append(cand[mask][:n_col - len(pts)])
    xy_col = torch.cat(pts)
    x_col, y_col = xy_col[:,0:1], xy_col[:,1:2]
    # Boundary noktaları burada None, ana kodda boundary sampling eklenebilir
    return (x_col, y_col), (None, None)
