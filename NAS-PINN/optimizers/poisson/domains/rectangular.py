import torch

def true_solution(x, y):
    """ Analitik çözüm: cos(πx) cos(πy) """
    return torch.cos(torch.pi * x) * torch.cos(torch.pi * y)

def poisson_rhs(x, y):
    """ f(x,y) = -2π² cos(πx) cos(πy) """
    return -2.0 * (torch.pi ** 2) * torch.cos(torch.pi * x) * torch.cos(torch.pi * y)

def sample_points(n_col, n_bc, device):
    x_col = torch.rand(n_col, 1, device=device) * 2 - 1
    y_col = torch.rand(n_col, 1, device=device) * 2 - 1
    n_side = n_bc // 4
    xb, yb = [], []
    for val in [-1.0, 1.0]:
        xb.append(torch.rand(n_side, 1, device=device) * 2 - 1)
        yb.append(torch.full((n_side, 1), val, device=device))
        xb.append(torch.full((n_side, 1), val, device=device))
        yb.append(torch.rand(n_side, 1, device=device) * 2 - 1)
    x_bc = torch.cat(xb)
    y_bc = torch.cat(yb)
    return (x_col, y_col), (x_bc, y_bc)
