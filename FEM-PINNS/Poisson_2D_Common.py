import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_EVAL_PATH = "./Eval_Points/2D_Poisson_eval-points.json"


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def start_run_logging(save_dir, log_name="run.log"):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, log_name)
    log_file = open(log_path, "a", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = _TeeStream(original_stdout, log_file)
    sys.stderr = _TeeStream(original_stderr, log_file)

    def stop_logging():
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    return log_path, stop_logging


def sync_cuda():
    if device.type == "cuda":
        torch.cuda.synchronize()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lhs(dim, samples):
    result = np.empty((samples, dim), dtype=np.float32)
    for j in range(dim):
        perm = np.random.permutation(samples)
        result[:, j] = (perm + np.random.rand(samples)) / samples
    return result


def analytic_sol(x, y):
    return (x ** 2) * ((x - 1) ** 2) * y * ((y - 1) ** 2)


def rhs_poisson(x, y):
    return 2.0 * (
        (x ** 4) * (3.0 * y - 2.0)
        + (x ** 3) * (4.0 - 6.0 * y)
        + (x ** 2) * (6.0 * (y ** 3) - 12.0 * (y ** 2) + 9.0 * y - 2.0)
        - 6.0 * x * ((y - 1.0) ** 2) * y
        + ((y - 1.0) ** 2) * y
    )


def sample_points(n_col, n_bc, n_val=None, *, dtype=torch.float32, target_device=device):
    domain_points = torch.tensor(lhs(2, n_col), device=target_device, dtype=dtype)
    boundary_points = torch.tensor(lhs(2, n_bc), device=target_device, dtype=dtype)
    if n_val is None:
        return domain_points, boundary_points
    val_points = torch.tensor(lhs(2, n_val), device=target_device, dtype=dtype)
    return domain_points, boundary_points, val_points


def _forward(model, points, masks=None):
    if masks is None:
        return model(points)
    return model(points, masks)


def residual(model, points, masks=None):
    points = points.clone().detach().requires_grad_(True)

    u = _forward(model, points, masks)
    grad_u = torch.autograd.grad(
        u,
        points,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]

    u_xx = torch.autograd.grad(
        u_x,
        points,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        u_y,
        points,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
    )[0][:, 1:2]

    return u_xx + u_yy - rhs_poisson(points[:, 0:1], points[:, 1:2])


def pde_residual(model, points, masks=None):
    return torch.mean(residual(model, points, masks=masks) ** 2)


def pde_true(model, points, masks=None):
    pred = _forward(model, points, masks)
    true = analytic_sol(points[:, 0:1], points[:, 1:2])
    return torch.mean((pred - true) ** 2)


def boundary_dirichlet(model, points, masks=None):
    x = points[:, 0:1]
    y0 = torch.zeros_like(x)
    boundary_points = torch.cat([x, y0], dim=1)
    return torch.mean(_forward(model, boundary_points, masks) ** 2)


def boundary_neumann(model, points, masks=None):
    x = points[:, 0:1].clone().detach()
    y = points[:, 1:2].clone().detach()

    left = torch.cat([torch.zeros_like(y), y], dim=1).requires_grad_(True)
    right = torch.cat([torch.ones_like(y), y], dim=1).requires_grad_(True)
    top = torch.cat([x, torch.ones_like(x)], dim=1).requires_grad_(True)

    u_left = _forward(model, left, masks)
    u_right = _forward(model, right, masks)
    u_top = _forward(model, top, masks)

    grad_left = torch.autograd.grad(
        u_left,
        left,
        grad_outputs=torch.ones_like(u_left),
        create_graph=True,
    )[0]
    grad_right = torch.autograd.grad(
        u_right,
        right,
        grad_outputs=torch.ones_like(u_right),
        create_graph=True,
    )[0]
    grad_top = torch.autograd.grad(
        u_top,
        top,
        grad_outputs=torch.ones_like(u_top),
        create_graph=True,
    )[0]

    du_dx_0 = grad_left[:, 0:1]
    du_dx_1 = grad_right[:, 0:1]
    du_dy_1 = grad_top[:, 1:2]

    return (
        torch.mean(du_dx_0 ** 2)
        + torch.mean(du_dx_1 ** 2)
        + torch.mean(du_dy_1 ** 2)
    )


def supervised_outer_loss(model, points, masks=None):
    pred = _forward(model, points, masks)
    true = analytic_sol(points[:, 0:1], points[:, 1:2])
    return torch.mean((pred - true) ** 2)


def total_physics_loss(model, domain_points, boundary_points, masks=None):
    loss_pde = pde_residual(model, domain_points, masks=masks)
    loss_bc_d = boundary_dirichlet(model, boundary_points, masks=masks)
    loss_bc_n = boundary_neumann(model, boundary_points, masks=masks)
    total = loss_pde + loss_bc_d + loss_bc_n
    return total, loss_pde, loss_bc_d, loss_bc_n


def load_eval_points(
    eval_path=DEFAULT_EVAL_PATH,
    *,
    dtype=torch.float32,
    target_device=device,
):
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pts = np.array(data["mesh_coord"]["0"], dtype=np.float32)
    else:
        gx = np.linspace(0.0, 1.0, 201, dtype=np.float32)
        gy = np.linspace(0.0, 1.0, 201, dtype=np.float32)
        xx, yy = np.meshgrid(gx, gy, indexing="xy")
        pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    return torch.tensor(pts, device=target_device, dtype=dtype)


def evaluate_pinn_model(model, masks=None, *, dtype=torch.float32, target_device=device):
    eval_points = load_eval_points(dtype=dtype, target_device=target_device)

    model.eval()
    sync_cuda()
    start_eval = time.time()
    with torch.no_grad():
        u_pred = _forward(model, eval_points, masks).squeeze(-1)
    sync_cuda()
    eval_time = time.time() - start_eval

    u_true = analytic_sol(eval_points[:, 0], eval_points[:, 1]).squeeze(-1)
    rel_l2 = (torch.linalg.norm(u_pred - u_true) / torch.linalg.norm(u_true)).item()

    return (
        eval_points.detach().cpu().tolist(),
        u_pred.detach().cpu().tolist(),
        u_true.detach().cpu().tolist(),
        rel_l2,
        eval_time,
    )


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path, payload, *, indent=None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, default=_json_default)
