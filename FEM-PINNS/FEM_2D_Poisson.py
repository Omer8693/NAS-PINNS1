import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from Poisson_2D_Common import (
    analytic_sol,
    device,
    load_eval_points,
    rhs_poisson,
    save_json,
    start_run_logging,
    sync_cuda,
)


dtype = torch.float64


def forcing(x, y):
    return -rhs_poisson(x, y)


def apply_operator(u_unknown, h):
    nx, ny = u_unknown.shape

    u_full = torch.zeros((nx, ny + 1), device=u_unknown.device, dtype=u_unknown.dtype)
    u_full[:, 1:] = u_unknown

    center = u_full[:, 1:]

    left = torch.empty_like(center)
    left[1:, :] = u_full[:-1, 1:]
    left[0, :] = u_full[1, 1:]

    right = torch.empty_like(center)
    right[:-1, :] = u_full[1:, 1:]
    right[-1, :] = u_full[-2, 1:]

    down = u_full[:, :-1]

    up = torch.empty_like(center)
    up[:, :-1] = u_full[:, 2:]
    up[:, -1] = u_full[:, -2]

    lap = (left + right + down + up - 4.0 * center) / (h * h)
    return -lap


@torch.no_grad()
def conjugate_gradient(linear_op, b, tol=1e-10, max_iter=5000):
    x = torch.zeros_like(b)
    r = b - linear_op(x)
    p = r.clone()
    rs_old = torch.sum(r * r)

    if torch.sqrt(rs_old).item() < tol:
        return x, 0, torch.sqrt(rs_old).item()

    for it in range(1, max_iter + 1):
        ap = linear_op(p)
        alpha = rs_old / torch.sum(p * ap)

        x = x + alpha * p
        r = r - alpha * ap

        rs_new = torch.sum(r * r)
        if torch.sqrt(rs_new).item() < tol:
            return x, it, torch.sqrt(rs_new).item()

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, max_iter, torch.sqrt(rs_old).item()


@torch.no_grad()
def solve_poisson(num, cg_tol=1e-10, cg_max_iter=5000):
    h = 1.0 / num

    x = torch.linspace(0.0, 1.0, num + 1, device=device, dtype=dtype)
    y = torch.linspace(h, 1.0, num, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    b = forcing(xx, yy)
    u_unknown, iters, final_res = conjugate_gradient(
        lambda u: apply_operator(u, h),
        b,
        tol=cg_tol,
        max_iter=cg_max_iter,
    )

    u_full = torch.zeros((num + 1, num + 1), device=device, dtype=dtype)
    u_full[:, 1:] = u_unknown
    return u_full, iters, final_res


@torch.no_grad()
def evaluate_on_points(u_full, eval_points):
    u_image = u_full.t().unsqueeze(0).unsqueeze(0).contiguous()
    grid = (2.0 * eval_points - 1.0).view(1, -1, 1, 2)

    values = F.grid_sample(
        u_image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return values.view(-1)


def main():
    save_dir = "./2D-Poisson-FEM"
    _, stop_logging = start_run_logging(save_dir)
    try:
        print(device)

        eval_points_t = load_eval_points(dtype=dtype, target_device=device)
        eval_points_np = eval_points_t.detach().cpu().numpy()
        nums = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

        y_results = {}
        times_solve = {}
        times_eval = {}
        l2_rel = {}

        for num in nums:
            tot_solve = 0.0
            tot_eval = 0.0

            for _ in range(10):
                sync_cuda()
                start_solve = time.time()
                u_grid, cg_iters, final_res = solve_poisson(num, cg_tol=1e-10, cg_max_iter=5000)
                sync_cuda()
                tot_solve += time.time() - start_solve

                sync_cuda()
                start_eval = time.time()
                y_approx_t = evaluate_on_points(u_grid, eval_points_t)
                sync_cuda()
                tot_eval += time.time() - start_eval

            time_solving = tot_solve / 10.0
            time_evaluation = tot_eval / 10.0

            print("Start comparing to GT", num)

            y_approx = y_approx_t.detach().cpu().numpy()
            y_true = analytic_sol(eval_points_np[:, 0], eval_points_np[:, 1])

            l2 = np.linalg.norm(y_true - y_approx)
            l2_rel_single = l2 / np.linalg.norm(y_true)

            print("Average solution time", time_solving)
            print("Average evaluation time", time_evaluation)
            print("Average accuracy:", l2_rel_single)
            print("Last CG iterations:", cg_iters)
            print("Last CG residual:", final_res)

            y_results[num] = y_approx.tolist()
            times_solve[num] = time_solving
            times_eval[num] = time_evaluation
            l2_rel[num] = l2_rel_single

            results = {
                "y_results": y_results,
                "times_solve": times_solve,
                "times_eval": times_eval,
                "l2_rel": l2_rel,
            }

            save_json(os.path.join(save_dir, "FEM_results.json"), results)
    finally:
        stop_logging()


if __name__ == "__main__":
    main()
