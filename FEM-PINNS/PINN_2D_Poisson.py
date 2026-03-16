import os
import time

import numpy as np
import torch
import torch.nn as nn

from Poisson_2D_Common import (
    device,
    evaluate_pinn_model,
    sample_points,
    save_json,
    set_seed,
    start_run_logging,
    sync_cuda,
    total_physics_loss,
)


architecture_list = [
    [20, 1],
    [60, 1],
    [20, 20, 1],
    [60, 60, 1],
    [20, 20, 20, 1],
    [60, 60, 60, 1],
    [20, 20, 20, 20, 1],
    [60, 60, 60, 60, 1],
    [20, 20, 20, 20, 20, 1],
    [60, 60, 60, 60, 60, 1],
    [120, 120, 120, 120, 120, 1],
]
lr = 1e-3
num_epochs = 20000


class PDESolution(nn.Module):
    def __init__(self, features):
        super().__init__()
        layers = []
        in_dim = 2

        for feat in features[:-1]:
            layers.append(nn.Linear(in_dim, feat))
            layers.append(nn.Tanh())
            in_dim = feat

        layers.append(nn.Linear(in_dim, features[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def training_step(model, optimizer):
    domain_points, boundary_points = sample_points(2000, 250)

    optimizer.zero_grad(set_to_none=True)
    loss, loss_pde, loss_bc_d, loss_bc_n = total_physics_loss(
        model,
        domain_points,
        boundary_points,
    )
    loss.backward()
    optimizer.step()

    return (
        float(loss.detach().cpu()),
        float(loss_pde.detach().cpu()),
        float(loss_bc_d.detach().cpu()),
        float(loss_bc_n.detach().cpu()),
    )


def train_loop(model, optimizer):
    losses = []
    for _ in range(num_epochs):
        loss_val, _, _, _ = training_step(model, optimizer)
        losses.append(loss_val)
    return losses


def main():
    save_dir = "./2D-Poisson-PINN"
    _, stop_logging = start_run_logging(save_dir)
    try:
        print(device)

        y_results = {}
        domain_pts = {}
        times_adam = {}
        times_lbfgs = {}
        times_total = {}
        times_eval = {}
        l2_rel = {}
        var = {}
        arch = {}

        for idx, feature in enumerate(architecture_list):
            times_adam_temp = []
            times_lbfgs_temp = []
            times_total_temp = []
            times_eval_temp = []
            accuracy_temp = []

            for _ in range(10):
                set_seed(17)
                model = PDESolution(feature).to(device)
                adam = torch.optim.Adam(model.parameters(), lr=lr)

                sync_cuda()
                start_time = time.time()
                train_loop(model, adam)
                sync_cuda()
                adam_time = time.time() - start_time
                times_adam_temp.append(adam_time)
                print("Adam training time:", adam_time)

                domain_lbfgs, boundary_lbfgs = sample_points(1000, 25)
                lbfgs = torch.optim.LBFGS(
                    model.parameters(),
                    lr=1.0,
                    max_iter=50000,
                    history_size=50,
                    tolerance_grad=1e-12,
                    tolerance_change=np.finfo(float).eps,
                    line_search_fn="strong_wolfe",
                )

                def closure():
                    lbfgs.zero_grad(set_to_none=True)
                    loss, _, _, _ = total_physics_loss(model, domain_lbfgs, boundary_lbfgs)
                    loss.backward()
                    return loss

                print("Starting L-BFGS Optimisation")
                sync_cuda()
                start_lbfgs = time.time()
                lbfgs.step(closure)
                sync_cuda()
                lbfgs_time = time.time() - start_lbfgs

                times_total_temp.append(time.time() - start_time)
                times_lbfgs_temp.append(lbfgs_time)

                eval_domain_pts, y_pred, y_true, run_accuracy, eval_time = evaluate_pinn_model(model)
                times_eval_temp.append(eval_time)
                accuracy_temp.append(run_accuracy)

            y_results[idx] = y_pred
            domain_pts[idx] = eval_domain_pts
            times_adam[idx] = float(np.mean(times_adam_temp))
            times_lbfgs[idx] = float(np.mean(times_lbfgs_temp))
            times_total[idx] = float(np.mean(times_total_temp))
            times_eval[idx] = float(np.mean(times_eval_temp))
            l2_rel[idx] = float(np.mean(accuracy_temp))
            var[idx] = float(np.var(accuracy_temp))
            arch[idx] = feature

            results = {
                "domain_pts": domain_pts,
                "y_results": y_results,
                "y_gt": y_true,
            }

            evaluation = {
                "arch": arch,
                "times_adam": times_adam,
                "times_lbfgs": times_lbfgs,
                "times_total": times_total,
                "times_eval": times_eval,
                "l2_rel": l2_rel,
                "var": var,
            }

            save_json(os.path.join(save_dir, "PINNs_results.json"), results)
            save_json(os.path.join(save_dir, "PINNs_evaluation.json"), evaluation)

        print(evaluation)
    finally:
        stop_logging()


if __name__ == "__main__":
    main()
