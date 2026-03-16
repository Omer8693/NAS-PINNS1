import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Poisson_2D_Common import (
    device,
    evaluate_pinn_model,
    sample_points,
    save_json,
    set_seed,
    start_run_logging,
    supervised_outer_loss,
    sync_cuda,
    total_physics_loss,
)


MASK_LEVELS = [30, 50, 70, 90, 110]
BASE_NEURONS = 110
NUM_LAYERS = 5

INNER_LR = 1e-3
OUTER_LR = 3e-4
OUTER_EVERY = 5

ADAM_EPOCHS = 12000
LBFGS_MAX_ITER = 1500
LBFGS_HISTORY_SIZE = 50

N_COL = 2000
N_BC = 250
N_VAL = 512

N_COL_LBFGS = 1000
N_BC_LBFGS = 250

SEED = 17


class MixedOp(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

        self.alpha_skip = nn.Parameter(torch.tensor(0.0))
        self.alpha_active = nn.Parameter(torch.tensor(0.0))
        self.alpha_mask = nn.Parameter(0.1 * torch.randn(len(MASK_LEVELS)))

    def forward(self, x):
        skip = x
        if x.shape[-1] < self.out_features:
            skip = F.pad(skip, (0, self.out_features - x.shape[-1]))
        elif x.shape[-1] > self.out_features:
            skip = skip[:, : self.out_features]

        active = torch.tanh(self.linear(x))

        w_skip = torch.sigmoid(self.alpha_skip)
        w_active = torch.sigmoid(self.alpha_active)
        mask_probs = F.softmax(self.alpha_mask, dim=0)

        masked_active = torch.zeros_like(active)
        for i, level in enumerate(MASK_LEVELS):
            mask = torch.zeros(self.out_features, device=x.device, dtype=x.dtype)
            mask[: min(level, self.out_features)] = 1.0
            masked_active = masked_active + mask_probs[i] * (active * mask)

        return w_skip * skip + w_active * masked_active


class NASPINN(nn.Module):
    def __init__(self, num_layers=NUM_LAYERS, base_neurons=BASE_NEURONS):
        super().__init__()
        dims = [2] + [base_neurons] * (num_layers - 1) + [1]
        self.layers = nn.ModuleList(
            [MixedOp(dims[i], dims[i + 1]) for i in range(num_layers)]
        )

    def forward(self, xy):
        out = xy
        for layer in self.layers:
            out = layer(out)
        return out


def split_params(model):
    weight_params = []
    arch_params = []

    for layer in model.layers:
        weight_params.extend(list(layer.linear.parameters()))
        arch_params.extend([layer.alpha_skip, layer.alpha_active, layer.alpha_mask])

    return weight_params, arch_params


def discrete_architecture(model):
    summary = []
    for idx, layer in enumerate(model.layers):
        skip_prob = torch.sigmoid(layer.alpha_skip).item()
        active_prob = torch.sigmoid(layer.alpha_active).item()
        mask_probs = F.softmax(layer.alpha_mask, dim=0).detach().cpu().numpy()
        best_mask_idx = int(np.argmax(mask_probs))
        neurons = int(min(MASK_LEVELS[best_mask_idx], layer.out_features))
        summary.append(
            {
                "layer": idx + 1,
                "connection": "skip" if skip_prob > active_prob else "active",
                "neurons": neurons,
                "skip_prob": skip_prob,
                "active_prob": active_prob,
            }
        )
    return summary


def print_architecture(model):
    print("\n" + "=" * 60)
    print("DISCRETE NAS-PINN ARCHITECTURE")
    print("=" * 60)
    for row in discrete_architecture(model):
        print(
            f"Layer {row['layer']:2d} | "
            f"{row['connection']:6s} | "
            f"neurons={row['neurons']:3d} | "
            f"skip={row['skip_prob']:.4f} | "
            f"active={row['active_prob']:.4f}"
        )
    print("=" * 60 + "\n")


def train_naspinn():
    model = NASPINN().to(device)
    weight_params, arch_params = split_params(model)

    optimizer_w = torch.optim.Adam(weight_params, lr=INNER_LR)
    optimizer_a = torch.optim.Adam(arch_params, lr=OUTER_LR)
    loss_history = []

    sync_cuda()
    start_adam = time.time()

    for epoch in range(ADAM_EPOCHS):
        domain_points, boundary_points, val_points = sample_points(N_COL, N_BC, N_VAL)

        optimizer_w.zero_grad(set_to_none=True)
        inner_loss, loss_pde, loss_bc_d, loss_bc_n = total_physics_loss(
            model,
            domain_points,
            boundary_points,
        )
        inner_loss.backward()
        optimizer_w.step()

        if epoch % OUTER_EVERY == 0:
            optimizer_a.zero_grad(set_to_none=True)
            outer_loss = supervised_outer_loss(model, val_points)
            outer_loss.backward()
            optimizer_a.step()
        else:
            outer_loss = torch.tensor(0.0, device=device)

        loss_history.append(float(inner_loss.detach().cpu()))

        if epoch % 1000 == 0 or epoch == ADAM_EPOCHS - 1:
            print(
                f"[{epoch:5d}] "
                f"PDE={loss_pde.item():.4e}  "
                f"BC_D={loss_bc_d.item():.4e}  "
                f"BC_N={loss_bc_n.item():.4e}  "
                f"OUTER={outer_loss.item():.4e}"
            )

    sync_cuda()
    adam_time = time.time() - start_adam

    for param in arch_params:
        param.requires_grad_(False)

    domain_lbfgs, boundary_lbfgs = sample_points(N_COL_LBFGS, N_BC_LBFGS)
    optimizer_lbfgs = torch.optim.LBFGS(
        weight_params,
        lr=0.8,
        max_iter=LBFGS_MAX_ITER,
        history_size=LBFGS_HISTORY_SIZE,
        tolerance_grad=1e-12,
        tolerance_change=np.finfo(float).eps,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer_lbfgs.zero_grad(set_to_none=True)
        loss, _, _, _ = total_physics_loss(model, domain_lbfgs, boundary_lbfgs)
        loss.backward()
        return loss

    print("\nStarting L-BFGS refinement...")
    sync_cuda()
    start_lbfgs = time.time()
    optimizer_lbfgs.step(closure)
    sync_cuda()
    lbfgs_time = time.time() - start_lbfgs

    return model, loss_history, adam_time, lbfgs_time


def main():
    save_dir = "./2D-Poisson-NASPINN"
    _, stop_logging = start_run_logging(save_dir)
    try:
        print(device)
        set_seed(SEED)

        total_start = time.time()
        model, _, adam_time, lbfgs_time = train_naspinn()
        print_architecture(model)

        domain_pts, y_pred, y_true, rel_l2, eval_time = evaluate_pinn_model(model)
        total_time = time.time() - total_start

        results = {
            "domain_pts": domain_pts,
            "y_results": y_pred,
            "y_gt": y_true,
        }

        evaluation = {
            "times_adam": adam_time,
            "times_lbfgs": lbfgs_time,
            "times_total": total_time,
            "times_eval": eval_time,
            "l2_rel": rel_l2,
            "arch": discrete_architecture(model),
            "mask_levels": MASK_LEVELS,
            "num_layers": NUM_LAYERS,
            "base_neurons": BASE_NEURONS,
        }

        save_json(os.path.join(save_dir, "NASPINN_results.json"), results)
        save_json(os.path.join(save_dir, "NASPINN_evaluation.json"), evaluation, indent=4)

        print(f"Adam time   : {adam_time:.4f} s")
        print(f"L-BFGS time : {lbfgs_time:.4f} s")
        print(f"Eval time   : {eval_time:.4f} s")
        print(f"Total time  : {total_time:.4f} s")
        print(f"Relative L2 : {rel_l2:.8e}")
    finally:
        stop_logging()


if __name__ == "__main__":
    main()
