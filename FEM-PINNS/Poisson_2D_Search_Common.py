import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Poisson_2D_Common import (
    device,
    evaluate_pinn_model,
    lhs,
    sample_points,
    set_seed,
    supervised_outer_loss,
    sync_cuda,
    total_physics_loss,
)


MASK_LEVELS = [30, 50, 70, 90, 110]
BASE_NEURONS = 110
NUM_HIDDEN_LAYERS = 4

INNER_LR = 1e-3
OUTER_LR = 3e-4
OUTER_EVERY = 5

PROXY_EPOCHS = 300
ADAM_EPOCHS = 12000

LBFGS_MAX_ITER = 1500
LBFGS_HISTORY_SIZE = 50

N_COL = 2000
N_BC = 250
N_VAL = 512

N_COL_PROXY = 1000
N_BC_PROXY = 128
N_VAL_PROXY = 256

N_COL_LBFGS = 1000
N_BC_LBFGS = 250

SEED = 17


class MixedHiddenOp(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.alpha_skip = nn.Parameter(torch.tensor(0.0))
        self.alpha_active = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, mask_idx):
        skip = x
        if x.shape[-1] < self.out_features:
            skip = F.pad(skip, (0, self.out_features - x.shape[-1]))
        elif x.shape[-1] > self.out_features:
            skip = skip[:, : self.out_features]

        active = torch.tanh(self.linear(x))

        mask = torch.zeros(self.out_features, device=x.device, dtype=x.dtype)
        width = min(MASK_LEVELS[int(mask_idx)], self.out_features)
        mask[:width] = 1.0

        w_skip = torch.sigmoid(self.alpha_skip)
        w_active = torch.sigmoid(self.alpha_active)
        return w_skip * skip + w_active * (active * mask)


class SearchPINN(nn.Module):
    def __init__(self, base_neurons=BASE_NEURONS, num_hidden_layers=NUM_HIDDEN_LAYERS):
        super().__init__()
        self.hidden_layers = nn.ModuleList()

        in_dim = 2
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(MixedHiddenOp(in_dim, base_neurons))
            in_dim = base_neurons

        self.out = nn.Linear(base_neurons, 1)

    def forward(self, xy, mask_choices):
        x = xy
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x, mask_choices[i])
        return self.out(x)


def split_params(model):
    weight_params = []
    arch_params = []

    for layer in model.hidden_layers:
        weight_params.extend(list(layer.linear.parameters()))
        arch_params.extend([layer.alpha_skip, layer.alpha_active])

    weight_params.extend(list(model.out.parameters()))
    return weight_params, arch_params


def architecture_complexity(mask_choices):
    widths = [min(MASK_LEVELS[int(m)], BASE_NEURONS) for m in mask_choices]
    dims = [2] + widths + [1]
    total = 0
    for i in range(len(dims) - 1):
        total += dims[i] * dims[i + 1] + dims[i + 1]
    return float(total)


def proxy_seed(mask_choices):
    return SEED + int(sum((i + 1) * (int(m) + 1) for i, m in enumerate(mask_choices)))


def proxy_evaluate(mask_choices, *, model_factory=SearchPINN):
    set_seed(proxy_seed(mask_choices))

    model = model_factory().to(device)
    weight_params, arch_params = split_params(model)

    opt_w = torch.optim.Adam(weight_params, lr=INNER_LR)
    opt_a = torch.optim.Adam(arch_params, lr=OUTER_LR)

    for epoch in range(PROXY_EPOCHS):
        domain_points, boundary_points, val_points = sample_points(
            N_COL_PROXY,
            N_BC_PROXY,
            N_VAL_PROXY,
        )

        opt_w.zero_grad(set_to_none=True)
        inner_loss, _, _, _ = total_physics_loss(
            model,
            domain_points,
            boundary_points,
            masks=mask_choices,
        )
        inner_loss.backward()
        opt_w.step()

        if epoch % OUTER_EVERY == 0:
            opt_a.zero_grad(set_to_none=True)
            outer_loss = supervised_outer_loss(model, val_points, masks=mask_choices)
            outer_loss.backward()
            opt_a.step()

    domain_points, boundary_points, _ = sample_points(
        N_COL_PROXY,
        N_BC_PROXY,
        N_VAL_PROXY,
    )
    final_loss, _, _, _ = total_physics_loss(
        model,
        domain_points,
        boundary_points,
        masks=mask_choices,
    )
    return float(final_loss.detach().cpu()), architecture_complexity(mask_choices)


def train_final_model(best_masks, *, model_factory=SearchPINN):
    set_seed(SEED)

    model = model_factory().to(device)
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
            masks=best_masks,
        )
        inner_loss.backward()
        optimizer_w.step()

        if epoch % OUTER_EVERY == 0:
            optimizer_a.zero_grad(set_to_none=True)
            outer_loss = supervised_outer_loss(model, val_points, masks=best_masks)
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

    domain_lbfgs = torch.tensor(lhs(2, N_COL_LBFGS), device=device)
    boundary_lbfgs = torch.tensor(lhs(2, N_BC_LBFGS), device=device)

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
        loss, _, _, _ = total_physics_loss(
            model,
            domain_lbfgs,
            boundary_lbfgs,
            masks=best_masks,
        )
        loss.backward()
        return loss

    print("\nStarting L-BFGS refinement...")
    sync_cuda()
    start_lbfgs = time.time()
    optimizer_lbfgs.step(closure)
    sync_cuda()
    lbfgs_time = time.time() - start_lbfgs

    return model, loss_history, adam_time, lbfgs_time


def architecture_summary(model, best_masks):
    rows = []
    for idx, (layer, mask_idx) in enumerate(zip(model.hidden_layers, best_masks)):
        skip_prob = float(torch.sigmoid(layer.alpha_skip).item())
        active_prob = float(torch.sigmoid(layer.alpha_active).item())
        rows.append(
            {
                "layer": idx + 1,
                "neurons": int(MASK_LEVELS[int(mask_idx)]),
                "skip_prob": skip_prob,
                "active_prob": active_prob,
                "mode": "skip" if skip_prob > active_prob else "active",
            }
        )
    return rows


def print_architecture(model, best_masks, title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for row in architecture_summary(model, best_masks):
        print(
            f"Layer {row['layer']:2d} | "
            f"neurons={row['neurons']:3d} | "
            f"mode={row['mode']:6s} | "
            f"skip={row['skip_prob']:.4f} | "
            f"active={row['active_prob']:.4f}"
        )
    print("=" * 60 + "\n")


def evaluate_search_model(model, best_masks):
    return evaluate_pinn_model(model, masks=best_masks)
