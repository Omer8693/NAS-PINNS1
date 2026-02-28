import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pyswarm import pso

from poisson.main import DOMAIN_MODULES, evaluate_and_plot, plot_loss_history, save_run_metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


MASK_LEVELS = [30, 50, 70, 90, 110]
BASE_NEURONS = 110
LAYERS = 5
EPOCHS_ADAM = 12000
PROXY_EPOCHS = 600
INNER_LR = 1e-3
OUTER_LR = 3e-4
OUTER_EVERY = 5
LBFGS_MAX_ITER = 1500
N_COL = 4000
N_BC = 400
N_COL_PROXY = 1500
N_BC_PROXY = 200
POP_SIZE = 30
N_GEN = 20
PSO_SPAN = 0.25


class MixedOp(nn.Module):
    def __init__(self, in_f, max_out):
        super().__init__()
        self.max_out = max_out
        self.linear = nn.Linear(in_f, max_out)
        self.skip_weight = nn.Parameter(torch.tensor(0.0))
        self.active_weight = nn.Parameter(torch.tensor(0.0))
        self.mask_logits = nn.Parameter(torch.randn(len(MASK_LEVELS)))

    def forward(self, x, mask_idx):
        skip = x if x.shape[-1] == self.max_out else F.pad(x, (0, self.max_out - x.shape[-1]))
        active = torch.tanh(self.linear(x))
        mask = torch.zeros(self.max_out, device=device)
        lvl = MASK_LEVELS[int(mask_idx)]
        mask[: min(lvl, self.max_out)] = 1.0
        w_s = torch.sigmoid(self.skip_weight)
        w_a = torch.sigmoid(self.active_weight)
        return w_s * skip + w_a * (active * mask)


class NAS_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [2] + [BASE_NEURONS] * (LAYERS - 1) + [1]
        self.layers = nn.ModuleList([MixedOp(dims[i], dims[i + 1]) for i in range(LAYERS)])

    def forward(self, xy, mask_choices):
        x = xy
        for i, layer in enumerate(self.layers):
            x = layer(x, mask_choices[i])
        return x


class FixedMaskModel(nn.Module):
    def __init__(self, model, masks):
        super().__init__()
        self.model = model
        self.masks = [int(m) for m in masks]

    def forward(self, xy):
        return self.model(xy, self.masks)


def pde_loss(model, x, y, masks, domain_mod):
    xy = torch.cat([x, y], 1).requires_grad_(True)
    u = model(xy, masks)
    grad = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    ux, uy = grad[:, 0:1], grad[:, 1:2]
    uxx = torch.autograd.grad(ux.sum(), xy, create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(uy.sum(), xy, create_graph=True)[0][:, 1:2]
    residual = uxx + uyy - domain_mod.poisson_rhs(x, y)
    return residual.pow(2).mean()


def bc_loss(model, xb, yb, masks, domain_mod):
    xy = torch.cat([xb, yb], 1)
    return F.mse_loss(model(xy, masks), domain_mod.true_solution(xb, yb))


def sample_points(domain_mod, n_col=N_COL, n_bc=N_BC):
    return domain_mod.sample_points(n_col, n_bc, device=device)


def proxy_evaluate(mask_choices, domain_mod, proxy_epochs):
    model = NAS_PINN().to(device)
    opt = optim.Adam(model.parameters(), lr=INNER_LR)
    for _ in range(proxy_epochs):
        (xc, yc), (xb, yb) = sample_points(domain_mod, N_COL_PROXY, N_BC_PROXY)
        loss = pde_loss(model, xc, yc, mask_choices, domain_mod) + bc_loss(model, xb, yb, mask_choices, domain_mod)
        opt.zero_grad()
        loss.backward()
        opt.step()

    final_loss = pde_loss(model, xc, yc, mask_choices, domain_mod) + bc_loss(model, xb, yb, mask_choices, domain_mod)
    n_param = sum(p.numel() for p in model.parameters())
    return float(final_loss.item()), float(n_param)


class NSGA2Problem(ElementwiseProblem):
    def __init__(self, domain_mod, proxy_epochs):
        super().__init__(
            n_var=LAYERS,
            n_obj=2,
            n_constr=0,
            xl=np.zeros(LAYERS),
            xu=np.full(LAYERS, len(MASK_LEVELS) - 1),
        )
        self.domain_mod = domain_mod
        self.proxy_epochs = proxy_epochs

    def _evaluate(self, x, out, *args, **kwargs):
        masks = x.round().astype(int)
        loss, nparam = proxy_evaluate(masks, self.domain_mod, proxy_epochs=self.proxy_epochs)
        out["F"] = [loss, nparam]


def run_architecture_search(domain_mod, args):
    print(f"Starting NSGA-II architecture search: domain={args.domain}")
    problem = NSGA2Problem(domain_mod, proxy_epochs=args.proxy_epochs)
    algorithm = NSGA2(pop_size=args.pop_size)
    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", args.n_gen),
        seed=args.seed,
        verbose=True,
    )
    best_idx = int(np.argmin(res.F[:, 0]))
    best_masks = res.X[best_idx].round().astype(int)
    return best_masks


def train_with_best_masks(domain_mod, best_masks, args):
    model = NAS_PINN().to(device)
    opt_w = optim.Adam(model.parameters(), lr=INNER_LR)
    arch_p = [p for l in model.layers for p in [l.skip_weight, l.active_weight, l.mask_logits]]
    opt_a = optim.Adam(arch_p, lr=OUTER_LR)
    loss_hist = []

    for ep in range(args.epochs):
        (xc, yc), (xb, yb) = sample_points(domain_mod)
        opt_w.zero_grad()
        loss = pde_loss(model, xc, yc, best_masks, domain_mod) + bc_loss(model, xb, yb, best_masks, domain_mod)
        loss.backward()
        opt_w.step()

        if ep % OUTER_EVERY == 0:
            opt_a.zero_grad()
            val_loss = F.mse_loss(model(torch.cat([xc, yc], 1), best_masks), domain_mod.true_solution(xc, yc))
            val_loss.backward()
            opt_a.step()

        loss_hist.append(float(loss.item()))
        if ep % 2000 == 0 or ep == args.epochs - 1:
            print(f"Adam [{ep:5d}] loss: {loss:.4e}")

    if not args.skip_lbfgs:
        print("\nL-BFGS refinement...")
        opt_l = optim.LBFGS(
            model.parameters(),
            lr=0.8,
            max_iter=LBFGS_MAX_ITER,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt_l.zero_grad()
            total = pde_loss(model, xc, yc, best_masks, domain_mod) + bc_loss(model, xb, yb, best_masks, domain_mod)
            total.backward()
            return total

        opt_l.step(closure)

    if not args.skip_pso:
        print("\nPSO refinement...")
        lower = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()]) - PSO_SPAN
        upper = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()]) + PSO_SPAN

        def pso_objective(flat):
            offset = 0
            with torch.no_grad():
                for param in model.parameters():
                    n_param = param.numel()
                    chunk = torch.from_numpy(flat[offset : offset + n_param]).view_as(param).to(device)
                    param.copy_(chunk)
                    offset += n_param
            total = pde_loss(model, xc, yc, best_masks, domain_mod) + bc_loss(model, xb, yb, best_masks, domain_mod)
            return float(total.item())

        _, best_loss = pso(
            pso_objective,
            lower,
            upper,
            swarmsize=args.pso_swarm,
            maxiter=args.pso_iters,
        )
        print(f"PSO best objective: {best_loss:.4e}")

    return model, loss_hist


def run_single(args, domain_name, save_dir):
    start = time.perf_counter()
    os.makedirs(save_dir, exist_ok=True)
    domain_mod = DOMAIN_MODULES[domain_name]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    best_masks = run_architecture_search(domain_mod, args)
    print(f"Best mask selections: {best_masks}")
    model, loss_hist = train_with_best_masks(domain_mod, best_masks, args)

    plot_loss_history(
        loss_hist,
        save_path=os.path.join(save_dir, "poisson_nsga2_loss_curve.png"),
        title=f"Poisson NSGA-II Loss ({domain_name})",
    )
    eval_model = FixedMaskModel(model, best_masks).to(device)
    rel_l2 = evaluate_and_plot(eval_model, domain_mod, save_dir=save_dir, method_label="NSGA-II")

    run_time = time.perf_counter() - start
    save_run_metrics(save_dir, "nsga2", domain_name, args.seed, rel_l2, run_time)

    with open(os.path.join(save_dir, "best_masks.txt"), "w", encoding="utf-8") as f:
        f.write(",".join(str(int(x)) for x in best_masks.tolist()) + "\n")

    print(f"Run time: {run_time:.2f} s")
    return rel_l2, run_time


def run_multi_domain(args):
    domains = [d.strip() for d in args.domain_list.split(",") if d.strip()]
    summary = []

    for idx, domain_name in enumerate(domains):
        if domain_name not in DOMAIN_MODULES:
            raise ValueError(f"Unknown domain in --domain-list: {domain_name}")
        args.seed = args.base_seed + idx
        out_dir = os.path.join(args.save_dir, f"domain_{domain_name}")
        rel_l2, run_time = run_single(args, domain_name=domain_name, save_dir=out_dir)
        summary.append((domain_name, rel_l2, run_time))

    out_csv = os.path.join(args.save_dir, "domain_comparison.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("domain,rel_l2,run_time_seconds\n")
        for domain_name, rel_l2, run_time in summary:
            f.write(f"{domain_name},{rel_l2:.8e},{run_time:.6f}\n")
    print(f"Saved summary: {out_csv}")

    domains_plot = [row[0] for row in summary]
    errs = [row[1] for row in summary]
    plt.figure(figsize=(7, 4))
    plt.plot(domains_plot, errs, marker="o", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Domain")
    plt.ylabel("Relative L2 Error")
    plt.title("Poisson NSGA-II: Domain Comparison")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "domain_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Poisson NAS-PINN with NSGA-II")
    parser.add_argument(
        "--domain",
        type=str,
        default="rectangular",
        choices=list(DOMAIN_MODULES.keys()),
        help="single domain type",
    )
    parser.add_argument("--multi-domain", action="store_true", help="run multiple domains from --domain-list")
    parser.add_argument(
        "--domain-list",
        type=str,
        default="rectangular,circle,lshape,flower,annulus",
        help="comma-separated domains for --multi-domain",
    )
    parser.add_argument("--save-dir", type=str, default="results/poisson/nsga2", help="output directory")
    parser.add_argument("--epochs", type=int, default=EPOCHS_ADAM, help="final Adam epochs")
    parser.add_argument("--proxy-epochs", type=int, default=PROXY_EPOCHS, help="proxy epochs per NSGA candidate")
    parser.add_argument("--pop-size", type=int, default=POP_SIZE, help="NSGA-II population size")
    parser.add_argument("--n-gen", type=int, default=N_GEN, help="NSGA-II generation count")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--skip-lbfgs", action="store_true", help="skip L-BFGS refinement")
    parser.add_argument("--skip-pso", action="store_true", help="skip PSO refinement")
    parser.add_argument("--pso-iters", type=int, default=8, help="PSO max iterations")
    parser.add_argument("--pso-swarm", type=int, default=16, help="PSO swarm size")
    return parser.parse_args()


def main():
    args = parse_args()
    args.base_seed = args.seed
    os.makedirs(args.save_dir, exist_ok=True)

    if args.multi_domain:
        run_multi_domain(args)
    else:
        run_single(args, domain_name=args.domain, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
