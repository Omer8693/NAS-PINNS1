import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
from poisson.domains import rectangular, circle, lshape, flower, annulus
from poisson.main import evaluate_and_plot
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pyswarm import pso

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
MASK_LEVELS     = [30, 50, 70, 90, 110]
BASE_NEURONS    = 110
LAYERS          = 5
EPOCHS_ADAM     = 12000
PROXY_EPOCHS    = 800
INNER_LR        = 1e-3
OUTER_LR        = 3e-4
OUTER_EVERY     = 5
LBFGS_MAX_ITER  = 1500
N_COL           = 4000
N_BC            = 400
N_COL_PROXY     = 1500
N_BC_PROXY      = 200
POP_SIZE        = 40
N_GEN           = 25

DOMAIN_MODULES = {
    "rectangular": rectangular,
    "circle": circle,
    "lshape": lshape,
    "flower": flower,
    "annulus": annulus,
}

def true_solution(x, y, domain_mod):
    return domain_mod.true_solution(x, y)

def poisson_rhs(x, y, domain_mod):
    return domain_mod.poisson_rhs(x, y)

def sample_points(domain_mod, n_col=N_COL, n_bc=N_BC, device=device):
    return domain_mod.sample_points(n_col, n_bc, device)

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
        mask[:min(lvl, self.max_out)] = 1.0
        w_s = torch.sigmoid(self.skip_weight)
        w_a = torch.sigmoid(self.active_weight)
        return w_s * skip + w_a * (active * mask)

class NAS_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [2] + [BASE_NEURONS] * (LAYERS - 1) + [1]
        self.layers = nn.ModuleList([MixedOp(dims[i], dims[i+1]) for i in range(LAYERS)])
    def forward(self, xy, mask_choices):
        x = xy
        for i, layer in enumerate(self.layers):
            x = layer(x, mask_choices[i])
        return x

def pde_loss(model, x, y, masks, domain_mod):
    xy = torch.cat([x, y], 1).requires_grad_(True)
    u = model(xy, masks)
    ux = torch.autograd.grad(u.sum(), xy, create_graph=True)[0][:,0:1]
    uy = torch.autograd.grad(u.sum(), xy, create_graph=True)[0][:,1:2]
    uxx = torch.autograd.grad(ux.sum(), xy, create_graph=True)[0][:,0:1]
    uyy = torch.autograd.grad(uy.sum(), xy, create_graph=True)[0][:,1:2]
    res = uxx + uyy - poisson_rhs(x, y, domain_mod)
    return res.pow(2).mean()

def bc_loss(model, xb, yb, masks, domain_mod):
    xy = torch.cat([xb, yb], 1)
    return F.mse_loss(model(xy, masks), true_solution(xb, yb, domain_mod))

def proxy_evaluate(mask_choices, domain_mod):
    model = NAS_PINN().to(device)
    opt = optim.Adam(model.parameters(), lr=INNER_LR)
    for _ in range(PROXY_EPOCHS):
        (xc, yc), (xb, yb) = sample_points(domain_mod, N_COL_PROXY, N_BC_PROXY, device)
        loss = pde_loss(model, xc, yc, mask_choices, domain_mod) + bc_loss(model, xb, yb, mask_choices, domain_mod)
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_loss = pde_loss(model, xc, yc, mask_choices, domain_mod) + bc_loss(model, xb, yb, mask_choices, domain_mod)
    n_param = sum(p.numel() for p in model.parameters())
    return final_loss.item(), n_param

class NSGA2Problem(ElementwiseProblem):
    def __init__(self, domain_mod):
        super().__init__(n_var=LAYERS, n_obj=2, n_constr=0,
                         xl=np.zeros(LAYERS), xu=np.full(LAYERS, len(MASK_LEVELS)-1))
        self.domain_mod = domain_mod
    def _evaluate(self, x, out, *args, **kwargs):
        masks = x.round().astype(int)
        loss, nparam = proxy_evaluate(masks, self.domain_mod)
        out["F"] = [loss, nparam]

def main():
    parser = argparse.ArgumentParser(description="NAS-PINN Poisson Solver with NSGA-II")
    parser.add_argument("--domain", type=str, default="rectangular", choices=list(DOMAIN_MODULES.keys()), help="Domain type")
    parser.add_argument("--results-dir", type=str, default="results_nsga2", help="Results directory")
    args = parser.parse_args()
    domain_mod = DOMAIN_MODULES[args.domain]
    os.makedirs(args.results_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    problem = NSGA2Problem(domain_mod)
    algorithm = NSGA2(pop_size=POP_SIZE)
    print(f"Starting NSGA-II architecture search for {args.domain} domain...")
    t_start = time.time()
    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', N_GEN),
                   seed=42,
                   verbose=True)
    search_time = time.time() - t_start
    print(f"Search time: {search_time/60:.1f} min")
    best_idx = np.argmin(res.F[:, 0])
    best_masks = res.X[best_idx].round().astype(int)
    best_proxy_loss, best_nparam = proxy_evaluate(best_masks, domain_mod)
    print(f"\nBest proxy loss: {best_proxy_loss:.4e}")
    print(f"Number of parameters: {best_nparam}")
    print(f"Mask selections: {best_masks}")
    print("\nFULL TRAINING + REFINEMENT with best architecture")
    model = NAS_PINN().to(device)
    opt_w = optim.Adam(model.parameters(), lr=INNER_LR)
    arch_p = [p for l in model.layers for p in [l.skip_weight, l.active_weight, l.mask_logits]]
    opt_a = optim.Adam(arch_p, lr=OUTER_LR)
    loss_hist = []
    for ep in range(EPOCHS_ADAM):
        (xc, yc), (xb, yb) = sample_points(domain_mod)
        opt_w.zero_grad()
        loss = pde_loss(model, xc, yc, best_masks, domain_mod) + bc_loss(model, xb, yb, best_masks, domain_mod)
        loss.backward()
        opt_w.step()
        if ep % OUTER_EVERY == 0:
            opt_a.zero_grad()
            val_loss = F.mse_loss(model(torch.cat([xc,yc],1), best_masks), true_solution(xc, yc, domain_mod))
            val_loss.backward()
            opt_a.step()
        loss_hist.append(loss.item())
        if ep % 3000 == 0:
            print(f"Adam [{ep:5d}] loss: {loss:.4e}")
    print("\nL-BFGS refinement...")
    opt_l = optim.LBFGS(model.parameters(), lr=0.8, max_iter=LBFGS_MAX_ITER,
                        line_search_fn='strong_wolfe')
    def closure():
        opt_l.zero_grad()
        (pde_loss(model, xc, yc, best_masks, domain_mod) + bc_loss(model, xb, yb, best_masks, domain_mod)).backward()
        return pde_loss(model, xc, yc, best_masks, domain_mod) + bc_loss(model, xb, yb, best_masks, domain_mod)
    opt_l.step(closure)
    print("\nPSO refinement...")
    lb = np.concatenate([p.cpu().detach().numpy().flatten() for p in model.parameters()]) - 0.5
    ub = np.concatenate([p.cpu().detach().numpy().flatten() for p in model.parameters()]) + 0.5
    def pso_obj(flat):
        offset = 0
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(flat[offset:offset+n]).view_as(p).to(device))
            offset += n
        return (pde_loss(model, xc, yc, best_masks, domain_mod) + bc_loss(model, xb, yb, best_masks, domain_mod)).item()
    xopt, fopt = pso(pso_obj, lb, ub, swarmsize=POP_SIZE, maxiter=PSO_MAX_ITER)
    print(f"PSO loss after refinement: {fopt:.4e}")
    evaluate_and_plot(model, domain_mod, save_dir=args.results_dir)

if __name__ == "__main__":
    main()
