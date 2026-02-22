import argparse
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.misc import has_feasible
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from tqdm import tqdm
from .plots import (
    plot_burgers_full_exact_vs_pred,
    plot_burgers_heatmap,
    plot_burgers_time_slices,
    plot_burgers_time_slices_with_exact,
    should_use_exact_plots,
)


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
interactive_plots = os.environ.get("DISPLAY") is not None

nu = 0.01 / np.pi
N_col = 10000
N_ic = 200
N_bc = 200
x_min, x_max = -1.0, 1.0
t_min, t_max = 0.0, 1.0


def binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(n_tournaments, np.nan)

    random_state = getattr(algorithm, "random_state", None)

    for i in range(n_tournaments):
        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
        rank_a, cd_a = pop[a].get("rank", "crowding")
        rank_b, cd_b = pop[b].get("rank", "crowding")

        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(
                a,
                a_cv,
                b,
                b_cv,
                method="smaller_is_better",
                return_random_if_equal=True,
                random_state=random_state,
            )
        else:
            if tournament_type == "comp_by_dom_and_crowding":
                rel = Dominator.get_relation(a_f, b_f)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == "comp_by_rank_and_crowding":
                S[i] = compare(a, rank_a, b, rank_b, method="smaller_is_better")

            else:
                raise Exception("Unknown tournament type.")

            if np.isnan(S[i]):
                S[i] = compare(
                    a,
                    cd_a,
                    b,
                    cd_b,
                    method="larger_is_better",
                    return_random_if_equal=True,
                    random_state=random_state,
                )

    return S[:, None].astype(int, copy=False)


class RankAndCrowdingSurvival(RankAndCrowding):
    def __init__(self, nds=None, crowding_func="cd"):
        super().__init__(nds, crowding_func)


class NSGA2(GeneticAlgorithm):
    def __init__(
        self,
        pop_size=100,
        sampling=FloatRandomSampling(),
        selection=TournamentSelection(func_comp=binary_tournament),
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        survival=RankAndCrowding(),
        output=MultiObjectiveOutput(),
        **kwargs,
    ):
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            advance_after_initial_infill=True,
            **kwargs,
        )

        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = "comp_by_dom_and_crowding"

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]


parse_doc_string(NSGA2.__init__)


def finalize_plot(save_path):
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {save_path}")
    if interactive_plots:
        plt.show()
    else:
        plt.close()


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class NASPINNFixed(nn.Module):
    def __init__(self, layer_sizes, activation="tanh", use_residual=False):
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        self.activation_name = activation
        self.use_residual = bool(use_residual)
        self.layers = nn.ModuleList(
            [nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)]
        )
        self.activation = SinActivation() if activation == "sin" else nn.Tanh()

    @property
    def model_config(self):
        return {
            "layer_sizes": self.layer_sizes,
            "activation": self.activation_name,
            "use_residual": self.use_residual,
        }

    def forward(self, xt):
        h = xt
        for layer in self.layers[:-1]:
            out = self.activation(layer(h))
            if self.use_residual and out.shape[-1] == h.shape[-1]:
                out = out + h
            h = out
        return self.layers[-1](h)


def sample_points():
    x_c = torch.rand(N_col, 1, device=device) * (x_max - x_min) + x_min
    t_c = torch.rand(N_col, 1, device=device) * (t_max - t_min) + t_min
    x_ic = torch.rand(N_ic, 1, device=device) * (x_max - x_min) + x_min
    t_ic = torch.zeros_like(x_ic)
    t_bc = torch.rand(N_bc, 1, device=device) * (t_max - t_min) + t_min
    x_left = torch.full((N_bc, 1), x_min, device=device)
    x_right = torch.full((N_bc, 1), x_max, device=device)
    return (x_c, t_c), (x_ic, t_ic), (x_left, t_bc), (x_right, t_bc)


def sample_points_paper(train_nx=250, train_nt=21):
    x_vals = torch.linspace(x_min, x_max, train_nx, device=device)
    t_vals = torch.linspace(t_min, t_max, train_nt, device=device)
    X, T = torch.meshgrid(x_vals, t_vals, indexing="ij")

    x_c = X.reshape(-1, 1)
    t_c = T.reshape(-1, 1)
    x_ic = x_vals.unsqueeze(1)
    t_ic = torch.zeros_like(x_ic)
    t_bc = t_vals.unsqueeze(1)
    x_left = torch.full_like(t_bc, x_min)
    x_right = torch.full_like(t_bc, x_max)
    return (x_c, t_c), (x_ic, t_ic), (x_left, t_bc), (x_right, t_bc)


def pde_residual(model, x, t, nu_coef):
    xt = torch.cat([x, t], dim=1).requires_grad_(True)
    u = model(xt)
    grads = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
    u_t = grads[:, 1:2]
    u_x = grads[:, 0:1]
    u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
    return torch.mean((u_t + u * u_x - nu_coef * u_xx) ** 2)


def ic_loss(model, x, t):
    xt = torch.cat([x, t], 1)
    u_pred = model(xt)
    u_true = -torch.sin(np.pi * x)
    return torch.mean((u_pred - u_true) ** 2)


def bc_loss(model, x_l, t_l, x_r, t_r):
    u_l = model(torch.cat([x_l, t_l], 1))
    u_r = model(torch.cat([x_r, t_r], 1))
    return torch.mean(u_l ** 2) + torch.mean(u_r ** 2)


def total_loss(model, points, nu_coef, lambda_pde=1.0, lambda_ic=100.0, lambda_bc=100.0):
    x_c, t_c, x_ic, t_ic, x_l, t_l, x_r, t_r = points
    l_pde = pde_residual(model, x_c, t_c, nu_coef)
    l_ic = ic_loss(model, x_ic, t_ic)
    l_bc = bc_loss(model, x_l, t_l, x_r, t_r)
    total = lambda_pde * l_pde + lambda_ic * l_ic + lambda_bc * l_bc
    return total, l_pde, l_ic, l_bc


def predict_on_grid(model, x_values, t_values):
    Xg, Tg = torch.meshgrid(x_values, t_values, indexing="ij")
    XT = torch.cat([Xg.reshape(-1, 1), Tg.reshape(-1, 1)], dim=1)
    with torch.no_grad():
        u_pred = model(XT).cpu().numpy().reshape(len(x_values), len(t_values))
    return u_pred


def reference_solution_fd(nu_coef, x_vals_np, t_vals_np):
    nx = len(x_vals_np)
    dx = x_vals_np[1] - x_vals_np[0]
    u0 = -np.sin(np.pi * x_vals_np)
    u0[0] = 0.0
    u0[-1] = 0.0

    def rhs(_t, u_inner):
        u = np.zeros(nx, dtype=np.float64)
        u[1:-1] = u_inner
        ux = (u[2:] - u[:-2]) / (2.0 * dx)
        uxx = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx ** 2)
        return -u[1:-1] * ux + nu_coef * uxx

    attempts = [
        ("BDF", 1e-5, 1e-7),
        ("Radau", 1e-5, 1e-7),
        ("LSODA", 1e-4, 1e-6),
    ]

    for method, rtol, atol in attempts:
        sol = solve_ivp(
            rhs,
            t_span=(float(t_vals_np[0]), float(t_vals_np[-1])),
            y0=u0[1:-1],
            t_eval=t_vals_np,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        if sol.success:
            U = np.zeros((nx, len(t_vals_np)), dtype=np.float64)
            U[1:-1, :] = sol.y
            return U

    U = np.zeros((nx, len(t_vals_np)), dtype=np.float64)
    U[:, 0] = u0
    eps = 1e-12

    for n in range(len(t_vals_np) - 1):
        dt_total = float(t_vals_np[n + 1] - t_vals_np[n])
        u_curr = U[:, n].copy()
        if not np.all(np.isfinite(u_curr)):
            u_curr = np.nan_to_num(u_curr, nan=0.0, posinf=0.0, neginf=0.0)
        u_curr = np.clip(u_curr, -5.0, 5.0)

        max_u = max(np.max(np.abs(u_curr)), eps)
        dt_adv = 0.4 * dx / max_u
        dt_diff = 0.4 * dx * dx / max(nu_coef, eps)
        dt_stable = max(min(dt_adv, dt_diff), 1e-6)
        if not np.isfinite(dt_stable):
            dt_stable = 1e-4
        n_sub = max(int(np.ceil(dt_total / dt_stable)), 1)
        dt = dt_total / n_sub

        for _ in range(n_sub):
            un = u_curr.copy()
            ux = (un[2:] - un[:-2]) / (2.0 * dx)
            uxx = (un[2:] - 2.0 * un[1:-1] + un[:-2]) / (dx ** 2)
            rhs_inner = -un[1:-1] * ux + nu_coef * uxx
            rhs_inner = np.nan_to_num(rhs_inner, nan=0.0, posinf=0.0, neginf=0.0)
            u_curr[1:-1] = un[1:-1] + dt * rhs_inner
            u_curr[1:-1] = np.clip(u_curr[1:-1], -5.0, 5.0)
            u_curr[0] = 0.0
            u_curr[-1] = 0.0

        if not np.all(np.isfinite(u_curr)):
            u_curr = np.nan_to_num(u_curr, nan=0.0, posinf=0.0, neginf=0.0)
        U[:, n + 1] = u_curr

    return U


def get_reference_solution(nu_coef, x_test, t_test):
    x_np = x_test.detach().cpu().numpy()
    t_np = t_test.detach().cpu().numpy()

    if should_use_exact_plots(nu_coef) and os.path.exists("burgers_shock.mat"):
        data = loadmat("burgers_shock.mat")
        x_exact = data["x"].squeeze()
        t_exact = data["t"].squeeze()
        u_exact = np.real(data["usol"])
        if len(x_exact) == len(x_np) and len(t_exact) == len(t_np):
            return u_exact

    return reference_solution_fd(nu_coef, x_np, t_np)


def decode_solution(x, max_hidden_layers):
    n_layers = int(np.clip(round(x[0]), 3, max_hidden_layers))
    neurons = [int(np.clip(round(v), 32, 192)) for v in x[1:1 + n_layers]]
    lr = float(np.clip(x[1 + max_hidden_layers], 1e-4, 5e-3))
    act_idx = int(np.clip(round(x[2 + max_hidden_layers]), 0, 1))
    res_idx = int(np.clip(round(x[3 + max_hidden_layers]), 0, 1))
    lambda_ic = float(np.clip(x[4 + max_hidden_layers], 40.0, 180.0))
    lambda_bc = float(np.clip(x[5 + max_hidden_layers], 40.0, 180.0))

    model_cfg = {
        "layer_sizes": [2] + neurons + [1],
        "activation": "sin" if act_idx == 1 else "tanh",
        "use_residual": bool(res_idx),
    }
    train_cfg = {
        "lr": lr,
        "lambda_pde": 1.0,
        "lambda_ic": lambda_ic,
        "lambda_bc": lambda_bc,
    }
    return model_cfg, train_cfg


class BurgersNSGA2Problem(ElementwiseProblem):
    def __init__(self, nu_coef, fixed_points, eval_epochs=400, max_hidden_layers=6, test_nt=21, test_nx=300):
        self.nu_coef = nu_coef
        self.fixed_points = fixed_points
        self.eval_epochs = eval_epochs
        self.max_hidden_layers = max_hidden_layers

        x_test = np.linspace(x_min, x_max, test_nx)
        t_test = np.linspace(t_min, t_max, test_nt)
        self.x_test_torch = torch.tensor(x_test, dtype=torch.float32, device=device)
        self.t_test_torch = torch.tensor(t_test, dtype=torch.float32, device=device)
        self.exact_u = reference_solution_fd(nu_coef, x_test, t_test)

        n_var = 1 + self.max_hidden_layers + 5
        xl = np.array([3] + [32] * self.max_hidden_layers + [1e-4, 0, 0, 40.0, 40.0], dtype=float)
        xu = np.array([self.max_hidden_layers] + [192] * self.max_hidden_layers + [5e-3, 1, 1, 180.0, 180.0], dtype=float)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        model_cfg, train_cfg = decode_solution(x, self.max_hidden_layers)
        model = NASPINNFixed(**model_cfg).to(device)
        try:
            (x_c, t_c), (x_ic, t_ic), (x_l, t_l), (x_r, t_r) = self.fixed_points
            points = (x_c, t_c, x_ic, t_ic, x_l, t_l, x_r, t_r)

            optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])
            for _ in range(self.eval_epochs):
                optimizer.zero_grad()
                loss, _, _, _ = total_loss(
                    model,
                    points,
                    self.nu_coef,
                    lambda_pde=train_cfg["lambda_pde"],
                    lambda_ic=train_cfg["lambda_ic"],
                    lambda_bc=train_cfg["lambda_bc"],
                )
                loss.backward()
                optimizer.step()

            pred_u = predict_on_grid(model, self.x_test_torch, self.t_test_torch)
            rel_l2 = np.linalg.norm(pred_u - self.exact_u) / (np.linalg.norm(self.exact_u) + 1e-12)
            params = sum(p.numel() for p in model.parameters())
            if np.isnan(rel_l2) or np.isinf(rel_l2):
                rel_l2 = 1.0
            out["F"] = [float(rel_l2), float(params)]
        except Exception as exc:
            print(f"  ⚠️ NSGA candidate failed: {str(exc)[:100]}")
            out["F"] = [1.0, 1e9]


class NSGAIICallback:
    def __init__(self, n_gen):
        self.n_gen = n_gen
        self.pbar = tqdm(total=n_gen, desc="NSGA-II generation", unit="gen")

    def __call__(self, algorithm):
        self.pbar.update(1)
        if hasattr(algorithm, "pop") and algorithm.pop is not None:
            F = algorithm.pop.get("F")
            best_idx = np.argmin(F[:, 0])
            print(f"  Gen {algorithm.n_gen:2d}/{self.n_gen} -> best relL2={F[best_idx, 0]:.2e}, params={int(F[best_idx,1]):,}")

    def close(self):
        self.pbar.close()


def run_nsga2_search(nu_coef, fixed_points, pop=10, ngen=5, eval_epochs=400, max_hidden_layers=6, test_nt=21, test_nx=300, seed=42):
    print("\n" + "=" * 90)
    print("NSGA-II NAS-PINN Search")
    print("=" * 90)
    print(f"  Viscosity: nu={nu_coef:.6f} | pop={pop} | gen={ngen} | eval_epochs={eval_epochs}")
    print("  Objective: min (relative L2, params)")
    print("=" * 90 + "\n")

    problem = BurgersNSGA2Problem(nu_coef, fixed_points, eval_epochs, max_hidden_layers, test_nt, test_nx)
    algorithm = NSGA2(
        pop_size=pop,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(eta=25),
        eliminate_duplicates=True,
    )
    termination = get_termination("n_gen", ngen)

    start = time.time()
    callback = NSGAIICallback(ngen)
    result = minimize(problem, algorithm, termination, seed=seed, verbose=False, callback=callback)
    callback.close()

    best_idx = np.argmin(result.F[:, 0])
    model_cfg, train_cfg = decode_solution(result.X[best_idx], max_hidden_layers)
    return {
        "model_config": model_cfg,
        "train_cfg": train_cfg,
        "rel_l2": float(result.F[best_idx, 0]),
        "params": int(result.F[best_idx, 1]),
        "search_time": time.time() - start,
    }


def save_checkpoint(path, model, optimizer, epoch, points, train_cfg, search_result=None):
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "model_config": model.model_config,
        "train_cfg": train_cfg,
        "search_result": search_result,
        "points": {
            "x_c": points[0].detach().cpu(),
            "t_c": points[1].detach().cpu(),
            "x_ic": points[2].detach().cpu(),
            "t_ic": points[3].detach().cpu(),
            "x_l": points[4].detach().cpu(),
            "t_l": points[5].detach().cpu(),
            "x_r": points[6].detach().cpu(),
            "t_r": points[7].detach().cpu(),
        },
    }
    torch.save(payload, path)


def read_checkpoint_metadata(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("model_config"), ckpt.get("train_cfg"), ckpt.get("search_result")


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    p = ckpt.get("points")
    if p is not None:
        points = (
            p["x_c"].to(device), p["t_c"].to(device), p["x_ic"].to(device), p["t_ic"].to(device),
            p["x_l"].to(device), p["t_l"].to(device), p["x_r"].to(device), p["t_r"].to(device),
        )
    else:
        (x_c, t_c), (x_ic, t_ic), (x_l, t_l), (x_r, t_r) = sample_points()
        points = (x_c, t_c, x_ic, t_ic, x_l, t_l, x_r, t_r)

    return ckpt.get("epoch", 0), points, ckpt.get("train_cfg"), ckpt.get("search_result")


def train_with_resume(model, args, train_cfg, fixed_points=None, search_result=None):
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])

    if args.resume and os.path.exists(args.checkpoint):
        start_epoch, points, ckpt_train_cfg, _ = load_checkpoint(args.checkpoint, model, optimizer)
        if ckpt_train_cfg is not None:
            train_cfg = ckpt_train_cfg
    else:
        start_epoch = 0
        if fixed_points is None:
            fixed_points = sample_points()
        (x_c, t_c), (x_ic, t_ic), (x_l, t_l), (x_r, t_r) = fixed_points
        points = (x_c, t_c, x_ic, t_ic, x_l, t_l, x_r, t_r)

    x_c, t_c, x_ic, t_ic, x_l, t_l, x_r, t_r = points
    for epoch in range(start_epoch, args.epochs):
        optimizer.zero_grad()
        loss, l_pde, l_ic, l_bc = total_loss(
            model,
            points,
            args.nu,
            lambda_pde=train_cfg["lambda_pde"],
            lambda_ic=train_cfg["lambda_ic"],
            lambda_bc=train_cfg["lambda_bc"],
        )
        loss.backward()
        optimizer.step()

        if epoch % 2000 == 0:
            print(f"[{epoch:5d}] PDE: {l_pde:.4e}  IC: {l_ic:.4e}  BC: {l_bc:.4e}")

        if ((epoch + 1) % 1000 == 0) or (epoch + 1 == args.epochs):
            save_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                epoch + 1,
                (x_c, t_c, x_ic, t_ic, x_l, t_l, x_r, t_r),
                train_cfg,
                search_result,
            )

    if not args.skip_lbfgs:
        lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=3000, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            total, _, _, _ = total_loss(
                model,
                points,
                args.nu,
                lambda_pde=train_cfg["lambda_pde"],
                lambda_ic=train_cfg["lambda_ic"],
                lambda_bc=train_cfg["lambda_bc"],
            )
            total.backward()
            return total

        lbfgs.step(closure)


def plot_heatmap(model, save_dir):
    plot_burgers_heatmap(
        model,
        device,
        x_min,
        x_max,
        t_min,
        t_max,
        os.path.join(save_dir, "burgers_heatmap.png"),
        use_interactive=interactive_plots,
    )


def plot_time_slices(model, save_dir, t_values=None, nx=400):
    plot_burgers_time_slices(
        model,
        device,
        x_min,
        x_max,
        os.path.join(save_dir, "burgers_time_slices.png"),
        t_values=t_values,
        nx=nx,
        use_interactive=interactive_plots,
    )


def plot_time_slices_with_exact(model, save_dir, mat_path="burgers_shock.mat", t_values=None):
    plot_burgers_time_slices_with_exact(
        model,
        device,
        os.path.join(save_dir, "burgers_exact_vs_pred_time_slices.png"),
        mat_path=mat_path,
        t_values=t_values,
        use_interactive=interactive_plots,
    )


def plot_full_exact_vs_pred(model, save_dir, mat_path="burgers_shock.mat"):
    plot_burgers_full_exact_vs_pred(
        model,
        device,
        os.path.join(save_dir, "burgers_full_exact_vs_pred.png"),
        mat_path=mat_path,
        pred_title="Predicted (NAS-PINN)",
        use_interactive=interactive_plots,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="NAS-PINN Burgers training and plotting (NSGA-II)")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_last_nsga2.pth", help="checkpoint path")
    parser.add_argument("--resume", action="store_true", help="resume training from checkpoint")
    parser.add_argument("--plot-only", action="store_true", help="skip training and only run plots from checkpoint")
    parser.add_argument("--skip-lbfgs", action="store_true", help="skip L-BFGS refinement")
    parser.add_argument("--nu", type=float, default=nu, help="single-run viscosity coefficient")
    parser.add_argument("--multi-nu", action="store_true", help="run three viscosity values and save comparison")
    parser.add_argument("--nu-list", type=str, default="0.01,0.04,0.07", help="comma-separated viscosities for --multi-nu")
    parser.add_argument("--epochs", type=int, default=15000, help="number of Adam epochs")
    parser.add_argument("--save-dir", type=str, default="results/burgers/nsga2", help="directory to save plot images")
    parser.add_argument("--seed", type=int, default=42, help="base random seed")

    parser.add_argument("--skip-nsga", action="store_true", help="skip NSGA-II and use manual/default architecture")
    parser.add_argument("--nsga-pop", type=int, default=10, help="NSGA-II population size")
    parser.add_argument("--nsga-gen", type=int, default=5, help="NSGA-II generations")
    parser.add_argument("--nsga-eval-epochs", type=int, default=400, help="training epochs per NSGA candidate")
    parser.add_argument("--nsga-max-layers", type=int, default=6, help="max hidden layers in NSGA search")
    parser.add_argument("--nsga-test-nt", type=int, default=21, help="NSGA validation t-grid points")
    parser.add_argument("--nsga-test-nx", type=int, default=300, help="NSGA validation x-grid points")

    parser.add_argument("--lr", type=float, default=1e-3, help="manual/default Adam learning rate")
    parser.add_argument("--lambda-ic", type=float, default=100.0, help="manual/default IC weight")
    parser.add_argument("--lambda-bc", type=float, default=100.0, help="manual/default BC weight")
    parser.add_argument("--activation", type=str, choices=["tanh", "sin"], default="tanh", help="manual/default activation")
    parser.add_argument("--use-residual", action="store_true", help="manual/default residual connections")
    return parser.parse_args()


def run_single(args):
    run_start = time.perf_counter()
    os.makedirs(args.save_dir, exist_ok=True)
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(args.save_dir, args.checkpoint)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.plot_only:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found for plot-only mode: {args.checkpoint}")
        model_cfg, train_cfg, search_result = read_checkpoint_metadata(args.checkpoint)
        if model_cfg is None or train_cfg is None:
            model_cfg = {"layer_sizes": [2, 128, 128, 128, 1], "activation": "tanh", "use_residual": False}
            train_cfg = {"lr": args.lr, "lambda_pde": 1.0, "lambda_ic": args.lambda_ic, "lambda_bc": args.lambda_bc}
            search_result = None
        model = NASPINNFixed(**model_cfg).to(device)
        load_checkpoint(args.checkpoint, model)
    else:
        fixed_points = sample_points()
        if args.skip_nsga:
            model_cfg = {"layer_sizes": [2, 128, 128, 128, 1], "activation": args.activation, "use_residual": args.use_residual}
            train_cfg = {"lr": args.lr, "lambda_pde": 1.0, "lambda_ic": args.lambda_ic, "lambda_bc": args.lambda_bc}
            search_result = None
        else:
            search_result = run_nsga2_search(
                nu_coef=args.nu,
                fixed_points=fixed_points,
                pop=args.nsga_pop,
                ngen=args.nsga_gen,
                eval_epochs=args.nsga_eval_epochs,
                max_hidden_layers=args.nsga_max_layers,
                test_nt=args.nsga_test_nt,
                test_nx=args.nsga_test_nx,
                seed=args.seed,
            )
            model_cfg = search_result["model_config"]
            train_cfg = search_result["train_cfg"]

        model = NASPINNFixed(**model_cfg).to(device)
        train_with_resume(model, args, train_cfg, fixed_points=fixed_points, search_result=search_result)

    print("\nDiscovered architecture (NSGA-II):")
    print(f"  Layers: {model.model_config['layer_sizes']}")
    print(f"  Activation: {model.model_config['activation']} | Residual: {model.model_config['use_residual']}")

    plot_heatmap(model, args.save_dir)
    if should_use_exact_plots(args.nu):
        plot_time_slices(model, args.save_dir)
        plot_time_slices_with_exact(model, args.save_dir)
        plot_full_exact_vs_pred(model, args.save_dir, "burgers_shock.mat")
    else:
        print("Exact/time-slice comparison is only generated for --nu=0.01. Saving heatmap + rel L2 for this run.")

    x_test = torch.linspace(x_min, x_max, args.nsga_test_nx, device=device)
    t_test = torch.linspace(t_min, t_max, args.nsga_test_nt, device=device)
    u_pred = predict_on_grid(model, x_test, t_test)
    u_exact = get_reference_solution(args.nu, x_test, t_test)
    rel_l2 = np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-12)
    with open(os.path.join(args.save_dir, "l2_error.txt"), "w", encoding="utf-8") as f:
        f.write(f"nu,{args.nu:.6f}\nrel_l2,{rel_l2:.8e}\n")
    run_time = time.perf_counter() - run_start
    with open(os.path.join(args.save_dir, "run_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"run_time_seconds,{run_time:.6f}\n")
    print(f"Run time: {run_time:.2f} s")
    return float(rel_l2), float(run_time)


def run_multi_nu(args):
    nu_values = [float(v.strip()) for v in args.nu_list.split(",") if v.strip()]
    summary = []
    base_dir = args.save_dir
    os.makedirs(base_dir, exist_ok=True)

    for idx, nu_val in enumerate(nu_values):
        args_local = argparse.Namespace(**vars(args))
        args_local.nu = nu_val
        args_local.seed = args.seed + idx
        args_local.save_dir = os.path.join(base_dir, f"nu_{nu_val:.3f}")
        args_local.checkpoint = "checkpoint_last_nsga2.pth"
        rel_l2, run_time = run_single(args_local)
        summary.append((nu_val, rel_l2, run_time))

    csv_path = os.path.join(base_dir, "viscosity_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("nu,rel_l2,run_time_seconds\n")
        for nu_val, rel_l2, run_time in summary:
            f.write(f"{nu_val:.6f},{rel_l2:.8e},{run_time:.6f}\n")
    print(f"Saved summary: {csv_path}")

    nus = [row[0] for row in summary]
    errs = [row[1] for row in summary]
    plt.figure(figsize=(7, 4))
    plt.plot(nus, errs, marker="o", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Viscosity (nu)")
    plt.ylabel("Relative L2 Error")
    plt.title("Burgers NSGA-II: Viscosity Comparison")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    finalize_plot(os.path.join(base_dir, "viscosity_comparison.png"))


def main():
    args = parse_args()
    if args.multi_nu:
        run_multi_nu(args)
        return

    run_single(args)


if __name__ == "__main__":
    main()
