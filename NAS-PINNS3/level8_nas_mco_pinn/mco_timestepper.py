"""
mco_timestepper.py — MCO-Enhanced NAS-PINN Time-Step Skip Operator
===================================================================
TRUE LEVEL 8: Level 2 enhanced with MCO (Multi-Loss Consistency Optimization).

Thesis question: "Can FEM's step-by-step computation be replaced by PINN time-stepping?"
Level 2 answer: skip=2 → 2× speedup, MAE=33°C (Bayesian)
Level 8 question: Do MCO weights reduce this MAE further?

Shen 2023 (MCO-PINN) method:
  w_i = 1/σᵢ²  where σᵢ → adapted via Gaussian MLE
  Innovation: each time window performs its own MCO adaptation
  → Early windows (high ΔT, large gradient) get different balance
  → Late windows (cold, smooth) get different balance

Reference comparison:
  FEM (baseline)           : 21 steps, ~183s, MAE≈0°C (reference)
  NAS-PINN L2 skip=1 Bay.  : 21 steps, ~184s, MAE=43.6°C
  NAS-PINN L2 skip=2 Bay.  : 11 steps,  ~90s, MAE=33.2°C ← current best
  NAS-MCO-PINN L8 skip=2   : 11 steps,   ?s,  MAE=?°C ← this file
  NAS-MCO-PINN L8 skip=4   :  6 steps,   ?s,  MAE=?°C ← this file
  MCO-PINN Shen 2023        :  —  steps,   —s, RMSE=0.0041 (industrial 3D)
"""

import time
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from typing import List, Dict, Optional, Callable, Tuple

try:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.config import DEVICE
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Physical constants (A356 aluminum, Mortensen 2026) ─────
T_INIT  = 540.0    # °C — after SHT
T_WATER = 20.0     # °C — water bath
T_SPAN  = 520.0    # °C
ALPHA   = 1.75e-3  # 1/s — cooling constant (analytical)
K       = 160.0    # W/mK — thermal conductivity
RHO_CP  = 2.4e6    # J/m³K
H_CONV  = 4000.0   # W/m²K — convection coefficient
# Domain dimensions (Mortensen 2026: 2D → same as Level 2)
LX, LY  = 1.3, 0.6  # meters
T_END   = 30.0
N_STEPS = 20


# ─── Aktivasyonlar ──────────────────────────────────────────

class SinAct(nn.Module):
    def forward(self, x): return torch.sin(x)

class GaussianAct(nn.Module):
    def forward(self, x): return torch.exp(-x**2)

ACTS = {
    "tanh":     nn.Tanh,
    "relu":     nn.ReLU,
    "swish":    nn.SiLU,
    "gelu":     nn.GELU,
    "sin":      SinAct,
    "gaussian": GaussianAct,   # recommended by Shen 2023
}


# ─── MCO Loss Function (Shen 2023 adaptation) ────────────────

class MCOLoss(nn.Module):
    """
    Shen et al. 2023: w_i = 1/σᵢ², log-likelihood optimized.
    Separate MCO instance used for each time window.
    3 losses: physics residual, boundary condition (BC), initial condition (IC).
    """
    def __init__(self, n=3, init_log_sigma: float = 0.0):
        super().__init__()
        self.log_sigma = nn.Parameter(
            torch.full((n,), init_log_sigma)
        )

    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, List[float]]:
        total = torch.zeros(1, device=self.log_sigma.device)
        weights = []
        for i, L in enumerate(losses):
            sigma2 = torch.exp(2 * self.log_sigma[i])
            total  = total + L / sigma2 + self.log_sigma[i]
            weights.append(float(1.0 / sigma2.detach().item()))
        return total, weights

    def sigmas(self) -> List[float]:
        return [float(torch.exp(s).item()) for s in self.log_sigma]


# ─── Network architecture ───────────────────────────────────

def build_net(config: dict) -> nn.Module:
    """
    Input: [x_norm, y_norm, t_local_norm, theta_prev]  → 4 dims
    Output: theta_next → 1 dim
    """
    act_cls = ACTS.get(config.get("activation", "tanh"), nn.Tanh)
    layers, in_dim = [], 4
    for w in config["neurons"]:
        layers += [nn.Linear(in_dim, w), act_cls()]
        in_dim = w
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


# ─── Analytical reference (same as Level 2) ─────────────────

def analytical_T(t: float) -> float:
    """T(t) = T_water + (T_init - T_water) · exp(-α·t)"""
    return T_WATER + T_SPAN * np.exp(-ALPHA * t)


def analytical_theta(t: float) -> float:
    return np.exp(-ALPHA * t)  # θ = (T - T_w)/T_span normalization


# ─── Window training (with MCO) ─────────────────────────────

def _normalize_xy(x: torch.Tensor, y: torch.Tensor):
    return x / LX, y / LY


def train_window_mco(
    net:       nn.Module,
    mco:       MCOLoss,
    t_start:   float,    # window start [s]
    t_end:     float,    # window end [s]
    theta_ref: float,    # initial θ value (for IC)
    n_domain:  int   = 600,
    n_bc:      int   = 150,
    n_adam:    int   = 800,
    lr:        float = 1e-3,
    device     = None,
) -> dict:
    """
    Train a single time window with MCO.

    Losses:
      L_phys: ρCp ∂θ/∂t = K ∇²θ in interior (normalized)
      L_bc:   θ_boundary ≈ θ_lc(t)  (Robin BC approximation)
      L_ic:   θ(x,y,0) = theta_ref  (temperature from previous step)
    """
    device = device or DEVICE
    dt = t_end - t_start

    # Fourier number for normalization
    Fo = K * dt / (RHO_CP * min(LX, LY)**2)   # Fourier number

    all_params = list(net.parameters()) + list(mco.parameters())
    adam = optim.Adam(all_params, lr=lr)
    sched = optim.lr_scheduler.ExponentialLR(adam, gamma=0.9995)

    history = []

    for step in range(n_adam):
        adam.zero_grad()
        rng_seed = step  # reproducibility

        # ── Interior points ─────────────────────────────
        rng = torch.Generator(device=device)
        rng.manual_seed(rng_seed)
        x_f  = torch.rand(n_domain, 1, device=device, generator=rng) * LX
        y_f  = torch.rand(n_domain, 1, device=device, generator=rng) * LY
        t_f  = torch.rand(n_domain, 1, device=device, generator=rng) * dt
        xn, yn = x_f/LX, y_f/LY
        tn = t_f / dt  # [0,1]

        # Previous step temperature (IC θ at window start)
        theta_prev_f = torch.full_like(xn, theta_ref)
        inp_f = torch.cat([xn, yn, tn, theta_prev_f], dim=1).requires_grad_(True)
        theta_pred = net(inp_f)

        # Physics residual — normalized PDE: ∂θ/∂τ = Fo·∇²θ
        grads = torch.autograd.grad(
            theta_pred, inp_f,
            grad_outputs=torch.ones_like(theta_pred),
            create_graph=True, retain_graph=True
        )[0]
        dtheta_dt = grads[:, 2:3] / dt     # time derivative
        dtheta_dx = grads[:, 0:1] / LX
        dtheta_dy = grads[:, 1:2] / LY

        d2x = torch.autograd.grad(
            dtheta_dx, inp_f, torch.ones_like(dtheta_dx),
            create_graph=True, retain_graph=True
        )[0][:, 0:1] / LX

        d2y = torch.autograd.grad(
            dtheta_dy, inp_f, torch.ones_like(dtheta_dy),
            create_graph=True, retain_graph=True
        )[0][:, 1:2] / LY

        pde_res = dtheta_dt - (K / RHO_CP) * (d2x + d2y)
        L_phys = torch.mean(pde_res**2) / (Fo**2 + 1e-8)  # normalize

        # ── Boundary points (BC) ────────────────────────
        rng.manual_seed(rng_seed + 1000)
        x_b  = torch.rand(n_bc, 1, device=device, generator=rng) * LX
        y_b  = torch.rand(n_bc, 1, device=device, generator=rng) * LY
        t_b  = torch.rand(n_bc, 1, device=device, generator=rng) * dt
        # Project boundary points onto edges
        side = torch.randint(0, 4, (n_bc,), device=device, generator=rng)
        x_b = torch.where(side.unsqueeze(1) == 0, torch.zeros_like(x_b), x_b)
        x_b = torch.where(side.unsqueeze(1) == 1, torch.full_like(x_b, LX), x_b)
        y_b = torch.where(side.unsqueeze(1) == 2, torch.zeros_like(y_b), y_b)
        y_b = torch.where(side.unsqueeze(1) == 3, torch.full_like(y_b, LY), y_b)

        # Robin BC: θ_surface ≈ θ_lc(t_start + t_local) (lumped capacitance)
        t_global_b = t_start + t_b
        theta_bc_ref = torch.exp(torch.tensor(-ALPHA, device=device) * t_global_b)
        theta_prev_b = torch.full_like(x_b/LX, theta_ref)
        inp_b = torch.cat([x_b/LX, y_b/LY, t_b/dt, theta_prev_b], dim=1)
        theta_b_pred = net(inp_b)
        L_bc = torch.mean((theta_b_pred - theta_bc_ref)**2)

        # ── Initial condition (IC: t=0, T=T_prev) ───────
        rng.manual_seed(rng_seed + 2000)
        x_i = torch.rand(n_bc, 1, device=device, generator=rng) * LX
        y_i = torch.rand(n_bc, 1, device=device, generator=rng) * LY
        t_i = torch.zeros(n_bc, 1, device=device)
        theta_prev_i = torch.full_like(x_i/LX, theta_ref)
        inp_i = torch.cat([x_i/LX, y_i/LY, t_i, theta_prev_i], dim=1)
        theta_i_pred = net(inp_i)
        L_ic = torch.mean((theta_i_pred - theta_ref)**2)

        # ── MCO total loss ───────────────────────────────
        total, weights = mco([L_phys, L_bc, L_ic])
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        adam.step()
        sched.step()

        if step % 200 == 0:
            history.append({
                "step": step,
                "L_phys": float(L_phys.item()),
                "L_bc": float(L_bc.item()),
                "L_ic": float(L_ic.item()),
                "w_phys": weights[0], "w_bc": weights[1], "w_ic": weights[2],
            })

    return {
        "final_L_phys": float(L_phys.item()),
        "final_L_bc":   float(L_bc.item()),
        "final_L_ic":   float(L_ic.item()),
        "final_weights": weights,
        "final_sigmas":  mco.sigmas(),
        "history": history,
    }


# ─── Skip table (same interface as Level 2) ─────────────────

def evaluate_skip_mco(
    config:       dict,
    skip:         int,
    t_steps:      np.ndarray,
    train_kwargs: dict,
    device        = None,
    verbose:      bool = True,
) -> dict:
    """
    Evaluate a specific skip value with MCO.
    Returns same format as Level 2 evaluate_skip() function.
    """
    device = device or DEVICE
    anchor_times = t_steps[::skip]
    n_anchors = len(anchor_times)

    if verbose:
        print(f"  skip={skip}: {n_anchors} FEM anchor → {len(t_steps)-n_anchors} PINN steps")

    errors_mae = []
    window_logs = []
    t0 = time.time()

    for i in range(1, n_anchors):
        t_start  = float(anchor_times[i-1])
        t_end    = float(anchor_times[i])
        theta_ref = analytical_theta(t_start)   # IC: analytical initial condition

        # Fresh network + MCO for each window
        net = build_net(config).to(device)
        mco = MCOLoss(n=3).to(device)

        log = train_window_mco(
            net, mco,
            t_start=t_start, t_end=t_end,
            theta_ref=theta_ref,
            device=device,
            **train_kwargs,
        )
        window_logs.append(log)

        # Prediction at window end
        n_eval = 800
        xv = torch.rand(n_eval, 1, device=device) * LX
        yv = torch.rand(n_eval, 1, device=device) * LY
        tp = torch.full((n_eval, 1), t_end - t_start, device=device)
        th_prev = torch.full((n_eval, 1), theta_ref, device=device)
        inp = torch.cat([xv/LX, yv/LY, tp/(t_end-t_start), th_prev], dim=1)

        with torch.no_grad():
            theta_pred = net(inp).cpu().numpy().squeeze()

        T_pred = T_WATER + theta_pred * T_SPAN
        T_ref  = analytical_T(t_end)
        mae    = float(np.mean(np.abs(T_pred - T_ref)))
        errors_mae.append(mae)

        if verbose:
            w = log["final_weights"]
            print(f"    [{i}/{n_anchors-1}] t={t_start:.1f}→{t_end:.1f}s  "
                  f"MAE={mae:.1f}°C  "
                  f"w=[{w[0]:.0f},{w[1]:.0f},{w[2]:.0f}]")

    elapsed = time.time() - t0
    mae_mean = float(np.mean(errors_mae))

    return {
        "skip":        skip,
        "steps_used":  n_anchors,
        "steps_total": len(t_steps),
        "mae_C":       mae_mean,
        "mae_per_window": errors_mae,
        "runtime_s":   elapsed,
        "window_logs": window_logs,
    }


# ─── NAS search (simple grid, comparable to Level 2) ─────────

SEARCH_SPACE = {
    "bayesian": {"n_layers": 5, "neurons": [151]*5, "activation": "relu"},   # L1 best
    "nsga2":    {"n_layers": 3, "neurons": [153]*3, "activation": "tanh"},   # L2 arch
    "nsga3":    {"n_layers": 3, "neurons": [75]*3,  "activation": "tanh"},   # L2 arch
    "mco_shen": {"n_layers": 4, "neurons": [64]*4,  "activation": "gaussian"}, # Shen 2023 architecture
}

SKIP_VALUES = [1, 2, 4, 6]

TRAIN_KWARGS = {
    "n_domain": 600,
    "n_bc":     150,
    "n_adam":   800,
    "lr":       1e-3,
}


def run_full_comparison(
    out_path:   str = "results/level8_skip_results.json",
    skip_values = SKIP_VALUES,
    verbose:    bool = True,
):
    """
    Run MCO time-stepper for all architectures × all skip values.
    Produces output comparable with Level 2 results.
    """
    t_steps = np.linspace(0, T_END, N_STEPS + 1)

    # Level 2 reference results (existing)
    level2_ref = {
        "bayesian": {1: 43.6, 2: 33.2, 4: 57.5, 6: 93.6},
        "nsga2":    {1: 38.1, 2: 56.0, 4: 154.8, 6: 354.6},
        "nsga3":    {1: 44.4, 2: 75.6, 4: 202.1, 6: 428.6},
    }
    shen_ref = {"rmse_norm": 0.0041, "note": "Shen et al. 2023, 3D industrial furnace"}

    print("\n" + "="*65)
    print("  NAS-MCO-PINN Level 8 — Time-Step Skip Comparison")
    print("  Shen 2023 MCO weights + Level 2 skip operator")
    print("="*65)

    results = {"level2_ref": level2_ref, "shen_2023_ref": shen_ref, "level8_mco": {}}

    for arch_name, config in SEARCH_SPACE.items():
        print(f"\n{'─'*55}")
        print(f"  Architecture: {arch_name}  "
              f"({config['n_layers']}L × {config['neurons'][0]}N, {config['activation']})")
        print(f"{'─'*55}")
        results["level8_mco"][arch_name] = {}

        for skip in skip_values:
            res = evaluate_skip_mco(
                config=config, skip=skip,
                t_steps=t_steps,
                train_kwargs=TRAIN_KWARGS,
                verbose=verbose,
            )
            results["level8_mco"][arch_name][skip] = res

            # Comparison
            l2_ref = level2_ref.get(arch_name, {}).get(skip, None)
            if l2_ref:
                delta = res["mae_C"] - l2_ref
                sign  = "▲" if delta > 0 else "▼"
                print(f"  skip={skip}: MCO={res['mae_C']:.1f}°C  "
                      f"L2ref={l2_ref:.1f}°C  {sign}{abs(delta):.1f}°C  "
                      f"({res['steps_used']}/{res['steps_total']} FEM steps)")

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        # Remove window_logs (large)
        clean = json.loads(json.dumps(results, default=str))
        for arch in clean.get("level8_mco", {}):
            for skip_k in clean["level8_mco"][arch]:
                clean["level8_mco"][arch][skip_k].pop("window_logs", None)
        json.dump(clean, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Summary table
    _print_summary_table(results)
    return results


def _print_summary_table(results: dict):
    """Level 2 vs Level 8 MCO comparison table."""
    l2_ref = results["level2_ref"]
    l8     = results["level8_mco"]

    print("\n" + "="*72)
    print("  SUMMARY: NAS-PINN L2 (fixed weights) vs NAS-MCO-PINN L8 (MCO)")
    print("="*72)
    print(f"  {'Architecture':<12} {'skip':<6} {'L2 MAE':>10} {'L8 MCO MAE':>12} "
          f"{'Δ':>8}  {'FEM steps':>10}")
    print("  " + "─"*68)

    for arch in ["bayesian", "nsga2", "nsga3"]:
        for skip in SKIP_VALUES:
            l2_mae = l2_ref.get(arch, {}).get(skip, None)
            l8_row = l8.get(arch, {}).get(skip, None)
            if l8_row is None:
                continue
            l8_mae = l8_row["mae_C"]
            steps  = l8_row["steps_used"]
            total  = l8_row["steps_total"]
            if l2_mae:
                delta = l8_mae - l2_mae
                sign  = "+" if delta > 0 else ""
                delta_str = f"{sign}{delta:.1f}°C"
            else:
                delta_str = "—"
            print(f"  {arch:<12} {skip:<6} {l2_mae if l2_mae else '—':>10} "
                  f"{l8_mae:>10.1f}°C  {delta_str:>8}  "
                  f"{steps}/{total}")

    print("\n  Reference: MCO-PINN Shen 2023 → RMSE=0.0041 (normalized), "
          "3D industrial furnace\n")


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="+", type=int, default=SKIP_VALUES)
    parser.add_argument("--arch", default="all",
                        help="bayesian/nsga2/nsga3/mco_shen/all")
    parser.add_argument("--out", default="results/level8_skip_results.json")
    args = parser.parse_args()

    run_full_comparison(
        out_path=args.out,
        skip_values=args.skip,
    )
