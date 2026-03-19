"""
eval_skip_l5.py — Level 5 Skip Evaluation
==========================================
Evaluates Level 5 (L-BFGS refined) PINNNet models for temporal skip capability.

Unlike Level 2 (window-based TimeStepperPINN), Level 5 models are global
single-shot PINNs (input: t, x, y). Skip evaluation here means:
  - FEM anchors at t_steps[::skip]
  - PINN predicts at all SKIPPED time points
  - L2_rel and MAE measured against analytical reference

Results saved to: results/level5_refinement/skip_table_l5.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.pinn_network import PINNNet

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_END    = 30.0
N_STEPS  = 20
ALPHA    = 1.75e-3
T_WATER  = 20.0
T_INIT   = 540.0
N_EVAL   = 2000   # spatial sample points per time step

SKIP_VALUES = [1, 2, 4, 6]

CONFIGS = {
    "bayesian": {"n_layers": 5, "neurons": [151]*5, "activation": "relu"},
    "nsga2":    {"n_layers": 3, "neurons": [153]*3, "activation": "tanh"},
    "nsga3":    {"n_layers": 3, "neurons": [75]*3,  "activation": "tanh"},
}

# Convergence threshold: MAE ratio vs skip=1 baseline
CONV_FACTOR = 1.5


def T_analytical(t: float) -> float:
    return T_WATER + (T_INIT - T_WATER) * np.exp(-ALPHA * t)


def load_model(opt_name: str) -> PINNNet:
    cfg = CONFIGS[opt_name]
    net = PINNNet(
        n_input      = 3,
        n_output     = 1,
        hidden_sizes = cfg["neurons"],
        activation   = cfg["activation"],
        residual     = False,
    )
    wt = _ROOT / "level5_refinement" / "results" / opt_name / "model_lbfgs.pt"
    net.load_state_dict(torch.load(wt, map_location="cpu", weights_only=False))
    net.eval()
    return net.to(DEVICE)


def evaluate_skip(net: PINNNet, skip: int) -> dict:
    """
    For a given skip value:
      - anchor steps: t_steps[::skip]  (FEM would solve these)
      - skipped steps: all other t_steps (PINN predicts these)
    If skip=1: no steps are skipped, evaluate at all non-zero steps.
    """
    t_steps     = np.linspace(0.0, T_END, N_STEPS + 1)
    anchor_idx  = set(range(0, len(t_steps), skip))

    # For skip=1: no skipped steps → evaluate ALL steps (same as level 1)
    if skip == 1:
        eval_times = t_steps[1:]   # skip t=0 (initial condition, always known)
    else:
        eval_times = [t_steps[i] for i in range(1, len(t_steps))
                      if i not in anchor_idx]

    if len(eval_times) == 0:
        # All points are anchors — return zero error (FEM handles everything)
        return {
            "skip": skip,
            "steps_used":  len(anchor_idx),
            "steps_total": len(t_steps),
            "l2_rel": 0.0,
            "mae_C":  0.0,
            "runtime_s": 0.0,
        }

    t0 = time.time()
    errors_l2  = []
    errors_mae = []

    x_eval = torch.rand(N_EVAL, 1, device=DEVICE) * 1.3   # x ∈ [0, 1.3] m
    y_eval = torch.rand(N_EVAL, 1, device=DEVICE) * 0.6   # y ∈ [0, 0.6] m

    for t_val in eval_times:
        t_tensor = torch.full((N_EVAL, 1), t_val, device=DEVICE, dtype=torch.float32)
        pts      = torch.cat([t_tensor, x_eval, y_eval], dim=1)

        with torch.no_grad():
            T_pred = net(pts).squeeze()   # (N_EVAL,)

        T_ref = torch.full((N_EVAL,), T_analytical(t_val),
                           device=DEVICE, dtype=torch.float32)

        l2  = float((torch.norm(T_pred - T_ref) /
                     torch.norm(T_ref - T_WATER)).item())
        mae = float(torch.mean(torch.abs(T_pred - T_ref)).item())
        errors_l2.append(l2)
        errors_mae.append(mae)

    return {
        "skip":        skip,
        "steps_used":  len(anchor_idx),
        "steps_total": len(t_steps),
        "l2_rel":      float(np.mean(errors_l2)),
        "mae_C":       float(np.mean(errors_mae)),
        "runtime_s":   time.time() - t0,
    }


def run():
    all_results = {}

    for opt_name in ["bayesian", "nsga2", "nsga3"]:
        print(f"\n{'='*55}")
        print(f"  {opt_name.upper()}  — Level 5 L-BFGS model")
        print(f"{'='*55}")

        net  = load_model(opt_name)
        rows = []

        for skip in SKIP_VALUES:
            row = evaluate_skip(net, skip)
            rows.append(row)
            conv = ""
            if len(rows) > 1:
                ratio = row["mae_C"] / (rows[0]["mae_C"] + 1e-9)
                conv  = "✓ CONVERGED" if ratio < CONV_FACTOR else "✗ DIVERGED"
                conv += f" ({ratio:.2f}×)"
            print(f"  skip={skip:2d}  steps={row['steps_used']:2d}/{row['steps_total']:2d}"
                  f"  L2={row['l2_rel']:.4f}  MAE={row['mae_C']:.1f}°C"
                  f"  {row['runtime_s']:.1f}s  {conv}")

        all_results[opt_name] = rows

    out_path = _ROOT / "level5_refinement" / "results" / "skip_table_l5.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")
    return all_results


if __name__ == "__main__":
    run()
