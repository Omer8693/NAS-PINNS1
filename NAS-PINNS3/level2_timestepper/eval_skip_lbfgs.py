"""
eval_skip_lbfgs.py — Level 2 + L-BFGS per window Skip Evaluation
=================================================================
Runs Level 2 window-based skip evaluation with L-BFGS refinement
applied after Adam in each time window.

Hypothesis: NSGA-II (skip=2 ratio=1.47×, just above threshold) and
NSGA-III should converge once per-window L-BFGS is applied.

Results saved to: results/skip_table_lbfgs.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT    = Path(__file__).resolve().parent.parent
_L2_ROOT = Path(__file__).resolve().parent

import os
os.chdir(str(_L2_ROOT))
# level2_timestepper must come BEFORE project root so its src/ takes priority
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_L2_ROOT))

from src.ts_model    import TimeStepperPINN
from src.ts_evaluate import skip_table

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_END   = 30.0
N_STEPS = 20

CONFIGS = {
    "bayesian": {"n_layers": 5, "neurons": [151]*5, "activation": "relu"},
    "nsga2":    {"n_layers": 3, "neurons": [153]*3, "activation": "tanh"},
    "nsga3":    {"n_layers": 3, "neurons": [75]*3,  "activation": "tanh"},
}

# Adam phase same as Level 2, plus L-BFGS refinement per window
TRAIN_KWARGS = {
    "n_domain":    500,
    "n_bc":        100,
    "n_epochs":    300,
    "lr":          1e-3,
    "lr_min":      1e-5,
    "lbfgs_iters": 30,    # L-BFGS per window — key addition
}

SKIP_VALUES  = [1, 2, 4, 6]
CONV_FACTOR  = 1.5


def run():
    t_steps     = np.linspace(0.0, T_END, N_STEPS + 1)
    all_results = {}

    for opt_name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  {opt_name.upper()}  [L={cfg['n_layers']} N={cfg['neurons'][0]}"
              f" act={cfg['activation']}]  +L-BFGS per window")
        print(f"{'='*60}")

        rows = skip_table(
            model_fn     = lambda c=cfg: TimeStepperPINN(c).to(DEVICE),
            skip_values  = SKIP_VALUES,
            t_steps      = t_steps,
            train_kwargs = TRAIN_KWARGS,
            device       = DEVICE,
        )
        all_results[opt_name] = rows

        base_mae = rows[0]["mae_C"]
        for row in rows:
            ratio = row["mae_C"] / (base_mae + 1e-9)
            conv  = "✓ CONVERGED" if ratio < CONV_FACTOR else "✗ DIVERGED"
            print(f"  skip={row['skip']:2d}  steps={row['steps_used']:2d}/{row['steps_total']:2d}"
                  f"  L2={row['l2_rel']:.4f}  MAE={row['mae_C']:.1f}°C"
                  f"  {row['runtime_s']:.0f}s  {conv} ({ratio:.2f}×)")

    out_path = _ROOT / "level2_timestepper" / "results" / "skip_table_lbfgs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")
    return all_results


if __name__ == "__main__":
    run()
