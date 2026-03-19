"""
eval_skip_5k.py — Level 2 High-Epoch Skip Test
================================================
Runs NSGA-II and NSGA-III only at skip=1 and skip=2
with 2000 Adam epochs to test whether they can converge.

Result saved to: results/skip_table_5k.json
"""
import json, sys, os
from pathlib import Path

_ROOT    = Path(__file__).resolve().parent.parent
_L2_ROOT = Path(__file__).resolve().parent
os.chdir(str(_L2_ROOT))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_L2_ROOT))   # L2 first → wins over project-root src

from src.ts_model    import TimeStepperPINN
from src.ts_evaluate import skip_table
import numpy as np, torch

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_END   = 30.0
N_STEPS = 20
CONV    = 1.5

CONFIGS = {
    "nsga2": {"n_layers": 3, "neurons": [153]*3, "activation": "tanh"},
    "nsga3": {"n_layers": 3, "neurons": [75]*3,  "activation": "tanh"},
}

TRAIN_KWARGS = {
    "n_domain": 500,
    "n_bc":     100,
    "n_epochs": 2000,
    "lr":       1e-3,
    "lr_min":   1e-5,
}

SKIP_VALUES = [1, 2]

def run():
    t_steps     = np.linspace(0.0, T_END, N_STEPS + 1)
    all_results = {}

    # Load Level 2 Adam-only baseline for reference
    ref_path = _L2_ROOT / "results" / "skip_table.json"
    with open(ref_path) as f:
        ref = json.load(f)

    for opt, cfg in CONFIGS.items():
        ref_mae = {r["skip"]: r for r in ref.get(opt, [])}.get(1, {}).get("mae_C", 999)
        print(f"\n{'='*60}")
        print(f"  {opt.upper()}  2000 epoch  (ref skip=1 MAE={ref_mae:.1f}°C)")
        print(f"{'='*60}")

        rows = skip_table(
            model_fn     = lambda c=cfg: TimeStepperPINN(c).to(DEVICE),
            skip_values  = SKIP_VALUES,
            t_steps      = t_steps,
            train_kwargs = TRAIN_KWARGS,
            device       = DEVICE,
        )
        all_results[opt] = rows

        base = rows[0]["mae_C"]
        for row in rows:
            ratio = row["mae_C"] / (ref_mae + 1e-9)
            conv  = "✓ CONVERGED" if ratio < CONV else "✗ DIVERGED"
            print(f"  skip={row['skip']}  MAE={row['mae_C']:.1f}°C"
                  f"  ratio={ratio:.2f}×  {conv}  {row['runtime_s']:.0f}s")

    out = _L2_ROOT / "results" / "skip_table_5k.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    run()
