"""
main_level6.py — Level 6: Poisson-Assisted Quenching Enhancement
==================================================================
Improves the Level 5 quenching solution by adding a Poisson auxiliary
loss that enforces spatial consistency at critical time slices.

Physics basis (Fizik temeli):
  Heat equation:   ρCp·∂T/∂t = k·ΔT
  At each t=t_k:  ΔT = (ρCp/k)·∂T/∂t   ← Poisson source term

  The Poisson auxiliary loss samples a DENSE SPATIAL GRID at fixed
  critical times {0.5, 2, 5, 10, 20}s, giving the network a targeted
  spatial gradient signal → better temperature field accuracy.

Pipeline:
  Load Level 5 model  →  Adam fine-tune (5 000 ep)  →  Poisson aux loss
  λ_p: 0 → 1.0 (warm-up over 500 epochs to avoid destabilization)

Prerequisite:
  Level 5 must be complete:  results/level5_refinement/{optimizer}/model_lbfgs.pt

Usage:
    cd /home/coder/NAS-PINNS1/NAS-PINNS3
    python level6_poisson_benchmark/main_level6.py
    python level6_poisson_benchmark/main_level6.py --optimizer bayesian
    python level6_poisson_benchmark/main_level6.py --epochs 3000 --lambda_p 0.5
"""

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from level6_poisson_benchmark.src.level6_finetune import (
    run_one, OUT_BASE,
    FINETUNE_EPOCHS, LR_START, LR_END, LAMBDA_P,
)

OPTIMIZERS = ["bayesian", "nsga2", "nsga3"]


def main():
    parser = argparse.ArgumentParser(
        description="Level 6 — Poisson-assisted fine-tuning of Level 5 quenching model"
    )
    parser.add_argument("--optimizer", choices=[*OPTIMIZERS, "all"], default="all",
                        help="Optimizer to fine-tune (default: all)")
    parser.add_argument("--epochs",    type=int,   default=FINETUNE_EPOCHS,
                        help=f"Fine-tune epochs (default: {FINETUNE_EPOCHS})")
    parser.add_argument("--lr_start",  type=float, default=LR_START,
                        help=f"Initial LR (default: {LR_START})")
    parser.add_argument("--lr_end",    type=float, default=LR_END,
                        help=f"Final LR (default: {LR_END})")
    parser.add_argument("--lambda_p",  type=float, default=LAMBDA_P,
                        help=f"Poisson loss weight (default: {LAMBDA_P})")
    parser.add_argument("--n_spatial", type=int,   default=1000,
                        help="Spatial points per time slice for Poisson loss (default: 1000)")
    args = parser.parse_args()

    opts    = OPTIMIZERS if args.optimizer == "all" else [args.optimizer]
    t_start = time.time()
    results = {}

    for opt in opts:
        try:
            r = run_one(
                opt_name        = opt,
                finetune_epochs = args.epochs,
                lr_start        = args.lr_start,
                lr_end          = args.lr_end,
                lambda_p        = args.lambda_p,
                n_spatial       = args.n_spatial,
            )
            results[opt] = r
        except FileNotFoundError as e:
            print(f"\n  [SKIP] {opt}: {e}")
            results[opt] = {"error": str(e)}
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] {opt}: {e}")
            traceback.print_exc()
            results[opt] = {"error": str(e)}

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  LEVEL 6 — Poisson-Assisted Quenching Enhancement Summary")
    print(f"{'='*68}")
    print(f"  {'Optimizer':<12}  {'L5 L2':<10}  {'L6 L2':<10}  {'Improv.':<10}  {'MAE_L6'}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")
    for opt, res in results.items():
        if "error" in res:
            print(f"  {opt:<12}  ERROR: {res['error'][:46]}")
            continue
        b = res.get("level5_before", {})
        a = res.get("level6_after",  {})
        i = res.get("improvement",   {})
        print(f"  {opt:<12}  {b.get('L2_rel',0):<10.5f}  "
              f"{a.get('L2_rel',0):<10.5f}  "
              f"{i.get('L2_pct',0):+.1f}%      "
              f"{a.get('MAE_C',0):.1f}°C")

    print(f"\n  Target: rel_L2 < 0.100")

    # ── Save summary ───────────────────────────────────────────────────────────
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    summary = {
        opt: {k: v for k, v in res.items()
              if k not in ("model", "loss_history", "poisson_history")}
        for opt, res in results.items()
    }
    summary_path = OUT_BASE / "level6_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")
    print(f"  Total time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
