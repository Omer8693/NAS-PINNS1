"""
main_level5.py — Extended Adam with Cosine LR Schedule
=======================================================
Uses Level 1 best architectures with improved training:
  - Cosine LR decay (lr=1e-3 → 1e-5) instead of fixed LR
  - Target: bring NSGA-II (L2=0.252) and NSGA-III (L2=0.513) to target < 0.10

Why not L-BFGS?
  Tested — strong_wolfe fails after 37-73 iters on multi-physics PINN loss.
  Hessian approx becomes ill-conditioned with autograd-of-autograd physics
  residuals and piecewise-nonlinear Robin BC. See DRAFT.md for full analysis.

Prerequisite (Ön koşul):
    Level 1 must be complete: results/run2/quenching/{bayesian,nsga2,nsga3}/

Usage:
    python level5_refinement/main_level5.py
    python level5_refinement/main_level5.py --optimizer nsga2
    python level5_refinement/main_level5.py --adam_epochs 20000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.pinn_network import PINNNet
from src.config import DEVICE
from problems.quenching import QuenchingProblem, CollocationSampler
from level5_refinement.src.batch_sampler import sample_fixed_batch, COL_POINTS, IC_POINTS, BC_POINTS
from level5_refinement.src.lbfgs_refiner import LBFGSRefiner, load_level1_model, LBFGS_CONFIG

OPTIMIZERS = ["bayesian", "nsga2", "nsga3"]
LEVEL1_DIR = _ROOT / "level1_single_shot" / "results"  # passed as str to load_level1_model
OUT_BASE   = _ROOT / "level5_refinement" / "results"

# Adam with cosine LR decay — same epochs as Level 1, better schedule
ADAM_EPOCHS = 20000
ADAM_LR_MAX = 1e-3    # starting LR
ADAM_LR_MIN = 1e-5    # final LR (cosine annealing end)


def run_adam_phase(model, loss_fn, val_fn, sampler, adam_epochs: int,
                   lr: float) -> dict:
    """Adam with cosine annealing LR: lr_max → lr_min over adam_epochs."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=adam_epochs, eta_min=ADAM_LR_MIN
    )

    history = []
    t0 = time.time()
    model.train()

    print(f"\n  Phase 1 — Adam ({adam_epochs} epochs, lr={lr})")
    for epoch in range(adam_epochs):
        # Random mini-batch each epoch (correct for Adam)
        batch = {
            "domain":   sampler.sample_domain(),
            "boundary": None,
            "normals":  None,
            "initial":  sampler.sample_initial(),
        }
        bc_pts, normals = sampler.sample_boundary()
        batch["boundary"] = bc_pts
        batch["normals"]  = normals

        optimizer.zero_grad()
        loss, _ = loss_fn(model, batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        v = float(loss.item())
        if np.isfinite(v):
            history.append(v)

        if (epoch + 1) % 500 == 0:
            l2 = val_fn(model)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"    epoch {epoch+1:5d}/{adam_epochs}  "
                  f"loss={v:.4e}  L2={l2:.4f}  lr={lr_now:.2e}")

    elapsed = time.time() - t0
    l2_final = val_fn(model)
    print(f"  Adam done: {elapsed:.0f}s  |  L2={l2_final:.4f}")
    return {"loss_history": history, "l2_final": l2_final, "elapsed_s": elapsed}


def run_one(opt_name: str, args: argparse.Namespace) -> dict:
    print(f"\n{'='*60}")
    print(f"  Optimizer: {opt_name.upper()}")
    print(f"{'='*60}")

    out_dir = OUT_BASE / opt_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Level 1 architecture (weights NOT used) ───────────────────────
    _, arch_cfg, level1_results = load_level1_model(opt_name, str(LEVEL1_DIR))
    print(f"  Architecture: {arch_cfg['n_layers']}×{arch_cfg['neurons'][0]} "
          f"{arch_cfg['activation']}")
    print(f"  Training: Adam(cosine LR {ADAM_LR_MAX}→{ADAM_LR_MIN}, {args.adam_epochs}ep)")

    # ── 2. Fresh model (random init) ──────────────────────────────────────────
    model = PINNNet(
        n_input      = arch_cfg.get("n_input",  3),
        n_output     = arch_cfg.get("n_output", 1),
        hidden_sizes = arch_cfg["neurons"],
        activation   = arch_cfg["activation"],
        residual     = arch_cfg.get("residual", False),
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}  (fresh init — not loading Level 1 weights)")

    # ── 3. Problem setup ──────────────────────────────────────────────────────
    problem = QuenchingProblem(time_mode="full")
    loss_fn = problem.make_loss_fn()
    val_fn  = problem.make_val_fn(n_val=1000)
    sampler = CollocationSampler(
        n_domain   = args.col_points,
        n_boundary = args.bc_points,
        n_initial  = args.ic_points,
    )

    # ── 4. Adam with cosine LR ────────────────────────────────────────────────
    adam_res = run_adam_phase(model, loss_fn, val_fn, sampler,
                              adam_epochs=args.adam_epochs, lr=ADAM_LR_MAX)

    # ── 5. (Optional) L-BFGS pass ─────────────────────────────────────────────
    # NOTE: L-BFGS stalls after 37-73 iters on this multi-physics loss.
    # Kept here for experimentation only. Primary improvement is cosine LR.
    lbfgs_res = {"l2_before": adam_res["l2_final"], "l2_after": adam_res["l2_final"],
                 "loss_before": float("nan"), "loss_after": float("nan"),
                 "iters_done": 0, "elapsed_s": 0.0,
                 "lbfgs_config": {}, "batch_sizes": {}}

    # ── 6. Paper-based thermal metrics ────────────────────────────────────────
    compare_fn = problem.make_paper_comparison_fn()
    thermal = compare_fn(model)

    # ── 7. Save refined model ─────────────────────────────────────────────────
    model_path = out_dir / "model_lbfgs.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n  Saved: {model_path}")

    # ── 8. Save loss histories ─────────────────────────────────────────────────
    log_path = out_dir / "refinement_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "optimizer":          opt_name,
            "adam_loss_history":  adam_res["loss_history"],
            "lbfgs_loss_history": lbfgs_res.get("loss_history", []),
            "adam_elapsed_s":     adam_res["elapsed_s"],
            "lbfgs_elapsed_s":    lbfgs_res["elapsed_s"],
        }, f, indent=2)

    # ── 9. Comparison JSON ────────────────────────────────────────────────────
    level1_tm = (level1_results.get("paper_baseline") or {}).get("thermal_metrics") or {}
    comparison = {
        "optimizer":    opt_name,
        "architecture": arch_cfg,
        "n_params":     n_params,
        "level1_adam_only": {
            "L2_rel":    level1_tm.get("L2_rel"),
            "MAE_C":     level1_tm.get("MAE_C"),
            "MaxErr_C":  level1_tm.get("MaxErr_C"),
            "adam_epochs": level1_results.get("adam_epochs"),
        },
        "level5_adam_lbfgs": {
            "L2_rel":           thermal.get("L2_rel"),
            "MAE_C":            thermal.get("MAE_C"),
            "MaxErr_C":         thermal.get("MaxErr_C"),
            "adam_l2":          adam_res["l2_final"],
            "lbfgs_l2_before":  lbfgs_res["l2_before"],
            "lbfgs_l2_after":   lbfgs_res["l2_after"],
            "lbfgs_loss_before": lbfgs_res["loss_before"],
            "lbfgs_loss_after":  lbfgs_res["loss_after"],
            "lbfgs_iters":      lbfgs_res["iters_done"],
            "adam_time_s":      adam_res["elapsed_s"],
            "lbfgs_time_s":     lbfgs_res["elapsed_s"],
        },
        "training_config": {
            "adam_epochs":   args.adam_epochs,
            "adam_lr":       ADAM_LR_MAX,
            "lbfgs_config":  lbfgs_res["lbfgs_config"],
            "batch_sizes":   lbfgs_res["batch_sizes"],
        },
    }
    cmp_path = out_dir / "comparison.json"
    with open(cmp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Saved: {cmp_path}")
    print(f"  Saved: {log_path}")

    return comparison


def run_all(args: argparse.Namespace) -> None:
    t_start = time.time()
    opts = OPTIMIZERS if args.optimizer == "all" else [args.optimizer]

    all_results = {}
    for opt in opts:
        try:
            all_results[opt] = run_one(opt, args)
        except Exception as e:
            print(f"\n  [ERROR] {opt}: {e}")
            import traceback; traceback.print_exc()
            all_results[opt] = {"error": str(e)}

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  LEVEL 5 — Adam({args.adam_epochs}ep) + L-BFGS({args.max_iter}it) SUMMARY")
    print(f"{'='*68}")
    print(f"  {'Optimizer':<12}  {'L1 Adam-only L2':<17}  {'L5 Adam+LBFGS L2':<18}  {'Gain':<8}  {'Time'}")
    print(f"  {'─'*12}  {'─'*17}  {'─'*18}  {'─'*8}  {'─'*8}")
    for opt, res in all_results.items():
        if "error" in res:
            print(f"  {opt:<12}  ERROR: {res['error']}")
            continue
        l2_l1 = res["level1_adam_only"].get("L2_rel") or float("nan")
        l2_l5 = res["level5_adam_lbfgs"].get("L2_rel") or float("nan")
        gain  = l2_l1 / (l2_l5 + 1e-12) if l2_l5 else float("nan")
        t_a   = res["level5_adam_lbfgs"].get("adam_time_s", 0)
        t_l   = res["level5_adam_lbfgs"].get("lbfgs_time_s", 0)
        print(f"  {opt:<12}  {l2_l1:<17.4f}  {l2_l5:<18.4f}  {gain:<8.2f}×  {t_a+t_l:.0f}s")

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_BASE / "level5_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {summary_path}")
    print(f"  Total time: {time.time() - t_start:.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Level 5 — Adam (early stop) + L-BFGS hybrid training"
    )
    parser.add_argument("--optimizer",    choices=["bayesian","nsga2","nsga3","all"], default="all")
    parser.add_argument("--adam_epochs",  type=int,   default=ADAM_EPOCHS,
                        help=f"Adam training epochs with cosine LR (default: {ADAM_EPOCHS})")
    parser.add_argument("--max_iter",     type=int,   default=LBFGS_CONFIG["max_iter"])
    parser.add_argument("--history_size", type=int,   default=LBFGS_CONFIG["history_size"])
    parser.add_argument("--tol_grad",     type=float, default=1e-7,
                        help="Gradient tolerance (default: 1e-7, NAS-PINNS2 validated)")
    parser.add_argument("--tol_change",   type=float, default=1e-9,
                        help="Loss change tolerance (default: 1e-9)")
    parser.add_argument("--col_points",   type=int,   default=COL_POINTS)
    parser.add_argument("--ic_points",    type=int,   default=IC_POINTS)
    parser.add_argument("--bc_points",    type=int,   default=BC_POINTS)
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
