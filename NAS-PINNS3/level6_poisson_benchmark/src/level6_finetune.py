"""
level6_finetune.py — Poisson-Assisted Fine-Tuning of Level 5 Model
====================================================================
Loads the best model from Level 5 (cosine LR Adam, 20K epochs) and
applies a short fine-tuning phase where an additional Poisson auxiliary
loss enforces spatial consistency at critical time slices.

Training schedule:
  Load Level 5 weights  →  Adam (5 000 epochs, lr=1e-5 → 1e-6)
  with loss = λ_heat·L_heat + λ_ic·L_ic + λ_bc·L_bc + λ_p(epoch)·L_poisson

  λ_p warm-up: 0 → LAMBDA_P over first WARMUP_EPOCHS steps
  (avoids destabilizing the pretrained model at the start)
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

import sys
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.pinn_network  import PINNNet
from src.config        import DEVICE
from problems.quenching import QuenchingProblem, CollocationSampler

from level6_poisson_benchmark.src.poisson_aux_loss import (
    poisson_auxiliary_loss, DEFAULT_T_SLICES, N_SPATIAL
)

# ── Paths ─────────────────────────────────────────────────────────────────────
LEVEL1_DIR = _ROOT / "level1_single_shot" / "results"
LEVEL5_DIR = _ROOT / "level5_refinement" / "results"
OUT_BASE   = _ROOT / "level6_poisson_benchmark" / "results"

# ── Fine-tuning hyper-parameters ──────────────────────────────────────────────
FINETUNE_EPOCHS = 5_000
LR_START        = 1e-5     # start from where Level 5 left off
LR_END          = 1e-6
LAMBDA_P        = 1.0      # final Poisson loss weight
WARMUP_EPOCHS   = 500      # epochs to ramp λ_p from 0 → LAMBDA_P

# ── Collocation sizes (same as Level 5) ──────────────────────────────────────
COL_POINTS = 2048
IC_POINTS  = 512
BC_POINTS  = 512
LOG_EVERY  = 200


def _load_arch(opt_name: str) -> dict:
    """Load architecture config from Level 1 results."""
    p = LEVEL1_DIR / opt_name / "best_arch.json"
    if not p.exists():
        raise FileNotFoundError(f"Level 1 arch not found: {p}")
    with open(p) as f:
        data = json.load(f)
    # best_arch.json may have a 'config' key or be the config itself
    cfg = data.get("config") or data.get("best_config") or data
    return cfg


def _load_level5_weights(opt_name: str) -> Path:
    """Return path to Level 5 saved model weights."""
    p = LEVEL5_DIR / opt_name / "model_lbfgs.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"Level 5 model not found: {p}\n"
            f"Run level5_refinement/main_level5.py --optimizer {opt_name} first."
        )
    return p


def _lambda_p(epoch: int) -> float:
    """Linear warm-up of Poisson loss weight."""
    if epoch >= WARMUP_EPOCHS:
        return LAMBDA_P
    return LAMBDA_P * (epoch / WARMUP_EPOCHS)


def run_one(opt_name: str,
            finetune_epochs: int = FINETUNE_EPOCHS,
            lr_start: float      = LR_START,
            lr_end: float        = LR_END,
            lambda_p: float      = LAMBDA_P,
            n_spatial: int       = N_SPATIAL,
            ) -> dict:
    """
    Fine-tune Level 5 model for `opt_name` optimizer with Poisson auxiliary loss.

    Returns result dict.
    """
    t_total = time.time()
    out_dir = OUT_BASE / opt_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Level 6 — Poisson-Assisted Fine-Tune  [{opt_name.upper()}]")
    print(f"  epochs={finetune_epochs}  lr={lr_start:.0e}→{lr_end:.0e}  λ_p={lambda_p}")
    print(f"{'='*62}")

    # ── 1. Load architecture and Level 5 weights ──────────────────────────────
    arch_cfg     = _load_arch(opt_name)
    weights_path = _load_level5_weights(opt_name)

    model = PINNNet(
        n_input      = arch_cfg.get("n_input",  3),
        n_output     = arch_cfg.get("n_output", 1),
        hidden_sizes = arch_cfg["neurons"],
        activation   = arch_cfg["activation"],
        residual     = arch_cfg.get("residual", False),
    ).to(DEVICE)

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE,
                                     weights_only=True))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Arch:   {arch_cfg['n_layers']}×{arch_cfg['neurons'][0]}"
          f" {arch_cfg['activation']}  ({n_params:,} params)")
    print(f"  Loaded: {weights_path}")

    # ── 2. Problem setup ───────────────────────────────────────────────────────
    problem  = QuenchingProblem(time_mode="full")
    loss_fn  = problem.make_loss_fn()
    val_fn   = problem.make_val_fn(n_val=1000)
    sampler  = CollocationSampler(
        n_domain   = COL_POINTS,
        n_boundary = BC_POINTS,
        n_initial  = IC_POINTS,
    )

    # ── 3. Level 5 baseline metric ────────────────────────────────────────────
    l2_before = val_fn(model)
    compare_fn = problem.make_paper_comparison_fn()
    thermal_before = compare_fn(model)
    print(f"\n  Level 5 baseline:  L2={l2_before:.5f}"
          f"  MAE={thermal_before.get('MAE_C', 0):.1f}°C")

    # ── 4. Fine-tuning ─────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=finetune_epochs, eta_min=lr_end
    )

    loss_history    = []
    poisson_history = []
    l2_history      = []
    t0              = time.time()

    model.train()
    print(f"\n  Fine-tuning with Poisson auxiliary loss...")
    for epoch in range(finetune_epochs):
        # Standard quenching batch
        batch = {
            "domain":   sampler.sample_domain(),
            "initial":  sampler.sample_initial(),
        }
        bc_pts, normals = sampler.sample_boundary()
        batch["boundary"] = bc_pts
        batch["normals"]  = normals

        optimizer.zero_grad()

        # Standard quenching loss
        L_heat, _ = loss_fn(model, batch)

        # Poisson auxiliary loss (with warm-up)
        lp = _lambda_p(epoch) * lambda_p
        if lp > 0:
            L_poisson = poisson_auxiliary_loss(
                model, t_slices=DEFAULT_T_SLICES, n_spatial=n_spatial
            )
        else:
            L_poisson = torch.tensor(0.0, device=DEVICE)

        loss = L_heat + lp * L_poisson
        loss.backward()
        optimizer.step()
        scheduler.step()

        v  = float(loss.item())
        vp = float(L_poisson.item()) if lp > 0 else 0.0
        if np.isfinite(v):
            loss_history.append(v)
            poisson_history.append(vp)

        if (epoch + 1) % LOG_EVERY == 0:
            l2     = val_fn(model)
            lr_now = optimizer.param_groups[0]["lr"]
            l2_history.append(l2)
            print(f"    epoch {epoch+1:5d}/{finetune_epochs}  "
                  f"loss={v:.4e}  L_p={vp:.4e}  L2={l2:.5f}  "
                  f"λ_p={lp:.3f}  lr={lr_now:.2e}")
            model.train()

    elapsed = time.time() - t0

    # ── 5. Final metrics ───────────────────────────────────────────────────────
    l2_after      = val_fn(model)
    thermal_after = compare_fn(model)

    print(f"\n  Fine-tune done: {elapsed:.0f}s")
    print(f"  L2  :  {l2_before:.5f}  →  {l2_after:.5f}"
          f"  ({100*(l2_before-l2_after)/l2_before:+.1f}%)")
    print(f"  MAE :  {thermal_before.get('MAE_C',0):.1f}°C"
          f"  →  {thermal_after.get('MAE_C',0):.1f}°C")

    # ── 6. Save model ──────────────────────────────────────────────────────────
    model_path = out_dir / "model_l6.pt"
    torch.save(model.state_dict(), model_path)
    print(f"  Saved: {model_path}")

    # ── 7. Save result JSON ────────────────────────────────────────────────────
    result = {
        "optimizer":     opt_name,
        "architecture":  arch_cfg,
        "n_params":      n_params,
        "level5_before": {
            "L2_rel": l2_before,
            "MAE_C":  thermal_before.get("MAE_C"),
            "MaxErr_C": thermal_before.get("MaxErr_C"),
        },
        "level6_after": {
            "L2_rel": l2_after,
            "MAE_C":  thermal_after.get("MAE_C"),
            "MaxErr_C": thermal_after.get("MaxErr_C"),
        },
        "improvement": {
            "L2_abs":  round(l2_before - l2_after, 6),
            "L2_pct":  round(100 * (l2_before - l2_after) / (l2_before + 1e-10), 2),
        },
        "training": {
            "finetune_epochs": finetune_epochs,
            "lr_start":        lr_start,
            "lr_end":          lr_end,
            "lambda_p":        lambda_p,
            "warmup_epochs":   WARMUP_EPOCHS,
            "t_slices":        DEFAULT_T_SLICES,
            "n_spatial":       n_spatial,
            "elapsed_s":       elapsed,
        },
        "loss_history":    loss_history,
        "poisson_history": poisson_history,
        "l2_history":      l2_history,
        "total_time_s":    time.time() - t_total,
    }

    result_path = out_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {result_path}")

    result["model"] = model   # attach model for downstream use
    return result
