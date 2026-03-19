"""
lbfgs_refiner.py — Post-NAS L-BFGS Refinement
================================================
Loads a Level-1 Adam-trained model and applies L-BFGS fine-tuning.

Hyperparameters selected from multi-folder analysis:
  Source              max_iter  history_size  tol_grad  line_search
  NAS-PINNS2 quench   5000      20            1e-7      strong_wolfe  (best for quenching)
  FEM-PINNS Poisson   1500      50            1e-12     strong_wolfe  (strictest convergence)
  NAS-PINN Burgers    3000      100           default   strong_wolfe

→ Level 5: max_iter=5000, history_size=50, tol_grad=1e-12, tol_change=machine_eps
"""

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.pinn_network import PINNNet
from src.config import DEVICE

# ── Validated L-BFGS hyperparameters ─────────────────────────────────────────
LBFGS_CONFIG = {
    "lr":               1.0,
    "max_iter":         5000,
    "history_size":     50,           # FEM-PINNS: more curvature history → better convergence
    "tolerance_grad":   1e-12,        # FEM-PINNS: strict — don't stop early
    "tolerance_change": float(np.finfo(float).eps),   # machine epsilon
    "line_search_fn":   "strong_wolfe",
}


def load_level1_model(opt_name: str,
                      results_dir: str = "level1_single_shot/results") -> tuple:
    """
    Load Level 1 architecture + weights for a given optimizer.

    Returns: (model, config_dict, results_dict)
    """
    base = Path(results_dir) / opt_name

    arch_path    = base / "best_arch.json"
    weights_path = base / "model.pt"
    results_path = base / "results.json"

    if not arch_path.exists():
        raise FileNotFoundError(f"best_arch.json not found: {arch_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"model.pt not found: {weights_path}")

    with open(arch_path) as f:
        cfg = json.load(f)
    with open(results_path) as f:
        results = json.load(f)

    # Rebuild network with same architecture
    model = PINNNet(
        n_input      = cfg.get("n_input",  3),
        n_output     = cfg.get("n_output", 1),
        hidden_sizes = cfg["neurons"],
        activation   = cfg["activation"],
        residual     = cfg.get("residual", False),
    ).to(DEVICE)

    state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)

    return model, cfg, results


class LBFGSRefiner:
    """
    Apply L-BFGS fine-tuning to an Adam-trained PINN.

    Usage:
        refiner = LBFGSRefiner(model, loss_fn, val_fn)
        result  = refiner.run(fixed_batch)
    """

    def __init__(self,
                 model:       nn.Module,
                 loss_fn:     Callable,
                 val_fn:      Optional[Callable] = None,
                 max_iter:    int   = LBFGS_CONFIG["max_iter"],
                 history_size: int  = LBFGS_CONFIG["history_size"],
                 tol_grad:    float = LBFGS_CONFIG["tolerance_grad"],
                 tol_change:  float = LBFGS_CONFIG["tolerance_change"]):

        self.model    = model
        self.loss_fn  = loss_fn
        self.val_fn   = val_fn
        self._cfg     = {
            "max_iter":         max_iter,
            "history_size":     history_size,
            "tolerance_grad":   tol_grad,
            "tolerance_change": tol_change,
        }

        self.optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr               = LBFGS_CONFIG["lr"],
            max_iter         = max_iter,
            history_size     = history_size,
            tolerance_grad   = tol_grad,
            tolerance_change = tol_change,
            line_search_fn   = LBFGS_CONFIG["line_search_fn"],
        )

        self._loss_history: list = []

    def run(self, fixed_batch: dict) -> Dict:
        """
        Run L-BFGS refinement on fixed_batch.

        fixed_batch must be generated ONCE before this call — never
        re-sampled inside the closure (would break quasi-Newton updates).

        Returns metrics dict with before/after L2 and loss history.
        """
        # ── Before-refinement snapshot ────────────────────────────────────────
        l2_before = self._evaluate(silent=False, label="before L-BFGS")
        loss_before, _ = self.loss_fn(self.model, fixed_batch)
        loss_before = float(loss_before.item())

        print(f"\n{'─'*56}")
        print(f"  L-BFGS REFINEMENT  "
              f"max_iter={self._cfg['max_iter']}  "
              f"history={self._cfg['history_size']}  "
              f"tol_grad={self._cfg['tolerance_grad']:.0e}")
        print(f"  col={fixed_batch['domain'].shape[0]}  "
              f"bc={fixed_batch['boundary'].shape[0]}  "
              f"ic={fixed_batch['initial'].shape[0]}")
        print(f"{'─'*56}")

        # ── Closure (deterministic — fixed_batch never changes) ───────────────
        def closure():
            self.optimizer.zero_grad()
            loss, _ = self.loss_fn(self.model, fixed_batch)
            loss.backward()
            v = float(loss.item())
            if np.isfinite(v):
                self._loss_history.append(v)
            return loss

        # ── Single step() — internally runs up to max_iter evaluations ────────
        t0 = time.time()
        try:
            self.optimizer.step(closure)
        except Exception as exc:
            print(f"  [L-BFGS] step() raised: {exc}")
        elapsed = time.time() - t0

        # ── After-refinement evaluation ───────────────────────────────────────
        l2_after = self._evaluate(silent=False, label="after  L-BFGS")
        loss_after = float(self._loss_history[-1]) if self._loss_history else float("nan")

        iters = len(self._loss_history)
        print(f"\n  L-BFGS complete: {elapsed:.1f}s  |  iters={iters}")
        print(f"  Loss:  {loss_before:.4e}  →  {loss_after:.4e}")
        if l2_before is not None and l2_after is not None:
            ratio = l2_before / (l2_after + 1e-12)
            print(f"  L2:    {l2_before:.4f}   →  {l2_after:.4f}   ({ratio:.1f}× improvement)")

        return {
            "lbfgs_config":    self._cfg,
            "batch_sizes": {
                "col": fixed_batch["domain"].shape[0],
                "bc":  fixed_batch["boundary"].shape[0],
                "ic":  fixed_batch["initial"].shape[0],
            },
            "l2_before":       l2_before,
            "l2_after":        l2_after,
            "loss_before":     loss_before,
            "loss_after":      loss_after,
            "iters_done":      iters,
            "elapsed_s":       elapsed,
            "loss_history":    self._loss_history,
        }

    def _evaluate(self, silent: bool = False, label: str = "") -> Optional[float]:
        if self.val_fn is None:
            return None
        self.model.eval()
        l2 = float(self.val_fn(self.model))
        self.model.train()
        if not silent:
            print(f"  L2 {label}: {l2:.6f}")
        return l2
