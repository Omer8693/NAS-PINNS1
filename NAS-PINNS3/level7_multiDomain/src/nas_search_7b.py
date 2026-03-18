"""
nas_search_7b.py — NAS for Multi-Domain Poisson (Level 7B)
===========================================================
Architecture search space:
  n_layers   : 2 – 5
  neurons    : 20 – 150
  activation : tanh | relu | gelu

Three optimizers: Bayesian  |  NSGA-II  |  NSGA-III
Each returns the best (config, model, metrics).
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from level7_multiDomain.src.domains import BaseDomain
from level7_multiDomain.src.poisson_pinn import PoissonNet, train

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Search space ──────────────────────────────────────────────────────────────

LAYER_CHOICES      = [2, 3, 4, 5]
NEURON_CHOICES     = [20, 30, 40, 50, 64, 80, 100, 128, 150]
ACTIVATION_CHOICES = ["tanh", "relu", "gelu"]

# Training budget per candidate
EVAL_EPOCHS   = 1000   # quick eval during search
FINAL_EPOCHS  = 5000   # final training of best architecture


def _build_model(n_layers: int, neurons: int, activation: str) -> PoissonNet:
    return PoissonNet(hidden_sizes=[neurons] * n_layers, activation=activation)


def _evaluate_candidate(n_layers: int, neurons: int, activation: str,
                         domain: BaseDomain, seed: int = 0) -> float:
    """Train a candidate for EVAL_EPOCHS and return validation L2."""
    model = _build_model(n_layers, neurons, activation)
    res   = train(model, domain, n_epochs=EVAL_EPOCHS, seed=seed)
    return res["final_val"]


# ── Bayesian Optimization ─────────────────────────────────────────────────────

def search_bayesian(domain: BaseDomain,
                    n_calls:   int = 20,
                    n_initial: int = 5,
                    seed:      int = 42) -> Dict:
    """
    Bayesian optimization over (n_layers, neurons, activation_idx).
    Uses bayes_opt if available, else falls back to random search.
    """
    try:
        from bayes_opt import BayesianOptimization
        _bayes_available = True
    except ImportError:
        _bayes_available = False

    best = {"val": float("inf"), "n_layers": 3, "neurons": 64, "activation": "tanh"}
    history = []

    if _bayes_available:
        def objective(n_layers_f, neurons_f, act_idx_f):
            nl  = LAYER_CHOICES[int(np.clip(round(n_layers_f), 0, len(LAYER_CHOICES)-1))]
            nw  = NEURON_CHOICES[int(np.clip(round(neurons_f),  0, len(NEURON_CHOICES)-1))]
            act = ACTIVATION_CHOICES[int(np.clip(round(act_idx_f), 0, len(ACTIVATION_CHOICES)-1))]
            val = _evaluate_candidate(nl, nw, act, domain, seed=seed)
            if val < best["val"]:
                best.update({"val": val, "n_layers": nl, "neurons": nw, "activation": act})
                print(f"    ★ Bayesian new best: L={nl} N={nw} act={act}  val={val:.5f}")
            history.append({"n_layers": nl, "neurons": nw, "activation": act, "val": val})
            return -val

        bo = BayesianOptimization(
            f       = objective,
            pbounds = {
                "n_layers_f": (0, len(LAYER_CHOICES)   - 1.001),
                "neurons_f":  (0, len(NEURON_CHOICES)  - 1.001),
                "act_idx_f":  (0, len(ACTIVATION_CHOICES) - 1.001),
            },
            random_state = seed, verbose = 0,
        )
        bo.maximize(init_points=n_initial, n_iter=max(1, n_calls - n_initial))
    else:
        # Random search fallback
        rng = np.random.default_rng(seed)
        for _ in range(n_calls):
            nl  = rng.choice(LAYER_CHOICES)
            nw  = rng.choice(NEURON_CHOICES)
            act = rng.choice(ACTIVATION_CHOICES)
            val = _evaluate_candidate(nl, nw, act, domain, seed=seed)
            if val < best["val"]:
                best.update({"val": val, "n_layers": nl, "neurons": nw, "activation": act})
            history.append({"n_layers": nl, "neurons": nw, "activation": act, "val": val})

    return {"best": best, "history": history}


# ── NSGA-II ───────────────────────────────────────────────────────────────────

def _pymoo_search(domain: BaseDomain, algo_name: str,
                  pop_size: int = 10, n_gen: int = 5,
                  seed: int = 42) -> Dict:
    """
    Single-objective NSGA-II/III implemented as (1+1) EA over discrete space
    using pymoo's genetic algorithm infrastructure.
    We use a simple GA with integer encoding.
    """
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.optimize import minimize

    n_layer_opts  = len(LAYER_CHOICES)
    n_neuron_opts = len(NEURON_CHOICES)
    n_act_opts    = len(ACTIVATION_CHOICES)

    class PoissonNASProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=3, n_obj=1, n_ieq_constr=0,
                xl=np.array([0, 0, 0]),
                xu=np.array([n_layer_opts-1, n_neuron_opts-1, n_act_opts-1]),
                vtype=int,
            )

        def _evaluate(self, x, out, *args, **kw):
            nl  = LAYER_CHOICES[int(np.clip(x[0], 0, n_layer_opts-1))]
            nw  = NEURON_CHOICES[int(np.clip(x[1], 0, n_neuron_opts-1))]
            act = ACTIVATION_CHOICES[int(np.clip(x[2], 0, n_act_opts-1))]
            val = _evaluate_candidate(nl, nw, act, domain, seed=seed)
            out["F"] = [val]

    problem = PoissonNASProblem()

    if algo_name == "nsga2":
        algo = NSGA2(pop_size=pop_size)
    else:  # nsga3
        ref_dirs = get_reference_directions("energy", 1, pop_size)
        algo     = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)

    res = minimize(problem, algo,
                   termination=("n_gen", n_gen),
                   seed=seed, verbose=False)

    best_x   = res.X.flatten().astype(int) if res.X is not None else [1, 5, 0]
    best_val = float(res.F.min()) if res.F is not None else 999.0
    nl  = LAYER_CHOICES[int(np.clip(best_x[0], 0, n_layer_opts-1))]
    nw  = NEURON_CHOICES[int(np.clip(best_x[1], 0, n_neuron_opts-1))]
    act = ACTIVATION_CHOICES[int(np.clip(best_x[2], 0, n_act_opts-1))]

    best = {"val": best_val, "n_layers": nl, "neurons": nw, "activation": act}
    return {"best": best, "history": []}


def search_nsga2(domain: BaseDomain, pop_size: int = 8, n_gen: int = 4,
                 seed: int = 42) -> Dict:
    return _pymoo_search(domain, "nsga2", pop_size, n_gen, seed)


def search_nsga3(domain: BaseDomain, pop_size: int = 8, n_gen: int = 4,
                 seed: int = 42) -> Dict:
    return _pymoo_search(domain, "nsga3", pop_size, n_gen, seed)


# ── Full pipeline for one domain × optimizer ──────────────────────────────────

def run_one(domain_name: str, optimizer: str,
            out_dir: Path,
            n_calls: int = 20, pop_size: int = 8, n_gen: int = 4,
            seed:    int = 42) -> Dict:
    """
    Full NAS pipeline: search → final train → save model + metrics.
    Returns results dict.
    """
    from level7_multiDomain.src.domains import get_domain
    domain = get_domain(domain_name, seed=seed)

    print(f"\n{'='*62}")
    print(f"  Level 7B  |  Domain: {domain_name.upper():8s}  |  Optimizer: {optimizer.upper()}")
    print(f"  Search: n_calls={n_calls}  pop={pop_size}  n_gen={n_gen}")
    print(f"{'='*62}")

    # ── 1. Architecture search ────────────────────────────────────────────────
    t0 = time.time()
    if optimizer == "bayesian":
        search_res = search_bayesian(domain, n_calls=n_calls, seed=seed)
    elif optimizer == "nsga2":
        search_res = search_nsga2(domain, pop_size=pop_size, n_gen=n_gen, seed=seed)
    elif optimizer == "nsga3":
        search_res = search_nsga3(domain, pop_size=pop_size, n_gen=n_gen, seed=seed)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer}'")

    t_search = time.time() - t0
    best_cfg  = search_res["best"]
    print(f"\n  Best arch: {best_cfg['n_layers']}×{best_cfg['neurons']} {best_cfg['activation']}"
          f"  (val={best_cfg['val']:.5f}, search={t_search:.1f}s)")

    # ── 2. Final training of best architecture ─────────────────────────────
    print(f"\n  Final training: {FINAL_EPOCHS} epochs...")
    model = _build_model(best_cfg["n_layers"], best_cfg["neurons"],
                          best_cfg["activation"])
    # Optimizer-specific seed: ensures two optimizers finding the same arch
    # still produce distinct (non-identical) training runs.
    # Use a fixed offset table (not hash()) — Python randomizes string hashes per process.
    _OPT_OFFSET = {"bayesian": 0, "nsga2": 137, "nsga3": 271}
    opt_seed = seed + _OPT_OFFSET.get(optimizer, 0)
    train_res = train(model, domain, n_epochs=FINAL_EPOCHS, seed=opt_seed, verbose=True)

    final_val = train_res["final_val"]
    print(f"\n  Final val={final_val:.5f}  ({train_res['elapsed_s']:.1f}s)")

    # ── 3. Compute L2 on a large validation set ────────────────────────────
    N_FINAL_VAL = 5000
    domain_eval = get_domain(domain_name, seed=1234)
    xy_val_np   = domain_eval.sample_interior(N_FINAL_VAL)
    xy_val      = torch.tensor(xy_val_np, device=DEVICE)

    model.eval()
    with torch.no_grad():
        u_pred = model(xy_val).squeeze().cpu().numpy()

    if domain.has_exact:
        u_ref = domain.u_exact(xy_val_np)
        l2    = float(np.linalg.norm(u_pred - u_ref) /
                      (np.linalg.norm(u_ref) + 1e-12))
        mae   = float(np.mean(np.abs(u_pred - u_ref)))
    else:
        l2  = float(np.sqrt(np.mean(u_pred**2)))  # RMS of prediction
        mae = float(np.mean(np.abs(u_pred)))

    # ── 4. Save ───────────────────────────────────────────────────────────
    save_dir = out_dir / domain_name / optimizer
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_dir / "model_l7b.pt")

    result = {
        "domain":     domain_name,
        "optimizer":  optimizer,
        "best_arch":  best_cfg,
        "n_params":   sum(p.numel() for p in model.parameters()),
        "final_l2":   round(l2,  6),
        "final_mae":  round(mae, 6),
        "has_exact":  domain.has_exact,
        "search_s":   round(t_search, 1),
        "train_s":    round(train_res["elapsed_s"], 1),
        "loss_history":  [round(x, 8) for x in train_res["loss_history"]],
        "val_l2_history": train_res["val_l2_history"],
    }
    with open(save_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Saved → {save_dir}")
    return result
