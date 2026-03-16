"""
opt_bayesian.py — Bayesian Optimization Mimari Arama
======================================================
NAS-PINNS1 opt_bayes.py ile hizalanmış:
  BayesianOptimization(f, pbounds, random_state, verbose=2)
  maximize(init_points=4, n_iter=12)
  Objective: -proxy_loss  (maximize eder → minimize eder)

Kullanım:
    from src.opt_bayesian import run_bayesian
    best_config = run_bayesian(train_fn, n_calls=16)
"""

import time
from copy import deepcopy
from typing import Callable

import numpy as np

from src.config import LAYERS_MIN, LAYERS_MAX, NEURONS_MIN, NEURONS_MAX
from src.arch_search import decode_x_to_config, evaluate_architecture

try:
    from bayes_opt import BayesianOptimization
    BAYES_OPT_AVAILABLE = True
except ImportError:
    BAYES_OPT_AVAILABLE = False


def run_bayesian(train_fn:  Callable,
                 n_calls:   int = 16,
                 n_initial: int = 4,
                 n_epochs:  int = 300,
                 seed:      int = 42,
                 n_input:   int = 3,
                 n_output:  int = 1) -> dict:
    """
    bayes_opt tabanlı Bayesian Optimization — NAS-PINNS1 opt_bayes.py ile hizalanmış.

    NAS-PINNS1 opt_bayes.py:
      BayesianOptimization(f, pbounds, random_state, verbose=2)
      maximize(init_points=4, n_iter=12)
      Objective: -proxy_loss (maximize eder → minimize eder)

    Arama uzayı: layers [3,6], neurons [64,160].
    """
    if not BAYES_OPT_AVAILABLE:
        raise ImportError(
            "bayesian-optimization gerekli: pip install bayesian-optimization"
        )

    n_iter = max(1, n_calls - n_initial)   # NAS-PINNS1: bo_iters=12

    print(f"\n{'='*55}")
    print(f"  Bayesian Optimization  [NAS-PINNS1 / bayes_opt]")
    print(f"  init_points: {n_initial}  |  n_iter: {n_iter}  |  total: {n_initial + n_iter}")
    print(f"{'='*55}")

    best = {"l2": float("inf"), "config": None}

    def objective(layers, neurons, activation_idx):
        """bayes_opt maximize eder — negatif L2 döndür (NAS-PINNS1 ile aynı).
        activation_idx ∈ [0,4]: 0=tanh, 1=sin, 2=swish, 3=relu, 4=gelu
        """
        x = np.array([layers, neurons, activation_idx])
        config = decode_x_to_config(x, n_input, n_output)
        res = evaluate_architecture(config, train_fn, n_epochs)
        if res["l2_error"] < best["l2"]:
            best["l2"]     = res["l2_error"]
            best["config"] = deepcopy(config)
            print(f"  → Yeni en iyi: L2={res['l2_error']:.6f}  "
                  f"L={config['n_layers']}  N={config['neurons'][0]}  "
                  f"act={config['activation']}")
        return -float(res["l2_error"])   # NAS-PINNS1: return -float(proxy_loss)

    t0 = time.time()
    optimizer = BayesianOptimization(
        f            = objective,
        pbounds      = {
            "layers":         (LAYERS_MIN,  LAYERS_MAX),
            "neurons":        (NEURONS_MIN, NEURONS_MAX),
            "activation_idx": (0.0, 4.0),
        },
        random_state = seed,
        verbose      = 2,
    )
    optimizer.maximize(init_points=n_initial, n_iter=n_iter)
    elapsed = time.time() - t0

    # En iyi konfigürasyonu döndür
    if best["config"] is None:
        p = optimizer.max["params"]
        best["config"] = decode_x_to_config(
            np.array([p["layers"], p["neurons"], p["activation_idx"]]),
            n_input, n_output
        )

    print(f"\n  Bayesian completed: {elapsed:.1f}s  |  Best L2: {best['l2']:.6f}")
    print(f"  En iyi mimari: L={best['config']['n_layers']}  "
          f"N={best['config']['neurons'][0]}")
    return best["config"]
