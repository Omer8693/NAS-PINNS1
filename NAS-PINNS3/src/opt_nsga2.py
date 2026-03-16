"""
opt_nsga2.py — NSGA-II Mimari Arama
=====================================
NAS-PINNS1 opt_nsga2.py parametreleriyle hizalanmış:
  pop_size=24, n_gen=16
  SBX: prob=0.9, eta=15
  PM:  eta=20

Kullanım:
    from src.opt_nsga2 import run_nsga2
    pareto = run_nsga2(train_fn, pop_size=24, n_gen=16)
"""

import time
from typing import Callable, List

import numpy as np

from src.config import LAYERS_MIN, LAYERS_MAX, NEURONS_MIN, NEURONS_MAX
from src.arch_search import PINNArchProblem, decode_x_to_config

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.optimize import minimize as pymoo_minimize
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False


def run_nsga2(train_fn:       Callable,
              pop_size:       int = 24,
              n_gen:          int = 16,
              n_epochs:       int = 300,
              seed:           int = 42,
              n_input:        int = 3,
              n_output:       int = 1,
              objective_mode: str = "balanced") -> List[dict]:
    """
    NSGA-II ile PINN mimari arama — NAS-PINNS1 parametreleri.

    NAS-PINNS1 opt_nsga2.py:
      pop_size=24, n_gen=16
      SBX: prob=0.9, eta=15
      PM:  eta=20
    Arama uzayı: layers [3,6], neurons [64,160].
    """
    if not PYMOO_AVAILABLE:
        raise ImportError("pymoo gerekli: pip install pymoo")

    print(f"\n{'='*55}")
    print(f"  NSGA-II Architecture Search  [NAS-PINNS1]")
    print(f"  Pop: {pop_size}  |  Gens: {n_gen}  "
          f"|  L: [{LAYERS_MIN},{LAYERS_MAX}]  N: [{NEURONS_MIN},{NEURONS_MAX}]")
    print(f"  SBX: prob=0.9 eta=15  |  PM: eta=20")
    print(f"{'='*55}")

    problem   = PINNArchProblem(train_fn, n_epochs, n_input, n_output, objective_mode)
    algorithm = NSGA2(
        pop_size             = pop_size,
        sampling             = FloatRandomSampling(),
        crossover            = SBX(prob=0.9, eta=15, repair=RoundingRepair()),
        mutation             = PM(eta=20,    repair=RoundingRepair()),
        eliminate_duplicates = True,
    )

    t0     = time.time()
    result = pymoo_minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=True)
    elapsed = time.time() - t0

    print(f"\n  NSGA-II completed: {elapsed:.1f}s  |  Pareto: {len(result.X)} solutions")

    pareto = []
    for x, f in zip(result.X, result.F):
        cfg = decode_x_to_config(x, n_input, n_output)
        cfg["objectives"] = {"l2": float(f[0]), "n_params": float(f[1] * 1e4)}
        pareto.append(cfg)
    pareto.sort(key=lambda c: c["objectives"]["l2"])
    return pareto
