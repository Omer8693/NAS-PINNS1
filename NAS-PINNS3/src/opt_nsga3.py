"""
opt_nsga3.py — NSGA-III Mimari Arama
======================================
NAS-PINNS1 opt_nsga3.py + Pymoo_NSGA3.py parametreleriyle hizalanmış:
  ref_partitions=10, pop=max(pop_size, len(ref_dirs))
  SBX: prob=1.0, eta=30
  PM:  eta=20
  Das-Dennis referans yönleri: 2 amaç, 10 bölüm → 11 yön.

Kullanım:
    from src.opt_nsga3 import run_nsga3
    pareto = run_nsga3(train_fn, pop_size=24, n_gen=16)
"""

import time
from typing import Callable, List

import numpy as np

from src.config import LAYERS_MIN, LAYERS_MAX, NEURONS_MIN, NEURONS_MAX
from src.arch_search import PINNArchProblem, decode_x_to_config

try:
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.optimize import minimize as pymoo_minimize
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False


def run_nsga3(train_fn:       Callable,
              pop_size:       int = 24,
              n_gen:          int = 16,
              n_epochs:       int = 300,
              seed:           int = 42,
              n_input:        int = 3,
              n_output:       int = 1,
              objective_mode: str = "balanced") -> List[dict]:
    """
    NSGA-III ile PINN mimari arama — NAS-PINNS1 parametreleri.

    NAS-PINNS1 opt_nsga3.py + Pymoo_NSGA3.py:
      ref_partitions=10, pop=max(pop_size, len(ref_dirs))
      SBX: prob=1.0, eta=30  (Pymoo_NSGA3.py referans değerleri)
      PM:  eta=20
    Das-Dennis referans yönleri: 2 amaç, 10 bölüm → 11 yön.
    """
    if not PYMOO_AVAILABLE:
        raise ImportError("pymoo gerekli: pip install pymoo")

    # NAS-PINNS1: ref_partitions=10, pop=max(pop_size, len(ref_dirs))
    ref_dirs   = get_reference_directions("das-dennis", 2, n_partitions=10)
    actual_pop = max(pop_size, len(ref_dirs))

    print(f"\n{'='*55}")
    print(f"  NSGA-III Architecture Search  [NAS-PINNS1]")
    print(f"  Pop: {actual_pop}  |  Gens: {n_gen}  |  Ref dirs: {len(ref_dirs)}")
    print(f"  SBX: prob=1.0 eta=30  |  PM: eta=20  (Pymoo_NSGA3 referans)")
    print(f"{'='*55}")

    problem   = PINNArchProblem(train_fn, n_epochs, n_input, n_output, objective_mode)
    algorithm = NSGA3(
        ref_dirs             = ref_dirs,
        pop_size             = actual_pop,
        sampling             = FloatRandomSampling(),
        crossover            = SBX(prob=1.0, eta=30, repair=RoundingRepair()),
        mutation             = PM(eta=20,    repair=RoundingRepair()),
        eliminate_duplicates = True,
    )

    t0     = time.time()
    result = pymoo_minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=True)
    elapsed = time.time() - t0

    print(f"\n  NSGA-III completed: {elapsed:.1f}s  |  Pareto: {len(result.X)} solutions")

    pareto = []
    for x, f in zip(result.X, result.F):
        cfg = decode_x_to_config(x, n_input, n_output)
        cfg["objectives"] = {"l2": float(f[0]), "n_params": float(f[1] * 1e4)}
        pareto.append(cfg)
    pareto.sort(key=lambda c: c["objectives"]["l2"])
    return pareto
