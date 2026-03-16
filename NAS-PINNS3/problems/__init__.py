"""
problems/ — NAS-PINN Benchmark Problem Kütüphanesi
===================================================
Wang & Zhong (2023) NAS-PINN paper'ındaki benchmark denklemler.

Her problem modülü aynı arayüzü sunar:
  problem = BurgersProblem()
  loss_fn   = problem.make_loss_fn()
  val_fn    = problem.make_val_fn()
  batch_fn  = problem.make_batch_fn(scheduler=None)

Desteklenen problemler:
  burgers    — Burgers 1D
  poisson    — Poisson 2D
  allen_cahn — Allen-Cahn 1D+t
  quenching  — Mortensen 2026 quenching
"""

from .base import PINNProblem


def get_problem(name: str, **kwargs) -> "PINNProblem":
    """Problem adına göre uygun problem sınıfını döndür."""
    name = name.lower()
    if name == "burgers":
        from .burgers import BurgersProblem
        return BurgersProblem(**kwargs)
    elif name == "poisson":
        from .poisson import PoissonProblem
        return PoissonProblem(**kwargs)
    elif name == "allen_cahn":
        from .allen_cahn import AllenCahnProblem
        return AllenCahnProblem(**kwargs)
    elif name == "quenching":
        from .quenching import QuenchingProblem
        return QuenchingProblem(**kwargs)
    else:
        raise ValueError(
            f"Bilinmeyen problem: '{name}'. "
            f"Options: burgers, poisson, allen_cahn, quenching"
        )

__all__ = ["PINNProblem", "get_problem"]
