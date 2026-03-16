"""
PINNProblem — Tüm Benchmark Problemleri İçin Soyut Temel Sınıf
===============================================================
Her problem modülü bu sınıfı miras alır ve şu metotları uygular:
  make_loss_fn()   → loss_fn(model, batch) callable
  make_val_fn()    → val_fn(model) → float  (L2 hatası)
  make_batch_fn()  → get_batch(epoch) callable
  analytical_solution(coords) → Tensor

Bu sayede arch_search.py ve trainers.py problem-agnostik kalır.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional
import torch
import numpy as np


class PINNProblem(ABC):
    """
    Soyut PINN problem arayüzü.

    Alt sınıflar:
      name        : str   — problem adı (log ve dosya isimleri için)
      n_input     : int   — ağ girdi boyutu  (2=xt, 3=xyt, ...)
      n_output    : int   — ağ çıktı boyutu  (genellikle 1)
      domain_bounds : dict — {'t':(t0,t1), 'x':(x0,x1), ...}
    """

    name:          str
    n_input:       int
    n_output:      int
    domain_bounds: dict   # {'t': (0,1), 'x': (-1,1), ...}

    # ── Zorunlu metotlar ──────────────────────────────────────

    @abstractmethod
    def make_loss_fn(self) -> Callable:
        """loss_fn(model, batch) → (loss_tensor, details_dict)"""
        ...

    @abstractmethod
    def make_val_fn(self, n_val: int = 500) -> Callable:
        """val_fn(model) → float  (normalize L2 hatası)"""
        ...

    @abstractmethod
    def make_batch_fn(self,
                      n_domain:   int = 2000,
                      n_boundary: int = 200,
                      n_initial:  int = 200,
                      scheduler=None) -> Callable:
        """get_batch(epoch) → {'domain': Tensor, 'boundary': Tensor, 'initial': Tensor}"""
        ...

    @abstractmethod
    def analytical_solution(self, coords: torch.Tensor) -> torch.Tensor:
        """Referans / analitik çözüm — L2 hatası ve görselleştirme için."""
        ...

    # ── Ortak yardımcı metotlar ───────────────────────────────

    def l2_relative(self,
                    pred:  torch.Tensor,
                    exact: torch.Tensor) -> float:
        """L2 bağıl hata: ||pred - exact|| / ||exact||"""
        return (torch.norm(pred - exact) /
                (torch.norm(exact) + 1e-10)).item()

    def info(self) -> str:
        """Problem özeti."""
        return (f"{self.name.upper()}  |  "
                f"input={self.n_input}  output={self.n_output}  |  "
                f"domain={self.domain_bounds}")
