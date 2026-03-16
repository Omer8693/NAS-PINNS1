"""
adaptive_skip.py — Adaptif Zaman Adımı Atlama Kararı
======================================================
PDE rezidüsü büyükse FEM çalıştır, küçükse PINN ile devam et.
Karar mekanizması Level 2 NAS sonucundaki eşikle kalibre edilir.

Kullanım:
    scheduler = AdaptiveSkipScheduler(threshold=0.1, max_skip=6)
    use_fem   = scheduler.decide(model, x, y, t_local, T_prev, dt)
"""

import torch
from src.fem_interface import FEMCheckpoint

# Fiziksel sabitler — baseline_paper [2026]
K_THERMAL = 150.0   # [W/mK]
RHO_CP    = 2.4e6   # [J/m³K]
PDE_SCALE = RHO_CP * 520.0 / 30.0   # ≈ 4.16e7 W/m³ — normalize için


def compute_pde_residual(model,
                         x:       torch.Tensor,
                         y:       torch.Tensor,
                         t_local: torch.Tensor,
                         T_prev:  torch.Tensor,
                         dt:      float) -> float:
    """
    PINN'in mevcut penceredeki PDE rezidüsünü hesapla.
    Döndürür: normalize rezidü normu (skaler, float)

    Büyük rezidü → PINN fizikten sapıyor → FEM gerekli.
    Küçük rezidü → PINN yeterince doğru → adımı atla.
    """
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t_local.requires_grad_(True)

    T_pred = model(x, y, t, T_prev.detach(), dt)

    grads = torch.autograd.grad(T_pred.sum(), [x, y, t], create_graph=True)
    T_x, T_y, T_t_raw = grads
    T_t = T_t_raw / dt   # zincir kuralı

    T_xx = torch.autograd.grad(T_x.sum(), x, create_graph=False, retain_graph=True)[0]
    T_yy = torch.autograd.grad(T_y.sum(), y, create_graph=False)[0]

    residual = RHO_CP * T_t - K_THERMAL * (T_xx + T_yy)
    norm_val = float(torch.mean((residual / PDE_SCALE) ** 2).item())
    return norm_val


class AdaptiveSkipScheduler:
    """
    Her adımda 'FEM mi PINN mi?' kararını verir.

    threshold  : normalize PDE rezidü eşiği (varsayılan 0.1)
    max_skip   : arka arkaya en fazla kaç adım atlanabilir (varsayılan 6)
    """

    def __init__(self, threshold: float = 0.1, max_skip: int = 6):
        self.threshold   = threshold
        self.max_skip    = max_skip
        self._skip_count = 0   # arka arkaya atlanmış adım sayacı
        self.history     = []  # {t, residual, decision} geçmişi

    def decide(self,
               model,
               x:       torch.Tensor,
               y:       torch.Tensor,
               t_local: torch.Tensor,
               T_prev:  torch.Tensor,
               dt:      float,
               t_now:   float = 0.0) -> bool:
        """
        FEM mi kullanılmalı?
        True  → FEM çalıştır (düzeltme adımı)
        False → PINN ile atla
        """
        residual = compute_pde_residual(model, x, y, t_local, T_prev, dt)

        # Rezidü eşiği aşıldı veya max_skip doldu → FEM
        if residual > self.threshold or self._skip_count >= self.max_skip:
            decision = "fem"
            self._skip_count = 0
        else:
            decision = "pinn"
            self._skip_count += 1

        self.history.append({"t": t_now, "residual": residual, "decision": decision})
        return decision == "fem"

    def summary(self) -> dict:
        """
        Çalışma özeti: kaç adım FEM, kaç adım PINN kullanıldı.
        """
        fem_steps  = sum(1 for h in self.history if h["decision"] == "fem")
        pinn_steps = sum(1 for h in self.history if h["decision"] == "pinn")
        total      = len(self.history)
        return {
            "total_steps":   total,
            "fem_steps":     fem_steps,
            "pinn_steps":    pinn_steps,
            "skip_rate_pct": round(100.0 * pinn_steps / total, 1) if total > 0 else 0.0,
        }
