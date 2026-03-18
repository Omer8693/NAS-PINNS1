"""
ts_trainer.py — Pencere Bazlı Eğitim Döngüsü
===============================================
Her [t_n, t_{n+k}] penceresi için mini-PINN eğitimi.
IC olarak önceki adım sıcaklığı kullanılır (teacher forcing).
"""

import time
from typing import Callable

import torch
import torch.nn as nn

from src.ts_model import TimeStepperPINN

# Fiziksel sabitler — baseline_paper [2026]
K_THERMAL = 150.0    # [W/mK] ısıl iletkenlik
RHO_CP    = 2.4e6    # [J/m³K] hacimsel ısı kapasitesi
H_CONV    = 5000.0   # [W/m²K] yüzey taşınım katsayısı
T_WATER   = 20.0     # [°C] su sıcaklığı
X_MAX     = 1.3      # [m] alan genişliği
Y_MAX     = 0.6      # [m] alan yüksekliği

# Kayıp ağırlıkları
W_PDE      = 1.0
W_IC       = 10.0    # IC kaybı — ilk adım en kritik kısıt
W_BOUNDARY = 5.0


def _pde_loss(model: TimeStepperPINN,
              x: torch.Tensor, y: torch.Tensor,
              t_local: torch.Tensor, T_prev: torch.Tensor,
              dt: float) -> torch.Tensor:
    """
    Isı denklemi PDE kaybı.
    r = ρCp·∂T/∂t - K·(∂²T/∂x² + ∂²T/∂y²) = 0
    Ölçeklendirilmiş form: r / PDE_SCALE → O(1)
    """
    # Gradyan hesabı için requires_grad gerekli
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t_local.requires_grad_(True)
    T_p = T_prev.detach()   # IC sabit — sadece model parametrelerine gradyan

    T_pred = model(x, y, t, T_p, dt)

    # Birinci türevler
    grads = torch.autograd.grad(T_pred.sum(), [x, y, t], create_graph=True)
    T_x, T_y, T_t = grads[0], grads[1], grads[2] / dt   # zincir kuralı: dt

    # İkinci türevler
    T_xx = torch.autograd.grad(T_x.sum(), x, create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y.sum(), y, create_graph=True)[0]

    residual = RHO_CP * T_t - K_THERMAL * (T_xx + T_yy)

    # PDE ölçeği: ρCp·ΔT/T_END ≈ 4.16e7 W/m³
    pde_scale = RHO_CP * 520.0 / 30.0
    return torch.mean((residual / pde_scale) ** 2)


def _ic_loss(model: TimeStepperPINN,
             x: torch.Tensor, y: torch.Tensor,
             T_prev: torch.Tensor, dt: float) -> torch.Tensor:
    """
    IC kaybı: t_local=0 → T = T_prev olmalı.
    T_prev doğrudan bilindiğinden bu kısıt güçlü bir çapa.
    """
    t_zero  = torch.zeros_like(x)
    T_pred  = model(x, y, t_zero, T_prev, dt)
    # IC ölçeği: 520°C
    return torch.mean(((T_pred - T_prev) / 520.0) ** 2)


def _boundary_loss(model: TimeStepperPINN,
                   t_local: torch.Tensor, T_prev_mean: float,
                   dt: float, n_bc: int,
                   device: torch.device) -> torch.Tensor:
    """
    Robin BC kaybı: K·∂T/∂n + h·(T - T_water) = 0
    Dört kenar için uygulanır.
    BC_SCALE = (h/K)·ΔT ≈ 17333 °C/m
    """
    bc_scale = (H_CONV / K_THERMAL) * 520.0   # ≈ 17333

    losses = []
    T_prev_bc = torch.full((n_bc, 1), T_prev_mean, device=device)

    for side in ["left", "right", "bottom", "top"]:
        t_bc = torch.rand(n_bc, 1, device=device) * dt

        if side == "left":
            x_bc = torch.zeros(n_bc, 1, device=device).requires_grad_(True)
            y_bc = torch.rand(n_bc, 1, device=device) * Y_MAX
            T_bc = model(x_bc, y_bc, t_bc, T_prev_bc, dt)
            dT_dn = torch.autograd.grad(T_bc.sum(), x_bc, create_graph=True)[0]
            normal_sign = -1.0   # dış normal x- yönü

        elif side == "right":
            x_bc = torch.full((n_bc, 1), X_MAX, device=device).requires_grad_(True)
            y_bc = torch.rand(n_bc, 1, device=device) * Y_MAX
            T_bc = model(x_bc, y_bc, t_bc, T_prev_bc, dt)
            dT_dn = torch.autograd.grad(T_bc.sum(), x_bc, create_graph=True)[0]
            normal_sign = 1.0

        elif side == "bottom":
            x_bc = torch.rand(n_bc, 1, device=device) * X_MAX
            y_bc = torch.zeros(n_bc, 1, device=device).requires_grad_(True)
            T_bc = model(x_bc, y_bc, t_bc, T_prev_bc, dt)
            dT_dn = torch.autograd.grad(T_bc.sum(), y_bc, create_graph=True)[0]
            normal_sign = -1.0

        else:   # top
            x_bc = torch.rand(n_bc, 1, device=device) * X_MAX
            y_bc = torch.full((n_bc, 1), Y_MAX, device=device).requires_grad_(True)
            T_bc = model(x_bc, y_bc, t_bc, T_prev_bc, dt)
            dT_dn = torch.autograd.grad(T_bc.sum(), y_bc, create_graph=True)[0]
            normal_sign = 1.0

        # Robin BC normalize rezidüsü
        robin = (normal_sign * dT_dn + (H_CONV / K_THERMAL) * (T_bc - T_WATER)) / bc_scale
        losses.append(torch.mean(robin ** 2))

    return sum(losses) / len(losses)


def train_window(model: TimeStepperPINN,
                 T_prev_fn: Callable,
                 t_start: float,
                 t_end: float,
                 n_domain: int = 1000,
                 n_bc: int = 200,
                 n_epochs: int = 500,
                 lr: float = 1e-3,
                 lr_min: float = 0.0,
                 lbfgs_iters: int = 0,
                 device: torch.device = torch.device("cpu")) -> tuple:
    """
    Single time window [t_start, t_end] training.

    T_prev_fn(x, y) → tensor(N,1): previous step temperature (teacher forcing).

    lr_min > 0    : cosine annealing LR from lr → lr_min over n_epochs
    lbfgs_iters>0 : L-BFGS refinement after Adam (Level 5 idea applied per window)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr_min)
        if lr_min > 0 else None
    )
    dt = t_end - t_start

    history = {"pde": [], "ic": [], "bc": [], "total": []}
    t0 = time.time()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        x = torch.rand(n_domain, 1, device=device) * X_MAX
        y = torch.rand(n_domain, 1, device=device) * Y_MAX
        t = torch.rand(n_domain, 1, device=device) * dt

        T_prev = T_prev_fn(x, y).detach()
        T_prev_mean = float(T_prev.mean().item())

        loss_pde = _pde_loss(model, x, y, t, T_prev, dt)
        loss_ic  = _ic_loss(model, x, y, T_prev, dt)
        loss_bc  = _boundary_loss(model, t, T_prev_mean, dt, n_bc, device)

        loss = W_PDE * loss_pde + W_IC * loss_ic + W_BOUNDARY * loss_bc
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if epoch % 100 == 0:
            history["pde"].append(loss_pde.item())
            history["ic"].append(loss_ic.item())
            history["bc"].append(loss_bc.item())
            history["total"].append(loss.item())

    # ── L-BFGS refinement phase (Level 5 idea applied per window) ────────────
    if lbfgs_iters > 0:
        x_f = torch.rand(n_domain, 1, device=device) * X_MAX
        y_f = torch.rand(n_domain, 1, device=device) * Y_MAX
        t_f = torch.rand(n_domain, 1, device=device) * dt
        T_prev_f     = T_prev_fn(x_f, y_f).detach()
        T_prev_mean_f = float(T_prev_f.mean().item())

        lbfgs_opt = torch.optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_iters,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs_opt.zero_grad()
            l_pde = _pde_loss(model, x_f, y_f, t_f, T_prev_f, dt)
            l_ic  = _ic_loss(model, x_f, y_f, T_prev_f, dt)
            l_bc  = _boundary_loss(model, t_f, T_prev_mean_f, dt, n_bc, device)
            l     = W_PDE * l_pde + W_IC * l_ic + W_BOUNDARY * l_bc
            l.backward()
            return l

        lbfgs_opt.step(closure)

    elapsed = time.time() - t0
    return model, {"history": history, "train_time": elapsed, "final_loss": loss.item()}
