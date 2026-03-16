"""
Poisson Denklemi (2D) — Ortak Benchmark Problemi
=================================================
Tum optimizerlar icin sabitlenen benchmark problem.

PDE  : u_xx + u_yy = f(x,y),   (x,y) ∈ [-1,1]²
BC   : u = 0   (Dirichlet homojen, tum kenarlarda)

Analitik cozum:
  u(x,y) = sin(a1*pi*x) * sin(a2*pi*y)
  a1 = 1, a2 = 4

Kaynak terimi:
  f(x,y) = -[(a1*pi)^2 + (a2*pi)^2] * sin(a1*pi*x) * sin(a2*pi*y)

Kollokasyon:
  N_f = 10 000
  N_b =  1 000

NOT: Poisson zamansiz PDE'dir; temporal skip uygulanmaz.
"""

import torch
import numpy as np
from typing import Callable

from .base import PINNProblem

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# Bölüm 1 — Problem Sabitleri
# ──────────────────────────────────────────────────────────────

A1  = 1.0
A2  = 4.0

X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0

# Kayıp ağırlıkları — paper eşit ağırlık kullanıyor
W_PDE = 1.0
W_BC  = 1.0


# ──────────────────────────────────────────────────────────────
# Bölüm 2 — Analitik Çözüm ve Kaynak Terimi
# ──────────────────────────────────────────────────────────────

def _u_exact(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """u(x,y) = sin(a₁πx)·sin(a₂πy)"""
    return torch.sin(A1 * np.pi * x) * torch.sin(A2 * np.pi * y)


def _f_source(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Kaynak terimi:  f = -[(a1*pi)^2 + (a2*pi)^2] * sin(a1*pi*x) * sin(a2*pi*y)
    """
    coeff = -((A1 * np.pi)**2 + (A2 * np.pi)**2)
    return coeff * torch.sin(A1 * np.pi * x) * torch.sin(A2 * np.pi * y)


# ──────────────────────────────────────────────────────────────
# Bölüm 3 — PDE Rezidüeli
# ──────────────────────────────────────────────────────────────

def poisson_residual(net_out: torch.Tensor,
                     coords:  torch.Tensor) -> torch.Tensor:
    """
    Poisson PDE rezidüeli:  R = u_xx + u_yy - f(x,y)

    coords  : [N, 2]  — (x, y) sütunları, requires_grad=True
    net_out : [N, 1]  — ağın tahmin ettiği u değerleri
    """
    u     = net_out[:, 0:1]
    grads = torch.autograd.grad(
        u, coords,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]

    u_xx = torch.autograd.grad(
        u_x, coords,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        u_y, coords,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True
    )[0][:, 1:2]

    x = coords[:, 0:1]
    y = coords[:, 1:2]
    f = _f_source(x, y)

    return u_xx + u_yy - f


# ──────────────────────────────────────────────────────────────
# Bölüm 4 — PoissonProblem Sınıfı
# ──────────────────────────────────────────────────────────────

class PoissonProblem(PINNProblem):
    """
    Poisson 2D ortak benchmark problemi.

    Ağ: f(x, y) → u(x, y)   [2 girdi, 1 çıktı]

    Zamansız PDE — temporal skip uygulanmaz.
    """

    name          = "poisson"
    n_input       = 2
    n_output      = 1
    domain_bounds = {"x": (X_MIN, X_MAX), "y": (Y_MIN, Y_MAX)}

    def analytical_solution(self, coords: torch.Tensor) -> torch.Tensor:
        """
        u(x,y) = sin(a₁πx)·sin(a₂πy)
        coords : [N, 2]  (x, y) sütunları
        """
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        return _u_exact(x, y)

    def make_loss_fn(self) -> Callable:
        """
        Poisson kayıp fonksiyonu.
          L = λ_pde·||R_pde||² + λ_bc·||u_boundary - 0||²
        """
        def loss_fn(model, batch):
            # ── PDE rezidüeli ────────────────────────────────
            coords_f = batch["domain"]
            coords_f.requires_grad_(True)
            pred_f   = model(coords_f)
            r_pde    = poisson_residual(pred_f, coords_f)
            L_pde    = W_PDE * (r_pde ** 2).mean()

            # ── Sınır koşulu: u=0 tüm kenarlarda ────────────
            coords_b = batch["boundary"]
            pred_b   = model(coords_b)
            L_bc     = W_BC * (pred_b ** 2).mean()

            # Poisson zamansız → başlangıç koşulu yok
            # batch["initial"] varsa görmezden gel
            loss = L_pde + L_bc
            details = {
                "L_total":    loss.item(),
                "L_physics":  L_pde.item(),
                "L_boundary": L_bc.item(),
                "L_initial":  0.0,
            }
            return loss, details

        return loss_fn

    def make_val_fn(self, n_val: int = 2000) -> Callable:
        """
        Doğrulama: analitik çözüme karşı normalize L2 hatası.
        """
        rng = np.random.default_rng(seed=0)
        x_v = torch.tensor(
            rng.uniform(X_MIN, X_MAX, (n_val, 1)), dtype=torch.float32, device=DEVICE
        )
        y_v = torch.tensor(
            rng.uniform(Y_MIN, Y_MAX, (n_val, 1)), dtype=torch.float32, device=DEVICE
        )
        coords_v = torch.cat([x_v, y_v], dim=1)
        u_ref    = _u_exact(x_v, y_v)   # [n_val, 1]

        def val_fn(model) -> float:
            model.eval()
            with torch.no_grad():
                pred = model(coords_v)
            return (torch.norm(pred - u_ref) /
                    (torch.norm(u_ref) + 1e-10)).item()

        return val_fn

    def make_batch_fn(self,
                      n_domain:   int = 10000,
                      n_boundary: int = 1000,
                      n_initial:  int = 0,       # Helmholtz için kullanılmaz
                      scheduler=None) -> Callable:
        """
        Poisson kollokasyon: N_f=10000, N_b=1000
        Zamansız problem — scheduler yoksayılır.
        """
        def _sample():
            # Alan içi noktalar
            x_f = torch.rand(n_domain, 1, device=DEVICE) * (X_MAX - X_MIN) + X_MIN
            y_f = torch.rand(n_domain, 1, device=DEVICE) * (Y_MAX - Y_MIN) + Y_MIN
            domain = torch.cat([x_f, y_f], dim=1)

            # Sınır noktaları — 4 kenara eşit dağıt
            n_per_edge = n_boundary // 4
            coords_b_list = []

            for _ in range(n_per_edge):
                pass   # batch olarak aşağıda üretilecek

            s = torch.rand(n_per_edge, 1, device=DEVICE)

            # Sol  (x=-1),  sağ  (x=+1)
            left   = torch.cat([torch.full((n_per_edge,1), X_MIN, device=DEVICE),
                                 s * (Y_MAX-Y_MIN) + Y_MIN], dim=1)
            right  = torch.cat([torch.full((n_per_edge,1), X_MAX, device=DEVICE),
                                 s * (Y_MAX-Y_MIN) + Y_MIN], dim=1)
            # Alt  (y=-1),  üst  (y=+1)
            bottom = torch.cat([s * (X_MAX-X_MIN) + X_MIN,
                                 torch.full((n_per_edge,1), Y_MIN, device=DEVICE)], dim=1)
            top    = torch.cat([s * (X_MAX-X_MIN) + X_MIN,
                                 torch.full((n_per_edge,1), Y_MAX, device=DEVICE)], dim=1)

            boundary = torch.cat([left, right, bottom, top], dim=0)

            # Zamansız problem — initial boş tensor döndür
            initial = torch.zeros(0, 2, device=DEVICE)

            return {"domain": domain, "boundary": boundary, "initial": initial}

        def get_batch(epoch: int) -> dict:
            return _sample()

        # freeze/unfreeze: Poisson için no-op (zamansız)
        get_batch.set_residual = lambda r: None
        get_batch.freeze       = lambda: None
        get_batch.unfreeze     = lambda: None
        return get_batch
