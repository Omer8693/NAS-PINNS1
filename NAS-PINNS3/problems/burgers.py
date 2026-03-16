"""
Burgers Denklemi (1D) — NAS-PINN Paper Benchmark
==================================================
Kaynak: Wang & Zhong (2023) NAS-PINN, Tablo benchmark seti.
        Raissi, Perdikaris & Karniadakis (2019) J. Comput. Phys.

PDE  : u_t + u·u_x = ν·u_xx,   x∈[-1,1], t∈[0,1]
ν    : 0.01/π   (NAS-PINN paper / Raissi 2019 standart değeri)
IC   : u(x, 0) = -sin(πx)
BC   : u(-1, t) = u(1, t) = 0   (Dirichlet homojen)

Kayıp ağırlıkları (NAS-PINN paper, burgers/naspinn.py lines 52-54):
  λ_pde = 1.0,  λ_ic = 100.0,  λ_bc = 100.0

Kollokasiyon boyutu (paper Tablo 1):
  N_f = 10 000  (alan içi PDE noktaları)
  N_b =    200  (sınır koşulu)
  N_0 =    200  (başlangıç koşulu)

Referans çözüm: Method-of-Lines + RK45 (scipy.integrate.solve_ivp)
  Nx=256, Nt=201 çözüm ızgarası → RegularGridInterpolator ile kesintisiz
"""

import torch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from typing import Callable, Optional
from copy import deepcopy

from .base import PINNProblem

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# Bölüm 1 — Problem Sabitleri (NAS-PINN / Raissi 2019)
# ──────────────────────────────────────────────────────────────

NU     = 0.01 / np.pi   # viskozite katsayısı
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX =  0.0, 1.0

# Kayıp ağırlıkları — NAS-PINN paper burgers/naspinn.py lines 52-54
W_PDE = 1.0
W_IC  = 100.0
W_BC  = 100.0


# ──────────────────────────────────────────────────────────────
# Bölüm 2 — Referans Çözüm (Method of Lines, RK45)
# ──────────────────────────────────────────────────────────────

def _compute_reference(nx: int = 256, nt: int = 200) -> tuple:
    """
    Burgers denklemini Method of Lines + RK45 ile çöz.
    Upwind fark şeması + ikinci mertebe diffüzyon.

    Döndürür: (x_1d, t_1d, u_2d)  shapes: (nx,), (nt+1,), (nt+1, nx)
    """
    x  = np.linspace(X_MIN, X_MAX, nx)
    dx = x[1] - x[0]
    u0 = -np.sin(np.pi * x)          # IC: u(x,0) = -sin(πx)

    def rhs(t, u):
        dudt = np.zeros_like(u)
        # İkinci mertebe diffüzyon: ν·u_xx  (merkezi fark)
        dudt[1:-1] = NU * (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        # Birinci mertebe adveksiyon: upwind u·u_x
        pos = u[1:-1] >= 0
        dudt[1:-1][pos]  -= u[1:-1][pos]  * (u[1:-1][pos]  - u[:-2][pos])  / dx
        dudt[1:-1][~pos] -= u[1:-1][~pos] * (u[2:][~pos]   - u[1:-1][~pos]) / dx
        # Homojen Dirichlet BC
        dudt[0] = dudt[-1] = 0.0
        return dudt

    t_eval = np.linspace(T_MIN, T_MAX, nt + 1)
    sol = solve_ivp(rhs, [T_MIN, T_MAX], u0,
                    t_eval=t_eval, method="RK45",
                    rtol=1e-7, atol=1e-9)

    if not sol.success:
        raise RuntimeError(f"[Burgers] Reference solution failed: {sol.message}")

    return x, sol.t, sol.y.T   # u: [nt+1, nx]


# Modül ilk yüklendiğinde referans çözümü hesapla + interpolatör kur
print("  [Burgers] Computing reference solution (RK45)...", end=" ", flush=True)
_X_REF, _T_REF, _U_REF = _compute_reference()
_INTERP = RegularGridInterpolator(
    (_T_REF, _X_REF), _U_REF,
    method="linear", bounds_error=False, fill_value=None
)
print(f"tamam  (grid: {len(_T_REF)}t × {len(_X_REF)}x)")


# ──────────────────────────────────────────────────────────────
# Bölüm 3 — PDE Rezidüeli
# ──────────────────────────────────────────────────────────────

def burgers_residual(net_out: torch.Tensor,
                     coords:  torch.Tensor) -> torch.Tensor:
    """
    Burgers PDE rezidüeli:  R = u_t + u·u_x − ν·u_xx

    coords  : [N, 2]  — (t, x) sütunları, requires_grad=True
    net_out : [N, 1]  — ağın tahmin ettiği u değerleri

    autograd ile türevler alınır (create_graph=True → L-BFGS uyumlu).
    """
    u      = net_out[:, 0:1]
    grads  = torch.autograd.grad(
        u, coords,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    u_t = grads[:, 0:1]
    u_x = grads[:, 1:2]

    u_xx = torch.autograd.grad(
        u_x, coords,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0][:, 1:2]

    return u_t + u * u_x - NU * u_xx


# ──────────────────────────────────────────────────────────────
# Bölüm 4 — BurgersProblem Sınıfı
# ──────────────────────────────────────────────────────────────

class BurgersProblem(PINNProblem):
    """
    Burgers 1D — NAS-PINN paper birincil benchmark problemi.

    Ağ: f(t, x) → u(t, x)   [2 girdi, 1 çıktı]

    Temporal skip için en uygun problem:
      t=0..0.3 → şok dalgası gelişiyor (yüksek rezidüel → atlama yapma)
      t=0.3..1 → şok kararlı, yavaş evrim (düşük rezidüel → atla)
    """

    name          = "burgers"
    n_input       = 2
    n_output      = 1
    domain_bounds = {"t": (T_MIN, T_MAX), "x": (X_MIN, X_MAX)}

    def analytical_solution(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Referans çözüm (Method of Lines + RK45 ile hesaplanmış).
        coords : [N, 2]  (t, x) sütunları
        """
        tx = coords[:, :2].detach().cpu().numpy()
        u  = _INTERP(tx)
        return torch.tensor(u[:, None], dtype=torch.float32, device=DEVICE)

    def make_loss_fn(self) -> Callable:
        """
        Burgers kayıp fonksiyonu.
          L = λ_pde·||R_pde||² + λ_ic·||u(x,0)+sin(πx)||² + λ_bc·||u(±1,t)||²
        """
        def loss_fn(model, batch):
            # ── PDE rezidüeli ────────────────────────────────
            coords_f = batch["domain"]   # [N_f, 2]
            coords_f.requires_grad_(True)
            pred_f   = model(coords_f)
            r_pde    = burgers_residual(pred_f, coords_f)
            L_pde    = W_PDE * (r_pde ** 2).mean()

            # ── Başlangıç koşulu: u(x,0) = -sin(πx) ─────────
            coords_i = batch["initial"]   # [N_0, 2]
            pred_i   = model(coords_i)
            true_i   = -torch.sin(np.pi * coords_i[:, 1:2])
            L_ic     = W_IC * ((pred_i - true_i) ** 2).mean()

            # ── Sınır koşulu: u(-1,t)=u(1,t)=0 ─────────────
            coords_b = batch["boundary"]   # [N_b, 2]
            pred_b   = model(coords_b)
            true_b   = torch.zeros_like(pred_b)
            L_bc     = W_BC * ((pred_b - true_b) ** 2).mean()

            loss = L_pde + L_ic + L_bc
            details = {
                "L_total":    loss.item(),
                "L_physics":  L_pde.item(),
                "L_initial":  L_ic.item(),
                "L_boundary": L_bc.item(),
            }
            return loss, details

        return loss_fn

    def make_val_fn(self, n_val: int = 1000) -> Callable:
        """
        Doğrulama fonksiyonu: normalize L2 hatası.
        Referans: Method of Lines çözümünden interpolasyon.
        """
        rng = np.random.default_rng(seed=0)
        t_v = rng.uniform(T_MIN, T_MAX, n_val).astype(np.float32)
        x_v = rng.uniform(X_MIN, X_MAX, n_val).astype(np.float32)

        # Referans değerler: numpy → torch
        u_ref_np = _INTERP(np.stack([t_v, x_v], axis=1))
        coords_v  = torch.tensor(
            np.stack([t_v, x_v], axis=1), dtype=torch.float32, device=DEVICE
        )
        u_ref = torch.tensor(u_ref_np[:, None], dtype=torch.float32, device=DEVICE)

        def val_fn(model) -> float:
            model.eval()
            with torch.no_grad():
                pred = model(coords_v)
            return (torch.norm(pred - u_ref) /
                    (torch.norm(u_ref) + 1e-10)).item()

        return val_fn

    def make_batch_fn(self,
                      n_domain:   int = 10000,
                      n_boundary: int = 200,
                      n_initial:  int = 200,
                      scheduler=None) -> Callable:
        """
        NAS-PINN paper kollokasiyon boyutları:
          N_f=10000, N_b=200, N_0=200

        scheduler bağlanırsa zaman aralığı [t_win] ile kısıtlanır.
        """
        current_residual = [float("inf")]
        is_frozen        = [False]

        def _sample(window=None):
            # Zaman penceresi
            t0 = T_MIN if window is None else window[0]
            t1 = T_MAX if window is None else window[1]

            # Alan içi (Latin Hypercube benzeri uniform)
            t_f = torch.rand(n_domain,   1, device=DEVICE) * (t1 - t0) + t0
            x_f = torch.rand(n_domain,   1, device=DEVICE) * (X_MAX - X_MIN) + X_MIN
            domain = torch.cat([t_f, x_f], dim=1)

            # Başlangıç koşulu: t=0, x∈[-1,1]
            x_i = torch.rand(n_initial, 1, device=DEVICE) * (X_MAX - X_MIN) + X_MIN
            t_i = torch.zeros(n_initial, 1, device=DEVICE)
            initial = torch.cat([t_i, x_i], dim=1)

            # Sınır koşulu: x=-1 veya x=1, t∈[t0,t1]
            t_b  = torch.rand(n_boundary, 1, device=DEVICE) * (t1 - t0) + t0
            x_b  = torch.where(
                torch.rand(n_boundary, 1, device=DEVICE) > 0.5,
                torch.ones(n_boundary,  1, device=DEVICE) *  X_MAX,
                torch.ones(n_boundary,  1, device=DEVICE) *  X_MIN,
            )
            boundary = torch.cat([t_b, x_b], dim=1)

            return {"domain": domain, "boundary": boundary, "initial": initial}

        def get_batch(epoch: int) -> dict:
            window = None
            if scheduler is not None and not is_frozen[0]:
                t = scheduler.next_timestep(current_residual[0])
                if t is None:
                    scheduler.reset()
                    t = scheduler.next_timestep()
                n_steps = max(len(scheduler.t_full) - 1, 1)
                dt = (scheduler.t_end - scheduler.t_start) / n_steps
                window = (max(scheduler.t_start, t - dt / 2),
                          min(scheduler.t_end,   t + dt / 2))
            return _sample(window)

        def set_residual(r: float): current_residual[0] = float(r)
        def freeze():               is_frozen[0] = True
        def unfreeze():             is_frozen[0] = False

        get_batch.set_residual = set_residual
        get_batch.freeze       = freeze
        get_batch.unfreeze     = unfreeze
        return get_batch
