"""
Allen-Cahn Denklemi (1D+t) — NAS-PINN Paper Benchmark
======================================================
Kaynak: Wang & Zhong (2023) NAS-PINN, benchmark seti.
        Lu et al. (2021) DeepXDE, Wang et al. (2022) gradual curriculum.

PDE  : u_t − ε²·u_xx + 5(u³ − u) = 0,   x∈[-1,1], t∈[0,1]
ε²   : 0.0001   (NAS-PINN paper / Wang 2022 "failure modes" paper)
IC   : u(x,0) = x²·cos(πx)
BC   : periyodik  u(-1,t) = u(1,t),  u_x(-1,t) = u_x(1,t)

Analitik çözüm: mevcut değil → referans Method-of-Lines (RK45) ile.

Kollokasiyon (paper):
  N_f = 10 000  (alan içi)
  N_0 =    512  (başlangıç)
  BC  : periyodik (kayıpta ceza terimi)

Temporal skip için neden önemli:
  • t≈0 → IC'den yüksek dinamik, hızlı faz ayrışması
  • t→1 → denge durumu (±1 bölgeleri), yavaş değişim → atlanabilir

NOT: Bu denklem NAS-PINN'de "failure modes" göstergesi olarak da kullanılır.
     u = ±1 plateolarına yakınsama süresi optimizer seçimine hassas.
"""

import torch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from typing import Callable

from .base import PINNProblem

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# Bölüm 1 — Problem Sabitleri (NAS-PINN paper)
# ──────────────────────────────────────────────────────────────

EPS2  = 0.0001   # ε² katsayısı — NAS-PINN paper
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX =  0.0, 1.0

W_PDE = 1.0
W_IC  = 1.0
W_BC  = 1.0


# ──────────────────────────────────────────────────────────────
# Bölüm 2 — Referans Çözüm (Method of Lines, RK45)
# ──────────────────────────────────────────────────────────────

def _compute_reference(nx: int = 512, nt: int = 200) -> tuple:
    """
    Allen-Cahn denklemini MOL + RK45 ile çöz.
    Periyodik BC: sarmal (circular) fark şeması.
    Döndürür: (x_1d, t_1d, u_2d)  shapes: (nx,), (nt+1,), (nt+1, nx)
    """
    x  = np.linspace(X_MIN, X_MAX, nx, endpoint=False)   # periyodik: son nokta tekrar
    dx = x[1] - x[0]
    u0 = x**2 * np.cos(np.pi * x)   # IC: u(x,0) = x²·cos(πx)

    def rhs(t, u):
        # Periyodik ikinci türev: merkezi fark + sarmal
        u_xx        = np.zeros_like(u)
        u_xx[1:-1]  = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        u_xx[0]     = (u[1]  - 2*u[0]  + u[-1])  / dx**2   # sol periyodik
        u_xx[-1]    = (u[0]  - 2*u[-1] + u[-2])  / dx**2   # sağ periyodik
        # Allen-Cahn RHS
        return EPS2 * u_xx - 5 * (u**3 - u)

    t_eval = np.linspace(T_MIN, T_MAX, nt + 1)
    sol = solve_ivp(rhs, [T_MIN, T_MAX], u0,
                    t_eval=t_eval, method="RK45",
                    rtol=1e-7, atol=1e-9)

    if not sol.success:
        raise RuntimeError(f"[Allen-Cahn] Reference solution failed: {sol.message}")

    return x, sol.t, sol.y.T   # [nt+1, nx]


print("  [Allen-Cahn] Computing reference solution (RK45)...", end=" ", flush=True)
_X_REF, _T_REF, _U_REF = _compute_reference()
_INTERP = RegularGridInterpolator(
    (_T_REF, _X_REF), _U_REF,
    method="linear", bounds_error=False, fill_value=None
)
print(f"tamam  (grid: {len(_T_REF)}t × {len(_X_REF)}x)")


# ──────────────────────────────────────────────────────────────
# Bölüm 3 — PDE Rezidüeli
# ──────────────────────────────────────────────────────────────

def allen_cahn_residual(net_out: torch.Tensor,
                        coords:  torch.Tensor) -> torch.Tensor:
    """
    Allen-Cahn PDE rezidüeli:  R = u_t − ε²·u_xx + 5(u³ − u)

    coords  : [N, 2]  (t, x), requires_grad=True
    net_out : [N, 1]
    """
    u     = net_out[:, 0:1]
    grads = torch.autograd.grad(
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

    return u_t - EPS2 * u_xx + 5.0 * (u**3 - u)


# ──────────────────────────────────────────────────────────────
# Bölüm 4 — AllenCahnProblem Sınıfı
# ──────────────────────────────────────────────────────────────

class AllenCahnProblem(PINNProblem):
    """
    Allen-Cahn 1D+t — NAS-PINN paper üçüncü benchmark.

    Ağ: f(t, x) → u(t, x)   [2 girdi, 1 çıktı]

    Temporal skip için mükemmel:
      şok dinamiği erken t'de yüksek rezidüel üretiyor,
      geç t'de plateau → düşük rezidüel → atlanabilir.

    Periyodik BC: kayıpta u(-1,t)−u(1,t) cezası.
    """

    name          = "allen_cahn"
    n_input       = 2
    n_output      = 1
    domain_bounds = {"t": (T_MIN, T_MAX), "x": (X_MIN, X_MAX)}

    def analytical_solution(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Referans çözüm (MOL interpolasyonu).
        coords : [N, 2]  (t, x)
        """
        tx    = coords[:, :2].detach().cpu().numpy()
        u_ref = _INTERP(tx)
        return torch.tensor(u_ref[:, None], dtype=torch.float32, device=DEVICE)

    def make_loss_fn(self) -> Callable:
        """
        Allen-Cahn kayıp:
          L = λ_pde·||R||² + λ_ic·||u(x,0)−x²cos(πx)||²
                           + λ_bc·(||u(-1,t)−u(1,t)||² + ||u_x(-1,t)−u_x(1,t)||²)
        """
        def loss_fn(model, batch):
            # ── PDE rezidüeli ────────────────────────────────
            coords_f = batch["domain"]
            coords_f.requires_grad_(True)
            pred_f   = model(coords_f)
            r_pde    = allen_cahn_residual(pred_f, coords_f)
            L_pde    = W_PDE * (r_pde ** 2).mean()

            # ── Başlangıç koşulu ────────────────────────────
            coords_i = batch["initial"]
            pred_i   = model(coords_i)
            x_i      = coords_i[:, 1:2]
            true_i   = x_i**2 * torch.cos(np.pi * x_i)
            L_ic     = W_IC * ((pred_i - true_i) ** 2).mean()

            # ── Periyodik BC: u(-1,t) = u(1,t) ──────────────
            t_b   = batch["boundary"][:, 0:1]
            x_neg = torch.full_like(t_b, X_MIN)
            x_pos = torch.full_like(t_b, X_MAX)
            c_neg = torch.cat([t_b, x_neg], dim=1)
            c_pos = torch.cat([t_b, x_pos], dim=1)
            u_neg = model(c_neg)
            u_pos = model(c_pos)
            L_bc  = W_BC * ((u_neg - u_pos) ** 2).mean()

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
        """L2 hatası — MOL referans çözümüne karşı."""
        rng = np.random.default_rng(seed=0)
        t_v = rng.uniform(T_MIN, T_MAX, n_val).astype(np.float32)
        x_v = rng.uniform(X_MIN, X_MAX, n_val).astype(np.float32)

        u_ref_np = _INTERP(np.stack([t_v, x_v], axis=1))
        coords_v = torch.tensor(
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
                      n_initial:  int = 512,
                      scheduler=None) -> Callable:
        """
        NAS-PINN: N_f=10000, N_0=512, periyodik BC noktaları.
        """
        current_residual = [float("inf")]
        is_frozen        = [False]

        def _sample(window=None):
            t0 = T_MIN if window is None else window[0]
            t1 = T_MAX if window is None else window[1]

            # Alan içi
            t_f = torch.rand(n_domain, 1, device=DEVICE) * (t1 - t0) + t0
            x_f = torch.rand(n_domain, 1, device=DEVICE) * (X_MAX - X_MIN) + X_MIN
            domain = torch.cat([t_f, x_f], dim=1)

            # IC: t=0
            x_i = torch.rand(n_initial, 1, device=DEVICE) * (X_MAX - X_MIN) + X_MIN
            initial = torch.cat([torch.zeros(n_initial, 1, device=DEVICE), x_i], dim=1)

            # BC periyodik: sadece t değerleri gerekli
            t_b    = torch.rand(n_boundary, 1, device=DEVICE) * (t1 - t0) + t0
            x_zero = torch.zeros(n_boundary, 1, device=DEVICE)  # placeholder
            boundary = torch.cat([t_b, x_zero], dim=1)

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
