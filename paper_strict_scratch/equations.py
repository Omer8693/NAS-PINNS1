from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from scipy.io import loadmat

from .config import EquationConfig


Tensor = torch.Tensor


POISSON_DOMAIN_MODULES = {
    "rectangular": "rectangular",
    "circle": "circle",
    "annulus": "annulus",
    "flower": "flower",
    "lshape": "lshape",
}

POISSON_DOMAIN_BOUNDS = {
    "rectangular": (-1.0, 1.0, -1.0, 1.0),
    "circle": (-1.0, 1.0, -1.0, 1.0),
    "annulus": (-1.0, 1.0, -1.0, 1.0),
    "flower": (-1.4, 1.4, -1.4, 1.4),
    "lshape": (-1.0, 2.0, -1.0, 2.0),
}

POISSON_ANNULUS_R_INNER = 0.3
POISSON_ANNULUS_R_OUTER = 1.0
POISSON_FLOWER_N_PETALS = 6
POISSON_FLOWER_AMP = 0.3


def _forward(model, x: Tensor, mask_indices: Optional[Sequence[int]]) -> Tensor:
    return model(x, mask_indices=mask_indices)


@dataclass
class TrainData1D:
    x_col: Tensor
    t_col: Tensor
    x_ic: Tensor
    t_ic: Tensor
    x_l: Tensor
    t_l: Tensor
    x_r: Tensor
    t_r: Tensor


@dataclass
class TrainData2D:
    x_col: Tensor
    y_col: Tensor
    t_col: Tensor
    x_ic: Tensor
    y_ic: Tensor
    t_ic: Tensor
    x_bc: Tensor
    y_bc: Tensor
    t_bc: Tensor


@dataclass
class TrainDataPoisson:
    x_col: Tensor
    y_col: Tensor
    x_bc: Tensor
    y_bc: Tensor


class Burgers1DEquation:
    """Paper-style Burgers 1D setup."""

    def __init__(
        self,
        cfg: EquationConfig,
        nu: float,
        train_nt: int = 21,
        train_nx: int = 250,
        test_nt: int = 21,
        test_nx: int = 500,
    ) -> None:
        self.cfg = cfg
        self.nu = float(nu)
        self.train_nt = int(train_nt)
        self.train_nx = int(train_nx)
        self.test_nt = int(test_nt)
        self.test_nx = int(test_nx)
        self._reference_cache: Dict[Tuple[int, int, float], np.ndarray] = {}
        self._mat_reference_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def sample_train(self, device: torch.device) -> TrainData1D:
        x_vals = torch.linspace(-1.0, 1.0, self.train_nx, device=device)
        t_vals = torch.linspace(0.0, 1.0, self.train_nt, device=device)

        Xc, Tc = torch.meshgrid(x_vals, t_vals, indexing="ij")
        x_col = Xc.reshape(-1, 1)
        t_col = Tc.reshape(-1, 1)

        x_ic = x_vals.unsqueeze(1)
        t_ic = torch.zeros_like(x_ic)

        t_bc = t_vals.unsqueeze(1)
        x_l = torch.full_like(t_bc, -1.0)
        x_r = torch.full_like(t_bc, 1.0)

        return TrainData1D(
            x_col=x_col,
            t_col=t_col,
            x_ic=x_ic,
            t_ic=t_ic,
            x_l=x_l,
            t_l=t_bc,
            x_r=x_r,
            t_r=t_bc,
        )

    def loss_components(self, model, data: TrainData1D, mask_indices: Optional[Sequence[int]]) -> Tuple[Tensor, Tensor, Tensor]:
        xt = torch.cat([data.x_col, data.t_col], dim=1).requires_grad_(True)
        u = _forward(model, xt, mask_indices)
        grads = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
        f = u_t + u * u_x - self.nu * u_xx
        l_pde = torch.mean(f.pow(2))

        u_ic_pred = _forward(model, torch.cat([data.x_ic, data.t_ic], dim=1), mask_indices)
        u_ic_true = -torch.sin(np.pi * data.x_ic)
        l_ic = torch.mean((u_ic_pred - u_ic_true).pow(2))

        u_l = _forward(model, torch.cat([data.x_l, data.t_l], dim=1), mask_indices)
        u_r = _forward(model, torch.cat([data.x_r, data.t_r], dim=1), mask_indices)
        l_bc = torch.mean(u_l.pow(2)) + torch.mean(u_r.pow(2))
        return l_pde, l_ic, l_bc

    def weighted_loss(self, l_pde: Tensor, l_ic: Tensor, l_bc: Tensor) -> Tensor:
        return self.cfg.lambda_pde * l_pde + self.cfg.lambda_ic * l_ic + self.cfg.lambda_bc * l_bc

    def _reference_solution_fd(self, x_vals_np: np.ndarray, t_vals_np: np.ndarray) -> np.ndarray:
        key = (len(x_vals_np), len(t_vals_np), round(self.nu, 12))
        if key in self._reference_cache:
            return self._reference_cache[key]

        nx = len(x_vals_np)
        dx = x_vals_np[1] - x_vals_np[0]
        u0 = -np.sin(np.pi * x_vals_np)
        u0[0] = 0.0
        u0[-1] = 0.0

        def rhs(_t, u_inner):
            u = np.zeros(nx, dtype=np.float64)
            u[1:-1] = u_inner
            ux = (u[2:] - u[:-2]) / (2.0 * dx)
            uxx = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx**2)
            return -u[1:-1] * ux + self.nu * uxx

        sol = solve_ivp(
            rhs,
            t_span=(float(t_vals_np[0]), float(t_vals_np[-1])),
            y0=u0[1:-1],
            t_eval=t_vals_np,
            method="BDF",
            rtol=1e-5,
            atol=1e-7,
        )
        if not sol.success:
            raise RuntimeError(f"Reference solver failed: {sol.message}")

        U = np.zeros((nx, len(t_vals_np)), dtype=np.float64)
        U[1:-1, :] = sol.y
        self._reference_cache[key] = U
        return U

    def _reference_solution_mat(self, x_vals_np: np.ndarray, t_vals_np: np.ndarray) -> Optional[np.ndarray]:
        key = (len(x_vals_np), len(t_vals_np))
        if key in self._mat_reference_cache:
            return self._mat_reference_cache[key]

        mat_path = Path(__file__).resolve().parents[1] / "burgers_shock.mat"
        if not mat_path.exists():
            return None

        try:
            data = loadmat(str(mat_path))
            x_exact = np.asarray(data["x"]).squeeze()
            t_exact = np.asarray(data["t"]).squeeze()
            u_exact = np.real(np.asarray(data["usol"]))
        except Exception:
            return None

        if x_exact.ndim != 1 or t_exact.ndim != 1 or u_exact.ndim != 2:
            return None
        if u_exact.shape != (len(x_exact), len(t_exact)):
            return None

        if np.array_equal(x_exact, x_vals_np) and np.array_equal(t_exact, t_vals_np):
            out = u_exact.astype(np.float64, copy=False)
            self._mat_reference_cache[key] = out
            return out

        # Match target test grid by separable interpolation over x then t.
        u_x = np.empty((len(x_vals_np), len(t_exact)), dtype=np.float64)
        for j in range(len(t_exact)):
            u_x[:, j] = np.interp(x_vals_np, x_exact, u_exact[:, j])

        u_xt = np.empty((len(x_vals_np), len(t_vals_np)), dtype=np.float64)
        for i in range(len(x_vals_np)):
            u_xt[i, :] = np.interp(t_vals_np, t_exact, u_x[i, :])

        self._mat_reference_cache[key] = u_xt
        return u_xt

    def reference_solution(self, x_vals_np: np.ndarray, t_vals_np: np.ndarray) -> np.ndarray:
        # Keep consistency with existing NAS-PINN exact-plot behavior:
        # prefer burgers_shock.mat for nu≈0.01, otherwise FD reference.
        if abs(float(self.nu) - 0.01) <= 1e-12:
            mat_ref = self._reference_solution_mat(x_vals_np, t_vals_np)
            if mat_ref is not None:
                return mat_ref
        return self._reference_solution_fd(x_vals_np, t_vals_np)

    def relative_l2(self, model, mask_indices: Optional[Sequence[int]], device: torch.device) -> float:
        x_test = torch.linspace(-1.0, 1.0, self.test_nx, device=device)
        t_test = torch.linspace(0.0, 1.0, self.test_nt, device=device)
        Xg, Tg = torch.meshgrid(x_test, t_test, indexing="ij")
        XT = torch.cat([Xg.reshape(-1, 1), Tg.reshape(-1, 1)], dim=1)
        with torch.no_grad():
            pred = _forward(model, XT, mask_indices).reshape(self.test_nx, self.test_nt).detach().cpu().numpy()
        ref = self.reference_solution(x_test.detach().cpu().numpy(), t_test.detach().cpu().numpy())
        return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12))

    def case_label(self) -> str:
        return f"nu_{self.nu:.3f}"


class Advection1DEquation:
    """Paper-style Advection 1D setup."""

    def __init__(
        self,
        cfg: EquationConfig,
        beta: float,
        train_nt: int = 40,
        train_nx: int = 120,
        test_nt: int = 40,
        test_nx: int = 120,
    ) -> None:
        self.cfg = cfg
        self.beta = float(beta)
        self.train_nt = int(train_nt)
        self.train_nx = int(train_nx)
        self.test_nt = int(test_nt)
        self.test_nx = int(test_nx)

    @staticmethod
    def exact_solution(beta: float, x: Tensor, t: Tensor) -> Tensor:
        return 0.8 * torch.sin(4.0 * np.pi * (x - beta * t) + np.pi / 4.0)

    def sample_train(self, device: torch.device) -> TrainData1D:
        x_vals = torch.linspace(0.0, 1.0, self.train_nx, device=device)
        t_vals = torch.linspace(0.0, 2.0, self.train_nt, device=device)

        Xc, Tc = torch.meshgrid(x_vals, t_vals, indexing="ij")
        x_col = Xc.reshape(-1, 1)
        t_col = Tc.reshape(-1, 1)

        x_ic = x_vals.unsqueeze(1)
        t_ic = torch.zeros_like(x_ic)

        t_bc = t_vals.unsqueeze(1)
        x_l = torch.zeros_like(t_bc)
        x_r = torch.ones_like(t_bc)

        return TrainData1D(
            x_col=x_col,
            t_col=t_col,
            x_ic=x_ic,
            t_ic=t_ic,
            x_l=x_l,
            t_l=t_bc,
            x_r=x_r,
            t_r=t_bc,
        )

    def loss_components(self, model, data: TrainData1D, mask_indices: Optional[Sequence[int]]) -> Tuple[Tensor, Tensor, Tensor]:
        xt = torch.cat([data.x_col, data.t_col], dim=1).requires_grad_(True)
        u = _forward(model, xt, mask_indices)
        grads = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        l_pde = torch.mean((u_t + self.beta * u_x).pow(2))

        u_ic_pred = _forward(model, torch.cat([data.x_ic, data.t_ic], dim=1), mask_indices)
        u_ic_true = self.exact_solution(self.beta, data.x_ic, data.t_ic)
        l_ic = torch.mean((u_ic_pred - u_ic_true).pow(2))

        # Periodic BC: u(0,t) == u(1,t)
        u_l = _forward(model, torch.cat([data.x_l, data.t_l], dim=1), mask_indices)
        u_r = _forward(model, torch.cat([data.x_r, data.t_r], dim=1), mask_indices)
        l_bc = torch.mean((u_l - u_r).pow(2))
        return l_pde, l_ic, l_bc

    def weighted_loss(self, l_pde: Tensor, l_ic: Tensor, l_bc: Tensor) -> Tensor:
        return self.cfg.lambda_pde * l_pde + self.cfg.lambda_ic * l_ic + self.cfg.lambda_bc * l_bc

    def relative_l2(self, model, mask_indices: Optional[Sequence[int]], device: torch.device) -> float:
        x_test = torch.linspace(0.0, 1.0, self.test_nx, device=device)
        t_test = torch.linspace(0.0, 2.0, self.test_nt, device=device)
        Xg, Tg = torch.meshgrid(x_test, t_test, indexing="ij")
        XT = torch.cat([Xg.reshape(-1, 1), Tg.reshape(-1, 1)], dim=1)
        with torch.no_grad():
            pred = _forward(model, XT, mask_indices).reshape(self.test_nx, self.test_nt)
        exact = self.exact_solution(self.beta, Xg, Tg)
        pred_np = pred.detach().cpu().numpy()
        exact_np = exact.detach().cpu().numpy()
        return float(np.linalg.norm(pred_np - exact_np) / (np.linalg.norm(exact_np) + 1e-12))

    def case_label(self) -> str:
        return f"beta_{self.beta:.3f}"


class Burgers2DEquation:
    """Paper-style 2D Burgers setup."""

    def __init__(
        self,
        cfg: EquationConfig,
        train_nt: int = 20,
        train_nx: int = 25,
        train_ny: int = 25,
        test_nt: int = 41,
        test_nx: int = 500,
        test_ny: int = 500,
        nu: float = 0.1,
    ) -> None:
        self.cfg = cfg
        self.train_nt = int(train_nt)
        self.train_nx = int(train_nx)
        self.train_ny = int(train_ny)
        self.test_nt = int(test_nt)
        self.test_nx = int(test_nx)
        self.test_ny = int(test_ny)
        self.nu = float(nu)

    @staticmethod
    def exact_solution(x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        return 1.0 / (1.0 + torch.exp((x + y - t) / 0.2))

    def sample_train(self, device: torch.device) -> TrainData2D:
        t_vals = torch.linspace(0.0, 2.0, self.train_nt, device=device)
        x_vals = torch.linspace(0.0, 1.0, self.train_nx, device=device)
        y_vals = torch.linspace(0.0, 1.0, self.train_ny, device=device)

        Xc, Yc, Tc = torch.meshgrid(x_vals, y_vals, t_vals, indexing="ij")
        x_col = Xc.reshape(-1, 1)
        y_col = Yc.reshape(-1, 1)
        t_col = Tc.reshape(-1, 1)

        Xi, Yi = torch.meshgrid(x_vals, y_vals, indexing="ij")
        x_ic = Xi.reshape(-1, 1)
        y_ic = Yi.reshape(-1, 1)
        t_ic = torch.zeros_like(x_ic)

        # Boundary points from all four faces.
        Yb, Tb = torch.meshgrid(y_vals, t_vals, indexing="ij")
        y_edge = Yb.reshape(-1, 1)
        t_edge = Tb.reshape(-1, 1)
        x_l = torch.zeros_like(y_edge)
        x_r = torch.ones_like(y_edge)

        Xb, Tb2 = torch.meshgrid(x_vals, t_vals, indexing="ij")
        x_edge = Xb.reshape(-1, 1)
        t_edge2 = Tb2.reshape(-1, 1)
        y_b = torch.zeros_like(x_edge)
        y_t = torch.ones_like(x_edge)

        x_bc = torch.cat([x_l, x_r, x_edge, x_edge], dim=0)
        y_bc = torch.cat([y_edge, y_edge, y_b, y_t], dim=0)
        t_bc = torch.cat([t_edge, t_edge, t_edge2, t_edge2], dim=0)

        return TrainData2D(
            x_col=x_col,
            y_col=y_col,
            t_col=t_col,
            x_ic=x_ic,
            y_ic=y_ic,
            t_ic=t_ic,
            x_bc=x_bc,
            y_bc=y_bc,
            t_bc=t_bc,
        )

    def loss_components(self, model, data: TrainData2D, mask_indices: Optional[Sequence[int]]) -> Tuple[Tensor, Tensor, Tensor]:
        xyt = torch.cat([data.x_col, data.y_col, data.t_col], dim=1).requires_grad_(True)
        u = _forward(model, xyt, mask_indices)
        grads = torch.autograd.grad(u.sum(), xyt, create_graph=True)[0]
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        u_t = grads[:, 2:3]
        u_xx = torch.autograd.grad(u_x.sum(), xyt, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), xyt, create_graph=True)[0][:, 1:2]
        f = u_t + u * (u_x + u_y) - self.nu * (u_xx + u_yy)
        l_pde = torch.mean(f.pow(2))

        u_ic_pred = _forward(model, torch.cat([data.x_ic, data.y_ic, data.t_ic], dim=1), mask_indices)
        u_ic_true = self.exact_solution(data.x_ic, data.y_ic, data.t_ic)
        l_ic = F.mse_loss(u_ic_pred, u_ic_true)

        u_bc_pred = _forward(model, torch.cat([data.x_bc, data.y_bc, data.t_bc], dim=1), mask_indices)
        u_bc_true = self.exact_solution(data.x_bc, data.y_bc, data.t_bc)
        l_bc = F.mse_loss(u_bc_pred, u_bc_true)
        return l_pde, l_ic, l_bc

    def weighted_loss(self, l_pde: Tensor, l_ic: Tensor, l_bc: Tensor) -> Tensor:
        return self.cfg.lambda_pde * l_pde + self.cfg.lambda_ic * l_ic + self.cfg.lambda_bc * l_bc

    def relative_l2(self, model, mask_indices: Optional[Sequence[int]], device: torch.device, batch_size: int = 65536) -> float:
        x_vals = np.linspace(0.0, 1.0, self.test_nx, dtype=np.float64)
        y_vals = np.linspace(0.0, 1.0, self.test_ny, dtype=np.float64)
        t_vals = np.linspace(0.0, 2.0, self.test_nt, dtype=np.float64)

        Xg, Yg = np.meshgrid(x_vals, y_vals, indexing="ij")
        xy_flat = np.stack([Xg.reshape(-1), Yg.reshape(-1)], axis=1)
        n_xy = xy_flat.shape[0]

        sq_err = 0.0
        sq_ref = 0.0
        model.eval()
        with torch.no_grad():
            for t in t_vals:
                t_col = np.full((n_xy, 1), t, dtype=np.float64)
                xyt_np = np.concatenate([xy_flat, t_col], axis=1)
                preds = np.empty((n_xy, 1), dtype=np.float64)
                for i in range(0, n_xy, batch_size):
                    j = min(i + batch_size, n_xy)
                    xyt = torch.from_numpy(xyt_np[i:j]).to(device=device, dtype=torch.float32)
                    out = _forward(model, xyt, mask_indices).detach().cpu().numpy()
                    preds[i:j] = out
                ref = 1.0 / (1.0 + np.exp((xy_flat[:, 0:1] + xy_flat[:, 1:2] - t) / 0.2))
                diff = preds - ref
                sq_err += float(np.sum(diff * diff))
                sq_ref += float(np.sum(ref * ref))
        return float(np.sqrt(sq_err / (sq_ref + 1e-12)))

    def case_label(self) -> str:
        return "default"


class PoissonEquation:
    """Poisson setup aligned with the original NAS-PINN Poisson benchmark."""

    def __init__(
        self,
        cfg: EquationConfig,
        domain_name: str = "rectangular",
        n_col: int = 4000,
        n_bc: int = 400,
        test_grid_size: int = 500,
    ) -> None:
        if domain_name not in POISSON_DOMAIN_MODULES:
            raise ValueError(
                f"Unsupported poisson domain: {domain_name}. "
                f"Available: {sorted(POISSON_DOMAIN_MODULES)}"
            )
        self.cfg = cfg
        self.domain_name = str(domain_name)
        self.n_col = int(n_col)
        self.n_bc = int(n_bc)
        self.test_grid_size = int(test_grid_size)

    @staticmethod
    def true_solution(x: Tensor, y: Tensor) -> Tensor:
        return torch.cos(torch.pi * x) * torch.cos(torch.pi * y)

    @staticmethod
    def poisson_rhs(x: Tensor, y: Tensor) -> Tensor:
        return -2.0 * (torch.pi**2) * torch.cos(torch.pi * x) * torch.cos(torch.pi * y)

    def _sample_rectangular(self, device: torch.device) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        x_col = torch.rand(self.n_col, 1, device=device) * 2.0 - 1.0
        y_col = torch.rand(self.n_col, 1, device=device) * 2.0 - 1.0
        n_side = max(self.n_bc // 4, 1)
        xb: list[Tensor] = []
        yb: list[Tensor] = []
        for val in (-1.0, 1.0):
            xb.append(torch.rand(n_side, 1, device=device) * 2.0 - 1.0)
            yb.append(torch.full((n_side, 1), val, device=device))
            xb.append(torch.full((n_side, 1), val, device=device))
            yb.append(torch.rand(n_side, 1, device=device) * 2.0 - 1.0)
        return (x_col, y_col), (torch.cat(xb, dim=0), torch.cat(yb, dim=0))

    def _sample_circle(self, device: torch.device) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        r = torch.sqrt(torch.rand(self.n_col, device=device))
        theta = 2.0 * torch.pi * torch.rand(self.n_col, device=device)
        x_col = (r * torch.cos(theta)).unsqueeze(1)
        y_col = (r * torch.sin(theta)).unsqueeze(1)
        theta_bc = torch.linspace(0.0, 2.0 * torch.pi, self.n_bc, device=device)
        x_bc = torch.cos(theta_bc).unsqueeze(1)
        y_bc = torch.sin(theta_bc).unsqueeze(1)
        return (x_col, y_col), (x_bc, y_bc)

    def _sample_annulus(self, device: torch.device) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        collected = []
        total = 0
        while total < self.n_col:
            cand = torch.rand(self.n_col * 4, 2, device=device) * 2.0 - 1.0
            r = torch.norm(cand, dim=1)
            inside = (r >= POISSON_ANNULUS_R_INNER) & (r <= POISSON_ANNULUS_R_OUTER)
            valid = cand[inside]
            if valid.numel() == 0:
                continue
            take = min(self.n_col - total, valid.shape[0])
            collected.append(valid[:take])
            total += take

        xy_col = torch.cat(collected, dim=0)
        x_col, y_col = xy_col[:, 0:1], xy_col[:, 1:2]

        theta = torch.linspace(0.0, 2.0 * torch.pi, max(self.n_bc // 2, 2), device=device)
        x_inner = POISSON_ANNULUS_R_INNER * torch.cos(theta).unsqueeze(1)
        y_inner = POISSON_ANNULUS_R_INNER * torch.sin(theta).unsqueeze(1)
        x_outer = POISSON_ANNULUS_R_OUTER * torch.cos(theta).unsqueeze(1)
        y_outer = POISSON_ANNULUS_R_OUTER * torch.sin(theta).unsqueeze(1)
        x_bc = torch.cat([x_inner, x_outer], dim=0)
        y_bc = torch.cat([y_inner, y_outer], dim=0)
        return (x_col, y_col), (x_bc, y_bc)

    def _sample_flower(self, device: torch.device) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        collected = []
        total = 0
        while total < self.n_col:
            cand = torch.rand(self.n_col * 6, 2, device=device) * 2.8 - 1.4
            r = torch.norm(cand, dim=1)
            theta = torch.atan2(cand[:, 1], cand[:, 0])
            r_max = 1.0 + POISSON_FLOWER_AMP * torch.sin(POISSON_FLOWER_N_PETALS * theta)
            valid = cand[r <= r_max]
            if valid.numel() == 0:
                continue
            take = min(self.n_col - total, valid.shape[0])
            collected.append(valid[:take])
            total += take

        xy_col = torch.cat(collected, dim=0)
        x_col, y_col = xy_col[:, 0:1], xy_col[:, 1:2]

        theta_b = torch.linspace(0.0, 2.0 * torch.pi, self.n_bc, device=device)
        r_b = (1.0 + POISSON_FLOWER_AMP * torch.sin(POISSON_FLOWER_N_PETALS * theta_b)).unsqueeze(1)
        x_bc = r_b * torch.cos(theta_b).unsqueeze(1)
        y_bc = r_b * torch.sin(theta_b).unsqueeze(1)
        return (x_col, y_col), (x_bc, y_bc)

    def _sample_lshape(self, device: torch.device) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        collected = []
        total = 0
        while total < self.n_col:
            cand = torch.rand(self.n_col * 4, 2, device=device) * 3.0 - 1.0
            x, y = cand[:, 0], cand[:, 1]
            inside = (
                ((x >= -1.0) & (x <= 2.0) & (y >= -1.0) & (y <= 1.0))
                | ((x >= -1.0) & (x <= 1.0) & (y >= 1.0) & (y <= 2.0))
            )
            valid = cand[inside]
            if valid.numel() == 0:
                continue
            take = min(self.n_col - total, valid.shape[0])
            collected.append(valid[:take])
            total += take

        xy_col = torch.cat(collected, dim=0)
        x_col, y_col = xy_col[:, 0:1], xy_col[:, 1:2]

        n_seg = max(self.n_bc // 6, 1)
        rem = self.n_bc - (n_seg * 6)
        counts = [n_seg] * 6
        for i in range(rem):
            counts[i % 6] += 1

        x_bc_parts = []
        y_bc_parts = []
        x_bc_parts.append(torch.rand(counts[0], 1, device=device) * 3.0 - 1.0)
        y_bc_parts.append(torch.full((counts[0], 1), -1.0, device=device))
        x_bc_parts.append(torch.full((counts[1], 1), 2.0, device=device))
        y_bc_parts.append(torch.rand(counts[1], 1, device=device) * 2.0 - 1.0)
        x_bc_parts.append(torch.rand(counts[2], 1, device=device) + 1.0)
        y_bc_parts.append(torch.full((counts[2], 1), 1.0, device=device))
        x_bc_parts.append(torch.full((counts[3], 1), 1.0, device=device))
        y_bc_parts.append(torch.rand(counts[3], 1, device=device) + 1.0)
        x_bc_parts.append(torch.rand(counts[4], 1, device=device) * 2.0 - 1.0)
        y_bc_parts.append(torch.full((counts[4], 1), 2.0, device=device))
        x_bc_parts.append(torch.full((counts[5], 1), -1.0, device=device))
        y_bc_parts.append(torch.rand(counts[5], 1, device=device) * 3.0 - 1.0)
        x_bc = torch.cat(x_bc_parts, dim=0)
        y_bc = torch.cat(y_bc_parts, dim=0)
        return (x_col, y_col), (x_bc, y_bc)

    def _sample_points(self, device: torch.device) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        if self.domain_name == "rectangular":
            return self._sample_rectangular(device)
        if self.domain_name == "circle":
            return self._sample_circle(device)
        if self.domain_name == "annulus":
            return self._sample_annulus(device)
        if self.domain_name == "flower":
            return self._sample_flower(device)
        if self.domain_name == "lshape":
            return self._sample_lshape(device)
        raise ValueError(f"Unsupported poisson domain: {self.domain_name}")

    def sample_train(self, device: torch.device) -> TrainDataPoisson:
        (x_col, y_col), (x_bc, y_bc) = self._sample_points(device=device)
        return TrainDataPoisson(
            x_col=x_col,
            y_col=y_col,
            x_bc=x_bc,
            y_bc=y_bc,
        )

    def loss_components(self, model, data: TrainDataPoisson, mask_indices: Optional[Sequence[int]]) -> Tuple[Tensor, Tensor, Tensor]:
        xy = torch.cat([data.x_col, data.y_col], dim=1).requires_grad_(True)
        u = _forward(model, xy, mask_indices)
        grads = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), xy, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), xy, create_graph=True)[0][:, 1:2]

        rhs = self.poisson_rhs(data.x_col, data.y_col)
        l_pde = torch.mean((u_xx + u_yy - rhs).pow(2))

        bc_xy = torch.cat([data.x_bc, data.y_bc], dim=1)
        u_bc_pred = _forward(model, bc_xy, mask_indices)
        u_bc_true = self.true_solution(data.x_bc, data.y_bc)
        l_bc = F.mse_loss(u_bc_pred, u_bc_true)

        l_ic = torch.zeros((), device=l_pde.device, dtype=l_pde.dtype)
        return l_pde, l_ic, l_bc

    def weighted_loss(self, l_pde: Tensor, l_ic: Tensor, l_bc: Tensor) -> Tensor:
        del l_ic
        return self.cfg.lambda_pde * l_pde + self.cfg.lambda_bc * l_bc

    def _domain_mask(self, x: Tensor, y: Tensor) -> Tensor:
        if self.domain_name == "rectangular":
            return torch.ones_like(x, dtype=torch.bool)
        if self.domain_name == "circle":
            return (x**2 + y**2) <= 1.0
        if self.domain_name == "annulus":
            r = torch.sqrt(x**2 + y**2)
            return (r >= POISSON_ANNULUS_R_INNER) & (r <= POISSON_ANNULUS_R_OUTER)
        if self.domain_name == "flower":
            r = torch.sqrt(x**2 + y**2)
            theta = torch.atan2(y, x)
            r_max = 1.0 + POISSON_FLOWER_AMP * torch.sin(POISSON_FLOWER_N_PETALS * theta)
            return r <= r_max
        if self.domain_name == "lshape":
            return (
                ((x >= -1.0) & (x <= 2.0) & (y >= -1.0) & (y <= 1.0))
                | ((x >= -1.0) & (x <= 1.0) & (y >= 1.0) & (y <= 2.0))
            )
        return torch.ones_like(x, dtype=torch.bool)

    def evaluate_on_grid(
        self,
        model,
        mask_indices: Optional[Sequence[int]],
        device: torch.device,
        batch_size: int = 65536,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_min, x_max, y_min, y_max = POISSON_DOMAIN_BOUNDS[self.domain_name]
        x_vals = np.linspace(x_min, x_max, self.test_grid_size, dtype=np.float64)
        y_vals = np.linspace(y_min, y_max, self.test_grid_size, dtype=np.float64)

        X, Y = np.meshgrid(x_vals, y_vals, indexing="xy")
        xy_np = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
        pred = np.empty((xy_np.shape[0], 1), dtype=np.float64)

        model.eval()
        with torch.no_grad():
            for i in range(0, xy_np.shape[0], batch_size):
                j = min(i + batch_size, xy_np.shape[0])
                xy = torch.from_numpy(xy_np[i:j]).to(device=device, dtype=torch.float32)
                out = _forward(model, xy, mask_indices).detach().cpu().numpy()
                pred[i:j] = out

        x_t = torch.from_numpy(X).to(device=device, dtype=torch.float32)
        y_t = torch.from_numpy(Y).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            true = self.true_solution(x_t, y_t).detach().cpu().numpy()
            mask = self._domain_mask(x_t, y_t).detach().cpu().numpy().astype(bool)

        pred_field = pred.reshape(self.test_grid_size, self.test_grid_size)
        return x_vals, y_vals, pred_field, true, mask

    def relative_l2(
        self,
        model,
        mask_indices: Optional[Sequence[int]],
        device: torch.device,
    ) -> float:
        _, _, pred, true, mask = self.evaluate_on_grid(model, mask_indices, device=device)
        err_sq = (pred - true) ** 2
        if np.any(mask):
            num = float(np.sqrt(np.mean(err_sq[mask])))
            den = float(np.sqrt(np.mean((true[mask]) ** 2)) + 1e-12)
            return num / den
        num = float(np.sqrt(np.mean(err_sq)))
        den = float(np.sqrt(np.mean(true**2)) + 1e-12)
        return num / den

    def case_label(self) -> str:
        return f"domain_{self.domain_name}"
