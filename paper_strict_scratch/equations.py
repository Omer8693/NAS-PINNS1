from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.integrate import solve_ivp

from .config import EquationConfig


Tensor = torch.Tensor


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

    def relative_l2(self, model, mask_indices: Optional[Sequence[int]], device: torch.device) -> float:
        x_test = torch.linspace(-1.0, 1.0, self.test_nx, device=device)
        t_test = torch.linspace(0.0, 1.0, self.test_nt, device=device)
        Xg, Tg = torch.meshgrid(x_test, t_test, indexing="ij")
        XT = torch.cat([Xg.reshape(-1, 1), Tg.reshape(-1, 1)], dim=1)
        with torch.no_grad():
            pred = _forward(model, XT, mask_indices).reshape(self.test_nx, self.test_nt).detach().cpu().numpy()
        ref = self._reference_solution_fd(x_test.detach().cpu().numpy(), t_test.detach().cpu().numpy())
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
