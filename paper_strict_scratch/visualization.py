from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .equations import Advection1DEquation, Burgers1DEquation, Burgers2DEquation


CMAP = "YlGnBu"


def _finalize(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


def plot_loss_curve(loss_history: Sequence[float], save_path: Path, title: str) -> None:
    if not loss_history:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(np.asarray(loss_history, dtype=np.float64), linewidth=1.3, color="royalblue")
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _finalize(save_path)


def _plot_xt_comparison(
    x_vals: np.ndarray,
    t_vals: np.ndarray,
    exact_u: np.ndarray,
    pred_u: np.ndarray,
    rel_l2: float,
    save_path: Path,
    title_prefix: str,
) -> None:
    Xg, Tg = np.meshgrid(x_vals, t_vals, indexing="ij")
    err = np.abs(pred_u - exact_u)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    cs0 = axes[0].contourf(Xg, Tg, exact_u, levels=60, cmap=CMAP)
    axes[0].set_title("Exact")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(cs0, ax=axes[0])

    cs1 = axes[1].contourf(Xg, Tg, pred_u, levels=60, cmap=CMAP)
    axes[1].set_title("Predicted")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    fig.colorbar(cs1, ax=axes[1])

    cs2 = axes[2].contourf(Xg, Tg, err, levels=60, cmap=CMAP)
    axes[2].set_title("|Pred-Exact|")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")
    fig.colorbar(cs2, ax=axes[2])

    fig.suptitle(f"{title_prefix} | Relative L2={rel_l2:.4e}")
    _finalize(save_path)


def _plot_burgers1d_time_slices(
    x_vals: np.ndarray,
    t_vals: np.ndarray,
    exact_u: np.ndarray,
    pred_u: np.ndarray,
    save_path: Path,
    t_slices: Optional[Sequence[float]] = None,
) -> None:
    if t_slices is None:
        t_slices = (0.0, 0.25, 0.5, 0.75, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(t_slices)))

    for idx, t_sel in enumerate(t_slices):
        t_idx = int(np.argmin(np.abs(t_vals - float(t_sel))))
        t_used = float(t_vals[t_idx])
        axes[0].plot(x_vals, exact_u[:, t_idx], color=colors[idx], linewidth=2.2, label=f"t={t_used:.2f}")
        axes[1].plot(x_vals, pred_u[:, t_idx], color=colors[idx], linewidth=2.2, label=f"t={t_used:.2f}")

    axes[0].set_title("Exact Time Slices")
    axes[1].set_title("Predicted Time Slices")
    axes[0].set_xlabel("x")
    axes[1].set_xlabel("x")
    axes[0].set_ylabel("u(x,t)")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(title="Exact")
    axes[1].legend(title="Pred")
    fig.suptitle("Burgers1D: Exact vs Predicted Time Slices")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _finalize(save_path)


def _predict_1d_field(model, mask_indices, x_vals: torch.Tensor, t_vals: torch.Tensor) -> np.ndarray:
    Xg, Tg = torch.meshgrid(x_vals, t_vals, indexing="ij")
    xt = torch.cat([Xg.reshape(-1, 1), Tg.reshape(-1, 1)], dim=1)
    with torch.no_grad():
        pred = model(xt, mask_indices=mask_indices).reshape(len(x_vals), len(t_vals))
    return pred.detach().cpu().numpy()


def _predict_2d_slice(
    model,
    mask_indices,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    t_value: float,
    device: torch.device,
    batch_size: int = 65536,
) -> np.ndarray:
    Xg, Yg = np.meshgrid(x_vals, y_vals, indexing="ij")
    xy = np.stack([Xg.reshape(-1), Yg.reshape(-1)], axis=1)
    t_col = np.full((xy.shape[0], 1), float(t_value), dtype=np.float64)
    xyt_np = np.concatenate([xy, t_col], axis=1)

    pred = np.empty((xy.shape[0], 1), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, xy.shape[0], batch_size):
            j = min(i + batch_size, xy.shape[0])
            xyt = torch.from_numpy(xyt_np[i:j]).to(device=device, dtype=torch.float32)
            out = model(xyt, mask_indices=mask_indices).detach().cpu().numpy()
            pred[i:j] = out
    return pred.reshape(len(x_vals), len(y_vals))


def _plot_xy_comparison(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    exact_u: np.ndarray,
    pred_u: np.ndarray,
    rel_l2_global: float,
    t_value: float,
    save_path: Path,
) -> None:
    Xg, Yg = np.meshgrid(x_vals, y_vals, indexing="ij")
    err = np.abs(pred_u - exact_u)
    rel_l2_slice = float(np.linalg.norm(pred_u - exact_u) / (np.linalg.norm(exact_u) + 1e-12))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    cs0 = axes[0].contourf(Xg, Yg, exact_u, levels=60, cmap=CMAP)
    axes[0].set_title(f"Exact (t={t_value:.2f})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(cs0, ax=axes[0])

    cs1 = axes[1].contourf(Xg, Yg, pred_u, levels=60, cmap=CMAP)
    axes[1].set_title(f"Predicted (t={t_value:.2f})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(cs1, ax=axes[1])

    cs2 = axes[2].contourf(Xg, Yg, err, levels=60, cmap=CMAP)
    axes[2].set_title(f"|Pred-Exact| (t={t_value:.2f})")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(cs2, ax=axes[2])

    fig.suptitle(
        "Burgers2D Comparison "
        f"| Slice Rel L2={rel_l2_slice:.4e} | Global Rel L2={rel_l2_global:.4e}"
    )
    _finalize(save_path)


def save_equation_plots(
    equation,
    model,
    mask_indices,
    device: torch.device,
    stage_dir: Path,
    rel_l2: float,
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(equation, Burgers1DEquation):
        x_test = torch.linspace(-1.0, 1.0, equation.test_nx, device=device)
        t_test = torch.linspace(0.0, 1.0, equation.test_nt, device=device)
        pred = _predict_1d_field(model, mask_indices, x_test, t_test)
        x_np = x_test.detach().cpu().numpy()
        t_np = t_test.detach().cpu().numpy()
        exact = equation._reference_solution_fd(x_np, t_np)

        _plot_xt_comparison(
            x_vals=x_np,
            t_vals=t_np,
            exact_u=exact,
            pred_u=pred,
            rel_l2=rel_l2,
            save_path=stage_dir / "result_comparison.png",
            title_prefix=f"Burgers1D ({equation.case_label()})",
        )
        _plot_burgers1d_time_slices(
            x_vals=x_np,
            t_vals=t_np,
            exact_u=exact,
            pred_u=pred,
            save_path=stage_dir / "burgers1d_time_slices_exact_vs_pred.png",
        )
        return

    if isinstance(equation, Advection1DEquation):
        x_test = torch.linspace(0.0, 1.0, equation.test_nx, device=device)
        t_test = torch.linspace(0.0, 2.0, equation.test_nt, device=device)
        pred = _predict_1d_field(model, mask_indices, x_test, t_test)

        Xg, Tg = torch.meshgrid(x_test, t_test, indexing="ij")
        exact_t = equation.exact_solution(equation.beta, Xg, Tg)
        exact = exact_t.detach().cpu().numpy()

        _plot_xt_comparison(
            x_vals=x_test.detach().cpu().numpy(),
            t_vals=t_test.detach().cpu().numpy(),
            exact_u=exact,
            pred_u=pred,
            rel_l2=rel_l2,
            save_path=stage_dir / "result_comparison.png",
            title_prefix=f"Advection1D ({equation.case_label()})",
        )
        return

    if isinstance(equation, Burgers2DEquation):
        plot_grid = int(min(200, equation.test_nx, equation.test_ny))
        x_vals = np.linspace(0.0, 1.0, plot_grid, dtype=np.float64)
        y_vals = np.linspace(0.0, 1.0, plot_grid, dtype=np.float64)

        for t_value in (0.0, 1.0, 2.0):
            pred = _predict_2d_slice(model, mask_indices, x_vals, y_vals, t_value, device=device)
            Xg, Yg = np.meshgrid(x_vals, y_vals, indexing="ij")
            exact = 1.0 / (1.0 + np.exp((Xg + Yg - t_value) / 0.2))
            _plot_xy_comparison(
                x_vals=x_vals,
                y_vals=y_vals,
                exact_u=exact,
                pred_u=pred,
                rel_l2_global=rel_l2,
                t_value=t_value,
                save_path=stage_dir / f"slice_t_{t_value:.2f}_comparison.png",
            )

        # Convenience link: keep one default comparison name.
        first = stage_dir / "slice_t_1.00_comparison.png"
        if first.exists():
            data = first.read_bytes()
            (stage_dir / "result_comparison.png").write_bytes(data)
        return

    raise TypeError(f"Unsupported equation type for plotting: {type(equation).__name__}")
