from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import StageConfig
from .model import SearchPINN, clone_model_state, load_model_state


@dataclass
class AdamOutcome:
    best_state: Dict[str, torch.Tensor]
    best_loss: float
    history: List[float]
    elapsed_sec: float


@dataclass
class StageOutcome:
    train_loss: float
    rel_l2: float
    elapsed_sec: float
    history: List[float]


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def compute_total_loss(
    model: SearchPINN,
    equation,
    train_data,
    mask_indices: Optional[Sequence[int]],
) -> Tuple[torch.Tensor, Tuple[float, float, float]]:
    l_pde, l_ic, l_bc = equation.loss_components(model, train_data, mask_indices)
    total = equation.weighted_loss(l_pde, l_ic, l_bc)
    return total, (float(l_pde.item()), float(l_ic.item()), float(l_bc.item()))


def _set_requires_grad(params: Sequence[torch.nn.Parameter], enabled: bool) -> None:
    for p in params:
        p.requires_grad_(enabled)


def _flatten_params(params: Sequence[torch.nn.Parameter]) -> np.ndarray:
    parts = [p.detach().cpu().numpy().reshape(-1) for p in params]
    if not parts:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(parts).astype(np.float64, copy=False)


def _set_params_from_flat(
    params: Sequence[torch.nn.Parameter],
    flat: np.ndarray,
    device: torch.device,
) -> None:
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            chunk = torch.from_numpy(flat[offset : offset + n]).view_as(p).to(device=device, dtype=p.dtype)
            p.copy_(chunk)
            offset += n


def train_adam(
    model: SearchPINN,
    equation,
    train_data,
    stage_cfg: StageConfig,
    mask_indices: Optional[Sequence[int]],
    optimize_arch: bool,
    verbose_every: int = 2000,
) -> AdamOutcome:
    mask_params = list(model.mask_parameters())
    arch_params = list(model.arch_parameters())
    arch_ids = {id(p) for p in arch_params}
    mask_ids = {id(p) for p in mask_params}

    if optimize_arch:
        inner_params = [p for p in model.parameters() if id(p) not in arch_ids]
        outer_params = arch_params
    else:
        # Fixed-mask training: keep alpha_masks frozen but allow alpha_ops to adapt.
        inner_params = [p for p in model.parameters() if id(p) not in mask_ids]
        outer_params = []
        _set_requires_grad(mask_params, False)

    opt_inner = torch.optim.Adam(inner_params, lr=stage_cfg.inner_lr)
    opt_outer = torch.optim.Adam(outer_params, lr=stage_cfg.outer_lr) if outer_params else None

    best_loss = float("inf")
    best_state = clone_model_state(model)
    history: List[float] = []

    t0 = time.perf_counter()
    for epoch in range(stage_cfg.epochs):
        if outer_params:
            _set_requires_grad(outer_params, False)

        opt_inner.zero_grad(set_to_none=True)
        loss, _ = compute_total_loss(model, equation, train_data, mask_indices)
        loss.backward()
        opt_inner.step()

        if outer_params:
            _set_requires_grad(outer_params, True)

        loss_val = float(loss.detach().item())
        history.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = clone_model_state(model)

        if opt_outer is not None and (epoch + 1) % stage_cfg.outer_every == 0:
            _set_requires_grad(inner_params, False)
            opt_outer.zero_grad(set_to_none=True)
            outer_loss, _ = compute_total_loss(model, equation, train_data, mask_indices)
            outer_loss.backward()
            opt_outer.step()
            _set_requires_grad(inner_params, True)

        if (epoch == 0) or ((epoch + 1) % max(1, verbose_every) == 0) or (epoch + 1 == stage_cfg.epochs):
            print(f"Adam [{epoch:5d}] loss: {loss_val:.4e}")

    elapsed = time.perf_counter() - t0

    # Re-enable all params for downstream stages.
    _set_requires_grad(mask_params, True)
    return AdamOutcome(best_state=best_state, best_loss=best_loss, history=history, elapsed_sec=elapsed)


def run_lbfgs(
    model: SearchPINN,
    equation,
    train_data,
    stage_cfg: StageConfig,
    mask_indices: Sequence[int],
    device: torch.device,
) -> Tuple[float, List[float], float]:
    del device
    mask_params = list(model.mask_parameters())
    _set_requires_grad(mask_params, False)

    params = [p for p in model.parameters() if p.requires_grad and id(p) not in {id(mp) for mp in mask_params}]
    if not params:
        _set_requires_grad(mask_params, True)
        return float("inf"), [], 0.0

    optimizer = torch.optim.LBFGS(params, lr=1.0, max_iter=stage_cfg.lbfgs_max_iter, line_search_fn="strong_wolfe")
    history: List[float] = []
    best_loss = float("inf")
    best_state = clone_model_state(model)

    def closure() -> torch.Tensor:
        nonlocal best_loss, best_state
        optimizer.zero_grad(set_to_none=True)
        loss, _ = compute_total_loss(model, equation, train_data, mask_indices)
        loss.backward()
        loss_val = float(loss.detach().item())
        history.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = clone_model_state(model)
        return loss

    t0 = time.perf_counter()
    optimizer.step(closure)
    elapsed = time.perf_counter() - t0

    load_model_state(model, best_state)
    _set_requires_grad(mask_params, True)
    return best_loss, history, elapsed


def run_pso(
    model: SearchPINN,
    equation,
    train_data,
    stage_cfg: StageConfig,
    mask_indices: Sequence[int],
    device: torch.device,
) -> Tuple[float, List[float], float]:
    mask_params = list(model.mask_parameters())
    _set_requires_grad(mask_params, False)

    mask_ids = {id(p) for p in mask_params}
    params = [p for p in model.parameters() if id(p) not in mask_ids]

    center = _flatten_params(params)
    if center.size == 0:
        _set_requires_grad(mask_params, True)
        return float("inf"), [], 0.0

    swarm = max(2, int(stage_cfg.pso_swarm))
    iters = max(1, int(stage_cfg.pso_iters))

    scale = float(stage_cfg.pso_span) * np.maximum(1.0, np.abs(center))
    lower = center - scale
    upper = center + scale

    def objective(vec: np.ndarray) -> float:
        _set_params_from_flat(params, vec, device)
        loss, _ = compute_total_loss(model, equation, train_data, mask_indices)
        return float(loss.detach().item())

    history: List[float] = []
    best_x = center.copy()
    best_f = float("inf")

    t0 = time.perf_counter()
    try:
        # Use pymoo's adaptive fuzzy-PSO implementation (same family as the provided code).
        from pymoo.algorithms.soo.nonconvex.pso import PSO
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize

        class WeightsProblem(ElementwiseProblem):
            def __init__(self) -> None:
                super().__init__(n_var=center.size, n_obj=1, n_ieq_constr=0, xl=lower, xu=upper)

            def _evaluate(self, x, out, *args, **kwargs):
                out["F"] = objective(np.asarray(x, dtype=np.float64))

        problem = WeightsProblem()
        algorithm = PSO(
            pop_size=swarm,
            w=0.9,
            c1=2.0,
            c2=2.0,
            adaptive=True,
            initial_velocity="random",
            max_velocity_rate=0.20,
            pertube_best=True,
        )
        res = minimize(
            problem,
            algorithm,
            termination=("n_gen", iters),
            save_history=True,
            verbose=True,
        )

        if res.X is not None and res.F is not None:
            best_x = np.asarray(res.X, dtype=np.float64)
            best_f = float(np.asarray(res.F, dtype=np.float64).reshape(-1)[0])
        if getattr(res, "history", None):
            for h in res.history:
                try:
                    fval = float(np.asarray(h.opt[0].F, dtype=np.float64).reshape(-1)[0])
                except Exception:
                    fvals = np.asarray(h.pop.get("F"), dtype=np.float64).reshape(-1)
                    fval = float(np.min(fvals)) if fvals.size else np.nan
                history.append(fval)
    except Exception:
        # Fallback to deterministic bounded PSO if pymoo PSO is unavailable.
        rng = np.random.default_rng()
        positions = rng.uniform(lower, upper, size=(swarm, center.size))
        positions[0] = center.copy()
        velocities = np.zeros_like(positions)

        personal_best_pos = positions.copy()
        personal_best_val = np.full((swarm,), np.inf, dtype=np.float64)
        global_best_pos = center.copy()
        global_best_val = float("inf")

        w = 0.729
        c1 = 1.49445
        c2 = 1.49445

        for _ in range(iters):
            for i in range(swarm):
                val = objective(positions[i])
                if val < personal_best_val[i]:
                    personal_best_val[i] = val
                    personal_best_pos[i] = positions[i].copy()
                if val < global_best_val:
                    global_best_val = val
                    global_best_pos = positions[i].copy()

            history.append(float(global_best_val))

            r1 = rng.random(size=positions.shape)
            r2 = rng.random(size=positions.shape)
            velocities = (
                w * velocities
                + c1 * r1 * (personal_best_pos - positions)
                + c2 * r2 * (global_best_pos[None, :] - positions)
            )
            positions = np.clip(positions + velocities, lower, upper)

        best_x = global_best_pos
        best_f = float(global_best_val)

    _set_params_from_flat(params, best_x, device)
    elapsed = time.perf_counter() - t0

    _set_requires_grad(mask_params, True)
    return float(best_f), history, elapsed


def evaluate_stage(
    model: SearchPINN,
    equation,
    train_data,
    stage_name: str,
    mask_indices: Sequence[int],
    device: torch.device,
    train_loss: float,
    history: List[float],
    elapsed_sec: float,
) -> StageOutcome:
    del stage_name, train_data
    rel = equation.relative_l2(model, mask_indices, device)
    return StageOutcome(train_loss=float(train_loss), rel_l2=float(rel), elapsed_sec=float(elapsed_sec), history=history)
