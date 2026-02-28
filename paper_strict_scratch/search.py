from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from .config import EquationConfig, MASK_LEVELS
from .model import SearchPINN
from .trainer import compute_total_loss, set_seed


@dataclass
class SearchOutcome:
    masks: Sequence[int]
    proxy_loss: float
    effective_params: int


def _effective_param_count(cfg: EquationConfig, masks: Sequence[int], mask_levels: Sequence[int]) -> int:
    prev = int(cfg.input_dim)
    total = 0
    for m in masks:
        width = int(mask_levels[int(m)])
        total += prev * width + width
        prev = width
    total += prev * 1 + 1
    return int(total)


def _proxy_train_loss(
    cfg: EquationConfig,
    equation,
    masks: Sequence[int],
    device: torch.device,
    seed: int,
) -> float:
    set_seed(seed)
    model = SearchPINN(
        input_dim=cfg.input_dim,
        hidden_layers=cfg.hidden_layers,
        base_neurons=cfg.base_neurons,
        mask_levels=MASK_LEVELS,
    ).to(device)

    for p in model.mask_parameters():
        p.requires_grad_(False)

    params = model.non_mask_parameters()
    opt = torch.optim.Adam(params, lr=cfg.stage.inner_lr)
    train_data = equation.sample_train(device)

    final_loss = float("inf")
    for _ in range(cfg.search.proxy_epochs):
        opt.zero_grad(set_to_none=True)
        loss, _ = compute_total_loss(model, equation, train_data, masks)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().item())

    return final_loss


def _evaluate_cached(
    cache: Dict[Tuple[int, ...], Tuple[float, int]],
    cfg: EquationConfig,
    equation,
    masks: Sequence[int],
    device: torch.device,
    seed: int,
) -> Tuple[float, int]:
    key = tuple(int(m) for m in masks)
    if key in cache:
        return cache[key]

    loss = _proxy_train_loss(cfg, equation, key, device, seed)
    nparam = _effective_param_count(cfg, key, MASK_LEVELS)
    cache[key] = (loss, nparam)
    return loss, nparam


def _pick_best_from_cache(cache: Dict[Tuple[int, ...], Tuple[float, int]]) -> SearchOutcome:
    if not cache:
        raise RuntimeError("architecture search cache is empty")
    best_masks, (best_loss, best_params) = min(cache.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    return SearchOutcome(masks=list(best_masks), proxy_loss=float(best_loss), effective_params=int(best_params))


def run_nsga2(
    cfg: EquationConfig,
    equation,
    device: torch.device,
    seed: int,
) -> SearchOutcome:
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
    except Exception as exc:
        raise ImportError("pymoo is required for NSGA-II search") from exc

    cache: Dict[Tuple[int, ...], Tuple[float, int]] = {}
    n_layers = int(cfg.hidden_layers)
    n_levels = len(MASK_LEVELS)

    class MaskProblem(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(n_var=n_layers, n_obj=2, n_ieq_constr=0, xl=0, xu=n_levels - 1)

        def _evaluate(self, x, out, *args, **kwargs):
            masks = [int(np.clip(np.rint(v), 0, n_levels - 1)) for v in x]
            eval_seed = int(seed + 997 * sum((i + 1) * m for i, m in enumerate(masks)))
            loss, nparam = _evaluate_cached(cache, cfg, equation, masks, device, eval_seed)
            out["F"] = np.array([loss, float(nparam)], dtype=np.float64)

    print(f"Starting NSGA-II architecture search: case={equation.case_label()}")
    problem = MaskProblem()
    algorithm = NSGA2(pop_size=cfg.search.pop_size)
    termination = get_termination("n_gen", cfg.search.n_gen)

    res = minimize(problem, algorithm, termination, seed=seed, save_history=False, verbose=True)

    if res.X is not None and res.F is not None:
        X = np.atleast_2d(res.X)
        F = np.atleast_2d(res.F)
        best_idx = int(np.lexsort((F[:, 1], F[:, 0]))[0])
        masks = [int(np.clip(np.rint(v), 0, n_levels - 1)) for v in X[best_idx]]
        loss, nparam = _evaluate_cached(cache, cfg, equation, masks, device, seed + 99991)
        return SearchOutcome(masks=masks, proxy_loss=loss, effective_params=nparam)

    return _pick_best_from_cache(cache)


def run_nsga3(
    cfg: EquationConfig,
    equation,
    device: torch.device,
    seed: int,
) -> SearchOutcome:
    try:
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
        from pymoo.util.ref_dirs import get_reference_directions
    except Exception as exc:
        raise ImportError("pymoo is required for NSGA-III search") from exc

    cache: Dict[Tuple[int, ...], Tuple[float, int]] = {}
    n_layers = int(cfg.hidden_layers)
    n_levels = len(MASK_LEVELS)

    class MaskProblem(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(n_var=n_layers, n_obj=2, n_ieq_constr=0, xl=0, xu=n_levels - 1)

        def _evaluate(self, x, out, *args, **kwargs):
            masks = [int(np.clip(np.rint(v), 0, n_levels - 1)) for v in x]
            eval_seed = int(seed + 997 * sum((i + 1) * m for i, m in enumerate(masks)))
            loss, nparam = _evaluate_cached(cache, cfg, equation, masks, device, eval_seed)
            out["F"] = np.array([loss, float(nparam)], dtype=np.float64)

    print(f"Starting NSGA-III architecture search: case={equation.case_label()}")
    problem = MaskProblem()
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=cfg.search.ref_partitions)
    algorithm = NSGA3(pop_size=max(cfg.search.pop_size, len(ref_dirs)), ref_dirs=ref_dirs)
    termination = get_termination("n_gen", cfg.search.n_gen)

    res = minimize(problem, algorithm, termination, seed=seed, save_history=False, verbose=True)

    if res.X is not None and res.F is not None:
        X = np.atleast_2d(res.X)
        F = np.atleast_2d(res.F)
        best_idx = int(np.lexsort((F[:, 1], F[:, 0]))[0])
        masks = [int(np.clip(np.rint(v), 0, n_levels - 1)) for v in X[best_idx]]
        loss, nparam = _evaluate_cached(cache, cfg, equation, masks, device, seed + 99991)
        return SearchOutcome(masks=masks, proxy_loss=loss, effective_params=nparam)

    return _pick_best_from_cache(cache)


def run_bayesian(
    cfg: EquationConfig,
    equation,
    device: torch.device,
    seed: int,
) -> SearchOutcome:
    try:
        from bayes_opt import BayesianOptimization
    except Exception as exc:
        raise ImportError("bayesian-optimization is required for Bayesian architecture search") from exc

    cache: Dict[Tuple[int, ...], Tuple[float, int]] = {}
    n_layers = int(cfg.hidden_layers)
    n_levels = len(MASK_LEVELS)

    pbounds = {f"m{i}": (0.0, float(n_levels - 1)) for i in range(n_layers)}

    def objective(**kwargs) -> float:
        masks = [int(np.clip(np.rint(kwargs[f"m{i}"]), 0, n_levels - 1)) for i in range(n_layers)]
        eval_seed = int(seed + 997 * sum((i + 1) * m for i, m in enumerate(masks)))
        loss, nparam = _evaluate_cached(cache, cfg, equation, masks, device, eval_seed)
        # Maximize negative objective.
        return -(loss + 1e-7 * float(nparam))

    print(f"Starting Bayesian architecture search: case={equation.case_label()}")
    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=seed, verbose=2)
    optimizer.maximize(init_points=cfg.search.bo_init_points, n_iter=cfg.search.bo_iters)

    return _pick_best_from_cache(cache)


def search_architecture(
    method: str,
    cfg: EquationConfig,
    equation,
    device: torch.device,
    seed: int,
) -> SearchOutcome:
    method_l = method.lower()
    if method_l == "nsga2":
        return run_nsga2(cfg, equation, device, seed)
    if method_l == "nsga3":
        return run_nsga3(cfg, equation, device, seed)
    if method_l == "bayesian":
        return run_bayesian(cfg, equation, device, seed)
    raise ValueError(f"Unsupported search method: {method}")
