from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


MASK_LEVELS: Tuple[int, ...] = (20, 40, 64, 96, 128, 192)
POISSON_MASK_LEVELS: Tuple[int, ...] = (30, 50, 70, 90, 110)


@dataclass(frozen=True)
class StageConfig:
    epochs: int
    inner_lr: float
    outer_lr: float
    outer_every: int
    lbfgs_max_iter: int
    pso_iters: int
    pso_swarm: int
    pso_span: float


@dataclass(frozen=True)
class SearchConfig:
    proxy_epochs: int
    pop_size: int
    n_gen: int
    ref_partitions: int
    bo_init_points: int
    bo_iters: int


@dataclass(frozen=True)
class EquationConfig:
    name: str
    input_dim: int
    hidden_layers: int
    base_neurons: int
    mask_levels: Tuple[int, ...]
    lambda_pde: float
    lambda_ic: float
    lambda_bc: float
    repeats: int
    base_seed: int
    stage: StageConfig
    search: SearchConfig


BURGERS1D = EquationConfig(
    name="burgers1d",
    input_dim=2,
    hidden_layers=5,
    base_neurons=128,
    mask_levels=MASK_LEVELS,
    lambda_pde=1.0,
    lambda_ic=100.0,
    lambda_bc=100.0,
    repeats=5,
    base_seed=42,
    stage=StageConfig(
        epochs=30000,
        inner_lr=1e-3,
        outer_lr=3e-4,
        outer_every=5,
        lbfgs_max_iter=6000,
        pso_iters=8,
        pso_swarm=16,
        pso_span=0.25,
    ),
    search=SearchConfig(
        proxy_epochs=400,
        pop_size=12,
        n_gen=6,
        ref_partitions=12,
        bo_init_points=4,
        bo_iters=12,
    ),
)


ADVECTION1D = EquationConfig(
    name="advection1d",
    input_dim=2,
    hidden_layers=4,
    base_neurons=128,
    mask_levels=MASK_LEVELS,
    lambda_pde=1.0,
    lambda_ic=100.0,
    lambda_bc=10.0,
    repeats=5,
    base_seed=42,
    stage=StageConfig(
        epochs=30000,
        inner_lr=1e-3,
        outer_lr=3e-4,
        outer_every=5,
        lbfgs_max_iter=6000,
        pso_iters=8,
        pso_swarm=16,
        pso_span=0.25,
    ),
    search=SearchConfig(
        proxy_epochs=300,
        pop_size=12,
        n_gen=6,
        ref_partitions=12,
        bo_init_points=4,
        bo_iters=12,
    ),
)


BURGERS2D = EquationConfig(
    name="burgers2d",
    input_dim=3,
    hidden_layers=5,
    base_neurons=128,
    mask_levels=MASK_LEVELS,
    lambda_pde=1.0,
    lambda_ic=100.0,
    lambda_bc=100.0,
    repeats=5,
    base_seed=42,
    stage=StageConfig(
        epochs=30000,
        inner_lr=1e-3,
        outer_lr=3e-4,
        outer_every=5,
        lbfgs_max_iter=6000,
        pso_iters=8,
        pso_swarm=16,
        pso_span=0.25,
    ),
    search=SearchConfig(
        proxy_epochs=200,
        pop_size=12,
        n_gen=6,
        ref_partitions=12,
        bo_init_points=4,
        bo_iters=12,
    ),
)


POISSON = EquationConfig(
    name="poisson",
    input_dim=2,
    hidden_layers=5,
    base_neurons=110,
    mask_levels=POISSON_MASK_LEVELS,
    lambda_pde=1.0,
    lambda_ic=0.0,
    lambda_bc=1.0,
    repeats=1,
    base_seed=42,
    stage=StageConfig(
        epochs=30000,
        inner_lr=1e-3,
        outer_lr=3e-4,
        outer_every=5,
        lbfgs_max_iter=6000,
        pso_iters=8,
        pso_swarm=16,
        pso_span=0.25,
    ),
    search=SearchConfig(
        proxy_epochs=600,
        pop_size=30,
        n_gen=20,
        ref_partitions=12,
        bo_init_points=4,
        bo_iters=12,
    ),
)


EQUATION_CONFIGS = {
    BURGERS1D.name: BURGERS1D,
    ADVECTION1D.name: ADVECTION1D,
    BURGERS2D.name: BURGERS2D,
    POISSON.name: POISSON,
}


# Paper protocol sweeps
BURGERS1D_NU_LIST = (0.01, 0.04, 0.07)
ADVECTION1D_BETA_LIST = (1.0, 0.5, 0.1)
POISSON_DOMAIN_LIST = ("rectangular", "circle", "lshape", "flower", "annulus")
