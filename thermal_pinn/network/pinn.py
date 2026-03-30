"""
thermal_pinn/network/pinn.py
=============================
IC-consistent residual PINN for transient heat conduction.

Architecture motivation
-----------------------
Standard PINNs (Raissi et al., 2019) struggle with the quenching problem
because the initial condition (T=540°C, uniform) and the boundary condition
(Robin, T_surface → T_water) create a steep transient gradient in the first
few seconds. A flat network typically fits neither the early-time behaviour
nor the long-time decay simultaneously.

We adopt the IC-consistent residual design from the Level 8 MCO-PINN
experiments, which was itself inspired by the time-decomposition idea in
Wang & Zhong (2023). The key idea:

    output = θ_ic + τ · net(inputs)

where τ = t_local / dt_window ∈ [0,1] is the normalised local time, and
θ_ic is the dimensionless initial temperature at the anchor point. At τ=0
the output exactly equals θ_ic regardless of the network weights — the
IC loss term is identically zero by construction. The network only needs
to learn the *change* from the initial condition, which is a much smoother
function and converges faster.

Skip parameter k
----------------
The PINN covers a time window of width k·Δt_single, where Δt_single is the
nominal FEM time step and k ∈ {1, 2, 3, 4, 5} is the skip factor. Larger k
means fewer FEM anchor points are needed (cheaper overall) but the PINN must
extrapolate further, which increases error. The optimal k is problem-dependent
and is found by the sweep in training/train_all.py.

Network inputs (2D mode — 4 inputs):
    (x_norm, y_norm, τ, θ_ic)

Network inputs (3D mode — 5 inputs):
    (x_norm, y_norm, z_norm, τ, θ_ic)

Network output:
    θ_next ∈ [0, 1]  (dimensionless temperature, clipped for stability)

NAS architectures
-----------------
Three architectures from Wang & Zhong (2023) extended to the quenching problem
via Bayesian, NSGA-II, and NSGA-III search (see training/train_all.py):

    bayesian : 5 hidden layers × 151 neurons, ReLU
    nsga2    : 3 hidden layers × 153 neurons, tanh
    nsga3    : 3 hidden layers × 75 neurons,  tanh

References:
    [1] Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019).
        Physics-informed neural networks. J. Comput. Phys., 378, 686–707.
        DOI: 10.1016/j.jcp.2018.10.045
    [2] Wang, Y., & Zhong, L. (2023). NAS-PINN: Neural Architecture
        Search-Guided PINN for Solving PDEs. arXiv:2305.10127.
    [3] Mortensen et al. (2026). DOI: 10.1007/s00170-026-17515-w
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
# NAS-optimised architecture configurations
# Source: Wang & Zhong (2023), extended via Bayesian / NSGA-II / NSGA-III
# ──────────────────────────────────────────────────────────────────────────────
ARCH_CONFIGS: dict[str, dict] = {
    "bayesian": {
        "n_layers":   5,
        "neurons":    [151, 151, 151, 151, 151],
        "activation": "relu",
        "label":      "Bayesian Optimisation (TPE)",
    },
    "nsga2": {
        "n_layers":   3,
        "neurons":    [153, 153, 153],
        "activation": "tanh",
        "label":      "NSGA-II (multi-objective evolutionary)",
    },
    "nsga3": {
        "n_layers":   3,
        "neurons":    [75, 75, 75],
        "activation": "tanh",
        "label":      "NSGA-III (reference-point evolutionary)",
    },
}

_ACT_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


class ThermalPINN(nn.Module):
    """
    IC-consistent residual network for a single time window [t_start, t_end].

    The network is stateless across windows — each window is a fresh forward
    pass with the FEM anchor field as a conditioning input (θ_ic). This is the
    key difference from a rolling-step PINN: we always restart from a trusted
    FEM solution, so errors do not accumulate across windows.

    Parameters
    ----------
    dim : int
        Spatial dimension: 2 for 2D problems, 3 for 3D problems.
    arch : str
        Architecture name: "bayesian", "nsga2", or "nsga3".
    config : dict, optional
        Override specific architecture parameters.
    """

    def __init__(self, dim: int = 2, arch: str = "nsga2",
                 config: dict | None = None):
        super().__init__()

        cfg = dict(ARCH_CONFIGS[arch])
        if config:
            cfg.update(config)

        self.dim   = dim
        self.arch  = arch
        n_in       = dim + 2   # spatial coords + τ + θ_ic

        act_cls = _ACT_MAP.get(cfg["activation"], nn.Tanh)
        layers  = []
        prev    = n_in
        for width in cfg["neurons"]:
            layers += [nn.Linear(prev, width), act_cls()]
            prev = width
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)

        # Xavier initialisation: better gradient flow for PINN training
        # than the default Kaiming uniform. Empirically validated for
        # tanh-activated networks — see Lu et al. (2021), DeepXDE.
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor,
                tau: torch.Tensor,
                theta_ic: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        coords : (N, dim) — normalised spatial coordinates in [0, 1]^dim
        tau    : (N, 1)   — normalised local time τ = t_local / dt_window ∈ [0,1]
        theta_ic : (N, 1) — dimensionless IC temperature θ = (T_ic − T_w) / ΔT

        Returns
        -------
        theta_next : (N, 1) — dimensionless predicted temperature, clamped to [0,1]
        """
        x = torch.cat([coords, tau, theta_ic], dim=1)
        delta = self.net(x)
        # IC-consistent output: at τ=0 the correction τ·net(x) is zero,
        # so the output exactly equals θ_ic — satisfying the IC by construction.
        theta_next = theta_ic + tau * delta
        return torch.clamp(theta_next, 0.0, 1.0)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        cfg  = ARCH_CONFIGS[self.arch]
        return (f"ThermalPINN(dim={self.dim}, arch={self.arch}, "
                f"layers={cfg['n_layers']}×{cfg['neurons'][0]}, "
                f"act={cfg['activation']}, params={self.n_params():,})")


def make_pinn(dim: int = 2, arch: str = "nsga2",
              device: torch.device | None = None) -> ThermalPINN:
    """Convenience constructor — creates and moves to device."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ThermalPINN(dim=dim, arch=arch).to(device)
