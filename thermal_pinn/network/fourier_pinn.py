"""
thermal_pinn/network/fourier_pinn.py
=====================================
IC-consistent residual PINN with Fourier Feature Embedding (v2).

Motivation
----------
Standard PINNs suffer from *spectral bias* — the network preferentially learns
low-frequency components. For the quenching problem the rapid early-time
cooling (t < 5 s) and the curved boundaries of cylinder / L-shape domains
introduce high-frequency components that a plain MLP under-represents.

Fourier feature embedding (Tancik et al., 2020) maps the inputs through:

    γ(v) = [sin(2π·B·v), cos(2π·B·v)]    B ∈ ℝ^{d_in × n_fourier} ~ N(0, σ²)

before feeding them to the MLP. The random matrix B is fixed during training
(not a learned parameter). This lifts the network out of the spectral-bias
regime and is particularly effective for NSGA-II / NSGA-III architectures
whose smaller capacity makes them most susceptible to spectral bias.

IC-consistent output is preserved:
    output = θ_ic + τ · net(γ(coords, τ, θ_ic))

At τ=0 the output is exactly θ_ic regardless of net(·).

Level 9 results (thesis-pinn-fem/level9_sa_fourier):
    NSGA-III 2D: 7.19× L2 improvement over plain MLP
    NSGA-II  2D: 2.04× L2 improvement
    Bayesian 2D: 1.19× L2 improvement

References:
    [1] Tancik, M. et al. (2020). Fourier features let networks learn high
        frequency functions in low dimensional domains. NeurIPS 2020.
    [2] Mortensen et al. (2026). DOI: 10.1007/s00170-026-17515-w
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from .pinn import ARCH_CONFIGS, _ACT_MAP

# Default Fourier hyperparameters (tuned on 2D annulus / 3D cylinder)
N_FOURIER_DEFAULT = 64
SIGMA_2D          = 1.0
SIGMA_3D          = 1.5     # slightly wider bandwidth helps for 3D curved geometry


class FourierEmbedding(nn.Module):
    """
    Random Fourier feature layer.

    Transforms input v ∈ ℝ^d → [sin(2π·B·v), cos(2π·B·v)] ∈ ℝ^{2F}.

    The projection matrix B is sampled once at construction and kept fixed.
    Using register_buffer ensures B is moved to the correct device with the
    module, and is included in state_dict for reproducibility.

    Parameters
    ----------
    in_dim   : number of input features
    n_fourier: number of random frequencies F  (output dim = 2F)
    sigma    : standard deviation of the random projection N(0, σ²)
    seed     : RNG seed for reproducible B matrix
    """

    def __init__(self, in_dim: int, n_fourier: int = N_FOURIER_DEFAULT,
                 sigma: float = SIGMA_2D, seed: int = 42):
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)
        B = torch.randn(in_dim, n_fourier, generator=rng) * sigma
        self.register_buffer("B", B)          # (in_dim, n_fourier) — fixed
        self.out_dim = 2 * n_fourier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, in_dim) → (N, 2·n_fourier)"""
        proj = (2.0 * math.pi) * (x @ self.B)   # (N, n_fourier)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)


class FourierPINN(nn.Module):
    """
    IC-consistent residual PINN with Fourier feature embedding (v2).

    Architecture
    ------------
    Input → FourierEmbedding → MLP → δθ → θ_ic + τ · δθ

    The Fourier embedding replaces the raw input to the MLP.
    All spatial coordinates, τ, and θ_ic are embedded jointly.

    The MLP hidden layers and activations are identical to ThermalPINN
    (same NAS-discovered configurations: bayesian / nsga2 / nsga3).

    Parameters
    ----------
    dim       : 2 for 2D, 3 for 3D
    arch      : "bayesian" | "nsga2" | "nsga3"
    n_fourier : number of random Fourier frequencies (default 64)
    sigma     : bandwidth of the random projection (default 1.0 for 2D, 1.5 for 3D)
    """

    def __init__(self, dim: int = 2, arch: str = "nsga2",
                 n_fourier: int = N_FOURIER_DEFAULT,
                 sigma: float | None = None):
        super().__init__()

        if sigma is None:
            sigma = SIGMA_3D if dim == 3 else SIGMA_2D

        cfg   = dict(ARCH_CONFIGS[arch])
        self.dim  = dim
        self.arch = arch

        # Input dimension before embedding: dim spatial + 1 τ + 1 θ_ic
        in_dim = dim + 2

        self.fourier = FourierEmbedding(in_dim, n_fourier=n_fourier, sigma=sigma)
        embed_dim    = self.fourier.out_dim    # 2 * n_fourier

        act_cls = _ACT_MAP.get(cfg["activation"], nn.Tanh)
        layers  = []
        prev    = embed_dim
        for width in cfg["neurons"]:
            layers += [nn.Linear(prev, width), act_cls()]
            prev = width
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)
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
        Parameters
        ----------
        coords   : (N, dim) — normalised spatial coords in [0, 1]^dim
        tau      : (N, 1)   — normalised local time τ ∈ [0, 1]
        theta_ic : (N, 1)   — dimensionless IC temperature θ ∈ [0, 1]

        Returns
        -------
        theta_next : (N, 1) — clamped to [0, 1]
        """
        x_raw = torch.cat([coords, tau, theta_ic], dim=1)   # (N, dim+2)
        x_emb = self.fourier(x_raw)                          # (N, 2F)
        delta  = self.net(x_emb)                             # (N, 1)
        theta_next = theta_ic + tau * delta
        return torch.clamp(theta_next, 0.0, 1.0)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        cfg = ARCH_CONFIGS[self.arch]
        return (f"FourierPINN(dim={self.dim}, arch={self.arch}, "
                f"fourier={self.fourier.out_dim//2}×2={self.fourier.out_dim}, "
                f"layers={cfg['n_layers']}×{cfg['neurons'][0]}, "
                f"act={cfg['activation']}, params={self.n_params():,})")


def make_fourier_pinn(dim: int = 2, arch: str = "nsga2",
                      n_fourier: int = N_FOURIER_DEFAULT,
                      sigma: float | None = None,
                      device: torch.device | None = None) -> FourierPINN:
    """Convenience constructor — creates and moves to device."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return FourierPINN(dim=dim, arch=arch,
                       n_fourier=n_fourier, sigma=sigma).to(device)
