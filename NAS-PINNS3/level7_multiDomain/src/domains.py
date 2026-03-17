"""
domains.py — Multi-Domain Poisson Geometry Definitions
=======================================================
Three 2-D domains for the Poisson problem  -Δu = f :

  CircleDomain   x² + y² ≤ 1
                 exact: u = (1 − x² − y²)/4,  f = 1

  LShapeDomain   [0,1]^2 \\ [0.5,1]x[0,0.5]
                 f = 1,  u = 0 on all boundaries  (no closed-form)

  FlowerDomain   r ≤ R(θ) = 1 + 0.3·cos(5θ)   (5 petals)
                 f = 1,  u = 0 on petal boundary  (no closed-form)

For domains without a closed-form, L2 is measured against a high-accuracy
"reference PINN" trained beforehand (see nas_search_7b.py).
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# ── Base class ────────────────────────────────────────────────────────────────

class BaseDomain:
    name: str = "base"
    has_exact: bool = False      # True if u_exact is available analytically
    _rng: np.random.Generator

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    # --- subclasses must implement ------------------------------------------
    def is_inside(self, xy: np.ndarray) -> np.ndarray:
        """xy: [N,2]  →  bool [N]"""
        raise NotImplementedError

    def sample_boundary(self, n: int) -> np.ndarray:
        """Return [n, 2] points on ∂Ω."""
        raise NotImplementedError

    def f_rhs(self, xy: np.ndarray) -> np.ndarray:
        """Right-hand side f(x,y) for -Δu = f.  Returns [N]."""
        raise NotImplementedError

    def u_exact(self, xy: np.ndarray) -> Optional[np.ndarray]:
        """Exact solution (or None if unavailable).  Returns [N]."""
        return None

    # --- provided ------------------------------------------------------------
    def sample_interior(self, n: int) -> np.ndarray:
        """Rejection-sample n points strictly inside Ω."""
        pts = []
        batch = max(n * 4, 1024)
        while len(pts) < n:
            cands = self._sample_bbox(batch)
            mask  = self.is_inside(cands)
            pts.extend(cands[mask].tolist())
        return np.array(pts[:n], dtype=np.float32)

    def _sample_bbox(self, n: int) -> np.ndarray:
        """Uniform samples from bounding box — override if needed."""
        lo, hi = self.bbox()
        return self._rng.uniform(lo, hi, (n, 2)).astype(np.float32)

    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """Bounding box [lo, hi] for rejection sampling."""
        raise NotImplementedError


# ── Circle  x²+y² ≤ 1 ────────────────────────────────────────────────────────

class CircleDomain(BaseDomain):
    """
    Unit disk.   -Δu = 1,  u = 0 on r=1.
    Exact solution:  u = (1 − x² − y²) / 4
    """
    name      = "circle"
    has_exact = True
    R         = 1.0

    def bbox(self):
        return np.array([-self.R, -self.R]), np.array([self.R, self.R])

    def is_inside(self, xy):
        return (xy[:, 0]**2 + xy[:, 1]**2) <= self.R**2

    def sample_boundary(self, n):
        theta = self._rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        x = self.R * np.cos(theta)
        y = self.R * np.sin(theta)
        return np.stack([x, y], axis=1)

    def f_rhs(self, xy):
        return np.ones(len(xy), dtype=np.float32)

    def u_exact(self, xy):
        return (1.0 - xy[:, 0]**2 - xy[:, 1]**2) / 4.0


# ── L-shape  [0,1]² \ [0.5,1]×[0,0.5] ───────────────────────────────────────

class LShapeDomain(BaseDomain):
    """
    L-shape domain.  -Δu = 1,  u = 0 on all boundary segments.
    No closed-form solution.
    Boundary: 6 line segments forming the L outline.
    """
    name      = "lshape"
    has_exact = False

    def bbox(self):
        return np.array([0.0, 0.0]), np.array([1.0, 1.0])

    def is_inside(self, xy):
        x, y = xy[:, 0], xy[:, 1]
        in_full = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
        in_cut  = (x >= 0.5) & (x <= 1.0) & (y >= 0.0) & (y <= 0.5)
        return in_full & ~in_cut

    def sample_boundary(self, n):
        """
        6 boundary segments of the L-shape:
          Bottom outer: (0,0)–(1,0)  — but x in [0.5,1] is cut boundary
          We handle the full boundary properly.

        Segments (parametric):
          S1: (0,0)→(0.5,0)  bottom-left    length 0.5
          S2: (0.5,0)→(0.5,0.5)  inner-vert  length 0.5
          S3: (0.5,0.5)→(1,0.5)  inner-horiz length 0.5
          S4: (1,0.5)→(1,1)     right        length 0.5
          S5: (1,1)→(0,1)       top          length 1.0
          S6: (0,1)→(0,0)       left         length 1.0
        Total length = 4.0
        """
        lengths = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 1.0])
        total   = lengths.sum()
        ns      = np.round(lengths / total * n).astype(int)
        ns[-1] += n - ns.sum()   # fix rounding

        segs = [
            ((0.0, 0.0),  (0.5, 0.0)),
            ((0.5, 0.0),  (0.5, 0.5)),
            ((0.5, 0.5),  (1.0, 0.5)),
            ((1.0, 0.5),  (1.0, 1.0)),
            ((1.0, 1.0),  (0.0, 1.0)),
            ((0.0, 1.0),  (0.0, 0.0)),
        ]
        pts = []
        for k, ((x0, y0), (x1, y1)) in enumerate(segs):
            t = self._rng.uniform(0, 1, ns[k]).astype(np.float32)
            pts.append(np.stack([x0 + t * (x1 - x0),
                                  y0 + t * (y1 - y0)], axis=1))
        return np.concatenate(pts, axis=0)[:n]

    def f_rhs(self, xy):
        return np.ones(len(xy), dtype=np.float32)

    def u_exact(self, xy):
        return None


# ── Flower  r ≤ 1 + 0.3·cos(5θ)  (5 petals) ─────────────────────────────────

class FlowerDomain(BaseDomain):
    """
    Flower-shaped domain with 5 petals.
    Boundary:  r = R(θ) = 1 + A·cos(n_petals·θ)  with A=0.3, n_petals=5.
    Interior:  r < R(θ).
    -Δu = 1,  u = 0 on petal boundary.  No closed-form solution.
    """
    name      = "flower"
    has_exact = False
    A         = 0.3
    N_PETALS  = 5
    R_MAX     = 1.0 + 0.3   # = 1.3

    def R_boundary(self, theta: np.ndarray) -> np.ndarray:
        return 1.0 + self.A * np.cos(self.N_PETALS * theta)

    def bbox(self):
        return np.array([-self.R_MAX, -self.R_MAX]), \
               np.array([ self.R_MAX,  self.R_MAX])

    def is_inside(self, xy):
        x, y    = xy[:, 0], xy[:, 1]
        r       = np.sqrt(x**2 + y**2)
        theta   = np.arctan2(y, x)
        r_bound = self.R_boundary(theta)
        return r <= r_bound

    def sample_boundary(self, n):
        """Parametric sampling along the flower boundary curve."""
        theta = self._rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        r     = self.R_boundary(theta).astype(np.float32)
        x     = r * np.cos(theta)
        y     = r * np.sin(theta)
        return np.stack([x, y], axis=1)

    def sample_interior(self, n):
        """
        Polar sampling: sample θ uniformly, r uniformly in [0, R(θ)].
        More efficient than rejection sampling for the flower shape.
        """
        theta = self._rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        r_max = self.R_boundary(theta).astype(np.float32)
        r     = (self._rng.uniform(0, 1, n) * r_max).astype(np.float32)
        x     = r * np.cos(theta)
        y     = r * np.sin(theta)
        return np.stack([x, y], axis=1)

    def f_rhs(self, xy):
        return np.ones(len(xy), dtype=np.float32)

    def u_exact(self, xy):
        return None


# ── Factory ───────────────────────────────────────────────────────────────────

DOMAINS = {
    "circle":  CircleDomain,
    "lshape":  LShapeDomain,
    "flower":  FlowerDomain,
}


def get_domain(name: str, seed: int = 0) -> BaseDomain:
    if name not in DOMAINS:
        raise ValueError(f"Unknown domain '{name}'. Choose from {list(DOMAINS)}")
    return DOMAINS[name](seed=seed)
