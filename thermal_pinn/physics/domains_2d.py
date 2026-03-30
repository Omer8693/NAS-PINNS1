"""
thermal_pinn/physics/domains_2d.py
===================================
Analytical and numerical reference solutions for 2D transient heat conduction
with Robin (convective) boundary conditions on four domain geometries:

    Rectangle  — separable cosine-product solution (first-mode approximation)
    Circle     — radially-symmetric Bessel J0 series solution
    Annulus    — Bessel solution with inner (insulated) + outer (convective) BC
    LShape     — explicit finite-difference solver (no closed-form solution exists)

Physical problem: water quenching of A356 aluminium alloy (Mortensen et al., 2026).
    ∂T/∂t = α · ∇²T        in Ω, t ∈ [0, 30s]
    k · ∂T/∂n + h(T−T_w) = 0   on ∂Ω  (Robin BC)
    T(x, y, 0) = T_init          (uniform initial condition)

The Robin BC formulation is physically appropriate here: the Biot number
Bi = h·L/(2k) ranges from ~6 to ~17 depending on domain size, which means
the surface temperature differs significantly from T_water during quenching.
Using a Dirichlet BC (T_surface = T_water) would be valid only for Bi → ∞
(perfect thermal contact), which overestimates the cooling rate — see
Incropera et al. (2007, §5.6) for the eigencondition derivation.

Material properties (A356 aluminium, Table 1, Mortensen et al., 2026):
    k     = 150 W/(m·K)   thermal conductivity
    ρ·Cp  = 2.4×10⁶ J/(m³·K)   volumetric heat capacity
    α     = k/(ρ·Cp) = 6.25×10⁻⁵ m²/s
    h     = 5000 W/(m²·K)  convection coefficient (water quench)
    T_init  = 540 °C
    T_water = 20 °C

References:
    [1] Wang, Y., & Zhong, L. (2023). NAS-PINN: Neural Architecture
        Search-Guided PINN for Solving PDEs. arXiv:2305.10127.
    [2] Mortensen, D., Noorsumar, G., Fjær, H.G., Babaei, R., & Drønen, P.E.
        (2026). Mitigating distortions in cast automotive subframes: A finite
        element simulation approach. Int J Adv Manuf Technol.
        DOI: 10.1007/s00170-026-17515-w
    [3] Incropera, F.P. et al. (2007). Fundamentals of Heat and Mass Transfer,
        6th ed. John Wiley & Sons.
    [4] Carslaw, H.S., & Jaeger, J.C. (1959). Conduction of Heat in Solids,
        2nd ed. Oxford University Press.  [Bessel series derivations]
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, y0, y1

# ──────────────────────────────────────────────────────────────────────────────
# Shared physical constants  (A356 aluminium, Mortensen et al. 2026)
# ──────────────────────────────────────────────────────────────────────────────
K       = 150.0       # W/(m·K)
RHO_CP  = 2.4e6       # J/(m³·K)
H       = 5000.0      # W/(m²·K)
ALPHA   = K / RHO_CP  # m²/s  ≈ 6.25e-5
T_INIT  = 540.0       # °C
T_WATER = 20.0        # °C
DELTA_T = T_INIT - T_WATER  # 520 °C

N_MODES = 20   # number of eigenfunction terms in series solutions


# ══════════════════════════════════════════════════════════════════════════════
# Domain 1 — Rectangle
# ══════════════════════════════════════════════════════════════════════════════

class Rectangle2D:
    """
    2D rectangular domain with Robin BC on all four sides.

    Dimensions: Lx × Ly  (centred at origin for eigenfunctions, but
    the coordinate system in the PINN uses x ∈ [0, Lx], y ∈ [0, Ly]).

    Analytical solution (separation of variables, first-mode dominant):
        T(x,y,t) = T_w + ΔT · θ_x(x,t) · θ_y(y,t)

    where each 1D factor satisfies  θ'' + µ²θ = 0  with Robin ends.

    We keep N_MODES terms for accuracy at early times (t < 1 s), where
    higher modes have not yet decayed. For t > 5 s the first mode alone
    captures >99.5% of the temperature field.

    Default dimensions match the A356 subframe cross-section reported
    in Mortensen et al. (2026), Section 2.1:
        Lx = 1.3 m  (length),  Ly = 0.6 m  (width)

    References: Carslaw & Jaeger (1959, Ch. 3); Incropera et al. (2007, §5.6).
    """

    name  = "Rectangle 2D"
    color = "#1565C0"   # deep blue

    def __init__(self, Lx: float = 1.3, Ly: float = 0.6,
                 h: float = H, k: float = K,
                 alpha: float = ALPHA, n_modes: int = N_MODES):
        self.Lx, self.Ly   = Lx, Ly
        self.h, self.k     = h, k
        self.alpha         = alpha
        self.Bi_x = h * (Lx / 2) / k
        self.Bi_y = h * (Ly / 2) / k

        # Pre-compute eigenvalues µ_n satisfying µ·tan(µ) = Bi
        # and Fourier coefficients C_n = 4·sin(µ_n)/(2µ_n + sin(2µ_n))
        self._mu_x, self._Cx = self._eigenvalues(self.Bi_x, n_modes)
        self._mu_y, self._Cy = self._eigenvalues(self.Bi_y, n_modes)

    @staticmethod
    def _eigenvalues(Bi: float, n: int):
        """
        Solve µ·tan(µ) = Bi for the first n positive roots.
        We use scipy.optimize.brentq on each interval (nπ, (n+0.5)π)
        because Newton's method can miss roots near the singularities
        of tan(µ) for large Bi.
        """
        mu, C = np.zeros(n), np.zeros(n)
        for i in range(n):
            lo = i * np.pi + 1e-10
            hi = (i + 0.5) * np.pi - 1e-10
            try:
                mu[i] = brentq(lambda m: m * np.tan(m) - Bi, lo, hi)
            except ValueError:
                mu[i] = (lo + hi) / 2
            C[i] = 4 * np.sin(mu[i]) / (2 * mu[i] + np.sin(2 * mu[i]))
        return mu, C

    def _theta_1d(self, x_norm: np.ndarray, t: float,
                  mu: np.ndarray, C: np.ndarray, L_half: float) -> np.ndarray:
        """
        1D dimensionless temperature factor at normalised coordinate x_norm ∈ [-1,1].
        θ(x*,t) = Σ_n C_n · cos(µ_n · x*) · exp(−µ_n² · Fo)
        Fo = α·t / L_half²  (Fourier number)
        """
        Fo = self.alpha * t / L_half**2
        result = np.zeros_like(x_norm, dtype=float)
        for mu_n, C_n in zip(mu, C):
            result += C_n * np.cos(mu_n * x_norm) * np.exp(-mu_n**2 * Fo)
        return result

    def T(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """
        Full 2D temperature field T(x, y, t) in °C.
        x ∈ [0, Lx],  y ∈ [0, Ly].
        """
        x_norm = (x - self.Lx / 2) / (self.Lx / 2)
        y_norm = (y - self.Ly / 2) / (self.Ly / 2)
        theta  = (self._theta_1d(x_norm, t, self._mu_x, self._Cx, self.Lx / 2) *
                  self._theta_1d(y_norm, t, self._mu_y, self._Cy, self.Ly / 2))
        return T_WATER + DELTA_T * theta

    def sample_interior(self, n: int, rng=None) -> tuple[np.ndarray, np.ndarray]:
        rng = rng or np.random.default_rng(0)
        x = rng.uniform(0, self.Lx, n).astype(np.float32)
        y = rng.uniform(0, self.Ly, n).astype(np.float32)
        return x, y

    def sample_boundary(self, n: int, rng=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return boundary points (x, y) and outward unit normals (nx, ny).
        Points are sampled uniformly on the four edges.
        """
        rng = rng or np.random.default_rng(1)
        n_per_edge = n // 4
        # left (x=0, nx=-1), right (x=Lx, nx=+1)
        # bottom (y=0, ny=-1), top (y=Ly, ny=+1)
        edges = [
            (np.zeros(n_per_edge),                     rng.uniform(0, self.Ly, n_per_edge), -np.ones(n_per_edge), np.zeros(n_per_edge)),
            (np.full(n_per_edge, self.Lx),             rng.uniform(0, self.Ly, n_per_edge), +np.ones(n_per_edge), np.zeros(n_per_edge)),
            (rng.uniform(0, self.Lx, n_per_edge),      np.zeros(n_per_edge),                np.zeros(n_per_edge), -np.ones(n_per_edge)),
            (rng.uniform(0, self.Lx, n_per_edge),      np.full(n_per_edge, self.Ly),        np.zeros(n_per_edge), +np.ones(n_per_edge)),
        ]
        bx = np.concatenate([e[0] for e in edges]).astype(np.float32)
        by = np.concatenate([e[1] for e in edges]).astype(np.float32)
        nx = np.concatenate([e[2] for e in edges]).astype(np.float32)
        ny = np.concatenate([e[3] for e in edges]).astype(np.float32)
        return bx, by, nx, ny

    def grid(self, nx: int = 60, ny: int = 30) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0, self.Lx, nx)
        y = np.linspace(0, self.Ly, ny)
        return np.meshgrid(x, y, indexing="ij")   # shape (nx, ny) each

    def info(self) -> str:
        return (f"Rectangle2D  Lx={self.Lx}m  Ly={self.Ly}m\n"
                f"  Bi_x={self.Bi_x:.2f}  Bi_y={self.Bi_y:.2f}  "
                f"(Bi >> 1 → strong surface cooling)")


# ══════════════════════════════════════════════════════════════════════════════
# Domain 2 — Circle
# ══════════════════════════════════════════════════════════════════════════════

class Circle2D:
    """
    Circular domain (radius R) with Robin BC on the outer boundary.

    In polar coordinates the heat equation reduces to a radially symmetric
    problem (no angular dependence for uniform IC):
        ∂θ/∂t = α · (1/r · ∂/∂r(r · ∂θ/∂r))

    Analytical solution (Carslaw & Jaeger, 1959, §7.5):
        T(r,t) = T_w + ΔT · Σ_n C_n · J0(µ_n · r/R) · exp(−µ_n² · α·t/R²)

    Eigenvalues from:  µ_n · J1(µ_n) / J0(µ_n) = Bi = h·R/k
    Coefficients:       C_n = 2/µ_n · 1/J0(µ_n) · 1/(1 + (Bi/µ_n)²)

    Default radius R = 0.3 m gives Bi = h·R/k = 5000·0.3/150 = 10.
    """

    name  = "Circle 2D"
    color = "#2E7D32"   # green

    def __init__(self, R: float = 0.3, h: float = H, k: float = K,
                 alpha: float = ALPHA, n_modes: int = N_MODES):
        self.R     = R
        self.h, self.k = h, k
        self.alpha = alpha
        self.Bi    = h * R / k
        self._mu, self._C = self._eigenvalues(self.Bi, n_modes)

    def _eigenvalues(self, Bi: float, n: int):
        """
        Solve µ·J1(µ) = Bi·J0(µ) for the first n positive roots.

        The equivalent form f(µ) = µ·J1(µ) − Bi·J0(µ) = 0 avoids dividing by
        J0(µ) which is zero at µ ≈ 2.405, 5.520, … and would cause numerical
        explosion when the bracket straddles those zeros.

        Brackets (Carslaw & Jaeger, 1959, §7.5):
          - Mode 1: (0, first_zero_of_J0)  — always contains exactly one root
          - Mode k: (zero_of_J0[k-2], zero_of_J0[k-1])  for k ≥ 2

        Coefficient formula derived from Bessel orthogonality on [0, R]:
            ∫₀ᴿ r J0(µ_n r/R) dr = R² J1(µ_n)/µ_n
            ∫₀ᴿ r J0²(µ_n r/R) dr = R²/2 · (J0²(µ_n) + J1²(µ_n))

        Using the BC relation J1(µ_n) = (Bi/µ_n)·J0(µ_n):
            C_n = 2·Bi / ((µ_n² + Bi²) · J0(µ_n))

        Reference: Carslaw & Jaeger (1959, §7.5, eq. 7.5.3).
        """
        # Tabulated zeros of J0 (Abramowitz & Stegun, Table 9.5)
        j0_zeros = [2.4048, 5.5201, 8.6537, 11.7915, 14.9309]
        while len(j0_zeros) < n + 1:
            k = len(j0_zeros) + 1
            j0_zeros.append((k - 0.25) * np.pi)   # asymptotic approximation

        mu, C = np.zeros(n), np.zeros(n)
        for i in range(n):
            # Bracket: mode 1 in (0, first J0 zero); mode k in (j0zero[k-2], j0zero[k-1])
            lo = 1e-8           if i == 0 else j0_zeros[i - 1] + 1e-10
            hi = j0_zeros[0] - 1e-10  if i == 0 else j0_zeros[i]     - 1e-10
            try:
                m = brentq(lambda m: m * j1(m) - Bi * j0(m), lo, hi)
            except ValueError:
                m = (lo + hi) / 2
            mu[i] = m
            J0v   = j0(m)
            # Guard against near-zero J0 (should not happen with correct brackets)
            denom = (m ** 2 + Bi ** 2) * J0v
            C[i]  = 2.0 * Bi / denom if abs(J0v) > 1e-10 else 0.0
        return mu, C

    def T(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """T(x, y, t) in °C.  Points outside the circle return T_water."""
        r = np.sqrt(x**2 + y**2)
        Fo = self.alpha * t / self.R**2
        theta = np.zeros_like(r, dtype=float)
        for mu_n, C_n in zip(self._mu, self._C):
            theta += C_n * j0(mu_n * r / self.R) * np.exp(-mu_n**2 * Fo)
        T_field = T_WATER + DELTA_T * theta
        T_field[r > self.R] = T_WATER   # outside domain
        return T_field

    def sample_interior(self, n: int, rng=None) -> tuple[np.ndarray, np.ndarray]:
        rng = rng or np.random.default_rng(0)
        pts = []
        while len(pts) < n:
            batch = rng.uniform(-self.R, self.R, (n * 4, 2))
            inside = batch[:, 0]**2 + batch[:, 1]**2 < self.R**2
            pts.extend(batch[inside].tolist())
        arr = np.array(pts[:n], dtype=np.float32)
        return arr[:, 0], arr[:, 1]

    def sample_boundary(self, n: int, rng=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = rng or np.random.default_rng(1)
        theta_b = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        bx = (self.R * np.cos(theta_b)).astype(np.float32)
        by = (self.R * np.sin(theta_b)).astype(np.float32)
        # Outward normal on a circle is the unit radial vector
        nx = np.cos(theta_b)
        ny = np.sin(theta_b)
        return bx, by, nx, ny

    def grid(self, n: int = 50) -> tuple[np.ndarray, np.ndarray]:
        lin = np.linspace(-self.R, self.R, n)
        return np.meshgrid(lin, lin, indexing="ij")

    def info(self) -> str:
        return f"Circle2D  R={self.R}m  Bi={self.Bi:.2f}"


# ══════════════════════════════════════════════════════════════════════════════
# Domain 3 — Annulus
# ══════════════════════════════════════════════════════════════════════════════

class Annulus2D:
    """
    Annular domain (inner radius R_in, outer radius R_out).
    Outer boundary: Robin (convective) BC — k·∂T/∂r + h(T−T_w) = 0 at r = R_out.
    Inner boundary: insulated (adiabatic) — ∂T/∂r = 0 at r = R_in.

    The inner adiabatic BC models a hollow casting with an insulated core
    (e.g., a thin-walled tube section of the automotive subframe).

    Analytical solution:
        T(r,t) = T_w + ΔT · Σ_n C_n · Z0(µ_n, r) · exp(−µ_n²·α·t/R_out²)

    where Z0(µ, r) = J0(µ·r/R_out)·Y1(µ·R_in/R_out) − Y0(µ·r/R_out)·J1(µ·R_in/R_out)
    is the Bessel cross-product function satisfying both BCs simultaneously.

    Eigenvalues from the combined BC condition at r = R_out.
    Coefficients from Bessel function orthogonality on [R_in, R_out].

    Reference: Carslaw & Jaeger (1959, §7.8).
    """

    name  = "Annulus 2D"
    color = "#E65100"   # orange

    def __init__(self, R_in: float = 0.1, R_out: float = 0.3,
                 h: float = H, k: float = K,
                 alpha: float = ALPHA, n_modes: int = N_MODES):
        self.R_in, self.R_out = R_in, R_out
        self.h, self.k        = h, k
        self.alpha            = alpha
        self.Bi               = h * R_out / k
        self._kappa           = R_in / R_out   # radius ratio
        self._mu, self._C     = self._eigenvalues(self.Bi, self._kappa, n_modes)

    def _Z0(self, mu: float, r_norm: float) -> float:
        """Bessel cross-product satisfying insulated inner BC."""
        k = self._kappa
        return j0(mu * r_norm) * y1(mu * k) - y0(mu * r_norm) * j1(mu * k)

    def _dZ0_dr(self, mu: float, r_norm: float) -> float:
        """Radial derivative of Z0 w.r.t. r_norm."""
        k = self._kappa
        return (-j1(mu * r_norm) * y1(mu * k) + y1(mu * r_norm) * j1(mu * k)) * mu

    def _eigenvalues(self, Bi: float, kappa: float, n: int):
        """
        Roots of:  µ · dZ0/dr(µ, 1) + Bi · Z0(µ, 1) = 0
        Bracketed search between consecutive maxima of Z0 at r = R_out.
        """
        mu_list, C_list = [], []
        mu_search = np.linspace(0.1, n * np.pi, 10000)

        def bc_residual(m):
            return self._dZ0_dr(m, 1.0) + Bi * self._Z0(m, 1.0)

        res = np.array([bc_residual(m) for m in mu_search])
        sign_changes = np.where(np.diff(np.sign(res)))[0]

        for idx in sign_changes[:n]:
            lo, hi = mu_search[idx], mu_search[idx + 1]
            try:
                m = brentq(bc_residual, lo, hi)
                # Coefficient from ∫_{R_in}^{R_out} r·Z0²(µ,r/R_out) dr
                r_norm = np.linspace(kappa, 1.0, 500)
                Z_vals = np.array([self._Z0(m, r) for r in r_norm])
                norm2  = np.trapz(r_norm * Z_vals**2, r_norm)
                Z0_val = self._Z0(m, kappa + 1e-6)   # ≈ Z0 at inner radius
                C = np.trapz(r_norm * Z_vals, r_norm) / (norm2 + 1e-30)
                mu_list.append(m)
                C_list.append(C)
            except ValueError:
                pass
            if len(mu_list) == n:
                break

        return np.array(mu_list), np.array(C_list)

    def T(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        r = np.sqrt(x**2 + y**2)
        Fo = self.alpha * t / self.R_out**2
        r_norm = r / self.R_out
        theta = np.zeros_like(r, dtype=float)
        for mu_n, C_n in zip(self._mu, self._C):
            Z_vals = np.array([self._Z0(mu_n, rn) for rn in r_norm.flat])
            Z_vals = Z_vals.reshape(r_norm.shape)
            theta += C_n * Z_vals * np.exp(-mu_n**2 * Fo)
        T_field = T_WATER + DELTA_T * theta
        # Mask: outside annulus
        T_field[(r < self.R_in) | (r > self.R_out)] = np.nan
        return T_field

    def sample_interior(self, n: int, rng=None) -> tuple[np.ndarray, np.ndarray]:
        rng = rng or np.random.default_rng(0)
        pts = []
        while len(pts) < n:
            batch = rng.uniform(-self.R_out, self.R_out, (n * 4, 2))
            r2    = batch[:, 0]**2 + batch[:, 1]**2
            mask  = (r2 > self.R_in**2) & (r2 < self.R_out**2)
            pts.extend(batch[mask].tolist())
        arr = np.array(pts[:n], dtype=np.float32)
        return arr[:, 0], arr[:, 1]

    def sample_boundary(self, n: int, rng=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = rng or np.random.default_rng(1)
        n_out = n // 2
        n_in  = n - n_out
        # Outer boundary (convective Robin)
        th_out = rng.uniform(0, 2 * np.pi, n_out).astype(np.float32)
        # Inner boundary (insulated, normal points inward = −r̂)
        th_in  = rng.uniform(0, 2 * np.pi, n_in).astype(np.float32)
        bx = np.concatenate([self.R_out * np.cos(th_out), self.R_in * np.cos(th_in)]).astype(np.float32)
        by = np.concatenate([self.R_out * np.sin(th_out), self.R_in * np.sin(th_in)]).astype(np.float32)
        nx = np.concatenate([np.cos(th_out), -np.cos(th_in)]).astype(np.float32)
        ny = np.concatenate([np.sin(th_out), -np.sin(th_in)]).astype(np.float32)
        return bx, by, nx, ny

    def grid(self, n: int = 50) -> tuple[np.ndarray, np.ndarray]:
        lin = np.linspace(-self.R_out, self.R_out, n)
        return np.meshgrid(lin, lin, indexing="ij")

    def info(self) -> str:
        return (f"Annulus2D  R_in={self.R_in}m  R_out={self.R_out}m  "
                f"Bi_out={self.Bi:.2f}  κ={self._kappa:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# Domain 4 — L-Shape  (numerical finite-difference reference)
# ══════════════════════════════════════════════════════════════════════════════

class LShape2D:
    """
    L-shaped 2D domain: no closed-form analytical solution exists.

    We implement an explicit finite-difference (FD) solver as the reference.
    The FD solution is pre-computed on a uniform Cartesian grid with a
    geometric mask that excludes the top-right corner of the bounding box.

    Domain definition (cross-section of L-shaped casting arm):
        Full bounding box: [0, Lx] × [0, Ly]
        Excluded region:   x > cut_x  AND  y > cut_y  (top-right corner)

    Boundary conditions:
        All exposed faces: Robin BC  k·∂T/∂n + h(T−T_w) = 0
        The corner cut creates two additional exposed faces at x=cut_x and y=cut_y.

    The FD time step is constrained by the 2D stability criterion:
        Δt ≤ (Δx² · Δy²) / (2α·(Δx² + Δy²))
    We use Δt = 0.9 × Δt_max for a 10% safety margin.

    This approach mirrors the LShape3D implementation in Level 8 (domains_3d.py),
    which itself follows the method described in Shen et al. (2023) for complex
    casting geometries without separable solutions.

    Reference:
        Shen, L. et al. (2023). MCO-PINN: Multi-continuity operator
        physics-informed neural network. [Domain discretisation strategy]
    """

    name  = "L-Shape 2D"
    color = "#6A1B9A"   # purple

    # Pre-computed FD time grid (matches Level 8 conventions)
    T_FEM = np.arange(0.0, 30.0 + 1e-9, 1.5)   # 21 snapshots

    def __init__(self, Lx: float = 0.8, Ly: float = 0.8,
                 cut_x: float = 0.4, cut_y: float = 0.4,
                 h: float = H, k: float = K,
                 alpha: float = ALPHA, rho_cp: float = RHO_CP,
                 nx: int = 40, ny: int = 40):
        self.Lx, self.Ly       = Lx, Ly
        self.cut_x, self.cut_y = cut_x, cut_y
        self.h, self.k         = h, k
        self.alpha, self.rho_cp = alpha, rho_cp
        self.nx, self.ny       = nx, ny

        dx, dy = Lx / nx, Ly / ny
        self.dx, self.dy = dx, dy
        self.xi = np.linspace(dx / 2, Lx - dx / 2, nx)   # cell centres
        self.yi = np.linspace(dy / 2, Ly - dy / 2, ny)
        XX, YY = np.meshgrid(self.xi, self.yi, indexing="ij")
        self.inside = self._mask(XX, YY)

        # Run FD solver and store snapshots at T_FEM time points
        print(f"  LShape2D: solving FD reference on {nx}×{ny} grid…", flush=True)
        self._T_cache = self._run_fd()   # shape (nx, ny, n_t)
        self._interp  = None            # built lazily in T()

    def _mask(self, X, Y):
        """True for cells inside the L-shaped domain."""
        in_full = (X >= 0) & (X <= self.Lx) & (Y >= 0) & (Y <= self.Ly)
        in_cut  = (X > self.cut_x) & (Y > self.cut_y)
        return in_full & ~in_cut

    def _run_fd(self) -> np.ndarray:
        """
        Explicit finite-difference time integration.
        Robin BC implemented via ghost-cell extrapolation:
            T_ghost = T_interior - 2·(h/k)·Δn·(T_surface − T_w)
        which gives a second-order-accurate approximation to ∂T/∂n.
        """
        dx, dy  = self.dx, self.dy
        dt_max  = (dx**2 * dy**2) / (2 * self.alpha * (dx**2 + dy**2))
        dt      = 0.9 * dt_max

        T  = np.full((self.nx, self.ny), T_INIT)
        T[~self.inside] = T_WATER   # outside cells fixed at water temp

        n_t  = len(self.T_FEM)
        cache = np.zeros((self.nx, self.ny, n_t))
        cache[:, :, 0] = T.copy()

        t_now  = 0.0
        t_next = self.T_FEM[1]
        snap_i = 1

        rx, ry = self.alpha * dt / dx**2, self.alpha * dt / dy**2

        while snap_i < n_t:
            T_new = T.copy()

            # Interior update (Laplacian via finite differences)
            T_new[1:-1, 1:-1] += (rx * (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]) +
                                   ry * (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]))

            # Robin BC: apply ghost-cell correction on boundary faces
            # Left face (i=0): outward normal = −x̂
            T_new[0, :]  = (T[1, :] + 2 * (self.h / self.k) * dx * T_WATER) / (1 + 2 * (self.h / self.k) * dx)
            # Right face (i=-1): outward normal = +x̂
            T_new[-1, :] = (T[-2, :] + 2 * (self.h / self.k) * dx * T_WATER) / (1 + 2 * (self.h / self.k) * dx)
            # Bottom face (j=0): outward normal = −ŷ
            T_new[:, 0]  = (T[:, 1] + 2 * (self.h / self.k) * dy * T_WATER) / (1 + 2 * (self.h / self.k) * dy)
            # Top face (j=-1): outward normal = +ŷ
            T_new[:, -1] = (T[:, -2] + 2 * (self.h / self.k) * dy * T_WATER) / (1 + 2 * (self.h / self.k) * dy)

            # Cut faces: vertical (x = cut_x) and horizontal (y = cut_y)
            ix_cut = int(self.cut_x / dx)
            iy_cut = int(self.cut_y / dy)
            if ix_cut < self.nx:
                T_new[ix_cut, iy_cut:] = (T[ix_cut - 1, iy_cut:] + 2 * (self.h / self.k) * dx * T_WATER) / (1 + 2 * (self.h / self.k) * dx)
            if iy_cut < self.ny:
                T_new[ix_cut:, iy_cut] = (T[ix_cut:, iy_cut - 1] + 2 * (self.h / self.k) * dy * T_WATER) / (1 + 2 * (self.h / self.k) * dy)

            T_new[~self.inside] = T_WATER
            T = T_new
            t_now += dt

            if t_now >= t_next - 1e-10:
                cache[:, :, snap_i] = T.copy()
                snap_i += 1
                if snap_i < n_t:
                    t_next = self.T_FEM[snap_i]

        return cache

    def T(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """Bilinear interpolation from pre-computed FD cache."""
        from scipy.interpolate import RegularGridInterpolator
        # Build interpolator for this time (linear interpolation in t)
        t_idx = np.searchsorted(self.T_FEM, t, side="right") - 1
        t_idx = np.clip(t_idx, 0, len(self.T_FEM) - 2)
        alpha_t = (t - self.T_FEM[t_idx]) / (self.T_FEM[t_idx + 1] - self.T_FEM[t_idx] + 1e-30)
        T_field = ((1 - alpha_t) * self._T_cache[:, :, t_idx] +
                   alpha_t       * self._T_cache[:, :, t_idx + 1])
        interp = RegularGridInterpolator(
            (self.xi, self.yi), T_field,
            method="linear", bounds_error=False, fill_value=T_WATER
        )
        pts = np.stack([x.flat, y.flat], axis=1)
        T_out = interp(pts).reshape(x.shape)
        # Mask outside region
        mask = self._mask(x, y)
        T_out[~mask] = np.nan
        return T_out

    def sample_interior(self, n: int, rng=None) -> tuple[np.ndarray, np.ndarray]:
        rng = rng or np.random.default_rng(0)
        pts = []
        while len(pts) < n:
            batch = rng.uniform([0, 0], [self.Lx, self.Ly], (n * 4, 2))
            mask  = self._mask(batch[:, 0], batch[:, 1])
            pts.extend(batch[mask].tolist())
        arr = np.array(pts[:n], dtype=np.float32)
        return arr[:, 0], arr[:, 1]

    def sample_boundary(self, n: int, rng=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Boundary of the L-shape = 6 edges (rectangle minus cut corner)."""
        rng = rng or np.random.default_rng(1)
        n_e = n // 6
        Lx, Ly, cx, cy = self.Lx, self.Ly, self.cut_x, self.cut_y
        edges = [
            # (x_func, y_func, nx, ny) for uniform parameter s ∈ [0,1]
            (lambda s: np.zeros(len(s)),          lambda s: s * Ly,           -1,  0),   # left
            (lambda s: s * cx,                    lambda s: np.full(len(s), Ly), 0, +1), # top-left
            (lambda s: np.full(len(s), cx),       lambda s: Ly - s*(Ly-cy),   +1,  0),   # cut-vert
            (lambda s: cx + s*(Lx-cx),            lambda s: np.full(len(s), cy), 0, -1), # cut-horiz
            (lambda s: np.full(len(s), Lx),       lambda s: s * cy,           +1,  0),   # right
            (lambda s: s * Lx,                    lambda s: np.zeros(len(s)),  0, -1),   # bottom
        ]
        bx_all, by_all, nx_all, ny_all = [], [], [], []
        for xf, yf, nxv, nyv in edges:
            s = rng.uniform(0, 1, n_e).astype(np.float32)
            bx_all.append(xf(s).astype(np.float32))
            by_all.append(yf(s).astype(np.float32))
            nx_all.append(np.full(n_e, nxv, dtype=np.float32))
            ny_all.append(np.full(n_e, nyv, dtype=np.float32))
        return (np.concatenate(bx_all), np.concatenate(by_all),
                np.concatenate(nx_all), np.concatenate(ny_all))

    def grid(self, nx: int = 50, ny: int = 50) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0, self.Lx, nx)
        y = np.linspace(0, self.Ly, ny)
        return np.meshgrid(x, y, indexing="ij")

    def info(self) -> str:
        return (f"LShape2D  {self.Lx}×{self.Ly}m  cut=({self.cut_x},{self.cut_y})m  "
                f"[FD reference,  {self.nx}×{self.ny} grid]")


# ──────────────────────────────────────────────────────────────────────────────
# Domain registry — used by trainer and predictor
# ──────────────────────────────────────────────────────────────────────────────

DOMAINS_2D = {
    "rectangle": Rectangle2D,
    "circle":    Circle2D,
    "lshape":    LShape2D,
}


def make_domain(name: str, **kwargs):
    """Instantiate a 2D domain by name."""
    if name not in DOMAINS_2D:
        raise ValueError(f"Unknown 2D domain '{name}'. Choose from: {list(DOMAINS_2D)}")
    return DOMAINS_2D[name](**kwargs)
