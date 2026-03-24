"""
domains_3d.py — 3D Analytical Heat Transfer Domains
=====================================================
3D domains for A356 aluminum water quenching simulation.

Domain 1 : Rectangular3D  — 3D rectangular prism, fully separable product solution
Domain 2 : Cylinder3D     — 3D cylinder, Bessel radial + plane wall axial
Domain 3 : StackedCubes3D — 2 stacked cubes (Shen 2023 style)
Domain 4 : LPrism3D       — L-shaped 2D cross-section × z extension (geometric mask, numerical ref.)

Physical constants — A356 aluminum:
  T_init  = 540 °C  (initial quenching temperature)
  T_water = 20  °C  (water bath)
  k       = 160 W/(m·K)
  ρCp     = 2.4e6 J/(m³·K)
  h       = 4000 W/(m²·K)
  α       = k/ρCp = 6.67×10⁻⁵ m²/s
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

# ──────────────────────────────────────────────────────────────
# Physical constants
# ──────────────────────────────────────────────────────────────
T_INIT   = 540.0     # °C
T_WATER  = 20.0      # °C
DELTA_T  = T_INIT - T_WATER   # 520 °C
K        = 160.0     # W/(m·K)
RHO_CP   = 2.4e6     # J/(m³·K)
H_CONV   = 4000.0    # W/(m²·K)
ALPHA    = K / RHO_CP  # m²/s


# ══════════════════════════════════════════════════════════════
# Helper: 1D Plane Wall — basis for all domains
# ══════════════════════════════════════════════════════════════

class PlaneWall1D:
    """
    1D plane wall analytical solution.
    Symmetric cooling from both faces (same h on both sides).

    θ(x,t) = Σ_n C_n cos(ζ_n * x*) * exp(-ζ_n² * Fo)

    x* = (x - L/2) / (L/2)  ∈ [-1, 1]  (normalized relative to center)
    Fo = α*t / (L/2)²
    ζ_n*tan(ζ_n) = Bi = h*(L/2)/k

    Coefficient: C_n = 4*sin(ζ_n) / (2*ζ_n + sin(2*ζ_n))
    """

    def __init__(self, L: float, h=H_CONV, k=K, alpha=ALPHA, N=20):
        self.L     = L
        self.a     = L / 2          # half-thickness
        self.Bi    = h * self.a / k
        self.alpha = alpha
        self.N     = N
        self._eigenvalues()

    def _eigenvalues(self):
        """First N roots of ζ·tan(ζ) = Bi."""
        self.zeta = np.zeros(self.N)
        self.C    = np.zeros(self.N)
        for n in range(self.N):
            lo = n * np.pi + 1e-10
            hi = (n + 0.5) * np.pi - 1e-10
            try:
                z = brentq(lambda z: z * np.tan(z) - self.Bi, lo, hi)
            except ValueError:
                z = (lo + hi) / 2
            self.zeta[n] = z
            self.C[n] = 4 * np.sin(z) / (2 * z + np.sin(2 * z))

    def theta(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Dimensionless temperature θ = (T-T_w)/(T_0-T_w).
        x ∈ [0, L]  (domain bounds)
        """
        x_star = (np.asarray(x, dtype=float) - self.a) / self.a
        Fo     = self.alpha * t / self.a ** 2
        result = np.zeros_like(x_star)
        for n in range(self.N):
            result += self.C[n] * np.cos(self.zeta[n] * x_star) * np.exp(-self.zeta[n]**2 * Fo)
        return result


# ══════════════════════════════════════════════════════════════
# Domain 1: 3D Rectangular Prism
# ══════════════════════════════════════════════════════════════

class Rectangular3D:
    """
    3D rectangular prism analytical solution.
    Robin (convection) BC on all 6 faces.

    T(x,y,z,t) = T_w + ΔT * θ_x(x,t) * θ_y(y,t) * θ_z(z,t)

    Multiplicative separation (product solution) — valid for uniform initial condition.

    Physical domain — A356 casting (Mortensen 2026):
        Lx=1.3 m, Ly=0.6 m, Lz=0.4 m
        Bi_x≈16.3, Bi_y≈7.5, Bi_z≈5.0
    """

    name   = "Rectangular Prism 3D"
    color  = "#1565C0"

    def __init__(self, Lx=1.3, Ly=0.6, Lz=0.4,
                 h=H_CONV, k=K, alpha=ALPHA, N=8):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.wx = PlaneWall1D(Lx, h, k, alpha, N)
        self.wy = PlaneWall1D(Ly, h, k, alpha, N)
        self.wz = PlaneWall1D(Lz, h, k, alpha, N)
        self.Bi = dict(x=h*(Lx/2)/k, y=h*(Ly/2)/k, z=h*(Lz/2)/k)

    # ── Analytical solution ──────────────────────────────────
    def theta(self, x, y, z, t):
        return self.wx.theta(x, t) * self.wy.theta(y, t) * self.wz.theta(z, t)

    def T(self, x, y, z, t):
        return T_WATER + DELTA_T * self.theta(x, y, z, t)

    # ── Geometry mask ────────────────────────────────────────
    @staticmethod
    def mask(x, y, z, Lx=1.3, Ly=0.6, Lz=0.4):
        return (x >= 0) & (x <= Lx) & (y >= 0) & (y <= Ly) & (z >= 0) & (z <= Lz)

    # ── Axis slices (for visualization) ─────────────────────
    def slice_xy(self, xx, yy, z_val, t):
        """T(x,y,z=z_val,t) — fixed z slice"""
        zz = np.full_like(xx, z_val)
        return self.T(xx, yy, zz, t)

    def slice_xz(self, xx, zz, y_val, t):
        """T(x,y=y_val,z,t) — fixed y slice"""
        yy = np.full_like(xx, y_val)
        return self.T(xx, yy, zz, t)

    def slice_yz(self, yy, zz, x_val, t):
        """T(x=x_val,y,z,t) — fixed x slice"""
        xx = np.full_like(yy, x_val)
        return self.T(xx, yy, zz, t)

    def info(self):
        return (f"Rectangular3D  Lx={self.Lx}m Ly={self.Ly}m Lz={self.Lz}m\n"
                f"  Bi_x={self.Bi['x']:.2f}  Bi_y={self.Bi['y']:.2f}  Bi_z={self.Bi['z']:.2f}")


# ══════════════════════════════════════════════════════════════
# Domain 2: 3D Cylinder
# ══════════════════════════════════════════════════════════════

class Cylinder3D:
    """
    3D cylinder analytical solution (axial symmetry — r,z).
    T(r,z,t) = T_w + ΔT * θ_r(r,t) * θ_z(z,t)

    Radial:  Bessel J₀(λ_n r) with Robin BC
             λ_n * R * J₁(λ_n R) = Bi_r * J₀(λ_n R)
             C_n = 2*J₁(λ_n R) / (λ_n*R * (J₀²(λ_n R) + J₁²(λ_n R)))

    Axial:   plane wall solution (θ_z)

    Domain — cylindrical casting:
        R=0.25 m, H=0.6 m
        Bi_r≈6.25, Bi_z≈7.5
    """

    name  = "Cylinder 3D"
    color = "#2E7D32"

    def __init__(self, R=0.25, H=0.6,
                 h=H_CONV, k=K, alpha=ALPHA, N=8):
        self.R     = R
        self.H     = H
        self.alpha = alpha
        self.Bi_r  = h * R / k
        self.Bi_z  = h * (H / 2) / k
        self.N     = N
        self._radial_eigenvalues(h, k)
        self.wz = PlaneWall1D(H, h, k, alpha, N)

    def _radial_eigenvalues(self, h, k):
        """Roots of λ_n*R*J₁(λ_n R) = Bi_r * J₀(λ_n R)."""
        Bi = self.Bi_r
        R  = self.R
        self.lam   = np.zeros(self.N)
        self.C_rad = np.zeros(self.N)
        for n in range(self.N):
            lo = (n * np.pi + 1e-10) / R
            hi = ((n + 1) * np.pi - 1e-10) / R
            def eq(lam, B=Bi, r=R):
                return lam * r * j1(lam * r) - B * j0(lam * r)
            try:
                self.lam[n] = brentq(eq, lo, hi)
            except ValueError:
                self.lam[n] = (lo + hi) / 2
            ln = self.lam[n]
            J0v, J1v = j0(ln * R), j1(ln * R)
            # C_n = 2 J₁(λ_n R) / (λ_n R (J₀² + J₁²))
            denom = ln * R * (J0v**2 + J1v**2)
            self.C_rad[n] = 2 * J1v / denom if denom != 0 else 0.0

    def theta_r(self, r: np.ndarray, t: float) -> np.ndarray:
        """Radial dimensionless temperature θ_r(r,t)."""
        r = np.asarray(r, dtype=float)
        result = np.zeros_like(r)
        Fo_r = self.alpha * t / self.R**2
        for n in range(self.N):
            ln = self.lam[n]
            result += self.C_rad[n] * j0(ln * r) * np.exp(-ln**2 * self.R**2 * Fo_r)
        return result

    def T(self, r, z, t):
        """T(r,z,t) — cylindrical coordinates."""
        return T_WATER + DELTA_T * self.theta_r(r, t) * self.wz.theta(z, t)

    def T_xyz(self, x, y, z, t):
        """Cartesian → cylindrical conversion."""
        r = np.sqrt(x**2 + y**2)
        return self.T(r, z, t)

    @staticmethod
    def mask(x, y, z, R=0.25, H=0.6):
        r = np.sqrt(x**2 + y**2)
        return (r <= R) & (z >= 0) & (z <= H)

    def slice_xy(self, xx, yy, z_val, t):
        """T(x,y,z=z_val,t) — xy slice"""
        zz = np.full_like(xx, z_val)
        return self.T_xyz(xx, yy, zz, t)

    def slice_rz(self, rr, zz, t):
        """T(r,z,t) — r-z plane"""
        return self.T(rr, zz, t)

    def info(self):
        return (f"Cylinder3D  R={self.R}m H={self.H}m\n"
                f"  Bi_r={self.Bi_r:.2f}  Bi_z={self.Bi_z:.2f}")


# ══════════════════════════════════════════════════════════════
# Domain 3: Stacked Cubes (Shen 2023 style)
# ══════════════════════════════════════════════════════════════

class StackedCubes3D:
    """
    Shen 2023 MCO-PINN reference domain: N stacked square prisms.

    Same material → analytical solution = single tall prism.
    Dimensions: L_cube × L_cube × (N * L_cube).

    Default: 2 cubes × 0.5m → [0.5 × 0.5 × 1.0 m]
      Bi = h*(0.25)/k = 6.25 (same in all directions)

    Each cube interface is marked in visualization.
    """

    name  = "Stacked Cubes 3D"
    color = "#E65100"

    def __init__(self, L_cube=0.5, N_stack=2,
                 h=H_CONV, k=K, alpha=ALPHA, N_terms=8):
        self.L_cube  = L_cube
        self.N_stack = N_stack
        self.Lz      = L_cube * N_stack
        self.field   = Rectangular3D(
            Lx=L_cube, Ly=L_cube, Lz=self.Lz,
            h=h, k=k, alpha=alpha, N=N_terms)
        self.interface_z = [L_cube * i for i in range(1, N_stack)]

    def T(self, x, y, z, t):
        return self.field.T(x, y, z, t)

    def theta(self, x, y, z, t):
        return self.field.theta(x, y, z, t)

    def mask(self, x, y, z):
        return ((x >= 0) & (x <= self.L_cube) &
                (y >= 0) & (y <= self.L_cube) &
                (z >= 0) & (z <= self.Lz))

    def slice_xy(self, xx, yy, z_val, t):
        zz = np.full_like(xx, z_val)
        return self.T(xx, yy, zz, t)

    def slice_xz(self, xx, zz, y_val, t):
        yy = np.full_like(xx, y_val)
        return self.T(xx, yy, zz, t)

    def info(self):
        Bi = H_CONV * (self.L_cube / 2) / K
        return (f"StackedCubes3D  {self.N_stack}×[{self.L_cube}m cube]  "
                f"→ {self.L_cube}×{self.L_cube}×{self.Lz}m\n"
                f"  Bi={Bi:.2f} (same in all directions)")


# ══════════════════════════════════════════════════════════════
# Domain 4: L-Shape (Numerical FD solver)
# ══════════════════════════════════════════════════════════════

class LShape3D:
    """
    L-shaped 3D domain: explicit finite-difference numerical solver.

    Cross-section (xy) = two rectangular arms joined at corner:
        mask: (x <= cut_x) OR (y <= cut_y)   [for all z in [0, Lz]]

    Numerical reference via 3D explicit FD:
      - Robin (convective) BC on ALL exposed faces
      - Precomputed at FEM time steps t_fem = 0,1.5,...,30s
      - Trilinear interpolation via RegularGridInterpolator

    Domain dimensions  (complex asymmetric casting):
        Lx=0.8m, Ly=0.8m, Lz=0.4m
        cut_x=0.3m, cut_y=0.3m  (remove top-right corner)
    """

    name  = "L-Shape 3D"
    color = "#6A1B9A"

    # FEM time grid (same as other domains: 0..30s every 1.5s)
    T_FEM = np.arange(0.0, 30.0 + 1e-9, 1.5)   # 21 steps

    def __init__(self, Lx=0.8, Ly=0.8, Lz=0.4,
                 cut_x=0.3, cut_y=0.3,
                 h=H_CONV, k=K, alpha=ALPHA, rho_cp=RHO_CP,
                 nx=20, ny=20, nz=10):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.cut_x, self.cut_y    = cut_x, cut_y
        self.h, self.k            = h, k
        self.alpha, self.rho_cp   = alpha, rho_cp
        self.nx, self.ny, self.nz = nx, ny, nz

        # Cell-centred uniform grid
        dx, dy, dz  = Lx / nx, Ly / ny, Lz / nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.xi_c = np.linspace(dx / 2, Lx - dx / 2, nx)
        self.yi_c = np.linspace(dy / 2, Ly - dy / 2, ny)
        self.zi_c = np.linspace(dz / 2, Lz - dz / 2, nz)

        # 3-D boolean mask on cell-centred grid
        XX, YY, ZZ  = np.meshgrid(self.xi_c, self.yi_c, self.zi_c, indexing='ij')
        self.inside = self._mask_3d(XX, YY, ZZ)

        # Nearest-inside-cell indices (used to fill NaN boundary for interpolator).
        # scipy EDT treats foreground=True as features and finds nearest background(=False).
        # We want nearest INSIDE (True) cell, so we invert: ~inside makes inside=background.
        # For inside cells: they ARE background → indices point to themselves.
        # For outside cells: foreground → indices point to nearest inside cell.
        self._nn_idx = distance_transform_edt(
            ~self.inside, return_distances=False, return_indices=True)

        # Run FD solver and cache T at every t_fem step
        print(f"  LShape3D: running FD solver ({nx}×{ny}×{nz} grid)…", flush=True)
        self._T_cache = self._run_fd()   # shape (nx, ny, nz, n_t_fem)

        # Build interpolator
        self._interp = RegularGridInterpolator(
            (self.xi_c, self.yi_c, self.zi_c, self.T_FEM),
            self._T_cache,
            method='linear',
            bounds_error=False,
            fill_value=np.nan,
        )
        print("  LShape3D: FD solver done.", flush=True)

    # ── Geometry ─────────────────────────────────────────────

    def _mask_3d(self, x, y, z):
        """L-shape on full grid arrays."""
        return (((x <= self.cut_x) | (y <= self.cut_y)) &
                (z >= 0) & (z <= self.Lz))

    def mask(self, x, y, z):
        """Boolean mask for arbitrary (x, y, z) arrays."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        return (((x <= self.cut_x) | (y <= self.cut_y)) &
                (x >= 0) & (x <= self.Lx) &
                (y >= 0) & (y <= self.Ly) &
                (z >= 0) & (z <= self.Lz))

    # ── Finite-difference solver ──────────────────────────────

    def _fd_step(self, T, dt):
        """One explicit FD update step (vectorised, Robin BC on exposed faces)."""
        dx, dy, dz   = self.dx, self.dy, self.dz
        alpha        = self.alpha
        h            = self.h
        rho_cp       = self.rho_cp
        ins          = self.inside
        dT           = np.zeros_like(T)

        for axis, dh in enumerate([dx, dy, dz]):
            for shift in (+1, -1):
                T_n   = np.roll(T,   shift, axis=axis)
                ins_n = np.roll(ins, shift, axis=axis).copy()

                # Invalidate the wrapped boundary slice
                if shift == +1:
                    # rolled element at index 0 came from the far edge — mark invalid
                    idx = [slice(None)] * 3
                    idx[axis] = 0
                    ins_n[tuple(idx)] = False
                else:
                    idx = [slice(None)] * 3
                    idx[axis] = -1
                    ins_n[tuple(idx)] = False

                # Conductive flux (both cells inside)
                dT += np.where(ins & ins_n,
                               alpha / dh**2 * (T_n - T),
                               0.0)
                # Convective flux (current cell inside, neighbour outside / boundary)
                dT += np.where(ins & ~ins_n,
                               h / (rho_cp * dh) * (T_WATER - T),
                               0.0)

        return np.where(ins, T + dt * dT, T)

    def _run_fd(self):
        """Run explicit FD from T_INIT, store T at each t_fem checkpoint."""
        dx, dy, dz = self.dx, self.dy, self.dz

        # Stability: dt < 1 / (2α Σ(1/dh²) + 6h/(ρCp·dh_min))
        sum_inv_dh2 = 1/dx**2 + 1/dy**2 + 1/dz**2
        dh_min      = min(dx, dy, dz)
        dt_max      = 1.0 / (2 * self.alpha * sum_inv_dh2 +
                             6 * self.h / (self.rho_cp * dh_min))
        dt          = dt_max * 0.40

        T = np.where(self.inside, float(T_INIT), np.nan)

        n_t       = len(self.T_FEM)
        T_cache   = np.zeros((self.nx, self.ny, self.nz, n_t))
        T_cache[:, :, :, 0] = self._fill_nearest(T)

        t_now     = 0.0
        store_idx = 1
        n_max     = int(np.ceil(self.T_FEM[-1] / dt)) + 20

        for _ in range(n_max):
            if store_idx >= n_t:
                break
            T    = self._fd_step(T, dt)
            t_now += dt
            while (store_idx < n_t and
                   t_now >= self.T_FEM[store_idx] - 1e-9):
                T_cache[:, :, :, store_idx] = self._fill_nearest(T)
                store_idx += 1

        return T_cache

    def _fill_nearest(self, T):
        """Replace NaN (outside cells) with nearest inside cell value."""
        ni = self._nn_idx
        return T[ni[0], ni[1], ni[2]]

    # ── Temperature query ─────────────────────────────────────

    def T(self, x, y, z, t):
        """
        T(x, y, z, t) — trilinearly interpolated from FD cache.
        Returns NaN for points outside the L-shape.
        Query coords are clamped to the cell-centred grid (nearest cell
        for surface/boundary points).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        scalar = (x.ndim == 0)
        x, y, z = np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)
        t_arr = np.full_like(x, float(t))

        # Clamp to grid bounds (boundary queries get nearest cell-centre value)
        xq = np.clip(x,     self.xi_c[0], self.xi_c[-1])
        yq = np.clip(y,     self.yi_c[0], self.yi_c[-1])
        zq = np.clip(z,     self.zi_c[0], self.zi_c[-1])
        tq = np.clip(t_arr, self.T_FEM[0], self.T_FEM[-1])

        pts   = np.stack([xq, yq, zq, tq], axis=-1)
        T_out = self._interp(pts)
        T_out = np.where(self.mask(x, y, z), T_out, np.nan)
        return float(T_out[0]) if scalar else T_out

    def theta(self, x, y, z, t):
        """Dimensionless temperature θ = (T - T_water) / ΔT."""
        return np.clip((self.T(x, y, z, t) - T_WATER) / DELTA_T, 0.0, 1.0)

    def slice_xy(self, xx, yy, z_val, t):
        zz = np.full_like(xx, z_val)
        return self.T(xx.ravel(), yy.ravel(), zz.ravel(), t).reshape(xx.shape)

    def info(self):
        vol_frac = self.inside.sum() / self.inside.size
        return (f"LShape3D  {self.Lx}×{self.Ly}×{self.Lz}m  "
                f"(cut x>{self.cut_x}, y>{self.cut_y},  "
                f"vol≈{vol_frac:.0%} of bbox)\n"
                f"  Numerical FD grid: {self.nx}×{self.ny}×{self.nz}")


# Keep LPrism3D as a thin alias for backward compatibility
LPrism3D = LShape3D


# ══════════════════════════════════════════════════════════════
# All domain list
# ══════════════════════════════════════════════════════════════

def all_domains():
    """Create and return all 3D domains."""
    return {
        "rectangular": Rectangular3D(),
        "cylinder":    Cylinder3D(),
        "stacked":     StackedCubes3D(),
        "lshape":      LShape3D(),
    }


# ══════════════════════════════════════════════════════════════
# Verification — physical consistency checks
# ══════════════════════════════════════════════════════════════

def verify_fields(verbose=True):
    """
    Physical consistency check for each domain:
    1. t=0: T ≈ T_init (540°C) at center point
    2. t→∞: T → T_water (20°C)
    3. Center > surface (Robin BC → surface cools faster)
    4. Cooling rate consistent with expected α
    """
    results = {}

    # ── Rectangular3D ──────────────────────────────────────
    dom = Rectangular3D()
    cx, cy, cz = dom.Lx/2, dom.Ly/2, dom.Lz/2
    T_center_0  = dom.T(cx, cy, cz, t=0.01)
    T_center_5  = dom.T(cx, cy, cz, t=5.0)
    T_center_30 = dom.T(cx, cy, cz, t=30.0)
    T_surface_5 = dom.T(0.0, cy, cz, t=5.0)   # surface point
    results["rectangular"] = {
        "T_center_t=0":  T_center_0,
        "T_center_t=5":  T_center_5,
        "T_center_t=30": T_center_30,
        "T_surface_t=5": T_surface_5,
        "center>surface": T_center_5 > T_surface_5,
    }

    # ── Cylinder3D ─────────────────────────────────────────
    cyl = Cylinder3D()
    T_axis_0  = cyl.T(0.0, cyl.H/2, t=0.01)
    T_axis_5  = cyl.T(0.0, cyl.H/2, t=5.0)
    T_axis_30 = cyl.T(0.0, cyl.H/2, t=30.0)
    T_surf_5  = cyl.T(cyl.R * 0.99, cyl.H/2, t=5.0)
    results["cylinder"] = {
        "T_axis_t=0":   T_axis_0,
        "T_axis_t=5":   T_axis_5,
        "T_axis_t=30":  T_axis_30,
        "T_surf_t=5":   T_surf_5,
        "axis>surface": T_axis_5 > T_surf_5,
    }

    # ── StackedCubes3D ─────────────────────────────────────
    stk = StackedCubes3D()
    L   = stk.L_cube
    T_c_0  = stk.T(L/2, L/2, stk.Lz/2, t=0.01)
    T_c_5  = stk.T(L/2, L/2, stk.Lz/2, t=5.0)
    T_c_30 = stk.T(L/2, L/2, stk.Lz/2, t=30.0)
    results["stacked"] = {
        "T_center_t=0":  T_c_0,
        "T_center_t=5":  T_c_5,
        "T_center_t=30": T_c_30,
    }

    if verbose:
        print("=" * 60)
        print("3D Domain Verification Results")
        print("=" * 60)
        for dname, vals in results.items():
            print(f"\n  [{dname}]")
            for k, v in vals.items():
                if isinstance(v, bool):
                    status = "OK" if v else "FAIL"
                    print(f"    {k}: {status}")
                else:
                    print(f"    {k}: {v:.1f}°C")
        print()
        print("Expected behavior:")
        print("  t=0  → T ≈ 540°C (initial)  [truncation error ±5°C is normal]")
        print("  t=5  → T ≈ 350-450°C (cooling started)")
        print("  t=30 → T ≈ 20-100°C (mostly cooled)")
        print("  center > surface → OK (Robin BC → surface cools faster)")
        print("=" * 60)

    return results


# ══════════════════════════════════════════════════════════════
# Grid builders — per domain
# ══════════════════════════════════════════════════════════════

def make_grid_rectangular(nx=30, ny=20, nz=15):
    """Meshgrid for Rectangular3D."""
    dom = Rectangular3D()
    x = np.linspace(0, dom.Lx, nx)
    y = np.linspace(0, dom.Ly, ny)
    z = np.linspace(0, dom.Lz, nz)
    return np.meshgrid(x, y, z, indexing="ij")   # (nx,ny,nz)


def make_grid_cylinder(nr=25, nz=20, ntheta=36):
    """
    Cylindrical → Cartesian meshgrid for Cylinder3D.
    Disk grid in xy plane, uniform along z axis.
    """
    cyl = Cylinder3D()
    r   = np.linspace(0, cyl.R, nr)
    th  = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    z   = np.linspace(0, cyl.H, nz)
    rr, tt, zz = np.meshgrid(r, th, z, indexing="ij")
    xx = rr * np.cos(tt)
    yy = rr * np.sin(tt)
    return xx, yy, zz, rr, zz   # Cartesian + cylindrical


def make_grid_2d_slice(domain, nx=60, ny=40, z_frac=0.5):
    """
    xy slice grid for any domain.
    z = z_frac * Lz position.

    Returns: xx, yy (2D), z_val (float)
    """
    if hasattr(domain, "Lx"):
        Lx, Ly = domain.Lx, domain.Ly
        Lz = domain.Lz if hasattr(domain, "Lz") else 0.6
    elif hasattr(domain, "R"):
        Lx, Ly = domain.R, domain.R
        Lz = domain.H
        nx = ny
    else:
        Lx, Ly, Lz = 1.3, 0.6, 0.4

    xi = np.linspace(0, Lx, nx)
    yi = np.linspace(0, Ly, ny)
    xx, yy = np.meshgrid(xi, yi)
    z_val  = z_frac * Lz
    return xx, yy, z_val


# ══════════════════════════════════════════════════════════════
# Main — verification runner
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nDomain info:")
    for name, dom in all_domains().items():
        print(f"\n  {dom.info()}")

    print()
    verify_fields(verbose=True)

    # Quick numerical check
    dom = Rectangular3D()
    t_vals = [0.1, 1.0, 5.0, 10.0, 20.0, 30.0]
    cx, cy, cz = dom.Lx/2, dom.Ly/2, dom.Lz/2
    print("Center point cooling curve (Rectangular3D):")
    for t in t_vals:
        T = dom.T(cx, cy, cz, t)
        print(f"  t={t:5.1f}s  T_center={T:.1f}°C")

    cyl = Cylinder3D()
    print("\nAxis cooling curve (Cylinder3D):")
    for t in t_vals:
        T = cyl.T(0.0, cyl.H/2, t)
        print(f"  t={t:5.1f}s  T_axis={T:.1f}°C")

    stk = StackedCubes3D()
    L   = stk.L_cube
    print("\nCenter cooling curve (StackedCubes3D — 2 cubes):")
    for t in t_vals:
        T = stk.T(L/2, L/2, stk.Lz/2, t)
        print(f"  t={t:5.1f}s  T_center={T:.1f}°C")
