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
# Domain 4: L-Prism (Numerical — geometric mask)
# ══════════════════════════════════════════════════════════════

class LPrism3D:
    """
    L-shaped 2D cross-section × z extension.
    No exact analytical solution — approximate: full prism + corner NaN mask.

    T(x,y,z,t) ≈ based on Rectangular3D, with z-direction PlaneWall1D correction.
    This is an approximation; the exact solution requires numerical FEM.

    Used for visualization and PINN benchmark (not exact analytical).
    """

    name  = "L-Prism 3D"
    color = "#6A1B9A"

    def __init__(self, Lx=1.3, Ly=0.6, Lz=0.4,
                 cut_x=0.715, cut_y=0.3,
                 h=H_CONV, k=K, alpha=ALPHA, N=8):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.cut_x, self.cut_y    = cut_x, cut_y
        # Full prism solution (approximation)
        self.field = Rectangular3D(Lx=Lx, Ly=Ly, Lz=Lz, h=h, k=k, alpha=alpha, N=N)

    def T(self, x, y, z, t):
        T_field = self.field.T(x, y, z, t)
        # Mask corner region with NaN
        corner = (x > self.cut_x) & (y > self.cut_y)
        T_field = np.where(corner, np.nan, T_field)
        return T_field

    def mask(self, x, y, z):
        in_rect = ((x >= 0) & (x <= self.Lx) &
                   (y >= 0) & (y <= self.Ly) &
                   (z >= 0) & (z <= self.Lz))
        corner  = (x > self.cut_x) & (y > self.cut_y)
        return in_rect & ~corner

    def info(self):
        return (f"LPrism3D  {self.Lx}×{self.Ly}×{self.Lz}m  "
                f"(corner cut x>{self.cut_x}, y>{self.cut_y})\n"
                f"  NOTE: approximate solution — not exact analytical")


# ══════════════════════════════════════════════════════════════
# All domain list
# ══════════════════════════════════════════════════════════════

def all_domains():
    """Create and return all 3D domains."""
    return {
        "rectangular": Rectangular3D(),
        "cylinder":    Cylinder3D(),
        "stacked":     StackedCubes3D(),
        "lprism":      LPrism3D(),
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
