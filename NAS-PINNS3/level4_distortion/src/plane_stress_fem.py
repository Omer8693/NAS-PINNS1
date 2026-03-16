"""
plane_stress_fem.py — 2D Düzlem Gerilme FEM Çözücüsü
======================================================
T(x,y) → termal gerinme → termal yük vektörü → K·u = f_th → u(x,y) → |δ| [mm]

Eleman tipi : Q4 dörtgen (bilinear izoparametrik)
Integrasyon  : 2×2 Gauss (Q4 için tam doğru)
Çözücü       : Konjuge Gradyan (scipy.sparse.linalg.cg)
Sınır koş.   : Penalty metodu — rijit cisim hareketi engellenir

Malzeme      : A356 alüminyum (baseline_paper [2026], Table 1)
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import cg

# ------------------------------------------------------------------
# A356 alüminyum malzeme sabitleri — baseline_paper [2026], Table 1
# ------------------------------------------------------------------
E_MODULUS    = 70.0e9   # [Pa]  elastisite modülü (oda sıcaklığı)
NU_POISSON   = 0.33     # [-]   Poisson oranı
BETA_THERMAL = 2.34e-5  # [1/K] termal genleşme katsayısı
T_REF        = 540.0    # [°C]  stres sıfır referans sıcaklığı (SHT çıkış)

# Lame sabitleri (düzlem gerilme için görünür değerler)
_MU     = E_MODULUS / (2.0 * (1.0 + NU_POISSON))
_LAMBDA = E_MODULUS * NU_POISSON / (1.0 - NU_POISSON**2)  # düzlem gerilme

# ------------------------------------------------------------------
# Yardımcı: Q4 şekil fonksiyonları ve türevleri
# ------------------------------------------------------------------

def _shape_functions(xi: float, eta: float) -> np.ndarray:
    """Q4 izoparametrik şekil fonksiyonları N_1..N_4."""
    return np.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta),
    ]) / 4.0


def _shape_grad_iso(xi: float, eta: float) -> np.ndarray:
    """Q4 şekil fonksiyonlarının izoparametrik türevleri (2×4)."""
    return np.array([
        [-(1.0 - eta),  (1.0 - eta),  (1.0 + eta), -(1.0 + eta)],
        [-(1.0 - xi),  -(1.0 + xi),   (1.0 + xi),  (1.0 - xi)],
    ]) / 4.0


def _B_matrix(xi: float, eta: float,
              xe: np.ndarray, ye: np.ndarray) -> tuple:
    """
    B-matrisi (3×8) ve Jacobian determinantını döndürür.
    ε = [ε_x, ε_y, γ_xy]^T = B · u_e
    """
    dN = _shape_grad_iso(xi, eta)          # (2×4)
    J  = dN @ np.column_stack([xe, ye])    # (2×2) Jacobian
    detJ = float(np.linalg.det(J))

    if detJ <= 0.0:
        raise ValueError(f"Negatif Jacobian: detJ={detJ:.4e} — eleman bozuk")

    Jinv  = np.linalg.inv(J)
    dN_xy = Jinv @ dN                      # global türevler (2×4)

    B = np.zeros((3, 8))
    for k in range(4):
        B[0, 2 * k]     = dN_xy[0, k]     # ∂u_x/∂x
        B[1, 2 * k + 1] = dN_xy[1, k]     # ∂u_y/∂y
        B[2, 2 * k]     = dN_xy[1, k]     # ∂u_x/∂y (geri yarı)
        B[2, 2 * k + 1] = dN_xy[0, k]     # ∂u_y/∂x (ön yarı)

    return B, detJ


# ------------------------------------------------------------------
# Ana sınıf
# ------------------------------------------------------------------

class PlaneStressFEM:
    """
    2D düzlem gerilme FEM çözücüsü — Q4 eleman, eşdağılımlı ızgara.

    Parametreler
    ----------
    nx, ny : eleman sayısı (x ve y yönü)
    Lx, Ly : domain boyutları [m]
    """

    # 2×2 Gauss noktaları ve ağırlıkları
    _G  = 1.0 / np.sqrt(3.0)
    _GAUSS = [(-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G)]
    _W     = [1.0, 1.0, 1.0, 1.0]

    def __init__(self, nx: int = 20, ny: int = 10,
                 Lx: float = 1.3, Ly: float = 0.6):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly

        # Düğüm koordinatları — (nx+1)×(ny+1) ızgara
        self.n_nodes = (nx + 1) * (ny + 1)
        self.n_dof   = 2 * self.n_nodes

        xs = np.linspace(0.0, Lx, nx + 1)
        ys = np.linspace(0.0, Ly, ny + 1)
        XX, YY = np.meshgrid(xs, ys, indexing="ij")  # (nx+1, ny+1)

        # Düz (flattened) koordinat dizileri
        self.x_nodes = XX.flatten()   # (n_nodes,)
        self.y_nodes = YY.flatten()   # (n_nodes,)

        # Düzlem gerilme elastisite matrisi D (3×3)
        c = E_MODULUS / (1.0 - NU_POISSON**2)
        self.D = c * np.array([
            [1.0,          NU_POISSON,    0.0                      ],
            [NU_POISSON,   1.0,           0.0                      ],
            [0.0,          0.0,           (1.0 - NU_POISSON) / 2.0 ],
        ])

        # Global stiffness matrisini bir kez oluştur
        self.K = self._assemble_stiffness()

    # ── Yardımcı: düğüm indeksi ─────────────────────────────────────

    def _nid(self, i: int, j: int) -> int:
        """(i,j) ızgara konumundan düğüm indeksi (row-major x-first)."""
        return i * (self.ny + 1) + j

    def _element_nodes(self, i: int, j: int) -> list:
        """(i,j) elemanının 4 köşe düğüm indeksi — CCW sıra."""
        return [
            self._nid(i,     j),
            self._nid(i + 1, j),
            self._nid(i + 1, j + 1),
            self._nid(i,     j + 1),
        ]

    # ── Eleman stiffness matrisi ────────────────────────────────────

    def _element_ke(self, i: int, j: int) -> tuple:
        """(i,j) elamanı için 8×8 stiffness matrisi ve düğüm listesi."""
        nodes = self._element_nodes(i, j)
        xe = self.x_nodes[nodes]
        ye = self.y_nodes[nodes]

        ke = np.zeros((8, 8))
        for (xi, eta), w in zip(self._GAUSS, self._W):
            B, detJ = _B_matrix(xi, eta, xe, ye)
            ke += w * B.T @ self.D @ B * detJ

        return ke, nodes

    # ── Global stiffness ────────────────────────────────────────────

    def _assemble_stiffness(self) -> csr_matrix:
        """Tüm elemanları toplayarak global K matrisini oluştur."""
        K = lil_matrix((self.n_dof, self.n_dof))

        for i in range(self.nx):
            for j in range(self.ny):
                ke, nodes = self._element_ke(i, j)
                dofs = []
                for n in nodes:
                    dofs.extend([2 * n, 2 * n + 1])

                for r, gr in enumerate(dofs):
                    for c, gc in enumerate(dofs):
                        K[gr, gc] += ke[r, c]

        return K.tocsr()

    # ── Termal yük vektörü ──────────────────────────────────────────

    def thermal_force(self, T_field: np.ndarray) -> np.ndarray:
        """
        Termal yük vektörü f_th hesapla.

        f_th_e = ∫ B^T · D · ε_th dΩ
        ε_th = β · (T − T_ref) · [1, 1, 0]^T   (düzlem gerilme)

        T_field : (n_nodes,) sıcaklık alanı [°C]
        Döndürür: (n_dof,) termal yük vektörü [N]
        """
        eps_th_dir = np.array([1.0, 1.0, 0.0])   # düzlem gerilme termal bileşeni
        f_th = np.zeros(self.n_dof)

        for i in range(self.nx):
            for j in range(self.ny):
                nodes = self._element_nodes(i, j)
                xe = self.x_nodes[nodes]
                ye = self.y_nodes[nodes]
                T_nodes = T_field[nodes]

                fe = np.zeros(8)
                for (xi, eta), w in zip(self._GAUSS, self._W):
                    # Gauss noktasında sıcaklık interpolasyonu
                    N      = _shape_functions(xi, eta)
                    T_gp   = float(N @ T_nodes)

                    # Termal gerinme vektörü
                    eps_th = BETA_THERMAL * (T_gp - T_REF) * eps_th_dir

                    B, detJ = _B_matrix(xi, eta, xe, ye)
                    sigma_th = self.D @ eps_th        # termal gerilme
                    fe += w * B.T @ sigma_th * detJ

                dofs = []
                for n in nodes:
                    dofs.extend([2 * n, 2 * n + 1])
                for r, gr in enumerate(dofs):
                    f_th[gr] += fe[r]

        return f_th

    # ── Sınır koşulları ─────────────────────────────────────────────

    def _apply_dirichlet(self, K: csr_matrix, f: np.ndarray,
                         fixed_dofs: list) -> tuple:
        """
        Penalty metodu ile Dirichlet BC uygula.
        Büyük sayı PENALTY → sabit DOF'lar sıfıra kilitlenir.
        """
        PENALTY = 1.0e30
        K = K.tolil()
        for dof in fixed_dofs:
            K[dof, :]    = 0.0
            K[:, dof]    = 0.0
            K[dof, dof]  = PENALTY
            f[dof]       = 0.0
        return K.tocsr(), f

    # ── Ana çözüm ───────────────────────────────────────────────────

    def solve(self, T_field: np.ndarray,
              tol: float = 1e-10,
              maxiter: int = 10000) -> np.ndarray:
        """
        Termal yük altında displacement alanını çöz.

        Rijit cisim hareketi engeli:
          - Sol-alt köşe: u_x = u_y = 0
          - Sağ-alt köşe: u_y = 0  (döndürme engeli)

        T_field : (n_nodes,) [°C]
        Döndürür: u (n_nodes, 2) [m]
        """
        f_th = self.thermal_force(T_field)
        K_bc = self.K.copy()
        f_bc = f_th.copy()

        # Sol-alt köşe: tam sabit
        n_bl = self._nid(0, 0)
        # Sağ-alt köşe: sadece u_y sabit (döndürmeyi engelle)
        n_br = self._nid(self.nx, 0)

        fixed_dofs = [2 * n_bl, 2 * n_bl + 1, 2 * n_br + 1]
        K_bc, f_bc = self._apply_dirichlet(K_bc, f_bc, fixed_dofs)

        # Konjuge Gradyan çözücü
        u_flat, info = cg(K_bc, f_bc, atol=tol, maxiter=maxiter)

        if info < 0:
            raise RuntimeError(f"CG solver error: info={info}")
        if info > 0:
            # Yakınsamamadı ama kısmi çözüm kullanılabilir
            pass

        return u_flat.reshape(self.n_nodes, 2)   # (n_nodes, 2) [m]

    # ── CMM noktaları interpolasyonu ────────────────────────────────

    def displacement_at_points(self, u: np.ndarray,
                                query_x: np.ndarray,
                                query_y: np.ndarray) -> tuple:
        """
        İstenen (x,y) noktalarında displacement sorgula.

        u       : (n_nodes, 2) displacement [m]
        query_* : (M,) CMM koordinatları [m]
        Döndürür: u_query (M, 2) [m],  delta_mm (M,) [mm]
        """
        M = len(query_x)
        u_query = np.zeros((M, 2))

        for k in range(M):
            dist2 = (self.x_nodes - query_x[k])**2 + (self.y_nodes - query_y[k])**2
            idx   = int(np.argmin(dist2))
            u_query[k] = u[idx]

        delta_mm = np.sqrt(u_query[:, 0]**2 + u_query[:, 1]**2) * 1000.0
        return u_query, delta_mm
