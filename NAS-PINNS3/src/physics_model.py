"""
Physics Model — Mortensen et al. (2026) Baseline
=================================================
Kaynak makale:
  "Mitigating distortions in cast automotive subframes:
   A finite element simulation approach"
  Int J Adv Manuf Technol (2026)

Bu modül, FEM yerine PINN ile çözülecek denklemleri içerir:
  1. Isı transfer denklemi  (quenching: film boiling → nucleate → konveksiyon)
  2. Termal gerinim         (εT — Eq. 3)
  3. Viskoplastik akış      (A356 — Eq. 6)
  4. Momentum dengesi       (Eq. 5)
  5. Zaman atlama           (TemporalSkipScheduler — projenin ana yeniliği)
"""

import numpy as np
import torch
from typing import Dict, Optional

from src.config import (
    DEVICE,
    K_THERMAL, RHO_CP, RHO_AL,
    BETA_THERMAL, T_COHERENCY, T_HARDENING, GS_COHERENCY,
)

# ─────────────────────────────────────────────────────────────
# Bölüm 1 — A356 Malzeme Parametreleri  (Table 1, Mortensen 2026)
# ─────────────────────────────────────────────────────────────
# Sıcaklık [°C] referans noktaları — tablo satırları
A356_TEMPS = torch.tensor(
    [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
    dtype=torch.float32, device=DEVICE
)
# F: akış gerilmesi ölçek katsayısı [MPa]
A356_F = torch.tensor(
    [22.0, 21.5, 21.0, 20.0, 18.9, 17.5, 16.4, 16.0, 15.6, 12.3, 7.2, 2.0],
    dtype=torch.float32, device=DEVICE
)
# n: izotermal pekleşme üssü [-]
A356_N = torch.tensor(
    [0.3, 0.3, 0.3, 0.3, 0.27, 0.21, 0.15, 0.125, 0.02, 0.0, 0.0, 0.0],
    dtype=torch.float32, device=DEVICE
)
# m: strain-rate duyarlılık üssü [-]
A356_M = torch.tensor(
    [0.0, 0.0, 0.0, 0.006, 0.016, 0.04, 0.1, 0.15, 0.19, 0.22, 0.25, 0.277],
    dtype=torch.float32, device=DEVICE
)


def interp_material_param(T: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """
    Sıcaklığa bağlı malzeme parametresi için doğrusal interpolasyon.

    Vektörleştirilmiş — GPU'da batch boyutunda çalışır.
    A356_TEMPS aralığı dışındaki değerler sınır noktalarına sabitlenir.
    """
    T_clamped = T.clamp(A356_TEMPS[0], A356_TEMPS[-1])
    idx   = torch.searchsorted(A356_TEMPS, T_clamped).clamp(1, len(A356_TEMPS) - 1)
    T_lo  = A356_TEMPS[idx - 1];  T_hi = A356_TEMPS[idx]
    p_lo  = param[idx - 1];       p_hi = param[idx]
    alpha = (T_clamped - T_lo) / (T_hi - T_lo + 1e-10)
    return p_lo + alpha * (p_hi - p_lo)


# ─────────────────────────────────────────────────────────────
# Bölüm 2 — Su Quenching Isı Transfer Katsayısı  (Sec. 2.3)
# ─────────────────────────────────────────────────────────────

class WaterQuenchHTC:
    """
    Su ile quenching sırasında yüzey-su ısı transfer katsayısı (HTC).

    Mortensen Fig. 6 ve Section 2.3'teki bölgesel korelasyonlar:
      T < T_sat        → Zorla konveksiyon
      T_sat .. T_chf   → Nucleate boiling (baloncuk kaynama)
      T_chf .. T_mfb   → Geçiş bölgesi (lineer interpolasyon)
      T > T_mfb        → Film boiling (Leidenfrost — buhar tabakası)
    """

    def __init__(self,
                 T_sat:    float = 100.0,    # [°C]   — suyun doyma sıcaklığı
                 T_mfb:    float = 300.0,    # [°C]   — minimum film boiling (Leidenfrost)
                 q_chf:    float = 3e6,      # [W/m²] — kritik ısı akısı değeri
                 h_conv:   float = 5000.0,   # [W/m²K]— zorla konveksiyon katsayısı
                 velocity: float = 0.5):     # [m/s]  — raf daldırma hızı
        self.T_sat    = T_sat
        self.T_mfb    = T_mfb
        self.q_chf    = q_chf
        self.h_conv   = h_conv
        self.velocity = velocity

    def htc(self, T_surface: torch.Tensor, T_water: torch.Tensor) -> torch.Tensor:
        """
        Anlık HTC hesabı. Tüm bölgeler torch.where ile GPU'da paralel hesaplanır.

        Girdi : T_surface, T_water — °C cinsinden tensörler
        Çıktı : h — W/m²K cinsinden, [100, 1e5] aralığında sınırlandırılmış

        DÜZELTİLDİ: Film boiling Stefan-Boltzmann formülünde T_surface da
        Kelvin'e çevrilmeli. Eskiden: T_surface**4 (yanlış — °C idi)
        Yeni:    (T_surface + 273.15)**4  (doğru — Kelvin)
        """
        delta_T_sub = self.T_sat - T_water.clamp(max=self.T_sat)
        dT          = (T_surface - T_water).clamp(min=0.1)

        # Bölge 1: zorla konveksiyon — sabit h değeri
        h_con = torch.full_like(T_surface, self.h_conv)

        # Bölge 2: nucleate boiling — Rohsenow benzeri güç yasası korelasyonu
        h_ncb = self.h_conv * (dT / 10.0) ** 2.33

        # Kritik ısı akısı noktasındaki sıcaklık ve katsayı
        T_chf = self.T_sat + 80.0 + 0.5 * delta_T_sub
        h_chf = self.q_chf / (T_chf - T_water + 1.0)

        # Bölge 3: geçiş — CHF noktasından Leidenfrost'a lineer düşüş
        alpha_tr = ((T_surface - T_chf) / (self.T_mfb - T_chf + 1e-6)).clamp(0, 1)
        h_tr     = h_chf * (1 - alpha_tr) + 200.0 * alpha_tr

        # Bölge 4: film boiling — Stefan-Boltzmann radyasyonu + buhar iletimi
        # HATA DÜZELTİLDİ: °C → Kelvin dönüşümü eklendi (T_surface + 273.15)
        T_s_K = T_surface + 273.15
        T_w_K = T_water   + 273.15
        h_fb = 200.0 + 5e-9 * (T_s_K ** 4 - T_w_K ** 4) / (
            T_surface - T_water + 1.0)

        # Sıcaklık bölgesine göre doğru katsayıyı seç
        h = torch.where(T_surface < self.T_sat, h_con,
            torch.where(T_surface < T_chf,      h_ncb,
            torch.where(T_surface < self.T_mfb, h_tr, h_fb)))

        return h.clamp(min=100.0, max=1e5)


# ─────────────────────────────────────────────────────────────
# Bölüm 3 — PDE Rezidüelleri  (PINN kayıp fonksiyonları için)
# ─────────────────────────────────────────────────────────────

def heat_equation_residual(net_out: torch.Tensor,
                            coords:  torch.Tensor,
                            k:       float = K_THERMAL,
                            rho_cp:  float = RHO_CP,
                            ) -> torch.Tensor:
    """
    Isı denklemi rezidüeli  [Eq. içi fizik kısıtı]:
        ρCp · ∂T/∂t = k · ∇²T

    coords : [N, 3] tensör — [t, x, y] sütunları
    net_out: [N, 1] tensör — ağın tahmin ettiği T değerleri

    autograd ile türevler hesaplanır (create_graph=True → L-BFGS için gerekli).
    """
    T     = net_out[:, 0:1]
    grads = torch.autograd.grad(
        T, coords,
        grad_outputs=torch.ones_like(T),
        create_graph=True
    )[0]

    dT_dt = grads[:, 0:1]
    dT_dx = grads[:, 1:2]
    dT_dy = grads[:, 2:3]

    # İkinci mertebe türevler — Laplacian için
    d2T_dx2 = torch.autograd.grad(
        dT_dx, coords, grad_outputs=torch.ones_like(dT_dx), create_graph=True
    )[0][:, 1:2]
    d2T_dy2 = torch.autograd.grad(
        dT_dy, coords, grad_outputs=torch.ones_like(dT_dy), create_graph=True
    )[0][:, 2:3]

    laplacian = d2T_dx2 + d2T_dy2
    return rho_cp * dT_dt - k * laplacian


def thermal_strain_residual(T:      torch.Tensor,
                             T_coh:  float = T_COHERENCY,
                             beta_T: float = BETA_THERMAL,
                             ) -> torch.Tensor:
    """
    Termal gerinim rezidüeli  (Eq. 3, Mortensen 2026):
        εT = -1/3 · βT · (T - T_coh)
    """
    return -1.0 / 3.0 * beta_T * (T - T_coh)


def viscoplastic_residual(sigma_eff: torch.Tensor,
                           eps_dot:   torch.Tensor,
                           phi:       torch.Tensor,
                           T:         torch.Tensor,
                           phi0:      float = 0.01
                           ) -> torch.Tensor:
    """
    Viskoplastik akış denklemi rezidüeli  (Eq. 6, Mortensen 2026):
        σ̄ = F(T) · (φ₀ + φ)^n(T) · (ε̇ᵖ)^m(T)

    Table 1'den sıcaklığa bağlı interpolasyonla alınan F, n, m kullanılır.
    """
    F = interp_material_param(T, A356_F)
    n = interp_material_param(T, A356_N)
    m = interp_material_param(T, A356_M)

    sigma_pred = F * (phi0 + phi.abs()) ** n * (eps_dot.abs() + 1e-12) ** m
    return sigma_eff - sigma_pred


def momentum_balance_residual(sigma_grad: torch.Tensor,
                               rho: float = RHO_AL,
                               g:   float = 9.81
                               ) -> torch.Tensor:
    """
    Momentum denge denklemi  (Eq. 5, Mortensen 2026):
        ∇·σ + ρg = 0   (fully solid bölge)
    """
    gravity        = torch.zeros_like(sigma_grad)
    gravity[:, -1] = -rho * g
    return sigma_grad + gravity


# ─────────────────────────────────────────────────────────────
# Bölüm 4 — Adaptif Zaman Atlama  (Projenin Ana Yeniliği)
# ─────────────────────────────────────────────────────────────

class TemporalSkipScheduler:
    """
    Sabit adımlı FEM zaman ilerlemesinin (Mortensen) yerine,
    rezidüel tabanlı adaptif zaman atlama stratejisi.

    Amaç: aynı doğrulukta %30-50 daha az hesaplama adımı.

    Temel mantık:
      - Quenching başlangıcında yüksek gradyan → atlama yapma, küçük adımlarla ilerle
      - Termal denge yakınında düşük gradyan  → max_skip kadar adım atla
      - Kriter: son_rezidüel < skip_threshold → bir sonraki zaman dilimine atla
    """

    def __init__(self,
                 t_start:         float,
                 t_end:           float,
                 n_initial_steps: int   = 20,
                 skip_threshold:  float = 1e-4,
                 max_skip:        int   = 5):
        self.t_start        = t_start
        self.t_end          = t_end
        self.skip_threshold = skip_threshold
        self.max_skip       = max_skip

        self.t_full      = np.linspace(t_start, t_end, n_initial_steps)
        self.visited     = []
        self.skipped     = []
        self.current_idx = 0

    def next_timestep(self, last_residual: Optional[float] = None) -> Optional[float]:
        """
        Bir sonraki zaman noktasını döndür.
        None döndürmesi: tüm zaman adımları tükendi, eğitim bitti.
        """
        if self.current_idx >= len(self.t_full):
            return None

        if last_residual is not None and last_residual < self.skip_threshold:
            skip = min(self.max_skip, len(self.t_full) - self.current_idx - 1)
            if skip > 1:
                for s in range(1, skip):
                    self.skipped.append(float(self.t_full[self.current_idx + s]))
                self.current_idx += skip

        t = float(self.t_full[self.current_idx])
        self.visited.append(t)
        self.current_idx += 1
        return t

    def get_stats(self) -> Dict:
        """Zaman atlama istatistiklerini döndür."""
        total     = len(self.t_full)
        n_visited = len(self.visited)
        n_skipped = len(self.skipped)
        return {
            "total_steps":    total,
            "visited":        n_visited,
            "skipped":        n_skipped,
            "efficiency_pct": round(100.0 * n_skipped / max(total, 1), 1),
            "visited_times":  self.visited,
            "skipped_times":  self.skipped,
        }

    def reset(self):
        """Zamanlayıcıyı sıfırla — yeni eğitim turu için."""
        self.visited     = []
        self.skipped     = []
        self.current_idx = 0
