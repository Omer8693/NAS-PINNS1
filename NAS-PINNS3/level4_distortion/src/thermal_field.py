"""
thermal_field.py — 2D Uzaysal Sıcaklık Alanı Üretici
======================================================
Level 3 T_means geçmişinden 2D uzaysal T(x,y) alanı oluşturur.

Fizik modeli:
  t=30s sonunda quenching sıcaklığı yaklaşık düzgündür.
  Uzaysal gradyan T_mean etrafında sınırlı bir pertürbasyon olarak modellenir:
    T(x,y) = T_mean + ΔT · f(x,y)
  burada f sıfır-ortalamalı ve [-1,1] aralığında normalize edilmiştir,
  ΔT = gradient_frac · (T_mean - T_water) fiziksel bir sınır sağlar.

Profil seçenekleri:
  "gradient"   : doğrusal x-gradyanı — asimetrik quench girişi (bir uç önce soğur)
  "cosine"     : 2D kosinüs — merkez sıcak, kenar soğuk (sınırlı ΔT ile)
  "parabolic"  : 2D parabolik — merkez sıcak, kenar soğuk (sınırlı ΔT ile)
  "uniform"    : sabit T_mean — uzaysal gradyan yok
"""

import numpy as np

# Domain boyutları — fem_interface.py ile aynı
LX = 1.3   # [m]
LY = 0.6   # [m]

# Su/quench sıcaklığı
T_WATER = 20.0    # [°C]

# Uzaysal gradyan fraksiyonu — ΔT = gradient_frac * (T_mean - T_water)
# 0.15: %15 sapma → fiziksel olarak makul (t=30s quenching sonrası)
DEFAULT_GRADIENT_FRAC = 0.15


def analytical_profile(x_nodes: np.ndarray,
                        y_nodes: np.ndarray,
                        T_mean: float,
                        profile: str = "gradient",
                        gradient_frac: float = DEFAULT_GRADIENT_FRAC) -> np.ndarray:
    """
    T_mean değerinden 2D uzaysal T(x,y) alanı oluştur.

    T(x,y) = T_mean + ΔT · f_normalized(x,y)
    ΔT = gradient_frac * (T_mean - T_water)  → T her zaman [T_mean-ΔT, T_mean+ΔT] içinde

    T_mean        : ortalama sıcaklık [°C] (Level 3 son zaman adımından)
    gradient_frac : uzaysal gradyan büyüklüğü (varsayılan 0.15)
    Döndürür      : T_field (n_nodes,) [°C]
    """
    # Normalize koordinatlar: [0,Lx] → [-1,1] ve [0,Ly] → [-1,1]
    xi  = 2.0 * x_nodes / LX - 1.0   # [-1, 1]
    eta = 2.0 * y_nodes / LY - 1.0   # [-1, 1]

    # Uzaysal gradyan büyüklüğü — T_mean - T_water kadar ısı farkının fraksiyonu
    delta_T = gradient_frac * (T_mean - T_WATER)

    if profile == "gradient":
        # Doğrusal x-gradyanı: sol taraf sıcak, sağ taraf soğuk
        # Asimetrik quench girişini simüle eder (parça bir ucundan girer)
        f_norm = xi   # [-1, 1], sıfır ortalamalı (doğrusal → integral=0)
        T_field = T_mean + delta_T * f_norm

    elif profile == "cosine":
        # Kosinüs temel modu — sıfır-ortalamalı hale getir
        f_raw  = np.cos(np.pi * xi / 2.0) * np.cos(np.pi * eta / 2.0)
        f_mean = float(f_raw.mean())
        f_max  = float(f_raw.max() - f_mean)
        f_norm = (f_raw - f_mean) / (f_max + 1e-12)   # [-1, 1], sıfır ortalamalı
        T_field = T_mean + delta_T * f_norm

    elif profile == "parabolic":
        # Parabolik profil — sıfır-ortalamalı hale getir
        f_raw  = (1.0 - xi**2) * (1.0 - eta**2)
        f_mean = float(f_raw.mean())
        f_max  = float(f_raw.max() - f_mean)
        f_norm = (f_raw - f_mean) / (f_max + 1e-12)   # [-1, 1], sıfır ortalamalı
        T_field = T_mean + delta_T * f_norm

    else:  # "uniform"
        T_field = np.full_like(x_nodes, T_mean)

    return T_field.astype(np.float64)


def field_stats(T_field: np.ndarray, label: str = "") -> dict:
    """Sıcaklık alanının özet istatistiklerini döndür."""
    return {
        "label":  label,
        "T_min":  float(T_field.min()),
        "T_max":  float(T_field.max()),
        "T_mean": float(T_field.mean()),
        "T_std":  float(T_field.std()),
        "delta_T": float(T_field.max() - T_field.min()),
    }
