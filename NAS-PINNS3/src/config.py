"""
config.py — Ortak Sabitler ve Ayarlar
======================================
Tüm modüller bu dosyadan içe aktarır.
Değer değiştirmek için sadece buraya bakın.

Kullanım:
    from src.config import DEVICE, LAYERS_MIN, T_INIT, W_BOUNDARY
"""

import torch

# ─────────────────────────────────────────────────────────────
# Cihaz: GPU varsa kullan, yoksa CPU
# ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# NAS Arama Uzayı (NAS-PINNS1 ile hizalanmış)
# ─────────────────────────────────────────────────────────────
LAYERS_MIN  = 3
LAYERS_MAX  = 6
NEURONS_MIN = 64
NEURONS_MAX = 160

# ─────────────────────────────────────────────────────────────
# Mortensen 2026 — Proses Parametreleri
# ─────────────────────────────────────────────────────────────
T_INIT       = 540.0   # [°C] — SHT çıkış sıcaklığı (kohezyon)
T_WATER_INIT = 20.0    # [°C] — quenching suyu başlangıç sıcaklığı
T_END        = 30.0    # [s]  — toplam quenching süresi
X_MAX        = 1.3     # [m]  — subframe uzunluğu
Y_MAX        = 0.6     # [m]  — subframe genişliği

# Analitik soğuma yaklaşımı katsayısı (2D Robin BC'nin temel modu)
# T(t) ≈ T_water + (T_init - T_water) · exp(-ALPHA · t)
# 2D slab (1.3m×0.6m), Robin BC (h=5000, K=150):
#   β_1: β_1*tan(β_1*L_x) = h/K → β_1 ≈ 2.31/m,  L_x = X_MAX/2 = 0.65m
#   μ_1: μ_1*tan(μ_1*L_y) = h/K → μ_1 ≈ 4.76/m,  L_y = Y_MAX/2 = 0.30m
#   ALPHA = D*(β_1² + μ_1²) = 6.25e-5 * (5.34 + 22.7) = 1.75e-3 [1/s]
# Not: 0.05 /s değeri bu geometri için ~29× fazla hızlı soğuma verir.
ALPHA = 1.75e-3        # [1/s] — 2D Robin BC temel mod soğuma katsayısı

# ─────────────────────────────────────────────────────────────
# Fiziksel Sabitler — A356 Alüminyum (Mortensen 2026)
# ─────────────────────────────────────────────────────────────
K_THERMAL    = 150.0   # [W/mK]  — ısıl iletkenlik
RHO_CP       = 2.4e6   # [J/m³K] — hacimsel ısı kapasitesi
RHO_AL       = 2700.0  # [kg/m³] — yoğunluk
BETA_THERMAL = 2.34e-5 # [1/K]   — termal genleşme katsayısı
T_COHERENCY  = 540.0   # [°C]    — kohezyon sıcaklığı
T_HARDENING  = 420.0   # [°C]    — work-hardening başlangıç sıcaklığı
GS_COHERENCY = 0.9     # [-]     — mushy zone → fully-solid eşiği

# ─────────────────────────────────────────────────────────────
# PINN Kayıp Ağırlıkları
# ─────────────────────────────────────────────────────────────
W_PHYSICS  = 1.0
W_BOUNDARY = 10.0
W_INITIAL  = 10.0
W_DATA     = 5.0

# ─────────────────────────────────────────────────────────────
# Kollokasiyon Nokta Sayıları (varsayılan)
# ─────────────────────────────────────────────────────────────
N_DOMAIN   = 2000
N_BOUNDARY = 400
N_INITIAL  = 400
N_TIME_STEPS = 20
