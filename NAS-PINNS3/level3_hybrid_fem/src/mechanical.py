"""
mechanical.py — Termal Genleşme → Distortion Zinciri
======================================================
T(x,y,t_final) → ε_thermal → σ → δ [mm]

Basit lineer elastik varsayım.
Fig17 CMM noktalarıyla karşılaştırma için gerekli minimum model.

Not: Bu yaklaşım gerçek FEM mekaniğinin yerini tutmaz.
Sadece birinci derece tahmin (order-of-magnitude) içindir.
Gerçek distortion için Abaqus / FEniCS gerekir.
"""

import numpy as np
import torch

# A356 alüminyum malzeme sabitleri — baseline_paper [2026], Table 1
BETA_THERMAL = 2.34e-5   # [1/K] termal genleşme katsayısı
E_MODULUS    = 70.0e9    # [Pa]  oda sıcaklığı elastisite modülü
NU_POISSON   = 0.33      # [-]   Poisson oranı
T_REF        = 540.0     # [°C]  kohezyon sıcaklığı (sıfır stress referansı)

# Fig17 CMM ölçüm noktaları — paper'dan alınacak gerçek koordinatlar
# Şimdilik normalize tahmini koordinatlar (m cinsinden)
# TODO: paper Fig17'den gerçek x,y koordinatlarını oku
CMM_POINT_LABELS = [
    "L1206", "R2206", "LS01",  "RS01",  "LS13",  "RS13",
    "LA518", "RA518", "LA525", "RA525", "LA537", "RA537",
    "LA538", "RA538", "LA550", "LA551", "LS35",  "RS35",
    "LS51",  "RS51",  "LS52",  "RS52",  "LA555", "RA555",
]

# Tahmini koordinatlar — paper okunduğunda güncellenecek
CMM_COORDS_M = np.array([
    [0.05, 0.10], [1.25, 0.10], [0.10, 0.05], [1.20, 0.05],
    [0.10, 0.55], [1.20, 0.55], [0.20, 0.15], [1.10, 0.15],
    [0.25, 0.20], [1.05, 0.20], [0.30, 0.30], [1.00, 0.30],
    [0.35, 0.35], [0.95, 0.35], [0.40, 0.40], [0.45, 0.40],
    [0.50, 0.05], [0.80, 0.05], [0.50, 0.55], [0.80, 0.55],
    [0.55, 0.50], [0.75, 0.50], [0.60, 0.30], [0.70, 0.30],
])


def thermal_strain(T_field: torch.Tensor) -> torch.Tensor:
    """
    ε_thermal = β · (T - T_ref)
    T_ref: kohezyon sıcaklığı — bu noktadan itibaren stress birikmesi başlar.
    """
    return BETA_THERMAL * (T_field - T_REF)


def estimate_distortion_at_cmm(T_field:  torch.Tensor,
                                x_grid:   torch.Tensor,
                                y_grid:   torch.Tensor,
                                cmm_coords: np.ndarray = None,
                                L_char:   float = 0.01) -> np.ndarray:
    """
    CMM noktalarında basit termal distortion tahmini.

    Yöntem:
      δ ≈ ε_thermal · L_char  [m]  →  [mm]

    cmm_coords: (M, 2) array [m], varsayılan CMM_COORDS_M
    L_char: karakteristik uzunluk [m] — eleman boyutu tahmini
    Döndürür: (M,) distortion değerleri [mm]
    """
    if cmm_coords is None:
        cmm_coords = CMM_COORDS_M

    eps = thermal_strain(T_field)   # (N,) alan üzerinde
    distortions = []

    for px, py in cmm_coords:
        # En yakın grid noktasını bul
        dist = (x_grid - px) ** 2 + (y_grid - py) ** 2
        idx  = int(torch.argmin(dist).item())

        eps_local = float(eps[idx].item())
        delta_mm  = eps_local * L_char * 1000.0   # m → mm
        distortions.append(delta_mm)

    return np.array(distortions)


def compare_with_paper(pinn_distortions: np.ndarray,
                       paper_fem:        np.ndarray,
                       paper_measured:   np.ndarray,
                       labels:           list = None) -> dict:
    """
    PINN distortion tahminini paper Fig17 verileriyle karşılaştır.

    paper_fem, paper_measured: (M,) paper'dan okunan referans değerler [mm]
    Döndürür: karşılaştırma metrik sözlüğü
    """
    if labels is None:
        labels = CMM_POINT_LABELS

    mae_pinn_vs_fem      = float(np.mean(np.abs(pinn_distortions - paper_fem)))
    mae_pinn_vs_measured = float(np.mean(np.abs(pinn_distortions - paper_measured)))
    mae_fem_vs_measured  = float(np.mean(np.abs(paper_fem        - paper_measured)))

    return {
        "mae_pinn_vs_fem_mm":      mae_pinn_vs_fem,
        "mae_pinn_vs_measured_mm": mae_pinn_vs_measured,
        "mae_fem_vs_measured_mm":  mae_fem_vs_measured,
        "per_point": [
            {
                "label":    labels[i],
                "pinn_mm":  float(pinn_distortions[i]),
                "fem_mm":   float(paper_fem[i])        if paper_fem is not None        else None,
                "meas_mm":  float(paper_measured[i])   if paper_measured is not None   else None,
            }
            for i in range(len(labels))
        ],
    }
