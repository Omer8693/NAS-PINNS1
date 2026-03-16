"""
cmm_points.py — CMM Ölçüm Noktaları Tanımlamaları
====================================================
Fig17 CMM nokta etiketleri ve koordinatları.
Paper baseline_paper [2026] Fig17'den alınan gerçek koordinatlar
elinde olmadığından normalize tahmin değerler kullanılmaktadır.

TODO: Paper Fig17 okunduğunda gerçek koordinatlar güncellenmeli.
"""

import numpy as np

# Fig17 CMM nokta etiketleri
CMM_LABELS = [
    "L1206", "R2206", "LS01",  "RS01",  "LS13",  "RS13",
    "LA518", "RA518", "LA525", "RA525", "LA537", "RA537",
    "LA538", "RA538", "LA550", "LA551", "LS35",  "RS35",
    "LS51",  "RS51",  "LS52",  "RS52",  "LA555", "RA555",
]

# Tahmini koordinatlar [m] — simetrik motor braket çerçevesi varsayımı
# Sol (L) noktalar küçük x, Sağ (R) noktalar büyük x
CMM_COORDS = np.array([
    [0.05, 0.10], [1.25, 0.10],  # L1206, R2206  — ön bağlantı
    [0.10, 0.05], [1.20, 0.05],  # LS01,  RS01   — alt sırt
    [0.10, 0.55], [1.20, 0.55],  # LS13,  RS13   — üst sırt
    [0.20, 0.15], [1.10, 0.15],  # LA518, RA518
    [0.25, 0.20], [1.05, 0.20],  # LA525, RA525
    [0.30, 0.30], [1.00, 0.30],  # LA537, RA537
    [0.35, 0.35], [0.95, 0.35],  # LA538, RA538
    [0.40, 0.40], [0.45, 0.40],  # LA550, LA551  — merkez yakın
    [0.50, 0.05], [0.80, 0.05],  # LS35,  RS35   — alt orta
    [0.50, 0.55], [0.80, 0.55],  # LS51,  RS51   — üst orta
    [0.55, 0.50], [0.75, 0.50],  # LS52,  RS52
    [0.60, 0.30], [0.70, 0.30],  # LA555, RA555  — merkez
])

# Paper Fig17 — Measured distortions (Experimental), Bottom layer [mm]
# baseline_paper [2026], Fig17: bar chart'tan manuel olarak okundu (±0.05 mm hassasiyet)
# İşaret kuralı: +CAD yüzeyinin dışına, -CAD yüzeyinin içine doğru
PAPER_MEASURED_MM = np.array([
    -0.60,  # L1206
    -0.20,  # R2206
    +1.20,  # LS01
    +1.10,  # RS01
    +0.15,  # LS13
    -0.05,  # RS13
    -0.15,  # LA518
    -0.05,  # RA518
    +1.35,  # LA525
    +1.00,  # RA525
    -0.10,  # LA537
    -0.10,  # RA537
    +2.10,  # LA538  ← en büyük ölçülen distortion
    -1.00,  # RA538
    +0.90,  # LA550
    -0.20,  # LA551
    -2.05,  # LS35   ← en büyük negatif
    -2.00,  # RS35
    +1.15,  # LS51
    +1.10,  # RS51
    -0.30,  # LS52
    -0.20,  # RS52
    -1.00,  # LA555
    +0.70,  # RA555
])

# Paper Fig18 — Simulated distortions (FEM Modelling), Bottom layer [mm]
# baseline_paper [2026], Fig18: bar chart'tan manuel olarak okundu (±0.05 mm hassasiyet)
PAPER_FEM_MM = np.array([
    -0.10,  # L1206
    -0.10,  # R2206
    +1.10,  # LS01
    +1.10,  # RS01
    +0.40,  # LS13
    +0.40,  # RS13
    -0.10,  # LA518
    -0.10,  # RA518
    +1.10,  # LA525
    +1.00,  # RA525
    +0.50,  # LA537
    +0.50,  # RA537
    +2.10,  # LA538
    -0.50,  # RA538
    +0.75,  # LA550
    -0.20,  # LA551
    -2.00,  # LS35
    -2.00,  # RS35
    +0.95,  # LS51
    +0.95,  # RS51
    -0.30,  # LS52
    -0.25,  # RS52
    -0.90,  # LA555
    +0.55,  # RA555
])

assert len(CMM_LABELS) == len(CMM_COORDS), "Label/coordinate count mismatch"
