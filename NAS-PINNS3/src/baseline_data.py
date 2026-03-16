"""
baseline_data.py
================
baseline_paper et al. [2026] paper'ından manuel olarak çıkarılan
tüm sayısal veriler.

KAYNAK: "Mitigating distortions in cast automotive subframes:
         A finite element simulation approach"
         Int J Adv Manuf Technol (2026)
         doi:10.1007/s00170-026-17515-w

NASIL KULLANILIR:
    from src.baseline_data import BaselinePaper
    bl = BaselinePaper()
    df = bl.get("fig15_ac_distortion")
    bl.compare_with_pinn(pinn_predictions, dataset="fig17_ht")

VERİ KAYNAĞI NOTU:
    - Table 1  → doğrudan sayısal tablo
    - Table 2  → doğrudan sayısal tablo
    - Fig 15   → bar chart'tan manuel okuma (±0.05mm tolerans)
    - Fig 16   → bar chart'tan manuel okuma
    - Fig 17   → bar chart'tan manuel okuma
    - Fig 19   → bar chart'tan manuel okuma
    - Fig 21   → bar chart'tan manuel okuma
    - Fig 7    → grafik'ten manuel okuma (±2°C tolerans)
"""

import numpy as np
import json
import os
from typing import Dict, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────
# 1. TABLO 1 – A356 Akış Gerilmesi Parametreleri
#    Kaynak: Table 1, baseline_paper [2026] (DOĞRUDAN TABLO)
# ─────────────────────────────────────────────────────────────
TABLE1_A356 = {
    "description": "A356 empirical flow stress parameters (Table 1, baseline_paper [2026])",
    "source":      "Table 1",
    "columns":     ["Temp_C", "F_MPa", "n", "m"],
    "data": [
        [0,    22.0,  0.300, 0.000],
        [50,   21.5,  0.300, 0.000],
        [100,  21.0,  0.300, 0.000],
        [150,  20.0,  0.300, 0.006],
        [200,  18.9,  0.270, 0.016],
        [250,  17.5,  0.210, 0.040],
        [300,  16.4,  0.150, 0.100],
        [350,  16.0,  0.125, 0.150],
        [400,  15.6,  0.020, 0.190],
        [450,  12.3,  0.000, 0.220],
        [500,   7.2,  0.000, 0.250],
        [550,   2.0,  0.000, 0.277],
    ]
}

# ─────────────────────────────────────────────────────────────
# 2. TABLO 2 – Deplasman-Rijitlik İlişkisi
#    Kaynak: Table 2, baseline_paper [2026] (DOĞRUDAN TABLO)
# ─────────────────────────────────────────────────────────────
TABLE2_STIFFNESS = {
    "description": "Displacement vs stiffness for air-gap boundary (Table 2, baseline_paper [2026])",
    "source":      "Table 2",
    "columns":     ["displacement_m", "stiffness_Pa_per_m"],
    "data": [
        [-0.10000, 1e5],
        [-0.00020, 5e5],
        [-0.00008, 5e6],
        [-0.00005, 5e7],
        [-0.00002, 5e8],
        [-0.00001, 1e9],
        [ 0.00000, 1e10],
        [ 0.00001, 1e10],
        [ 0.00002, 1e11],
        [ 0.00003, 1e12],
        [ 0.00005, 1e13],
    ]
}

# ─────────────────────────────────────────────────────────────
# 3. FIG 15 – As-Cast Distorsiyonları (Ölçülen vs Simüle)
#    Kaynak: Fig 15, bar chart, manuel okuma (±0.05mm)
#    Birim: mm, pozitif = CAD'dan dışarı, negatif = içeri
# ─────────────────────────────────────────────────────────────
FIG15_AC_DISTORTION = {
    "description": "As-cast distortions: measured vs simulated (Fig 15, baseline_paper [2026])",
    "source":      "Fig 15",
    "unit":        "mm",
    "note":        "Manuel bar chart okuma, tolerans ±0.05mm",
    "points": ["L1206","R2206","LS01","RS01","LS13","RS13",
               "LA518","RA518","LA525","RA525","LA537","RA537",
               "LA538","RA538","LA550","LA551","LS35","RS35",
               "LS51","RS51","LS52","RS52","LA555","RA555"],
    # Bar chart'tan okunan değerler (Fig 15):
    "AC_Measured": [
         0.20,  0.15,  0.05, -0.05,  0.25,  0.20,
        -0.10, -0.15,  0.50,  0.45,  0.10,  0.05,
        -0.30, -0.35,  0.00,  0.05,  0.80,  0.75,
         1.00,  0.95,  1.30,  1.25, -1.80, -1.75
    ],
    "Simulated": [
         0.18,  0.12,  0.03, -0.08,  0.22,  0.18,
        -0.12, -0.18,  0.48,  0.42,  0.08,  0.02,
        -0.32, -0.38, -0.02,  0.02,  0.75,  0.70,
         0.95,  0.90,  1.25,  1.20, -1.70, -1.65
    ]
}

# ─────────────────────────────────────────────────────────────
# 4. FIG 16 – Yalnızca Creep İle Isıl İşlem Distorsiyonları
#    Kaynak: Fig 16, bar chart (GT1..GT9 noktaları)
# ─────────────────────────────────────────────────────────────
FIG16_CREEP_DISTORTION = {
    "description": "HT distortions (creep only, air cooling) (Fig 16, baseline_paper [2026])",
    "source":      "Fig 16",
    "unit":        "mm",
    "points":      ["GT1","GT2","GT3","GT4","GT5","GT6","GT7","GT8","GT9"],
    "Measured": [
         0.50,  0.45, -0.20, -0.15, -2.40, -0.10,  1.70,  1.60,  0.30
    ],
    "Simulated": [
         0.45,  0.40, -0.22, -0.18, -2.30, -0.08,  1.60,  1.50,  0.28
    ]
}

# ─────────────────────────────────────────────────────────────
# 5. FIG 17 – Quenching Distorsiyonları (İlk Denemeler, 3 Katman)
#    Kaynak: Fig 17 (ölçülen) ve Fig 18 (simüle edilmiş)
#    3 katman: Bottom (yeşil), Middle (turuncu), Top (gri)
# ─────────────────────────────────────────────────────────────
FIG17_HT_QUENCH_3LAYER = {
    "description": "HT quenching distortions, 3-layer rack, first trials (Fig 17-18, baseline_paper [2026])",
    "source":      "Fig 17 (measured) + Fig 18 (simulated)",
    "unit":        "mm",
    "points": ["L1206","R2206","LS01","RS01","LS13","RS13",
               "LA518","RA518","LA525","RA525","LA537","RA537",
               "LA538","RA538","LA550","LA551","LS35","RS35",
               "LS51","RS51","LS52","RS52","LA555","RA555"],
    # Ölçülen (Fig 17)
    "Measured_Bottom": [
        -0.50, -0.55,  0.00,  0.05, -0.70, -0.60,
         0.10,  0.05,  1.30,  1.20, -0.20, -0.25,
        -0.80, -0.90, -0.50, -0.40,  0.00,  0.05,
        -1.80, -1.90, -1.20, -1.30,  2.10,  2.00
    ],
    "Measured_Middle": [
        -0.20, -0.25,  0.10,  0.08, -0.30, -0.25,
         0.05,  0.02,  0.50,  0.45, -0.10, -0.12,
        -0.30, -0.35, -0.15, -0.10,  0.10,  0.12,
        -0.60, -0.65, -0.40, -0.45,  0.80,  0.75
    ],
    "Measured_Top": [
        -0.10, -0.12,  0.15,  0.12, -0.15, -0.10,
         0.03,  0.01,  0.25,  0.20, -0.05, -0.08,
        -0.10, -0.12, -0.05, -0.02,  0.20,  0.22,
        -0.30, -0.35, -0.20, -0.25,  0.40,  0.38
    ],
    # Simüle edilmiş (Fig 18)
    "Simulated_Bottom": [
        -0.48, -0.52,  0.02,  0.04, -0.68, -0.58,
         0.08,  0.03,  1.25,  1.15, -0.22, -0.27,
        -0.82, -0.92, -0.52, -0.42,  0.02,  0.06,
        -1.75, -1.85, -1.15, -1.25,  2.05,  1.95
    ],
    "Simulated_Middle": [
        -0.18, -0.22,  0.08,  0.06, -0.28, -0.22,
         0.03,  0.00,  0.45,  0.40, -0.12, -0.14,
        -0.28, -0.32, -0.12, -0.08,  0.08,  0.10,
        -0.55, -0.60, -0.38, -0.42,  0.75,  0.70
    ],
    "Simulated_Top": [
        -0.08, -0.10,  0.12,  0.10, -0.12, -0.08,
         0.01, -0.01,  0.20,  0.15, -0.07, -0.10,
        -0.08, -0.10, -0.03,  0.00,  0.18,  0.20,
        -0.28, -0.32, -0.18, -0.22,  0.38,  0.35
    ]
}

# ─────────────────────────────────────────────────────────────
# 6. FIG 19 – Optimize Edilmiş Raf, 5 Katman
#    Kaynak: Fig 19 (ölçülen) + Fig 20 (simüle)
#    Toplam distorsiyon ~1mm'ye düştü
# ─────────────────────────────────────────────────────────────
FIG19_HT_OPTIMIZED_5LAYER = {
    "description": "HT quenching distortions, optimized 5-layer rack (Fig 19-20, baseline_paper [2026])",
    "source":      "Fig 19 (measured) + Fig 20 (simulated)",
    "unit":        "mm",
    "note":        "Optimized rack: distortion reduced to ~1mm",
    "points": ["L1206","R2206","LS01","RS01","LS13","RS13",
               "LA518","RA518","LA525","RA525","LA537","RA537",
               "LA538","RA538","LA550","LA551","LS35","RS35",
               "LS51","RS51","LS52","RS52","LA555","RA555"],
    "Measured_L1": [ 0.15, 0.12, 0.05,-0.05, 0.20, 0.15,-0.05,-0.08, 0.40, 0.35,-0.08,-0.10,-0.25,-0.28,-0.15,-0.12, 0.05, 0.08,-0.50,-0.55,-0.35,-0.38, 0.70, 0.65],
    "Measured_L2": [ 0.08, 0.06, 0.08, 0.05, 0.10, 0.08,-0.02,-0.04, 0.20, 0.15,-0.04,-0.06,-0.10,-0.12,-0.05,-0.03, 0.08, 0.10,-0.25,-0.28,-0.15,-0.18, 0.30, 0.28],
    "Measured_L3": [ 0.03, 0.02, 0.10, 0.08, 0.05, 0.03, 0.00,-0.02, 0.10, 0.08,-0.02,-0.03,-0.05,-0.06,-0.02, 0.00, 0.10, 0.12,-0.10,-0.12,-0.08,-0.10, 0.15, 0.12],
    "Measured_L4": [-0.02,-0.03, 0.12, 0.10, 0.02, 0.00, 0.02, 0.00, 0.05, 0.03,-0.01,-0.02,-0.02,-0.03, 0.00, 0.02, 0.12, 0.14,-0.05,-0.06,-0.03,-0.04, 0.08, 0.06],
    "Measured_L5": [-0.05,-0.06, 0.14, 0.12, 0.00,-0.02, 0.04, 0.02, 0.02, 0.00, 0.00,-0.01, 0.00,-0.01, 0.02, 0.04, 0.15, 0.16,-0.02,-0.03,-0.01,-0.02, 0.04, 0.02],
    "Simulated_L1":[ 0.14, 0.11, 0.04,-0.06, 0.18, 0.13,-0.06,-0.09, 0.38, 0.32,-0.09,-0.11,-0.26,-0.30,-0.16,-0.13, 0.04, 0.07,-0.48,-0.52,-0.33,-0.36, 0.68, 0.62],
    "Simulated_L5":[-0.06,-0.07, 0.12, 0.10,-0.02,-0.04, 0.03, 0.01, 0.01,-0.01,-0.01,-0.02,-0.01,-0.02, 0.01, 0.03, 0.13, 0.15,-0.03,-0.04,-0.02,-0.03, 0.03, 0.01],
}

# ─────────────────────────────────────────────────────────────
# 7. FIG 21 – Cross-Member CMM Noktaları (6 Katman)
#    Kaynak: Fig 21 (ölçülen) + Fig 22 (simüle)
#    MP1..MP6, MP24..MP31
# ─────────────────────────────────────────────────────────────
FIG21_CROSSMEMBER_6LAYER = {
    "description": "Cross-member CMM distortions, 6-layer rack (Fig 21-22, baseline_paper [2026])",
    "source":      "Fig 21 (measured) + Fig 22 (simulated)",
    "unit":        "mm",
    "points":      ["MP1","MP2","MP3","MP4","MP5","MP6","MP24","MP25","MP26","MP27","MP28","MP31"],
    # Layer 1 en büyük distorsiyon, layer 6 en küçük
    "Measured_L1": [-1.10,  0.80, -0.30,  0.60,  0.40, -0.20, -0.90,  1.20,  0.70, -0.50,  0.30, -0.40],
    "Measured_L2": [-0.70,  0.50, -0.20,  0.40,  0.25, -0.12, -0.55,  0.75,  0.45, -0.30,  0.18, -0.25],
    "Measured_L3": [-0.45,  0.30, -0.12,  0.25,  0.15, -0.08, -0.35,  0.50,  0.28, -0.18,  0.10, -0.15],
    "Measured_L4": [-0.20,  0.15, -0.06,  0.12,  0.08, -0.04, -0.16,  0.22,  0.12, -0.08,  0.05, -0.07],
    "Measured_L5": [-0.10,  0.08, -0.03,  0.06,  0.04, -0.02, -0.08,  0.12,  0.06, -0.04,  0.02, -0.03],
    "Measured_L6": [-0.05,  0.04, -0.01,  0.03,  0.02, -0.01, -0.04,  0.06,  0.03, -0.02,  0.01, -0.02],
    "Simulated_L1":[-1.05,  0.75, -0.28,  0.55,  0.38, -0.18, -0.85,  1.15,  0.65, -0.45,  0.28, -0.38],
    "Simulated_L6":[-0.06,  0.05, -0.02,  0.04,  0.02, -0.01, -0.05,  0.07,  0.04, -0.02,  0.01, -0.02],
}

# ─────────────────────────────────────────────────────────────
# 8. FIG 7 – Su Tankı Sıcaklığı (Zaman Serisi)
#    Kaynak: Fig 7, manuel grafik okuma (~±2°C tolerans)
#    0-30 saniye, 12 kanal
# ─────────────────────────────────────────────────────────────
FIG7_WATER_TEMP = {
    "description": "Water tank temperature vs time, 12 channels (Fig 7, baseline_paper [2026])",
    "source":      "Fig 7",
    "unit":        "°C",
    "time_s":      [0, 2, 4, 6, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30],
    # Kanallar: farklı derinlik / konum
    "Channel_1":  [62, 65, 72, 80, 88, 92, 94, 93, 91, 89, 87, 85, 83, 82],  # üst
    "Channel_2":  [61, 63, 68, 75, 83, 88, 91, 90, 88, 86, 84, 82, 81, 80],
    "Channel_3":  [60, 62, 66, 72, 79, 85, 88, 87, 85, 83, 82, 80, 79, 78],
    "Channel_6":  [60, 61, 63, 66, 70, 74, 78, 80, 79, 77, 76, 75, 74, 73],
    "Channel_9":  [60, 61, 62, 64, 67, 70, 73, 76, 75, 74, 73, 72, 71, 70],
    "Channel_12": [60, 60, 61, 62, 64, 66, 68, 71, 72, 71, 70, 69, 68, 68],  # alt
}

# ─────────────────────────────────────────────────────────────
# 9. PROSES PARAMETRELER (metin'den çıkarılan)
# ─────────────────────────────────────────────────────────────
PROCESS_PARAMS = {
    "description": "Key process parameters from baseline_paper [2026]",
    "solution_heat_treatment_temp_C": 540,
    "quenching_medium": "water",
    "T_water_initial_C": 20,
    "gs_coherency_threshold": 0.9,
    "T0_work_hardening_C": 420,
    "alloy": "A356 (AlSi7Mg)",
    "subframe_diagonal_m": 1.3,
    "subframe_weight_kg": 18,
    "rack_parts": 32,
    "quenching_time_s": 30,
    "air_cooling_before_quench_s": "few seconds",
    "tangential_stiffness_Pa_per_m": 1e8,
    "cold_mould_expansion_pct": "0.5 to 1.0",
    "mould_corner_gap_mm": 1.5,
    "FEM_elements": 3e6,
    "FEM_domains": 27,
    "distortion_tolerance_mm": "few hundred microns",
    "max_observed_distortion_mm": 2.5,
}


# ─────────────────────────────────────────────────────────────
# BASELINE SINIFI – Ana arayüz
# ─────────────────────────────────────────────────────────────

class BaselinePaper:
    """
    baseline_paper et al. [2026] paper'ından çıkarılan tüm verileri
    yönetir ve PINN tahminleriyle karşılaştırma yapar.

    Kullanım:
        bl = BaselinePaper()
        bl.summary()
        bl.plot_baseline("fig17_ht")
        metrics = bl.compare_with_pinn(pinn_preds, "fig17_ht", layer="Bottom")
        bl.plot_comparison(pinn_preds, "fig17_ht", save_path="results/")
    """

    DATASETS = {
        "table1_a356":        TABLE1_A356,
        "table2_stiffness":   TABLE2_STIFFNESS,
        "fig15_ac":           FIG15_AC_DISTORTION,
        "fig16_creep":        FIG16_CREEP_DISTORTION,
        "fig17_ht":           FIG17_HT_QUENCH_3LAYER,
        "fig19_optimized":    FIG19_HT_OPTIMIZED_5LAYER,
        "fig21_crossmember":  FIG21_CROSSMEMBER_6LAYER,
        "fig7_water_temp":    FIG7_WATER_TEMP,
        "process_params":     PROCESS_PARAMS,
    }

    def get(self, name: str) -> dict:
        if name not in self.DATASETS:
            raise KeyError(f"Dataset '{name}' not found. Available: {list(self.DATASETS.keys())}")
        return self.DATASETS[name]

    def summary(self):
        print("\n" + "="*60)
        print("  Baseline Paper [2026] Data Inventory")
        print("="*60)
        for k, v in self.DATASETS.items():
            desc = v.get("description", "")[:55]
            print(f"  {k:<22} | {desc}")
        print("="*60)

    def list_layers(self, dataset: str) -> list:
        """Return measured layer names available in a dataset."""
        ds = self.get(dataset)
        layers = []
        for key in ds.keys():
            if key.startswith("Measured_"):
                layers.append(key.replace("Measured_", ""))
        if not layers:
            if "AC_Measured" in ds:
                return ["AC"]
            if "Measured" in ds:
                return ["Measured"]
        return layers

    # ── Karşılaştırma metriklerini hesapla ──────────────────

    def compare_with_pinn(self,
                           pinn_predictions: np.ndarray,
                           dataset: str = "fig17_ht",
                           layer: str = "Bottom",
                           key_measured: str = None) -> Dict:
        """
        PINN tahminlerini paper baseline ile karşılaştır.

        pinn_predictions: [N] numpy array, paper ile aynı CMM noktaları sırasında
        dataset: hangi veri seti kullanılacak
        layer: 'Bottom', 'Top', 'L1', vb.

        Döndürür: MAE, RMSE, MaxErr, R2, point-wise fark
        """
        ds = self.get(dataset)
        points = ds["points"]

        # Ölçülen değeri bul
        if key_measured is None:
            # Otomatik anahtar tahmini
            candidates = [k for k in ds.keys() if "Measured" in k and layer in k]
            if not candidates:
                candidates = [k for k in ds.keys() if k == "AC_Measured" or k == "Measured"]
            if not candidates:
                raise ValueError(f"Measurement key not found. Available: {[k for k in ds.keys() if 'Measured' in k or 'AC_' in k]}")
            key_measured = candidates[0]

        measured = np.array(ds[key_measured])

        if len(pinn_predictions) != len(measured):
            raise ValueError(f"Dimension mismatch: PINN={len(pinn_predictions)}, Baseline={len(measured)}")

        diff = pinn_predictions - measured
        mae  = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        max_err = np.max(np.abs(diff))
        ss_res = np.sum(diff**2)
        ss_tot = np.sum((measured - np.mean(measured))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        # Paper'ın kendi FEM simülasyonu ile de karşılaştır
        key_simulated = key_measured.replace("Measured", "Simulated")
        fem_metrics = {}
        if key_simulated in ds:
            fem_pred = np.array(ds[key_simulated])
            fem_diff = fem_pred - measured
            fem_metrics = {
                "FEM_MAE":     float(np.mean(np.abs(fem_diff))),
                "FEM_RMSE":    float(np.sqrt(np.mean(fem_diff**2))),
                "FEM_MaxErr":  float(np.max(np.abs(fem_diff))),
            }

        result = {
            "dataset":      dataset,
            "layer":        layer,
            "n_points":     len(points),
            "PINN_MAE_mm":  float(mae),
            "PINN_RMSE_mm": float(rmse),
            "PINN_MaxErr_mm": float(max_err),
            "PINN_R2":      float(r2),
            "point_diff_mm": diff.tolist(),
            "points":       points,
            **fem_metrics
        }

        self._print_comparison_table(result)
        return result

    def _print_comparison_table(self, result: dict):
        print(f"\n{'─'*55}")
        print(f"  Comparison: {result['dataset']} / {result['layer']}")
        print(f"{'─'*55}")
        print(f"  {'Metrik':<20} | {'PINN':>12} | {'FEM (Paper)':>12}")
        print(f"  {'─'*48}")
        metrics = [("MAE (mm)", "PINN_MAE_mm", "FEM_MAE"),
                   ("RMSE (mm)", "PINN_RMSE_mm", "FEM_RMSE"),
                   ("Max Error (mm)", "PINN_MaxErr_mm", "FEM_MaxErr")]
        for label, pk, fk in metrics:
            pv = f"{result.get(pk, 0):.4f}"
            fv = f"{result.get(fk, 'N/A'):.4f}" if fk in result else "N/A"
            print(f"  {label:<20} | {pv:>12} | {fv:>12}")
        print(f"  {'R²':<20} | {result.get('PINN_R2', 0):>12.4f} | {'─':>12}")
        print(f"{'─'*55}")

    # ── Görselleştirme ────────────────────────────────────────

    def plot_baseline(self, dataset: str, save_path: str = None):
        """Paper'daki baseline verisini görselleştir."""
        ds = self.get(dataset)
        points = ds.get("points", [])

        if dataset in ["fig15_ac", "fig16_creep"]:
            self._plot_bar_single(ds, points, save_path)
        elif dataset in ["fig17_ht", "fig19_optimized"]:
            self._plot_bar_layers(ds, points, save_path, dataset)
        elif dataset == "fig7_water_temp":
            self._plot_water_temp(ds, save_path)
        elif dataset == "table1_a356":
            self._plot_a356_params(ds, save_path)

    def export_reference_bundle(self, save_path: str) -> Dict:
        """
        Export the real paper reference layers used by the quenching study.

        This creates baseline plots and a manifest JSON so every optimizer run
        carries the same reference package.
        """
        os.makedirs(save_path, exist_ok=True)
        datasets = {
            "fig7_water_temp": self.get("fig7_water_temp"),
            "fig17_ht": self.get("fig17_ht"),
            "fig19_optimized": self.get("fig19_optimized"),
            "fig21_crossmember": self.get("fig21_crossmember"),
        }

        self.plot_baseline("fig7_water_temp", save_path)
        self.plot_baseline("fig17_ht", save_path)
        self.plot_baseline("fig19_optimized", save_path)

        manifest = {
            "datasets": {
                "fig7_water_temp": {
                    "type": "time_series",
                    "time_s": datasets["fig7_water_temp"]["time_s"],
                    "channels": [k for k in datasets["fig7_water_temp"].keys() if k.startswith("Channel_")],
                },
                "fig17_ht": {
                    "type": "distortion_layers",
                    "layers": self.list_layers("fig17_ht"),
                    "points": datasets["fig17_ht"]["points"],
                },
                "fig19_optimized": {
                    "type": "distortion_layers",
                    "layers": self.list_layers("fig19_optimized"),
                    "points": datasets["fig19_optimized"]["points"],
                },
                "fig21_crossmember": {
                    "type": "distortion_layers",
                    "layers": self.list_layers("fig21_crossmember"),
                    "points": datasets["fig21_crossmember"]["points"],
                },
            }
        }

        manifest_path = os.path.join(save_path, "reference_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return {
            "reference_dir": save_path,
            "manifest_path": manifest_path,
            "datasets": manifest["datasets"],
        }

    def plot_comparison(self,
                         pinn_predictions: np.ndarray,
                         dataset: str = "fig17_ht",
                         layer: str = "Bottom",
                         optimizer_name: str = "PINN",
                         save_path: str = "results/") -> str:
        """
        PINN tahmini vs FEM (paper) vs Ölçüm karşılaştırma grafiği.
        3 panel: bar karşılaştırma + hata dağılımı + nokta bazlı hata
        """
        ds      = self.get(dataset)
        points  = ds["points"]
        n       = len(points)
        x_idx   = np.arange(n)

        # Veri anahtarlarını bul
        key_m = f"Measured_{layer}" if f"Measured_{layer}" in ds else "AC_Measured"
        key_s = key_m.replace("Measured", "Simulated")

        measured  = np.array(ds.get(key_m, [0]*n))
        simulated = np.array(ds.get(key_s, [0]*n))

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(
            f"Comparison: {optimizer_name} vs FEM (baseline_paper [2026]) vs Measurement\n"
            f"Dataset: {dataset}  |  Layer: {layer}",
            fontsize=13, fontweight="bold"
        )

        colors = {"Measured": "#212121", "FEM": "#1565C0", "PINN": "#E53935"}
        width = 0.25

        # ── Panel 1: Bar Karşılaştırma ────────────────────
        ax = axes[0, 0]
        ax.bar(x_idx - width, measured,     width, label="Measurement (Paper)",   color=colors["Measured"], alpha=0.85)
        ax.bar(x_idx,         simulated,    width, label="FEM (baseline_paper)", color=colors["FEM"],      alpha=0.85)
        ax.bar(x_idx + width, pinn_predictions, width, label=f"{optimizer_name}", color=colors["PINN"], alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x_idx[::2])
        ax.set_xticklabels(points[::2], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Distortion [mm]")
        ax.set_title("CMM Point Comparison")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

        # ── Panel 2: Nokta Bazlı Hata ────────────────────
        ax = axes[0, 1]
        pinn_err = pinn_predictions - measured
        fem_err  = simulated - measured
        ax.bar(x_idx - width/2, fem_err,  width, label="FEM Error", color=colors["FEM"], alpha=0.8)
        ax.bar(x_idx + width/2, pinn_err, width, label=f"{optimizer_name} Error", color=colors["PINN"], alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x_idx[::2])
        ax.set_xticklabels(points[::2], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Error [mm]  (Pred - Measurement)")
        ax.set_title("Point-wise Error")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

        # ── Panel 3: Scatter (Pred vs Measured) ──────────
        ax = axes[1, 0]
        lim = max(np.abs(measured).max(), np.abs(pinn_predictions).max(),
                  np.abs(simulated).max()) * 1.15
        ax.scatter(measured, simulated,        c=colors["FEM"],  alpha=0.7, s=60, label="FEM")
        ax.scatter(measured, pinn_predictions, c=colors["PINN"], alpha=0.7, s=60, label=optimizer_name, marker="^")
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, label="Perfect")
        ax.set_xlabel("Measurement [mm]"); ax.set_ylabel("Prediction [mm]")
        ax.set_title("Pred vs Measurement (Scatter)")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # ── Panel 4: Metrik Özeti ─────────────────────────
        ax = axes[1, 1]
        ax.axis("off")
        pinn_mae  = np.mean(np.abs(pinn_err))
        pinn_rmse = np.sqrt(np.mean(pinn_err**2))
        fem_mae   = np.mean(np.abs(fem_err))
        fem_rmse  = np.sqrt(np.mean(fem_err**2))

        table_data = [
            ["Metric", "FEM (baseline_paper)", optimizer_name, "Gain"],
            ["MAE [mm]",  f"{fem_mae:.4f}",  f"{pinn_mae:.4f}",
             f"{'↓' if pinn_mae < fem_mae else '↑'}{abs(pinn_mae-fem_mae):.4f}"],
            ["RMSE [mm]", f"{fem_rmse:.4f}", f"{pinn_rmse:.4f}",
             f"{'↓' if pinn_rmse < fem_rmse else '↑'}{abs(pinn_rmse-fem_rmse):.4f}"],
            ["MaxErr [mm]",
             f"{np.max(np.abs(fem_err)):.4f}",
             f"{np.max(np.abs(pinn_err)):.4f}", "─"],
        ]
        col_widths = [0.28, 0.25, 0.25, 0.22]
        # BUG FIX: loc="center" tablo sınırları dışına taşabilir → tablo görünmez.
        # bbox=[left, bottom, width, height] eksen koordinatlarında kesin konumlama.
        tbl = ax.table(cellText=table_data, cellLoc="center",
                       colWidths=col_widths, bbox=[0.0, 0.05, 1.0, 0.88])
        tbl.auto_set_font_size(False); tbl.set_fontsize(10)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#BDBDBD")
            if r == 0:
                cell.set_facecolor("#1565C0")
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 1:
                cell.set_facecolor("#F5F5F5")
            else:
                cell.set_facecolor("#FFFFFF")
        ax.set_title("Comparison Summary", fontweight="bold", fontsize=11, pad=4)

        plt.tight_layout()
        os.makedirs(save_path, exist_ok=True)
        fname = f"{save_path}/{optimizer_name}_{dataset}_{layer}_comparison.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Grafik] {fname} kaydedildi")
        return fname

    def _plot_bar_single(self, ds, points, save_path):
        fig, ax = plt.subplots(figsize=(14, 5))
        n = len(points); x = np.arange(n)
        ax.bar(x - 0.2, ds.get("AC_Measured", ds.get("Measured", [])), 0.35,
               label="Measurement", color="#212121", alpha=0.85)
        ax.bar(x + 0.2, ds.get("Simulated", []), 0.35,
               label="FEM Simulation", color="#1565C0", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(points, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Distortion [mm]"); ax.legend()
        ax.set_title(f"Baseline: {ds['description'][:60]}")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--"); ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/baseline_{ds['source'].replace(' ','_')}.png", dpi=150)
        plt.close()

    def _plot_bar_layers(self, ds, points, save_path, dataset):
        layer_keys = [k for k in ds.keys() if k.startswith("Measured_")]
        n = len(points); x = np.arange(n)
        fig, axes = plt.subplots(len(layer_keys), 1, figsize=(14, 4*len(layer_keys)), squeeze=False)
        colors = plt.cm.tab10(np.linspace(0, 0.8, len(layer_keys)))
        for i, lk in enumerate(layer_keys):
            ax = axes[i][0]
            vals = np.array(ds[lk])
            ax.bar(x, vals, color=colors[i], alpha=0.85, label=lk)
            sim_k = lk.replace("Measured", "Simulated")
            if sim_k in ds:
                ax.bar(x, np.array(ds[sim_k]), 0.4, color="navy", alpha=0.5, label=sim_k)
            ax.set_xticks(x); ax.set_xticklabels(points, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Distortion [mm]"); ax.legend(fontsize=9)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--"); ax.grid(True, alpha=0.3, axis="y")
        axes[0][0].set_title(f"Baseline: {ds['description'][:60]}")
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/baseline_{dataset}.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_water_temp(self, ds, save_path):
        fig, ax = plt.subplots(figsize=(10, 5))
        t = ds["time_s"]
        for k, v in ds.items():
            if k.startswith("Channel"):
                ax.plot(t, v, marker="o", markersize=4, label=k, linewidth=1.5)
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Water Temperature [°C]")
        ax.set_title("Water Tank Temperature (Fig 7, baseline_paper [2026])")
        ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.3)
        ax.set_ylim(58, 97)
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/baseline_fig7_water_temp.png", dpi=150)
        plt.close()

    def _plot_a356_params(self, ds, save_path):
        data = np.array(ds["data"])
        T = data[:, 0]; F = data[:, 1]; n = data[:, 2]; m = data[:, 3]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(T, F, "bo-", linewidth=2); axes[0].set_title("F(T) [MPa]")
        axes[1].plot(T, n, "rs-", linewidth=2); axes[1].set_title("n(T)")
        axes[2].plot(T, m, "g^-", linewidth=2); axes[2].set_title("m(T)")
        for ax in axes:
            ax.set_xlabel("T [°C]"); ax.grid(True, alpha=0.3)
        fig.suptitle("A356 Flow Stress Parameters (Table 1, baseline_paper [2026])", fontweight="bold")
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/baseline_a356_params.png", dpi=150)
        plt.close()

    def save_all_to_json(self, path: str = "results/baseline_data.json"):
        """Tüm baseline veriyi JSON'a kaydet."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.DATASETS, f, indent=2)
        print(f"  Baseline veri kaydedildi: {path}")

    def get_process_param(self, key: str):
        return PROCESS_PARAMS.get(key)


# ─────────────────────────────────────────────────────────────
# Test / Demo
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bl = BaselinePaper()
    bl.summary()

    # Tüm baseline grafiklerini çiz
    os.makedirs("results/baseline", exist_ok=True)
    bl.plot_baseline("fig15_ac",       "results/baseline")
    bl.plot_baseline("fig17_ht",       "results/baseline")
    bl.plot_baseline("fig7_water_temp","results/baseline")
    bl.plot_baseline("table1_a356",    "results/baseline")
    bl.plot_baseline("fig16_creep",    "results/baseline")
    bl.save_all_to_json("results/baseline/baseline_data.json")

    # PINN simülasyonu yapıyormuş gibi demo karşılaştırma
    n = len(FIG17_HT_QUENCH_3LAYER["points"])
    fake_pinn = np.array(FIG17_HT_QUENCH_3LAYER["Measured_Bottom"]) + \
                np.random.normal(0, 0.05, n)   # %5 gürültü = simüle edilmiş PINN
    metrics = bl.compare_with_pinn(fake_pinn, "fig17_ht", layer="Bottom")
    bl.plot_comparison(fake_pinn, "fig17_ht", layer="Bottom",
                       optimizer_name="PINN_NSGA2", save_path="results/baseline")
    print("\nDemo complete.")
