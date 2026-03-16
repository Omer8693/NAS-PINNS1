"""
main_distortion.py — Level 4 Distortion Hesaplama Ana Çalıştırıcı
==================================================================
Level 3 hybrid FEM+PINN sonuçlarını alarak 2D düzlem gerilme FEM ile
CMM noktası distortion hesaplar ve Fig17 ile karşılaştırır.

Boru Hattı:
  1. Level 3 results JSON → T_mean (her optimizer için)
  2. Analitik 2D T(x,y) profili üret (gradient / cosine / parabolic)
  3. PlaneStressFEM.solve() → u(x,y) displacement alanı
  4. CMM noktalarında |δ| = sqrt(u_x² + u_y²) [mm]
  5. Karşılaştırma tablosu + JSON kaydet

Kullanım:
    python main_distortion.py
    python main_distortion.py --level3_json results/my_run/level3_results.json
    python main_distortion.py --nx 30 --ny 15 --profile gradient
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Level 4 modülleri
from src.plane_stress_fem import PlaneStressFEM, BETA_THERMAL, T_REF
from src.thermal_field    import analytical_profile, field_stats
from src.cmm_points       import (CMM_LABELS, CMM_COORDS,
                                   PAPER_FEM_MM, PAPER_MEASURED_MM)

# Varsayılan Level 3 sonuç dosyası
DEFAULT_L3_JSON = str(
    Path(__file__).parent.parent / "level3_hybrid_fem" / "results" / "level3_results.json"
)


# ── Logger kurulumu ────────────────────────────────────────────────────────────

def setup_logger(log_path: str) -> logging.Logger:
    """Hem terminale hem dosyaya yazan logger — Level 2/3 ile aynı format."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("level4")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ── Ana hesaplama ──────────────────────────────────────────────────────────────

def run_distortion(level3_json: str,
                   nx:      int   = 20,
                   ny:      int   = 10,
                   Lx:      float = 1.3,
                   Ly:      float = 0.6,
                   profile: str   = "gradient",
                   out_dir: str   = "results",
                   logger:  logging.Logger = None) -> dict:
    """
    Level 3 JSON'dan distortion hesapla.

    Döndürür: optimizer → distortion sonuç sözlüğü
    """
    # Level 3 sonuçlarını oku
    if not Path(level3_json).exists():
        raise FileNotFoundError(f"Level 3 JSON not found: {level3_json}")

    with open(level3_json) as f:
        l3_data = json.load(f)

    logger.info("Level 4 Distortion Pipeline")
    logger.info(f"Level 3 JSON : {level3_json}")
    logger.info(f"FEM mesh     : {nx}x{ny} Q4 elements  ({(nx+1)*(ny+1)} nodes, {2*(nx+1)*(ny+1)} DOF)")
    logger.info(f"Domain       : {Lx}m x {Ly}m")
    logger.info(f"T profile    : {profile}  (gradient_frac=0.15)")
    logger.info(f"Optimizers   : {list(l3_data.keys())}")
    logger.info("")

    # FEM nesnesini bir kez oluştur — tüm optimizer'lar aynı mesh kullanır
    t0 = time.perf_counter()
    logger.info("Assembling stiffness matrix ...")
    fem = PlaneStressFEM(nx=nx, ny=ny, Lx=Lx, Ly=Ly)
    logger.info(f"  K matrix ready — size={fem.n_dof}x{fem.n_dof}  "
                f"({time.perf_counter()-t0:.1f}s)")
    logger.info("")

    results = {}

    for opt_name, l3_res in l3_data.items():
        logger.info(f"{'='*60}")
        logger.info(f"Optimizer : {opt_name.upper()}")
        logger.info(f"{'='*60}")

        t_opt = time.perf_counter()

        # Level 3'ten T_mean alınır — son zaman adımı
        T_means = l3_res["history"]["T_means"]
        T_final = float(T_means[-1])
        logger.info(f"  T_mean(final) = {T_final:.1f} degC")

        # 2D uzaysal sıcaklık alanı oluştur
        T_field = analytical_profile(fem.x_nodes, fem.y_nodes, T_final, profile)
        stats   = field_stats(T_field, label=opt_name)
        logger.info(f"  T_field : min={stats['T_min']:.1f}C  "
                    f"max={stats['T_max']:.1f}C  "
                    f"deltaT={stats['delta_T']:.1f}C")

        # FEM displacement çözümü
        logger.info("  Solving FEM (K*u = f_th) ...")
        t_fem = time.perf_counter()
        try:
            u = fem.solve(T_field)
            fem_time = time.perf_counter() - t_fem
            logger.info(f"  CG solver done — {fem_time:.2f}s")
        except RuntimeError as e:
            logger.error(f"  FEM error: {e}")
            results[opt_name] = {"error": str(e)}
            continue

        # Displacement büyüklükleri
        u_mag = np.sqrt(u[:, 0]**2 + u[:, 1]**2)
        logger.info(f"  max |u| = {u_mag.max()*1000:.4f} mm")

        # CMM noktalarında interpolasyon
        u_query, delta_mm = fem.displacement_at_points(
            u,
            CMM_COORDS[:, 0],
            CMM_COORDS[:, 1],
        )

        logger.info(f"  CMM max delta = {delta_mm.max():.4f} mm  "
                    f"mean delta = {delta_mm.mean():.4f} mm")

        # Paper karşılaştırması
        mae_vs_fem  = float(np.mean(np.abs(delta_mm - PAPER_FEM_MM)))
        mae_vs_meas = float(np.mean(np.abs(delta_mm - PAPER_MEASURED_MM)))

        total_time = time.perf_counter() - t_opt
        logger.info(f"  RESULT {opt_name:>8} | "
                    f"max_delta={delta_mm.max():.4f}mm  "
                    f"MAE_FEM={mae_vs_fem:.4f}mm  "
                    f"time={total_time:.1f}s")

        results[opt_name] = {
            "T_final_mean":    T_final,
            "T_field_stats":   stats,
            "u_max_mm":        float(u_mag.max() * 1000.0),
            "delta_mean_mm":   float(delta_mm.mean()),
            "delta_max_mm":    float(delta_mm.max()),
            "mae_vs_fem_mm":   mae_vs_fem,
            "mae_vs_meas_mm":  mae_vs_meas,
            "per_point": [
                {
                    "label":         CMM_LABELS[i],
                    "x_m":           float(CMM_COORDS[i, 0]),
                    "y_m":           float(CMM_COORDS[i, 1]),
                    "u_x_mm":        float(u_query[i, 0] * 1000.0),
                    "u_y_mm":        float(u_query[i, 1] * 1000.0),
                    "delta_mm":      float(delta_mm[i]),
                    "paper_fem_mm":  float(PAPER_FEM_MM[i]),
                    "paper_meas_mm": float(PAPER_MEASURED_MM[i]),
                }
                for i in range(len(CMM_LABELS))
            ],
        }

    # Karşılaştırma tablosu
    logger.info("")
    logger.info(f"{'='*70}")
    logger.info("  LEVEL 4 — Distortion Comparison Table")
    logger.info(f"{'='*70}")
    header = (f"{'optimizer':>10} | {'T_final_C':>9} | "
              f"{'max_d_mm':>8} | {'mean_d_mm':>9} | {'MAE_FEM':>8} | {'MAE_meas':>9}")
    logger.info(header)
    logger.info("-" * len(header))
    for opt_name, res in results.items():
        if "error" in res:
            logger.info(f"{opt_name:>10} | ERROR: {res['error']}")
            continue
        logger.info(
            f"{opt_name:>10} | {res['T_final_mean']:>9.1f} | "
            f"{res['delta_max_mm']:>8.4f} | {res['delta_mean_mm']:>9.4f} | "
            f"{res['mae_vs_fem_mm']:>8.4f} | {res['mae_vs_meas_mm']:>9.4f}"
        )

    # JSON kaydet
    out_path = Path(out_dir) / "level4_distortion.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_path}")

    return results


# ── Giriş noktası ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Level 4 — 2D Plane Stress FEM Distortion"
    )
    parser.add_argument(
        "--level3_json", type=str, default=DEFAULT_L3_JSON,
        help="Level 3 results JSON file"
    )
    parser.add_argument("--nx",      type=int,   default=20,
                        help="Number of elements in x direction (default: 20)")
    parser.add_argument("--ny",      type=int,   default=10,
                        help="Number of elements in y direction (default: 10)")
    parser.add_argument("--Lx",      type=float, default=1.3,
                        help="Domain width [m] (default: 1.3)")
    parser.add_argument("--Ly",      type=float, default=0.6,
                        help="Domain height [m] (default: 0.6)")
    parser.add_argument("--profile", type=str,   default="gradient",
                        choices=["gradient", "cosine", "parabolic", "uniform"],
                        help="2D temperature profile type (default: gradient)")
    parser.add_argument("--out_dir", type=str,   default="results",
                        help="Output directory (default: results)")
    args = parser.parse_args()

    log_path = str(Path(args.out_dir) / "run.log")
    logger   = setup_logger(log_path)

    run_distortion(
        level3_json = args.level3_json,
        nx          = args.nx,
        ny          = args.ny,
        Lx          = args.Lx,
        Ly          = args.Ly,
        profile     = args.profile,
        out_dir     = args.out_dir,
        logger      = logger,
    )

    logger.info(f"Log  : {log_path}")
