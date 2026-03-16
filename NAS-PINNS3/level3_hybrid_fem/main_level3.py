"""
main_level3.py — Level 3 Ana Çalıştırıcı
==========================================
Üç Level 1 mimarisi (Bayesian, NSGA2, NSGA3) için hibrit FEM + PINN döngüsü.
Her optimizer: FEM anchor + PINN arası + adaptif atlama + distortion zinciri.

Log : results/run.log
JSON: results/level3_results.json

Kullanım:
    python main_level3.py
    python main_level3.py --threshold 0.1 --max_skip 4
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Level 3 modülleri
from src.fem_interface import FEMCheckpoint
from src.hybrid_runner import run_hybrid, history_to_array
from src.adaptive_skip import AdaptiveSkipScheduler
from src.mechanical    import estimate_distortion_at_cmm, compare_with_paper, CMM_POINT_LABELS

# Level 2 modeli import
sys.path.insert(0, str(Path(__file__).parent.parent / "level2_timestepper"))
from src.ts_model import TimeStepperPINN

# ─── Sabitler ──────────────────────────────────────────────────────────────────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_END   = 30.0
N_STEPS = 20

# Level 1 run2 sonuçlarından alınan üç mimari
LEVEL1_CONFIGS = {
    "bayesian": {           # L2_rel=0.076, MAE=39°C — en iyi
        "n_layers":   5,
        "neurons":    [151, 151, 151, 151, 151],
        "activation": "relu",
    },
    "nsga2": {              # L2_rel=0.252, MAE=133°C
        "n_layers":   3,
        "neurons":    [153, 153, 153],
        "activation": "tanh",
    },
    "nsga3": {              # L2_rel=0.513, MAE=270°C — en küçük model
        "n_layers":   3,
        "neurons":    [75, 75, 75],
        "activation": "tanh",
    },
}

# Pencere eğitim parametreleri — Level 2 ile aynı
TRAIN_KWARGS = {
    "n_domain": 500,
    "n_bc":     100,
    "n_epochs": 300,
    "lr":       1e-3,
    "lr_min":   1e-5,   # cosine annealing: 1e-3 → 1e-5 (Level 5 validated)
}
# ───────────────────────────────────────────────────────────────────────────────


def setup_logger(log_path: str) -> logging.Logger:
    # Hem terminale hem dosyaya yaz — Level 2 ile aynı format
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("level3")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def fem_solver(t: float) -> FEMCheckpoint:
    # FEM anchor: şimdilik analitik referans (α=1.75e-3 /s)
    return FEMCheckpoint.from_analytical(t, n_grid=50, device=DEVICE)


def run_all_optimizers(threshold: float = 0.1,
                       max_skip:  int   = 4,
                       log_path:  str   = "results/run.log",
                       save_path: str   = "results/level3_results.json") -> None:
    """
    Üç optimizer için hibrit pipeline çalıştır.
    Her biri için:
      1. Model oluştur
      2. FEM + PINN hibrit döngüsü (adaptif atlama)
      3. Distortion tahmin et
      4. Sonuçları kaydet
    """
    logger  = setup_logger(log_path)
    t_steps = np.linspace(0.0, T_END, N_STEPS + 1)
    all_results = {}

    logger.info(f"Level 3 hybrid pipeline started — device={DEVICE}")
    logger.info(f"threshold={threshold}  max_skip={max_skip}  n_steps={N_STEPS}")
    logger.info(f"Optimizers: {list(LEVEL1_CONFIGS.keys())}")

    for opt_name, config in LEVEL1_CONFIGS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimizer : {opt_name.upper()}")
        logger.info(f"Config    : L={config['n_layers']} N={config['neurons'][0]} "
                    f"act={config['activation']}")
        logger.info(f"{'='*60}")

        # Taze model
        model     = TimeStepperPINN(config).to(DEVICE)
        scheduler = AdaptiveSkipScheduler(threshold=threshold, max_skip=max_skip)

        # Hibrit döngü — her pencerede eğitim + karar
        final_chk, history, run_summary = run_hybrid(
            model        = model,
            config       = config,
            t_steps      = t_steps,
            scheduler    = scheduler,
            fem_solver   = fem_solver,
            train_kwargs = TRAIN_KWARGS,
            device       = DEVICE,
            logger       = logger,
        )

        # Distortion tahmin et
        distortions_pinn = estimate_distortion_at_cmm(
            T_field = final_chk.T_field,
            x_grid  = final_chk.x_grid,
            y_grid  = final_chk.y_grid,
        )

        # Paper karşılaştırma — placeholder (paper verisi eklenince güncelle)
        placeholder = np.zeros_like(distortions_pinn)
        comparison  = compare_with_paper(distortions_pinn, placeholder, placeholder)

        all_results[opt_name] = {
            "config":         config,
            "run_summary":    run_summary,
            "history":        history_to_array(history),
            "distortion_cmm": comparison,
        }

        # Optimizer özeti
        logger.info(f"  RESULT {opt_name:>8} | "
                    f"FEM={run_summary['fem_steps']} "
                    f"PINN={run_summary['pinn_steps']} "
                    f"skip_rate={run_summary['skip_rate_pct']}% "
                    f"time={run_summary['wall_time_s']:.1f}s | "
                    f"T_final={float(final_chk.T_field.mean()):.1f}C")

    # Birleşik karşılaştırma tablosu
    logger.info(f"\n{'='*70}")
    logger.info("  LEVEL 3 — Three Optimizer Comparison")
    logger.info(f"{'='*70}")
    header = f"{'optimizer':>10} | {'fem_steps':>9} | {'pinn_steps':>10} | {'skip_%':>6} | {'T_final_C':>9} | {'time_s':>7}"
    logger.info(header)
    logger.info("-" * len(header))
    for opt_name, res in all_results.items():
        s = res["run_summary"]
        T_final = res["history"]["T_means"][-1]
        logger.info(f"{opt_name:>10} | {s['fem_steps']:>9} | {s['pinn_steps']:>10} | "
                    f"{s['skip_rate_pct']:>6} | {T_final:>9.1f} | {s['wall_time_s']:>7.1f}")

    # JSON kaydet
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved: {save_path}")
    logger.info(f"Log  : {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Level 3 — Hybrid FEM + PINN")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Adaptif atlama PDE rezidu esigi")
    parser.add_argument("--max_skip",  type=int,   default=4,
                        help="Arka arkaya maksimum PINN adim sayisi")
    parser.add_argument("--out_dir",   type=str,   default=None,
                        help="Sonuc klasoru (varsayilan: results/thr{threshold}_skip{max_skip})")
    args = parser.parse_args()

    # Klasör adı belirtilmemişse parametrelerden otomatik üret
    out_dir = args.out_dir or f"results/thr{args.threshold}_skip{args.max_skip}"

    run_all_optimizers(
        threshold = args.threshold,
        max_skip  = args.max_skip,
        log_path  = f"{out_dir}/run.log",
        save_path = f"{out_dir}/level3_results.json",
    )
