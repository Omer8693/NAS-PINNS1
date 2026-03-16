"""
main_level2.py — Level 2 Ana Çalıştırıcı
==========================================
Level 1'in en iyi mimarisini alır, time-stepping operatörüne dönüştürür,
farklı skip değerleri için L2 tablosu üretir.

Kullanım:
    python main_level2.py --skip_values 1 2 4 6 --target_skip 4
    python main_level2.py --mode nas --target_skip 4 --threshold 0.15
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Level 2 modülleri
from src.ts_model    import TimeStepperPINN
from src.ts_evaluate import skip_table
from src.ts_nas      import run_grid_search


def setup_logger(log_path: str) -> logging.Logger:
    # Hem terminale hem dosyaya yaz
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("level2")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    # Terminal handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    # Dosya handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

# ─── Sabitler ──────────────────────────────────────────────────────────────────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_END   = 30.0
N_STEPS = 20

# Level 1 run2 sonuçlarından alınan en iyi mimariler — üç optimizer
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

# Varsayılan eğitim parametreleri
DEFAULT_TRAIN_KWARGS = {
    "n_domain": 1000,
    "n_bc":     200,
    "n_epochs": 500,
    "lr":       1e-3,
    "lr_min":   1e-5,   # cosine annealing: 1e-3 → 1e-5 (Level 5 validated)
}
# ───────────────────────────────────────────────────────────────────────────────


def _make_model(config: dict) -> TimeStepperPINN:
    # Verilen config için bağımsız model oluştur
    return TimeStepperPINN(config).to(DEVICE)


def run_skip_comparison(skip_values: list, save_path: str, log_path: str) -> None:
    """
    Üç Level 1 mimarisi (Bayesian, NSGA2, NSGA3) için skip tablosu üret.
    Her optimizer için ayrı tablo, sonunda birleşik karşılaştırma.
    İlerleme log dosyasına yazılır.
    """
    logger  = setup_logger(log_path)
    t_steps = np.linspace(0.0, T_END, N_STEPS + 1)
    all_results = {}

    logger.info(f"Level 2 skip comparison started — device={DEVICE}")
    logger.info(f"Skip values: {skip_values}  |  Optimizers: {list(LEVEL1_CONFIGS.keys())}")

    for optimizer_name, config in LEVEL1_CONFIGS.items():
        logger.info(f"{'='*55}")
        logger.info(f"Optimizer: {optimizer_name.upper()}  "
                    f"L={config['n_layers']} N={config['neurons'][0]} "
                    f"act={config['activation']}")
        logger.info(f"{'='*55}")

        rows = skip_table(
            model_fn     = lambda cfg=config: _make_model(cfg),
            skip_values  = skip_values,
            t_steps      = t_steps,
            train_kwargs = DEFAULT_TRAIN_KWARGS,
            device       = DEVICE,
            logger       = logger,
        )
        all_results[optimizer_name] = rows

        # Optimizer özeti
        for row in rows:
            logger.info(
                f"  RESULT {optimizer_name:>8} skip={row['skip']} "
                f"steps={row['steps_used']}/{row['steps_total']} "
                f"L2={row['l2_rel']:.4f} MAE={row['mae_C']:.1f}C "
                f"time={row['runtime_s']:.0f}s"
            )

    # Birleşik karşılaştırma tablosu
    logger.info(f"\n{'='*70}")
    logger.info(f"  LEVEL 2 — Three Optimizer Comparison")
    logger.info(f"{'='*70}")
    header = f"{'optimizer':>10} | {'skip':>4} | {'steps':>7} | {'l2_rel':>8} | {'mae_C':>8} | {'time_s':>7}"
    logger.info(header)
    logger.info("-" * len(header))
    for opt_name, rows in all_results.items():
        for row in rows:
            logger.info(
                f"{opt_name:>10} | {row['skip']:>4} | "
                f"{row['steps_used']:>3}/{row['steps_total']:<3} | "
                f"{row['l2_rel']:>8.4f} | {row['mae_C']:>8.1f} | "
                f"{row['runtime_s']:>7.0f}"
            )

    # JSON kaydet
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved: {save_path}")
    logger.info(f"Log  : {log_path}")


def run_nas(target_skip: int, threshold: float, save_path: str) -> None:
    """
    NAS grid search — target_skip için en iyi mimariyi bul.
    """
    t_steps = np.linspace(0.0, T_END, N_STEPS + 1)
    run_grid_search(
        target_skip  = target_skip,
        l2_threshold = threshold,
        t_steps      = t_steps,
        train_kwargs = DEFAULT_TRAIN_KWARGS,
        device       = DEVICE,
        save_path    = save_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Level 2 — Time-Stepping PINN")
    parser.add_argument("--mode", choices=["skip", "nas"], default="skip",
                        help="skip: karsilastirma tablosu | nas: mimari arama")
    parser.add_argument("--skip_values", nargs="+", type=int, default=[1, 2, 4, 6],
                        help="Degerlendirilecek skip degerleri")
    parser.add_argument("--target_skip", type=int, default=4,
                        help="NAS hedef skip miktari")
    parser.add_argument("--threshold",   type=float, default=0.15,
                        help="NAS L2 esigi")
    args = parser.parse_args()

    if args.mode == "skip":
        run_skip_comparison(
            skip_values = args.skip_values,
            save_path   = "results/skip_table.json",
            log_path    = "results/run.log",
        )
    elif args.mode == "nas":
        run_nas(
            target_skip = args.target_skip,
            threshold   = args.threshold,
            save_path   = "results/nas_grid.json",
        )
