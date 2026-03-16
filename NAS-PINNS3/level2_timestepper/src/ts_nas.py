"""
ts_nas.py — Time-Stepping PINN için NAS
=========================================
Hedef: minimum parametre sayısıyla maksimum skip değeri sağlayan
mimariyi bul.
Kısıt: L2_rel(skip=k) < threshold

Level 1 NAS altyapısıyla aynı arama uzayı:
  layers  ∈ [3, 6]
  neurons ∈ [64, 160]
  activation ∈ {tanh, sin, swish, relu, gelu}
"""

import json
import time
from typing import List, Dict, Callable

import numpy as np
import torch

from src.ts_model    import TimeStepperPINN
from src.ts_evaluate import skip_table

# Arama uzayı — Level 1 ile aynı sınırlar
LAYERS_MIN  = 3
LAYERS_MAX  = 6
NEURONS_MIN = 64
NEURONS_MAX = 160
ACTIVATIONS = ["tanh", "sin", "swish", "relu", "gelu"]


def _count_params(config: dict) -> int:
    # Mimari parametre sayısını hesapla (n_input=4 sabit)
    n_input = 4
    sizes   = [n_input] + config["neurons"] + [1]
    total   = sum(sizes[i] * sizes[i + 1] + sizes[i + 1]
                  for i in range(len(sizes) - 1))
    return total


def _config_from_x(x: np.ndarray) -> dict:
    """
    Optimizasyon vektörünü mimari konfigürasyonuna çevir.
    x[0]: layers     ∈ [3, 6]
    x[1]: neurons    ∈ [64, 160]
    x[2]: act_idx    ∈ [0, 4]
    """
    layers  = int(np.clip(round(float(x[0])), LAYERS_MIN, LAYERS_MAX))
    neurons = int(np.clip(round(float(x[1])), NEURONS_MIN, NEURONS_MAX))
    act_idx = int(np.clip(round(float(x[2])), 0, len(ACTIVATIONS) - 1))
    return {
        "n_layers":   layers,
        "neurons":    [neurons] * layers,
        "activation": ACTIVATIONS[act_idx],
    }


def evaluate_config_for_skip(config: dict,
                              target_skip: int,
                              t_steps: np.ndarray,
                              train_kwargs: dict,
                              device: torch.device) -> Dict:
    """
    Belirli bir konfigürasyonun target_skip değerindeki L2'sini ölç.
    Önbellekleme: aynı (L,N,act) çifti iki kez eğitilmez.
    """
    from src.ts_evaluate import evaluate_skip
    model = TimeStepperPINN(config).to(device)
    result = evaluate_skip(model, target_skip, t_steps, train_kwargs, device)
    result["n_params"] = _count_params(config)
    result["config"]   = config
    return result


def run_grid_search(target_skip: int = 4,
                    l2_threshold: float = 0.15,
                    t_steps: np.ndarray = None,
                    train_kwargs: dict = None,
                    device: torch.device = torch.device("cpu"),
                    save_path: str = "results/nas_grid.json") -> Dict:
    """
    Grid search NAS — Level 1'in Bayesian search'üne hazırlık.
    Tüm konfigürasyonları dener, threshold altındakileri kaydeder.
    TODO: pymoo / bayes_opt ile değiştir — Level 1 altyapısına bağla.
    """
    if t_steps is None:
        t_steps = np.linspace(0.0, 30.0, 21)
    if train_kwargs is None:
        train_kwargs = {"n_domain": 500, "n_bc": 100, "n_epochs": 300, "lr": 1e-3}

    # Temsili grid — tam arama için adım sayısını artır
    candidates = []
    for n_layers in range(LAYERS_MIN, LAYERS_MAX + 1):
        for neurons in [64, 96, 128, 151, 160]:
            for act in ["tanh", "sin", "relu"]:
                candidates.append({
                    "n_layers":   n_layers,
                    "neurons":    [neurons] * n_layers,
                    "activation": act,
                })

    print(f"\nNAS grid search — target_skip={target_skip}, "
          f"l2_threshold={l2_threshold}, {len(candidates)} candidates")

    results  = []
    feasible = []   # threshold altındaki adaylar
    t0 = time.time()

    for i, config in enumerate(candidates):
        res = evaluate_config_for_skip(config, target_skip, t_steps, train_kwargs, device)
        results.append(res)

        status = "OK" if res["l2_rel"] < l2_threshold else "--"
        print(f"  [{i+1:3d}/{len(candidates)}] L={config['n_layers']} "
              f"N={config['neurons'][0]} act={config['activation']:5s} "
              f"| L2={res['l2_rel']:.4f} | params={res['n_params']:6d} {status}")

        if res["l2_rel"] < l2_threshold:
            feasible.append(res)

    elapsed = time.time() - t0

    # En az parametreli geçerli mimariyi seç
    best = min(feasible, key=lambda r: r["n_params"]) if feasible else None

    output = {
        "target_skip":   target_skip,
        "l2_threshold":  l2_threshold,
        "total_time_s":  elapsed,
        "n_candidates":  len(candidates),
        "n_feasible":    len(feasible),
        "best":          best,
        "all_results":   results,
    }

    with open(save_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {save_path}")
    if best:
        print(f"Best: L={best['config']['n_layers']} N={best['config']['neurons'][0]} "
              f"act={best['config']['activation']} | L2={best['l2_rel']:.4f} "
              f"| params={best['n_params']}")
    return output
