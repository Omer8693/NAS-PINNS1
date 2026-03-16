"""
ts_evaluate.py — Skip Değerlendirme
=====================================
Farklı skip değerleri için time-stepping PINN'i değerlendirir.
Level 1 tablosuyla doğrudan karşılaştırılabilir çıktı üretir.

skip=1 → hiç atlamama (baseline, Level 1 ile aynı)
skip=4 → her 4 adımda bir anchor, 3 adımı PINN ile atla
"""

from typing import Callable, List, Dict

import numpy as np
import torch

from src.ts_model   import TimeStepperPINN
from src.ts_trainer import train_window


def _make_analytical_fn(t: float, device: torch.device) -> Callable:
    """
    Analitik referans fonksiyonu — 2D Robin BC temel mod soğuma.
    Level 1 ile aynı fizik: α = 1.75e-3 /s
    T(t) = T_water + (T_init - T_water) · exp(-α·t)
    """
    alpha, T_water, T_init = 1.75e-3, 20.0, 540.0
    T_val = T_water + (T_init - T_water) * np.exp(-alpha * t)

    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Analitik çözüm uzaydan bağımsız (0D soğuma modeli)
        return torch.full_like(x, T_val)
    return fn


def evaluate_skip(model: TimeStepperPINN,
                  skip: int,
                  t_steps: np.ndarray,
                  train_kwargs: dict,
                  device: torch.device) -> Dict:
    """
    Belirli bir skip değeri için L2 hatası hesapla.

    Yöntem:
      - Anchor adımları: t_steps[::skip]
      - Her anchor çiftinde model yeniden eğitilir (pencere bazlı)
      - Çıkış noktalarında L2 hatası hesaplanır

    Döndürür: {skip, steps_used, steps_total, l2_rel, mae_C, runtime_s}
    """
    anchor_times = t_steps[::skip]
    n_eval       = 500
    errors_l2    = []
    errors_mae   = []
    t0           = __import__("time").time()

    T_prev_fn = _make_analytical_fn(anchor_times[0], device)

    for i in range(1, len(anchor_times)):
        t_start = float(anchor_times[i - 1])
        t_end   = float(anchor_times[i])
        dt      = t_end - t_start

        # Pencere için eğit
        model, _ = train_window(
            model      = model,
            T_prev_fn  = T_prev_fn,
            t_start    = t_start,
            t_end      = t_end,
            device     = device,
            **train_kwargs,
        )

        # Değerlendirme noktaları — iç alan merkezi
        x = torch.rand(n_eval, 1, device=device) * 1.3
        y = torch.rand(n_eval, 1, device=device) * 0.6
        T_prev_vals = T_prev_fn(x, y)
        t_local_end = torch.full((n_eval, 1), dt, device=device)

        with torch.no_grad():
            T_pred = model(x, y, t_local_end, T_prev_vals, dt)

        # Referans: analitik çözüm
        T_ref_fn = _make_analytical_fn(t_end, device)
        T_ref    = T_ref_fn(x, y)

        l2  = float((torch.norm(T_pred - T_ref) / torch.norm(T_ref - 20.0)).item())
        mae = float(torch.mean(torch.abs(T_pred - T_ref)).item())
        errors_l2.append(l2)
        errors_mae.append(mae)

        # Bir sonraki pencere için T_prev güncelle
        T_prev_fn = _make_analytical_fn(t_end, device)

    runtime = __import__("time").time() - t0
    return {
        "skip":        skip,
        "steps_used":  len(anchor_times),
        "steps_total": len(t_steps),
        "l2_rel":      float(np.mean(errors_l2)),
        "mae_C":       float(np.mean(errors_mae)),
        "runtime_s":   runtime,
    }


def skip_table(model_fn: Callable,
               skip_values: List[int],
               t_steps: np.ndarray,
               train_kwargs: dict,
               device: torch.device,
               logger=None) -> List[Dict]:
    """
    Farklı skip değerleri için karşılaştırma tablosu üret.
    model_fn(): her skip için taze model döndürür.
    logger: varsa log dosyasına da yazar.
    """
    rows = []
    for skip in skip_values:
        model = model_fn()   # her skip için bağımsız model
        row   = evaluate_skip(model, skip, t_steps, train_kwargs, device)
        rows.append(row)
        msg = (f"  skip={skip:2d} | steps={row['steps_used']:2d}/{row['steps_total']:2d} "
               f"| L2={row['l2_rel']:.4f} | MAE={row['mae_C']:.1f}C "
               f"| {row['runtime_s']:.0f}s")
        if logger:
            logger.info(msg)
        else:
            print(msg)
    return rows
