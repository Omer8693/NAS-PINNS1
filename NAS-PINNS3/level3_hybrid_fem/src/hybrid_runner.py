"""
hybrid_runner.py — FEM + PINN Dönüşümlü Çalıştırıcı
======================================================
FEM anchor adımlarında çalışır, PINN aralarını doldurur.
Her pencerede önce model eğitilir, sonra adaptif karar verilir:
  - PDE rezidüsü küçük → PINN ile atla
  - PDE rezidüsü büyük → FEM anchor kullan

Çıkış geçmişi (history):
  [{"t", "source", "residual", "T_mean"}, ...]
"""

import sys
import time
from pathlib import Path
from typing import Callable, List, Dict

import numpy as np
import torch

from src.fem_interface import FEMCheckpoint
from src.adaptive_skip import AdaptiveSkipScheduler

# Level 2 trainer'ı import et — Level 3 kendi penceresinde eğitir
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "level2_timestepper"))
from src.ts_trainer import train_window


def run_hybrid(model,
               config:     dict,
               t_steps:    np.ndarray,
               scheduler:  AdaptiveSkipScheduler,
               fem_solver: Callable,
               train_kwargs: dict,
               n_query:    int = 300,
               device:     torch.device = torch.device("cpu"),
               logger=None) -> tuple:
    """
    Hibrit FEM + PINN zaman ilerleme döngüsü.

    Her pencerede:
      1. Model pencere için eğitilir (train_window)
      2. PDE rezidüsü hesaplanır
      3. Adaptif karar: FEM mi PINN mi?

    Döndürür: (final_checkpoint, history, scheduler_summary)
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # t=0: IC her zaman FEM/analitikten
    current  = fem_solver(t_steps[0])
    history: List[Dict] = [{"t": float(t_steps[0]), "source": "fem_ic",
                             "residual": 0.0, "T_mean": float(current.T_field.mean())}]
    t0_wall = time.time()

    for i in range(1, len(t_steps)):
        t_now  = float(t_steps[i])
        t_prev = float(t_steps[i - 1])
        dt     = t_now - t_prev

        # Sorgu noktaları — iç alan
        x = torch.rand(n_query, 1, device=device) * 1.3
        y = torch.rand(n_query, 1, device=device) * 0.6

        # Önceki checkpoint'ten T_prev al
        T_prev_vals = current.interpolate_at(x.squeeze(), y.squeeze()).unsqueeze(1)
        T_prev_mean = float(T_prev_vals.mean().item())

        # 1. Bu pencereyi eğit
        T_prev_fn = lambda _x, _y, _Tp=T_prev_vals, _x0=x, _y0=y: (
            current.interpolate_at(_x.squeeze(), _y.squeeze()).unsqueeze(1)
        )
        model, _ = train_window(
            model     = model,
            T_prev_fn = T_prev_fn,
            t_start   = t_prev,
            t_end     = t_now,
            device    = device,
            **train_kwargs,
        )

        # 2. Adaptif karar
        t_local = torch.full((n_query, 1), dt, device=device)
        use_fem  = scheduler.decide(model, x, y, t_local, T_prev_vals, dt, t_now=t_now)
        residual = scheduler.history[-1]["residual"]

        if use_fem:
            current = fem_solver(t_now)
            source  = "fem"
        else:
            with torch.no_grad():
                T_next = model(x, y, t_local, T_prev_vals, dt)
            current = FEMCheckpoint(t_now, T_next.squeeze(), x.squeeze(), y.squeeze())
            source  = "pinn"

        T_mean = float(current.T_field.mean().item())
        history.append({"t": t_now, "source": source,
                        "residual": residual, "T_mean": T_mean})
        log(f"  t={t_now:5.1f}s | {source:4s} | residual={residual:.4f} | T_mean={T_mean:.1f}C")

    elapsed = time.time() - t0_wall
    summary = scheduler.summary()
    summary["wall_time_s"] = elapsed
    log(f"Hybrid done: {elapsed:.1f}s | FEM={summary['fem_steps']} "
        f"PINN={summary['pinn_steps']} skip_rate={summary['skip_rate_pct']}%")

    return current, history, summary


def history_to_array(history: List[Dict]) -> dict:
    """
    Geçmişi analiz için serileştirilebilir dict'e dönüştür.
    PLAN.md Grafik 1 (T vs t) için veri hazırlar.
    """
    return {
        "times":     [h["t"]        for h in history],
        "sources":   [h["source"]   for h in history],
        "residuals": [h["residual"] for h in history],
        "T_means":   [h["T_mean"]   for h in history],
    }
