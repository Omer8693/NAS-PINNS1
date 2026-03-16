"""
arch_search.py — Ortak NAS Altyapısı
======================================
Tüm optimizasyon yöntemleri (NSGA-II, NSGA-III, Bayesian) tarafından
paylaşılan temel bileşenler:
  - decode_x_to_config : optimizasyon vektörü → mimari sözlüğü
  - evaluate_architecture: hızlı proxy eğitimi ve sonuç döndürme
  - PINNArchProblem    : pymoo problem sınıfı (NSGA-II ve III için ortak)

Her optimizer kendi dosyasındadır:
  src/opt_nsga2.py    → NSGA-II
  src/opt_nsga3.py    → NSGA-III
  src/opt_bayesian.py → Bayesian Optimization

NAS-PINNS1 referansı:
  Arama uzayı: layers [3,6], neurons [64,160]  (cfg.py)
  Proxy epochs: 300
  Fail penalty: 1e30
"""

import time
from typing import Callable, Dict

import numpy as np

from src.config import LAYERS_MIN, LAYERS_MAX, NEURONS_MIN, NEURONS_MAX

# pymoo ElementwiseProblem — sadece NSGA için gerekli
try:
    from pymoo.core.problem import ElementwiseProblem
    PYMOO_AVAILABLE = True
except ImportError:
    ElementwiseProblem = object   # tip ipucu için yer tutucu
    PYMOO_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Bölüm 1 — Kodlama / Çözme (NAS-PINNS1: 2 karar değişkeni)
# ─────────────────────────────────────────────────────────────

_ACTIVATIONS = ["tanh", "sin", "swish", "relu", "gelu"]  # indeks 0-4

def decode_x_to_config(x: np.ndarray,
                        n_input:  int = 3,
                        n_output: int = 1) -> dict:
    """
    3-değişkenli optimizasyon vektörünü mimari konfigürasyonuna çevir.

    Arama uzayı:
      x[0] : layers     ∈ [LAYERS_MIN,  LAYERS_MAX]   — gizli katman sayısı
      x[1] : neurons    ∈ [NEURONS_MIN, NEURONS_MAX]  — her katmanda aynı nöron
      x[2] : activation ∈ [0, 4]                       — aktivasyon fonksiyonu
               0=tanh, 1=sin, 2=swish, 3=relu, 4=gelu

    Not: aktivasyon PINN performansını büyük ölçüde etkiler;
         sin aktivasyon bu quenching problemi için tanh'dan ~2-4× daha iyi sonuç verir.
    """
    layers  = int(np.clip(round(float(x[0])), LAYERS_MIN, LAYERS_MAX))
    neurons = int(np.clip(round(float(x[1])), NEURONS_MIN, NEURONS_MAX))
    act_idx = int(np.clip(round(float(x[2])), 0, len(_ACTIVATIONS) - 1)) if len(x) > 2 else 0
    activation = _ACTIVATIONS[act_idx]
    return {
        "n_input":    n_input,
        "n_output":   n_output,
        "n_layers":   layers,
        "neurons":    [neurons] * layers,
        "activation": activation,
        "residual":   False,
    }


# ─────────────────────────────────────────────────────────────
# Bölüm 2 — Proxy Değerlendirme (NAS-PINNS1: proxy_epochs=300)
# ─────────────────────────────────────────────────────────────

def evaluate_architecture(config:          dict,
                           train_fn:        Callable,
                           n_epochs_search: int = 300) -> Dict:
    """
    Mimariyi hızlı proxy eğitimiyle değerlendir.

    NAS-PINNS1 proxy_fail_penalty=1e30 — hata durumunda büyük ceza.
    train_fn: (config, n_epochs) → (l2_error, n_params)
    """
    try:
        t0 = time.time()
        l2_err, n_params = train_fn(config, n_epochs_search)
        elapsed = time.time() - t0
        return {
            "l2_error":   float(l2_err),
            "train_time": float(elapsed),
            "n_params":   float(n_params),
            "config":     config,
        }
    except Exception as e:
        print(f"  [EvalError] {e}")
        return {
            "l2_error":   1e30,   # NAS-PINNS1: proxy_fail_penalty
            "train_time": 9999.0,
            "n_params":   9999.0,
            "config":     config,
        }


# ─────────────────────────────────────────────────────────────
# Bölüm 3 — pymoo Problem Tanımı (NSGA-II ve NSGA-III için ortak)
# ─────────────────────────────────────────────────────────────

class PINNArchProblem(ElementwiseProblem):
    """
    NSGA-II / NSGA-III için 2-değişkenli, 2-amaçlı problem.

    NAS-PINNS1 ile hizalanmış:
      - 2 karar değişkeni: [layers, neurons]
      - 2 amaç: L2 hatası + parametre sayısı
      - Önbellekleme: aynı (L,N) çifti iki kez eğitilmez
    """

    def __init__(self,
                 train_fn:       Callable,
                 n_epochs:       int = 300,
                 n_input:        int = 3,
                 n_output:       int = 1,
                 objective_mode: str = "balanced"):
        self.train_fn       = train_fn
        self.n_epochs       = n_epochs
        self.eval_cache     = {}   # "L{l}_N{n}" → result dict
        self.n_input        = n_input
        self.n_output       = n_output
        self.objective_mode = objective_mode

        xl = np.array([LAYERS_MIN,  NEURONS_MIN, 0.0], dtype=float)
        xu = np.array([LAYERS_MAX, NEURONS_MAX, 4.0], dtype=float)
        super().__init__(n_var=3, n_obj=2, n_ieq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        layers  = int(np.clip(round(float(x[0])), LAYERS_MIN, LAYERS_MAX))
        neurons = int(np.clip(round(float(x[1])), NEURONS_MIN, NEURONS_MAX))
        act_idx = int(np.clip(round(float(x[2])), 0, 4))
        key = f"L{layers}_N{neurons}_A{act_idx}"   # aktivasyon dahil önbellekleme

        if key in self.eval_cache:
            res = self.eval_cache[key]
        else:
            config = decode_x_to_config(x, self.n_input, self.n_output)
            res = evaluate_architecture(config, self.train_fn, self.n_epochs)
            self.eval_cache[key] = res

        if self.objective_mode == "l2_only":
            out["F"] = [res["l2_error"], 0.0]
        else:
            # NAS-PINNS1: [proxy_loss, param_count]
            out["F"] = [res["l2_error"], res["n_params"] / 1e4]
