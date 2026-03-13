"""
Architecture Search — NAS-PINN'den Evrimsel Optimizasyona Geçiş
================================================================
Kaynak: Wang & Zhong (2023) NAS-PINN — DARTS tabanlı mimari arama.

Bu projede DARTS'ın YERİNE kullanılan yöntemler:
  1. NSGA-II  — hız + doğruluk + model boyutu, çok amaçlı Pareto cephesi
  2. NSGA-III — referans noktalı çok amaçlı, 3+ amaç için NSGA-II'den üstün
  3. Bayesian Optimization — Gaussian Process, az değerlendirme ile etkin arama

Amaç fonksiyonları (3 boyutlu Pareto):
  f1: Validation L2 hatası    → minimize
  f2: Eğitim süresi [s]       → minimize
  f3: Parametre sayısı        → minimize (kompaktlık)

Karar değişkenleri (mimari):
  n_layers   : [1, 7]         — gizli katman sayısı
  neurons[i] : [16, 256]      — her katmanın nöron sayısı (bağımsız)
  activation : {tanh, sin, swish}
  residual   : {True, False}
"""

import torch
import numpy as np
import time
import json
import os
from typing import List, Dict, Callable, Optional
from copy import deepcopy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Opsiyonel bağımlılıklar — kurulu değilse uyarı ver ─────
try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.optimize import minimize as pymoo_minimize
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("[WARNING] pymoo not found. Install: pip install pymoo")

try:
    from skopt import gp_minimize
    from skopt.space import Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("[WARNING] scikit-optimize not found. Install: pip install scikit-optimize")


# ─────────────────────────────────────────────────────────────
# Bölüm 1 — Kodlama / Çözme Fonksiyonları
# ─────────────────────────────────────────────────────────────

def decode_x_to_config(x:          np.ndarray,
                        max_layers: int = 7,
                        n_input:    int = 3,
                        n_output:   int = 1) -> dict:
    """
    Optimizasyon vektörünü (sürekli) mimari konfigürasyonuna çevir.

    Vektör biçimi:
      x[0]          : n_layers  ∈ [1, max_layers]
      x[1..max_l]   : neurons[i] ∈ [16, 256]
      x[max_l+1]    : act_idx ∈ {0:tanh, 1:sin, 2:swish}
      x[max_l+2]    : residual ∈ {0, 1}

    NSGA ve Bayesian Optimization aynı vektör biçimini kullanır —
    bu sayede karşılaştırma tutarlı olur.
    """
    n_layers = int(np.clip(round(x[0]), 1, max_layers))
    neurons  = [int(np.clip(round(x[1 + i]), 16, 256)) for i in range(n_layers)]
    act_idx  = int(np.clip(round(x[1 + max_layers]), 0, 2))
    act_map  = {0: "tanh", 1: "sin", 2: "swish"}
    residual = bool(round(x[2 + max_layers]) > 0.5)

    return {
        "n_input":    n_input,
        "n_output":   n_output,
        "n_layers":   n_layers,
        "neurons":    neurons,
        "activation": act_map[act_idx],
        "residual":   residual,
    }


# ─────────────────────────────────────────────────────────────
# Bölüm 2 — Mimari Değerlendirme (Arama Sırasında Hızlı Eğitim)
# ─────────────────────────────────────────────────────────────

def evaluate_architecture(config:           dict,
                           train_fn:         Callable,
                           n_epochs_search:  int = 300) -> Dict:
    """
    Verilen mimariyi hızlı eğit (300 epoch) ve değerlendirme metriklerini döndür.

    train_fn: (config, n_epochs) → (l2_error, n_params)
    Hata durumunda kötü metrik döndürür (ceza değerleri) — arama çökmez.
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
            "l2_error":   1.0,
            "train_time": 9999.0,
            "n_params":   9999.0,
            "config":     config,
        }


# ─────────────────────────────────────────────────────────────
# Bölüm 3 — pymoo Problem Tanımı (NSGA-II ve NSGA-III için ortak)
# ─────────────────────────────────────────────────────────────

class PINNArchProblem(ElementwiseProblem):
    """
    NSGA-II / NSGA-III için 3-amaçlı sürekli optimizasyon problemi.

    Her birey bir mimari konfigürasyona karşılık gelir.
    Değerlendirme sonuçları önbelleklenir — aynı mimari iki kez eğitilmez.
    """

    def __init__(self, train_fn: Callable, max_layers: int = 7, n_epochs: int = 300):
        self.train_fn   = train_fn
        self.max_layers = max_layers
        self.n_epochs   = n_epochs
        self.eval_cache = {}   # tekrar eden değerlendirmeleri önlemek için

        # Karar değişkeni sınırları: [n_layers, n0..n6, act_idx, residual]
        n_var = 1 + max_layers + 2
        xl = np.array([1]         + [16]  * max_layers + [0, 0], dtype=float)
        xu = np.array([max_layers] + [256] * max_layers + [2, 1], dtype=float)

        super().__init__(n_var=n_var, n_obj=3, n_ieq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        config = decode_x_to_config(x, self.max_layers)
        key    = json.dumps(config, sort_keys=True)

        # Önbellek kontrolü
        if key in self.eval_cache:
            res = self.eval_cache[key]
        else:
            res = evaluate_architecture(config, self.train_fn, self.n_epochs)
            self.eval_cache[key] = res

        # 3 amaç: L2 hatası, eğitim süresi, parametre sayısı (normalleştirilmiş)
        out["F"] = [
            res["l2_error"],
            res["train_time"],
            res["n_params"] / 1e4,
        ]


# ─────────────────────────────────────────────────────────────
# Bölüm 4 — NSGA-II
# ─────────────────────────────────────────────────────────────

def run_nsga2(train_fn:   Callable,
              pop_size:   int = 20,
              n_gen:      int = 30,
              max_layers: int = 7,
              n_epochs:   int = 300,
              seed:       int = 42) -> List[dict]:
    """
    NSGA-II ile PINN mimari arama.

    Operatörler:
      SBX crossover (prob=0.9, eta=15) + PM mutation (eta=20) + RoundingRepair
    Döndürür: L2 hatasına göre sıralanmış Pareto optimal mimari listesi.
    """
    if not PYMOO_AVAILABLE:
        raise ImportError("pymoo required: pip install pymoo")

    print(f"\n{'='*52}")
    print(f"  NSGA-II Architecture Search")
    print(f"  Population: {pop_size}  |  Generations: {n_gen}  |  Max Layers: {max_layers}")
    print(f"{'='*52}")

    problem   = PINNArchProblem(train_fn, max_layers, n_epochs)
    algorithm = NSGA2(
        pop_size          = pop_size,
        sampling          = FloatRandomSampling(),
        crossover         = SBX(prob=0.9, eta=15, repair=RoundingRepair()),
        mutation          = PM(eta=20,    repair=RoundingRepair()),
        eliminate_duplicates = True,
    )

    t0     = time.time()
    result = pymoo_minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=True)
    elapsed = time.time() - t0

    print(f"\n  NSGA-II done: {elapsed:.1f}s  |  Pareto front: {len(result.X)} solutions")

    # Pareto konfigürasyonlarını L2'ye göre sırala
    pareto = []
    for x, f in zip(result.X, result.F):
        cfg = decode_x_to_config(x, max_layers)
        cfg["objectives"] = {"l2": float(f[0]), "time_s": float(f[1]),
                              "n_params": float(f[2] * 1e4)}
        pareto.append(cfg)
    pareto.sort(key=lambda c: c["objectives"]["l2"])
    return pareto


# ─────────────────────────────────────────────────────────────
# Bölüm 5 — NSGA-III
# ─────────────────────────────────────────────────────────────

def run_nsga3(train_fn:   Callable,
              pop_size:   int = 20,
              n_gen:      int = 30,
              max_layers: int = 7,
              n_epochs:   int = 300,
              seed:       int = 42) -> List[dict]:
    """
    NSGA-III ile PINN mimari arama.

    3 amaç için Das-Dennis referans noktaları (n_partitions=4 → 15 nokta).
    3 veya daha fazla amaçta NSGA-II'ye göre daha dengeli Pareto dağılımı sağlar.
    """
    if not PYMOO_AVAILABLE:
        raise ImportError("pymoo required: pip install pymoo")

    print(f"\n{'='*52}")
    print(f"  NSGA-III Architecture Search")
    print(f"  Population: {pop_size}  |  Generations: {n_gen}")
    print(f"{'='*52}")

    # Das-Dennis referans yönleri: 3 amaç, n_partitions=4 → C(4+3-1,3-1)=15 nokta
    ref_dirs  = get_reference_directions("das-dennis", 3, n_partitions=4)

    problem   = PINNArchProblem(train_fn, max_layers, n_epochs)
    algorithm = NSGA3(
        ref_dirs             = ref_dirs,
        pop_size             = pop_size,
        sampling             = FloatRandomSampling(),
        crossover            = SBX(prob=0.9, eta=15, repair=RoundingRepair()),
        mutation             = PM(eta=20,    repair=RoundingRepair()),
        eliminate_duplicates = True,
    )

    t0     = time.time()
    result = pymoo_minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=True)
    elapsed = time.time() - t0

    print(f"\n  NSGA-III done: {elapsed:.1f}s  |  Pareto front: {len(result.X)} solutions")

    pareto = []
    for x, f in zip(result.X, result.F):
        cfg = decode_x_to_config(x, max_layers)
        cfg["objectives"] = {"l2": float(f[0]), "time_s": float(f[1]),
                              "n_params": float(f[2] * 1e4)}
        pareto.append(cfg)
    pareto.sort(key=lambda c: c["objectives"]["l2"])
    return pareto


# ─────────────────────────────────────────────────────────────
# Bölüm 6 — Bayesian Optimization
# ─────────────────────────────────────────────────────────────

def run_bayesian(train_fn:   Callable,
                 n_calls:    int = 30,
                 n_initial:  int = 10,
                 max_layers: int = 7,
                 n_epochs:   int = 300,
                 seed:       int = 42) -> dict:
    """
    Gaussian Process tabanlı Bayesian Optimization.
    scikit-optimize kütüphanesini kullanır.

    Avantaj: NSGA'ya kıyasla çok daha az fonksiyon değerlendirmesiyle iyi sonuç.
    Tek amaç (L2 hatası) minimize edilir — hız ve parametre sayısı kısıt değil.

    n_initial: rasgele başlangıç noktaları (GP için prior oluşturur)
    n_calls:   toplam değerlendirme sayısı (n_initial dahil)
    """
    if not SKOPT_AVAILABLE:
        raise ImportError("scikit-optimize required: pip install scikit-optimize")

    print(f"\n{'='*52}")
    print(f"  Bayesian Optimization Architecture Search")
    print(f"  n_calls: {n_calls}  |  n_initial: {n_initial}")
    print(f"{'='*52}")

    # Arama uzayı: tamsayı değişkenler
    space = (
        [Integer(1, max_layers, name="n_layers")] +
        [Integer(16, 256, name=f"n{i}") for i in range(max_layers)] +
        [Integer(0, 2, name="act_idx"),
         Integer(0, 1, name="residual")]
    )

    best = {"l2": float("inf"), "config": None}

    def objective(x_list):
        config  = decode_x_to_config(np.array(x_list, dtype=float), max_layers)
        res     = evaluate_architecture(config, train_fn, n_epochs)
        if res["l2_error"] < best["l2"]:
            best["l2"]     = res["l2_error"]
            best["config"] = deepcopy(config)
            print(f"  → New best: L2={res['l2_error']:.6f}  arch={config['neurons']}")
        return res["l2_error"]

    t0     = time.time()
    result = gp_minimize(
        func             = objective,
        dimensions       = space,
        n_calls          = n_calls,
        n_initial_points = n_initial,
        random_state     = seed,
        verbose          = False,
    )
    elapsed = time.time() - t0

    best_config = decode_x_to_config(np.array(result.x, dtype=float), max_layers)
    print(f"\n  Bayesian done: {elapsed:.1f}s  |  Best L2: {result.fun:.6f}")
    print(f"  Best arch: {best_config['neurons']}  act={best_config['activation']}")
    return best_config
