"""
time_skip_analysis.py — Level 7A
=================================
NAS-PINN vs FEM Zaman Atlama Analizi.

Soru: "FEM'in adım adım ilerlediği yerde, NAS-PINN t=0'dan
      doğrudan t=30s'ye atlayabilir mi — ve ne kadar doğru?"

Referans: T_ref(t) = T_water + ΔT·exp(−α·t)   (lumped kapasitans)
Bu, model eğitiminde kullanılan analitik yaklaşımla aynıdır
(problems/quenching.py → analytical_solution).

FEM ile karşılaştırma:
  FEM  : T(t+Δt) = f(T(t)) — sıralı bağımlılık, global sparse sistem çözümü
  PINN : T(x,y,t) = NN(x,y,t) — tek fonksiyon, herhangi t'de ms-düzeyinde eval
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(_ROOT))

from src.config import (
    DEVICE, T_INIT, T_WATER_INIT, T_END, X_MAX, Y_MAX, ALPHA
)
from src.pinn_network import PINNNet

# ── Dizin sabitleri ────────────────────────────────────────────────────────────
LEVEL1_DIR = _ROOT / "level1_single_shot" / "results"
LEVEL6_DIR = _ROOT / "level6_poisson_benchmark" / "results"
OPTIMIZERS = ["bayesian", "nsga2", "nsga3"]

# ── FEM referans sabitleri ─────────────────────────────────────────────────────
# Mortensen (2026): implicit FEM, A356 quenching, 30s simülasyon
FEM_DT_FINE   = 0.01   # s — ince FEM adımı (~3000 adım, gerçek implementasyon)
FEM_DT_COARSE = 0.30   # s — kaba FEM adımı (~100 adım, hızlandırılmış)
FEM_N_STEPS_FINE   = int(round(T_END / FEM_DT_FINE))    # 3000
FEM_N_STEPS_COARSE = int(round(T_END / FEM_DT_COARSE))  # 100

# ── Değerlendirme senaryoları ──────────────────────────────────────────────────
# FEM zorunlu olarak BÜTÜN t_i ≤ t_query noktalarını çözmek zorundadır.
# NAS-PINN: SADECE bu noktalarda forward pass, aralar atlanır.
TIME_SCENARIOS: Dict[str, List[float]] = {
    "ultra_coarse (3 pts)":  [0.0, 10.0, 30.0],
    "coarse (5 pts)":        [0.0, 5.0, 10.0, 20.0, 30.0],
    "medium (8 pts)":        [0.0, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0],
    "L1 train equiv (20 pts)": list(np.linspace(0.0, T_END, 20)),
    "fine (61 pts)":         list(np.linspace(0.0, T_END, 61)),
}

# Uzamsal eval grid
NX_EVAL, NY_EVAL = 50, 25   # 1250 nokta / zaman dilimi


# ── Yardımcı fonksiyonlar ──────────────────────────────────────────────────────

def T_ref(t_arr: np.ndarray) -> np.ndarray:
    """
    Lumped kapasitans analitik çözümü — eğitimde kullanılan referans.
      T(t) = T_water + (T_init − T_water)·exp(−α·t)
    Girdi t_arr: [N], Çıktı: [N]
    """
    return T_WATER_INIT + (T_INIT - T_WATER_INIT) * np.exp(-ALPHA * t_arr)


def load_model(opt: str) -> Optional[PINNNet]:
    """Level 6 → Level 5 → Level 1 öncelik sırası ile model yükle."""
    arch_path = LEVEL1_DIR / opt / "best_arch.json"
    if not arch_path.exists():
        return None
    cfg = json.load(open(arch_path))
    cfg = cfg.get("config") or cfg.get("best_config") or cfg

    candidates = [
        LEVEL6_DIR / opt / "model_l6.pt",
        _ROOT / "level5_refinement" / "results" / opt / "model_lbfgs.pt",
        LEVEL1_DIR / opt / "model.pt",
    ]
    weights = next((p for p in candidates if p.exists()), None)
    if weights is None:
        return None

    model = PINNNet(
        n_input      = cfg.get("n_input",  3),
        n_output     = cfg.get("n_output", 1),
        hidden_sizes = cfg["neurons"],
        activation   = cfg["activation"],
        residual     = cfg.get("residual", False),
    ).to(DEVICE)
    model.load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def eval_grid(model: PINNNet, t_val: float) -> np.ndarray:
    """
    T(x,y,t_val) → [NX_EVAL, NY_EVAL] numpy array.
    Girdi sırası: (t, x, y) — quenching model standardı.
    """
    xs = torch.linspace(0.0, X_MAX, NX_EVAL, device=DEVICE)
    ys = torch.linspace(0.0, Y_MAX, NY_EVAL, device=DEVICE)
    xg, yg = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([
        torch.full_like(xg, t_val).reshape(-1),
        xg.reshape(-1),
        yg.reshape(-1),
    ], dim=1)
    with torch.no_grad():
        T = model(coords).squeeze(-1).cpu().numpy()
    return T.reshape(NX_EVAL, NY_EVAL)


def l2_at_t(model: PINNNet, t_val: float, n_pts: int = 2000) -> float:
    """
    Eğitimle tutarlı L2 metriği: rastgele (x,y) noktalarında
    PINN çıktısı vs T_ref(t).
    """
    t_col = torch.full((n_pts, 1), t_val, device=DEVICE)
    x_col = torch.rand(n_pts, 1, device=DEVICE) * X_MAX
    y_col = torch.rand(n_pts, 1, device=DEVICE) * Y_MAX
    coords = torch.cat([t_col, x_col, y_col], dim=1)

    with torch.no_grad():
        T_pinn = model(coords).squeeze(-1).cpu().numpy()

    T_ana = float(T_ref(np.array([t_val])))  # skaler
    diff  = T_pinn - T_ana
    return float(np.sqrt(np.mean(diff**2)) / (abs(T_ana) + 1e-8))


def run_analysis(opt: str) -> Optional[Dict]:
    """Tek optimizer için tam zaman-atlama analizi."""
    model = load_model(opt)
    if model is None:
        print(f"  [{opt}] Model bulunamadı, atlanıyor.")
        return None

    print(f"\n  [{opt}] Analiz başlıyor...")

    # ── 1. L2(t) eğrisi — 61 zaman noktasında
    t_dense = np.linspace(0.0, T_END, 61)
    l2_curve = []
    T_avg_pinn_dense = []
    T_avg_ref_dense  = []

    t0_eval = time.perf_counter()
    for t_val in t_dense:
        grid = eval_grid(model, float(t_val))
        l2   = l2_at_t(model, float(t_val))
        l2_curve.append(l2)
        T_avg_pinn_dense.append(float(grid.mean()))
        T_avg_ref_dense.append(float(T_ref(np.array([t_val]))))
    t_dense_eval = time.perf_counter() - t0_eval
    print(f"    Dense L2 eğrisi: 61 nokta, {t_dense_eval:.2f}s")
    print(f"    L2 aralığı: [{min(l2_curve):.4f}, {max(l2_curve):.4f}]  "
          f"ort={np.mean(l2_curve):.4f}")

    # ── 2. Her senaryo için metrikleri hesapla
    scenario_results = {}
    for name, t_pts in TIME_SCENARIOS.items():
        t_arr = np.array(t_pts)

        t0 = time.perf_counter()
        l2s   = [l2_at_t(model, float(t)) for t in t_arr]
        grids = [eval_grid(model, float(t)) for t in t_arr]
        t_inf = time.perf_counter() - t0

        T_avg_pinn = [float(g.mean()) for g in grids]
        T_avg_ref_ = [float(T_ref(np.array([t]))) for t in t_arr]

        # Ortalama sıcaklık farkı
        T_err = [abs(p - r) for p, r in zip(T_avg_pinn, T_avg_ref_)]

        scenario_results[name] = {
            "n_pts":       int(len(t_arr)),
            "t_values":    t_arr.tolist(),
            "l2_per_t":    [round(x, 6) for x in l2s],
            "mean_l2":     float(np.mean(l2s)),
            "T_avg_pinn":  [round(x, 3) for x in T_avg_pinn],
            "T_avg_ref":   [round(x, 3) for x in T_avg_ref_],
            "T_avg_err_C": [round(x, 3) for x in T_err],
            "mean_T_err_C": float(np.mean(T_err)),
            "inference_ms": round(t_inf * 1000, 2),
        }

        print(f"    {name:<28}: L2={np.mean(l2s):.4f}  "
              f"ΔT={np.mean(T_err):.1f}°C  "
              f"t={t_inf*1000:.1f}ms")

    # ── 3. Tek forward-pass zamanı
    t0 = time.perf_counter()
    for _ in range(20):
        eval_grid(model, 15.0)
    t_fwd_ms = (time.perf_counter() - t0) / 20 * 1000

    # ── 4. Isı haritaları (4 zaman dilimi: t=1,5,15,30s)
    heatmap_times = [1.0, 5.0, 15.0, 30.0]
    heatmap_grids = {}
    for t_val in heatmap_times:
        heatmap_grids[str(t_val)] = eval_grid(model, t_val).tolist()

    # ── 5. Kaydet
    result = {
        "optimizer":        opt,
        "t_dense":          t_dense.tolist(),
        "l2_curve":         [round(x, 6) for x in l2_curve],
        "T_avg_pinn_dense": [round(x, 3) for x in T_avg_pinn_dense],
        "T_avg_ref_dense":  [round(x, 3) for x in T_avg_ref_dense],
        "scenarios":        scenario_results,
        "heatmap_times":    heatmap_times,
        "heatmap_grids":    heatmap_grids,
        "mean_l2_dense":    round(float(np.mean(l2_curve)), 6),
        "fwd_pass_ms":      round(t_fwd_ms, 3),
        "fem_comparison": {
            "fem_fine_steps":   FEM_N_STEPS_FINE,
            "fem_coarse_steps": FEM_N_STEPS_COARSE,
            "note": (
                f"FEM (implicit): {FEM_N_STEPS_FINE} adım × global sparse çözüm. "
                f"NAS-PINN: herhangi t'de {t_fwd_ms:.1f}ms forward pass."
            ),
        },
    }

    out_dir = _ROOT / "results" / "level7" / opt
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "time_skip_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"    Forward-pass: {t_fwd_ms:.2f}ms/nokta")
    print(f"    Kaydedildi: {out_dir / 'time_skip_result.json'}")
    return result
