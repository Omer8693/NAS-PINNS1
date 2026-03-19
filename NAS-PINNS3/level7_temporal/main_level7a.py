"""
main_level7a.py — Level 7A: Zaman Atlama Analizi
=================================================
NAS-PINN'in FEM zaman adımlarını atlamasını test eder.

Kullanım:
    cd /home/coder/NAS-PINNS1/NAS-PINNS3
    python level7_temporal/main_level7a.py
    python level7_temporal/main_level7a.py --optimizer bayesian
"""

import argparse
import json
import time
from pathlib import Path

import sys
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from level7_temporal.src.time_skip_analysis import (
    OPTIMIZERS, TIME_SCENARIOS,
    FEM_N_STEPS_FINE, FEM_N_STEPS_COARSE,
    run_analysis,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", default="all",
                   choices=["all"] + OPTIMIZERS)
    return p.parse_args()


def print_summary(results: dict):
    print("\n" + "=" * 70)
    print("  LEVEL 7A — Zaman Atlama Özeti")
    print("  Soru: NAS-PINN, FEM'in adım adım ilerlediği yerde atlar mı?")
    print("=" * 70)

    # FEM referans
    print(f"\n  FEM referans:")
    print(f"    İnce FEM  : {FEM_N_STEPS_FINE:4d} zaman adımı  (Δt = 0.01s)")
    print(f"    Kaba FEM  : {FEM_N_STEPS_COARSE:4d} zaman adımı  (Δt = 0.50s)")
    print(f"    → Sıralı hesap: her adım öncekine bağımlı\n")

    # NAS-PINN sonuçları
    print(f"  NAS-PINN (Level 6) — farklı değerlendirme senaryoları:\n")

    header = f"  {'Senaryo':<20} {'Pts':>4}  "
    for opt in results:
        header += f"{opt.upper():>12} "
    print(header)
    print("  " + "-" * (len(header) - 2))

    for scenario in TIME_SCENARIOS:
        row = f"  {scenario:<20} {len(TIME_SCENARIOS[scenario]):4d}  "
        for opt, res in results.items():
            l2 = res["scenarios"][scenario]["mean_l2"]
            row += f"{l2:12.4f} "
        print(row)

    print(f"\n  {'Dense eval (61 pts)':<20} {'61':>4}  ", end="")
    for opt, res in results.items():
        print(f"{res['mean_l2_dense']:12.4f} ", end="")
    print()

    # Inference süresi
    print(f"\n  NAS-PINN inference süresi (tek t noktası):")
    for opt, res in results.items():
        ms = res.get("fwd_pass_ms", res["fem_comparison"].get("pinn_forward_ms", 0))
        print(f"    {opt.upper():12s}: {ms:.2f} ms")

    print(f"\n  Sonuç: NAS-PINN T(x,y,t)'yi herhangi bir t'de milisaniyeler")
    print(f"  içinde verir — FEM gibi ara adımları hesaplamak zorunda değil.")
    print("=" * 70)

    # Özet JSON kaydet
    out_dir = _ROOT / "level7_temporal" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "question": "NAS-PINN, FEM zaman marching olmadan aynı doğrulukta tahmin yapabilir mi?",
        "fem_reference": {
            "fine_steps":   FEM_N_STEPS_FINE,
            "coarse_steps": FEM_N_STEPS_COARSE,
        },
        "results_per_optimizer": {
            opt: {
                "mean_l2_dense": res["mean_l2_dense"],
                "scenarios": {
                    sc: {
                        "n_pts": len(TIME_SCENARIOS[sc]),
                        "mean_l2": res["scenarios"][sc]["mean_l2"],
                        "inference_ms": res["scenarios"][sc]["inference_ms"],
                    }
                    for sc in TIME_SCENARIOS
                },
                "pinn_forward_ms": res.get("fwd_pass_ms", 0),
            }
            for opt, res in results.items()
        },
    }
    with open(out_dir / "level7a_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Özet kaydedildi: {out_dir / 'level7a_summary.json'}")


def main():
    args = parse_args()
    opts = OPTIMIZERS if args.optimizer == "all" else [args.optimizer]

    print("\n" + "=" * 70)
    print("  Level 7A — NAS-PINN Zaman Atlama Analizi")
    print(f"  Optimizerlar: {opts}")
    print("=" * 70)

    t_total = time.time()
    all_results = {}

    for opt in opts:
        result = run_analysis(opt)
        if result:
            all_results[opt] = result

    if all_results:
        print_summary(all_results)

    print(f"\n  Toplam süre: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
