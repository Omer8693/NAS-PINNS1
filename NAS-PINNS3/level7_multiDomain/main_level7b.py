"""
main_level7b.py — Level 7B: Multi-Domain Poisson NAS Benchmark
===============================================================
Runs NAS for the Poisson problem on three domains:
  circle   — unit disk,   exact solution available
  lshape   — L-shaped,    no exact solution
  flower   — 5-petal,     no exact solution

Three optimizers: bayesian | nsga2 | nsga3

Usage:
    cd /home/coder/NAS-PINNS1/NAS-PINNS3
    python level7_multiDomain/main_level7b.py
    python level7_multiDomain/main_level7b.py --domain circle --optimizer bayesian
    python level7_multiDomain/main_level7b.py --domain all --optimizer all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from level7_multiDomain.src.nas_search_7b import run_one

DOMAINS    = ["square", "circle", "annulus", "lshape", "flower"]
OPTIMIZERS = ["bayesian", "nsga2", "nsga3"]

OUT_DIR = _ROOT / "level7_multiDomain" / "results"


def parse_args():
    p = argparse.ArgumentParser(description="Level 7B — Multi-Domain Poisson NAS")
    p.add_argument("--domain",    default="all", choices=["all"] + DOMAINS,
                   nargs="+", metavar="DOMAIN")
    p.add_argument("--optimizer", default="all", choices=["all"] + OPTIMIZERS)
    p.add_argument("--n_calls",   type=int, default=20,
                   help="Bayesian: total evaluations")
    p.add_argument("--pop_size",  type=int, default=8,
                   help="NSGA-II/III: population size")
    p.add_argument("--n_gen",     type=int, default=4,
                   help="NSGA-II/III: number of generations")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def print_summary(all_results: dict):
    print("\n" + "=" * 70)
    print("  LEVEL 7B — Multi-Domain Poisson NAS — Summary")
    print("=" * 70)

    header = f"  {'Domain':<10} {'Optimizer':<12} {'Arch':>20}  {'L2':>8}  {'MAE':>8}  {'Params':>8}  {'Time(s)':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for dom in DOMAINS:
        for opt in OPTIMIZERS:
            key = f"{dom}/{opt}"
            if key not in all_results:
                continue
            r = all_results[key]
            arch = r["best_arch"]
            arch_str = f"{arch['n_layers']}×{arch['neurons']} {arch['activation']}"
            exact_flag = " *" if r["has_exact"] else ""
            print(f"  {dom:<10}{exact_flag} {opt:<12} {arch_str:>20}  "
                  f"{r['final_l2']:>8.5f}  {r['final_mae']:>8.5f}  "
                  f"{r['n_params']:>8d}  {r['search_s']+r['train_s']:>8.1f}")

    print("  (* = exact solution available, L2 is true relative error)")
    print("=" * 70)


def main():
    args = parse_args()
    raw = args.domain
    domains    = DOMAINS    if raw == ["all"] or raw == "all" else (raw if isinstance(raw, list) else [raw])
    optimizers = OPTIMIZERS if args.optimizer == "all" else [args.optimizer]

    print("\n" + "=" * 70)
    print("  Level 7B — Multi-Domain Poisson NAS Benchmark")
    print(f"  Domains   : {domains}")
    print(f"  Optimizers: {optimizers}")
    print(f"  Search    : n_calls={args.n_calls}  pop={args.pop_size}  n_gen={args.n_gen}")
    print("=" * 70)

    t_total     = time.time()
    all_results = {}

    for dom in domains:
        for opt in optimizers:
            try:
                result = run_one(
                    domain_name = dom,
                    optimizer   = opt,
                    out_dir     = OUT_DIR,
                    n_calls     = args.n_calls,
                    pop_size    = args.pop_size,
                    n_gen       = args.n_gen,
                    seed        = args.seed,
                )
                all_results[f"{dom}/{opt}"] = result
            except Exception as e:
                print(f"\n  ERROR [{dom}/{opt}]: {e}")
                import traceback; traceback.print_exc()

    # Save summary
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "level7b_summary.json"
    with open(summary_path, "w") as f:
        # Strip large loss histories from summary
        slim = {}
        for k, v in all_results.items():
            slim[k] = {kk: vv for kk, vv in v.items()
                       if kk not in ("loss_history", "val_l2_history")}
        json.dump(slim, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    if all_results:
        print_summary(all_results)

    print(f"\n  Total time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
