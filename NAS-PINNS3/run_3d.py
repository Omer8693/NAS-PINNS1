"""
run_3d.py — 3D MCO-PINN Full Runner
=====================================
cd /home/coder/NAS-PINNS1/NAS-PINNS3
python -u run_3d.py 2>&1 | tee results/run_3d_log.txt
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from level8_nas_mco_pinn.domains_3d import (
    Rectangular3D, Cylinder3D, StackedCubes3D, LPrism3D,
)
from level8_nas_mco_pinn.pinn_3d import run_skip_3d
from level8_nas_mco_pinn.viz_3d  import run_all_figures

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
DOMAINS = {
    "rectangular": Rectangular3D(),
    "cylinder":    Cylinder3D(),
    "stacked":     StackedCubes3D(),
}
ARCHS      = ["bayesian", "nsga2", "nsga3"]
SKIP_VALS  = [2, 4]          # skip=1 is very slow, skip=6 can also be added

# Training parameters (speed/accuracy trade-off)
TRAIN_CFG = dict(
    n_col  = 1500,   # interior collocation points
    n_bc   = 150,    # BC points (per face)
    n_adam = 800,    # number of epochs
    nx=20, ny=12, nz=8,  # visualization grid
)

# ──────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────
print("=" * 65)
print("  3D MCO-PINN Skip Operator")
print("  Domains: Rectangular, Cylinder, Stacked Cubes")
print("  Arch: Bayesian, NSGA-II, NSGA-III")
print(f"  Skip: {SKIP_VALS}")
print("=" * 65)

all_results = {}   # {domain: {arch: {skip: result}}}
summary     = {}

t0_total = time.time()

for dname, domain in DOMAINS.items():
    all_results[dname] = {}
    summary[dname]     = {}
    print(f"\n{'='*65}")
    print(f"  Domain: {domain.name}")
    print(f"{'='*65}")

    for arch in ARCHS:
        print(f"\n  Arch: {arch}")
        res = run_skip_3d(
            domain     = domain,
            arch_name  = arch,
            skip_values= SKIP_VALS,
            verbose    = True,
            **TRAIN_CFG,
        )
        all_results[dname][arch] = res
        summary[dname][arch]     = {
            str(s): {"mae_C": v["mae_C"], "runtime_s": v["runtime_s"]}
            for s, v in res.items()
        }

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SUMMARY — 3D MCO-PINN Results")
print("="*65)
for dname in DOMAINS:
    print(f"\n  [{dname}]")
    for arch in ARCHS:
        for s in SKIP_VALS:
            mae = summary[dname][arch][str(s)]["mae_C"]
            rt  = summary[dname][arch][str(s)]["runtime_s"]
            print(f"    {arch:10s} skip={s}: MAE={mae:.2f}°C  t={rt:.0f}s")

# Save JSON
out_dir  = os.path.join(os.path.dirname(__file__), "results")
json_path= os.path.join(out_dir, "results_3d.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Results saved: {json_path}")

# ──────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────
print("\nGenerating visualizations...")
run_all_figures(all_results)

print(f"\nTotal runtime: {(time.time()-t0_total)/60:.1f} minutes")
print("Completed.")
