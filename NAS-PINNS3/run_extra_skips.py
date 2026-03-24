"""
run_extra_skips.py — 3D MCO-PINN for skip=1 and skip=6
Adds to existing v2 results (skip=2 and skip=4 already done).
Saves to same v2 directory: level8_nas_mco_pinn/results/v2/
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from level8_nas_mco_pinn.domains_3d import Rectangular3D, Cylinder3D, StackedCubes3D
from level8_nas_mco_pinn.pinn_3d    import run_skip_3d

OUT_DIR = os.path.join(os.path.dirname(__file__),
                       "level8_nas_mco_pinn", "results", "v2")
os.makedirs(OUT_DIR, exist_ok=True)

DOMAINS = {
    "rectangular": Rectangular3D(),
    "cylinder":    Cylinder3D(),
    "stacked":     StackedCubes3D(),
}
ARCHS  = ["bayesian", "nsga2", "nsga3"]
SKIPS  = [1, 6]   # only the missing ones

TRAIN_CFG = dict(
    n_col  = 3000,
    n_bc   = 200,
    n_adam = 2000,
    nx=24, ny=16, nz=10,
)

# Load existing v2 results to extend them
v2_json = os.path.join(OUT_DIR, "results_3d_v2.json")
with open(v2_json) as f:
    summary = json.load(f)

t0_total = time.time()
for dom_name, dom in DOMAINS.items():
    print(f"\n  Domain: {dom.name}")
    for arch in ARCHS:
        print(f"  Arch: {arch}")

        # Determine which skips still need to be done
        pending = [s for s in SKIPS
                   if str(s) not in summary.get(dom_name, {}).get(arch, {})]
        if not pending:
            print(f"    [{dom_name}] [{arch}]: all skips done, skipping")
            continue

        t0 = time.time()
        # run_skip_3d handles multiple skip values in one call
        rv_all = run_skip_3d(dom, arch_name=arch, skip_values=pending,
                             verbose=True, **TRAIN_CFG)
        elapsed_total = time.time() - t0

        for skip in pending:
            rv = rv_all[skip]
            mae_C   = float(rv["mae_C"])
            elapsed = float(rv["runtime_s"])
            print(f"    [{dom_name}] [{arch}] skip={skip}: MAE={mae_C:.2f}°C  ({elapsed:.0f}s)")

            # Save to summary
            summary.setdefault(dom_name, {}).setdefault(arch, {})[str(skip)] = {
                "mae_C":       mae_C,
                "mae_windows": [float(x) for x in rv["mae_windows"]],
                "runtime_s":   elapsed,
            }

            # Save z-mid slice
            grid  = rv["grid"]
            zi    = grid["zi"]
            k_mid = len(zi) // 2
            wins_out = {}
            for wi in range(len(rv["T_fields"])):
                T_pred = rv["T_fields"][wi][:, :, k_mid]
                T_fem  = rv["T_fem"][wi][:, :, k_mid]
                wins_out[str(wi)] = {
                    "T_pred": T_pred.tolist(),
                    "T_fem":  T_fem.tolist(),
                }
            slice_data = {
                "xi": grid["xi"].tolist(), "yi": grid["yi"].tolist(),
                "zi": grid["zi"].tolist(), "k_mid": int(k_mid),
                "z_val": float(zi[k_mid]), "windows": wins_out,
            }
            with open(os.path.join(OUT_DIR,
                      f"{dom_name}_{arch}_skip{skip}_slice.json"), "w") as f:
                json.dump(slice_data, f)

            # Save loss history
            with open(os.path.join(OUT_DIR,
                      f"{dom_name}_{arch}_skip{skip}_loss.json"), "w") as f:
                json.dump(rv["loss_history"], f)

        # Save updated summary after each arch
        with open(v2_json, "w") as f:
            json.dump(summary, f, indent=2)

print(f"\n  Total extra runtime: {(time.time()-t0_total)/60:.1f} min")
print("  Done. All results saved to", OUT_DIR)
