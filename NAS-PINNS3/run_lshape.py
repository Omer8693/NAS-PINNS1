"""
run_lshape.py — 3D MCO-PINN training for LShape3D domain
Trains all skip values (1,2,4,6) × 3 archs, saves to v2/ alongside other domains.
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from level8_nas_mco_pinn.domains_3d import LShape3D
from level8_nas_mco_pinn.pinn_3d    import run_skip_3d

OUT_DIR = os.path.join(os.path.dirname(__file__),
                       "level8_nas_mco_pinn", "results", "v2")
os.makedirs(OUT_DIR, exist_ok=True)

ARCHS  = ["bayesian", "nsga2", "nsga3"]
SKIPS  = [1, 2, 4, 6]

TRAIN_CFG = dict(
    n_col  = 3000,
    n_bc   = 200,
    n_adam = 2000,
    nx=20, ny=20, nz=10,
)

# Load or create summary
v2_json = os.path.join(OUT_DIR, "results_3d_v2.json")
if os.path.exists(v2_json):
    with open(v2_json) as f:
        summary = json.load(f)
else:
    summary = {}

print("  Initialising LShape3D FD solver…")
dom = LShape3D()
dom_name = "lshape"

t0_total = time.time()
for arch in ARCHS:
    print(f"\n  Arch: {arch}")

    pending = [s for s in SKIPS
               if str(s) not in summary.get(dom_name, {}).get(arch, {})]
    if not pending:
        print(f"    [lshape] [{arch}]: all skips done, skipping")
        continue

    t0 = time.time()
    rv_all = run_skip_3d(dom, arch_name=arch, skip_values=pending,
                         verbose=True, **TRAIN_CFG)

    for skip in pending:
        rv = rv_all[skip]
        mae_C   = float(rv["mae_C"])
        elapsed = float(rv["runtime_s"])
        print(f"    [lshape] [{arch}] skip={skip}: MAE={mae_C:.2f}°C  ({elapsed:.0f}s)")

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
            "zi": zi.tolist(), "k_mid": int(k_mid),
            "z_val": float(zi[k_mid]), "windows": wins_out,
        }
        with open(os.path.join(OUT_DIR,
                  f"{dom_name}_{arch}_skip{skip}_slice.json"), "w") as f:
            json.dump(slice_data, f)

        with open(os.path.join(OUT_DIR,
                  f"{dom_name}_{arch}_skip{skip}_loss.json"), "w") as f:
            json.dump(rv["loss_history"], f)

    # Save updated summary after each arch
    with open(v2_json, "w") as f:
        json.dump(summary, f, indent=2)

print(f"\n  Total LShape runtime: {(time.time()-t0_total)/60:.1f} min")
print("  Done. Results saved to", OUT_DIR)
