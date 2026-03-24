"""
run_3d_v2.py — 3D MCO-PINN  (improved training, v2)
=====================================================
Differences from v1:
  n_adam : 800  → 2000
  n_col  : 1500 → 3000
  nx/ny/nz: 20/12/8 → 24/16/10  (denser evaluation grid)
Output folder: results/v2/
Saves:
  results/v2/results_3d_v2.json  — MAE summary
  results/v2/<dom>_<arch>_skip<s>_slice.npy  — z-mid T slices (heat map)
  results/v2/<dom>_<arch>_skip<s>_loss.json  — loss history (all windows)
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from level8_nas_mco_pinn.domains_3d import Rectangular3D, Cylinder3D, StackedCubes3D, LShape3D
from level8_nas_mco_pinn.pinn_3d    import run_skip_3d

# ── Output ────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "level8_nas_mco_pinn", "results", "v2")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helper: Extract geometry metadata ──────────────────────────
def get_geometry_metadata(domain):
    """Extract geometry parameters from domain object."""
    domain_class = type(domain).__name__
    
    if domain_class == "Rectangular3D":
        return {
            "type": "rectangular",
            "params": {
                "Lx": float(domain.Lx),
                "Ly": float(domain.Ly),
                "Lz": float(domain.Lz),
            }
        }
    elif domain_class == "Cylinder3D":
        return {
            "type": "cylinder",
            "params": {
                "R": float(domain.R),
                "H": float(domain.H),
            }
        }
    elif domain_class == "StackedCubes3D":
        return {
            "type": "stacked",
            "params": {
                "L_cube": float(domain.L_cube),
                "N_stack": int(domain.N_stack),
                "Lz": float(domain.Lz),
            }
        }
    elif domain_class == "LShape3D":
        return {
            "type": "lshape",
            "params": {
                "Lx": float(domain.Lx),
                "Ly": float(domain.Ly),
                "Lz": float(domain.Lz),
                "cut_x": float(domain.cut_x),
                "cut_y": float(domain.cut_y),
            }
        }
    else:
        return {"type": "unknown", "params": {}}

# ── Config ────────────────────────────────────────────────────
DOMAINS = {
    "rectangular": Rectangular3D(),
    "cylinder":    Cylinder3D(),
    "stacked":     StackedCubes3D(),
    "lshape":      LShape3D(),
}
ARCHS     = ["bayesian", "nsga2", "nsga3"]
SKIP_VALS = [2, 4]

TRAIN_CFG = dict(
    n_col  = 3000,
    n_bc   = 200,
    n_adam = 2000,
    nx=24, ny=16, nz=10,
)

print("=" * 65)
print("  3D MCO-PINN Skip Operator  —  v2 (improved training)")
print(f"  n_adam={TRAIN_CFG['n_adam']}  n_col={TRAIN_CFG['n_col']}")
print("=" * 65)

summary    = {}
t0_total   = time.time()

for dname, domain in DOMAINS.items():
    summary[dname] = {}
    print(f"\n{'='*65}")
    print(f"  Domain: {domain.name}")
    print(f"{'='*65}")

    for arch in ARCHS:
        print(f"\n  Arch: {arch}")
        res = run_skip_3d(
            domain      = domain,
            arch_name   = arch,
            skip_values = SKIP_VALS,
            verbose     = True,
            **TRAIN_CFG,
        )

        summary[dname][arch] = {}

        for skip, rv in res.items():
            mae  = rv["mae_C"]
            rt   = rv["runtime_s"]
            grid = rv["grid"]
            zi   = grid["zi"]
            k_mid = len(zi) // 2

            summary[dname][arch][str(skip)] = {
                "mae_C":       mae,
                "mae_windows": rv["mae_windows"],
                "runtime_s":   rt,
            }

            # ── Save z-mid slices (FEM + PINN) for each window ──
            slices = {}
            for wi in rv["T_fields"]:
                T_pred = rv["T_fields"][wi][:, :, k_mid]
                T_fem  = rv["T_fem"][wi][:, :, k_mid]
                slices[str(wi)] = {
                    "T_pred": T_pred.tolist(),
                    "T_fem":  T_fem.tolist(),
                }
            # Also save grid axes and geometry metadata
            slice_data = {
                "xi":    grid["xi"].tolist(),
                "yi":    grid["yi"].tolist(),
                "zi":    grid["zi"].tolist(),
                "k_mid": int(k_mid),
                "z_val": float(zi[k_mid]),
                "geometry": get_geometry_metadata(domain),
                "windows": slices,
            }
            slice_path = os.path.join(OUT_DIR,
                f"{dname}_{arch}_skip{skip}_slice.json")
            with open(slice_path, "w") as f:
                json.dump(slice_data, f)

            # ── Save loss history (all windows) ──────────────────
            loss_data = []
            for hist in rv["loss_history"]:
                loss_data.append({
                    "L_total": hist["L_total"],
                    "L_bc":    hist["L_bc"],
                    "L_phys":  hist["L_phys"],
                    "L_ic":    hist["L_ic"],
                })
            loss_path = os.path.join(OUT_DIR,
                f"{dname}_{arch}_skip{skip}_loss.json")
            with open(loss_path, "w") as f:
                json.dump(loss_data, f)

            print(f"    [{dname}] [{arch}] skip={skip}: "
                  f"MAE={mae:.2f}°C  ({rt:.0f}s)")

# ── Save summary JSON ─────────────────────────────────────────
json_path = os.path.join(OUT_DIR, "results_3d_v2.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Summary saved: {json_path}")

# ── Print comparison table ────────────────────────────────────
v1_path = os.path.join(os.path.dirname(__file__), "level8_nas_mco_pinn", "results", "results_3d.json")
v1 = None
if os.path.exists(v1_path):
    with open(v1_path) as f:
        v1 = json.load(f)

print("\n" + "="*65)
print("  COMPARISON: v1 (800 ep) vs v2 (2000 ep) — MAE [°C]")
print("="*65)
hdr = f"{'Domain':12} {'Arch':10} {'skip':5} {'v1':>8} {'v2':>8} {'Δ':>8}"
print(hdr)
print("-"*65)
for dname in DOMAINS:
    for arch in ARCHS:
        for skip in SKIP_VALS:
            v2_mae = summary[dname][arch][str(skip)]["mae_C"]
            v1_mae = v1[dname][arch][str(skip)]["mae_C"] if v1 else float("nan")
            delta  = v2_mae - v1_mae
            sym    = "▼" if delta < 0 else "▲"
            print(f"{dname:12} {arch:10} {skip:5}  "
                  f"{v1_mae:7.2f}  {v2_mae:7.2f}  {sym}{abs(delta):6.2f}")

print(f"\nTotal runtime: {(time.time()-t0_total)/60:.1f} min")
print("Done.")
