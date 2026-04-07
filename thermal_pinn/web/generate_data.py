"""
thermal_pinn/web/generate_data.py
===================================
Pre-generate JSON data files and copy key images for the NAS-PINN website.

Run from NAS-PINNS1/:
    python -m thermal_pinn.web.generate_data [--dim 0|2|3]

Output:
    web/data/manifest.json
    web/data/2d_{domain}_{arch}_k{k}.json
    web/data/3d_{domain}_{arch}_k{k}.json
    web/img/*.png
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent        # thermal_pinn/
sys.path.insert(0, str(ROOT.parent))                 # NAS-PINNS1/

from thermal_pinn.plots.plot_results import (
    CKPT_DIR, RESULT_DIR, DEVICE,
    eval_grid_2d, eval_grid_3d_mid,
)
from thermal_pinn.physics.domains_2d import make_domain
from thermal_pinn.physics.domains_3d import make_domain_3d
from thermal_pinn.network.pinn import ThermalPINN
from thermal_pinn.plots.plot_thesis import DOMAINS_2D_LIST, DOMAINS_3D_LIST

WEB_DIR  = ROOT / "web"
DATA_DIR = WEB_DIR / "data"
IMG_DIR  = WEB_DIR / "img"
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

ARCHS  = ["bayesian", "nsga2", "nsga3"]
TIMES  = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]
N_GRID = 60


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_model(domain, arch, k, dim, suffix=""):
    path = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}{suffix}.pt"
    if not path.exists():
        return None
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    model = ThermalPINN(dim=dim, arch=arch).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _clean(arr):
    """Numpy 2-D array → nested list with NaN replaced by None."""
    out = []
    for row in arr:
        out.append([None if not np.isfinite(v) else round(float(v), 3) for v in row])
    return out


def _scalar(v):
    return None if (v is None or not np.isfinite(v)) else round(float(v), 5)


# ── Per-combination generators ────────────────────────────────────────────────

def generate_2d(domain_name, arch, k):
    model = _load_model(domain_name, arch, k, dim=2)
    if model is None:
        return None
    domain = make_domain(domain_name)

    xx_out = yy_out = None
    frames = []
    for t in TIMES:
        g = eval_grid_2d(model, domain, k, t, n_grid=N_GRID)
        if xx_out is None:
            xx_out = _clean(g["xx"])
            yy_out = _clean(g["yy"])
        frames.append({
            "t":      t,
            "T_pred": _clean(g["T_pred"]),
            "T_ref":  _clean(g["T_ref"]),
            "l2":     _scalar(g["l2"]),
            "mae":    _scalar(np.nanmean(g["err"]) if g["err"].size else np.nan),
        })
    return {"xx": xx_out, "yy": yy_out, "frames": frames}


def generate_3d(domain_name, arch, k):
    model = _load_model(domain_name, arch, k, dim=3)
    if model is None:
        return None
    domain = make_domain_3d(domain_name)

    xx_out = yy_out = None
    frames = []
    for t in TIMES:
        try:
            g = eval_grid_3d_mid(model, domain, k, t, n_grid=N_GRID)
        except Exception:
            continue
        if xx_out is None:
            xx_out = _clean(g["xx"])
            yy_out = _clean(g["yy"])
        frames.append({
            "t":      t,
            "T_pred": _clean(g["T_pred"]),
            "T_ref":  _clean(g["T_ref"]),
            "l2":     _scalar(g["l2"]),
            "mae":    _scalar(g["mae"]),
        })
    if not frames:
        return None
    return {"xx": xx_out, "yy": yy_out, "frames": frames}


# ── Image copy ────────────────────────────────────────────────────────────────

def copy_images():
    copies = [
        ("05_summary/fig_kskip_800ep_2d.png",          "summary_2d.png"),
        ("05_summary/fig_kskip_800ep_3d.png",          "summary_3d.png"),
        ("05_summary/fig_table_800ep_2d.png",          "table_2d.png"),
        ("05_summary/fig_table_800ep_3d.png",          "table_3d.png"),
        ("06_warmstart_stats/fig_ws_mae_comparison_2d.png", "ws_mae_2d.png"),
        ("06_warmstart_stats/fig_ws_mae_comparison_3d.png", "ws_mae_3d.png"),
        ("06_warmstart_stats/fig_ws_recommendation_2d.png", "ws_rec_2d.png"),
        ("06_warmstart_stats/fig_ws_recommendation_3d.png", "ws_rec_3d.png"),
        ("06_warmstart_stats/fig_ws_summary_table_2d.png",  "ws_table_2d.png"),
        ("06_warmstart_stats/fig_ws_summary_table_3d.png",  "ws_table_3d.png"),
    ]
    for rel, dst_name in copies:
        src = RESULT_DIR / rel
        dst = IMG_DIR / dst_name
        if src.exists():
            shutil.copy(src, dst)
            print(f"  img → {dst_name}")
        else:
            print(f"  img MISS: {src.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=0, choices=[0, 2, 3])
    args = parser.parse_args()

    # Load existing manifest so partial runs (--dim 2 or --dim 3) don't erase the other dim
    manifest_path = DATA_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest.setdefault("2d", {}); manifest.setdefault("3d", {})
    else:
        manifest = {"times": TIMES, "2d": {}, "3d": {}}
    dims = [2, 3] if args.dim == 0 else [args.dim]

    for dim in dims:
        domains = DOMAINS_2D_LIST if dim == 2 else DOMAINS_3D_LIST
        gen_fn  = generate_2d    if dim == 2 else generate_3d
        tag     = "2d"           if dim == 2 else "3d"
        print(f"\n{'='*50}\n  {dim}D data\n{'='*50}")

        for domain in domains:
            for arch in ARCHS:
                for k in range(1, 6):
                    ckpt = CKPT_DIR / f"{domain}_{arch}_k{k}_dim{dim}.pt"
                    if not ckpt.exists():
                        continue
                    print(f"  {domain}/{arch}/k={k}", end="", flush=True)
                    data = gen_fn(domain, arch, k)
                    if data is None:
                        print(" SKIP"); continue
                    fname = f"{tag}_{domain}_{arch}_k{k}.json"
                    with open(DATA_DIR / fname, "w") as f:
                        json.dump(data, f, separators=(",", ":"))
                    manifest[tag].setdefault(domain, {}).setdefault(arch, []).append(k)
                    size_kb = (DATA_DIR / fname).stat().st_size // 1024
                    print(f" → {fname} ({size_kb} KB)")

    with open(DATA_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest → {DATA_DIR}/manifest.json")

    print("\nCopying images:")
    copy_images()
    print(f"\nDone → {WEB_DIR}/")


if __name__ == "__main__":
    main()
