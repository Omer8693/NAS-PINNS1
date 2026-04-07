"""
thermal_pinn/web/generate_frames.py
=====================================
Generate PNG animation frames for the NAS-PINN website viewer.

2D: matplotlib pcolormesh heatmap (reads from existing JSON data files)
3D: matplotlib 3D volumetric scatter (PINN model inference + _render_fn)

Output:
    web/frames/2d_{domain}_{arch}_k{k}_t{idx:02d}.png
    web/frames/3d_{domain}_{arch}_k{k}_t{idx:02d}.png
    web/frames/manifest_frames.json

Run from NAS-PINNS1/:
    python -m thermal_pinn.web.generate_frames [--dim 0|2|3]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent        # thermal_pinn/
sys.path.insert(0, str(ROOT.parent))

from thermal_pinn.plots.plot_results import (
    CKPT_DIR, RESULT_DIR, DEVICE, T_WATER, T_INIT,
)
from thermal_pinn.plots.plot_thesis import DOMAINS_2D_LIST, DOMAINS_3D_LIST
from thermal_pinn.plots.plot_ws_heatmaps import (
    _load_model, _render_fn, _make_pinn_field_fn, _style_3d_ax,
)
from thermal_pinn.physics.domains_3d import make_domain_3d

WEB_DIR    = ROOT / "web"
DATA_DIR   = WEB_DIR / "data"
FRAMES_DIR = WEB_DIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

ARCHS  = ["bayesian", "nsga2", "nsga3"]
TIMES  = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]
NORM   = Normalize(vmin=T_WATER, vmax=T_INIT)
CMAP   = mpl.colormaps["plasma"]
DPI    = 90


# ── 2D frames (from JSON) ──────────────────────────────────────────────────────

def make_2d_frame(domain_name: str, arch: str, k: int, tidx: int, out: Path):
    data_path = DATA_DIR / f"2d_{domain_name}_{arch}_k{k}.json"
    if not data_path.exists():
        return False

    with open(data_path) as f:
        data = json.load(f)
    frame = data["frames"][tidx]
    t     = frame["t"]

    XX = np.array(data["xx"], dtype=float)
    YY = np.array(data["yy"], dtype=float)
    T  = np.array([[v if v is not None else np.nan for v in row]
                   for row in frame["T_pred"]], dtype=float)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    fig.patch.set_facecolor("white")
    im = ax.pcolormesh(XX, YY, T, cmap=CMAP, norm=NORM, shading="auto")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"t = {t:.0f} s", fontsize=9, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("T [°C]", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    plt.tight_layout(pad=0.5)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


# ── 3D frames (model inference + volumetric render) ───────────────────────────

def make_3d_frame(domain_name: str, arch: str, k: int, tidx: int, out: Path):
    model = _load_model(domain_name, arch, k, dim=3, suffix="")
    if model is None:
        return False

    domain    = make_domain_3d(domain_name)
    render_fn = _render_fn(domain)
    field_fn  = _make_pinn_field_fn(model, domain, k)
    t         = TIMES[tidx]

    fig = plt.figure(figsize=(4.5, 4.0))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")

    render_fn(ax, domain, t, CMAP, NORM, field_fn=field_fn)
    _style_3d_ax(ax, domain)
    ax.view_init(elev=25, azim=-60)

    if hasattr(domain, "R"):
        ax.set_xlim(-domain.R, domain.R)
        ax.set_ylim(-domain.R, domain.R)
        ax.set_zlim(0, domain.Hz)
        dims = np.array([2*domain.R, 2*domain.R, domain.Hz], dtype=float)
    else:
        Lx = getattr(domain, "Lx", 1.0)
        Ly = getattr(domain, "Ly", 1.0)
        Lz = getattr(domain, "Lz", getattr(domain, "Hz", 1.0))
        ax.set_xlim(0, Lx); ax.set_ylim(0, Ly); ax.set_zlim(0, Lz)
        dims = np.array([Lx, Ly, Lz], dtype=float)
    ax.set_box_aspect(dims / dims.max())

    ax.text2D(0.5, 0.97, f"t = {t:.0f} s", transform=ax.transAxes,
              ha="center", va="top", fontsize=9, fontweight="bold")

    # Colorbar
    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.08, shrink=0.7)
    cb.set_label("T [°C]", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    plt.tight_layout(pad=0.3)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=0, choices=[0, 2, 3])
    args = parser.parse_args()

    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {"2d": {}, "3d": {}}
    frames_manifest_path = DATA_DIR / "manifest_frames.json"
    if frames_manifest_path.exists():
        frames_manifest = json.load(open(frames_manifest_path))
        frames_manifest.setdefault("times", TIMES)
        frames_manifest.setdefault("2d", {}); frames_manifest.setdefault("3d", {})
    else:
        frames_manifest = {"times": TIMES, "2d": {}, "3d": {}}
    dims = [2, 3] if args.dim == 0 else [args.dim]

    for dim in dims:
        tag      = f"{dim}d"
        domains  = DOMAINS_2D_LIST if dim == 2 else DOMAINS_3D_LIST
        make_fn  = make_2d_frame   if dim == 2 else make_3d_frame
        avail    = manifest.get(tag, {})
        print(f"\n{'='*50}\n  {dim}D frames\n{'='*50}")

        for domain in domains:
            if domain not in avail:
                continue
            for arch in avail[domain]:
                for k in avail[domain][arch]:
                    print(f"  {domain}/{arch}/k={k}", end="", flush=True)
                    ok_count = 0
                    for tidx in range(len(TIMES)):
                        fname = f"{tag}_{domain}_{arch}_k{k}_t{tidx:02d}.png"
                        out   = FRAMES_DIR / fname
                        if out.exists():
                            ok_count += 1
                            continue
                        if make_fn(domain, arch, k, tidx, out):
                            ok_count += 1
                    if ok_count > 0:
                        frames_manifest[tag].setdefault(domain, {}).setdefault(arch, [])
                        if k not in frames_manifest[tag][domain][arch]:
                            frames_manifest[tag][domain][arch].append(k)
                        print(f" → {ok_count}/{len(TIMES)} frames")
                    else:
                        print(" SKIP")

    out_path = frames_manifest_path
    with open(out_path, "w") as f:
        json.dump(frames_manifest, f, indent=2)
    print(f"\nFrames manifest → {out_path}")
    print(f"Done → {FRAMES_DIR}/")


if __name__ == "__main__":
    main()
