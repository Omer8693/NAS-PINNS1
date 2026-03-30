"""
retrain_nsga.py
===============
nsga2 ve nsga3 icin n_epochs=1500, lbfgs_iters=150 ile yeniden egitim.
train_all.TRAIN_KWARGS'i monkey-patch ederek override eder.

Kullanim:
    python -m thermal_pinn.training.retrain_nsga --dim 2 --cuda
    python -m thermal_pinn.training.retrain_nsga --dim 3 --cuda
"""
from __future__ import annotations
import argparse
import torch
import thermal_pinn.training.train_all as _ta

# Override TRAIN_KWARGS before importing sweep_k
_ta.TRAIN_KWARGS = {
    "n_domain":    1500,
    "n_epochs":    1500,
    "lr":          1e-3,
    "lr_min":      1e-5,
    "lbfgs_iters": 150,
}

from thermal_pinn.training.train_all import sweep_k

ARCHS           = ["nsga2", "nsga3"]
DOMAINS_2D_LIST = ["rectangle", "circle", "lshape"]
DOMAINS_3D_LIST = ["rectangular", "cylinder", "stacked", "lshape"]
K_VALUES        = [1, 2, 3, 4, 5]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dim",  type=int, default=2, choices=[2, 3])
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  n_epochs=1500  lbfgs_iters=150", flush=True)

    domains = DOMAINS_2D_LIST if args.dim == 2 else DOMAINS_3D_LIST

    for domain in domains:
        for arch in ARCHS:
            print(f"\n{'='*70}")
            print(f"  RETRAIN  domain={domain}  arch={arch}  dim={args.dim}D")
            print(f"{'='*70}", flush=True)
            sweep_k(domain, arch, args.dim, K_VALUES, device)

    print("\nRetrain tamamlandi.")
