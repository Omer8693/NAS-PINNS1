# NAS-PINN Thermal Quenching — Fixed-k and Adaptive-k Framework

A Neural Architecture Search (NAS) driven Physics-Informed Neural Network (PINN) framework
for accelerating transient thermal quenching simulations. The physical problem is the cooling
of an aluminium casting (A356 alloy) submerged in water: temperature drops from T_init = 500 °C
to near T_water = 20 °C over 30 seconds, governed by the transient heat equation with a
convective boundary condition (h = 5000 W/m²K).

The framework couples FEM and PINN via a **multi-step window prediction (MSWP)** strategy.
Instead of solving every FEM timestep (Δt_FEM = 1.5 s), the PINN covers k consecutive steps
in a single forward pass. Two usage modes are now included:

1. **Fixed-k sweep:** evaluate k ∈ {1, 2, 3, 4, 5} for every domain / architecture pair.
2. **Adaptive-k control:** choose k online from the previous window error, so the pipeline
   stays conservative in the early transient and skips more FEM anchors later when the
   temperature field becomes smoother.

The network output is IC-consistent by construction: it exactly satisfies the initial condition
at the start of each window, avoiding additional penalty terms.

Three NAS optimizers — **Bayesian TPE**, **NSGA-II**, and **NSGA-III** — independently search
for optimal layer depth, width, and activation function over four 2D and four 3D domain
geometries (rectangle, circle, L-shape, cylinder, stacked, etc.). The search space follows
Wang & Zhong (2023) adapted to the quenching problem.

Training and evaluation now span four main tracks:
1. **Cold-start (v1):** 800 Adam epochs from random initialisation per window.
2. **v2 (Fourier + SA):** Fourier feature embedding combats spectral bias; self-adaptive loss
   weights are learned jointly with model parameters. NSGA-II/III additionally retrained with
   1500 epochs + 150 L-BFGS iterations.
3. **Warm-start:** Window 1 trained cold (800 ep); subsequent windows fine-tune from the
   previous window's weights (500 ep, lr = 1e-3), achieving comparable accuracy at 38% fewer epochs.
4. **Adaptive-k / 3D follow-up:** legacy adaptive-k, conservative adaptive-k v2, improved 3D
   normalization experiments, and 3D triplet comparison figures for thesis selection.

---

## Repository Structure

```
thermal_pinn/
├── network/
│   ├── pinn.py              # ThermalPINN — IC-consistent MLP (v1)
│   └── fourier_pinn.py      # FourierPINN — Fourier embedding + self-adaptive weights (v2)
├── physics/
│   ├── domains_2d.py        # 2D domains: rectangle, circle, lshape
│   └── domains_3d.py        # 3D domains: rectangular, cylinder, stacked, lshape
├── training/
│   ├── trainer.py           # Window trainer: Adam + L-BFGS (v1)
│   ├── train_all.py         # Full sweep launcher (v1, cold-start)
│   ├── train_warmstart.py   # Warm-start trainer (500 ep fine-tune per window)
│   ├── train_adaptive_k.py  # Legacy adaptive-k controller
│   ├── train_adaptive_k_v2.py
│   │                       # Conservative adaptive-k controller with burn-in + phase caps
│   ├── train_3d_improved.py
│   │                       # Experimental 3D normalization / residual scaling study
│   └── train_3d_improved_warmstart.py
│                           # Warm-start variant of the improved 3D study
├── plots/
│   ├── plot_results.py      # Core evaluation utilities (CKPT_DIR, RESULT_DIR, eval_grid_*)
│   ├── plot_thesis.py       # Publication-quality figures (heatmaps, tables, k-progression)
│   ├── plot_warmstart.py    # Cold vs v2 vs warm-start MAE comparison
│   ├── plot_adaptive_k.py   # Adaptive-k metric summaries and controller comparisons
│   ├── plot_adaptive_k_fields.py
│   │                       # Exact vs predicted fields at adaptive k-change events
│   ├── plot_adaptive_k_triplet_3d.py
│   │                       # 3D triplet snapshots: fixed k=1 vs adaptive v1 vs adaptive v2
│   ├── plot_ws_heatmaps.py  # Side-by-side cold/warm temperature field heatmaps
│   ├── plot_timeline.py     # Training timeline and window-step plots
│   └── plot_comparison.py   # Architecture comparison plots
├── gen_pptx_full.py         # Generate full 19-slide PPTX report
├── gen_docx_full.py         # Generate full DOCX report
├── checkpoints/             # Model weights (.pt) and metrics (.json)
└── results/
    ├── 01_all_k_fields/     # Per-k temperature field plots (all domains, all k)
    ├── 02_best_k/           # Best-k heatmaps per (domain, arch)
    ├── 03_timeline/         # Training timeline figures
    ├── 04_window_steps/     # Window-step progression plots
    ├── 05_summary/          # Summary tables and bar charts
    ├── 06_warmstart_stats/  # Cold vs v2 vs warm MAE comparison, summary table, recommendation
    ├── 07_warmstart_fields/ # Cold vs warm temperature field heatmaps (all k)
    ├── adaptive_k/          # Adaptive-k aggregate plots and summaries
    ├── adaptive_k_fields/   # Adaptive-k field comparisons at k-change events
    └── adaptive_k_triplet_3d/
                            # Common-time 3D triplet figures for thesis use
```

---

## Usage

### Training

```bash
# Cold-start — all domains/archs/k
python -m thermal_pinn.training.train_all --all --cuda

# Warm-start (500 ep fine-tune per window)
python -m thermal_pinn.training.train_warmstart --dim 2 --cuda

# Legacy adaptive-k
python -m thermal_pinn.training.train_adaptive_k \
  --domain cylinder --arch bayesian --dim 3 --cuda

# Conservative adaptive-k v2
python -m thermal_pinn.training.train_adaptive_k_v2 \
  --domain cylinder --arch bayesian --dim 3 --cuda \
  --tag adaptive_v2_all3d_rerun

# Experimental improved 3D study
python -m thermal_pinn.training.train_3d_improved \
  --domain cylinder --arch bayesian --k 1 --cuda
```

### Plotting

```bash
# All thesis figures
python -m thermal_pinn.plots.plot_thesis

# k-field heatmaps + summary
python -m thermal_pinn.plots.plot_results

# Warm-start comparison figures (2D + 3D)
python -m thermal_pinn.plots.plot_warmstart

# Adaptive-k event fields
python -m thermal_pinn.plots.plot_adaptive_k_fields

# 3D fixed-vs-adaptive triplet snapshots
python -m thermal_pinn.plots.plot_adaptive_k_triplet_3d \
  --all --v2-tag adaptive_v2_all3d_rerun

# Cold vs warm temperature field heatmaps
python -m thermal_pinn.plots.plot_ws_heatmaps

# Timeline plots
python -m thermal_pinn.plots.plot_timeline
```

### Reports

```bash
python -m thermal_pinn.gen_pptx_full   # → reports/NAS_PINN_Thermal_Quenching.pptx
python -m thermal_pinn.gen_docx_full   # → reports/NAS_PINN_Thermal_Quenching_Report.docx
```

---

## Key Parameters

| Parameter | Value |
|-----------|-------|
| T_init | 500 °C |
| T_water | 20 °C |
| h (HTC) | 5000 W/m²K |
| t_total | 30 s |
| Δt_FEM | 1.5 s |
| k range | 1–5 |
| Cold-start epochs | 800 (Adam) |
| Warm-start epochs | 500 (Adam, lr=1e-3) |
| v2 epochs | 1500 (Adam) + 150 (L-BFGS) for NSGA |

---

## Recent Additions

- Added adaptive-k training scripts for both the original controller and the more conservative
  `adaptive_k_v2` controller.
- Added experimental 3D scripts to study centered-domain normalization and larger 3D budgets.
- Added `plot_adaptive_k_fields.py` and `plot_adaptive_k_triplet_3d.py` for qualitative
  comparison of fixed-k and adaptive-k runs.
- Generated 3D triplet figures for all domain/architecture combinations under
  `results/adaptive_k_triplet_3d/`.

---

## References

[1] Lagaris et al. (1998). Artificial neural networks for solving ODEs/PDEs. *IEEE TNN*, 9(5).

[2] Tancik et al. (2020). Fourier features let networks learn high frequency functions. *NeurIPS*.

[3] McClenny & Braga-Neto (2023). Self-adaptive physics-informed neural networks. *J. Comput. Phys.*

[4] Bergstra et al. (2011). Algorithms for hyper-parameter optimization. *NeurIPS*.

[5] Deb et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE TEC*.

[6] Deb & Jain (2014). Many-objective optimization with reference-point NSGA-III. *IEEE TEC*.

[7] Wang & Zhong (2023). NAS-PINN: Neural Architecture Search-Guided PINN. *arXiv:2305.10127*.

[8] Mortensen et al. (2026). FEM simulation of quenching in aluminium casting. *Int. J. Adv. Manuf. Technol.*
