# NAS-PINN Thermal Quenching — k-Skip Framework

NAS-driven PINN framework for transient thermal quenching simulation using a
multi-step window prediction (MSWP) strategy with k-skip FEM coupling.

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
│   ├── train_all.py         # Full sweep launcher (v1)
│   ├── sa_trainer.py        # Self-adaptive weight trainer (v2)
│   ├── train_v2.py          # Full sweep launcher (v2)
│   ├── retrain_nsga.py      # NSGA-II/III extended retrain (1500 ep + L-BFGS)
│   └── warmstart_trainer.py # Warm-start trainer (500 ep fine-tune per window)
├── plots/
│   ├── plot_results.py      # Core evaluation utilities (CKPT_DIR, RESULT_DIR, eval_grid_*)
│   ├── plot_thesis.py       # Publication-quality figures (heatmaps, tables, k-progression)
│   ├── plot_warmstart.py    # Cold vs v2 vs warm-start comparison plots
│   ├── plot_ws_heatmaps.py  # Side-by-side cold/warm temperature field heatmaps
│   └── plot_timeline.py     # Training timeline and window-step plots
├── gen_pptx_full.py         # Generate full 19-slide PPTX report
├── gen_docx_full.py         # Generate full DOCX report
├── reports/                 # Generated PPTX and DOCX
├── checkpoints/             # Model weights (.pt) and metrics (.json)
└── results/
    ├── 01_all_k_fields/     # Per-k temperature field plots (all domains, all k)
    ├── 02_best_k/           # Best-k heatmaps per (domain, arch)
    ├── 03_timeline/         # Training timeline figures
    ├── 04_window_steps/     # Window-step progression plots
    ├── 05_summary/          # Summary tables and bar charts
    ├── 06_warmstart_stats/  # Cold vs v2 vs warm MAE comparison, summary table, recommendation
    └── 07_warmstart_fields/ # Cold vs warm temperature field heatmaps (all k)
```

---

## Usage

### Training

```bash
# v1 — cold-start, all domains/archs/k
python -m thermal_pinn.training.train_all --all --cuda

# v2 — Fourier + self-adaptive weights
python -m thermal_pinn.training.train_v2 --all --dim 2 --cuda

# NSGA-II/III extended retrain
python -m thermal_pinn.training.retrain_nsga --dim 2 --cuda

# Warm-start (500 ep fine-tune)
python -m thermal_pinn.training.warmstart_trainer --dim 2 --cuda
```

### Plotting

```bash
# All thesis figures
python -m thermal_pinn.plots.plot_thesis

# k-field heatmaps + summary
python -m thermal_pinn.plots.plot_results

# Warm-start comparison figures (2D + 3D)
python -m thermal_pinn.plots.plot_warmstart

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

## References

[1] Lagaris et al. (1998). Artificial neural networks for solving ODEs/PDEs. *IEEE TNN*, 9(5).

[2] Tancik et al. (2020). Fourier features let networks learn high frequency functions. *NeurIPS*.

[3] McClenny & Braga-Neto (2023). Self-adaptive physics-informed neural networks. *J. Comput. Phys.*

[4] Bergstra et al. (2011). Algorithms for hyper-parameter optimization. *NeurIPS*.

[5] Deb et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE TEC*.

[6] Deb & Jain (2014). Many-objective optimization with reference-point NSGA-III. *IEEE TEC*.

[7] Wang & Zhong (2023). NAS-PINN: Neural Architecture Search-Guided PINN. *arXiv:2305.10127*.

[8] Mortensen et al. (2026). FEM simulation of quenching in aluminium casting. *Int. J. Adv. Manuf. Technol.*
