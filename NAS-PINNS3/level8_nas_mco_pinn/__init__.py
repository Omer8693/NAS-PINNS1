"""
Level 8 — NAS-MCO-PINN Time-Step Skip Operator
================================================
Level 2 time-step skip operator enhanced with
MCO (Multi-Loss Consistency Optimization) weights.

Thesis question: Can FEM's step-by-step computation be replaced by PINN?
Answer: MCO with skip=2 → MAE ~2°C (L2 baseline: 33°C)

Reference: Shen et al. (2023) Sensors 23(21), 8885.
"""

from .mco_timestepper import (
    MCOLoss,
    build_net,
    train_window_mco,
    evaluate_skip_mco,
    run_full_comparison,
    analytical_T,
    SEARCH_SPACE,
    SKIP_VALUES,
    TRAIN_KWARGS,
)

from .domains_3d import (
    PlaneWall1D,
    Rectangular3D,
    Cylinder3D,
    StackedCubes3D,
    LPrism3D,
    all_domains,
    make_grid_2d_slice,
    verify_fields,
)
