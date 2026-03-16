# Level 5 — Refinement Strategy: Findings & Revised Plan
### (İyileştirme Stratejisi: Bulgular ve Revize Plan)

---

## What We Tried — L-BFGS (Denenenler — L-BFGS)

### Attempt 1: Load Level 1 weights → apply L-BFGS
- **Result:** 53–73 iterations, no improvement (1.0×)
- **Why it failed:** Adam 20K epochs → already well-converged local minimum.
  L-BFGS finds nothing left to improve from an over-trained model.

### Attempt 2: Fresh init → Adam(5000ep) → L-BFGS(5000it)
- **Result:** Adam L2=0.308, L-BFGS L2=0.239 (1.3×). Still worse than Level 1 (0.076).
- **Why it failed:** 5K Adam epochs insufficient to reach Level 1 quality basin.
  L-BFGS then stops at 37 iterations due to strong Wolfe line search failure.

### Attempt 3: L-BFGS without line search (lr=0.01)
- **Result:** Diverges to NaN at step 5.
- **Why it failed:** Hessian-scaled steps too large without line search control.

---

## Root Cause Analysis (Kök Sebep Analizi)

L-BFGS works well when:
- Loss is smooth and nearly-quadratic near minimum ✓ (Poisson, Burgers)
- Single loss term ✓ (NAS-PINNS2 proxy objective)
- Small network ✓

L-BFGS fails here because:
- **Multi-component loss** (physics + Robin BC + IC with different scales)
  → Hessian approximation B_k becomes ill-conditioned across loss components
- **Physics residual involves autograd of autograd** (∂²T/∂x², ∂²T/∂y²)
  → High-order gradients create highly non-linear loss landscape
- **Robin BC boiling curve HTC(T)** is piecewise-nonlinear
  → Violates L-BFGS curvature assumption

NAS-PINNS2 "20000× improvement" was on **proxy NAS objective** (single-term scalar
proxy for architecture search), not on the full multi-physics PINN loss.

---

## Revised Level 5: Extended Adam with LR Schedule
### (Revize Kademe 5: LR Planlı Uzatılmış Adam)

**Goal:** Improve NSGA-II (L2=0.252) and NSGA-III (L2=0.513) to reach
the L2 < 0.10 target using extended training with proper learning rate scheduling.
Bayesian is already at L2=0.076 ✓ (no need to improve).

### Strategy (Strateji)

```
Phase 1: Adam  lr=1e-3  →  epochs 0..5000    (fast descent to rough basin)
Phase 2: Adam  lr=1e-4  →  epochs 5000..12000 (fine descent)
Phase 3: Adam  lr=1e-5  →  epochs 12000..20000 (precision refinement)
```

This is simply **Level 1 with cosine LR decay** instead of fixed LR + milestones.
The target: bring NSGA-II and NSGA-III to Bayesian level.

### Why This Works (Neden İşe Yarar)

```
Level 1 (fixed LR):           Adam(20K, lr=1e-3)
  Bayesian:  L2=0.076 ✓
  NSGA-II:   L2=0.252 ✗
  NSGA-III:  L2=0.513 ✗

Level 5 (cosine LR decay):    Adam(20K, lr=1e-3→1e-5)
  Expected: all 3 → L2 < 0.10
```

### Implementation Plan (Uygulama Planı)

```python
# In main_level5.py — replace Adam fixed LR with CosineAnnealingLR
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=20000, eta_min=1e-5
)
```

Parameters to tune (Ayarlanacak parametreler):
- T_max: total Adam epochs (20000)
- eta_min: final LR (1e-5)
- Optional warmup: first 500 epochs lr=1e-4 → 1e-3 (linear warmup)

---

## Honest Comparison Table (Dürüst Karşılaştırma Tablosu)

| Method | Bayesian L2 | NSGA-II L2 | NSGA-III L2 | Notes |
|--------|------------|------------|-------------|-------|
| Level 1: Adam 20K fixed LR | 0.076 ✓ | 0.252 ✗ | 0.513 ✗ | Current best |
| Level 5 attempt: Adam+LBFGS | 0.239 ✗ | — | — | Worse than L1 |
| Level 5 revised: Adam cosine | **TBD** | **TBD** | **TBD** | In progress |
| Target | < 0.10 | < 0.10 | < 0.10 | |

---

## Files (Dosyalar)

```
level5_refinement/
├── DRAFT.md                 ← findings + revised plan (bu dosya)
├── main_level5.py           ← revised: Adam(cosine) + optional L-BFGS
├── plot_results.py          ← before/after plots
└── src/
    ├── batch_sampler.py     ← fixed-batch sampler (retained for future L-BFGS use)
    └── lbfgs_refiner.py     ← LBFGSRefiner (retained, not primary path)
```

## Run (Çalıştırma)

```bash
# Revised Level 5 — cosine LR Adam for all optimizers
python level5_refinement/main_level5.py --adam_epochs 20000 --optimizer all

# Single optimizer test
python level5_refinement/main_level5.py --adam_epochs 20000 --optimizer nsga2

# Visualize after completion
python level5_refinement/plot_results.py
```
