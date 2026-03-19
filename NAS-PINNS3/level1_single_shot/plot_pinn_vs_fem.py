"""
plot_pinn_vs_fem.py
===================
Generates three clean, publication-ready figures:

  fig1_cooling_curves.png  — FEM vs PINN temperature curves at 4 probe points
  fig2_spatial_fields.png  — 2D temperature heatmaps at t=0,10,20,30 s
  fig3_error_analysis.png  — Absolute error curves + summary metrics table

Reference: Mortensen et al. (2026), A356 aluminium water quenching
"""

import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pinn_network import PINNNet

# ── Physics ───────────────────────────────────────────────────────────────────
T_INIT, T_WATER, ALPHA = 540.0, 20.0, 1.75e-3
X_MAX, Y_MAX, T_END    = 1.3, 0.6, 30.0
N_STEPS                = 20

def T_fem(t):
    return T_WATER + (T_INIT - T_WATER) * np.exp(-ALPHA * np.asarray(t, float))

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "Bayesian — L1 Adam":         dict(path="level1_single_shot/results/bayesian/model.pt",
                                       neurons=[151]*5, act="relu",
                                       color="#1565C0", ls="--", lw=2.0, alpha=0.70),
    "Bayesian — L5 Adam+L-BFGS":  dict(path="level5_refinement/results/bayesian/model_lbfgs.pt",
                                       neurons=[151]*5, act="relu",
                                       color="#1565C0", ls="-",  lw=3.0, alpha=1.00),
    "NSGA-II — L5 Adam+L-BFGS":   dict(path="level5_refinement/results/nsga2/model_lbfgs.pt",
                                       neurons=[153]*3, act="tanh",
                                       color="#C62828", ls="-",  lw=2.5, alpha=1.00),
    "NSGA-III — L5 Adam+L-BFGS":  dict(path="level5_refinement/results/nsga3/model_lbfgs.pt",
                                       neurons=[75]*3,  act="tanh",
                                       color="#2E7D32", ls="-",  lw=2.5, alpha=1.00),
}

PROBES = [
    ("Center  (0.65, 0.30)",   0.65, 0.30),
    ("Corner  (0.00, 0.00)",   0.00, 0.00),
    ("Edge mid (0.00, 0.30)",  0.00, 0.30),
    ("Far tip  (1.30, 0.30)",  1.30, 0.30),
]

# ── Style ─────────────────────────────────────────────────────────────────────
FIG_BG  = "#f7f1e3"
AX_BG   = "#fffdf8"
SPINE_C = "#d8cfbf"
TXT_C   = "#2f2a24"
TICK_C  = "#4c463f"
FEM_C   = "#E65100"

def _style(ax, fs=11):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_color(SPINE_C); sp.set_linewidth(1.0)
    ax.tick_params(colors=TICK_C, labelsize=fs)
    ax.xaxis.label.set_color(TICK_C)
    ax.yaxis.label.set_color(TICK_C)
    ax.title.set_color(TXT_C)
    ax.grid(True, alpha=0.35, color=SPINE_C, linewidth=0.7)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_model(cfg):
    net = PINNNet(n_input=3, n_output=1, hidden_sizes=cfg["neurons"],
                  activation=cfg["act"], residual=False)
    net.load_state_dict(
        torch.load(ROOT / cfg["path"], map_location="cpu", weights_only=False))
    net.eval()
    return net

@torch.no_grad()
def predict_curve(net, x, y, t_arr):
    return np.array([net(torch.tensor([[float(t), float(x), float(y)]])).item()
                     for t in t_arr])

@torch.no_grad()
def predict_field(net, t_val, nx=60, ny=30):
    xs = np.linspace(0, X_MAX, nx)
    ys = np.linspace(0, Y_MAX, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    pts = torch.tensor(np.column_stack(
        [np.full(XX.size, t_val), XX.ravel(), YY.ravel()]), dtype=torch.float32)
    return xs, ys, net(pts).numpy().reshape(nx, ny)

def jget(d, *keys, fmt=".4f"):
    try:
        v = d
        for k in keys:
            v = v[k]
        return format(float(v), fmt)
    except Exception:
        return "—"

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading models...")
nets   = {n: load_model(c) for n, c in MODELS.items()}
t_arr  = np.linspace(0, T_END, 121)
T_ref  = T_fem(t_arr)
t_disc = np.linspace(0, T_END, N_STEPS + 1)
OUT    = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — Cooling curves
# ═════════════════════════════════════════════════════════════════════════════
print("Fig 1: cooling curves...")
fig1, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=FIG_BG)
fig1.suptitle(
    "Cooling Curves: FEM Reference vs PINN Predictions\n"
    r"A356 Aluminium Water Quenching — $T_{ref}(t)=20+520\,e^{-1.75\times10^{-3}t}$ °C",
    fontsize=15, color=TXT_C, fontweight="bold", y=1.01,
)

for ax, (label, px, py) in zip(axes.flat, PROBES):
    _style(ax, fs=11)
    ax.axhspan(T_fem(T_END) - 1, T_fem(0) + 1,
               alpha=0.10, color=FEM_C, zorder=0)
    ax.plot(t_arr, T_ref, color=FEM_C, lw=3.5, zorder=10,
            label="FEM Reference (analytical)")
    ax.scatter(t_disc, T_fem(t_disc), color=FEM_C, s=35, zorder=11, alpha=0.85)

    for name, cfg in MODELS.items():
        T_pred = predict_curve(nets[name], px, py, t_arr)
        ax.plot(t_arr, T_pred, color=cfg["color"], ls=cfg["ls"],
                lw=cfg["lw"], alpha=cfg["alpha"], label=name, zorder=5)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title(f"Probe Point — {label}", fontsize=12, color=TXT_C, pad=6)
    # auto y-limits including FEM reference
    all_preds = [predict_curve(nets[n], px, py, t_arr) for n in MODELS]
    y_lo = min(p.min() for p in all_preds + [T_ref])
    y_hi = max(p.max() for p in all_preds + [T_ref])
    pad  = (y_hi - y_lo) * 0.12
    ax.set_xlim(-0.5, 31)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    leg = ax.legend(fontsize=9.5, loc="lower left",
                    facecolor=AX_BG, edgecolor=SPINE_C, framealpha=0.92)
    for t in leg.get_texts():
        t.set_color(TXT_C)

fig1.tight_layout(pad=2.5)
fig1.savefig(OUT / "fig1_cooling_curves.png", dpi=150,
             bbox_inches="tight", facecolor=FIG_BG)
plt.close(fig1)
print("  Saved: fig1_cooling_curves.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — Spatial heatmaps: Row 1 = PINN field, Row 2 = Error |PINN − FEM|
# ═════════════════════════════════════════════════════════════════════════════
print("Fig 2: spatial heatmaps...")
snap_times = [0, 10, 20, 30]
best_net   = nets["Bayesian — L5 Adam+L-BFGS"]

all_fields = [predict_field(best_net, t)[2] for t in snap_times]
all_errors = [np.abs(f - float(T_fem(t))) for f, t in zip(all_fields, snap_times)]

# Shared colour scales
t_vmin = np.floor(min(f.min() for f in all_fields) / 5) * 5
t_vmax = np.ceil( max(f.max() for f in all_fields) / 5) * 5
e_vmax = np.ceil( max(e.max() for e in all_errors) / 5) * 5

fig2, axes2 = plt.subplots(
    2, 4, figsize=(22, 8),
    facecolor=FIG_BG,
    gridspec_kw={"hspace": 0.45, "wspace": 0.35,
                 "left": 0.05, "right": 0.96,
                 "top": 0.88,  "bottom": 0.08},
)
fig2.suptitle(
    "PINN Temperature Field vs FEM Reference — Bayesian NAS (L5: Adam + L-BFGS)\n"
    "Row 1: PINN Predicted Temperature  |  Row 2: Absolute Error  |T_PINN − T_FEM|",
    fontsize=14, color=TXT_C, fontweight="bold",
)

# Row labels
for row_i, row_lbl in enumerate(["PINN Temperature (°C)", "Error |PINN − FEM| (°C)"]):
    fig2.text(0.01, 0.73 - row_i * 0.47, row_lbl,
              fontsize=12, color=TXT_C, fontweight="bold",
              va="center", rotation="vertical")

xs_g = np.linspace(0, X_MAX, all_fields[0].shape[0])
ys_g = np.linspace(0, Y_MAX, all_fields[0].shape[1])

for col, (t_snap, T_field, E_field) in enumerate(
        zip(snap_times, all_fields, all_errors)):

    T_ref_snap = float(T_fem(t_snap))

    # ── Row 0: PINN temperature ──────────────────────────────────────────────
    ax0 = axes2[0, col]
    im0 = ax0.pcolormesh(xs_g, ys_g, T_field.T,
                         cmap="plasma", vmin=t_vmin, vmax=t_vmax,
                         shading="gouraud")
    cb0 = fig2.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03)
    cb0.ax.tick_params(labelsize=10, colors=TICK_C)
    cb0.set_label("°C", fontsize=10, color=TICK_C)
    ax0.contour(xs_g, ys_g, T_field.T, levels=5,
                colors="white", alpha=0.35, linewidths=0.8)
    ax0.set_title(
        f"t = {t_snap} s\nPINN mean = {T_field.mean():.1f} °C   "
        f"(FEM ref = {T_ref_snap:.1f} °C)",
        fontsize=11, color=TXT_C, pad=5,
    )
    for _, px, py in PROBES:
        ax0.scatter(px, py, s=70, color="white", edgecolors="#111",
                    linewidths=1.3, zorder=10)

    # ── Row 1: error map ─────────────────────────────────────────────────────
    ax1 = axes2[1, col]
    im1 = ax1.pcolormesh(xs_g, ys_g, E_field.T,
                         cmap="Reds", vmin=0, vmax=e_vmax,
                         shading="gouraud")
    cb1 = fig2.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cb1.ax.tick_params(labelsize=10, colors=TICK_C)
    cb1.set_label("°C", fontsize=10, color=TICK_C)
    ax1.contour(xs_g, ys_g, E_field.T, levels=4,
                colors="#333", alpha=0.40, linewidths=0.7)
    ax1.set_title(
        f"Max err = {E_field.max():.1f} °C   |   "
        f"Mean err = {E_field.mean():.1f} °C",
        fontsize=11, color=TXT_C, pad=5,
    )
    for _, px, py in PROBES:
        ax1.scatter(px, py, s=70, color="white", edgecolors="#111",
                    linewidths=1.3, zorder=10)

    for ax in (ax0, ax1):
        ax.set_xlabel("x (m)", fontsize=11)
        ax.set_ylabel("y (m)", fontsize=11)
        ax.tick_params(labelsize=10, colors=TICK_C)
        ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_color(SPINE_C)

fig2.savefig(OUT / "fig2_spatial_fields.png", dpi=150,
             bbox_inches="tight", facecolor=FIG_BG)
plt.close(fig2)
print("  Saved: fig2_spatial_fields.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — Error curves + metrics table
# ═════════════════════════════════════════════════════════════════════════════
print("Fig 3: error analysis...")

def load_json(path):
    p = ROOT / path
    return json.load(open(p)) if p.exists() else {}

l1_bay = load_json("level1_single_shot/results/bayesian/results.json")
l5_bay = load_json("level5_refinement/results/bayesian/comparison.json")
l5_n2  = load_json("level5_refinement/results/nsga2/comparison.json")
l5_n3  = load_json("level5_refinement/results/nsga3/comparison.json")

fig3, (ax_err, ax_tbl) = plt.subplots(1, 2, figsize=(18, 7), facecolor=FIG_BG)
fig3.suptitle(
    "Error Analysis: PINN vs FEM Reference — A356 Quenching (Mortensen et al. 2026)",
    fontsize=14, color=TXT_C, fontweight="bold",
)

# ── Error curves ──────────────────────────────────────────────────────────────
_style(ax_err, fs=11)
for name, cfg in MODELS.items():
    err = np.abs(predict_curve(nets[name], 0.65, 0.30, t_arr) - T_ref)
    ax_err.plot(t_arr, err, color=cfg["color"], ls=cfg["ls"],
                lw=cfg["lw"], alpha=cfg["alpha"], label=name)

ax_err.axhline(39.1, color=FEM_C, ls=":", lw=2.0, alpha=0.8,
               label="Bayesian L1 MAE baseline (39.1 °C)")
ax_err.fill_between(t_arr, 0, 39.1, alpha=0.05, color=FEM_C)
ax_err.set_xlabel("Time (s)", fontsize=13)
ax_err.set_ylabel(r"|$T_{PINN}$ − $T_{FEM}$|  (°C)", fontsize=13)
ax_err.set_title("Absolute Temperature Error at Center Point (0.65, 0.30 m)",
                 fontsize=12, color=TXT_C, pad=7)
ax_err.set_xlim(-0.5, 31)
ax_err.set_ylim(0)
leg = ax_err.legend(fontsize=10, loc="upper left",
                    facecolor=AX_BG, edgecolor=SPINE_C, framealpha=0.92)
for t in leg.get_texts():
    t.set_color(TXT_C)

# ── Metrics table ─────────────────────────────────────────────────────────────
ax_tbl.axis("off")
ax_tbl.set_facecolor(AX_BG)

col_labels = ["Model", "Training", "L2 Error\n(global)", "MAE (°C)", "Max Err (°C)", "Params"]
rows = [
    ["Bayesian",
     "L1 — Adam (20k)",
     jget(l1_bay, "paper_baseline", "thermal_metrics", "L2_rel"),
     jget(l1_bay, "paper_baseline", "thermal_metrics", "MAE_C",    fmt=".1f"),
     jget(l1_bay, "paper_baseline", "thermal_metrics", "MaxErr_C", fmt=".1f"),
     "92,564"],
    ["Bayesian",
     "L5 — Adam + L-BFGS",
     jget(l5_bay, "level5_adam_lbfgs", "L2_rel"),
     jget(l5_bay, "level5_adam_lbfgs", "MAE_C",    fmt=".1f"),
     jget(l5_bay, "level5_adam_lbfgs", "MaxErr_C", fmt=".1f"),
     "92,564"],
    ["NSGA-II",
     "L5 — Adam + L-BFGS",
     jget(l5_n2,  "level5_adam_lbfgs", "L2_rel"),
     jget(l5_n2,  "level5_adam_lbfgs", "MAE_C",    fmt=".1f"),
     jget(l5_n2,  "level5_adam_lbfgs", "MaxErr_C", fmt=".1f"),
     "47,890"],
    ["NSGA-III",
     "L5 — Adam + L-BFGS",
     jget(l5_n3,  "level5_adam_lbfgs", "L2_rel"),
     jget(l5_n3,  "level5_adam_lbfgs", "MAE_C",    fmt=".1f"),
     jget(l5_n3,  "level5_adam_lbfgs", "MaxErr_C", fmt=".1f"),
     "11,776"],
]

tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center", bbox=[0, 0.12, 1, 0.80])
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.6)

ROW_COLORS = ["#DCEEFA", "#C8E6C9", "#FDDEDE", "#FFF9C4"]
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#37474F")
    tbl[0, j].set_text_props(color="white", fontweight="bold", fontsize=11)
for i, rc in enumerate(ROW_COLORS):
    for j in range(len(col_labels)):
        tbl[i+1, j].set_facecolor(rc)
        tbl[i+1, j].set_text_props(color=TXT_C, fontsize=11)

# Highlight best L2
for i, row in enumerate(rows):
    try:
        if float(row[2]) < 0.05:
            tbl[i+1, 2].set_facecolor("#A5D6A7")
            tbl[i+1, 2].set_text_props(fontweight="bold", color="#1B5E20", fontsize=11)
        elif float(row[2]) < 0.10:
            tbl[i+1, 2].set_facecolor("#FFF176")
    except Exception:
        pass

ax_tbl.set_title("Summary: PINN Error Metrics vs FEM Reference",
                 fontsize=13, color=TXT_C, pad=12, fontweight="bold")

fig3.tight_layout(pad=2.5)
fig3.savefig(OUT / "fig3_error_analysis.png", dpi=150,
             bbox_inches="tight", facecolor=FIG_BG)
plt.close(fig3)
print("  Saved: fig3_error_analysis.png")
print("\nDone.")
