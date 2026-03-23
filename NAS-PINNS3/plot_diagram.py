"""
plot_diagram.py — NAS-MCO-PINN unified project overview diagram
Single figure, left-to-right flow, all components integrated.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(26, 16))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 26)
ax.set_ylim(0, 16)
ax.axis("off")

# ── Palette ────────────────────────────────────────────────────
C_PHYS  = "#0D47A1"   # physics / problem
C_NAS   = "#1B5E20"   # NAS search
C_NN    = "#E65100"   # neural network
C_AD    = "#6A1B9A"   # auto-diff
C_LOSS  = "#B71C1C"   # loss
C_MCO   = "#4A148C"   # MCO
C_OPT   = "#004D40"   # optimizer
C_SKIP  = "#01579B"   # skip operator
C_LEVEL = "#37474F"   # level boxes
C_IN    = "#1565C0"
C_HID   = "#F9A825"
C_OUT_N = "#E65100"

# ── Helpers ────────────────────────────────────────────────────
def box(x, y, w, h, fc, ec, lw=1.8, r="round,pad=0.08", alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=r,
                 fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=2))

def t(x, y, s, fs=8, fw="normal", c="#111", ha="center", va="center", st="normal"):
    ax.text(x, y, s, fontsize=fs, fontweight=fw, color=c,
            ha=ha, va=va, style=st, zorder=5, fontfamily="DejaVu Sans")

def arr(x1, y1, x2, y2, color="#333", lw=2.0, hw=0.28, hl=0.35, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=f"-|>,head_width={hw},head_length={hl}",
                    color=color, lw=lw,
                    connectionstyle=f"arc3,rad={rad}"), zorder=3)

def circ(cx, cy, r, fc, ec="white", lw=1.2):
    ax.add_patch(plt.Circle((cx, cy), r, fc=fc, ec=ec, lw=lw, zorder=4))

def badge(x, y, w, h, label, fc, fs=8.5):
    """Colored left-strip badge box."""
    box(x, y, w, h, fc=fc+"22", ec=fc, lw=2.0)
    box(x, y, 0.55, h, fc=fc, ec=fc, lw=0, r="square,pad=0")
    t(x+0.275, y+h/2, label, fs=fs, fw="bold", c="white")


# ══════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════
t(13, 15.5, "NAS-MCO-PINN: Neural Architecture Search for Thermal Time-Step Skip Operator",
  fs=16, fw="bold", c="#0D47A1")
t(13, 15.0, "A356 Aluminum Water Quenching  ·  3D Domains  ·  FEM Step Replacement",
  fs=10, c="#546E7A")

# ══════════════════════════════════════════════════════════════
#  ROW 1 — Main flow boxes (y=9.5 → 13.5)
# ══════════════════════════════════════════════════════════════
FLOW_Y  = 9.8
FLOW_H  = 3.5
FLOW_YC = FLOW_Y + FLOW_H / 2  # vertical center

# Box widths and x-starts
BW = [3.6, 3.6, 4.2, 2.2, 4.6]   # widths
BX = [0.3, 4.3, 8.3, 12.9, 15.5] # x-starts

# ── [1] Physical Problem ──────────────────────────────────────
box(BX[0], FLOW_Y, BW[0], FLOW_H, "#E3F2FD", C_PHYS)
t(BX[0]+BW[0]/2, FLOW_Y+FLOW_H-0.38, "Physical Problem", fs=10, fw="bold", c=C_PHYS)
t(BX[0]+BW[0]/2, FLOW_Y+FLOW_H-0.82, "A356 Aluminum Quenching", fs=8.5, c=C_PHYS)

lines_phys = [
    "Domain: Rectangular / Cylinder",
    "         Stacked Cubes (3D)",
    "IC: T₀ = 540 °C (uniform)",
    "BC: Robin  (h = 4000 W/m²K)",
    "FEM: 21 steps, Δt = 1.5 s",
    "Total: 0 → 30 s quenching",
]
for i, ln in enumerate(lines_phys):
    t(BX[0]+BW[0]/2, FLOW_Y+FLOW_H-1.35-i*0.37, ln, fs=7.5, c="#37474F")

# ── [2] NAS Architecture Search ───────────────────────────────
box(BX[1], FLOW_Y, BW[1], FLOW_H, "#E8F5E9", C_NAS)
t(BX[1]+BW[1]/2, FLOW_Y+FLOW_H-0.38, "NAS Search", fs=10, fw="bold", c=C_NAS)
t(BX[1]+BW[1]/2, FLOW_Y+FLOW_H-0.80, "Architecture Optimization", fs=8.5, c=C_NAS)

nas_lines = [
    "Bayesian (TPE):",
    "  5 layers × 151 neurons, ReLU",
    "NSGA-II:",
    "  3 layers × 153 neurons, Tanh",
    "NSGA-III:",
    "  3 layers × 75 neurons,  Tanh",
]
for i, ln in enumerate(nas_lines):
    fw_ = "bold" if ":" in ln and not ln.startswith(" ") else "normal"
    t(BX[1]+BW[1]/2, FLOW_Y+FLOW_H-1.30-i*0.37, ln, fs=7.5, c="#1B5E20", fw=fw_)

# ── [3] NAS-PINN Network ──────────────────────────────────────
box(BX[2], FLOW_Y, BW[2], FLOW_H, "#FFF3E0", C_NN)
t(BX[2]+BW[2]/2, FLOW_Y+FLOW_H-0.38, "NAS-PINN  NN(w, b)", fs=10, fw="bold", c=C_NN)
t(BX[2]+BW[2]/2, FLOW_Y+FLOW_H-0.78, "Residual Skip Network", fs=8.5, c=C_NN)

# Mini network drawing inside box
in_labels = ["x̄", "ȳ", "z̄", "τ", "θᵢc"]
in_y      = [FLOW_Y+2.7, FLOW_Y+2.15, FLOW_Y+1.6, FLOW_Y+1.05, FLOW_Y+0.5]
NX_IN = BX[2] + 0.55
NX_H1 = BX[2] + 1.55
NX_H2 = BX[2] + 2.50
NX_OUT= BX[2] + 3.55

for lbl, iy in zip(in_labels, in_y):
    circ(NX_IN, iy, 0.23, fc=C_IN, ec="white", lw=1.0)
    t(NX_IN, iy, lbl, fs=8, fw="bold", c="white")

hid_y = [FLOW_Y+2.7, FLOW_Y+2.05, FLOW_Y+1.4, FLOW_Y+0.75]
for hy in hid_y:
    circ(NX_H1, hy, 0.21, fc=C_HID, ec="white")
    t(NX_H1, hy, "σ", fs=8, fw="bold", c="#4E342E")
    circ(NX_H2, hy, 0.21, fc=C_HID, ec="white")
    t(NX_H2, hy, "σ", fs=8, fw="bold", c="#4E342E")

# Connections
for iy in in_y:
    for hy in hid_y:
        ax.plot([NX_IN+0.23, NX_H1-0.21], [iy, hy], "-",
                color="#FFB74D", lw=0.4, alpha=0.5, zorder=1)
for h1y in hid_y:
    for h2y in hid_y:
        ax.plot([NX_H1+0.21, NX_H2-0.21], [h1y, h2y], "-",
                color="#FFB74D", lw=0.4, alpha=0.5, zorder=1)

circ(NX_OUT, FLOW_Y+1.72, 0.28, fc=C_OUT_N, ec="white", lw=1.2)
t(NX_OUT, FLOW_Y+1.72, "θ̂", fs=9, fw="bold", c="white")

for hy in hid_y:
    ax.plot([NX_H2+0.21, NX_OUT-0.28], [hy, FLOW_Y+1.72], "-",
            color="#FFCCBC", lw=0.5, alpha=0.6, zorder=1)

t(BX[2]+BW[2]/2, FLOW_Y+0.20,
  "θ̂_next = θ_ic + τ · correction  [IC auto-satisfied at τ=0]",
  fs=7.5, c="#37474F")

# ── [4] Auto-Diff ─────────────────────────────────────────────
box(BX[3], FLOW_Y, BW[3], FLOW_H, "#F3E5F5", C_AD)
t(BX[3]+BW[3]/2, FLOW_Y+FLOW_H-0.38, "AD", fs=10, fw="bold", c=C_AD)
t(BX[3]+BW[3]/2, FLOW_Y+FLOW_H-0.78, "Auto-Differentiation", fs=8, c=C_AD)

ad_exprs = ["∂θ̂/∂τ", "∂²θ̂/∂ξ²", "∂²θ̂/∂η²", "∂²θ̂/∂ζ²"]
ad_ys    = [FLOW_Y+2.4, FLOW_Y+1.7, FLOW_Y+1.0, FLOW_Y+0.3]
for expr, ay in zip(ad_exprs, ad_ys):
    circ(BX[3]+BW[3]/2, ay, 0.28, fc="#AB47BC", ec="white")
    t(BX[3]+BW[3]/2, ay, expr, fs=8, fw="bold", c="white")

# ── [5] Loss + MCO ────────────────────────────────────────────
box(BX[4], FLOW_Y, BW[4], FLOW_H, "#FFEBEE", C_LOSS)
t(BX[4]+BW[4]/2, FLOW_Y+FLOW_H-0.36, "Loss Functions + MCO", fs=10, fw="bold", c=C_LOSS)

loss_lines = [
    ("L_target  =  ||θ̂(τ=1) − θ_exact(t_{k+s})||²", "#B71C1C"),
    ("       Target supervision (full field match)", "#78909C"),
    ("L_PDE    =  ||∂τθ̂ − Fo∇²θ̂||²", "#1565C0"),
    ("       3D heat equation residual (w=0.05)", "#78909C"),
    ("MCO: w_i = 1/σᵢ²  (Gaussian MLE, adaptive)", "#4A148C"),
    ("L_total = L_target + 0.05·L_PDE", "#E65100"),
]
for i, (ln, lc) in enumerate(loss_lines):
    fw_ = "bold" if i in [0, 2, 4, 5] else "normal"
    t(BX[4]+BW[4]/2, FLOW_Y+FLOW_H-0.90-i*0.42, ln, fs=7.5, fw=fw_, c=lc)

# Flow arrows between main boxes
arr(BX[0]+BW[0], FLOW_YC, BX[1], FLOW_YC, C_PHYS)
arr(BX[1]+BW[1], FLOW_YC, BX[2], FLOW_YC, C_NAS)
arr(BX[2]+BW[2], FLOW_YC, BX[3], FLOW_YC, C_NN)
arr(BX[3]+BW[3], FLOW_YC, BX[4], FLOW_YC, C_AD)

# ══════════════════════════════════════════════════════════════
#  ROW 2 — Optimization loop (y=7.2 → 9.4)
# ══════════════════════════════════════════════════════════════
OPT_Y = 7.4

# Total Loss → Minimize
box(15.7, OPT_Y, 3.6, 1.75, "#FFEE58", "#F57F17", lw=1.8)
t(17.5, OPT_Y+1.28, "Total Loss", fs=9.5, fw="bold", c="#E65100")
t(17.5, OPT_Y+0.82, "MCO-weighted", fs=8.5, c="#37474F")
t(17.5, OPT_Y+0.45, "combination", fs=8.5, c="#37474F")

# Arrow from Loss box down to Total Loss
arr(17.8, FLOW_Y, 17.8, OPT_Y+1.75, C_LOSS, lw=1.8)

# Minimize arrow
arr(15.7, OPT_Y+0.87, 13.9, OPT_Y+0.87, "#E65100", lw=1.8)
t(14.8, OPT_Y+1.10, "Minimize", fs=8.5, fw="bold", c="#E65100")

# Diamond
DX, DY = 13.2, OPT_Y + 0.87
dpoly = plt.Polygon([[DX, DY+0.65],[DX+1.0, DY],[DX, DY-0.65],[DX-1.0, DY]],
                    closed=True, fc="#FFF9C4", ec="#F57F17", lw=1.8, zorder=3)
ax.add_patch(dpoly)
t(DX, DY+0.22, "L < ε ?", fs=9, fw="bold", c="#E65100")
t(DX, DY-0.22, "converged?", fs=8, c="#37474F")

# Yes → Stop
arr(DX, DY-0.65, DX, OPT_Y-1.0, C_OPT, lw=1.8)
t(DX+0.5, OPT_Y-0.15, "Yes", fs=8.5, fw="bold", c=C_OPT)
box(DX-0.9, OPT_Y-1.65, 1.8, 0.60, "#C8E6C9", C_OPT, lw=1.8)
t(DX, OPT_Y-1.35, "STOP  →  Save Model", fs=9, fw="bold", c=C_OPT)

# No → NAS Optimizer
arr(DX-1.0, DY, 10.2, DY, C_NAS, lw=1.8)
t(11.5, DY+0.28, "No  →  Update weights", fs=8.5, fw="bold", c=C_NAS)

box(8.5, OPT_Y+0.25, 1.65, 1.25, "#E8F5E9", C_NAS, lw=1.6)
t(9.32, OPT_Y+1.12, "NAS", fs=9, fw="bold", c=C_NAS)
t(9.32, OPT_Y+0.78, "Optimizer", fs=8.5, fw="bold", c=C_NAS)
t(9.32, OPT_Y+0.45, "Adam + CosineLR", fs=7.5, c="#546E7A")

arr(8.5, OPT_Y+0.87, 6.7, OPT_Y+0.87, C_NAS, lw=1.8)
t(7.6, OPT_Y+1.12, "Update NN(w,b)", fs=8.5, fw="bold", c=C_NAS)

# Arc back up to NN box
ax.annotate("", xy=(10.4, FLOW_Y), xytext=(6.2, OPT_Y+0.87),
            arrowprops=dict(arrowstyle="-|>", color=C_NAS, lw=1.8,
                            connectionstyle="arc3,rad=-0.3"), zorder=3)

# ══════════════════════════════════════════════════════════════
#  ROW 3 — Level Progression strip (y=3.8 → 6.8)
# ══════════════════════════════════════════════════════════════
t(13, 6.55, "Project Level Progression  (L1 → L8)", fs=10, fw="bold", c=C_LEVEL)

LEVELS = [
    (1,   "#1565C0", "L1\nNAS Search",
     "Single-shot PINN\nBayesian/NSGA-II/NSGA-III"),
    (2,   "#00838F", "L2\nSkip Op.",
     "Time-step skip\nskip=1,2,4,6"),
    (3,   "#2E7D32", "L3\nHybrid",
     "FEM anchor\n+ PINN inter."),
    (4,   "#6A1B9A", "L4\nDistort.",
     "T→mechanical\ndisp. u(x,y)"),
    (5,   "#E65100", "L5\nRefine",
     "L-BFGS training\nStrong Wolfe"),
    (6,   "#C62828", "L6\nPoisson",
     "Aux Poisson\nloss fine-tune"),
    ("7A+B","#4527A0","L7\nMulti-D",
     "Temporal skip\nMulti-domain"),
    (8,   "#1A237E", "L8  ★\nMCO-PINN",
     "3D MCO adaptive\nskip=2,4 · 3 dom."),
]

LW = 2.9
LX0 = 0.25
LY0 = 4.0
LH  = 2.4

for i, (lvl, ec, badge_lbl, desc) in enumerate(LEVELS):
    x0 = LX0 + i * (LW + 0.25)
    fc = ec + "18"
    box(x0, LY0, LW, LH, fc=fc, ec=ec, lw=2.2)
    # top badge
    box(x0, LY0+LH-0.72, LW, 0.72, fc=ec, ec=ec, lw=0, r="square,pad=0")
    t(x0+LW/2, LY0+LH-0.36, badge_lbl, fs=8.5, fw="bold", c="white")
    # description
    for j, line in enumerate(desc.split("\n")):
        t(x0+LW/2, LY0+LH-1.10-j*0.48, line, fs=8, c=ec, fw="bold" if j==0 else "normal")

    # Arrow to next level
    if i < len(LEVELS)-1:
        arr(x0+LW, LY0+LH/2, x0+LW+0.25, LY0+LH/2,
            color=ec, lw=1.8, hw=0.18, hl=0.22)

# Arrow from L8 up to main flow
arr(LX0 + 7*(LW+0.25)+LW/2, LY0+LH,
    LX0 + 7*(LW+0.25)+LW/2, OPT_Y,
    color="#1A237E", lw=1.8, rad=0.0)
t(LX0+7*(LW+0.25)+LW/2+1.0, OPT_Y-0.6,
  "Level 8\nfinal model", fs=8, fw="bold", c="#1A237E", ha="left")

# ══════════════════════════════════════════════════════════════
#  ROW 4 — Skip Operator visual (y=0.3 → 3.5)
# ══════════════════════════════════════════════════════════════
box(0.25, 0.3, 25.5, 3.2, "#E8F5E9", C_SKIP, lw=2.0)
t(13, 3.35, "Time-Step Skip Operator  —  PINN Replaces FEM at Skipped Steps",
  fs=10, fw="bold", c=C_SKIP)

# Timeline
for skip, cy, label in [(2, 2.25, "skip = 2  (52% FEM steps saved)"),
                         (4, 1.05, "skip = 4  (71% FEM steps saved)")]:
    fem_set = set(range(0, 21, skip)) | {20}
    t(0.9, cy+0.50, label, fs=8.5, fw="bold", c=C_SKIP)

    for i in range(21):
        cx = 2.2 + i * 1.13
        is_fem = i in fem_set
        fc_c = C_IN if is_fem else C_OUT_N
        circ(cx, cy, 0.21, fc=fc_c, ec="white", lw=0.8)
        if is_fem:
            t(cx, cy, "●", fs=7, c="white")
        else:
            t(cx, cy, "▲", fs=7, c="white")
        if i % 2 == 0:
            t(cx, cy-0.38, f"t{i}", fs=6, c=fc_c)

    # Bracket for one skip window
    if skip == 2:
        ax.annotate("", xy=(2.2+2*1.13, cy+0.52), xytext=(2.2, cy+0.52),
                    arrowprops=dict(arrowstyle="-|>", color=C_OUT_N, lw=2.0,
                                    connectionstyle="arc3,rad=-0.4"), zorder=3)
        t(2.2+1.13, cy+0.88, "1 step skipped", fs=7.5, fw="bold", c=C_OUT_N)

# Legend
box(20.8, 0.45, 4.6, 2.7, "#F5F5F5", "#90A4AE", lw=1.2)
t(23.1, 2.9, "Legend", fs=9.5, fw="bold", c="#37474F")
legend_items = [
    ("●", C_IN,     "FEM step — computed"),
    ("▲", C_OUT_N,  "PINN step — predicted"),
    ("σ", C_HID,    "Hidden neuron (NAS-opt.)"),
    ("AD", C_AD,    "Automatic differentiation"),
    ("★", "#1A237E","Level 8 — Final model"),
]
for ii, (mk, lc, lbl) in enumerate(legend_items):
    circ(21.35, 2.52-ii*0.48, 0.18, fc=lc, ec="white", lw=0.8)
    t(21.35, 2.52-ii*0.48, mk, fs=7, c="white")
    t(23.5, 2.52-ii*0.48, lbl, fs=8, c="#37474F", ha="left")

plt.tight_layout(pad=0.3)
plt.savefig(os.path.join(OUT, "fig0_architecture_diagram.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("  [OK] fig0_architecture_diagram.png")
