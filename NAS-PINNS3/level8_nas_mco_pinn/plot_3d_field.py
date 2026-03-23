"""
plot_3d_field.py  —  Professional 3D Field Visualizations
==========================================================
FEM (analytical series solution) vs PINN skip operator
T(x,y,t) 2D field comparison — all optimizers

Domain 1: Rectangular (1.3 m × 0.6 m) — Mortensen 2026 casting
Domain 2: L-shaped (automotive subframe corner cross-section)

Run:
    cd /home/coder/NAS-PINNS1/NAS-PINNS3
    python -m level8_nas_mco_pinn.plot_3d_field
"""

import os, sys, time, warnings
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.optim as optim

warnings.filterwarnings("ignore")

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.config import DEVICE
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .mco_timestepper import (
    MCOLoss, build_net, train_window_mco,
    T_INIT, T_WATER, T_SPAN, ALPHA, K, RHO_CP, H_CONV, LX, LY, T_END,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

ARCH_LABEL = {"bayesian": "Bayesian (TPE)", "nsga2": "NSGA-II", "nsga3": "NSGA-III"}
ARCH_COLOR = {"bayesian": "#1565C0", "nsga2": "#2E7D32", "nsga3": "#E65100"}
ARCHS      = list(ARCH_LABEL.keys())

ARCH_CONFIGS = {
    "bayesian": {"n_layers": 5, "neurons": [151]*5, "activation": "relu"},
    "nsga2":    {"n_layers": 3, "neurons": [153]*3, "activation": "tanh"},
    "nsga3":    {"n_layers": 3, "neurons": [75]*3,  "activation": "tanh"},
}

N_GRID   = 60    # grid resolution
N_ADAM   = 1200  # training steps (reduced for speed)
CMAP_T   = "plasma"   # temperature colormap
CMAP_E   = "Reds"     # error colormap
FS_TITLE = 11
FS_LABEL = 9


# ══════════════════════════════════════════════════════════════
# 1. Analytical 2D Series Solution  (FEM reference)
# ══════════════════════════════════════════════════════════════

class AnalyticalField2D:
    """
    2D heat equation on rectangular domain with Robin BCs.
    N-term eigenfunction series:
      T(x,y,t) = T_w + (T₀-T_w) Σ_n Σ_m A_nm X_n(x) Y_m(y) exp(-λ_nm t)

    Large Biot number (Bi_x≈32, Bi_y≈15) → fast surface cooling,
    center stays much hotter → significant spatial gradient.
    """

    def __init__(self, Lx=LX, Ly=LY, n_terms=6):
        self.Lx, self.Ly = Lx, Ly
        self.n_terms = n_terms
        Bi_x = H_CONV * Lx / K          # ~32.5
        Bi_y = H_CONV * Ly / K          # ~15.0

        self.mu_x, self.beta_x, self.C_x = self._eigen(Bi_x, Lx, n_terms)
        self.mu_y, self.beta_y, self.C_y = self._eigen(Bi_y, Ly, n_terms)
        alpha_diff = K / RHO_CP

        # λ_nm = α(β_x_n² + β_y_m²)
        self.Lambda = np.array([
            [alpha_diff * (self.beta_x[n]**2 + self.beta_y[m]**2)
             for m in range(n_terms)]
            for n in range(n_terms)
        ])

    @staticmethod
    def _eigen(Bi, L, n):
        """First n eigenvalues and coefficients for Robin BC on [-L/2, L/2]."""
        Bi_half = Bi / 2
        mu_list, beta_list, C_list = [], [], []
        for k in range(n):
            lo = k * np.pi + 1e-8
            hi = (k + 0.5) * np.pi - 1e-8
            try:
                mu = brentq(lambda m: m*np.tan(m) - Bi_half, lo, hi)
            except Exception:
                mu = (lo + hi) / 2
            beta = 2 * mu / L
            s, c = np.sin(mu), np.cos(mu)
            C = 2*s / (mu + s*c)
            mu_list.append(mu); beta_list.append(beta); C_list.append(C)
        return np.array(mu_list), np.array(beta_list), np.array(C_list)

    def T(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """T(x,y,t) — scalar t, arrays x,y."""
        shape = x.shape
        x, y = x.ravel(), y.ravel()
        cx = (self.Lx / 2)
        cy = (self.Ly / 2)
        result = np.zeros(len(x))
        for n in range(self.n_terms):
            Xn = self.C_x[n] * np.cos(self.beta_x[n] * (x - cx))
            for m in range(self.n_terms):
                Ym = self.C_y[m] * np.cos(self.beta_y[m] * (y - cy))
                result += Xn * Ym * np.exp(-self.Lambda[n, m] * t)
        T_field = T_WATER + T_SPAN * result
        # t=0 result may deviate slightly from T_0 due to Cx×Cy truncation
        T_field = np.clip(T_field, T_WATER, T_INIT + 10)
        return T_field.reshape(shape)


# ══════════════════════════════════════════════════════════════
# 2. PINN Training & Field Evaluation
# ══════════════════════════════════════════════════════════════

def train_and_predict_field(
    arch_name: str,
    t_start: float, t_end: float,
    anal: AnalyticalField2D,
    xx: np.ndarray, yy: np.ndarray,
    n_adam: int = N_ADAM,
) -> np.ndarray:
    """
    Train on window [t_start → t_end], predict T(x,y) on dense grid.
    Analytical T(x,y,t_start) is used as IC.
    """
    config = ARCH_CONFIGS[arch_name]
    net    = build_net(config).to(DEVICE)
    mco    = MCOLoss(n=3).to(DEVICE)

    theta_ref = np.exp(-ALPHA * t_start)   # IC lumped θ

    train_window_mco(
        net, mco,
        t_start=t_start, t_end=t_end,
        theta_ref=theta_ref,
        n_domain=800, n_bc=200, n_adam=n_adam, lr=1e-3,
        device=DEVICE,
    )

    # Dense grid evaluation
    dt = t_end - t_start
    xf, yf = xx.ravel(), yy.ravel()
    T_prev_analytical = anal.T(xf, yf, t_start)
    theta_prev        = (T_prev_analytical - T_WATER) / T_SPAN

    inp = torch.tensor(
        np.stack([xf/LX, yf/LY,
                  np.ones(len(xf)),        # t_local=1 (window end)
                  theta_prev], axis=1),
        dtype=torch.float32, device=DEVICE
    )
    with torch.no_grad():
        theta_pred = net(inp).cpu().numpy().squeeze()

    T_pred = T_WATER + np.clip(theta_pred, -0.2, 1.3) * T_SPAN
    return T_pred.reshape(xx.shape)


# ══════════════════════════════════════════════════════════════
# 3. Domain Masks
# ══════════════════════════════════════════════════════════════

def rect_mask(xx, yy):
    """Rectangular domain — all points valid."""
    return np.ones(xx.shape, dtype=bool)

def lshape_mask(xx, yy, Lx=LX, Ly=LY):
    """L-shaped domain: upper-right quadrant of rectangle removed."""
    cut_x = Lx * 0.55
    cut_y = Ly * 0.50
    return ~((xx > cut_x) & (yy > cut_y))


# ══════════════════════════════════════════════════════════════
# 4. Plot Utilities
# ══════════════════════════════════════════════════════════════

def _apply_mask(arr, mask):
    out = arr.copy().astype(float)
    out[~mask] = np.nan
    return out

def _colorbar(fig, ax, im, label, fontsize=8):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.92)
    cb.set_label(label, fontsize=fontsize)
    cb.ax.tick_params(labelsize=7)
    return cb

def _ax_style(ax, xlabel="x [m]", ylabel="y [m]"):
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")


# ══════════════════════════════════════════════════════════════
# 5. Figure A: 3D Surface — FEM vs PINN (single optimizer, two skips)
# ══════════════════════════════════════════════════════════════

def fig_3d_surface(anal, xx, yy, mask, domain_name, t_snap=15.0, arch="bayesian"):
    """
    3D surface: FEM and PINN skip=2, skip=4 side by side.
    """
    fig = plt.figure(figsize=(18, 5.5))
    fig.suptitle(
        f"3D Temperature Field — {domain_name}   t = {t_snap:.0f} s\n"
        f"FEM (analytical series) vs NAS-MCO-PINN {ARCH_LABEL[arch]} — Skip Operator",
        fontsize=12, fontweight="bold", y=1.01,
    )

    vmin, vmax = T_WATER, T_INIT
    norm_T = Normalize(vmin=vmin, vmax=vmax)
    cmap_T = plt.get_cmap(CMAP_T)

    T_fem = _apply_mask(anal.T(xx, yy, t_snap), mask)

    # skip=2: t=0→7.5→15s (2 windows × 7.5s)
    T_skip2 = _apply_mask(
        train_and_predict_field(arch, 7.5, t_snap, anal, xx, yy), mask
    )
    # skip=4: t=0→15s single window
    T_skip4 = _apply_mask(
        train_and_predict_field(arch, 0.0, t_snap, anal, xx, yy), mask
    )

    panels = [
        ("FEM Reference",              T_fem,   cmap_T, norm_T),
        (f"PINN skip=2\n(11/21 FEM steps)", T_skip2, cmap_T, norm_T),
        (f"PINN skip=4\n(6/21 FEM steps)",  T_skip4, cmap_T, norm_T),
    ]

    for i, (title, Tdata, cmap, norm) in enumerate(panels):
        ax = fig.add_subplot(1, 4, i+1, projection="3d")
        xm = np.where(mask, xx, np.nan)
        ym = np.where(mask, yy, np.nan)
        zm = np.where(mask, Tdata, np.nan)
        fc = cmap(norm(zm))
        ax.plot_surface(xm, ym, zm, facecolors=fc,
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=True, alpha=0.95, shade=True)
        ax.set_xlabel("x [m]", fontsize=8, labelpad=2)
        ax.set_ylabel("y [m]", fontsize=8, labelpad=2)
        ax.set_zlabel("T [°C]",  fontsize=8, labelpad=2)
        ax.set_zlim(vmin, vmax)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=28, azim=-60)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    # Error panel
    ax_e = fig.add_subplot(1, 4, 4)
    err2 = np.abs(T_fem - T_skip2)
    err4 = np.abs(T_fem - T_skip4)
    err_max = max(np.nanpercentile(err2, 95), np.nanpercentile(err4, 95), 1)
    norm_e = Normalize(vmin=0, vmax=err_max)
    im_e = ax_e.pcolormesh(xx, yy, _apply_mask(err4, mask),
                           cmap=CMAP_E, norm=norm_e, shading="auto")
    ax_e.contour(xx, yy, _apply_mask(err4, mask), levels=5,
                 colors="black", linewidths=0.4, alpha=0.5)
    _ax_style(ax_e)
    ax_e.set_title(f"|FEM − PINN skip=4|\nMAE={np.nanmean(err4):.1f}°C",
                   fontsize=10, fontweight="bold", pad=6)
    _colorbar(fig, ax_e, im_e, "|ΔT| [°C]")

    fig.tight_layout()
    p = os.path.join(OUT, f"field_3d_{domain_name}_{arch}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(p)}")
    return p


# ══════════════════════════════════════════════════════════════
# 6. Figure B: 2D Field Comparison — 3 optimizers × time
# ══════════════════════════════════════════════════════════════

def fig_2d_field_comparison(anal, xx, yy, mask, domain_name,
                             skip=2, t_snaps=(5.0, 15.0, 25.0)):
    """
    Row = FEM / Bayesian / NSGA-II / NSGA-III / Error
    Column = different time snapshots
    Shows which FEM steps are skipped for each skip value.
    """
    n_col = len(t_snaps)
    n_row = 5  # FEM + 3 arch + error (skip4)

    fig, axes = plt.subplots(n_row, n_col, figsize=(5*n_col, 3.5*n_row))
    fig.suptitle(
        f"FEM vs NAS-MCO-PINN Skip={skip} — {domain_name}\n"
        f"Row 1: FEM analytical | Rows 2–4: PINN optimizer | Row 5: |Error| (Bayesian)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # Color scale for all T values
    all_T = [anal.T(xx, yy, t) for t in t_snaps]
    vmin_T = T_WATER
    vmax_T = T_INIT
    norm_T = Normalize(vmin=vmin_T, vmax=vmax_T)

    # Error color scale
    err_vals = []
    pinn_fields = {arch: [] for arch in ARCHS}

    print(f"  Training PINN fields (skip={skip})...")
    for ti, t in enumerate(t_snaps):
        t_start = t - skip * 1.5 if t >= skip*1.5 else 0.0
        for arch in ARCHS:
            Tp = train_and_predict_field(arch, t_start, t, anal, xx, yy)
            pinn_fields[arch].append(_apply_mask(Tp, mask))
        T_ref = _apply_mask(anal.T(xx, yy, t), mask)
        err_vals.append(np.abs(T_ref - pinn_fields["bayesian"][ti]))

    err_max = max(np.nanpercentile(e, 96) for e in err_vals)
    err_max = max(err_max, 1.0)
    norm_e  = Normalize(vmin=0, vmax=err_max)

    row_labels = ["FEM (analytical series)"] + \
                 [f"PINN {ARCH_LABEL[a]}" for a in ARCHS] + \
                 [f"|FEM − PINN Bayesian|\nskip={skip}"]

    for ri in range(n_row):
        for ci, t in enumerate(t_snaps):
            ax = axes[ri, ci]
            ax.set_facecolor("#F0F0F0")

            if ri == 0:
                data = _apply_mask(anal.T(xx, yy, t), mask)
                im   = ax.pcolormesh(xx, yy, data, cmap=CMAP_T,
                                     norm=norm_T, shading="auto")
                ax.contour(xx, yy, data, levels=6,
                           colors="white", linewidths=0.5, alpha=0.6)
            elif ri in (1, 2, 3):
                arch = ARCHS[ri - 1]
                data = pinn_fields[arch][ci]
                im   = ax.pcolormesh(xx, yy, data, cmap=CMAP_T,
                                     norm=norm_T, shading="auto")
                mae  = np.nanmean(np.abs(data - _apply_mask(anal.T(xx,yy,t), mask)))
                ax.set_title(f"MAE = {mae:.1f}°C", fontsize=8, pad=2, color="#333")
                ax.contour(xx, yy, data, levels=6,
                           colors="white", linewidths=0.4, alpha=0.5)
            else:  # error row
                data = err_vals[ci]
                im   = ax.pcolormesh(xx, yy, data, cmap=CMAP_E,
                                     norm=norm_e, shading="auto")
                ax.contour(xx, yy, data, levels=4,
                           colors="black", linewidths=0.4, alpha=0.5)

            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)

            if ci == 0:
                ax.set_ylabel(row_labels[ri], fontsize=9, fontweight="bold")
            if ri == 0:
                dt_fem = skip * 1.5
                ax.set_title(f"t = {t:.0f} s\n(skip={skip}: {dt_fem:.0f}s skipped)",
                             fontsize=9, fontweight="bold")
            if ri == n_row - 1:
                ax.set_xlabel("x [m]", fontsize=8)

            # Colorbar only on right column
            if ci == n_col - 1:
                unit = "T [°C]" if ri < 4 else "|ΔT| [°C]"
                _im   = im
                _norm = norm_T if ri < 4 else norm_e
                _colorbar(fig, ax, _im, unit)

    plt.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.05,
                        hspace=0.10, wspace=0.06)
    p = os.path.join(OUT, f"field_2d_{domain_name}_skip{skip}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(p)}")
    return p


# ══════════════════════════════════════════════════════════════
# 7. Figure C: FEM Step Skip — Time Series Visualization
# ══════════════════════════════════════════════════════════════

def fig_skip_timeline(anal, xx, yy, mask, domain_name, arch="bayesian"):
    """
    Rows = time steps (t=0, 1.5, 3, 4.5, 6, 7.5, 9)
    Cols = FEM | PINN skip=2 | |Error| s2 | PINN skip=4 | |Error| s4

    At every PINN prediction time, FEM result is in the same row for
    direct visual comparison. SKIPPED panels show FEM field (greyed) so
    the skipped information is still visible.
    """
    t_rows  = [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0]

    s2_anch = {0.0, 3.0, 6.0, 9.0}   # skip=2 anchors (dt=3s)
    s4_anch = {0.0, 6.0}              # skip=4 anchors (dt=6s)

    n_row = len(t_rows)
    n_col = 5   # FEM | PINN-s2 | err-s2 | PINN-s4 | err-s4

    fig, axes = plt.subplots(n_row, n_col, figsize=(3.5*n_col, 2.8*n_row))
    fig.suptitle(
        f"Time-Step Skip Operator — {domain_name}  (arch: {ARCH_LABEL[arch]})\n"
        "Each row = one time instant  |  FEM always shown for direct comparison",
        fontsize=12, fontweight="bold", y=1.01,
    )

    norm_T = Normalize(vmin=T_WATER, vmax=T_INIT)

    # ── Train PINN windows ────────────────────────────────────
    print(f"    Training skip=2 windows for timeline...")
    s2_pinn = {}
    for t_s, t_e in [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]:
        s2_pinn[t_e] = _apply_mask(
            train_and_predict_field(arch, t_s, t_e, anal, xx, yy), mask)

    print(f"    Training skip=4 window for timeline...")
    s4_pinn = {6.0: _apply_mask(
        train_and_predict_field(arch, 0.0, 6.0, anal, xx, yy), mask)}

    # shared error colour scale
    all_e = []
    for t_e, Tp in {**s2_pinn, **s4_pinn}.items():
        Tref = _apply_mask(anal.T(xx, yy, t_e), mask)
        all_e.append(np.nanpercentile(np.abs(Tref - Tp), 96))
    err_max = max(max(all_e), 1.0)
    norm_e  = Normalize(vmin=0, vmax=err_max)

    col_titles = [
        "FEM  (analytical ref.)",
        "PINN skip=2\n(dt = 3 s, skips 1 step)",
        "|FEM − PINN|  skip=2",
        "PINN skip=4\n(dt = 6 s, skips 3 steps)",
        "|FEM − PINN|  skip=4",
    ]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=8, fontweight="bold", pad=4)

    for ri, t in enumerate(t_rows):
        T_fem = _apply_mask(anal.T(xx, yy, t), mask)

        for ci in range(n_col):
            ax = axes[ri, ci]
            ax.set_aspect("equal")
            ax.tick_params(labelsize=5)
            im = None

            # ── Col 0: FEM (always data) ──────────────────
            if ci == 0:
                im = ax.pcolormesh(xx, yy, T_fem, cmap=CMAP_T,
                                   norm=norm_T, shading="auto")
                ax.contour(xx, yy, T_fem, levels=5,
                           colors="white", linewidths=0.4, alpha=0.6)
                ax.set_ylabel(f"t = {t:.1f} s", fontsize=9, fontweight="bold")

            # ── Col 1: PINN skip=2 ────────────────────────
            elif ci == 1:
                if t == 0.0:
                    im = ax.pcolormesh(xx, yy, T_fem, cmap=CMAP_T,
                                       norm=norm_T, shading="auto")
                    ax.set_title("IC (same as FEM)", fontsize=7,
                                 color="#1565C0", pad=2)
                elif t in s2_anch:
                    Tp  = s2_pinn[t]
                    mae = np.nanmean(np.abs(T_fem - Tp))
                    T_fem_mean = np.nanmean(T_fem)
                    T_pinn_mean = np.nanmean(Tp)
                    im = ax.pcolormesh(xx, yy, Tp, cmap=CMAP_T,
                                       norm=norm_T, shading="auto")
                    # PINN contours (white)
                    ax.contour(xx, yy, Tp, levels=5,
                               colors="white", linewidths=0.5, alpha=0.6,
                               linestyles="solid")
                    # FEM contours overlay (red dashed) — shows FEM isotherms on PINN field
                    ax.contour(xx, yy, T_fem, levels=5,
                               colors="#FF4444", linewidths=0.9, alpha=0.85,
                               linestyles="dashed")
                    ax.set_title(
                        f"PINN T̄={T_pinn_mean:.0f}°C  |  FEM T̄={T_fem_mean:.0f}°C\n"
                        f"MAE={mae:.1f}°C  (— — FEM isotherms)",
                        fontsize=7, color="#2E7D32", pad=2)
                else:
                    im = ax.pcolormesh(xx, yy, T_fem, cmap=CMAP_T,
                                       norm=norm_T, shading="auto", alpha=0.25)
                    ax.text(0.5, 0.5, "SKIPPED\n(FEM step not needed)",
                            ha="center", va="center", fontsize=8, color="#555",
                            style="italic", transform=ax.transAxes)
                    ax.set_facecolor("#EBEBEB")

            # ── Col 2: |Error| skip=2 ─────────────────────
            elif ci == 2:
                if t in s2_anch and t != 0.0:
                    err = np.abs(T_fem - s2_pinn[t])
                    im  = ax.pcolormesh(xx, yy, err, cmap=CMAP_E,
                                        norm=norm_e, shading="auto")
                    ax.contour(xx, yy, err, levels=4,
                               colors="black", linewidths=0.3, alpha=0.5)
                    ax.set_title(f"MAE={np.nanmean(err):.1f}°C  "
                                 f"max={np.nanmax(err):.1f}°C",
                                 fontsize=7, color="#B71C1C", pad=2)
                else:
                    ax.set_facecolor("#F5F5F5")
                    ax.text(0.5, 0.5, "—", ha="center", va="center",
                            fontsize=12, color="#CCC", transform=ax.transAxes)

            # ── Col 3: PINN skip=4 ────────────────────────
            elif ci == 3:
                if t == 0.0:
                    im = ax.pcolormesh(xx, yy, T_fem, cmap=CMAP_T,
                                       norm=norm_T, shading="auto")
                    ax.set_title("IC (same as FEM)", fontsize=7,
                                 color="#1565C0", pad=2)
                elif t in s4_anch:
                    Tp  = s4_pinn[t]
                    mae = np.nanmean(np.abs(T_fem - Tp))
                    T_fem_mean  = np.nanmean(T_fem)
                    T_pinn_mean = np.nanmean(Tp)
                    im = ax.pcolormesh(xx, yy, Tp, cmap=CMAP_T,
                                       norm=norm_T, shading="auto")
                    ax.contour(xx, yy, Tp, levels=5,
                               colors="white", linewidths=0.5, alpha=0.6,
                               linestyles="solid")
                    ax.contour(xx, yy, T_fem, levels=5,
                               colors="#FF4444", linewidths=0.9, alpha=0.85,
                               linestyles="dashed")
                    ax.set_title(
                        f"PINN T̄={T_pinn_mean:.0f}°C  |  FEM T̄={T_fem_mean:.0f}°C\n"
                        f"MAE={mae:.1f}°C  (— — FEM isotherms)",
                        fontsize=7, color="#E65100", pad=2)
                else:
                    im = ax.pcolormesh(xx, yy, T_fem, cmap=CMAP_T,
                                       norm=norm_T, shading="auto", alpha=0.25)
                    ax.text(0.5, 0.5, "SKIPPED\n(FEM step not needed)",
                            ha="center", va="center", fontsize=8, color="#555",
                            style="italic", transform=ax.transAxes)
                    ax.set_facecolor("#EBEBEB")

            # ── Col 4: |Error| skip=4 ─────────────────────
            elif ci == 4:
                if t in s4_anch and t != 0.0:
                    err = np.abs(T_fem - s4_pinn[t])
                    im  = ax.pcolormesh(xx, yy, err, cmap=CMAP_E,
                                        norm=norm_e, shading="auto")
                    ax.contour(xx, yy, err, levels=4,
                               colors="black", linewidths=0.3, alpha=0.5)
                    ax.set_title(f"MAE={np.nanmean(err):.1f}°C  "
                                 f"max={np.nanmax(err):.1f}°C",
                                 fontsize=7, color="#B71C1C", pad=2)
                else:
                    ax.set_facecolor("#F5F5F5")
                    ax.text(0.5, 0.5, "—", ha="center", va="center",
                            fontsize=12, color="#CCC", transform=ax.transAxes)

            # colorbars on last row
            if ri == n_row - 1:
                ax.set_xlabel("x [m]", fontsize=7)
                if im is not None:
                    lbl = "T [°C]" if ci in (0, 1, 3) else "|ΔT| [°C]"
                    _colorbar(fig, ax, im, lbl, fontsize=7)

    plt.subplots_adjust(left=0.10, right=0.96, top=0.93, bottom=0.06,
                        hspace=0.22, wspace=0.08)
    p = os.path.join(OUT, f"field_timeline_{domain_name}_{arch}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(p)}")
    return p


# ══════════════════════════════════════════════════════════════
# 8. Figure D: All Optimizers × Skip Values Summary
# ══════════════════════════════════════════════════════════════

def fig_optimizer_summary(anal, xx, yy, mask, domain_name, t_snap=15.0):
    """
    4×4 grid: row = FEM + 3 optimizers, column = skip=1,2,4,6
    """
    skips = [1, 2, 4, 6]
    dt    = 1.5   # FEM step interval

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle(
        f"All Optimizers × All Skip Values — {domain_name}   t = {t_snap:.0f} s\n"
        f"Row 1: FEM reference | Rows 2–4: NAS-MCO-PINN | Skip = number of FEM steps skipped",
        fontsize=13, fontweight="bold", y=1.01,
    )

    norm_T = Normalize(vmin=T_WATER, vmax=T_INIT)
    T_fem  = _apply_mask(anal.T(xx, yy, t_snap), mask)

    all_err, all_pinn = {a: [] for a in ARCHS}, {a: [] for a in ARCHS}
    print(f"  Training all optimizer × skip grids for t={t_snap}s...")
    for skip in skips:
        t_start = max(0.0, t_snap - skip * dt)
        for arch in ARCHS:
            Tp = _apply_mask(train_and_predict_field(arch, t_start, t_snap, anal, xx, yy), mask)
            all_pinn[arch].append(Tp)
            all_err[arch].append(np.abs(T_fem - Tp))

    err_max = max(np.nanpercentile(e, 96)
                  for errs in all_err.values() for e in errs)
    err_max = max(err_max, 1.0)
    norm_e  = Normalize(vmin=0, vmax=err_max)

    col_titles = [f"skip={s}\n({21 - s*((21-1)//s) if s<6 else 4}/{21} FEM)" for s in skips]

    for ri in range(4):
        for ci, skip in enumerate(skips):
            ax = axes[ri, ci]
            ax.set_facecolor("#F5F5F5")

            if ri == 0:
                im = ax.pcolormesh(xx, yy, T_fem, cmap=CMAP_T, norm=norm_T, shading="auto")
                ax.contour(xx, yy, T_fem, levels=6, colors="white", linewidths=0.4, alpha=0.5)
                if ci == 0: ax.set_ylabel("FEM (reference)", fontsize=9, fontweight="bold")
                if ci == 0: ax.set_title(col_titles[ci], fontsize=9, fontweight="bold")
                else:        ax.set_title(col_titles[ci], fontsize=9)
            else:
                arch = ARCHS[ri - 1]
                Tp   = all_pinn[arch][ci]
                im   = ax.pcolormesh(xx, yy, Tp, cmap=CMAP_T, norm=norm_T, shading="auto")
                ax.contour(xx, yy, Tp, levels=5, colors="white", linewidths=0.3, alpha=0.5)
                mae  = np.nanmean(all_err[arch][ci])
                ax.text(0.97, 0.03, f"MAE={mae:.1f}°C",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=8, fontweight="bold",
                        color="lime" if mae < 5 else "yellow" if mae < 10 else "red",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
                if ci == 0: ax.set_ylabel(ARCH_LABEL[arch], fontsize=9, fontweight="bold",
                                           color=ARCH_COLOR[arch])

            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            if ri == 3: ax.set_xlabel("x [m]", fontsize=8)

            if ci == 3:
                _colorbar(fig, ax, im, "T [°C]")

    plt.subplots_adjust(left=0.08, right=0.94, top=0.94, bottom=0.04,
                        hspace=0.06, wspace=0.06)
    p = os.path.join(OUT, f"field_optimizer_summary_{domain_name}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(p)}")
    return p


# ══════════════════════════════════════════════════════════════
# 9. Main
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*65)
    print("  Level 8 — 3D Field Visualizations")
    print("  FEM reference vs NAS-MCO-PINN Skip Operator")
    print("="*65 + "\n")

    anal = AnalyticalField2D()
    print(f"  Analytical solution: Λ₁₁={anal.Lambda[0,0]*1000:.2f}×10⁻³ s⁻¹")
    print(f"  Bi_x={H_CONV*LX/K:.1f}, Bi_y={H_CONV*LY/K:.1f}")

    # ── Rectangular domain ─────────────────────────────────
    xi = np.linspace(0, LX, N_GRID)
    yi = np.linspace(0, LY, N_GRID)
    xx_r, yy_r = np.meshgrid(xi, yi)
    mask_r     = rect_mask(xx_r, yy_r)

    print("\n── Domain 1: Rectangular (Casting Cross-Section) ──")
    fig_3d_surface(anal, xx_r, yy_r, mask_r, "Rectangular", t_snap=15.0)
    fig_2d_field_comparison(anal, xx_r, yy_r, mask_r, "Rectangular", skip=2)
    fig_2d_field_comparison(anal, xx_r, yy_r, mask_r, "Rectangular", skip=4)
    fig_skip_timeline(anal, xx_r, yy_r, mask_r, "Rectangular")
    fig_optimizer_summary(anal, xx_r, yy_r, mask_r, "Rectangular", t_snap=15.0)

    # ── L-shaped domain ─────────────────────────────────
    print("\n── Domain 2: L-Shaped (Automotive Subframe) ──")
    mask_l     = lshape_mask(xx_r, yy_r)

    fig_3d_surface(anal, xx_r, yy_r, mask_l, "LShape", t_snap=15.0)
    fig_2d_field_comparison(anal, xx_r, yy_r, mask_l, "LShape", skip=2)
    fig_2d_field_comparison(anal, xx_r, yy_r, mask_l, "LShape", skip=4)
    fig_skip_timeline(anal, xx_r, yy_r, mask_l, "LShape")
    fig_optimizer_summary(anal, xx_r, yy_r, mask_l, "LShape", t_snap=15.0)

    print("\n" + "="*65)
    print("  All figures complete →", OUT)
    files = [f for f in os.listdir(OUT) if f.startswith("field_")]
    for f in sorted(files):
        kb = os.path.getsize(os.path.join(OUT,f)) // 1024
        print(f"    {f}  ({kb} KB)")


if __name__ == "__main__":
    main()
