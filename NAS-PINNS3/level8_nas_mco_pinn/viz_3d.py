"""
viz_3d.py — Professional 3D Field Visualization
=================================================
1. fig_field_slices   : FEM vs PINN 3 orthogonal slices (xy, xz, yz)
2. fig_loss_curves    : Training loss curves + MCO weights
3. fig_domain_compare : All 4 domains side by side at z_mid slice
4. fig_3d_surface     : matplotlib 3D surface — single domain single time

Usage:
    from level8_nas_mco_pinn.viz_3d import run_all_figures
    run_all_figures(results_rect, results_cyl, results_stacked)
"""

import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore")

from .domains_3d import (
    Rectangular3D, Cylinder3D, StackedCubes3D, LPrism3D,
    T_INIT, T_WATER, DELTA_T,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

CMAP_T  = "plasma"
CMAP_E  = "Reds"
ARCH_LABEL = {
    "bayesian": "Bayesian (TPE)",
    "nsga2":    "NSGA-II",
    "nsga3":    "NSGA-III",
}
ARCH_COLOR = {
    "bayesian": "#1565C0",
    "nsga2":    "#2E7D32",
    "nsga3":    "#E65100",
}
DOMAIN_COLOR = {
    "rectangular": "#1565C0",
    "cylinder":    "#2E7D32",
    "stacked":     "#E65100",
    "lprism":      "#6A1B9A",
}
DOMAIN_LABEL = {
    "rectangular": "Rectangular Prism\n(1.3×0.6×0.4 m)",
    "cylinder":    "Cylinder\n(R=0.25 m, H=0.6 m)",
    "stacked":     "Stacked Cubes\n(2×0.5 m cubes)",
    "lprism":      "L-Prism\n(1.3×0.6×0.4 m, L-cut)",
}

norm_T = Normalize(vmin=T_WATER, vmax=T_INIT)


def _cb(fig, ax, im, label, fs=7):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.04)
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=fs)
    cb.ax.tick_params(labelsize=fs-1)


def _mid_z_slice(T_vol, zi):
    """Select z=mid slice."""
    k_mid = len(zi) // 2
    return T_vol[:, :, k_mid], zi[k_mid]


# ══════════════════════════════════════════════════════════════
# Figure 1: 3D Orthogonal Slices — FEM vs PINN vs Error
# ══════════════════════════════════════════════════════════════

def fig_field_slices(domain, domain_name, arch_name,
                     skip, result_skip, t_window=0):
    """
    3 slices × 3 columns (FEM / PINN / |Error|) for each domain.
    Slices: z=Lz/2 (xy), y=Ly/2 (xz), x=Lx/2 (yz)

    Parameters
    ----------
    result_skip : dict  (single skip value from run_skip_3d output)
    t_window    : which window to display (0 = first window)
    """
    T_fem  = result_skip["T_fem"][t_window]    # (ny,nx,nz)
    T_pred = result_skip["T_fields"][t_window]
    grid   = result_skip["grid"]
    xi, yi, zi = grid["xi"], grid["yi"], grid["zi"]
    xx2d, yy2d = grid["xx2d"], grid["yy2d"]

    k_mid = len(zi) // 2
    j_mid = len(yi) // 2
    i_mid = len(xi) // 2

    # z slice (xy)
    fem_xy  = T_fem[:, :, k_mid]
    pred_xy = T_pred[:, :, k_mid]
    err_xy  = np.abs(fem_xy - pred_xy)

    # y slice (xz): T_vol shape=(ny,nx,nz) → xz plane
    fem_xz  = T_fem[j_mid, :, :]    # (nx, nz)
    pred_xz = T_pred[j_mid, :, :]
    err_xz  = np.abs(fem_xz - pred_xz)

    # x slice (yz)
    fem_yz  = T_fem[:, i_mid, :]    # (ny, nz)
    pred_yz = T_pred[:, i_mid, :]
    err_yz  = np.abs(fem_yz - pred_yz)

    mae = result_skip["mae_C"]
    err_max = max(np.nanpercentile(
        np.concatenate([err_xy.ravel(), err_xz.ravel(), err_yz.ravel()]),
        97, ), 1.0)
    norm_e = Normalize(vmin=0, vmax=err_max)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    fig.suptitle(
        f"3D Field: FEM vs PINN  |  {DOMAIN_LABEL[domain_name]}  "
        f"|  {ARCH_LABEL[arch_name]}  skip={skip}\n"
        f"Overall MAE = {mae:.2f}°C   "
        f"(— — red contours = FEM isotherms on PINN panel)",
        fontsize=11, fontweight="bold",
    )

    row_labels = [
        f"z = {zi[k_mid]:.2f} m  (xy plane)",
        f"y = {yi[j_mid]:.2f} m  (xz plane)",
        f"x = {xi[i_mid]:.2f} m  (yz plane)",
    ]
    col_labels = ["FEM (analytical)", "PINN prediction", "|FEM − PINN|"]

    slices = [
        (fem_xy,  pred_xy,  err_xy,  xx2d,                    yy2d,                    "x [m]", "y [m]"),
        (fem_xz.T,pred_xz.T,err_xz.T,np.meshgrid(xi,zi)[0],   np.meshgrid(xi,zi)[1],   "x [m]", "z [m]"),
        (fem_yz.T,pred_yz.T,err_yz.T,np.meshgrid(yi,zi)[0],   np.meshgrid(yi,zi)[1],   "y [m]", "z [m]"),
    ]

    for ri, (fem_s, pred_s, err_s, XX, YY, xl, yl) in enumerate(slices):
        ax_f = axes[ri, 0]
        ax_p = axes[ri, 1]
        ax_e = axes[ri, 2]

        for ax in [ax_f, ax_p, ax_e]:
            ax.set_aspect("auto")
            ax.tick_params(labelsize=6)
            ax.set_xlabel(xl, fontsize=7)
            ax.set_ylabel(yl if ri == 0 else "", fontsize=7)

        # FEM
        im_f = ax_f.pcolormesh(XX, YY, fem_s, cmap=CMAP_T,
                                norm=norm_T, shading="auto")
        ax_f.contour(XX, YY, fem_s, levels=6,
                     colors="white", linewidths=0.4, alpha=0.7)
        T_mean = np.nanmean(fem_s)
        ax_f.set_title(f"FEM  T̄={T_mean:.0f}°C", fontsize=8, color="#333")

        # PINN + FEM contours overlay
        im_p = ax_p.pcolormesh(XX, YY, pred_s, cmap=CMAP_T,
                                norm=norm_T, shading="auto")
        ax_p.contour(XX, YY, pred_s, levels=6,
                     colors="white", linewidths=0.4, alpha=0.6, linestyles="solid")
        ax_p.contour(XX, YY, fem_s, levels=6,
                     colors="#FF4444", linewidths=0.8, alpha=0.85, linestyles="dashed")
        T_pmean = np.nanmean(pred_s)
        mae_s   = np.nanmean(err_s)
        ax_p.set_title(f"PINN  T̄={T_pmean:.0f}°C  MAE={mae_s:.1f}°C",
                       fontsize=8, color=ARCH_COLOR.get(arch_name, "#333"))

        # Error
        im_e = ax_e.pcolormesh(XX, YY, err_s, cmap=CMAP_E,
                                norm=norm_e, shading="auto")
        ax_e.contour(XX, YY, err_s, levels=4,
                     colors="black", linewidths=0.3, alpha=0.5)
        ax_e.set_title(f"|Error|  max={np.nanmax(err_s):.1f}°C",
                       fontsize=8, color="#B71C1C")

        if ri == 0:
            _cb(fig, ax_f, im_f, "T [°C]")
            _cb(fig, ax_p, im_p, "T [°C]")
            _cb(fig, ax_e, im_e, "|ΔT| [°C]")

        axes[ri, 0].set_ylabel(row_labels[ri], fontsize=8, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f"3d_slices_{domain_name}_{arch_name}_skip{skip}.png"
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return p


# ══════════════════════════════════════════════════════════════
# Figure 2: Loss Curves + MCO Weights
# ══════════════════════════════════════════════════════════════

def fig_loss_curves(results_by_arch, domain_name, skip=2):
    """
    Loss curve for first window skip=2 across all optimizers.
    Left: L_phys, L_bc, L_ic, L_total  |  Right: MCO weights w_i = 1/σ_i²
    """
    archs = [k for k in results_by_arch if skip in results_by_arch[k]]
    n_arch = len(archs)
    if n_arch == 0:
        return

    fig, axes = plt.subplots(n_arch, 2, figsize=(12, 3.5 * n_arch))
    if n_arch == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        f"MCO Training Loss Curves — {domain_name}  skip={skip}\n"
        f"Left: loss components  |  Right: MCO adaptive weights w_i = 1/σ_i²",
        fontsize=11, fontweight="bold",
    )

    for ri, arch in enumerate(archs):
        hist = results_by_arch[arch][skip]["loss_history"][0]  # first window
        epochs = np.linspace(0, len(hist["L_phys"]) - 1, len(hist["L_phys"]))

        ax_l = axes[ri, 0]
        ax_w = axes[ri, 1]
        c    = ARCH_COLOR.get(arch, "#333")

        # Loss curves
        ax_l.semilogy(epochs, hist["L_phys"],  label="L_phys",  color="#E53935", lw=1.5)
        ax_l.semilogy(epochs, hist["L_bc"],    label="L_bc",    color="#1565C0", lw=1.5)
        ax_l.semilogy(epochs, hist["L_ic"],    label="L_ic",    color="#2E7D32", lw=1.5)
        ax_l.semilogy(epochs, hist["L_total"], label="L_total", color="#6A1B9A", lw=2.0,
                      linestyle="--")
        ax_l.set_title(f"{ARCH_LABEL[arch]} — Loss", fontsize=9, color=c,
                       fontweight="bold")
        ax_l.set_xlabel("Epoch (log scale × 20)", fontsize=8)
        ax_l.set_ylabel("Loss (log)", fontsize=8)
        ax_l.legend(fontsize=7, loc="upper right")
        ax_l.tick_params(labelsize=7)
        ax_l.grid(True, alpha=0.3, which="both")

        # MCO weights
        ax_w.plot(epochs, hist["w_phys"], label="w_phys", color="#E53935", lw=1.5)
        ax_w.plot(epochs, hist["w_bc"],   label="w_bc",   color="#1565C0", lw=1.5)
        ax_w.plot(epochs, hist["w_ic"],   label="w_ic",   color="#2E7D32", lw=1.5)
        ax_w.set_title(f"{ARCH_LABEL[arch]} — MCO Weights", fontsize=9, color=c,
                       fontweight="bold")
        ax_w.set_xlabel("Epoch", fontsize=8)
        ax_w.set_ylabel("w_i = 1/σ_i²", fontsize=8)
        ax_w.legend(fontsize=7, loc="upper left")
        ax_w.tick_params(labelsize=7)
        ax_w.grid(True, alpha=0.3)

        # Convergence note
        final_w = [hist["w_phys"][-1], hist["w_bc"][-1], hist["w_ic"][-1]]
        note = " | ".join([f"w_{n}={v:.1f}" for n, v in
                           zip(["ph","bc","ic"], final_w)])
        ax_w.text(0.02, 0.05, f"Final: {note}",
                  transform=ax_w.transAxes, fontsize=7,
                  color=c, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f"loss_curves_{domain_name}_skip{skip}.png"
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return p


# ══════════════════════════════════════════════════════════════
# Figure 3: All Domains Comparison (z-mid slice)
# ══════════════════════════════════════════════════════════════

def fig_domain_compare(all_domain_results, arch_name="bayesian", skip=2, t_window=0):
    """
    4 domains × 3 columns (FEM / PINN / Error) — at z=mid slice.
    """
    domains = list(all_domain_results.keys())
    n_dom   = len(domains)

    fig, axes = plt.subplots(n_dom, 3, figsize=(12, 3.5 * n_dom))
    if n_dom == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        f"Domain Comparison — skip={skip}  arch={ARCH_LABEL.get(arch_name, arch_name)}\n"
        f"z = mid-plane slice  |  — — red = FEM isotherms on PINN panel",
        fontsize=11, fontweight="bold",
    )

    col_titles = ["FEM (analytical)", "PINN prediction", "|FEM − PINN| error"]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=9, fontweight="bold", pad=4)

    for ri, dname in enumerate(domains):
        if arch_name not in all_domain_results[dname]:
            continue
        res = all_domain_results[dname][arch_name]
        if skip not in res:
            continue
        rs   = res[skip]
        grid = rs["grid"]
        xi, yi, zi = grid["xi"], grid["yi"], grid["zi"]
        xx2d, yy2d = grid["xx2d"], grid["yy2d"]
        k_mid = len(zi) // 2

        T_fem  = rs["T_fem"][t_window][:, :, k_mid]
        T_pred = rs["T_fields"][t_window][:, :, k_mid]
        err    = np.abs(T_fem - T_pred)

        err_max = max(np.nanpercentile(err, 97), 1.0)
        norm_e  = Normalize(vmin=0, vmax=err_max)

        ax_f = axes[ri, 0]
        ax_p = axes[ri, 1]
        ax_e = axes[ri, 2]

        for ax in [ax_f, ax_p, ax_e]:
            ax.tick_params(labelsize=6)
            ax.set_xlabel("x [m]", fontsize=7)

        # FEM
        im_f = ax_f.pcolormesh(xx2d, yy2d, T_fem, cmap=CMAP_T,
                                norm=norm_T, shading="auto")
        ax_f.contour(xx2d, yy2d, T_fem, levels=6,
                     colors="white", linewidths=0.4, alpha=0.7)
        ax_f.set_ylabel(DOMAIN_LABEL[dname], fontsize=8, fontweight="bold",
                        color=DOMAIN_COLOR.get(dname, "#333"))

        # PINN + overlay
        im_p = ax_p.pcolormesh(xx2d, yy2d, T_pred, cmap=CMAP_T,
                                norm=norm_T, shading="auto")
        ax_p.contour(xx2d, yy2d, T_pred, levels=6,
                     colors="white", linewidths=0.4, alpha=0.6, linestyles="solid")
        ax_p.contour(xx2d, yy2d, T_fem, levels=6,
                     colors="#FF4444", linewidths=0.8, alpha=0.85, linestyles="dashed")
        mae = rs["mae_C"]
        ax_p.set_title(f"MAE = {mae:.2f}°C", fontsize=8,
                       color=ARCH_COLOR.get(arch_name, "#333"), pad=2)

        # Error
        im_e = ax_e.pcolormesh(xx2d, yy2d, err, cmap=CMAP_E,
                                norm=norm_e, shading="auto")
        ax_e.contour(xx2d, yy2d, err, levels=4,
                     colors="black", linewidths=0.3, alpha=0.5)

        if ri == 0:
            _cb(fig, ax_f, im_f, "T [°C]")
            _cb(fig, ax_p, im_p, "T [°C]")
        _cb(fig, ax_e, im_e, "|ΔT| [°C]")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f"domain_compare_{arch_name}_skip{skip}.png"
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return p


# ══════════════════════════════════════════════════════════════
# Figure 4: 3D Surface View (matplotlib 3D)
# ══════════════════════════════════════════════════════════════

def fig_3d_surface_compare(domain, domain_name, arch_name,
                           skip, result_skip, t_window=0):
    """
    2×2 subplot: FEM z-mid / PINN z-mid / Error z-mid / 3D surface comparison
    """
    T_fem  = result_skip["T_fem"][t_window]
    T_pred = result_skip["T_fields"][t_window]
    grid   = result_skip["grid"]
    xi, yi, zi = grid["xi"], grid["yi"], grid["zi"]
    xx2d, yy2d = grid["xx2d"], grid["yy2d"]
    k_mid = len(zi) // 2

    fem_z  = T_fem[:, :, k_mid]
    pred_z = T_pred[:, :, k_mid]
    err_z  = np.abs(fem_z - pred_z)

    err_max = max(np.nanpercentile(err_z, 97), 1.0)
    norm_e  = Normalize(vmin=0, vmax=err_max)

    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

    mae = result_skip["mae_C"]
    fig.suptitle(
        f"3D Thermal Field  —  {DOMAIN_LABEL[domain_name]}\n"
        f"Arch: {ARCH_LABEL.get(arch_name)}  |  skip={skip}  |  "
        f"Overall MAE = {mae:.2f}°C",
        fontsize=12, fontweight="bold",
    )

    # ── Panel 1: FEM 2D (z-mid) ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(xx2d, yy2d, fem_z, cmap=CMAP_T,
                          norm=norm_T, shading="auto")
    ax1.contour(xx2d, yy2d, fem_z, levels=8,
                colors="white", linewidths=0.5, alpha=0.7)
    ax1.set_title(f"FEM  (z={zi[k_mid]:.2f}m slice)",
                  fontsize=9, fontweight="bold")
    ax1.set_xlabel("x [m]", fontsize=8); ax1.set_ylabel("y [m]", fontsize=8)
    ax1.tick_params(labelsize=7)
    _cb(fig, ax1, im1, "T [°C]")

    # ── Panel 2: PINN 2D + FEM overlay ──────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.pcolormesh(xx2d, yy2d, pred_z, cmap=CMAP_T,
                          norm=norm_T, shading="auto")
    ax2.contour(xx2d, yy2d, pred_z, levels=8,
                colors="white", linewidths=0.5, alpha=0.6, linestyles="solid")
    ax2.contour(xx2d, yy2d, fem_z, levels=8,
                colors="#FF4444", linewidths=0.9, alpha=0.9, linestyles="dashed")
    T_pfem = np.nanmean(fem_z)
    T_ppinn = np.nanmean(pred_z)
    mae_z = np.nanmean(err_z)
    ax2.set_title(f"PINN  FEM T̄={T_pfem:.0f}°C  PINN T̄={T_ppinn:.0f}°C  MAE={mae_z:.1f}°C",
                  fontsize=8.5, fontweight="bold",
                  color=ARCH_COLOR.get(arch_name, "#333"))
    ax2.set_xlabel("x [m]", fontsize=8); ax2.set_ylabel("y [m]", fontsize=8)
    ax2.tick_params(labelsize=7)
    _cb(fig, ax2, im2, "T [°C]")

    # ── Panel 3: Error heatmap ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.pcolormesh(xx2d, yy2d, err_z, cmap=CMAP_E,
                          norm=norm_e, shading="auto")
    ax3.contour(xx2d, yy2d, err_z, levels=5,
                colors="black", linewidths=0.4, alpha=0.6)
    ax3.set_title(f"|FEM − PINN|  max={np.nanmax(err_z):.1f}°C",
                  fontsize=9, fontweight="bold", color="#B71C1C")
    ax3.set_xlabel("x [m]", fontsize=8); ax3.set_ylabel("y [m]", fontsize=8)
    ax3.tick_params(labelsize=7)
    _cb(fig, ax3, im3, "|ΔT| [°C]")

    # ── Panel 4: 3D matplotlib surface ──────────────────
    ax4 = fig.add_subplot(gs[1, 1], projection="3d")
    valid = ~np.isnan(fem_z)
    if valid.any():
        fem_plot = np.where(valid, fem_z, np.nanmean(fem_z))
        surf_fem = ax4.plot_surface(
            xx2d, yy2d, fem_plot,
            facecolors=plt.cm.plasma(norm_T(fem_plot)),
            alpha=0.85, linewidth=0, antialiased=True,
        )
        # Show PINN as transparent wireframe (overlaid)
        pred_plot = np.where(valid, pred_z, np.nanmean(pred_z))
        ax4.plot_wireframe(
            xx2d, yy2d, pred_plot,
            color=ARCH_COLOR.get(arch_name, "blue"),
            alpha=0.35, linewidth=0.5, rstride=3, cstride=3,
        )
    ax4.set_title("3D Surface: FEM (color) + PINN (wire)",
                  fontsize=8.5, fontweight="bold")
    ax4.set_xlabel("x [m]", fontsize=7)
    ax4.set_ylabel("y [m]", fontsize=7)
    ax4.set_zlabel("T [°C]", fontsize=7)
    ax4.tick_params(labelsize=6)
    ax4.view_init(elev=30, azim=-60)
    ax4.set_zlim(T_WATER - 20, T_INIT + 20)

    # Shared legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], color="white", lw=0.5, label="— FEM isotherms"),
        Line2D([0],[0], color="#FF4444", lw=0.9, linestyle="--", label="— — FEM on PINN"),
        Line2D([0],[0], color=ARCH_COLOR.get(arch_name,"blue"), lw=0.5,
               label="PINN wire (3D)"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    fname = f"3d_surface_{domain_name}_{arch_name}_skip{skip}.png"
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return p


# ══════════════════════════════════════════════════════════════
# Figure 5: Skip Comparison (all skip values, single arch)
# ══════════════════════════════════════════════════════════════

def fig_skip_compare_3d(domain_name, arch_name, arch_results, t_window=0):
    """
    Single arch, all skip values: skip=1,2,4,6 → 4 rows × 3 columns
    """
    skips = sorted(arch_results.keys())
    n_s   = len(skips)
    fig, axes = plt.subplots(n_s, 3, figsize=(12, 3.2 * n_s))
    if n_s == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"Skip Operator Comparison  —  {domain_name}  arch={ARCH_LABEL.get(arch_name)}\n"
        f"Rows = skip values  |  z-mid slice  |  — — red = FEM isotherms",
        fontsize=11, fontweight="bold",
    )
    axes[0, 0].set_title("FEM (analytical)", fontsize=9, fontweight="bold")
    axes[0, 1].set_title("PINN prediction", fontsize=9, fontweight="bold")
    axes[0, 2].set_title("|FEM − PINN| error", fontsize=9, fontweight="bold")

    for ri, skip in enumerate(skips):
        rs   = arch_results[skip]
        grid = rs["grid"]
        xi, yi, zi = grid["xi"], grid["yi"], grid["zi"]
        xx2d, yy2d = grid["xx2d"], grid["yy2d"]
        k_mid = len(zi) // 2

        T_fem  = rs["T_fem"][t_window][:, :, k_mid]
        T_pred = rs["T_fields"][t_window][:, :, k_mid]
        err    = np.abs(T_fem - T_pred)

        err_max = max(np.nanpercentile(err, 97), 1.0)
        norm_e  = Normalize(vmin=0, vmax=err_max)
        mae     = rs["mae_C"]

        ax_f = axes[ri, 0]
        ax_p = axes[ri, 1]
        ax_e = axes[ri, 2]

        # FEM
        ax_f.pcolormesh(xx2d, yy2d, T_fem, cmap=CMAP_T,
                        norm=norm_T, shading="auto")
        ax_f.contour(xx2d, yy2d, T_fem, levels=6,
                     colors="white", linewidths=0.4, alpha=0.7)
        ax_f.set_ylabel(
            f"skip={skip}\n({rs.get('steps_used', '?')}/{21} FEM steps)",
            fontsize=8, fontweight="bold")

        # PINN
        im_p = ax_p.pcolormesh(xx2d, yy2d, T_pred, cmap=CMAP_T,
                                norm=norm_T, shading="auto")
        ax_p.contour(xx2d, yy2d, T_pred, levels=6,
                     colors="white", linewidths=0.4, alpha=0.6, linestyles="solid")
        ax_p.contour(xx2d, yy2d, T_fem, levels=6,
                     colors="#FF4444", linewidths=0.8, alpha=0.85, linestyles="dashed")
        ax_p.set_title(f"MAE = {mae:.1f}°C", fontsize=8,
                       color=ARCH_COLOR.get(arch_name, "#333"))

        # Error
        im_e = ax_e.pcolormesh(xx2d, yy2d, err, cmap=CMAP_E,
                                norm=norm_e, shading="auto")
        ax_e.contour(xx2d, yy2d, err, levels=4,
                     colors="black", linewidths=0.3, alpha=0.5)
        ax_e.set_title(f"max={np.nanmax(err):.1f}°C", fontsize=8,
                       color="#B71C1C")

        for ax in [ax_f, ax_p, ax_e]:
            ax.tick_params(labelsize=5)
            if ri == n_s - 1:
                ax.set_xlabel("x [m]", fontsize=7)

        if ri == 0:
            _cb(fig, ax_e, im_e, "|ΔT| [°C]")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f"skip_compare_3d_{domain_name}_{arch_name}.png"
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return p


# ══════════════════════════════════════════════════════════════
# Generate all figures
# ══════════════════════════════════════════════════════════════

def run_all_figures(all_results):
    """
    all_results = {
        domain_name: {
            arch_name: {skip: result_skip_dict}
        }
    }
    """
    print("\nStarting visualization...")

    for dname, arch_dict in all_results.items():
        for arch, skip_dict in arch_dict.items():
            for skip, rs in skip_dict.items():
                print(f"  [{dname}] [{arch}] skip={skip}")
                fig_field_slices(None, dname, arch, skip, rs)
                fig_3d_surface_compare(None, dname, arch, skip, rs)

        # Loss curves (all arch, skip=2)
        fig_loss_curves(arch_dict, dname, skip=2)

        # Skip comparison (bayesian)
        if "bayesian" in arch_dict:
            fig_skip_compare_3d(dname, "bayesian", arch_dict["bayesian"])

    # Domain comparison
    for arch in ["bayesian", "nsga2", "nsga3"]:
        fig_domain_compare(all_results, arch_name=arch, skip=2)

    print("All visualizations complete.")
