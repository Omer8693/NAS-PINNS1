"""
Academic PDF — NAS-MCO-PINNs Skip Operator Results.
Top-to-bottom cursor layout. Figure titles placed BELOW images (caption style).
"""
import json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from datetime import date

BASE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.join(BASE, "..", "..", "..")
NP3    = os.path.join(ROOT, "NAS-PINNS3")
IMG    = os.path.join(NP3,  "web", "static", "img")
V2_RES = os.path.join(NP3,  "level8_nas_mco_pinn", "results", "v2", "results_3d_v2.json")
OUT    = os.path.join(BASE,  "NAS_PINNS3_professor_brief.pdf")

with open(V2_RES) as f:
    DATA = json.load(f)

DOMAINS  = ["rectangular", "cylinder", "stacked", "lshape"]
ARCHS    = ["bayesian", "nsga2", "nsga3"]
SKIPS    = [1, 2, 4, 6]
ARCH_LBL = {"bayesian": "Bayesian", "nsga2": "NSGA-II", "nsga3": "NSGA-III"}
DOM_LBL  = {"rectangular": "Rectangular", "cylinder": "Cylinder",
             "stacked": "Stacked", "lshape": "L-Shape"}
FEM_SAVE = {1: "0%", 2: "52%", 4: "71%", 6: "81%"}

PW, PH   = 8.27, 11.69
LM, RM   = 0.07, 0.93
CW       = RM - LM
PT       = 1.0 / (PH * 72.0)    # 1 point in figure-y coords

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif"],
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "text.color":       "#111111",
})

def load_img(fname):
    p = os.path.join(IMG, fname)
    return mpimg.imread(p) if os.path.exists(p) else None

def wrap(text, chars=108):
    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
        else:
            out.extend(textwrap.wrap(para, chars))
    return out

def lh(fs, sp=1.5):
    """Line height in figure coords for given fontsize and spacing."""
    return fs * sp * PT


class Page:
    def __init__(self):
        self.fig = plt.figure(figsize=(PW, PH), facecolor="white")
        self.y   = 0.975

    def gap(self, h):
        self.y -= h

    def hline(self, y=None, color="#bbbbbb", lw=0.8):
        yy = y if y is not None else self.y
        self.fig.add_artist(plt.Line2D([LM, RM], [yy, yy],
            transform=self.fig.transFigure, color=color, linewidth=lw))

    def text(self, txt, fs=8.5, bold=False, color="#111111",
             ha="left", x=None, sp=1.5):
        """Single line of text at cursor, advance cursor."""
        h = lh(fs, sp)
        xpos = x if x is not None else LM
        self.fig.text(xpos, self.y - h * 0.78, txt,
                      fontsize=fs,
                      fontweight="bold" if bold else "normal",
                      color=color, ha=ha, va="baseline")
        self.y -= h

    def lines(self, text_lines, fs=8.3, color="#222222", sp=1.5, italic=False):
        """Render a list of pre-wrapped lines, advance cursor."""
        h = lh(fs, sp)
        for ln in text_lines:
            self.fig.text(LM, self.y - h * 0.78, ln,
                          fontsize=fs, color=color, va="baseline",
                          style="italic" if italic else "normal")
            self.y -= h

    def section(self, num, title):
        self.gap(0.008)
        self.text(f"{num}.  {title}", fs=10, bold=True, color="#1a237e", sp=1.4)
        self.gap(0.005)

    def caption(self, label, text_lines, fs=7.8):
        """Bold label + italic text lines. E.g. 'Figure 1.' + description."""
        self.gap(0.004)
        # label on same line as first text line — rendered in one fig.text
        first = text_lines[0] if text_lines else ""
        rest  = text_lines[1:]
        h = lh(fs, 1.4)
        # label (bold)
        self.fig.text(LM, self.y - h * 0.78, label,
                      fontsize=fs, fontweight="bold", color="#333333", va="baseline")
        # rest of first line (italic) positioned after label
        label_w = len(label) * fs * 0.55 * PT / (PW / PH)   # rough char width in figure-x
        self.fig.text(LM + label_w, self.y - h * 0.78, " " + first,
                      fontsize=fs, color="#555555", va="baseline", style="italic")
        self.y -= h
        for ln in rest:
            self.fig.text(LM, self.y - h * 0.78, ln,
                          fontsize=fs, color="#555555", va="baseline", style="italic")
            self.y -= h
        self.gap(0.003)

    def image(self, fname, height):
        """Draw image at current cursor, advance cursor by height."""
        img = load_img(fname)
        y0  = self.y - height
        ax  = self.fig.add_axes([LM, y0, CW, height])
        if img is not None:
            ax.imshow(img, aspect="auto")
        ax.axis("off")
        self.y -= height

    def axes(self, height):
        """Reserve space and return an axes for custom plots."""
        y0 = self.y - height
        ax = self.fig.add_axes([LM, y0, CW, height])
        self.y -= height
        return ax

    def table(self, col_labels, row_data, row_colors, fontsize=7.0, row_h_scale=1.0):
        n   = len(row_data) + 1
        rh  = lh(fontsize, 1.5) * row_h_scale
        th  = n * rh + 0.005
        y0  = self.y - th
        ax  = self.fig.add_axes([LM, y0, CW, th])
        ax.axis("off")
        tbl = ax.table(
            cellText=row_data, colLabels=col_labels,
            cellLoc="center", loc="center",
            cellColours=row_colors,
            colColours=["#c5cae9"] * len(col_labels),
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)
        tbl.scale(1, 1.0)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#cccccc")
            if r == 0:
                cell.set_text_props(fontweight="bold")
        self.y -= th

    def footer(self, page, total=3):
        self.hline(y=0.026, color="#aaaaaa")
        self.fig.text(LM, 0.013,
            f"NAS-MCO-PINNs Progress Report \u2014 Page {page} of {total}",
            fontsize=7, color="#888888")
        self.fig.text(RM, 0.013, date.today().strftime("%B %Y"),
            fontsize=7, color="#888888", ha="right")

    def save(self, pdf):
        pdf.savefig(self.fig, dpi=180)
        plt.close(self.fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Introduction + Skip Operator + Method
# ═══════════════════════════════════════════════════════════════════════════════
def page1(pdf):
    p = Page()

    p.text("NAS-MCO-PINNs: Neural Architecture Search with Multi-Loss",
           fs=12, bold=True, ha="center", x=0.5, sp=1.4)
    p.text("Consistency Optimization and Skip Operator for 3D Thermal Simulation",
           fs=12, bold=True, ha="center", x=0.5, sp=1.4)
    p.gap(0.004)
    p.text("Project Progress Report  \u2014  " + date.today().strftime("%B %Y"),
           fs=9, color="#555555", ha="center", x=0.5, sp=1.3)
    p.gap(0.006)
    p.hline(color="#666666", lw=1.0)
    p.gap(0.008)

    p.section(1, "Introduction")
    p.lines(wrap(
        "Physics-Informed Neural Networks (PINNs) offer a mesh-free approach to solving "
        "partial differential equations governing heat transfer. In a standard PINN, the "
        "network is trained on each FEM time window [t_k, t_{k+1}], so the number of PINN "
        "training runs equals the total number of FEM time steps."
        "\n"
        "This project presents NAS-MCO-PINNs with three contributions: "
        "(i) a residual skip operator predicting s time steps ahead in a single run; "
        "(ii) Neural Architecture Search (NAS) via Bayesian Optimization, NSGA-II, and NSGA-III; "
        "and (iii) Multi-Loss Consistency Optimization (MCO) for adaptive loss weighting."
        "\n"
        "Validated on A356 aluminum water quenching across four 3D geometries: "
        "Rectangular Prism (1.3x0.6x0.4 m), Cylinder (R=0.25 m, H=0.6 m), "
        "Stacked Cubes (2x0.5 m), and L-Shape (0.8x0.8x0.4 m, numerical FD reference). "
        "Total: 48 runs (4 domains x 3 NAS optimizers x 4 skip values)."
    ))
    p.gap(0.010)

    p.section(2, "The Residual Skip Operator")
    p.lines(wrap(
        "Instead of one PINN per FEM step, a single network f_theta predicts s steps ahead "
        "from the current initial condition. The predicted temperature field is:"
    ))
    p.gap(0.006)
    p.text("theta_hat(t + s*dt)  =  theta_IC  +  tau * f_theta(x, y, z, tau, theta_IC)",
           fs=9.0, bold=False, color="#1a237e", ha="center", x=0.5, sp=1.5)
    p.gap(0.006)
    p.lines(wrap(
        "where tau = t/(s*dt) in [0,1] is the normalized time coordinate. This guarantees "
        "the initial condition exactly at tau=0. Skipping s steps reduces FEM calls from N "
        "to ceil(N/s), with savings shown in Table 1."
    ))
    p.gap(0.012)

    # Table 1
    col_lbl = ["Skip Value (s)", "FEM Steps Saved", "Reduction", "PINN Runs (of 21)"]
    row_d   = [["s = 1  (baseline)", "0 steps", "0%", "21 runs"],
               ["s = 2",            "10 steps", "52%", "11 runs"],
               ["s = 4",            "15 steps", "71%",  "6 runs"],
               ["s = 6",            "17 steps", "81%",  "4 runs"]]
    row_c   = [["#fafafa"]*4, ["#e3f2fd"]*4, ["#e8f5e9"]*4, ["#f3e5f5"]*4]
    p.table(col_lbl, row_d, row_c, fontsize=8.5, row_h_scale=1.6)
    p.gap(0.004)
    p.caption("Table 1.", ["FEM computation savings per skip value (21 total time steps)."])
    p.gap(0.010)

    p.section(3, "Training Method and Physical Parameters")
    p.lines(wrap(
        "Each PINN window is trained with AdamW (2000 epochs). The MCO loss function:"
    ))
    p.gap(0.006)
    p.text("L  =  w_target * L_target  +  0.05 * w_PDE * L_PDE     ,     w_i = 1 / sigma_i^2",
           fs=9.0, color="#1a237e", ha="center", x=0.5, sp=1.5)
    p.gap(0.006)
    p.lines(wrap(
        "Adaptive weights w_i are updated via the running standard deviation of each loss "
        "component, eliminating manual tuning. NAS search space: depth 2-6 layers, "
        "width 32-256 neurons, activations {Tanh, GELU, SiLU}. Optimizers: Bayesian "
        "Optimization, NSGA-II, NSGA-III (500 candidate evaluations each)."
        "\n"
        "A356 aluminum parameters: rho=2670 kg/m3, cp=963 J/(kg*K), k=151 W/(m*K). "
        "Boundary: convective quenching h=2000 W/(m2*K), T_inf=20 C, T_0=540 C. "
        "PDE: rho*cp * dT/dt = div(k * grad(T))."
    ))

    p.footer(1)
    p.save(pdf)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MAE Results Table + Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
def page2(pdf):
    p = Page()

    p.text("4.  Results: Skip Operator Accuracy",
           fs=11, bold=True, ha="center", x=0.5, sp=1.4)
    p.gap(0.005)
    p.hline(color="#666666", lw=1.0)
    p.gap(0.008)

    p.lines(wrap(
        "Table 2 presents MAE (deg C) for all 48 configurations. "
        "Color coding: dark green = best per row, light green < 8 C, "
        "yellow = 8-15 C, red > 15 C."
    ))
    p.gap(0.008)

    # MAE Table — all 48 runs
    col_lbl = ["Domain", "Optimizer",
               "skip=1\n(0%)", "skip=2\n(52%)", "skip=4\n(71%)", "skip=6\n(81%)"]
    rows, colors = [], []
    for dom in DOMAINS:
        for arch in ARCHS:
            row   = [DOM_LBL[dom], ARCH_LBL[arch]]
            color = ["#eeeeee", "#eeeeee"]
            vals  = [DATA[dom][arch].get(str(sk), {}).get("mae_C", None) for sk in SKIPS]
            minv  = min((v for v in vals if v is not None), default=None)
            for v in vals:
                if v is None:
                    row.append("--"); color.append("#ffffff")
                else:
                    row.append(f"{v:.2f}")
                    if v == minv:   color.append("#a5d6a7")
                    elif v < 8:     color.append("#e8f5e9")
                    elif v < 15:    color.append("#fff9c4")
                    else:           color.append("#ffcdd2")
            rows.append(row); colors.append(color)

    p.table(col_lbl, rows, colors, fontsize=7.2, row_h_scale=1.12)
    p.gap(0.004)
    p.caption("Table 2.", [
        "MAE (deg C) for all 48 training runs. FEM savings shown in column headers.",
        "Green shading = good accuracy; red shading = MAE > 15 deg C.",
    ])
    p.gap(0.014)

    # Bar chart — full width, tall
    bar_h = 0.370
    ax = p.axes(bar_h)
    dom_colors = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"]
    x = np.arange(len(SKIPS)); w = 0.18
    for di, (dom, col) in enumerate(zip(DOMAINS, dom_colors)):
        best = []
        for sk in SKIPS:
            vals = [DATA[dom][a].get(str(sk), {}).get("mae_C", np.nan) for a in ARCHS]
            vals = [v for v in vals if not np.isnan(v)]
            best.append(min(vals) if vals else np.nan)
        bars = ax.bar(x + (di - 1.5) * w, best, w,
                      label=DOM_LBL[dom], color=col, alpha=0.82,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, best):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{val:.1f}", ha="center", va="bottom",
                        fontsize=7.0, color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"skip = {s}\n({FEM_SAVE[s]} FEM saved)" for s in SKIPS], fontsize=9.5)
    ax.set_ylabel("MAE (deg C)", fontsize=10)
    ax.set_ylim(0, 32)
    ax.axhline(10, color="#999999", linewidth=0.9, linestyle="--")
    ax.text(3.58, 10.6, "10 deg C threshold", fontsize=8, color="#666666")
    ax.legend(fontsize=9, framealpha=0.85, loc="upper left", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)
    ax.tick_params(axis="y", labelsize=9)

    p.gap(0.006)
    p.caption("Figure 1.", [
        "Best MAE per domain and skip value (minimum over the three NAS optimizers).",
        "Each bar group = one skip value; four bars = four 3D domains.",
        "At skip=1 all domains reach 2.2-5.6 deg C. At skip=2 most stay below 11 deg C",
        "while saving 52% of FEM computation. At skip=6 up to 81% FEM savings but higher error.",
    ])

    p.footer(2)
    p.save(pdf)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Skip Timeline + Thermal Fields + Conclusions
# ═══════════════════════════════════════════════════════════════════════════════
def page3(pdf):
    p = Page()

    p.text("5.  Visualization and Conclusions",
           fs=11, bold=True, ha="center", x=0.5, sp=1.4)
    p.gap(0.005)
    p.hline(color="#666666", lw=1.0)
    p.gap(0.010)

    # Skip Timeline
    p.image("fig5_skip_timeline.png", height=0.270)
    p.gap(0.005)
    p.caption("Figure 2.", [
        "Skip Operator Timeline: MAE per time window (Rectangular domain, Bayesian NAS).",
        "Each line = one skip value (s=1,2,4,6). x-axis = time window index; y-axis = MAE (deg C).",
        "At skip=1 error is low and uniform. At skip=2/4 it grows moderately. At skip=6 it rises",
        "sharply in later windows, illustrating the accuracy-efficiency trade-off.",
    ])
    p.gap(0.012)

    # Thermal Fields
    p.image("fig1_thermal_fields.png", height=0.250)
    p.gap(0.005)
    p.caption("Figure 3.", [
        "Predicted temperature fields on the z-midplane (Turbo colormap: blue=20 C, yellow=540 C).",
        "All four 3D domains at a mid-simulation time step, skip=2, Bayesian NAS.",
        "Rectangular/Cylinder: smooth surface cooling. Stacked: interface effect. L-Shape: concave-corner.",
    ])
    p.gap(0.014)

    # Conclusions
    p.section(6, "Conclusions")
    conclusions = [
        ("Best accuracy (skip=1):",
         "2.19 deg C MAE on L-Shape with NSGA-II/III. All domains achieve < 5.6 deg C."),
        ("Practical optimum (skip=2):",
         "52% FEM savings while keeping MAE < 11 deg C for all domains (recommended default)."),
        ("Fast screening (skip=6):",
         "81% FEM reduction; MAE 5.6-22.5 deg C, suitable for rapid feasibility checks."),
        ("NAS comparison:",
         "Bayesian best on simple geometries; NSGA-II/III superior on the complex L-Shape."),
        ("MCO weighting:",
         "Adaptive w_i = 1/sigma_i^2 eliminates manual loss tuning across all geometries."),
    ]
    for bold, body in conclusions:
        p.gap(0.006)
        p.text(bold, fs=8.5, bold=True, color="#1a237e", sp=1.35)
        p.lines(wrap("    " + body, chars=106), fs=8.3, color="#333333", sp=1.4)

    p.footer(3)
    p.save(pdf)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating: {OUT}")
    with PdfPages(OUT) as pdf:
        page1(pdf); print("  Page 1 done")
        page2(pdf); print("  Page 2 done")
        page3(pdf); print("  Page 3 done")
        d = pdf.infodict()
        d["Title"]   = "NAS-MCO-PINNs Project Progress Report"
        d["Subject"] = "Skip Operator Results — 3D Thermal Simulation"
    print(f"Done: {OUT}")
