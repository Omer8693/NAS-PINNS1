"""
finalize_after_5k.py
====================
Run after eval_skip_5k.py completes.
  1. Adds 5k-epoch results to the skip comparison plot
  2. Updates cross-level summary table
  3. Re-saves the PowerPoint with updated data
"""
import json, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── 1. Check 5k results exist ─────────────────────────────────────────────────
p5k = ROOT / "level2_timestepper" / "results" / "skip_table_5k.json"
if not p5k.exists():
    print("skip_table_5k.json not ready yet — run eval_skip_5k.py first")
    raise SystemExit(1)

with open(p5k) as f:
    data5k = json.load(f)

# ── 2. Print summary ──────────────────────────────────────────────────────────
ref_path = ROOT / "level2_timestepper" / "results" / "skip_table.json"
with open(ref_path) as f:
    ref = json.load(f)

print("\n2000-Epoch Skip Results vs 500-Epoch Baseline")
print("=" * 60)
CONV = 1.5
for opt in ["nsga2", "nsga3"]:
    ref_s1 = {r["skip"]: r for r in ref.get(opt, [])}.get(1, {})
    ref_mae = ref_s1.get("mae_C", 999)
    print(f"\n{opt.upper()}  (ref skip=1 MAE={ref_mae:.1f}°C)")
    for row in data5k.get(opt, []):
        ratio = row["mae_C"] / (ref_mae + 1e-9)
        conv  = "✓ CONVERGED" if ratio < CONV else "✗ DIVERGED"
        print(f"  skip={row['skip']}  MAE={row['mae_C']:.1f}°C  "
              f"ratio={ratio:.2f}×  {conv}")

# ── 3. Update skip comparison plot ───────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

_FIG_BG = "#f7f1e3"; _AX_BG = "#fffdf8"; _SPINE_C = "#d8cfbf"
_TXT_C  = "#2f2a24"; _TICK_C = "#4c463f"

OPT_STYLE = {
    "bayesian": {"color": "#1565C0", "marker": "o", "ls": "-",
                 "label": "Bayesian NAS\n[5×151 relu]"},
    "nsga2":    {"color": "#C62828", "marker": "s", "ls": "--",
                 "label": "NSGA-II\n[3×153 tanh]"},
    "nsga3":    {"color": "#2E7D32", "marker": "^", "ls": ":",
                 "label": "NSGA-III\n[3×75 tanh]"},
}

def _style(ax):
    ax.set_facecolor(_AX_BG)
    for sp in ax.spines.values():
        sp.set_color(_SPINE_C); sp.set_linewidth(0.8)
    ax.tick_params(colors=_TICK_C, labelsize=8)
    ax.xaxis.label.set_color(_TICK_C); ax.yaxis.label.set_color(_TICK_C)

# Load all data
with open(ROOT / "level2_timestepper/results/skip_table.json") as f:
    l2 = json.load(f)
with open(ROOT / "level2_timestepper/results/skip_table_lbfgs.json") as f:
    l2l = json.load(f)
with open(ROOT / "level5_refinement/results/skip_table_l5.json") as f:
    l5 = json.load(f)

skip_vals = [1, 2, 4, 6]
opts      = ["bayesian", "nsga2", "nsga3"]
CONV_FAC  = 1.5

SERIES = [
    {"data": l2,   "label": "L2 Adam 500ep",        "ls": "-",   "marker": "o",  "lw": 2.2},
    {"data": l2l,  "label": "L2 Adam+LBFGS 300+30", "ls": "--",  "marker": "s",  "lw": 1.6},
    {"data": l5,   "label": "L5 Global PINN",        "ls": "-.",  "marker": "D",  "lw": 1.6},
]

# Add 5k data for nsga2/nsga3 only
data5k_full = {}
for opt in opts:
    if opt in data5k:
        data5k_full[opt] = data5k[opt]
    elif opt in l2:
        # Bayesian: use original L2 data as placeholder (not re-run)
        data5k_full[opt] = []

SERIES_5K = {"data": data5k_full, "label": "L2 Adam 2000ep\n(NSGA only)",
             "ls": (0,(3,1,1,1)), "marker": "*", "lw": 2.0}

fig = plt.figure(figsize=(18, 13), facecolor=_FIG_BG)
fig.suptitle(
    "Temporal Skip Capability: Four Approaches Compared\n"
    "L2 Adam-500  |  L2 Adam+L-BFGS  |  L5 Global PINN  |  L2 Adam-2000 (NSGA-II/III)",
    fontsize=11, color=_TXT_C, fontweight="bold", y=0.99,
)
gs = gridspec.GridSpec(3, 3, hspace=0.50, wspace=0.35,
                       left=0.07, right=0.97, top=0.93, bottom=0.03)

# Rows 0 & 1: MAE and L2
for col, opt in enumerate(opts):
    style = OPT_STYLE[opt]
    ref_mae_val = {r["skip"]: r for r in l2.get(opt,[])}.get(1,{}).get("mae_C", None)

    for row_idx, (metric_key, ylabel) in enumerate([("mae_C","MAE (°C)"),
                                                     ("l2_rel","L2 Relative Error")]):
        ax = fig.add_subplot(gs[row_idx, col])
        _style(ax)

        for ser in SERIES:
            rows_d = {r["skip"]: r for r in ser["data"].get(opt, [])}
            xs = [s for s in skip_vals if s in rows_d]
            ys = [rows_d[s][metric_key] for s in xs]
            if xs:
                ax.plot(xs, ys, marker=ser["marker"], color=style["color"],
                        ls=ser["ls"], lw=ser["lw"], ms=6, alpha=0.85,
                        label=ser["label"], zorder=5)

        # 5k series
        rows5k = {r["skip"]: r for r in data5k_full.get(opt, [])}
        xs5k = [s for s in [1,2] if s in rows5k]
        ys5k = [rows5k[s][metric_key] for s in xs5k]
        if xs5k:
            ax.plot(xs5k, ys5k, marker=SERIES_5K["marker"],
                    color=style["color"], ls=SERIES_5K["ls"],
                    lw=SERIES_5K["lw"], ms=9, alpha=1.0,
                    label=SERIES_5K["label"], zorder=6)

        if metric_key == "mae_C" and ref_mae_val:
            ax.axhline(ref_mae_val * CONV_FAC, color="#E65100", ls=":",
                       lw=1.3, alpha=0.6, label=f"Threshold (1.5×)")
        if metric_key == "l2_rel":
            ax.axhline(0.10, color="#E65100", ls=":", lw=1.3,
                       alpha=0.6, label="Target L2=0.10")

        if row_idx == 0:
            ax.set_title(style["label"], fontsize=9, color=_TXT_C, pad=5)
        ax.set_xlabel("Skip (k)", fontsize=9)
        ax.set_xticks(skip_vals)
        ax.set_xticklabels([f"skip={s}" for s in skip_vals], fontsize=7.5)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=9, color=_TICK_C)
        ax.legend(fontsize=6, loc="upper left", facecolor=_AX_BG,
                  edgecolor=_SPINE_C, ncol=1)

# Row 2: Summary table
ax_tbl = fig.add_subplot(gs[2, :])
ax_tbl.axis("off")

col_labels = ["Optimizer", "Approach", "skip=1 MAE",
              "skip=2 MAE", "skip=2 ✓/✗",
              "skip=4 MAE", "skip=4 ✓/✗"]

rows_all, rc_all = [], []
base_cols = {"bayesian":"#DCEEFA","nsga2":"#FDDEDE","nsga3":"#DFF3E0"}
lite_cols  = {"bayesian":"#EAF4FF","nsga2":"#FFF0F0","nsga3":"#F0FAF1"}
glob_cols  = {"bayesian":"#F0F4FF","nsga2":"#FFF5F5","nsga3":"#F5FFF5"}
fivek_cols = {"bayesian":"#FFFFFF","nsga2":"#FFFDE7","nsga3":"#F9FBE7"}

ser_triples = [
    ("L2 Adam-500",     l2,           base_cols),
    ("L2 Adam+LBFGS",   l2l,          lite_cols),
    ("L5 Global PINN",  l5,           glob_cols),
    ("L2 Adam-2000",    data5k_full,  fivek_cols),
]

for opt in opts:
    opt_lbl = OPT_STYLE[opt]["label"].replace("\n"," ")
    ref_m   = {r["skip"]:r for r in l2.get(opt,[])}.get(1,{}).get("mae_C", 999)
    for ser_lbl, ser_dat, ser_col in ser_triples:
        rows_d = {r["skip"]:r for r in ser_dat.get(opt,[])}
        if not rows_d:
            continue
        base = rows_d.get(1,{}).get("mae_C", float("nan"))
        row  = [opt_lbl, ser_lbl, f"{base:.1f}°C"]
        for sk in [2, 4]:
            r    = rows_d.get(sk, {})
            mae  = r.get("mae_C", float("nan"))
            ratio = mae / (ref_m + 1e-9)
            conv  = "✓" if ratio < CONV_FAC else "✗"
            row.append(f"{mae:.1f}°C" if not np.isnan(mae) else "—")
            row.append(f"{conv} {ratio:.2f}×" if not np.isnan(mae) else "—")
        rows_all.append(row)
        rc_all.append(ser_col[opt])

tbl = ax_tbl.table(cellText=rows_all, colLabels=col_labels,
                   cellLoc="center", loc="center", bbox=[0,0,1,1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1, 1.8)

for j in range(len(col_labels)):
    tbl[0,j].set_facecolor("#37474F")
    tbl[0,j].set_text_props(color="white", fontweight="bold")
for i, rc in enumerate(rc_all):
    for j in range(len(col_labels)):
        tbl[i+1,j].set_facecolor(rc)
    for j_c in [4, 6]:
        if j_c < len(col_labels):
            txt = rows_all[i][j_c]
            if txt.startswith("✓"):
                tbl[i+1,j_c].set_facecolor("#C8E6C9")
                tbl[i+1,j_c].set_text_props(color="#1B5E20", fontweight="bold")
            elif txt.startswith("✗"):
                tbl[i+1,j_c].set_facecolor("#FFCDD2")
                tbl[i+1,j_c].set_text_props(color="#B71C1C", fontweight="bold")

out_png = ROOT / "level5_refinement/results/level5_skip_comparison.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Updated plot: {out_png}")

# ── 4. Re-run PowerPoint generation ──────────────────────────────────────────
result = subprocess.run(
    ["python", str(ROOT / "make_presentation.py")],
    cwd=str(ROOT), capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)

print("\n✓ All done — skip_table_5k + updated plot + updated PPTX")
