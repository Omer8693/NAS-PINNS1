#!/usr/bin/env python3
"""Generate tables/plots/report for quench2026 comparison experiments."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("nsga2", "nsga3", "bayesian")
STAGES = ("adam", "lbfgs", "pso")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stage_map_from_csv(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    df = pd.read_csv(path)
    if "stage" not in df.columns or "objective" not in df.columns:
        return out
    for _, row in df.iterrows():
        stage = str(row["stage"]).strip().lower()
        try:
            out[stage] = float(row["objective"])
        except (TypeError, ValueError):
            continue
    return out


def first_arch_dir(parent: Path) -> Optional[Path]:
    if not parent.exists():
        return None
    dirs = sorted([p for p in parent.glob("L*_N*") if p.is_dir()])
    return dirs[0] if dirs else None


def build_pipeline_stage_matrix(results_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []

    # Baseline
    base_stage_csv = results_root / "baseline" / "L5_N96" / "stage_summary.csv"
    base_map = stage_map_from_csv(base_stage_csv)
    rows.append(
        {
            "method": "baseline",
            "arch": "L5_N96",
            "adam": base_map.get("adam"),
            "lbfgs": base_map.get("lbfgs"),
            "pso": base_map.get("pso"),
        }
    )

    # Pipeline finals
    cmp_df = read_csv(results_root / "pipeline" / "comparison.csv")
    for method in METHODS:
        row = cmp_df[cmp_df["method"] == method]
        if row.empty:
            continue
        layers = int(float(row.iloc[0]["best_layers"]))
        neurons = int(float(row.iloc[0]["best_neurons"]))
        arch = f"L{layers}_N{neurons}"
        stage_csv = results_root / "pipeline" / method / "final" / arch / "stage_summary.csv"
        s_map = stage_map_from_csv(stage_csv)
        rows.append(
            {
                "method": method,
                "arch": arch,
                "adam": s_map.get("adam"),
                "lbfgs": s_map.get("lbfgs"),
                "pso": s_map.get("pso"),
            }
        )

    out = pd.DataFrame(rows)
    return out


def build_refine5000_stage_matrix(results_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []

    # Baseline refine5000 (lbfgs + pso are in separate folders)
    b_lbfgs_csv = results_root / "baseline_adam_refine5000" / "lbfgs" / "L5_N96" / "stage_summary.csv"
    b_pso_csv = results_root / "baseline_adam_refine5000" / "pso" / "L5_N96" / "stage_summary.csv"
    b_l = stage_map_from_csv(b_lbfgs_csv)
    b_p = stage_map_from_csv(b_pso_csv)
    rows.append(
        {
            "method": "baseline",
            "arch": "L5_N96",
            "adam": b_l.get("adam", b_p.get("adam")),
            "lbfgs": b_l.get("lbfgs"),
            "pso": b_p.get("pso"),
        }
    )

    # Method refine5000 outputs
    for method in METHODS:
        lbfgs_parent = results_root / "best_adam_lbfgs5000" / "lbfgs" / method
        pso_parent = results_root / "best_adam_lbfgs5000" / "pso" / method
        lbfgs_arch = first_arch_dir(lbfgs_parent)
        pso_arch = first_arch_dir(pso_parent)
        if lbfgs_arch is None and pso_arch is None:
            continue

        l_map = stage_map_from_csv(lbfgs_arch / "stage_summary.csv") if lbfgs_arch else {}
        p_map = stage_map_from_csv(pso_arch / "stage_summary.csv") if pso_arch else {}
        arch = lbfgs_arch.name if lbfgs_arch else pso_arch.name
        rows.append(
            {
                "method": method,
                "arch": arch,
                "adam": l_map.get("adam", p_map.get("adam")),
                "lbfgs": l_map.get("lbfgs"),
                "pso": p_map.get("pso"),
            }
        )

    return pd.DataFrame(rows)


def plot_heatmap(df: pd.DataFrame, title: str, out_png: Path) -> None:
    if df.empty:
        return
    plot_df = df.set_index("method")[["adam", "lbfgs", "pso"]].astype(float)

    # Log scale to keep PSO values from dominating the color map.
    mat = np.log10(np.clip(plot_df.to_numpy(dtype=float), 1e-16, None))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(objective)")

    ax.set_xticks(np.arange(len(STAGES)))
    ax.set_xticklabels(STAGES)
    ax.set_yticks(np.arange(plot_df.shape[0]))
    ax.set_yticklabels(plot_df.index.tolist())
    ax.set_title(title)
    ax.set_xlabel("stage")
    ax.set_ylabel("method")

    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            val = plot_df.iloc[i, j]
            if np.isnan(val):
                txt = "NA"
            else:
                txt = f"{val:.2e}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_runtime_vs_objective(df: pd.DataFrame, out_png: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    markers = {
        "baseline": "o",
        "baseline_refine5000": "s",
        "pipeline_final": "^",
        "method_refine5000": "D",
    }
    for group, gdf in df.groupby("group"):
        ax.scatter(
            gdf["run_time_seconds"],
            gdf["best_objective"],
            label=group,
            marker=markers.get(group, "o"),
            s=52,
            alpha=0.9,
        )
        for _, row in gdf.iterrows():
            ax.annotate(
                row["scenario"],
                (row["run_time_seconds"], row["best_objective"]),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=7,
            )
    ax.set_yscale("log")
    ax.set_xlabel("run_time_seconds")
    ax.set_ylabel("best_objective (log scale, lower is better)")
    ax.set_title("Runtime vs Objective")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_search_pareto(results_root: Path, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    any_data = False
    for method in METHODS:
        pop_csv = results_root / "pipeline" / method / "search_population.csv"
        df = read_csv(pop_csv)
        if df.empty:
            continue
        needed = {"obj_proxy_loss", "obj_param_count"}
        if not needed.issubset(set(df.columns)):
            continue
        any_data = True
        ax.scatter(
            df["obj_param_count"],
            df["obj_proxy_loss"],
            s=28,
            alpha=0.75,
            label=method,
        )
    if not any_data:
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.set_xlabel("param_count")
    ax.set_ylabel("proxy_loss (log scale)")
    ax.set_title("Search Pareto Cloud")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_adam_convergence(results_root: Path, out_png: Path) -> None:
    runs = {
        "baseline": results_root / "baseline" / "L5_N96" / "metrics.csv",
        "nsga2_pipeline_final": results_root / "pipeline" / "nsga2" / "final" / "L6_N132" / "metrics.csv",
        "nsga3_pipeline_final": results_root / "pipeline" / "nsga3" / "final" / "L6_N141" / "metrics.csv",
        "bayesian_pipeline_final": results_root / "pipeline" / "bayesian" / "final" / "L5_N121" / "metrics.csv",
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    any_data = False
    for name, path in runs.items():
        df = read_csv(path)
        if df.empty or "epoch" not in df.columns or "total" not in df.columns:
            continue
        any_data = True
        ax.plot(df["epoch"], np.clip(df["total"], 1e-16, None), label=name, linewidth=1.3)
    if not any_data:
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("total_loss (Adam phase, log)")
    ax.set_title("Adam Convergence Curves")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def build_runtime_table(results_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []

    def push(group: str, scenario: str, method: str, run_dir: Path):
        meta = read_json(run_dir / "run_meta.json")
        if not isinstance(meta, dict):
            return
        rows.append(
            {
                "group": group,
                "scenario": scenario,
                "method": method,
                "run_dir": str(run_dir.resolve()),
                "layers": meta.get("layers"),
                "neurons": meta.get("base_neurons"),
                "param_count": meta.get("param_count"),
                "best_stage": meta.get("best_stage"),
                "best_objective": meta.get("best_objective"),
                "run_time_seconds": meta.get("run_time_seconds"),
            }
        )

    # Baseline family
    push("baseline", "baseline_original", "baseline", results_root / "baseline" / "L5_N96")
    push(
        "baseline_refine5000",
        "baseline_from_adam_lbfgs5000",
        "baseline",
        results_root / "baseline_adam_refine5000" / "lbfgs" / "L5_N96",
    )
    push(
        "baseline_refine5000",
        "baseline_from_adam_pso",
        "baseline",
        results_root / "baseline_adam_refine5000" / "pso" / "L5_N96",
    )

    # Pipeline finals
    cmp_df = read_csv(results_root / "pipeline" / "comparison.csv")
    for method in METHODS:
        row = cmp_df[cmp_df["method"] == method]
        if row.empty:
            continue
        arch = f"L{int(float(row.iloc[0]['best_layers']))}_N{int(float(row.iloc[0]['best_neurons']))}"
        push(
            "pipeline_final",
            f"{method}_pipeline_final",
            method,
            results_root / "pipeline" / method / "final" / arch,
        )

    # Method refine5000: lbfgs and pso
    for method in METHODS:
        lbfgs_arch = first_arch_dir(results_root / "best_adam_lbfgs5000" / "lbfgs" / method)
        pso_arch = first_arch_dir(results_root / "best_adam_lbfgs5000" / "pso" / method)
        if lbfgs_arch is not None:
            push("method_refine5000", f"{method}_lbfgs5000", method, lbfgs_arch)
        if pso_arch is not None:
            push("method_refine5000", f"{method}_pso", method, pso_arch)

    return pd.DataFrame(rows)


def load_fair_same_arch_tables(results_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fair_dir = results_root / "fair_same_arch_refine"
    per_seed_paths = sorted(fair_dir.glob("L*_N*_per_seed.csv"))
    agg_paths = sorted(fair_dir.glob("L*_N*_aggregate_mean_std.csv"))

    per_seed_all: List[pd.DataFrame] = []
    agg_all: List[pd.DataFrame] = []

    for p in per_seed_paths:
        df = read_csv(p)
        if df.empty:
            continue
        df["arch"] = p.name.replace("_per_seed.csv", "")
        per_seed_all.append(df)

    for p in agg_paths:
        df = read_csv(p)
        if df.empty:
            continue
        df["arch"] = p.name.replace("_aggregate_mean_std.csv", "")
        agg_all.append(df)

    per_seed_df = pd.concat(per_seed_all, ignore_index=True) if per_seed_all else pd.DataFrame()
    agg_df = pd.concat(agg_all, ignore_index=True) if agg_all else pd.DataFrame()
    return per_seed_df, agg_df


def append_fair_runtime_rows(runtime_df: pd.DataFrame, fair_per_seed_df: pd.DataFrame) -> pd.DataFrame:
    if fair_per_seed_df.empty:
        return runtime_df
    rows: List[Dict] = []
    for _, row in fair_per_seed_df.iterrows():
        seed = int(row["seed"])
        arch = str(row["arch"])
        layers = int(row.get("layers", np.nan))
        neurons = int(row.get("neurons", np.nan))
        rows.extend(
            [
                {
                    "group": "fair_same_arch",
                    "scenario": f"fair_seed{seed}_adam",
                    "method": "fixed_arch",
                    "run_dir": "",
                    "layers": layers,
                    "neurons": neurons,
                    "param_count": np.nan,
                    "best_stage": "adam",
                    "best_objective": float(row["adam_obj"]),
                    "run_time_seconds": float(row["adam_runtime_s"]),
                    "arch": arch,
                },
                {
                    "group": "fair_same_arch",
                    "scenario": f"fair_seed{seed}_lbfgs",
                    "method": "fixed_arch",
                    "run_dir": "",
                    "layers": layers,
                    "neurons": neurons,
                    "param_count": np.nan,
                    "best_stage": "lbfgs",
                    "best_objective": float(row["lbfgs_obj"]),
                    "run_time_seconds": float(row["lbfgs_runtime_s"]),
                    "arch": arch,
                },
                {
                    "group": "fair_same_arch",
                    "scenario": f"fair_seed{seed}_pso",
                    "method": "fixed_arch",
                    "run_dir": "",
                    "layers": layers,
                    "neurons": neurons,
                    "param_count": np.nan,
                    "best_stage": "pso",
                    "best_objective": float(row["pso_obj"]),
                    "run_time_seconds": float(row["pso_runtime_s"]),
                    "arch": arch,
                },
            ]
        )
    add_df = pd.DataFrame(rows)
    if runtime_df.empty:
        return add_df
    # Align columns and append.
    for col in runtime_df.columns:
        if col not in add_df.columns:
            add_df[col] = np.nan
    for col in add_df.columns:
        if col not in runtime_df.columns:
            runtime_df[col] = np.nan
    return pd.concat([runtime_df, add_df[runtime_df.columns]], ignore_index=True)


def plot_fair_same_arch_objective(per_seed_df: pd.DataFrame, out_png: Path) -> None:
    if per_seed_df.empty:
        return
    rows: List[Dict] = []
    for _, row in per_seed_df.iterrows():
        rows.append({"seed": int(row["seed"]), "stage": "adam", "objective": float(row["adam_obj"])})
        rows.append({"seed": int(row["seed"]), "stage": "lbfgs", "objective": float(row["lbfgs_obj"])})
        rows.append({"seed": int(row["seed"]), "stage": "pso", "objective": float(row["pso_obj"])})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    stage_order = ["adam", "lbfgs", "pso"]
    data = [df[df["stage"] == s]["objective"].to_numpy(dtype=float) for s in stage_order]
    ax.boxplot(data, tick_labels=stage_order, showmeans=True)
    ax.set_yscale("log")
    ax.set_ylabel("objective (log scale)")
    ax.set_title("Fair Same-Arch: Objective Distribution Across Seeds")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_fair_same_arch_runtime(per_seed_df: pd.DataFrame, out_png: Path) -> None:
    if per_seed_df.empty:
        return
    stage_cols = {
        "adam": "adam_runtime_s",
        "lbfgs": "lbfgs_runtime_s",
        "pso": "pso_runtime_s",
    }
    stages = list(stage_cols.keys())
    means = [float(per_seed_df[stage_cols[s]].astype(float).mean()) for s in stages]
    stds = [float(per_seed_df[stage_cols[s]].astype(float).std()) for s in stages]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(stages, means, yerr=stds, capsize=5)
    ax.set_ylabel("runtime (seconds)")
    ax.set_title("Fair Same-Arch: Runtime Mean±Std Across Seeds")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_markdown_report(
    out_md: Path,
    quality_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    fair_per_seed_df: pd.DataFrame,
    fair_agg_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Quench2026 Report Pack")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- `tables/quality_summary.csv`")
    lines.append(f"- `tables/runtime_summary.csv`")
    lines.append(f"- `tables/pipeline_stage_matrix.csv`")
    lines.append(f"- `tables/refine5000_stage_matrix.csv`")
    lines.append(f"- `plots/pipeline_stage_heatmap_log10.png`")
    lines.append(f"- `plots/refine5000_stage_heatmap_log10.png`")
    lines.append(f"- `plots/search_pareto_cloud.png`")
    lines.append(f"- `plots/runtime_vs_objective.png`")
    lines.append(f"- `plots/adam_convergence_curves.png`")
    if not fair_per_seed_df.empty:
        lines.append(f"- `tables/fair_same_arch_per_seed.csv`")
        lines.append(f"- `tables/fair_same_arch_aggregate.csv`")
        lines.append(f"- `plots/fair_same_arch_objective_boxplot.png`")
        lines.append(f"- `plots/fair_same_arch_runtime_bar.png`")
    lines.append("")

    if not quality_df.empty:
        top = quality_df.iloc[0]
        lines.append("## Top Result")
        lines.append(
            f"- best method: `{top['method']}` | arch `{top['best_architecture']}` | "
            f"best_after_refine_obj `{top['best_after_refine_obj']:.6g}`"
        )
        lines.append("")
        lines.append("## Quality Table")
        lines.append(quality_df.to_markdown(index=False))
        lines.append("")

    if not runtime_df.empty:
        lines.append("## Runtime Table")
        lines.append(runtime_df.to_markdown(index=False))
        lines.append("")

    if not fair_per_seed_df.empty:
        lines.append("## Fair Same-Architecture (Strict) Per-Seed")
        lines.append(fair_per_seed_df.to_markdown(index=False))
        lines.append("")
    if not fair_agg_df.empty:
        lines.append("## Fair Same-Architecture Aggregate")
        lines.append(fair_agg_df.to_markdown(index=False))
        lines.append("")

    lines.append("## Note")
    lines.append(
        "- `quality_score_vs_best_pct` objective tabanli bir skordur; klasik classification accuracy degildir."
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Build quench2026 tables/plots/report")
    parser.add_argument("--results-root", type=str, default="results/quench2026")
    parser.add_argument("--out-dir", type=str, default="results/quench2026/report_pack")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    results_root = (repo_root / args.results_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    plot_dir = out_dir / "plots"
    table_dir = out_dir / "tables"
    ensure_dir(plot_dir)
    ensure_dir(table_dir)

    # 1) Quality summary from best_architectures table
    quality_in = read_csv(results_root / "best_architectures_summary_ranked.csv")
    quality_df = quality_in.copy()
    if not quality_df.empty:
        quality_df["best_after_refine_obj"] = pd.to_numeric(quality_df["best_after_refine_obj"], errors="coerce")
        best_val = float(quality_df["best_after_refine_obj"].min())
        quality_df["quality_score_vs_best_pct"] = (
            (best_val / quality_df["best_after_refine_obj"]) * 100.0
        ).round(3)
        quality_df = quality_df.sort_values("best_after_refine_obj", ascending=True).reset_index(drop=True)
    quality_df.to_csv(table_dir / "quality_summary.csv", index=False)

    # 2) Stage matrices + heatmaps
    pipe_stage = build_pipeline_stage_matrix(results_root)
    ref_stage = build_refine5000_stage_matrix(results_root)
    pipe_stage.to_csv(table_dir / "pipeline_stage_matrix.csv", index=False)
    ref_stage.to_csv(table_dir / "refine5000_stage_matrix.csv", index=False)
    plot_heatmap(pipe_stage, "Pipeline Stage Heatmap (log10 objective)", plot_dir / "pipeline_stage_heatmap_log10.png")
    plot_heatmap(ref_stage, "Refine5000 Stage Heatmap (log10 objective)", plot_dir / "refine5000_stage_heatmap_log10.png")

    # 3) Search pareto
    plot_search_pareto(results_root, plot_dir / "search_pareto_cloud.png")

    # 4) Runtime summary + scatter
    runtime_df = build_runtime_table(results_root)

    # 4b) Fair same-arch strict tables (if available)
    fair_per_seed_df, fair_agg_df = load_fair_same_arch_tables(results_root)
    if not fair_per_seed_df.empty:
        fair_per_seed_df = fair_per_seed_df.copy()
        fair_per_seed_df["seed"] = pd.to_numeric(fair_per_seed_df["seed"], errors="coerce").astype("Int64")
        fair_per_seed_df.to_csv(table_dir / "fair_same_arch_per_seed.csv", index=False)
        runtime_df = append_fair_runtime_rows(runtime_df, fair_per_seed_df)
        plot_fair_same_arch_objective(fair_per_seed_df, plot_dir / "fair_same_arch_objective_boxplot.png")
        plot_fair_same_arch_runtime(fair_per_seed_df, plot_dir / "fair_same_arch_runtime_bar.png")

    if not fair_agg_df.empty:
        fair_agg_df.to_csv(table_dir / "fair_same_arch_aggregate.csv", index=False)

    if not runtime_df.empty:
        runtime_df["best_objective"] = pd.to_numeric(runtime_df["best_objective"], errors="coerce")
        runtime_df["run_time_seconds"] = pd.to_numeric(runtime_df["run_time_seconds"], errors="coerce")
        runtime_df = runtime_df.sort_values("best_objective", ascending=True).reset_index(drop=True)
    runtime_df.to_csv(table_dir / "runtime_summary.csv", index=False)
    plot_runtime_vs_objective(runtime_df, plot_dir / "runtime_vs_objective.png")

    # 5) Adam convergence
    plot_adam_convergence(results_root, plot_dir / "adam_convergence_curves.png")

    # 6) Markdown report
    report_md = out_dir / "report.md"
    write_markdown_report(report_md, quality_df, runtime_df, fair_per_seed_df, fair_agg_df, out_dir)

    print(f"Report pack written: {out_dir}")
    print(f"Tables: {table_dir}")
    print(f"Plots:  {plot_dir}")
    print(f"Report: {report_md}")


if __name__ == "__main__":
    main()
