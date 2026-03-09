#!/usr/bin/env python3
"""Build a comparison table and winner summary for quench2026 experiments."""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional


ARCH_RE = re.compile(r"^L(?P<layers>\d+)_N(?P<neurons>\d+)$")


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float(value) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def to_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_arch(name: str):
    m = ARCH_RE.match(name)
    if not m:
        return None, None
    return int(m.group("layers")), int(m.group("neurons"))


def collect_baseline_rows(baseline_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not baseline_dir.exists():
        return rows
    for run_dir in sorted([p for p in baseline_dir.iterdir() if p.is_dir()]):
        meta = load_json(run_dir / "run_meta.json", default=None)
        if not isinstance(meta, dict):
            continue
        layers, neurons = parse_arch(run_dir.name)
        rows.append(
            {
                "method": "baseline",
                "variant": run_dir.name,
                "status": "completed",
                "best_layers": layers if layers is not None else to_int(meta.get("layers")),
                "best_neurons": neurons if neurons is not None else to_int(meta.get("base_neurons")),
                "param_count": to_int(meta.get("param_count")),
                "best_stage": meta.get("best_stage"),
                "best_objective": to_float(meta.get("best_objective")),
                "run_time_seconds": to_float(meta.get("run_time_seconds")),
                "final_dir": str(run_dir.resolve()),
            }
        )
    return rows


def _fallback_best_final(method_dir: Path) -> Dict:
    best = {}
    best_obj = None
    for meta_path in sorted(method_dir.glob("final/L*_N*/run_meta.json")):
        meta = load_json(meta_path, default=None)
        if not isinstance(meta, dict):
            continue
        obj = to_float(meta.get("best_objective"))
        if obj is None:
            continue
        if best_obj is None or obj < best_obj:
            best_obj = obj
            arch_dir = meta_path.parent.name
            layers, neurons = parse_arch(arch_dir)
            best = {
                "best_layers": layers if layers is not None else to_int(meta.get("layers")),
                "best_neurons": neurons if neurons is not None else to_int(meta.get("base_neurons")),
                "param_count": to_int(meta.get("param_count")),
                "best_stage": meta.get("best_stage"),
                "best_objective": obj,
                "run_time_seconds": to_float(meta.get("run_time_seconds")),
                "final_dir": str(meta_path.parent.resolve()),
            }
    return best


def collect_method_row(pipeline_dir: Path, method: str) -> Dict:
    method_dir = pipeline_dir / method
    state = load_json(method_dir / "run_state.json", default={})
    search = state.get("search", {}) if isinstance(state, dict) else {}
    final = state.get("final", {}) if isinstance(state, dict) else {}

    status = "missing"
    if method_dir.exists():
        status = "in_progress"
    if bool(state.get("search_done")):
        status = "search_done"
    if bool(state.get("final_done")):
        status = "completed"

    row = {
        "method": method,
        "variant": "",
        "status": status,
        "best_layers": to_int(search.get("best_layers")),
        "best_neurons": to_int(search.get("best_neurons")),
        "param_count": to_int(final.get("param_count")),
        "best_stage": final.get("best_stage"),
        "best_objective": to_float(final.get("best_objective")),
        "run_time_seconds": to_float(final.get("run_time_seconds")),
        "final_dir": final.get("final_dir") if isinstance(final, dict) else "",
    }

    # Fallback when run_state is absent/stale but final run_meta exists.
    if row["best_objective"] is None:
        fallback = _fallback_best_final(method_dir)
        if fallback:
            row.update(fallback)
            row["status"] = "completed"
    return row


def rank_rows(rows: List[Dict]) -> None:
    valid = [r for r in rows if r.get("best_objective") is not None and r.get("status") == "completed"]
    for r in rows:
        r["objective_rank"] = ""
        r["is_winner"] = "0"
    valid_sorted = sorted(
        valid,
        key=lambda r: (
            float(r["best_objective"]),
            float(r["param_count"]) if r.get("param_count") is not None else float("inf"),
            float(r["run_time_seconds"]) if r.get("run_time_seconds") is not None else float("inf"),
        ),
    )
    for i, row in enumerate(valid_sorted, start=1):
        row["objective_rank"] = i
    if valid_sorted:
        valid_sorted[0]["is_winner"] = "1"


def write_csv(rows: List[Dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "variant",
        "status",
        "best_layers",
        "best_neurons",
        "param_count",
        "best_stage",
        "best_objective",
        "run_time_seconds",
        "objective_rank",
        "is_winner",
        "final_dir",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def print_summary(rows: List[Dict], output_csv: Path) -> None:
    print(f"Table written: {output_csv}")
    print("")
    print("method,variant,status,best_layers,best_neurons,param_count,best_stage,best_objective,run_time_seconds,objective_rank,is_winner")
    for r in rows:
        print(
            f"{r.get('method','')},{r.get('variant','')},{r.get('status','')},"
            f"{r.get('best_layers','')},{r.get('best_neurons','')},{r.get('param_count','')},"
            f"{r.get('best_stage','')},{r.get('best_objective','')},{r.get('run_time_seconds','')},"
            f"{r.get('objective_rank','')},{r.get('is_winner','')}"
        )
    winners = [r for r in rows if r.get("is_winner") == "1"]
    print("")
    if winners:
        w = winners[0]
        print(
            "winner="
            f"{w.get('method')} "
            f"variant={w.get('variant') or '-'} "
            f"objective={w.get('best_objective')}"
        )
    else:
        print("winner=none (no completed runs with valid objective)")


def parse_args():
    parser = argparse.ArgumentParser(description="Quench2026 winner table builder")
    parser.add_argument("--results-root", type=str, default="results/quench2026")
    parser.add_argument("--methods", type=str, default="nsga2,nsga3,bayesian")
    parser.add_argument("--output-csv", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_root).resolve()
    baseline_dir = root / "baseline"
    pipeline_dir = root / "pipeline"
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]

    rows = []
    rows.extend(collect_baseline_rows(baseline_dir))
    for method in methods:
        rows.append(collect_method_row(pipeline_dir, method))

    rank_rows(rows)

    output_csv = Path(args.output_csv).resolve() if args.output_csv else (pipeline_dir / "winner_table.csv")
    write_csv(rows, output_csv)
    print_summary(rows, output_csv)


if __name__ == "__main__":
    main()
