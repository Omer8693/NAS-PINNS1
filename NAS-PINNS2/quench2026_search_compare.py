#!/usr/bin/env python3
"""Compare search algorithms (NSGA2/NSGA3/Bayesian) on quench2026 results."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def to_int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_arch_key(key: str) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(key, str):
        return None, None
    if not key.startswith("L") or "_N" not in key:
        return None, None
    try:
        left, right = key.split("_N", 1)
        return int(left[1:]), int(right)
    except (TypeError, ValueError):
        return None, None


def read_search_population_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return max(0, len(rows) - 1)


def count_trial_dirs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_dir() and p.name.startswith("eval_"))


def find_best_cache_entry(cache: Dict, budget_cut: Optional[int] = None) -> Dict:
    best = {"proxy": None, "layers": None, "neurons": None, "eval_idx": None, "param_count": None}
    for key, val in cache.items():
        if not isinstance(val, dict):
            continue
        if val.get("status") != "ok":
            continue
        eval_idx = to_int(val.get("eval_idx"))
        if budget_cut is not None:
            if eval_idx is None or eval_idx > int(budget_cut):
                continue
        score = to_float(val.get("objective"))
        if score is None:
            continue
        if best["proxy"] is None or score < best["proxy"]:
            layers, neurons = parse_arch_key(key)
            best = {
                "proxy": score,
                "layers": layers,
                "neurons": neurons,
                "eval_idx": eval_idx,
                "param_count": to_float(val.get("param_count")),
            }
    return best


def collect_row(results_root: Path, method: str, budget_cut: Optional[int]) -> Dict:
    method_dir = results_root / method
    run_state = load_json(method_dir / "run_state.json", default={})
    search = run_state.get("search", {}) if isinstance(run_state, dict) else {}
    final = run_state.get("final", {}) if isinstance(run_state, dict) else {}

    cache = load_json(method_dir / "search_cache.json", default={})
    if not isinstance(cache, dict):
        cache = {}
    ok_count = sum(1 for v in cache.values() if isinstance(v, dict) and v.get("status") == "ok")
    fail_count = sum(1 for v in cache.values() if isinstance(v, dict) and v.get("status") == "failed")
    pop_rows = read_search_population_rows(method_dir / "search_population.csv")
    trial_evals = count_trial_dirs(method_dir / "search_trials")
    best_cache = find_best_cache_entry(cache, budget_cut=None)
    best_cache_cut = find_best_cache_entry(cache, budget_cut=budget_cut) if budget_cut else {}

    row = {
        "method": method,
        "status": "completed" if bool(run_state.get("final_done")) else "incomplete",
        "search_trial_evals": trial_evals,
        "search_population_rows": pop_rows,
        "search_unique_arch_ok": ok_count,
        "search_unique_arch_failed": fail_count,
        "search_best_layers": to_int(search.get("best_layers")),
        "search_best_neurons": to_int(search.get("best_neurons")),
        "search_best_proxy_loss": to_float(search.get("best_proxy_loss")),
        "search_best_param_count": to_float(search.get("best_param_count")),
        "cache_best_layers": best_cache["layers"],
        "cache_best_neurons": best_cache["neurons"],
        "cache_best_proxy_loss": best_cache["proxy"],
        "cache_best_eval_idx": best_cache["eval_idx"],
        "budget_cut": int(budget_cut) if budget_cut else "",
        "budget_cut_best_layers": best_cache_cut.get("layers"),
        "budget_cut_best_neurons": best_cache_cut.get("neurons"),
        "budget_cut_best_proxy_loss": best_cache_cut.get("proxy"),
        "budget_cut_best_eval_idx": best_cache_cut.get("eval_idx"),
        "budget_cut_best_param_count": best_cache_cut.get("param_count"),
        "final_best_stage": final.get("best_stage"),
        "final_best_objective": to_float(final.get("best_objective")),
        "final_run_time_seconds": to_float(final.get("run_time_seconds")),
        "proxy_rank": "",
        "proxy_rank_budget_cut": "",
        "final_rank": "",
        "is_proxy_winner": "0",
        "is_proxy_winner_budget_cut": "0",
        "is_final_winner": "0",
    }
    return row


def assign_rank(rows: List[Dict], metric: str, rank_col: str, winner_col: str) -> None:
    for r in rows:
        r[rank_col] = ""
        r[winner_col] = "0"
    valid = [r for r in rows if r.get(metric) is not None]
    valid.sort(key=lambda r: float(r[metric]))
    for i, row in enumerate(valid, start=1):
        row[rank_col] = i
    if valid:
        valid[0][winner_col] = "1"


def budget_fair_note(rows: List[Dict], budget_cut: Optional[int]) -> str:
    b_trials = sorted(set(int(r["search_trial_evals"]) for r in rows if r.get("search_trial_evals") is not None))
    b_unique = sorted(set(int(r["search_unique_arch_ok"]) for r in rows if r.get("search_unique_arch_ok") is not None))
    if len(b_trials) == 1 and len(b_unique) == 1:
        return "fair_by_trials_and_unique"
    if budget_cut and int(budget_cut) > 0:
        enough_budget = all(int(r.get("search_trial_evals") or 0) >= int(budget_cut) for r in rows)
        if enough_budget:
            return f"fair_by_budget_cut_{int(budget_cut)}"
        return f"budget_cut_{int(budget_cut)}_requested_but_missing_trials"
    return "not_fair_budget_mismatch"


def write_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "status",
        "search_trial_evals",
        "search_population_rows",
        "search_unique_arch_ok",
        "search_unique_arch_failed",
        "search_best_layers",
        "search_best_neurons",
        "search_best_proxy_loss",
        "search_best_param_count",
        "cache_best_layers",
        "cache_best_neurons",
        "cache_best_proxy_loss",
        "cache_best_eval_idx",
        "budget_cut",
        "budget_cut_best_layers",
        "budget_cut_best_neurons",
        "budget_cut_best_proxy_loss",
        "budget_cut_best_eval_idx",
        "budget_cut_best_param_count",
        "final_best_stage",
        "final_best_objective",
        "final_run_time_seconds",
        "proxy_rank",
        "proxy_rank_budget_cut",
        "final_rank",
        "is_proxy_winner",
        "is_proxy_winner_budget_cut",
        "is_final_winner",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(description="Search algorithm comparison for quench2026 pipeline")
    parser.add_argument("--results-root", type=str, default="results/quench2026/pipeline")
    parser.add_argument("--methods", type=str, default="nsga2,nsga3,bayesian")
    parser.add_argument("--budget-cut", type=int, default=0, help="Compare proxy quality at first N evals")
    parser.add_argument("--output-csv", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_root).resolve()
    budget_cut = int(args.budget_cut) if int(args.budget_cut) > 0 else None
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    rows = [collect_row(root, method, budget_cut) for method in methods]
    assign_rank(rows, metric="search_best_proxy_loss", rank_col="proxy_rank", winner_col="is_proxy_winner")
    if budget_cut:
        assign_rank(
            rows,
            metric="budget_cut_best_proxy_loss",
            rank_col="proxy_rank_budget_cut",
            winner_col="is_proxy_winner_budget_cut",
        )
    assign_rank(rows, metric="final_best_objective", rank_col="final_rank", winner_col="is_final_winner")
    fair_note = budget_fair_note(rows, budget_cut)

    out_csv = Path(args.output_csv).resolve() if args.output_csv else (root / "search_algorithm_comparison.csv")
    write_csv(rows, out_csv)

    print(f"Table written: {out_csv}")
    print(f"budget_check: {fair_note}")
    print("")
    for r in rows:
        print(
            f"{r['method']}: "
            f"trials={r['search_trial_evals']}, pop_rows={r['search_population_rows']}, unique_ok={r['search_unique_arch_ok']}, "
            f"proxy={r['search_best_proxy_loss']}, proxy@cut={r.get('budget_cut_best_proxy_loss')}, "
            f"final={r['final_best_objective']}, "
            f"proxy_rank={r['proxy_rank']}, proxy_rank@cut={r.get('proxy_rank_budget_cut')}, final_rank={r['final_rank']}"
        )


if __name__ == "__main__":
    main()
