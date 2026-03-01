#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-results/paper_strict_scratch/full_suite/${STAMP}}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"

mkdir -p "$BASE_OUT" "$LOG_DIR"

echo "Run root: $BASE_OUT"
echo "Log dir : $LOG_DIR"

run_job() {
  local method="$1"
  local equation="$2"

  local script="paper_strict_scratch/run_${method}.py"
  local out_dir="${BASE_OUT}/${equation}/${method}"
  local log_file="${LOG_DIR}/${equation}_${method}.log"

  mkdir -p "$out_dir"
  echo "============================================================" | tee -a "$log_file"
  echo "[RUN ] ${equation}/${method}" | tee -a "$log_file"
  echo "  out: ${out_dir}" | tee -a "$log_file"
  echo "  log: ${log_file}" | tee -a "$log_file"

  PYTHONUNBUFFERED=1 python "$script" \
    --equation "$equation" \
    --save-dir "$out_dir" \
    2>&1 | tee -a "$log_file"

  echo "[OK  ] ${equation}/${method}" | tee -a "$log_file"
}

# Paper-strict equations and methods.
equations=("burgers1d" "advection1d" "burgers2d")
methods=("naspinn" "nsga2" "nsga3" "bayesian")

for eq in "${equations[@]}"; do
  for m in "${methods[@]}"; do
    run_job "$m" "$eq"
  done
done

echo "============================================================"
echo "DONE. Outputs: $BASE_OUT"
echo "DONE. Logs   : $LOG_DIR"
