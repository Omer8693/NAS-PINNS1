#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-results/paper_strict_scratch/full_suite/${STAMP}}"
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs}"
EPOCHS="${EPOCHS:-30000}"
LBFGS_MAX_ITER="${LBFGS_MAX_ITER:-6000}"
RESUME="${RESUME:-1}"
# Optional global overrides (empty => equation defaults)
PROXY_EPOCHS="${PROXY_EPOCHS:-}"
PSO_ITERS="${PSO_ITERS:-}"
PSO_SWARM="${PSO_SWARM:-}"
PSO_SPAN="${PSO_SPAN:-}"

mkdir -p "$BASE_OUT" "$LOG_DIR"

echo "Run root: $BASE_OUT"
echo "Log dir : $LOG_DIR"
echo "Epochs  : $EPOCHS"
echo "LBFGS   : $LBFGS_MAX_ITER"
echo "Resume  : $RESUME"

run_job() {
  local method="$1"
  local equation="$2"

  local script="paper_strict_scratch/run_${method}.py"
  local out_dir="${BASE_OUT}/${equation}/${method}"
  local log_file="${LOG_DIR}/${equation}_${method}.log"
  local summary_file="${out_dir}/summary_${equation}_${method}.csv"

  mkdir -p "$out_dir"
  if [[ "$RESUME" == "1" && -f "$summary_file" ]]; then
    echo "[SKIP] ${equation}/${method} already completed: ${summary_file}" | tee -a "$log_file"
    return 0
  fi

  echo "============================================================" | tee -a "$log_file"
  echo "[RUN ] ${equation}/${method}" | tee -a "$log_file"
  echo "  out: ${out_dir}" | tee -a "$log_file"
  echo "  log: ${log_file}" | tee -a "$log_file"

  local extra_args=(
    --equation "$equation"
    --save-dir "$out_dir"
    --epochs "$EPOCHS"
    --lbfgs-max-iter "$LBFGS_MAX_ITER"
  )

  if [[ -n "$PSO_ITERS" ]]; then
    extra_args+=(--pso-iters "$PSO_ITERS")
  fi
  if [[ -n "$PSO_SWARM" ]]; then
    extra_args+=(--pso-swarm "$PSO_SWARM")
  fi
  if [[ -n "$PSO_SPAN" ]]; then
    extra_args+=(--pso-span "$PSO_SPAN")
  fi
  if [[ "$method" != "naspinn" && -n "$PROXY_EPOCHS" ]]; then
    extra_args+=(--proxy-epochs "$PROXY_EPOCHS")
  fi

  PYTHONUNBUFFERED=1 python "$script" \
    "${extra_args[@]}" \
    2>&1 | tee -a "$log_file"

  echo "[OK  ] ${equation}/${method}" | tee -a "$log_file"
}

# Paper-strict equations and methods.
equations=("burgers1d" "advection1d" "burgers2d" "poisson")
methods=("naspinn" "nsga2" "nsga3" "bayesian")

for eq in "${equations[@]}"; do
  for m in "${methods[@]}"; do
    run_job "$m" "$eq"
  done
done

echo "============================================================"
echo "DONE. Outputs: $BASE_OUT"
echo "DONE. Logs   : $LOG_DIR"
