#!/usr/bin/env bash
set -u -o pipefail

# Sequential batch runner for Advection + Burgers2D (NAS-PINN/NSGA2/NSGA3/Bayesian).
# - Keeps per-job logs
# - Writes a runner log and summary CSV
# - Supports resume via .done_ok marker
#
# Usage:
#   bash run_advection_burgers2d_paper_batch.sh [run_id] [seed] [mode]
#
# Args:
#   run_id : pipeline run directory id (default: current UTC timestamp)
#   seed   : base seed for all jobs (default: 42)
#   mode   : paper | single (default: paper)
#            paper  -> adds --paper-protocol
#            single -> single run (no --paper-protocol)
#
# Example (background):
#   nohup bash run_advection_burgers2d_paper_batch.sh 20260227_202723 42 paper \
#     > results/pipeline_runs/20260227_202723/logs/runner_advection_burgers2d.nohup.out 2>&1 &

RUN_ID="${1:-$(date -u +%Y%m%d_%H%M%S)}"
BASE_SEED="${2:-42}"
MODE="${3:-paper}"

if [[ "${MODE}" != "paper" && "${MODE}" != "single" ]]; then
  echo "Invalid mode: ${MODE} (expected: paper|single)" >&2
  exit 1
fi

RUN_ROOT="results/pipeline_runs/${RUN_ID}"
ART_ROOT="${RUN_ROOT}/artifacts/rep_01"
LOG_ROOT="${RUN_ROOT}/logs"
RUNNER_LOG="${LOG_ROOT}/runner_advection_burgers2d_paper_batch.out"
SUMMARY_CSV="${RUN_ROOT}/summary_advection_burgers2d.csv"

mkdir -p "${ART_ROOT}" "${LOG_ROOT}"

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  printf "equation,method,status,seed,start_utc,end_utc,duration_sec,save_dir,log_file\n" > "${SUMMARY_CSV}"
fi

log() {
  local msg="$1"
  printf "%s %s\n" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${msg}" | tee -a "${RUNNER_LOG}"
}

build_args() {
  local equation="$1"
  local method="$2"
  local save_dir="$3"
  local seed="$4"

  local args=("--seed" "${seed}" "--save-dir" "${save_dir}")

  if [[ "${MODE}" == "paper" ]]; then
    args+=("--paper-protocol")
  fi

  if [[ "${method}" != "naspinn" ]]; then
    args+=("--profile" "paper_baseline")
  fi

  printf "%s\n" "${args[@]}"
}

run_job() {
  local equation="$1"
  local method="$2"
  local script="$3"
  local seed="$4"

  local save_dir="${ART_ROOT}/${equation}/${method}/paper_baseline"
  local log_dir="${LOG_ROOT}/${equation}/${method}"
  local log_file="${log_dir}/rep_01_${MODE}.log"
  local done_file="${save_dir}/.done_ok"
  mkdir -p "${save_dir}" "${log_dir}"

  if [[ -f "${done_file}" ]]; then
    log "[SKIP] ${equation}/${method} (already done): ${save_dir}"
    return 0
  fi

  local start_ts end_ts duration rc
  start_ts="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  local start_epoch
  start_epoch="$(date +%s)"

  log "[RUN ] ${equation}/${method} seed=${seed} mode=${MODE}"
  local cmd=(python "${script}")
  while IFS= read -r arg; do
    cmd+=("${arg}")
  done < <(build_args "${equation}" "${method}" "${save_dir}" "${seed}")

  PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee -a "${RUNNER_LOG}" "${log_file}"
  rc=${PIPESTATUS[0]}

  end_ts="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  local end_epoch
  end_epoch="$(date +%s)"
  duration="$((end_epoch - start_epoch))"

  if [[ ${rc} -eq 0 ]]; then
    touch "${done_file}"
    log "[OK  ] ${equation}/${method} (${duration}s)"
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "${equation}" "${method}" "ok" "${seed}" "${start_ts}" "${end_ts}" "${duration}" "${save_dir}" "${log_file}" \
      >> "${SUMMARY_CSV}"
    return 0
  fi

  log "[FAIL] ${equation}/${method} rc=${rc} (${duration}s)"
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${equation}" "${method}" "fail(${rc})" "${seed}" "${start_ts}" "${end_ts}" "${duration}" "${save_dir}" "${log_file}" \
    >> "${SUMMARY_CSV}"
  return "${rc}"
}

declare -a JOBS=(
  "advection|naspinn|NAS_PINNs_advection.py"
  "advection|nsga2|NAS_PINNs_advection_nsga2.py"
  "advection|nsga3|NAS_PINNs_advection_nsga3.py"
  "advection|bayesian|NAS_PINNs_advection_bayesian.py"
  "burgers2d|naspinn|NAS_PINNs_burgers2d.py"
  "burgers2d|nsga2|NAS_PINNs_burgers2d_nsga2.py"
  "burgers2d|nsga3|NAS_PINNs_burgers2d_nsga3.py"
  "burgers2d|bayesian|NAS_PINNs_burgers2d_bayesian.py"
)

ok_count=0
fail_count=0

log "Run root : ${RUN_ROOT}"
log "Art root : ${ART_ROOT}"
log "Log root : ${LOG_ROOT}"
log "Mode     : ${MODE}"
log "Base seed: ${BASE_SEED}"

for entry in "${JOBS[@]}"; do
  IFS="|" read -r eq method script <<< "${entry}"
  if run_job "${eq}" "${method}" "${script}" "${BASE_SEED}"; then
    ok_count=$((ok_count + 1))
  else
    fail_count=$((fail_count + 1))
  fi
done

log "[DONE] ok=${ok_count} fail=${fail_count}"
log "Summary CSV: ${SUMMARY_CSV}"

if [[ ${fail_count} -gt 0 ]]; then
  exit 1
fi

