#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-results/pipeline_runs/20260303_222758_advection_burgers2d}"
PROFILE="${PROFILE:-paper_baseline}"
REPEATS="${REPEATS:-5}"
SEED="${SEED:-42}"

LOG_ROOT="${RUN_ROOT}/logs"
METHOD_DIR="${RUN_ROOT}/artifacts/burgers2d/nsga2"
LOG_FILE="${LOG_ROOT}/burgers2d_nsga2.log"
SUMMARY_FILE="${METHOD_DIR}/paper_protocol_summary.csv"

mkdir -p "${LOG_ROOT}" "${METHOD_DIR}"

is_run_complete() {
  local run_dir="$1"
  [[ -f "${run_dir}/metrics.csv" && -f "${run_dir}/run_time.txt" && -f "${run_dir}/search_summary.csv" ]]
}

write_summary_if_ready() {
  local run_id metrics_file rel_l2 run_time
  local -a metric_files=()

  for run_id in $(seq 1 "${REPEATS}"); do
    metrics_file="${METHOD_DIR}/run_$(printf "%02d" "${run_id}")/metrics.csv"
    if [[ ! -f "${metrics_file}" ]]; then
      echo "[INFO ] Summary bekliyor; eksik: ${metrics_file}"
      return 1
    fi
    metric_files+=("${metrics_file}")
  done

  {
    echo "run,rel_l2,run_time_seconds"
    for run_id in $(seq 1 "${REPEATS}"); do
      metrics_file="${METHOD_DIR}/run_$(printf "%02d" "${run_id}")/metrics.csv"
      read -r rel_l2 run_time < <(
        awk -F',' 'NR==2 {printf "%.8e %.6f\n", $4 + 0, $5 + 0}' "${metrics_file}"
      )
      printf "%d,%s,%s\n" "${run_id}" "${rel_l2}" "${run_time}"
    done

    local mean_rel std_rel
    read -r mean_rel std_rel < <(
      awk -F',' '
        FNR==2 {
          x = $4 + 0
          sum += x
          sumsq += x * x
          n += 1
        }
        END {
          if (n == 0) exit 1
          mean = sum / n
          var = sumsq / n - mean * mean
          if (var < 0) var = 0
          printf "%.8e %.8e\n", mean, sqrt(var)
        }
      ' "${metric_files[@]}"
    )
    printf "mean,%s,-\n" "${mean_rel}"
    printf "std,%s,-\n" "${std_rel}"
  } > "${SUMMARY_FILE}"

  echo "Saved summary: ${SUMMARY_FILE}"
}

echo "Run root : ${RUN_ROOT}"
echo "Profile  : ${PROFILE}"
echo "Repeats  : ${REPEATS}"
echo "Seed     : ${SEED}"
echo "Method   : burgers2d_nsga2"

for run_id in $(seq 1 "${REPEATS}"); do
  run_dir="${METHOD_DIR}/run_$(printf "%02d" "${run_id}")"
  mkdir -p "${run_dir}"

  if is_run_complete "${run_dir}"; then
    echo "[SKIP ] burgers2d_nsga2 run=${run_id} (already complete)"
    continue
  fi

  echo
  echo "================================================================"
  echo "[START] burgers2d_nsga2 run=${run_id}"
  echo "Command: ${PYTHON_BIN} NAS_PINNs_burgers2d_nsga2.py --profile ${PROFILE} --seed ${SEED} --paper-run-id ${run_id} --save-dir ${run_dir}"
  echo "Log    : ${LOG_FILE}"
  echo "================================================================"

  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" NAS_PINNs_burgers2d_nsga2.py \
    --profile "${PROFILE}" \
    --seed "${SEED}" \
    --paper-run-id "${run_id}" \
    --save-dir "${run_dir}" \
    2>&1 | tee -a "${LOG_FILE}"

  echo "[DONE ] burgers2d_nsga2 run=${run_id}"
done

write_summary_if_ready || true
echo "All requested burgers2d_nsga2 runs checked."
