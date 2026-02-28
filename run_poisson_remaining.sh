#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_poisson_remaining.sh [run_id] [base_seed]
# Example:
#   bash run_poisson_remaining.sh 20260227_202723 42

RUN_ID="${1:-20260227_202723}"
BASE_SEED="${2:-42}"

RUN_ROOT="results/pipeline_runs/${RUN_ID}"
POISSON_ROOT="${RUN_ROOT}/artifacts/rep_01/poisson"
LOG_ROOT="${RUN_ROOT}/logs/poisson"

DOMAINS=(rectangular circle lshape flower annulus)

mkdir -p "${LOG_ROOT}"

domain_seed() {
  local domain="$1"
  local i
  for i in "${!DOMAINS[@]}"; do
    if [[ "${DOMAINS[$i]}" == "${domain}" ]]; then
      echo $((BASE_SEED + i))
      return 0
    fi
  done
  echo "Unknown domain: ${domain}" >&2
  return 1
}

is_same_job_running() {
  local script="$1"
  local domain="$2"
  local seed="$3"
  local out_dir="$4"

  if pgrep -fa "python ${script} --domain ${domain} --seed ${seed}" | grep -Fq -- "--save-dir ${out_dir}"; then
    return 0
  fi
  return 1
}

run_if_needed() {
  local method="$1"
  local script="$2"
  local domain="$3"
  local seed="$4"
  shift 4
  local extra_args=("$@")

  local out_dir="${POISSON_ROOT}/${method}/domain_${domain}"
  local done_file="${out_dir}/metrics.csv"
  local log_dir="${LOG_ROOT}/poisson_${method}"
  local log_file="${log_dir}/rep_01_pso_${domain}.log"

  mkdir -p "${out_dir}" "${log_dir}"

  if [[ -f "${done_file}" ]]; then
    echo "[SKIP] ${method}/${domain} already completed."
    return 0
  fi

  if is_same_job_running "${script}" "${domain}" "${seed}" "${out_dir}"; then
    echo "[WAIT] ${method}/${domain} is already running. Waiting for completion..."
    while is_same_job_running "${script}" "${domain}" "${seed}" "${out_dir}"; do
      sleep 60
    done
    if [[ -f "${done_file}" ]]; then
      echo "[DONE] ${method}/${domain} completed by existing process."
      return 0
    fi
    echo "[WARN] Existing process ended but ${done_file} not found. Restarting..."
  fi

  echo "[RUN ] ${method}/${domain} (seed=${seed})"
  (
    set -x
    PYTHONUNBUFFERED=1 python "${script}" \
      --domain "${domain}" \
      --seed "${seed}" \
      --save-dir "${out_dir}" \
      "${extra_args[@]}"
  ) 2>&1 | tee -a "${log_file}"

  if [[ ! -f "${done_file}" ]]; then
    echo "[FAIL] ${method}/${domain} finished without ${done_file}" >&2
    return 1
  fi
  echo "[OK  ] ${method}/${domain}"
}

echo "Run root   : ${RUN_ROOT}"
echo "Poisson dir: ${POISSON_ROOT}"
echo "Log dir    : ${LOG_ROOT}"
echo "Base seed  : ${BASE_SEED}"
echo

# 1) NSGA-II remaining (pso active by default unless --skip-pso is provided)
for d in "${DOMAINS[@]}"; do
  s="$(domain_seed "${d}")"
  run_if_needed "nsga2" "NAS_PINNs_poisson_nsga2.py" "${d}" "${s}"
done

# 2) NSGA-III full/remaining
for d in "${DOMAINS[@]}"; do
  s="$(domain_seed "${d}")"
  run_if_needed "nsga3" "NAS_PINNs_poisson_nsga3.py" "${d}" "${s}"
done

# 3) Bayesian full/remaining
for d in "${DOMAINS[@]}"; do
  s="$(domain_seed "${d}")"
  run_if_needed "bayesian" "NAS_PINNs_poisson_bayesian.py" "${d}" "${s}"
done

echo
echo "All requested Poisson remaining jobs are completed."
