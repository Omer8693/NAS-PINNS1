#!/usr/bin/env bash
set -euo pipefail

# Re-run Poisson NSGA-III + Bayesian in staged mode (adam/lbfgs/pso).
#
# Usage:
#   bash run_poisson_nsga3_bayesian_staged.sh [run_id] [base_seed]
# Example:
#   bash run_poisson_nsga3_bayesian_staged.sh 20260227_202723 42

RUN_ID="${1:-20260227_202723}"
BASE_SEED="${2:-42}"

RUN_ROOT="results/pipeline_runs/${RUN_ID}"
ART_ROOT="${RUN_ROOT}/artifacts/rep_01/poisson"
LOG_ROOT="${RUN_ROOT}/logs/poisson"
RUNNER_LOG="${LOG_ROOT}/runner_poisson_nsga3_bayesian_staged.out"

DOMAINS=(rectangular circle lshape flower annulus)
METHODS=(nsga3 bayesian)

mkdir -p "${ART_ROOT}" "${LOG_ROOT}"

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

script_for_method() {
  local method="$1"
  case "${method}" in
    nsga3) echo "NAS_PINNs_poisson_nsga3.py" ;;
    bayesian) echo "NAS_PINNs_poisson_bayesian.py" ;;
    *)
      echo "Unknown method: ${method}" >&2
      return 1
      ;;
  esac
}

stage_metrics_path() {
  local method="$1"
  local domain="$2"
  local stage="$3"
  echo "${ART_ROOT}/${method}/domain_${domain}/stage_${stage}/metrics.csv"
}

stage_is_done() {
  local metrics_file="$1"
  [[ -s "${metrics_file}" ]] || return 1
  # Expect header + at least one row.
  awk 'END { exit (NR > 1 ? 0 : 1) }' "${metrics_file}"
}

is_same_stage_running() {
  local script="$1"
  local domain="$2"
  local seed="$3"
  local stage_dir="$4"
  if pgrep -fa "python ${script} --domain ${domain} --seed ${seed}" | grep -Fq -- "--save-dir ${stage_dir}"; then
    return 0
  fi
  return 1
}

run_stage() {
  local method="$1"
  local script="$2"
  local domain="$3"
  local seed="$4"
  local stage="$5"

  local stage_dir="${ART_ROOT}/${method}/domain_${domain}/stage_${stage}"
  local stage_metrics
  stage_metrics="$(stage_metrics_path "${method}" "${domain}" "${stage}")"
  local log_dir="${LOG_ROOT}/poisson_${method}"
  local domain_log="${log_dir}/rep_01_stage_${domain}.log"

  mkdir -p "${stage_dir}" "${log_dir}"

  local flags=()
  case "${stage}" in
    adam) flags=(--skip-lbfgs --skip-pso) ;;
    lbfgs) flags=(--skip-pso) ;;
    pso) flags=() ;;
    *)
      echo "Unknown stage: ${stage}" | tee -a "${RUNNER_LOG}" "${domain_log}" >&2
      return 1
      ;;
  esac

  if stage_is_done "${stage_metrics}"; then
    echo "[SKIP] ${method}/${domain} stage_${stage} already completed." | tee -a "${RUNNER_LOG}" "${domain_log}"
    return 0
  fi

  if is_same_stage_running "${script}" "${domain}" "${seed}" "${stage_dir}"; then
    echo "[WAIT] ${method}/${domain} stage_${stage} is already running. Waiting..." | tee -a "${RUNNER_LOG}" "${domain_log}"
    while is_same_stage_running "${script}" "${domain}" "${seed}" "${stage_dir}"; do
      sleep 60
    done
    if stage_is_done "${stage_metrics}"; then
      echo "[DONE] ${method}/${domain} stage_${stage} completed by existing process." | tee -a "${RUNNER_LOG}" "${domain_log}"
      return 0
    fi
    echo "[WARN] Existing process ended without complete metrics. Restarting stage_${stage}..." | tee -a "${RUNNER_LOG}" "${domain_log}"
  fi

  echo "[RUN] ${method}/${domain} stage_${stage} (seed=${seed})" | tee -a "${RUNNER_LOG}" "${domain_log}"
  PYTHONUNBUFFERED=1 python "${script}" \
    --domain "${domain}" \
    --seed "${seed}" \
    --save-dir "${stage_dir}" \
    "${flags[@]}" 2>&1 | tee -a "${RUNNER_LOG}" "${domain_log}"

  if ! stage_is_done "${stage_metrics}"; then
    echo "[FAIL] ${method}/${domain} stage_${stage} finished but metrics are missing/incomplete: ${stage_metrics}" | tee -a "${RUNNER_LOG}" "${domain_log}" >&2
    return 1
  fi
}

finalize_domain() {
  local method="$1"
  local domain="$2"
  local domain_dir="${ART_ROOT}/${method}/domain_${domain}"
  local domain_log="${LOG_ROOT}/poisson_${method}/rep_01_stage_${domain}.log"

  python - "${domain_dir}" "${method}" <<'PY' 2>&1 | tee -a "${RUNNER_LOG}" "${domain_log}"
import csv
import os
import shutil
import sys

domain_dir = sys.argv[1]
method = sys.argv[2]
stages = ["adam", "lbfgs", "pso"]
rows = []

for stage in stages:
    metrics = os.path.join(domain_dir, f"stage_{stage}", "metrics.csv")
    if not os.path.exists(metrics):
        continue
    with open(metrics, "r", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    if data:
        rows.append((stage, data[-1]))

if not rows:
    raise SystemExit(f"No stage metrics found in {domain_dir}")

summary_csv = os.path.join(domain_dir, "stage_summary.csv")
with open(summary_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["stage", "method", "domain", "seed", "rel_l2", "run_time_seconds"])
    for stage, row in rows:
        w.writerow([stage, row["method"], row["domain"], row["seed"], row["rel_l2"], row["run_time_seconds"]])

best_stage, best_row = min(rows, key=lambda t: float(t[1]["rel_l2"]))
best_src = os.path.join(domain_dir, f"stage_{best_stage}")
best_dst = os.path.join(domain_dir, "stage_best")
if os.path.exists(best_dst):
    shutil.rmtree(best_dst)
shutil.copytree(best_src, best_dst)
with open(os.path.join(best_dst, "selected_stage.txt"), "w", encoding="utf-8") as f:
    f.write(f"{best_stage},{best_row['rel_l2']}\n")

print(f"[OK ] {method}/{best_row['domain']} best_stage={best_stage}")
PY
}

{
  echo "Run root : ${RUN_ROOT}"
  echo "Art root : ${ART_ROOT}"
  echo "Log root : ${LOG_ROOT}"
  echo "Base seed: ${BASE_SEED}"
  echo
} | tee -a "${RUNNER_LOG}"

for method in "${METHODS[@]}"; do
  script="$(script_for_method "${method}")"
  for domain in "${DOMAINS[@]}"; do
    seed="$(domain_seed "${domain}")"
    run_stage "${method}" "${script}" "${domain}" "${seed}" "adam"
    run_stage "${method}" "${script}" "${domain}" "${seed}" "lbfgs"
    run_stage "${method}" "${script}" "${domain}" "${seed}" "pso"
    finalize_domain "${method}" "${domain}"
  done
done

echo "[DONE] Poisson NSGA3 + Bayesian staged re-run completed." | tee -a "${RUNNER_LOG}"
