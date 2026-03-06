#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PROFILE="${PROFILE:-paper_baseline}"     # paper_baseline | ours_fast
REPEATS="${REPEATS:-5}"
SEED="${SEED:-42}"
BETA_LIST="${BETA_LIST:-1.0,0.5,0.1}"
FORCE_NEW_RUN="${FORCE_NEW_RUN:-0}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"          # 1: fail-fast if CUDA unavailable, 0: allow CPU fallback
RUN_FAMILIES="${RUN_FAMILIES:-advection,burgers2d}"
ADVECTION_METHODS="${ADVECTION_METHODS:-naspinn,nsga2,nsga3,bayesian}"
BURGERS2D_METHODS="${BURGERS2D_METHODS:-naspinn,nsga2,nsga3,bayesian}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
if [[ -z "${RUN_ROOT:-}" ]]; then
  latest_run=""
  if compgen -G "results/pipeline_runs/*_advection_burgers2d" > /dev/null; then
    latest_run="$(ls -1dt results/pipeline_runs/*_advection_burgers2d | head -n 1)"
  fi
  if [[ "${FORCE_NEW_RUN}" == "1" || -z "${latest_run}" ]]; then
    RUN_ROOT="results/pipeline_runs/${STAMP}_advection_burgers2d"
  else
    RUN_ROOT="${latest_run}"
  fi
fi
ARTIFACT_ROOT="${RUN_ROOT}/artifacts"
LOG_ROOT="${RUN_ROOT}/logs"

mkdir -p "${ARTIFACT_ROOT}" "${LOG_ROOT}"

run_job() {
  local name="$1"
  shift
  local log_file="${LOG_ROOT}/${name}.log"

  echo
  echo "================================================================"
  echo "[START] ${name}"
  echo "Command: $*"
  echo "Log    : ${log_file}"
  echo "================================================================"

  # Force unbuffered Python stdout/stderr so progress appears in logs immediately.
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee -a "${log_file}"

  echo "[DONE ] ${name}"
}

is_run_complete() {
  local run_dir="$1"
  shift
  local rel_path
  for rel_path in "$@"; do
    if [[ ! -f "${run_dir}/${rel_path}" ]]; then
      return 1
    fi
  done
  return 0
}

parse_beta_values() {
  BETA_VALUES=()
  local beta_raw beta_clean
  local -a raw_values=()
  IFS=',' read -r -a raw_values <<< "${BETA_LIST}"
  for beta_raw in "${raw_values[@]}"; do
    beta_clean="${beta_raw//[[:space:]]/}"
    if [[ -n "${beta_clean}" ]]; then
      BETA_VALUES+=("${beta_clean}")
    fi
  done
  if [[ "${#BETA_VALUES[@]}" -eq 0 ]]; then
    echo "ERROR: BETA_LIST does not contain any usable values: ${BETA_LIST}" >&2
    exit 1
  fi
}

csv_has() {
  local csv="$1"
  local target="$2"
  local item
  local -a values=()
  IFS=',' read -r -a values <<< "${csv}"
  for item in "${values[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" && "${item}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

require_gpu_check() {
  if [[ "${REQUIRE_GPU}" != "1" ]]; then
    return
  fi

  echo "[CHECK] REQUIRE_GPU=1, verifying CUDA availability..."
  if ! "${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.stderr.write("ERROR: CUDA is not available (torch.cuda.is_available()=False).\n")
    sys.stderr.write("ERROR: Refusing to run on CPU because REQUIRE_GPU=1.\n")
    sys.exit(2)

count = torch.cuda.device_count()
name = torch.cuda.get_device_name(0)
print(f"CUDA ready: device_count={count}, device0={name}")
PY
  then
    echo "Hint    : nvidia-smi/cuInit failures indicate a host-container GPU runtime issue."
    echo "Hint    : use REQUIRE_GPU=0 only if you intentionally want CPU fallback."
    exit 2
  fi
}

profile_seed_base() {
  local needs_profile="$1"
  if [[ "${needs_profile}" == "1" && "${PROFILE}" == "paper_baseline" ]]; then
    echo "42"
  else
    echo "${SEED}"
  fi
}

write_advection_paper_summary() {
  local method_dir="$1"
  local out_csv="${method_dir}/paper_protocol_summary.csv"
  local beta beta_fmt run_id run_dir metrics_file
  local mean_rel std_rel

  for beta in "${BETA_VALUES[@]}"; do
    beta_fmt="$(printf "%.3f" "${beta}")"
    for run_id in $(seq 1 "${REPEATS}"); do
      run_dir="${method_dir}/paper_beta_${beta_fmt}/run_$(printf "%02d" "${run_id}")"
      metrics_file="${run_dir}/metrics.csv"
      if [[ ! -f "${metrics_file}" ]]; then
        echo "[INFO ] Summary bekliyor (${method_dir}); eksik: ${metrics_file}"
        return 1
      fi
    done
  done

  mkdir -p "${method_dir}"
  {
    echo "beta,mean_rel_l2,std_rel_l2"
    for beta in "${BETA_VALUES[@]}"; do
      beta_fmt="$(printf "%.3f" "${beta}")"
      local -a metric_files=()
      for run_id in $(seq 1 "${REPEATS}"); do
        metric_files+=("${method_dir}/paper_beta_${beta_fmt}/run_$(printf "%02d" "${run_id}")/metrics.csv")
      done
      read -r mean_rel std_rel < <(
        awk -F',' '
          FNR==2 {
            x = $5 + 0
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
      printf "%.6f,%s,%s\n" "${beta}" "${mean_rel}" "${std_rel}"
    done
  } > "${out_csv}"

  echo "Saved summary: ${out_csv}"
}

write_burgers2d_paper_summary() {
  local method_dir="$1"
  local out_csv="${method_dir}/paper_protocol_summary.csv"
  local run_id metrics_file
  local rel_l2 run_time mean_rel std_rel
  local -a metric_files=()

  for run_id in $(seq 1 "${REPEATS}"); do
    metrics_file="${method_dir}/run_$(printf "%02d" "${run_id}")/metrics.csv"
    if [[ ! -f "${metrics_file}" ]]; then
      echo "[INFO ] Summary bekliyor (${method_dir}); eksik: ${metrics_file}"
      return 1
    fi
    metric_files+=("${metrics_file}")
  done

  mkdir -p "${method_dir}"
  {
    echo "run,rel_l2,run_time_seconds"
    for run_id in $(seq 1 "${REPEATS}"); do
      metrics_file="${method_dir}/run_$(printf "%02d" "${run_id}")/metrics.csv"
      read -r rel_l2 run_time < <(
        awk -F',' 'NR==2 {printf "%.8e %.6f\n", $4 + 0, $5 + 0}' "${metrics_file}"
      )
      printf "%d,%s,%s\n" "${run_id}" "${rel_l2}" "${run_time}"
    done

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
  } > "${out_csv}"

  echo "Saved summary: ${out_csv}"
}

run_advection_remaining() {
  local job_name="$1"
  local entrypoint="$2"
  local method_dir="$3"
  local needs_profile="$4"
  local use_paper_run_id="$5"
  shift 5
  local required_files=("$@")
  local summary_file="${method_dir}/paper_protocol_summary.csv"

  if [[ -f "${summary_file}" ]]; then
    echo "[SKIP ] ${job_name} (summary exists: ${summary_file})"
    return
  fi

  local beta beta_fmt run_id run_dir seed seed_base
  seed_base="$(profile_seed_base "${needs_profile}")"
  for beta in "${BETA_VALUES[@]}"; do
    beta_fmt="$(printf "%.3f" "${beta}")"
    for run_id in $(seq 1 "${REPEATS}"); do
      run_dir="${method_dir}/paper_beta_${beta_fmt}/run_$(printf "%02d" "${run_id}")"
      seed=$((seed_base + run_id))
      if is_run_complete "${run_dir}" "${required_files[@]}"; then
        echo "[SKIP ] ${job_name} beta=${beta_fmt} run=${run_id} (already complete)"
        continue
      fi

      local -a cmd=("${PYTHON_BIN}" "${entrypoint}")
      if [[ "${needs_profile}" == "1" ]]; then
        cmd+=(--profile "${PROFILE}")
      fi
      cmd+=(--beta "${beta}")
      if [[ "${use_paper_run_id}" == "1" ]]; then
        cmd+=(--seed "${seed_base}" --paper-run-id "${run_id}")
      else
        cmd+=(--seed "${seed}")
      fi
      cmd+=(--save-dir "${run_dir}")
      run_job "${job_name}" "${cmd[@]}"
    done
  done

  write_advection_paper_summary "${method_dir}" || true
}

run_burgers2d_remaining() {
  local job_name="$1"
  local entrypoint="$2"
  local method_dir="$3"
  local needs_profile="$4"
  local use_paper_run_id="$5"
  shift 5
  local required_files=("$@")
  local summary_file="${method_dir}/paper_protocol_summary.csv"

  if [[ -f "${summary_file}" ]]; then
    echo "[SKIP ] ${job_name} (summary exists: ${summary_file})"
    return
  fi

  local run_id run_dir seed seed_base
  seed_base="$(profile_seed_base "${needs_profile}")"
  for run_id in $(seq 1 "${REPEATS}"); do
    run_dir="${method_dir}/run_$(printf "%02d" "${run_id}")"
    seed=$((seed_base + run_id - 1))
    if is_run_complete "${run_dir}" "${required_files[@]}"; then
      echo "[SKIP ] ${job_name} run=${run_id} (already complete)"
      continue
    fi

    local -a cmd=("${PYTHON_BIN}" "${entrypoint}")
    if [[ "${needs_profile}" == "1" ]]; then
      cmd+=(--profile "${PROFILE}")
    fi
    if [[ "${use_paper_run_id}" == "1" ]]; then
      cmd+=(--seed "${seed_base}" --paper-run-id "${run_id}")
    else
      cmd+=(--seed "${seed}")
    fi
    cmd+=(--save-dir "${run_dir}")
    run_job "${job_name}" "${cmd[@]}"
  done

  write_burgers2d_paper_summary "${method_dir}" || true
}

parse_beta_values
require_gpu_check

echo "Run root : ${RUN_ROOT}"
echo "Profile  : ${PROFILE}"
echo "Repeats  : ${REPEATS}"
echo "Seed     : ${SEED}"
echo "Betas    : ${BETA_LIST}"
echo "ForceNew : ${FORCE_NEW_RUN}"
echo "RequireG : ${REQUIRE_GPU}"
echo "Families : ${RUN_FAMILIES}"
echo "AdvMthds : ${ADVECTION_METHODS}"
echo "B2DMthds : ${BURGERS2D_METHODS}"
if [[ "${PROFILE}" == "paper_baseline" && "${SEED}" != "42" ]]; then
  echo "Note    : profile '${PROFILE}' locks seed=42 for profile-based methods."
fi

# ---------------------------------------------------------------------------
# Advection (resume per beta/run)
# ---------------------------------------------------------------------------
if csv_has "${RUN_FAMILIES}" "advection"; then
  if csv_has "${ADVECTION_METHODS}" "naspinn"; then
    run_advection_remaining \
      "advection_naspinn" \
      "NAS_PINNs_advection.py" \
      "${ARTIFACT_ROOT}/advection/naspinn" \
      "0" \
      "0" \
      "metrics.csv" "run_time.txt"
  else
    echo "[SKIP ] advection_naspinn (filtered by ADVECTION_METHODS)"
  fi

  if csv_has "${ADVECTION_METHODS}" "nsga2"; then
    run_advection_remaining \
      "advection_nsga2" \
      "NAS_PINNs_advection_nsga2.py" \
      "${ARTIFACT_ROOT}/advection/nsga2" \
      "1" \
      "1" \
      "metrics.csv" "run_time.txt" "search_summary.csv"
  else
    echo "[SKIP ] advection_nsga2 (filtered by ADVECTION_METHODS)"
  fi

  if csv_has "${ADVECTION_METHODS}" "nsga3"; then
    run_advection_remaining \
      "advection_nsga3" \
      "NAS_PINNs_advection_nsga3.py" \
      "${ARTIFACT_ROOT}/advection/nsga3" \
      "1" \
      "1" \
      "metrics.csv" "run_time.txt" "search_summary.csv"
  else
    echo "[SKIP ] advection_nsga3 (filtered by ADVECTION_METHODS)"
  fi

  if csv_has "${ADVECTION_METHODS}" "bayesian"; then
    run_advection_remaining \
      "advection_bayesian" \
      "NAS_PINNs_advection_bayesian.py" \
      "${ARTIFACT_ROOT}/advection/bayesian" \
      "1" \
      "1" \
      "metrics.csv" "run_time.txt" "search_summary.csv"
  else
    echo "[SKIP ] advection_bayesian (filtered by ADVECTION_METHODS)"
  fi
else
  echo "[SKIP ] advection family (filtered by RUN_FAMILIES)"
fi

# ---------------------------------------------------------------------------
# Burgers2D (resume per run)
# ---------------------------------------------------------------------------
if csv_has "${RUN_FAMILIES}" "burgers2d"; then
  if csv_has "${BURGERS2D_METHODS}" "naspinn"; then
    run_burgers2d_remaining \
      "burgers2d_naspinn" \
      "NAS_PINNs_burgers2d.py" \
      "${ARTIFACT_ROOT}/burgers2d/naspinn" \
      "0" \
      "0" \
      "metrics.csv" "run_time.txt"
  else
    echo "[SKIP ] burgers2d_naspinn (filtered by BURGERS2D_METHODS)"
  fi

  if csv_has "${BURGERS2D_METHODS}" "nsga2"; then
    run_burgers2d_remaining \
      "burgers2d_nsga2" \
      "NAS_PINNs_burgers2d_nsga2.py" \
      "${ARTIFACT_ROOT}/burgers2d/nsga2" \
      "1" \
      "1" \
      "metrics.csv" "run_time.txt" "search_summary.csv"
  else
    echo "[SKIP ] burgers2d_nsga2 (filtered by BURGERS2D_METHODS)"
  fi

  if csv_has "${BURGERS2D_METHODS}" "nsga3"; then
    run_burgers2d_remaining \
      "burgers2d_nsga3" \
      "NAS_PINNs_burgers2d_nsga3.py" \
      "${ARTIFACT_ROOT}/burgers2d/nsga3" \
      "1" \
      "1" \
      "metrics.csv" "run_time.txt" "search_summary.csv"
  else
    echo "[SKIP ] burgers2d_nsga3 (filtered by BURGERS2D_METHODS)"
  fi

  if csv_has "${BURGERS2D_METHODS}" "bayesian"; then
    run_burgers2d_remaining \
      "burgers2d_bayesian" \
      "NAS_PINNs_burgers2d_bayesian.py" \
      "${ARTIFACT_ROOT}/burgers2d/bayesian" \
      "1" \
      "1" \
      "metrics.csv" "run_time.txt" "search_summary.csv"
  else
    echo "[SKIP ] burgers2d_bayesian (filtered by BURGERS2D_METHODS)"
  fi
else
  echo "[SKIP ] burgers2d family (filtered by RUN_FAMILIES)"
fi

echo
echo "All remaining runs completed successfully."
echo "Artifacts: ${ARTIFACT_ROOT}"
echo "Logs     : ${LOG_ROOT}"
