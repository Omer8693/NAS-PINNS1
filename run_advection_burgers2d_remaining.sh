#!/usr/bin/env bash
set -euo pipefail

# Sequential runner for the remaining equations:
# 1) Advection (NAS-PINN, NSGA-II, NSGA-III, Bayesian)
# 2) Burgers2D (NAS-PINN, NSGA-II, NSGA-III, Bayesian)

PYTHON_BIN="${PYTHON_BIN:-python}"
PROFILE="${PROFILE:-paper_baseline}"     # paper_baseline | ours_fast
REPEATS="${REPEATS:-5}"
SEED="${SEED:-42}"
BETA_LIST="${BETA_LIST:-1.0,0.5,0.1}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-results/pipeline_runs/${STAMP}_advection_burgers2d}"
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

  "$@" 2>&1 | tee "${log_file}"

  echo "[DONE ] ${name}"
}

echo "Run root : ${RUN_ROOT}"
echo "Profile  : ${PROFILE}"
echo "Repeats  : ${REPEATS}"
echo "Seed     : ${SEED}"
echo "Betas    : ${BETA_LIST}"

# ---------------------------------------------------------------------------
# Advection (paper-style repeated protocol over beta list)
# ---------------------------------------------------------------------------
run_job "advection_naspinn" \
  "${PYTHON_BIN}" NAS_PINNs_advection.py \
  --paper-protocol --paper-betas "${BETA_LIST}" \
  --repeats "${REPEATS}" --seed "${SEED}" \
  --save-dir "${ARTIFACT_ROOT}/advection/naspinn"

run_job "advection_nsga2" \
  "${PYTHON_BIN}" NAS_PINNs_advection_nsga2.py \
  --profile "${PROFILE}" \
  --paper-protocol --paper-betas "${BETA_LIST}" \
  --repeats "${REPEATS}" --seed "${SEED}" \
  --save-dir "${ARTIFACT_ROOT}/advection/nsga2"

run_job "advection_nsga3" \
  "${PYTHON_BIN}" NAS_PINNs_advection_nsga3.py \
  --profile "${PROFILE}" \
  --paper-protocol --paper-betas "${BETA_LIST}" \
  --repeats "${REPEATS}" --seed "${SEED}" \
  --save-dir "${ARTIFACT_ROOT}/advection/nsga3"

run_job "advection_bayesian" \
  "${PYTHON_BIN}" NAS_PINNs_advection_bayesian.py \
  --profile "${PROFILE}" \
  --paper-protocol --paper-betas "${BETA_LIST}" \
  --repeats "${REPEATS}" --seed "${SEED}" \
  --save-dir "${ARTIFACT_ROOT}/advection/bayesian"

# ---------------------------------------------------------------------------
# Burgers2D (paper-style repeated protocol)
# ---------------------------------------------------------------------------
run_job "burgers2d_naspinn" \
  "${PYTHON_BIN}" NAS_PINNs_burgers2d.py \
  --paper-protocol \
  --repeats "${REPEATS}" --seed "${SEED}" \
  --save-dir "${ARTIFACT_ROOT}/burgers2d/naspinn"

run_job "burgers2d_nsga2" \
  "${PYTHON_BIN}" NAS_PINNs_burgers2d_nsga2.py \
  --profile "${PROFILE}" \
  --paper-protocol \
  --repeats "${REPEATS}" --seed "${SEED}" \
  --save-dir "${ARTIFACT_ROOT}/burgers2d/nsga2"

run_job "burgers2d_nsga3" \
  "${PYTHON_BIN}" NAS_PINNs_burgers2d_nsga3.py \
  --profile "${PROFILE}" \
  --paper-protocol \
  --repeats "${REPEATS}" --seed "${SEED}" \
  --save-dir "${ARTIFACT_ROOT}/burgers2d/nsga3"

run_job "burgers2d_bayesian" \
  "${PYTHON_BIN}" NAS_PINNs_burgers2d_bayesian.py \
  --profile "${PROFILE}" \
  --paper-protocol \
  --repeats "${REPEATS}" --seed "${SEED}" \
  --save-dir "${ARTIFACT_ROOT}/burgers2d/bayesian"

echo
echo "All remaining runs completed successfully."
echo "Artifacts: ${ARTIFACT_ROOT}"
echo "Logs     : ${LOG_ROOT}"
