#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ALL_DIR="${RUN_ALL_DIR:-${SCRIPT_DIR}/2D-Poisson-RunAll}"
LOG_FILE="${RUN_ALL_DIR}/run_all.log"

mkdir -p "$RUN_ALL_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

timestamp() {
  date -u +"%Y-%m-%d %H:%M:%S UTC"
}

run_step() {
  local label="$1"
  shift

  echo
  echo "============================================================"
  echo "[$(timestamp)] START ${label}"
  echo "Command: ${PYTHON_BIN} $*"
  echo "============================================================"
  "${PYTHON_BIN}" "$@"
  echo "[$(timestamp)] DONE  ${label}"
}

echo "[$(timestamp)] 2D Poisson full run started"
echo "Script dir : ${SCRIPT_DIR}"
echo "Python bin : ${PYTHON_BIN}"
echo "Master log : ${LOG_FILE}"

run_step "FEM" FEM_2D_Poisson.py
run_step "PINN" PINN_2D_Poisson.py
run_step "NAS_PINN" NAS_PINN_2D_Poisson.py
run_step "NSGA2" NSGA2_2D_Poisson.py
run_step "NSGA3" NSGA3_2D_Poisson.py
run_step "Bayesian" Bayesian_2D_Poisson.py
run_step "Compare" Compare_2D_Poisson.py --results-root .

echo
echo "[$(timestamp)] All runs completed successfully."
echo "Comparison outputs:"
echo "  ${SCRIPT_DIR}/2D-Poisson-Comparison/comparison_summary.csv"
echo "  ${SCRIPT_DIR}/2D-Poisson-Comparison/comparison_summary.json"
echo "  ${SCRIPT_DIR}/2D-Poisson-Comparison/comparison_details.json"
echo "  ${SCRIPT_DIR}/2D-Poisson-Comparison/comparison.log"
