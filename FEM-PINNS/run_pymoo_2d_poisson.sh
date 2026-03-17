#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/2D-Poisson-RunPymoo}"
LOG_FILE="${RUN_DIR}/run_pymoo.log"

mkdir -p "$RUN_DIR"

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

echo "[$(timestamp)] 2D Poisson local-pymoo rerun started"
echo "Script dir : ${SCRIPT_DIR}"
echo "Python bin : ${PYTHON_BIN}"
echo "Master log : ${LOG_FILE}"

run_step "NSGA2_Pymoo" NSGA2_Pymoo_2D_Poisson.py
run_step "NSGA3_Pymoo" NSGA3_Pymoo_2D_Poisson.py
run_step "Compare" Compare_2D_Poisson.py --results-root .
run_step "Visualize" Visualize_2D_Poisson.py --results-root .

echo
echo "[$(timestamp)] Local-pymoo reruns completed successfully."
echo "New result folders:"
echo "  ${SCRIPT_DIR}/2D-Poisson-NSGA2-Pymoo"
echo "  ${SCRIPT_DIR}/2D-Poisson-NSGA3-Pymoo"
echo "Updated comparison outputs:"
echo "  ${SCRIPT_DIR}/2D-Poisson-Comparison/comparison_summary.csv"
echo "  ${SCRIPT_DIR}/2D-Poisson-Comparison/comparison_summary.json"
echo "  ${SCRIPT_DIR}/2D-Poisson-Comparison/comparison_details.json"
echo "  ${SCRIPT_DIR}/2D-Poisson-Comparison/poisson_comparison_single_figure.png"
echo "  ${SCRIPT_DIR}/2D-Poisson-Comparison/poisson_comparison_single_figure.pdf"
