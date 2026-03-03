#!/usr/bin/env bash
set -euo pipefail

# Run only Advection + Burgers2D jobs sequentially (all methods).
# This is a thin wrapper around run_advection_burgers2d_paper_batch.sh.
#
# Usage:
#   bash run_advection_burgers2d_remaining.sh [run_id] [base_seed] [mode]
#
# Args:
#   run_id    : pipeline run id (default: current UTC timestamp)
#   base_seed : base seed for all jobs (default: 42)
#   mode      : single | paper (default: single)
#
# Examples:
#   bash run_advection_burgers2d_remaining.sh 20260302_204438 42 single
#   bash run_advection_burgers2d_remaining.sh 20260302_204438 42 paper

RUN_ID="${1:-$(date -u +%Y%m%d_%H%M%S)}"
BASE_SEED="${2:-42}"
MODE="${3:-single}"

if [[ "${MODE}" != "single" && "${MODE}" != "paper" ]]; then
  echo "Invalid mode: ${MODE} (expected: single|paper)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_advection_burgers2d_paper_batch.sh" "${RUN_ID}" "${BASE_SEED}" "${MODE}"
