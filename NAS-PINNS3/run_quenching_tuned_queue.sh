#!/bin/bash
# ============================================================
# NAS-PINNS3 — Quenching Deney Kuyruğu
# PLAN.md: full → fixed → adaptive  (3 time-mode × 3 optimizer)
# NAS-PINNS1 parametreleri kullanılır.
# ============================================================

set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

ADAM_EPOCHS=${ADAM_EPOCHS:-10000}
NAS_POP=${NAS_POP:-24}
NAS_GEN=${NAS_GEN:-16}
NAS_CALLS=${NAS_CALLS:-16}
NAS_EPOCHS=${NAS_EPOCHS:-300}
FIXED_SKIP=${FIXED_SKIP:-2}
RUN_LBFGS=${RUN_LBFGS:-""}
RUN_PSO=${RUN_PSO:-""}

COMMON_ARGS="--compare_all --adam_epochs $ADAM_EPOCHS \
  --nas_pop $NAS_POP --nas_gen $NAS_GEN \
  --nas_calls $NAS_CALLS --nas_epochs $NAS_EPOCHS \
  $RUN_LBFGS $RUN_PSO"

echo "======================================"
echo " NAS-PINNS3 Queue  —  $(date)"
echo " Adam=$ADAM_EPOCHS NAS=pop$NAS_POP/gen$NAS_GEN/calls$NAS_CALLS/proxy$NAS_EPOCHS"
echo "======================================"

echo "[1/3] FULL  — $(date)"
python main.py $COMMON_ARGS --time_mode full --output_dir results/full
echo "[1/3] done  — $(date)"

echo "[2/3] FIXED (stride=$FIXED_SKIP) — $(date)"
python main.py $COMMON_ARGS --time_mode fixed --fixed_skip $FIXED_SKIP --output_dir results/fixed
echo "[2/3] done  — $(date)"

echo "[3/3] ADAPTIVE — $(date)"
python main.py $COMMON_ARGS --time_mode adaptive --output_dir results/adaptive
echo "[3/3] done  — $(date)"

echo "======================================"
echo " TAMAMLANDI — $(date)"
echo "======================================"
