#!/usr/bin/env bash
# Sampled runs for one pool over an explicit arm list.
# Usage: run_sample_arms.sh <run_id> <pool> <comma-ids> <arm> [arm...]
set -u
export PYTHONUTF8=1
RUN_ID="$1"; POOL="$2"; IDS="$3"; shift 3
cd "$(dirname "$0")/.."
mkdir -p ttfa_real/logs
for ARM in "$@"; do
  LOG="ttfa_real/logs/sample_${POOL}_${ARM}.log"
  echo ">>> SAMPLE start ${POOL}/${ARM} $(date -u +%H:%M:%SZ)"
  modal run modal_ttfa_bench.py::run_bench \
    --pool "$POOL" --arm "$ARM" --workers 3 --run-id "$RUN_ID" \
    --ids "$IDS" >> "$LOG" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo ">>> SAMPLE retry ${POOL}/${ARM} rc=$RC"
    modal run modal_ttfa_bench.py::run_bench \
      --pool "$POOL" --arm "$ARM" --workers 3 --run-id "$RUN_ID" \
      --ids "$IDS" >> "$LOG" 2>&1
    RC=$?
  fi
  echo ">>> SAMPLE done ${POOL}/${ARM} rc=$RC $(date -u +%H:%M:%SZ)"
done
echo ">>> SAMPLE pool ${POOL} arms $* all done $(date -u +%H:%M:%SZ)"
