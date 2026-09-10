#!/usr/bin/env bash
# Sampled-measurement lane: one pool, 5 arms sequential, fixed id list.
# Usage: run_sample_lane.sh <run_id> <pool> <comma-ids>
set -u
export PYTHONUTF8=1
RUN_ID="$1"; POOL="$2"; IDS="$3"
cd "$(dirname "$0")/.."
mkdir -p ttfa_real/logs
for ARM in local conservative balanced aggressive always; do
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
echo ">>> SAMPLE pool ${POOL} all done $(date -u +%H:%M:%SZ)"
