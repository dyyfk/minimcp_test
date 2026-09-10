#!/usr/bin/env bash
# TTFA full-run lane driver. Usage: run_lane.sh <run_id> <pool> [pool...]
# Arms run sequentially per pool; one attempt per query (resume skips
# recorded ids, so re-invoking after a crash only runs the remainder).
set -u
export PYTHONUTF8=1
RUN_ID="$1"; shift
cd "$(dirname "$0")/.."     # interactive_paper
mkdir -p ttfa_real/logs
for POOL in "$@"; do
  for ARM in local conservative balanced aggressive always; do
    LOG="ttfa_real/logs/${POOL}_${ARM}.log"
    echo ">>> LANE start ${POOL}/${ARM} $(date -u +%H:%M:%SZ)"
    modal run modal_ttfa_bench.py::run_bench \
      --pool "$POOL" --arm "$ARM" --workers 8 --run-id "$RUN_ID" \
      >> "$LOG" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
      echo ">>> LANE retry-once ${POOL}/${ARM} rc=$RC $(date -u +%H:%M:%SZ)"
      modal run modal_ttfa_bench.py::run_bench \
        --pool "$POOL" --arm "$ARM" --workers 8 --run-id "$RUN_ID" \
        >> "$LOG" 2>&1
      RC=$?
    fi
    echo ">>> LANE done ${POOL}/${ARM} rc=$RC $(date -u +%H:%M:%SZ)"
  done
done
echo ">>> LANE all done: $* $(date -u +%H:%M:%SZ)"
