#!/usr/bin/env bash
# Pull all ttfa-v3 formal shard logs from the volume into results/<run_id>.
# Usage: fetch_results.sh [run_id]
set -u
export PYTHONUTF8=1
RUN_ID="${1:-ttfa1}"
cd "$(dirname "$0")"
for P in frozen striviaqa swebq sllama sdqa; do
  mkdir -p "results/$RUN_ID/$P"
  modal volume ls gate-data "ttfa_real/$RUN_ID/$P" 2>/dev/null \
    | grep -oE "ttfa_real/$RUN_ID/$P/(local|conservative|balanced|aggressive|always)\.jsonl\.shard[0-9]+" \
    | sort -u | while read -r f; do
        modal volume get gate-data "$f" "results/$RUN_ID/$P/" --force >/dev/null 2>&1 \
          || echo "FAIL $f"
      done
  echo "$P: $(cat "results/$RUN_ID/$P"/*.jsonl.shard* 2>/dev/null | wc -l) rows"
done
