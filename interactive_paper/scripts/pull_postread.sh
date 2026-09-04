#!/usr/bin/env bash
# Pull the 8bw later-read-point dumps (feats shards + judged parquets) for the *k tags.
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."
for f in $(modal volume ls gate-data 2>/dev/null | grep -oE "frozen_native_[a-z0-9]+k_(feats\.shard[0-9]+\.npz|judged\.parquet)" | sort -u); do
  [ -f "data/$f" ] && [[ "$f" == *feats* ]] && continue
  modal volume get gate-data "$f" "data/$f" --force >/dev/null 2>&1 && echo "pulled $f"
done
