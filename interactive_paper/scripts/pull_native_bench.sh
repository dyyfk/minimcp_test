#!/usr/bin/env bash
# Pull the judged native-bench parquets from the gate-data volume.
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."
mkdir -p data/native_bench
for f in $(modal volume ls gate-data native_bench 2>/dev/null | grep -oE "[a-z]+_[a-z]+(_tts)?_judged\.parquet" | sort -u); do
  modal volume get gate-data "native_bench/$f" "data/native_bench/$f" --force >/dev/null 2>&1 && echo "pulled $f"
done
