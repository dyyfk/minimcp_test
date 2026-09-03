#!/usr/bin/env bash
# Stream B (TTS relay, new cleaner): three live mid tiers, remaining pools.
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd "$(dirname "$0")"
LOG=native_bench_B.log
for pool in swebq sllama sdqa sreason frozen valpaca; do
  for tier in balanced aggressive conservative; do
    echo "=== $(date '+%F %T') RUN $pool/$tier tts ===" >> "$LOG"
    modal run modal_native_bench.py::run_bench --pool "$pool" --tier "$tier" --workers 8 --relay tts 2>&1 | grep -E "^\s*\[|>>>|Error|Traceback|complete" >> "$LOG"
    echo "=== $(date '+%F %T') JUDGE $pool/$tier tts ===" >> "$LOG"
    modal run modal_native_bench.py::judge --pool "$pool" --tier "$tier" --relay tts 2>&1 | grep -E ">>>|Error|Traceback" >> "$LOG"
  done
done
echo "=== $(date '+%F %T') STREAM-B-TTS DONE ===" >> "$LOG"
