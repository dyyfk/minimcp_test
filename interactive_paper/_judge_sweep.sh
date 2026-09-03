#!/usr/bin/env bash
# Final pass: re-judge every (pool, tier) so rows whose verdict was lost to
# judge rate limits get filled (judge_pool re-tries verdict-less rows).
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd "$(dirname "$0")"
LOG=native_bench_sweep.log
for pool in striviaqa swebq sllama sdqa sreason frozen valpaca; do
  modal run modal_native_bench.py::judge --pool "$pool" --tier never 2>&1 | grep -E ">>> .*n=|Error|Traceback" >> "$LOG"
  for tier in conservative balanced aggressive always; do
    modal run modal_native_bench.py::judge --pool "$pool" --tier "$tier" --relay tts 2>&1 | grep -E ">>> .*n=|Error|Traceback" >> "$LOG"
  done
  modal run modal_native_bench.py::judge --pool "$pool" --tier always --relay tts --field expert 2>&1 | grep -E ">>> .*n=|Error|Traceback" >> "$LOG"
done
echo "=== $(date '+%F %T') SWEEP DONE ===" >> "$LOG"
