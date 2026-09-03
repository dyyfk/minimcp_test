#!/usr/bin/env bash
# Stream A (TTS relay, new cleaner): judge the in-flight sllama run, then always arms on the rest.
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd "$(dirname "$0")"
LOG=native_bench_A.log
while kill -0 19588 2>/dev/null; do sleep 30; done
echo "=== $(date '+%F %T') JUDGE sllama/always tts ===" >> "$LOG"
modal run modal_native_bench.py::judge --pool sllama --tier always --relay tts 2>&1 | grep -E ">>>|Error|Traceback" >> "$LOG"
modal run modal_native_bench.py::judge --pool sllama --tier always --relay tts --field expert 2>&1 | grep -E ">>>|Error|Traceback" >> "$LOG"
for pool in swebq sdqa sreason frozen valpaca; do
  echo "=== $(date '+%F %T') RUN $pool/always tts ===" >> "$LOG"
  modal run modal_native_bench.py::run_bench --pool "$pool" --tier always --workers 8 --relay tts 2>&1 | grep -E "^\s*\[|>>>|Error|Traceback|complete" >> "$LOG"
  echo "=== $(date '+%F %T') JUDGE $pool/always tts ===" >> "$LOG"
  modal run modal_native_bench.py::judge --pool "$pool" --tier always --relay tts 2>&1 | grep -E ">>>|Error|Traceback" >> "$LOG"
  modal run modal_native_bench.py::judge --pool "$pool" --tier always --relay tts --field expert 2>&1 | grep -E ">>>|Error|Traceback" >> "$LOG"
done
echo "=== $(date '+%F %T') STREAM-A-TTS DONE ===" >> "$LOG"
