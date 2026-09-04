#!/usr/bin/env bash
# 8bw: official-config native dump with later read points (X, X_k1..3), then judge.
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd "$(dirname "$0")"
LOG=postread.log
run() { # tag pool split
  echo "=== $(date '+%F %T') DUMP $1 ($2/$3) ===" >> "$LOG"
  if [ -n "$3" ]; then SPL="--split $3"; else SPL=""; fi
  modal run modal_native_dump.py::run_native --pool "$2" $SPL --tag "$1" --official 1 --workers 8 2>&1 | grep -E ">>>|Error|Traceback" >> "$LOG"
  echo "=== $(date '+%F %T') JUDGE $1 ===" >> "$LOG"
  modal run modal_native_dump.py::judge_all --tags "$1:$2" 2>&1 | grep -E ">>>|Error|Traceback" >> "$LOG"
}
run testk frozen test
run striviaqak striviaqa ""
run swebqk swebq ""
run sllamak sllama ""
run sdqak sdqa ""
run sreasonk sreason ""
run calibk frozen calib
run freshk fresh ""
run expk expansion ""
run exp2k expansion2 ""
run exp3zhk expansion3zh ""
run exp3k expansion3 ""
echo "=== $(date '+%F %T') POSTREAD DONE ===" >> "$LOG"
