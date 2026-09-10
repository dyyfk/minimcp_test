#!/usr/bin/env bash
# Pull every input the local ttfa-v3 rebuild needs from the Modal
# gate-data volume into ./data_bundle (queries, gate artifacts, audio).
# Verify against freeze.json hashes afterwards. Model weights are NOT
# here — download openbmb/MiniCPM-o-4_5 and check config.json sha
# against freeze.json (or copy /workspace/models from the
# minicpm-o45-weights volume).
set -eu
export PYTHONUTF8=1
DEST="${1:-data_bundle}"
mkdir -p "$DEST"
for f in queries.jsonl queries_striviaqa.jsonl queries_swebq.jsonl \
         queries_sllama.jsonl queries_sdqa.jsonl \
         gate_native.json gate_act.json; do
  modal volume get gate-data "$f" "$DEST/$f" --force
done
for d in audio_pool bench_audio sdqa_audio; do
  mkdir -p "$DEST/$d"
  modal volume get gate-data "$d" "$DEST/" --force
done
echo "done -> $DEST"
