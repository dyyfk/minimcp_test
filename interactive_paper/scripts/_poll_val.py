"""Poll repeat-run progress on the gate-data volume; print one status
line; exit 0 with DONE when all 10 (run, arm) sets reach 240 ids."""
import json
import os
import subprocess
import sys
import tempfile

ARMS = ["never_tts", "always_tts"]
env = dict(os.environ, PYTHONUTF8="1")

def ls(path):
    r = subprocess.run(["modal", "volume", "ls", "gate-data", path,
                        "--json"], capture_output=True, text=True, env=env)
    try:
        return json.loads(r.stdout)
    except Exception:
        return []

def count(run, arm):
    files = [e["filename"] for e in ls(f"native_bench_v2/{run}/validation")
             if e["filename"].split("/")[-1].startswith(arm + ".jsonl.shard")]
    ids = set()
    for f in files:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "x.jsonl")
            subprocess.run(["modal", "volume", "get", "gate-data", f, p,
                            "--force"], capture_output=True, env=env)
            try:
                for ln in open(p, encoding="utf-8"):
                    if ln.strip():
                        ids.add(json.loads(ln)["id"])
            except FileNotFoundError:
                pass
    return len(ids)

parts, done = [], 0
for run in ["val1"]:
    for arm in ARMS:
        n = count(run, arm)
        done += n >= 299
        parts.append(f"{run}/{arm.replace('_tts','')}={n}")
print(("DONE " if done == 10 else "") + " ".join(parts), flush=True)
sys.exit(0 if done == 10 else 1)
