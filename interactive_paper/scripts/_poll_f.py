"""Poll formal v2 batch progress; DONE when every (run,pool,arm) full."""
import json
import os
import subprocess
import sys
import tempfile

RUNS = sys.argv[1:] or ["f1"]
POOLS = {"frozen": 240, "striviaqa": 250, "sdqa": 200}
ARMS = ["never_tts", "conservative_tts", "balanced_tts",
        "aggressive_tts", "always_tts"]
env = dict(os.environ, PYTHONUTF8="1")


def ls(path):
    r = subprocess.run(["modal", "volume", "ls", "gate-data", path,
                        "--json"], capture_output=True, text=True, env=env)
    try:
        return json.loads(r.stdout)
    except Exception:
        return []


parts, done, total = [], 0, 0
for run in RUNS:
    for pool, want in POOLS.items():
        listing = ls(f"native_bench_v2/{run}/{pool}")
        for arm in ARMS:
            total += 1
            files = [e["filename"] for e in listing
                     if e["filename"].split("/")[-1].startswith(
                         arm + ".jsonl.shard")]
            ids = set()
            for f in files:
                with tempfile.TemporaryDirectory() as td:
                    p = os.path.join(td, "x.jsonl")
                    subprocess.run(["modal", "volume", "get", "gate-data",
                                    f, p, "--force"],
                                   capture_output=True, env=env)
                    try:
                        for ln in open(p, encoding="utf-8"):
                            if ln.strip():
                                ids.add(json.loads(ln)["id"])
                    except FileNotFoundError:
                        pass
            n = len(ids)
            done += n >= want
            parts.append(f"{run}/{pool[:4]}/{arm.replace('_tts', '')}={n}")
print(("DONE " if done == total else "") + " ".join(parts), flush=True)
sys.exit(0 if done == total else 1)
