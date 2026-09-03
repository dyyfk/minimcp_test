"""Run the frozen 36-cell P29 structural screen across available GPUs."""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path


def label(value):
    return str(value).replace(".", "p").replace("-", "m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--windows-dir", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--live-artifact", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    script = Path(__file__).with_name("90_train_copied_block_probe.py")
    grid = list(itertools.product(
        (10, 22, 30), ("attention_only", "full_block"),
        (1e-5, 3e-5, 1e-4), (0., .01)))
    queue = []
    for tap, mode, learning_rate, weight_decay in grid:
        name = (f"tap{tap}_{mode}_lr{label(learning_rate)}_"
                f"wd{label(weight_decay)}_fold0")
        output = args.output_dir / f"{name}.json"
        if output.exists():
            print(f"skip {name}", flush=True)
            continue
        queue.append((name, output, tap, mode, learning_rate, weight_decay))

    running = {}
    failures = []
    while queue or running:
        for gpu in gpus:
            if gpu in running or not queue:
                continue
            name, output, tap, mode, learning_rate, weight_decay = queue.pop(0)
            command = [
                sys.executable, str(script),
                "--selection", str(args.selection),
                "--windows-dir", str(args.windows_dir),
                "--original-dir", str(args.original_dir),
                "--local", str(args.local), "--expert", str(args.expert),
                "--live-artifact", str(args.live_artifact),
                "--model-dir", str(args.model_dir),
                "--tap-layer", str(tap), "--train-mode", mode,
                "--learning-rate", str(learning_rate),
                "--weight-decay", str(weight_decay),
                "--fold-index", "0", "--output", str(output),
            ]
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            log_handle = (logs / f"{name}.log").open("w")
            process = subprocess.Popen(command, stdout=log_handle,
                                       stderr=subprocess.STDOUT, env=environment)
            running[gpu] = (process, name, log_handle)
            print(f"gpu {gpu}: start {name}", flush=True)
        time.sleep(2)
        for gpu, (process, name, log_handle) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log_handle.close()
            print(f"gpu {gpu}: finish {name} rc={code}", flush=True)
            if code:
                failures.append((name, code))
            del running[gpu]
    if failures:
        raise RuntimeError(f"screen failures: {failures}")
    print(f"screen complete: {len(grid)} configurations", flush=True)


if __name__ == "__main__":
    main()
