"""Run the four remaining folds for the two frozen P29 screen winners."""
from __future__ import annotations

import argparse
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
    winners = (
        (22, "full_block", 1e-4, 0.),
        (30, "attention_only", 3e-5, .01),
    )
    queue = [(tap, mode, lr, wd, fold) for tap, mode, lr, wd in winners
             for fold in range(1, 5)]
    script = Path(__file__).with_name("90_train_copied_block_probe.py")
    running, failures = {}, []
    while queue or running:
        for gpu in gpus:
            if gpu in running or not queue:
                continue
            tap, mode, lr, wd, fold = queue.pop(0)
            name = (f"tap{tap}_{mode}_lr{label(lr)}_wd{label(wd)}_"
                    f"fold{fold}")
            output = args.output_dir / f"{name}.json"
            if output.exists():
                print(f"skip {name}", flush=True)
                continue
            command = [
                sys.executable, str(script),
                "--selection", str(args.selection),
                "--windows-dir", str(args.windows_dir),
                "--original-dir", str(args.original_dir),
                "--local", str(args.local), "--expert", str(args.expert),
                "--live-artifact", str(args.live_artifact),
                "--model-dir", str(args.model_dir), "--tap-layer", str(tap),
                "--train-mode", mode, "--learning-rate", str(lr),
                "--weight-decay", str(wd), "--fold-index", str(fold),
                "--output", str(output),
            ]
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            handle = (logs / f"{name}.log").open("w")
            process = subprocess.Popen(command, stdout=handle,
                                       stderr=subprocess.STDOUT, env=environment)
            running[gpu] = (process, name, handle)
            print(f"gpu {gpu}: start {name}", flush=True)
        time.sleep(2)
        for gpu, (process, name, handle) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            print(f"gpu {gpu}: finish {name} rc={code}", flush=True)
            if code:
                failures.append((name, code))
            del running[gpu]
    if failures:
        raise RuntimeError(f"OOF failures: {failures}")
    print("remaining folds complete", flush=True)


if __name__ == "__main__":
    main()
