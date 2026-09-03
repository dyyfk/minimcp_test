"""Public, persistent Modal deployment for the human-evaluation site.

Deploy from the repository root:
    modal deploy human_eval/modal_app.py
"""

from __future__ import annotations

from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent.parent
DATA_MOUNT = "/data"

app = modal.App("minicpm-human-eval")
study_data = modal.Volume.from_name("human-eval-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.115,<1",
        "httpx>=0.27,<1",
        "uvicorn[standard]>=0.30,<1",
        "websockets>=13,<16",
    )
    .add_local_dir(str(ROOT / "human_eval"), "/workspace/human_eval")
)


@app.function(
    image=image,
    volumes={DATA_MOUNT: study_data},
    secrets=[modal.Secret.from_name("human-eval-secrets")],
    env={
        "PYTHONPATH": "/workspace",
        "HUMAN_EVAL_DATA_DIR": DATA_MOUNT,
        "HUMAN_EVAL_PUBLIC": "1",
        "HUMAN_EVAL_REQUIRE_TRANSCRIPTS": "1",
        # HUMAN_EVAL_DEBUG_REMOVE_AFTER_PILOT: set to "0" before recruitment.
        "HUMAN_EVAL_DEBUG_MODEL_LOGS": "1",
        "MINICPM_DEMO_URL": "https://rhe9527--gate-duplex-voice.modal.run",
    },
    max_containers=1,
    timeout=60 * 60,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="minicpm-human-eval")
def web():
    from human_eval.backend import app as backend

    # The store writes atomically first, then commits JSON and WAV changes to
    # the named Volume before the participant request is acknowledged.
    backend.store.set_after_save(study_data.commit)
    return backend.app
