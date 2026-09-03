# Human Evaluation Integration Plan

The local end-to-end path is implemented:

1. FastAPI assigns two tasks and two blinded model arms per task.
2. The browser streams 16 kHz PCM through the study WebSocket.
3. The gateway connects to `demo_duplex.py` with the assigned `probe_on` setting. The service returns one private structured event per turn with full user audio, ASR, routing telemetry, and timings.
4. Session JSON, WAV audio, transcripts, model telemetry, ratings, and comparisons are persisted by the backend.
5. `modal_app.py` publishes a separate study URL and commits data to the `human-eval-data` Volume.

Before formal recruitment:

- Run a small pilot covering manual finish, two-minute timeout, model failure, refresh, and all task/arm orders.
- Decide consent copy, data retention, access rules, and transcript redaction.
- Deploy the duplex service first, then deploy the separate human-evaluation Modal app.
- Inspect exported sessions for missing audio/transcripts and confirm assignment balance.

Do not add a database or queue unless multiple backend processes or recruitment scale require them.
