# Human Evaluation Backend

The backend deliberately stays small: FastAPI, one JSON file per session, separate WAV files, and one blinded WebSocket proxy to the existing `demo_duplex.py` Voice service. There is no database, ORM, queue, or separate model adapter process.

Run instructions, model routing, allocation policy, collected metrics, API contracts, and the multi-turn open question are documented in the main [Human Evaluation README](../README.md).

Optional environment variables:

- `HUMAN_EVAL_DATA_DIR` — JSON/audio root; defaults to `human_eval/backend/data`.
- `HUMAN_EVAL_CONVERSATION_SECONDS` — server limit; defaults to `120`.
- `MINICPM_DEMO_URL`, `MINICPM_DEMO_TOKEN` — upstream Voice service override.
- `OPENAI_API_KEY` — post-hoc transcription for turns without upstream text.
- `HUMAN_EVAL_ASR_MODEL` — ASR model; defaults to `gpt-transcribe`.
- `HUMAN_EVAL_REQUIRE_TRANSCRIPTS` — set to `1` for formal/public collection; readiness fails when ASR is not configured.
- `HUMAN_EVAL_PUBLIC` — set to `1` on a public deployment.
- `HUMAN_EVAL_ADMIN_TOKEN` — Bearer token required by `/api/admin/*` in public mode.
- `HUMAN_EVAL_DEBUG_MODEL_LOGS` — temporary pilot flag that exposes the blinded model configuration to the browser console.

`human_eval/modal_app.py` provides the public one-replica deployment and commits each atomic save to the `human-eval-data` Volume. It stores session JSON under `sessions/`, WAV files under `audio/`, and timestamped structured event logs under `logs/`. The browser does not retain study records.

Conversation interaction and rating lifecycles are separate in schema 1.4. API access sweeps expired reservations into an explicit terminal `expired` session state, and analysis exports expose reservation and QC status directly.

See [DATA_SCHEMA.md](./DATA_SCHEMA.md) for the persisted record shape.
