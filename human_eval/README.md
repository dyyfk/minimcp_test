# MiniCPM Human Evaluation

English, blinded voice evaluation of MiniCPM and MiniCPM+. The participant UI and FastAPI backend are under `human_eval/`.

## Current status

- The browser checks the microphone, speaker, consent, and model readiness. During each conversation, a green **You** card shows live microphone movement while a separate blue **Assistant** card shows listening, thinking, and speaking states.
- Each participant receives two tasks. Each task contains one MiniCPM conversation and one MiniCPM+ conversation in blinded, balanced order.
- Each conversation lasts at most two minutes and uses one fixed model arm.
- The backend saves assignments, audio, transcripts, model telemetry, ratings, and pairwise feedback.
- The duplex model service returns complete turn audio, ASR, routing telemetry, and conversation-scoped history.
- MiniCPM+ uses the `aggressive` escalation tier; MiniCPM keeps escalation disabled.
- Routing correctness is reviewed per MiniCPM+ turn; it is not inferred from the task type.
- `modal_app.py` deploys this study as a separate public Modal site with its own persistent Volume.

## Recent changes

- Human evaluation now has its own public Modal app and URL; it does not share the demo page.
- MiniCPM+ uses the `aggressive` routing tier. MiniCPM keeps escalation off.
- The duplex service emits one structured event per turn with full user audio, ASR, model/expert transcript, routing scores, and latency.
- Conversation-scoped history is retained and passed to the expert for S3 follow-ups.
- Session JSON, WAV audio, and timestamped JSONL event logs persist in the `human-eval-data` Modal Volume.
- `/api/admin/*` requires a Bearer token on the public deployment.
- Temporary pilot logging shows the blinded model configuration in the browser console and Modal logs.
- Task examples are short conversation starters. S3 uses familiar date or meeting planning topics and lets the participant continue naturally.
- Per-conversation feedback is reduced from eight overlapping ratings to four focused ratings.
- Finishing a conversation shows a saving spinner while final turn data is drained and persisted.

## Open questions before launch

1. **Transcript and audio policy:** confirm consent language, retention, deletion, and redaction of personal information.
2. **Data ownership:** choose who can access the Modal Volume and who owns backups.
3. **Routing review:** decide who labels each MiniCPM+ turn as expected `local`, `escalate`, or `not_applicable`, and whether a second reviewer is needed.
4. **Study policy:** set the recruitment target, stopping rule, exclusion criteria, and treatment of incomplete sessions.
5. **ASR quality:** pilot English transcription accuracy and decide how failed or incorrect transcripts are corrected.
6. **Provider logs:** confirm whether the model and ASR providers retain request audio or text.

## Run and test locally

One-time setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r human_eval/backend/requirements.txt
```

Start the app:

```bash
source .venv/bin/activate
python3 human_eval/serve.py
```

Open <http://127.0.0.1:4173/>.

Optional post-hoc ASR fallback:

```bash
export OPENAI_API_KEY='your-key'
export HUMAN_EVAL_REQUIRE_TRANSCRIPTS=1
python3 human_eval/serve.py
```

To start another test in the same browser:

1. Restart `serve.py` if `scenarios.json` or backend code changed. Scenario definitions are loaded when the server starts.
2. To request another assignment within the same study version, open browser developer tools and run:

   ```js
   sessionStorage.removeItem("humanEvalUserId");
   location.reload();
   ```

3. Reload the page after deploying. The HTML, JavaScript, CSS, and scenario config use versioned URLs plus `Cache-Control: no-store`, so a hard refresh should not be necessary.

After a scenario-version change, the backend automatically creates a current-version session instead of resuming an older assignment. Clearing the browser user ID is only needed to repeat the same version as a fresh participant.

Do not delete old records to reset the UI. Local output is stored here:

- Session JSON: `human_eval/backend/data/sessions/`
- User and model WAV: `human_eval/backend/data/audio/`
- Structured conversation logs: `human_eval/backend/data/logs/`
- Live server logs: the terminal running `serve.py`

Useful exports:

- `/api/admin/export/sessions.jsonl`
- `/api/admin/export/tasks.jsonl`
- `/api/admin/export/conversations.jsonl`
- `/api/admin/export/turns.jsonl`

Run checks:

```bash
python3 -m unittest human_eval.backend.tests.test_core human_eval.backend.tests.test_api
node --check human_eval/app.js
```

## Study design and participant flow

| Task | Participant action | Expected MiniCPM+ behavior | Main measure |
|---|---|---|---|
| S1: simple conversation | Chat about simple, stable facts | Stay local | Quality, latency, unnecessary escalation |
| S2: real-time information | Chat about something happening now or today | Escalate when fresh information is needed | Freshness, correctness, responsiveness |
| S3: complex conversation | Plan something together and continue the conversation naturally | Escalate when deeper reasoning is needed | Helpfulness and context retention |

**S3 design decision (approved for version 13):** S3 intentionally uses familiar, open-ended planning instead of the earlier scripted constraint-satisfaction flow. Participants found the old flow difficult to follow naturally. The new task still asks for about three turns to measure context retention, but it is not a direct replication of the paper's original S3 task. Do not pool or directly compare version 5 and version 13 S3 results. Freeze the task wording before recruitment; any later task change requires a new study version.

Each participant completes:

1. Device check and consent.
2. Task 1: conversation A, rating, conversation B, rating, pairwise choice.
3. Task 2: the same sequence.
4. Completion page.

Participants may finish a conversation early. Model identity and escalation status remain hidden.

The task card presents a numbered example flow plus an explicit goal and an end-of-task self-check. The flow is guidance rather than a hard gate: participants may paraphrase naturally and may finish whenever they believe they have tested the goal. Short interactions receive a QC flag for later review, not an automatic invalid label.

Assignments continuously target these task-pair proportions: 25% S1+S2, 25% S1+S3, and 50% S2+S3. Scenario and model order are balanced without requiring a fixed participant count.

## Human feedback

After each conversation, participants rate 1–5:

- **Correctness:** was the answer accurate?
- **Task helpfulness:** did it complete the task while following the participant's requirements?
- **Context consistency:** did it remember and use earlier information across turns?
- **Conversation naturalness:** was the timing and turn-taking natural, with clear and concise spoken answers?

These four cover the paper's primary delivered-answer accuracy outcome, practical task success, the S3 multi-turn gap, and the quality/latency tradeoff of a duplex spoken system. Separate instruction-following, overall-quality, clarity, and responsiveness ratings were removed because they substantially overlap these measures.

After both conversations in a task, participants complete one pairwise comparison. This is not another 1–5 rubric. It stores:

- Overall preference: Conversation 1, Conversation 2, or About the same.
- Reasons: About the same, More accurate, More complete, Clearer, Used current information, Did not require repetition, or More natural pacing.
- An optional comment.

## Stored data and metrics

- **Identity and assignment:** user/session/task/conversation IDs, capability, scenario, order, model arm, probe setting, and threshold tier.
- **Feedback:** four ratings, conversation comment, pairwise preference, reasons, task comment, and submission times.
- **Lifecycle:** voice interaction status and rating/evaluation status are stored separately. Analysis completion requires a submitted rating.
- **Partial completion:** every submitted conversation rating remains a primary outcome. When both conversations in one task have ratings and its comparison is submitted, that task is analysis-complete even if the participant never finishes the other task and the session later expires. `response_record_status` and `telemetry_complete` separately identify whether transcripts, routing, and latency were fully captured.
- **Interaction:** user/model WAV, transcripts and source, expert answer, turn count, interruption state, and conversation end reason.
- **Routing:** local/escalated action, threshold, EOT score and series, plus manual expected action and correctness review.
- **Latency:** speech end, gate decision, first model audio, response completion, expert, relay, stall, EOT-read, and ASR timing.
- **Persistence evidence:** browser-observed model audio, completed-turn acknowledgements, response-record status, and stable rating/comparison IDs. Finish drains final upstream events and returns a persistence receipt before the rating screen opens.
- **Audio quality:** input/output duration, speech detection, RMS/VAD statistics, and short or missing audio flags.
- **Guardrails:** timeout, crash, disconnect, empty response, interruption, missing transcript, routing-review status, and manual conversation QC (`needs_review`, `valid`, or `invalid`).

Expired reservations are persisted as session status `expired`; an in-progress conversation in that session becomes `abandoned` with `end_reason=reservation_expired`. A returning participant receives a new assignment rather than silently reviving the stale one. Exports include `reservation_status` and `reservation_active`.

Expiration releases an inactive assignment from balancing; it does not delete or invalidate ratings already submitted. Analysis should use conversation- and task-level readiness fields rather than filtering only for `session.status=completed`.

Raw JSON is the audit record. Flattened JSONL exports are intended for analysis. Exact fields are in [backend/DATA_SCHEMA.md](./backend/DATA_SCHEMA.md).

## Modal setup and deployment

The public participant site is:

<https://rhe9527--minicpm-human-eval.modal.run/>

It uses the separate `minicpm-human-eval` app and `human-eval-data` Volume. Inference is proxied to the existing `gate-demo-duplex` Voice service.

### One-time CLI setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r human_eval/backend/requirements.txt modal
```

With a workspace API token, avoid placing credentials in shell history. This command prompts for the `ak-...` ID and `as-...` secret:

```bash
modal token set --profile rhe9527 --activate --verify
modal config set-environment main
modal profile current
```

If you own the Modal account, `modal setup` is the browser-login alternative. Never commit or paste Modal credentials into source files.

Create the admin secret once and save the generated value in a password manager:

```bash
export HUMAN_EVAL_ADMIN_TOKEN="$(openssl rand -hex 32)"
modal secret create human-eval-secrets \
  HUMAN_EVAL_ADMIN_TOKEN="$HUMAN_EVAL_ADMIN_TOKEN"
```

### Deploy

The first full deployment needs both services:

```bash
modal deploy interactive_paper/demo_duplex.py
modal deploy human_eval/modal_app.py
```

For later changes:

- `human_eval/` UI, API, scenarios, or storage: deploy `human_eval/modal_app.py`.
- `interactive_paper/demo_duplex.py` model/protocol changes: deploy `interactive_paper/demo_duplex.py` too.
- README, tests, and downloaded data do not affect the running app.

Use a temporary hot-reloading `-dev` URL while iterating:

```bash
modal serve human_eval/modal_app.py
```

## User data and logs on Modal

The browser stores only its anonymous user ID in `sessionStorage`. Modal persists:

| Volume path | Contents |
|---|---|
| `/sessions/session_<id>.json` | Complete participant record and metrics |
| `/audio/<conversation_id>/*.wav` | User and model audio |
| `/logs/<UTC timestamp>_<conversation_id>.jsonl` | Structured backend/model events without PCM/base64 audio |

List files:

```bash
modal volume ls human-eval-data /sessions
modal volume ls human-eval-data /audio
modal volume ls human-eval-data /logs
```

Read one JSON or log in the terminal:

```bash
modal volume get human-eval-data /sessions/SESSION_FILE.json -
modal volume get human-eval-data /logs/LOG_FILE.jsonl -
```

Download the complete Volume:

```bash
modal volume get human-eval-data / ./human_eval_data
```

Export analysis-ready JSONL without downloading audio:

```bash
curl -H "Authorization: Bearer $HUMAN_EVAL_ADMIN_TOKEN" \
  "https://rhe9527--minicpm-human-eval.modal.run/api/admin/export/turns.jsonl" \
  -o turns.jsonl
```

Replace `turns` with `sessions`, `tasks`, or `conversations` for the other tables. Keep `HUMAN_EVAL_ADMIN_TOKEN` outside Git.

Stream live container logs:

```bash
modal app logs minicpm-human-eval --timestamps
```

### Clear pilot data

This permanently deletes collected records. Back up and stop the app first:

```bash
modal volume get human-eval-data / ./human_eval_backup
modal app stop minicpm-human-eval
modal volume rm -r human-eval-data /sessions
modal volume rm -r human-eval-data /audio
modal volume rm -r human-eval-data /logs
modal deploy human_eval/modal_app.py
```

Do not delete the `human-eval-data` Volume itself.

## Temporary pilot debug logging

`HUMAN_EVAL_DEBUG_MODEL_LOGS=1` is currently enabled in `modal_app.py`. After the participant clicks **Start conversation** and the model sends `ready`, the browser console shows model arm, probe, aggressive tier, protocol, and runtime mode. Refresh alone does not emit this message.

This breaks blinding. Before recruitment, change the flag to `0` and remove the marked blocks:

```bash
rg -n "HUMAN_EVAL_DEBUG_REMOVE_AFTER_PILOT" human_eval
```

## API summary

- `POST /api/study-sessions`: create or resume an assignment.
- `GET /api/model/readiness`: warm and check the model; formal mode also checks ASR configuration.
- `WS /api/conversations/{conversation_id}/stream`: proxy blinded audio to the assigned model.
- `POST /api/conversations/{conversation_id}/finalize`: save the conversation result.
- `PUT /api/conversations/{conversation_id}/rating`: save `correctness`, `helpfulness`, `context_consistency`, and `conversation_naturalness` ratings plus optional feedback.
- `PUT /api/tasks/{task_id}/comparison`: save pairwise feedback.
- `POST /api/study-sessions/{session_id}/complete`: complete the study.
- `GET /api/admin/export/{table}.jsonl`: export analysis data.
- `PUT /api/admin/turns/{turn_id}/routing-review`: save a transcript-informed routing label.
- `PUT /api/admin/conversations/{conversation_id}/quality-review`: save a task-adherence/data-quality label.
