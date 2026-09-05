# Persisted Human-Evaluation Data

Each `sessions/{session_id}.json` document is the complete audit record for one participant. Audio stays in `audio/{conversation_id}/`. Structured backend events use `logs/{UTC timestamp}_{conversation_id}.jsonl`; PCM/base64 audio is excluded from these logs. Analysis should normally use the flattened `sessions`, `tasks`, `conversations`, and `turns` JSONL exports.

## Session and assignment

- `user_id`, `session_id`, `study_version`, `schema_version`
- `status`, `created_at`, `started_at`, `completed_at`, `updated_at`; stale active reservations are persisted with session status `expired`
- private `pair_cell`, assignment time, reservation expiry, and expiration audit timestamps/reason
- exports derive `reservation_status` (`active`, `expired`, `completed`, or `not_applicable`) and `reservation_active`
- task order, `task_id`, capability `S1`/`S2`/`S3`, scenario ID, and whether escalation is expected for that capability
- conversation order, `conversation_id`, assigned `model`, `probe_on`, and `threshold_tier`

## Primary outcome data

- Per-conversation 1–5 ratings (`correctness`, `helpfulness`, `context_consistency`, and `conversation_naturalness`) and free-form feedback
- Pairwise preference (`first`, `second`, or `same`), selected reasons, and feedback
- Rating/comparison submission timestamps
- Separate interaction and evaluation lifecycle: conversation `status` records how the voice interaction ended, while `evaluation_status` records whether its rating was submitted
- A submitted rating is the primary per-conversation outcome, so conversation `analysis_complete=true` means the rating was saved. Turn telemetry is tracked separately with `telemetry_complete` and `response_record_status`.
- `response_record_status` is `recorded` when a server turn exists, `client_observed` when the browser received model audio but the final turn record is missing, `unverified_legacy` for older ratings with neither signal, or `not_observed` before a rateable response.
- Conversation ratings remain analysis-ready even if the participant does not finish the full session. At task level, `ratings_complete` means both model arms submitted ratings and `analysis_complete` additionally requires the pairwise comparison. `recorded_ratings_complete`/`telemetry_complete` show whether both model turns are also available. These fields do not depend on session status, so completed tasks inside expired sessions remain usable.
- New ratings and comparisons have stable `rating_id`/`comparison_id` values. Comparison exports also resolve the blinded `first`/`second` choice to `preferred_model` for analysis.
- Session completion status and completion code
- Task and scenario identity needed to compare matched X/Y prompts

## Interaction data

For every model turn:

- User WAV path, byte count, sample rate, transcript, transcript status, and source (`upstream_asr` or `posthoc_asr`)
- Model WAV path, byte count, sample rate, final transcript, and expert transcript when escalated
- Input-stream start, user-speech start/end, gate decision, first model audio, and response-complete timestamps
- Upstream EOT-to-gate, EOT-to-first-audio, and EOT-to-response-complete latency, measured on one model-runtime clock
- Model-reported expert, stall, relay, EOT score-read, and optional post-hoc ASR latency
- Input/output duration, speech-detected flag, input RMS mean/max, mean VAD threshold, and silence before EOT

## MiniCPM+ escalation data

- Whether the turn was eligible for escalation and whether it escalated
- Threshold tier, numerical threshold, final EOT score, per-chunk score series, action score, information-request decision, and EOT read time
- Total observed escalation and local-routing counts across MiniCPM+ turns
- Per-turn routing review: expected action, actual action, correctness, reviewer, note, and timestamp
- Reviewed/correct/incorrect/unreviewed routing counts
- Number of S1 MiniCPM+ conversations with any escalation and expected-task MiniCPM+ conversations with zero escalation; these are screening signals, not correctness labels

## Guardrail and data-quality data

- Conversation interaction status (`assigned`, `in_progress`, `interaction_completed`, `failed`, or `abandoned`) and evaluation status (`not_ready`, `pending`, `completed`, or `not_submitted`)
- Manual `quality_review.status` (`needs_review`, `valid`, or `invalid`), reason, note, reviewer, timestamp, and automatic screening flags
- Suggested task-flow length is not enforced. Fewer recorded turns than the task target adds `fewer_than_target_turns` for review but never blocks the participant or automatically invalidates data.
- Timeout, model crash, disconnect, interruption, and empty-response flags
- The browser reports whether model audio was received, its approximate duration, and the number of completed-turn acknowledgements. This is supporting evidence only; model identity and authoritative turn telemetry remain server-side.
- A conversation with neither a recorded turn nor browser-observed model audio is marked `abandoned` with `no_observed_response`; it cannot accept a rating and may be retried while the session reservation is active.
- If audio was observed but the final turn is missing, the rating is preserved with `response_record_status=client_observed` and the conversation receives a `response_not_persisted` QC flag.
- On Finish, the backend asks the upstream model to stop and briefly drains final `turn`/`bye` events before closing the socket. It sends `conversation_finished` only after persistence; the browser waits for this receipt before opening the rating screen.
- Input and output audio anomaly flags
- Structured error timestamp/message
- Interaction-ended, evaluation-completed, analysis-complete, QC-status, total-turn, MiniCPM+-turn, and anomaly counts

Null interpretation:

- `user.transcript=null` records an upstream ASR error or a missing transcript. The WAV remains available for correction or optional post-hoc ASR.
- MiniCPM+ `routing_review.status=unreviewed` means correctness is intentionally unknown. Routing correctness is never inferred from the task capability alone.
- `quality_review.status=needs_review` is likewise not an invalid label; it asks a reviewer to check task adherence and data quality.
- Records from schema 1.3 and earlier are normalized on read. Legacy conversation `status=completed` becomes `interaction_completed`; rating presence determines evaluation completion.
- Expert, stall, and relay fields do not apply to local turns and are omitted in new records.
- Missing EOT-derived latency fields indicate a collection defect, not “not applicable.” `eot_read_ms` measures only the cost of reading the gate score and must never be substituted for EOT-to-gate latency. Older affected records remain missing rather than being exported as a false `0ms` value.
- Blank or whitespace-only transcripts are normalized to missing and receive `missing_transcript=true`.

Analysis exports:

- `sessions.jsonl`: one row per participant, assignment, duration, counts, and anomaly totals.
- `tasks.jsonl`: one row per capability task and pairwise judgment, including rating count, `preferred_model`, rating/comparison readiness, telemetry readiness, and task-level QC readiness.
- `conversations.jsonl`: one row per model arm with rating metrics, `response_record_status`, client receipt evidence, end state, escalation count, guardrails, and the enclosing task's paired-analysis readiness.
- `turns.jsonl`: one row per user–assistant turn with transcripts, audio references, routing, latency, and audio quality.

Recommended inclusion rule:

- For per-model rating summaries, include every conversation row with `analysis_complete=true`, regardless of session status. Report or sensitivity-check `response_record_status`; do not silently discard `client_observed` ratings.
- For MiniCPM versus MiniCPM+ paired comparisons, include task rows with `analysis_complete=true` and group by `capability`. Use `telemetry_complete=true` only when the analysis also needs transcripts, escalation, or latency.
- Use task `quality_valid` or conversation `quality_status` as a separate task-adherence/QC filter; never discard data solely because its parent session expired.
