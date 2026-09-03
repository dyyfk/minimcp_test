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

- Per-conversation rating metrics and free-form feedback
- Pairwise preference (`first`, `second`, or `same`), selected reasons, and feedback
- Rating/comparison submission timestamps
- Separate interaction and evaluation lifecycle: conversation `status` records how the voice interaction ended, while `evaluation_status` records whether its rating was submitted
- `analysis_complete=true` and completion counts require a submitted rating; an ended interaction alone is not counted as complete analysis data
- Session completion status and completion code
- Task and scenario identity needed to compare matched X/Y prompts

## Interaction data

For every model turn:

- User WAV path, byte count, sample rate, transcript, transcript status, and source (`upstream_asr` or `posthoc_asr`)
- Model WAV path, byte count, sample rate, final transcript, and expert transcript when escalated
- Input-stream start, user-speech start/end, gate decision, first model audio, and response-complete timestamps
- Derived speech-end-to-gate, speech-end-to-first-audio, and speech-end-to-complete latency
- Model-reported first-audio, expert, stall, relay, EOT-read, and optional post-hoc ASR latency
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
- Input and output audio anomaly flags
- Structured error timestamp/message
- Interaction-ended, evaluation-completed, analysis-complete, QC-status, total-turn, MiniCPM+-turn, and anomaly counts

Null interpretation:

- `user.transcript=null` records an upstream ASR error or a missing transcript. The WAV remains available for correction or optional post-hoc ASR.
- MiniCPM+ `routing_review.status=unreviewed` means correctness is intentionally unknown. Routing correctness is never inferred from the task capability alone.
- `quality_review.status=needs_review` is likewise not an invalid label; it asks a reviewer to check task adherence and data quality.
- Records from schema 1.3 and earlier are normalized on read. Legacy conversation `status=completed` becomes `interaction_completed`; rating presence determines evaluation completion.
- Expert, stall, and relay fields do not apply to local turns and are omitted in new records.
- Missing core speech timestamps or derived latency fields indicate a collection defect, not “not applicable.” For older affected records, the turn export estimates speech end from gate time minus EOT-read time and sets `speech_end_estimated=true`; raw files remain unchanged.

Analysis exports:

- `sessions.jsonl`: one row per participant, assignment, duration, counts, and anomaly totals.
- `tasks.jsonl`: one row per capability task and pairwise judgment.
- `conversations.jsonl`: one row per model arm with rating metrics, end state, escalation count, and guardrails.
- `turns.jsonl`: one row per user–assistant turn with transcripts, audio references, routing, latency, and audio quality.
