"""Assignment and file persistence with no framework dependencies.

The study is small enough that one JSON file per participant is easier to
inspect and export than a database. A process lock plus atomic file replacement
keeps writes safe for a single backend process. Move this interface to SQLite
or Postgres only if recruitment scale requires multiple backend processes.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


SCHEMA_VERSION = "1.4"
MODEL_MINICPM = "minicpm"
MODEL_MINICPM_PLUS = "minicpm_plus"
DEFAULT_TIER = "aggressive"
RESERVATION_MINUTES = 30
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
ACTIVE_SESSION_STATUSES = {"assigned", "in_progress"}
TERMINAL_INTERACTION_STATUSES = {
    "interaction_completed",
    "completed",  # Legacy schema <= 1.3.
    "failed",
    "abandoned",
}

CAPABILITIES = {
    "simple_guardrail": {"code": "S1", "expected_escalation": False},
    "realtime": {"code": "S2", "expected_escalation": True},
    "context_reasoning": {"code": "S3", "expected_escalation": True},
}

# 1:1:2 produces 25% S1+S2, 25% S1+S3, 50% S2+S3.
PAIR_CELLS = {
    "s1_s2": {"weight": 1, "tasks": ["simple_guardrail", "realtime"]},
    "s1_s3": {"weight": 1, "tasks": ["simple_guardrail", "context_reasoning"]},
    "s2_s3": {"weight": 2, "tasks": ["context_reasoning", "realtime"]},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reservation_expires_at() -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=RESERVATION_MINUTES)).isoformat()


def reservation_state(
    session: dict[str, Any], now: datetime | None = None
) -> str:
    """Return an explicit reservation state for UI and analysis exports."""
    status = session.get("status")
    if status == "completed":
        return "completed"
    if status == "expired":
        return "expired"
    if status not in ACTIVE_SESSION_STATUSES:
        return "not_applicable"
    expires_at = session.get("assignment", {}).get("reservation_expires_at")
    try:
        expiry = datetime.fromisoformat(expires_at) if expires_at else None
    except (TypeError, ValueError):
        expiry = None
    return "active" if expiry and expiry > (now or datetime.now(timezone.utc)) else "expired"


def counts_toward_balance(
    session: dict[str, Any], now: datetime | None = None
) -> bool:
    """Count completed sessions and active reservations, but not dropouts."""
    if session.get("status") == "completed":
        return True
    return reservation_state(session, now) == "active"


def target_turns_for(conversation: dict[str, Any], capability: str | None = None) -> int:
    """Read the suggested task flow length with a safe capability fallback."""
    configured = conversation.get("scenario", {}).get("targetTurns")
    if isinstance(configured, int) and configured > 0:
        return configured
    return 3 if capability == "S3" else 2


def normalize_session(session: dict[str, Any]) -> dict[str, Any]:
    """Populate lifecycle/QC fields when reading records from older schemas."""
    session["schema_version"] = SCHEMA_VERSION
    for task in session.get("tasks", []):
        capability = task.get("capability")
        for conversation in task.get("conversations", []):
            if conversation.get("status") == "completed":
                conversation["status"] = "interaction_completed"
            rating = conversation.get("rating")
            if "evaluation_status" not in conversation:
                if rating:
                    conversation["evaluation_status"] = "completed"
                elif conversation.get("status") in TERMINAL_INTERACTION_STATUSES:
                    conversation["evaluation_status"] = "pending"
                else:
                    conversation["evaluation_status"] = "not_ready"
            if rating and not conversation.get("evaluation_completed_at"):
                conversation["evaluation_completed_at"] = rating.get("submitted_at")
            quality = conversation.setdefault(
                "quality_review",
                {
                    "status": (
                        "needs_review"
                        if conversation.get("status") in TERMINAL_INTERACTION_STATUSES
                        else "not_ready"
                    ),
                    "reason": None,
                    "note": "",
                    "reviewer": None,
                    "reviewed_at": None,
                    "automatic_flags": [],
                },
            )
            quality.setdefault("automatic_flags", [])
            if (
                conversation.get("status") in TERMINAL_INTERACTION_STATUSES
                and len(conversation.get("turns", []))
                < target_turns_for(conversation, capability)
                and "fewer_than_target_turns" not in quality["automatic_flags"]
            ):
                quality["automatic_flags"].append("fewer_than_target_turns")
    return session


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _least_filled_weighted_cell(
    counts: dict[str, int], weights: dict[str, int], rng: secrets.SystemRandom
) -> str:
    ratios = {name: counts.get(name, 0) / weight for name, weight in weights.items()}
    lowest = min(ratios.values())
    candidates = [name for name, ratio in ratios.items() if ratio == lowest]
    return rng.choice(candidates)


def _least_used_sequence(
    existing_sessions: Iterable[dict[str, Any]], capability_key: str, rng: secrets.SystemRandom
) -> int:
    counts = {cell: 0 for cell in range(4)}
    for session in existing_sessions:
        for task in session.get("tasks", []):
            if task.get("capability_key") == capability_key:
                cell = task.get("sequence_cell")
                if cell in counts:
                    counts[cell] += 1
    lowest = min(counts.values())
    return rng.choice([cell for cell, count in counts.items() if count == lowest])


def create_assignment(
    scenarios: dict[str, Any],
    existing_sessions: Iterable[dict[str, Any]],
    user_id: str | None = None,
    rng: secrets.SystemRandom | None = None,
) -> dict[str, Any]:
    """Create one balanced, blinded two-task assignment."""
    rng = rng or secrets.SystemRandom()
    study_version = scenarios.get("version", "unknown")
    existing = [
        session
        for session in existing_sessions
        if counts_toward_balance(session) and session.get("study_version") == study_version
    ]
    pair_counts = {name: 0 for name in PAIR_CELLS}
    for session in existing:
        cell = session.get("assignment", {}).get("pair_cell")
        if cell in pair_counts:
            pair_counts[cell] += 1

    pair_cell = _least_filled_weighted_cell(
        pair_counts,
        {name: config["weight"] for name, config in PAIR_CELLS.items()},
        rng,
    )
    task_keys = list(PAIR_CELLS[pair_cell]["tasks"])
    rng.shuffle(task_keys)

    definitions = {item["key"]: item for item in scenarios["tasks"]}
    session_id = new_id("session")
    assigned_user_id = user_id.strip() if user_id and user_id.strip() else new_id("user")
    tasks: list[dict[str, Any]] = []

    for task_order, capability_key in enumerate(task_keys, start=1):
        definition = definitions[capability_key]
        sequence_cell = _least_used_sequence(existing, capability_key, rng)
        scenario_order = [0, 1] if sequence_cell < 2 else [1, 0]
        model_order = (
            [MODEL_MINICPM, MODEL_MINICPM_PLUS]
            if sequence_cell % 2 == 0
            else [MODEL_MINICPM_PLUS, MODEL_MINICPM]
        )
        capability = CAPABILITIES[capability_key]
        conversations = []
        for conversation_order, (scenario_index, model) in enumerate(
            zip(scenario_order, model_order), start=1
        ):
            conversations.append(
                {
                    "conversation_id": new_id("conversation"),
                    "order": conversation_order,
                    "scenario_id": definition["scenarios"][scenario_index]["id"],
                    "scenario": deepcopy(definition["scenarios"][scenario_index]),
                    # Private assignment fields. public_session() removes them.
                    "model": model,
                    "probe_on": model == MODEL_MINICPM_PLUS,
                    "threshold_tier": DEFAULT_TIER if model == MODEL_MINICPM_PLUS else None,
                    "status": "assigned",
                    "started_at": None,
                    "ended_at": None,
                    "end_reason": None,
                    "turns": [],
                    "rating": None,
                    "evaluation_status": "not_ready",
                    "evaluation_completed_at": None,
                    "quality_review": {
                        "status": "not_ready",
                        "reason": None,
                        "note": "",
                        "reviewer": None,
                        "reviewed_at": None,
                        "automatic_flags": [],
                    },
                    "errors": [],
                    "anomalies": {key: False for key in ANOMALY_KEYS},
                }
            )
        tasks.append(
            {
                "task_id": new_id("task"),
                "order": task_order,
                "capability": capability["code"],
                "capability_key": capability_key,
                "expected_escalation": capability["expected_escalation"],
                "title": definition["participantTitle"],
                "sequence_cell": sequence_cell,
                "conversations": conversations,
                "comparison": None,
            }
        )

    now = utc_now()
    session = {
        "schema_version": SCHEMA_VERSION,
        "study_version": study_version,
        "session_id": session_id,
        "user_id": assigned_user_id,
        "status": "assigned",
        "created_at": now,
        "started_at": None,
        "completed_at": None,
        "updated_at": now,
        "assignment": {
            "pair_cell": pair_cell,
            "assigned_at": now,
            "reservation_expires_at": reservation_expires_at(),
        },
        "tasks": tasks,
        "summary": {},
    }
    recompute_summary(session)
    return session


def public_session(session: dict[str, Any]) -> dict[str, Any]:
    """Return participant-safe assignment data without model or gate identity."""
    public = deepcopy(session)
    public.pop("summary", None)
    public.get("assignment", {}).pop("pair_cell", None)
    for task in public.get("tasks", []):
        task.pop("sequence_cell", None)
        task.pop("expected_escalation", None)
        for conversation in task.get("conversations", []):
            conversation["turn_count"] = len(conversation.get("turns", []))
            conversation["target_turns"] = target_turns_for(
                conversation, task.get("capability")
            )
            for private_key in (
                "model",
                "probe_on",
                "threshold_tier",
                "turns",
                "errors",
                "anomalies",
                "model_runtime",
                "quality_review",
            ):
                conversation.pop(private_key, None)
    return public


ANOMALY_KEYS = (
    "timeout",
    "crash",
    "empty_response",
    "input_audio_anomaly",
    "output_audio_anomaly",
    "disconnect",
    "interrupted",
    "missing_transcript",
)


def recompute_summary(session: dict[str, Any]) -> None:
    normalize_session(session)
    summary: dict[str, Any] = {
        "task_count": len(session.get("tasks", [])),
        "conversation_count": 0,
        "interaction_ended_conversation_count": 0,
        "evaluation_completed_conversation_count": 0,
        "analysis_complete_conversation_count": 0,
        # Backward-compatible alias. As of schema 1.4, completion requires a rating.
        "completed_conversation_count": 0,
        "quality_valid_conversation_count": 0,
        "quality_invalid_conversation_count": 0,
        "quality_needs_review_conversation_count": 0,
        "turn_count": 0,
        "minicpm_plus_turn_count": 0,
        "escalation_count": 0,
        "local_routing_count": 0,
        "routing_reviewed_turn_count": 0,
        "routing_correct_turn_count": 0,
        "routing_incorrect_turn_count": 0,
        "routing_unreviewed_turn_count": 0,
        "s1_plus_conversation_with_escalation_count": 0,
        "expected_task_plus_conversation_with_zero_escalation_count": 0,
        "missing_user_transcript_count": 0,
        "anomalies": {key: 0 for key in ANOMALY_KEYS},
    }
    for task in session.get("tasks", []):
        for conversation in task.get("conversations", []):
            summary["conversation_count"] += 1
            if conversation.get("status") in TERMINAL_INTERACTION_STATUSES:
                summary["interaction_ended_conversation_count"] += 1
            if conversation.get("rating"):
                summary["evaluation_completed_conversation_count"] += 1
                summary["analysis_complete_conversation_count"] += 1
                summary["completed_conversation_count"] += 1
            quality_status = conversation.get("quality_review", {}).get("status")
            if quality_status == "valid":
                summary["quality_valid_conversation_count"] += 1
            elif quality_status == "invalid":
                summary["quality_invalid_conversation_count"] += 1
            elif quality_status == "needs_review":
                summary["quality_needs_review_conversation_count"] += 1
            conversation_escalations = 0
            for turn in conversation.get("turns", []):
                summary["turn_count"] += 1
                gate = turn.get("gate", {})
                is_plus = conversation.get("model") == MODEL_MINICPM_PLUS
                escalated = bool(gate.get("escalated"))
                if is_plus:
                    summary["minicpm_plus_turn_count"] += 1
                    if escalated:
                        summary["escalation_count"] += 1
                        conversation_escalations += 1
                    else:
                        summary["local_routing_count"] += 1
                    review = turn.get("routing_review", {})
                    if review.get("status") == "reviewed":
                        summary["routing_reviewed_turn_count"] += 1
                        if review.get("correct") is True:
                            summary["routing_correct_turn_count"] += 1
                        elif review.get("correct") is False:
                            summary["routing_incorrect_turn_count"] += 1
                    else:
                        summary["routing_unreviewed_turn_count"] += 1
                if not turn.get("user", {}).get("transcript"):
                    summary["missing_user_transcript_count"] += 1
                anomalies = turn.get("anomalies", {})
                for key in (
                    "empty_response",
                    "input_audio_anomaly",
                    "output_audio_anomaly",
                    "interrupted",
                    "missing_transcript",
                ):
                    if anomalies.get(key):
                        summary["anomalies"][key] += 1
            if conversation.get("model") == MODEL_MINICPM_PLUS and conversation.get("turns"):
                if task.get("capability") == "S1" and conversation_escalations:
                    summary["s1_plus_conversation_with_escalation_count"] += 1
                if task.get("expected_escalation") and not conversation_escalations:
                    summary[
                        "expected_task_plus_conversation_with_zero_escalation_count"
                    ] += 1
            for key in ("timeout", "crash", "disconnect"):
                if conversation.get("anomalies", {}).get(key):
                    summary["anomalies"][key] += 1
    session["summary"] = summary
    session["updated_at"] = utc_now()


def _elapsed_ms(start: str | None, end: str | None) -> int | None:
    if not start or not end:
        return None
    return max(
        0,
        int(
            (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
            * 1000
        ),
    )


def analysis_rows(sessions: Iterable[dict[str, Any]], table: str) -> list[dict[str, Any]]:
    """Flatten raw session documents into four analysis-friendly tables."""
    rows: list[dict[str, Any]] = []
    for session in sessions:
        session_fields = {
            "schema_version": session.get("schema_version"),
            "study_version": session.get("study_version"),
            "session_id": session.get("session_id"),
            "user_id": session.get("user_id"),
        }
        if table == "sessions":
            summary = session.get("summary", {})
            row = {
                **session_fields,
                "status": session.get("status"),
                "reservation_status": reservation_state(session),
                "reservation_active": reservation_state(session) == "active",
                "pair_cell": session.get("assignment", {}).get("pair_cell"),
                "reservation_expires_at": session.get("assignment", {}).get(
                    "reservation_expires_at"
                ),
                "expired_at": session.get("expired_at"),
                "created_at": session.get("created_at"),
                "started_at": session.get("started_at"),
                "completed_at": session.get("completed_at"),
                "study_duration_ms": _elapsed_ms(
                    session.get("started_at"), session.get("completed_at")
                ),
                **{
                    key: value
                    for key, value in summary.items()
                    if key != "anomalies"
                },
                **{
                    f"anomaly_{key}_count": value
                    for key, value in summary.get("anomalies", {}).items()
                },
            }
            rows.append(row)
            continue

        for task in session.get("tasks", []):
            task_fields = {
                **session_fields,
                "task_id": task.get("task_id"),
                "task_order": task.get("order"),
                "capability": task.get("capability"),
                "capability_key": task.get("capability_key"),
            }
            if table == "tasks":
                comparison = task.get("comparison") or {}
                rows.append(
                    {
                        **task_fields,
                        "task_title": task.get("title"),
                        "expected_escalation": task.get("expected_escalation"),
                        "sequence_cell": task.get("sequence_cell"),
                        "preference": comparison.get("preference"),
                        "comparison_reasons": comparison.get("reasons", []),
                        "comparison_feedback": comparison.get("feedback", ""),
                        "comparison_submitted_at": comparison.get("submitted_at"),
                    }
                )
                continue

            for conversation in task.get("conversations", []):
                conversation_fields = {
                    **task_fields,
                    "conversation_id": conversation.get("conversation_id"),
                    "conversation_order": conversation.get("order"),
                    "scenario_id": conversation.get("scenario_id"),
                    "model": conversation.get("model"),
                    "probe_enabled": conversation.get("probe_on"),
                }
                if table == "conversations":
                    rating = conversation.get("rating") or {}
                    turns = conversation.get("turns", [])
                    anomalies = conversation.get("anomalies", {})
                    quality = conversation.get("quality_review", {})
                    rows.append(
                        {
                            **conversation_fields,
                            "status": conversation.get("status"),
                            "interaction_status": conversation.get("status"),
                            "evaluation_status": conversation.get(
                                "evaluation_status"
                            ),
                            "evaluation_completed_at": conversation.get(
                                "evaluation_completed_at"
                            ),
                            "analysis_complete": bool(rating),
                            "quality_status": quality.get("status"),
                            "quality_reason": quality.get("reason"),
                            "quality_note": quality.get("note", ""),
                            "quality_reviewer": quality.get("reviewer"),
                            "quality_reviewed_at": quality.get("reviewed_at"),
                            "quality_automatic_flags": quality.get(
                                "automatic_flags", []
                            ),
                            "started_at": conversation.get("started_at"),
                            "ended_at": conversation.get("ended_at"),
                            "duration_ms": _elapsed_ms(
                                conversation.get("started_at"),
                                conversation.get("ended_at"),
                            ),
                            "end_reason": conversation.get("end_reason"),
                            "event_log_path": conversation.get(
                                "model_runtime", {}
                            ).get("event_log_path"),
                            "turn_count": len(turns),
                            "escalation_count": sum(
                                bool(turn.get("gate", {}).get("escalated"))
                                for turn in turns
                            ),
                            "routing_reviewed_turn_count": sum(
                                turn.get("routing_review", {}).get("status") == "reviewed"
                                for turn in turns
                            ),
                            "routing_correct_turn_count": sum(
                                turn.get("routing_review", {}).get("correct") is True
                                for turn in turns
                            ),
                            "routing_incorrect_turn_count": sum(
                                turn.get("routing_review", {}).get("correct") is False
                                for turn in turns
                            ),
                            "missing_user_transcript_count": sum(
                                not turn.get("user", {}).get("transcript")
                                for turn in turns
                            ),
                            "rating_metrics": rating.get("metrics", {}),
                            "rating_feedback": rating.get("feedback", ""),
                            "rating_submitted_at": rating.get("submitted_at"),
                            "completed_after_error": bool(
                                conversation.get("status")
                                == "interaction_completed"
                                and (anomalies.get("crash") or anomalies.get("disconnect"))
                            ),
                            "error_count": len(conversation.get("errors", [])),
                            **{
                                f"anomaly_{key}": value
                                for key, value in anomalies.items()
                            },
                            "errors": conversation.get("errors", []),
                        }
                    )
                    continue

                if table != "turns":
                    continue
                for turn in conversation.get("turns", []):
                    user = turn.get("user", {})
                    response = turn.get("model_response", {})
                    gate = turn.get("gate", {})
                    timestamps = dict(turn.get("timestamps", {}))
                    speech_end_estimated = False
                    if (
                        not timestamps.get("user_speech_ended_at")
                        and timestamps.get("gate_decision_at")
                        and isinstance(gate.get("eot_read_ms"), (int, float))
                    ):
                        estimated = datetime.fromisoformat(
                            timestamps["gate_decision_at"]
                        ) - timedelta(milliseconds=gate["eot_read_ms"])
                        timestamps["user_speech_ended_at"] = estimated.isoformat()
                        speech_end_estimated = True
                    latency = dict(turn.get("latency_ms", {}))
                    if timestamps.get("user_speech_ended_at"):
                        derived_latency = {
                            "speech_end_to_gate": _elapsed_ms(
                                timestamps.get("user_speech_ended_at"),
                                timestamps.get("gate_decision_at"),
                            ),
                            "speech_end_to_first_audio": _elapsed_ms(
                                timestamps.get("user_speech_ended_at"),
                                timestamps.get("first_model_audio_at"),
                            ),
                            "speech_end_to_response_complete": _elapsed_ms(
                                timestamps.get("user_speech_ended_at"),
                                timestamps.get("response_completed_at"),
                            ),
                        }
                        for key, value in derived_latency.items():
                            if latency.get(key) is None:
                                latency[key] = value
                    row = {
                        **conversation_fields,
                        "turn_id": turn.get("turn_id"),
                        "turn_index": turn.get("turn_index"),
                        "user_audio_path": user.get("audio_path"),
                        "user_audio_bytes": user.get("audio_bytes"),
                        "user_transcript": user.get("transcript"),
                        "user_transcript_status": user.get("transcript_status"),
                        "user_transcript_source": user.get("transcript_source"),
                        "model_audio_path": response.get("audio_path"),
                        "model_audio_bytes": response.get("audio_bytes"),
                        "model_transcript": response.get("transcript"),
                        "expert_transcript": response.get("expert_transcript"),
                        "escalated": gate.get("escalated"),
                        "threshold": gate.get("threshold"),
                        "eot_score": gate.get("score"),
                        "score_series": gate.get("score_series", []),
                        "eot_read_ms": gate.get("eot_read_ms"),
                        "act_score": gate.get("act_score"),
                        "is_information_request": gate.get(
                            "is_information_request"
                        ),
                        "routing_review_status": turn.get("routing_review", {}).get("status"),
                        "routing_expected_action": turn.get("routing_review", {}).get("expected_action"),
                        "routing_actual_action": turn.get("routing_review", {}).get("actual_action"),
                        "routing_correct": turn.get("routing_review", {}).get("correct"),
                        "routing_review_note": turn.get("routing_review", {}).get("note"),
                        "routing_reviewer": turn.get("routing_review", {}).get("reviewer"),
                        "routing_reviewed_at": turn.get("routing_review", {}).get("reviewed_at"),
                        "speech_end_estimated": speech_end_estimated,
                        **timestamps,
                        **{
                            (f"latency_{key}" if key.endswith("_ms") else f"latency_{key}_ms"): value
                            for key, value in latency.items()
                            if value is not None
                        },
                        **{
                            f"audio_{key}": value
                            for key, value in turn.get("audio_quality", {}).items()
                        },
                        **{
                            f"anomaly_{key}": value
                            for key, value in turn.get("anomalies", {}).items()
                        },
                        **{
                            f"model_{key}": value
                            for key, value in turn.get("raw_model_metrics", {}).items()
                        },
                    }
                    rows.append(row)
    return rows


class JsonSessionStore:
    """One atomic JSON document per session plus separate WAV files."""

    def __init__(
        self, root: Path | str, after_save: Callable[[], None] | None = None
    ):
        self.root = Path(root)
        self.sessions_dir = self.root / "sessions"
        self.audio_dir = self.root / "audio"
        self.logs_dir = self.root / "logs"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._after_save = after_save

    def set_after_save(self, callback: Callable[[], None] | None) -> None:
        """Register a persistence flush hook, such as Modal Volume.commit."""
        self._after_save = callback

    def _safe(self, value: str) -> str:
        if not SAFE_ID.fullmatch(value):
            raise ValueError(f"Invalid identifier: {value!r}")
        return value

    def _path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{self._safe(session_id)}.json"

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = []
            for path in sorted(self.sessions_dir.glob("session_*.json")):
                with path.open(encoding="utf-8") as handle:
                    rows.append(normalize_session(json.load(handle)))
            return rows

    def get(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            path = self._path(session_id)
            if not path.exists():
                raise KeyError(session_id)
            with path.open(encoding="utf-8") as handle:
                return normalize_session(json.load(handle))

    def expire_stale_sessions(
        self, now: datetime | None = None
    ) -> list[str]:
        """Persist an explicit terminal state for reservations past their TTL."""
        checked_at = now or datetime.now(timezone.utc)
        expired_ids: list[str] = []
        with self._lock:
            for session in self.list_sessions():
                if (
                    session.get("status") not in ACTIVE_SESSION_STATUSES
                    or reservation_state(session, checked_at) != "expired"
                ):
                    continue
                expiry = session.get("assignment", {}).get(
                    "reservation_expires_at"
                )
                session.update(
                    {
                        "status": "expired",
                        "expired_at": expiry or checked_at.isoformat(),
                        "expiration_recorded_at": checked_at.isoformat(),
                        "expiration_reason": "reservation_timeout",
                    }
                )
                for task in session.get("tasks", []):
                    for conversation in task.get("conversations", []):
                        status = conversation.get("status")
                        if status == "in_progress":
                            conversation.update(
                                {
                                    "status": "abandoned",
                                    "ended_at": expiry or checked_at.isoformat(),
                                    "end_reason": "reservation_expired",
                                }
                            )
                            quality = conversation.setdefault(
                                "quality_review", {}
                            )
                            quality["status"] = "needs_review"
                            flags = quality.setdefault("automatic_flags", [])
                            if "reservation_expired" not in flags:
                                flags.append("reservation_expired")
                        if (
                            conversation.get("status")
                            in TERMINAL_INTERACTION_STATUSES
                            and not conversation.get("rating")
                        ):
                            conversation["evaluation_status"] = "not_submitted"
                self.save(session)
                expired_ids.append(session["session_id"])
        return expired_ids

    def save(
        self, session: dict[str, Any], *, renew_reservation: bool = True
    ) -> None:
        with self._lock:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            if renew_reservation and session.get("status") in ACTIVE_SESSION_STATUSES:
                session.setdefault("assignment", {})[
                    "reservation_expires_at"
                ] = reservation_expires_at()
            recompute_summary(session)
            path = self._path(session["session_id"])
            temporary = path.with_suffix(".json.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(session, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            if self._after_save:
                self._after_save()

    def event_log_path(self, conversation_id: str) -> Path:
        """Return one UTC-timestamped JSONL log path per conversation."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return self.logs_dir / f"{timestamp}_{self._safe(conversation_id)}.jsonl"

    def append_event_log(self, path: Path, event: dict[str, Any]) -> None:
        """Append a small structured event; the next session save commits it."""
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def create(self, session: dict[str, Any]) -> None:
        with self._lock:
            if self._path(session["session_id"]).exists():
                raise ValueError("Session already exists")
            self.save(session)

    def mutate(self, session_id: str, change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        with self._lock:
            session = self.get(session_id)
            change(session)
            self.save(session)
            return session

    def find_conversation(self, conversation_id: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
        self._safe(conversation_id)
        with self._lock:
            for session in self.list_sessions():
                for task in session.get("tasks", []):
                    for conversation in task.get("conversations", []):
                        if conversation.get("conversation_id") == conversation_id:
                            return session["session_id"], task, conversation
        raise KeyError(conversation_id)

    def find_task(self, task_id: str) -> tuple[str, dict[str, Any]]:
        self._safe(task_id)
        with self._lock:
            for session in self.list_sessions():
                for task in session.get("tasks", []):
                    if task.get("task_id") == task_id:
                        return session["session_id"], task
        raise KeyError(task_id)

    def mutate_turn(
        self,
        turn_id: str,
        change: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None],
    ) -> dict[str, Any]:
        """Atomically update one turn and return its containing session."""
        self._safe(turn_id)
        with self._lock:
            for session in self.list_sessions():
                for task in session.get("tasks", []):
                    for conversation in task.get("conversations", []):
                        for turn in conversation.get("turns", []):
                            if turn.get("turn_id") == turn_id:
                                change(task, conversation, turn)
                                self.save(session)
                                return session
        raise KeyError(turn_id)

    def mutate_conversation(
        self, conversation_id: str, change: Callable[[dict[str, Any], dict[str, Any]], None]
    ) -> dict[str, Any]:
        session_id, _, _ = self.find_conversation(conversation_id)

        def apply(session: dict[str, Any]) -> None:
            for task in session["tasks"]:
                for conversation in task["conversations"]:
                    if conversation["conversation_id"] == conversation_id:
                        change(task, conversation)
                        if conversation.get("status") == "in_progress":
                            session["status"] = "in_progress"
                            session["started_at"] = session.get("started_at") or utc_now()
                        return
            raise KeyError(conversation_id)

        return self.mutate(session_id, apply)

    def mutate_task(
        self, task_id: str, change: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        session_id, _ = self.find_task(task_id)

        def apply(session: dict[str, Any]) -> None:
            for task in session["tasks"]:
                if task["task_id"] == task_id:
                    change(task)
                    return
            raise KeyError(task_id)

        return self.mutate(session_id, apply)

    def audio_path(self, conversation_id: str, turn_id: str, side: str) -> Path:
        if side not in {"user", "model"}:
            raise ValueError("side must be user or model")
        directory = self.audio_dir / self._safe(conversation_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{self._safe(turn_id)}_{side}.wav"
