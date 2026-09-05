"""FastAPI entry point for the human-evaluation backend.

Run with:
    uvicorn human_eval.backend.app:app --host 127.0.0.1 --port 4173
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from urllib.parse import quote

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .core import (
    ACTIVE_SESSION_STATUSES,
    TERMINAL_INTERACTION_STATUSES,
    JsonSessionStore,
    analysis_rows,
    create_assignment,
    meaningful_text,
    new_id,
    public_session,
    response_record_status,
    response_was_observed,
    target_turns_for,
    utc_now,
)
from .model_gateway import (
    ConversationRecorder,
    DemoModelGateway,
    transcript_collection_settings,
)


HUMAN_EVAL_DIR = Path(__file__).resolve().parent.parent
SCENARIOS = json.loads((HUMAN_EVAL_DIR / "scenarios.json").read_text(encoding="utf-8"))
DATA_DIR = Path(os.getenv("HUMAN_EVAL_DATA_DIR", Path(__file__).parent / "data"))
CONVERSATION_LIMIT_SECONDS = int(os.getenv("HUMAN_EVAL_CONVERSATION_SECONDS", "120"))
FINISH_DRAIN_SECONDS = float(os.getenv("HUMAN_EVAL_FINISH_DRAIN_SECONDS", "8"))
NO_STORE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

store = JsonSessionStore(DATA_DIR)
model_gateway = DemoModelGateway()
assignment_lock = threading.Lock()
model_warm_task: asyncio.Task[dict[str, Any]] | None = None
app = FastAPI(title="MiniCPM Human Evaluation", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:4173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, max_length=128)
    force_new: bool = False


class RatingRequest(BaseModel):
    metrics: Dict[str, Any]
    feedback: str = Field(default="", max_length=5000)
    client_received_model_audio: bool = False
    client_completed_turn_count: int = Field(default=0, ge=0)
    client_model_audio_ms: int = Field(default=0, ge=0)


class ComparisonRequest(BaseModel):
    preference: str
    reasons: list[str] = Field(default_factory=list)
    feedback: str = Field(default="", max_length=5000)


class FinalizeConversationRequest(BaseModel):
    end_reason: str
    timeout: bool = False
    crash: bool = False
    disconnect: bool = False
    error: Optional[str] = Field(default=None, max_length=2000)
    client_received_model_audio: bool = False
    client_completed_turn_count: int = Field(default=0, ge=0)
    client_model_audio_ms: int = Field(default=0, ge=0)


class RoutingReviewRequest(BaseModel):
    expected_action: Literal["local", "escalate", "not_applicable"]
    reviewer: str = Field(min_length=1, max_length=128)
    note: str = Field(default="", max_length=2000)


class QualityReviewRequest(BaseModel):
    status: Literal["valid", "invalid", "needs_review"]
    reviewer: str = Field(min_length=1, max_length=128)
    reason: Optional[str] = Field(default=None, max_length=128)
    note: str = Field(default="", max_length=2000)


def _not_found(label: str, identifier: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{label} not found: {identifier}")


def _validated_rating_metrics(metrics: dict[str, Any]) -> dict[str, int]:
    metric_ids = tuple(question["id"] for question in SCENARIOS["ratingQuestions"])
    expected = set(metric_ids)
    provided = set(metrics)
    if provided != expected:
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unknown: {', '.join(extra)}")
        raise HTTPException(
            status_code=422,
            detail=f"Rating metrics must match the study rubric ({'; '.join(details)})",
        )
    normalized: dict[str, int] = {}
    for metric_id in metric_ids:
        value = metrics[metric_id]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not float(value).is_integer()
            or not 1 <= int(value) <= 5
        ):
            raise HTTPException(
                status_code=422,
                detail=f"Rating metric '{metric_id}' must be an integer from 1 to 5",
            )
        normalized[metric_id] = int(value)
    return normalized


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _merge_client_observation(
    conversation: dict[str, Any],
    *,
    received_model_audio: bool,
    completed_turn_count: int,
    model_audio_ms: int,
) -> None:
    """Merge the participant-side receipt signal without reducing prior values."""
    observation = conversation.setdefault("client_observation", {})
    observation.update(
        {
            "received_model_audio": bool(
                observation.get("received_model_audio") or received_model_audio
            ),
            "completed_turn_count": max(
                int(observation.get("completed_turn_count") or 0),
                completed_turn_count,
            ),
            "model_audio_ms": max(
                int(observation.get("model_audio_ms") or 0), model_audio_ms
            ),
            "reported_at": utc_now(),
        }
    )


def _record_client_observation(
    conversation_id: str,
    *,
    received_model_audio: bool,
    completed_turn_count: int,
    model_audio_ms: int,
) -> None:
    store.mutate_conversation(
        conversation_id,
        lambda _task, conversation: _merge_client_observation(
            conversation,
            received_model_audio=received_model_audio,
            completed_turn_count=completed_turn_count,
            model_audio_ms=model_audio_ms,
        ),
    )


def _require_active_conversation(
    conversation_id: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Reject participant writes after a reservation has expired."""
    store.expire_stale_sessions()
    try:
        session_id, task, conversation = store.find_conversation(conversation_id)
        session = store.get(session_id)
    except (KeyError, ValueError):
        raise _not_found("Conversation", conversation_id)
    if session.get("status") not in ACTIVE_SESSION_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Study session is no longer active ({session.get('status')})",
        )
    return session_id, task, conversation


def require_admin(authorization: Optional[str] = Header(default=None)) -> None:
    """Protect private exports on public deployments; keep local setup simple."""
    token = os.getenv("HUMAN_EVAL_ADMIN_TOKEN")
    public = os.getenv("HUMAN_EVAL_PUBLIC", "0").lower() in {"1", "true", "yes"}
    if not token and not public:
        return
    if not token:
        raise HTTPException(
            status_code=503,
            detail="HUMAN_EVAL_ADMIN_TOKEN is required for a public deployment",
        )
    supplied = authorization or ""
    expected = f"Bearer {token}"
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def ensure_model_ready() -> dict[str, Any]:
    """Share one in-flight GPU warm-up across pages and conversations."""
    global model_warm_task
    if model_warm_task is None:
        model_warm_task = asyncio.create_task(model_gateway.wait_until_ready())
    task = model_warm_task
    try:
        return await asyncio.shield(task)
    finally:
        if task.done() and model_warm_task is task:
            model_warm_task = None


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "study_version": SCENARIOS["version"],
        **transcript_collection_settings(),
    }


@app.get("/api/model/readiness")
async def model_readiness() -> dict[str, Any]:
    transcript_settings = transcript_collection_settings()
    if (
        transcript_settings["transcripts_required"]
        and not transcript_settings["transcript_collection_configured"]
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Full transcript collection is required, but neither the "
                "upstream protocol nor post-hoc ASR provides it"
            ),
        )
    try:
        payload = await ensure_model_ready()
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Model warm-up failed: {error}")
    return {
        "ready": True,
        "busy": bool(payload.get("busy")),
        "load_s": payload.get("load_s"),
        **transcript_settings,
    }


@app.put("/api/admin/turns/{turn_id}/routing-review")
def save_routing_review(
    turn_id: str,
    request: RoutingReviewRequest,
    _admin: None = Depends(require_admin),
) -> dict[str, Any]:
    """Save a transcript-informed routing label; never infer it from task alone."""
    result: dict[str, Any] = {}

    def update(
        _task: dict[str, Any], conversation: dict[str, Any], turn: dict[str, Any]
    ) -> None:
        if conversation.get("model") != "minicpm_plus":
            raise ValueError("Routing review only applies to MiniCPM+ turns")
        actual_action = (
            "escalate" if turn.get("gate", {}).get("escalated") else "local"
        )
        correct = (
            None
            if request.expected_action == "not_applicable"
            else actual_action == request.expected_action
        )
        review = {
            "status": "reviewed",
            "expected_action": request.expected_action,
            "actual_action": actual_action,
            "correct": correct,
            "note": request.note,
            "reviewer": request.reviewer,
            "reviewed_at": utc_now(),
        }
        turn["routing_review"] = review
        result.update(review)

    try:
        store.mutate_turn(turn_id, update)
    except KeyError:
        raise _not_found("Turn", turn_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))
    return result


@app.put("/api/admin/conversations/{conversation_id}/quality-review")
def save_quality_review(
    conversation_id: str,
    request: QualityReviewRequest,
    _admin: None = Depends(require_admin),
) -> dict[str, Any]:
    """Store a manual task-adherence/data-quality decision."""
    result: dict[str, Any] = {}

    def update(_task: dict[str, Any], conversation: dict[str, Any]) -> None:
        if conversation.get("status") not in TERMINAL_INTERACTION_STATUSES:
            raise ValueError("Quality review requires an ended interaction")
        existing = conversation.get("quality_review", {})
        review = {
            "status": request.status,
            "reason": request.reason,
            "note": request.note,
            "reviewer": request.reviewer,
            "reviewed_at": utc_now(),
            "automatic_flags": existing.get("automatic_flags", []),
        }
        conversation["quality_review"] = review
        result.update(review)

    try:
        store.mutate_conversation(conversation_id, update)
    except KeyError:
        raise _not_found("Conversation", conversation_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))
    return result


@app.post("/api/study-sessions")
def create_or_resume_session(request: CreateSessionRequest) -> dict[str, Any]:
    with assignment_lock:
        store.expire_stale_sessions()
        existing = store.list_sessions()
        if request.user_id and not request.force_new:
            for session in reversed(existing):
                if session.get("user_id") == request.user_id:
                    if session.get("study_version") != SCENARIOS["version"]:
                        continue
                    if session.get("status") in ACTIVE_SESSION_STATUSES:
                        store.save(session)  # renew the active reservation
                        return public_session(session)
                    if session.get("status") == "completed":
                        return public_session(session)
        session = create_assignment(SCENARIOS, existing, request.user_id)
        store.create(session)
    print(
        "[assignment]",
        json.dumps(
            {
                "session_id": session["session_id"],
                "user_id": session["user_id"],
                "pair_cell": session["assignment"]["pair_cell"],
                "tasks": [
                    {
                        "capability": task["capability"],
                        "sequence_cell": task["sequence_cell"],
                        "models": [item["model"] for item in task["conversations"]],
                    }
                    for task in session["tasks"]
                ],
            }
        ),
        flush=True,
    )
    return public_session(session)


@app.get("/api/study-sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    store.expire_stale_sessions()
    try:
        return public_session(store.get(session_id))
    except (KeyError, ValueError):
        raise _not_found("Session", session_id)


@app.put("/api/conversations/{conversation_id}/rating")
def save_rating(conversation_id: str, request: RatingRequest) -> dict[str, Any]:
    _require_active_conversation(conversation_id)
    metrics = _validated_rating_metrics(request.metrics)

    def update(_task: dict[str, Any], conversation: dict[str, Any]) -> None:
        if conversation.get("status") not in TERMINAL_INTERACTION_STATUSES:
            raise ValueError("Finish the interaction before submitting its rating")
        _merge_client_observation(
            conversation,
            received_model_audio=request.client_received_model_audio,
            completed_turn_count=request.client_completed_turn_count,
            model_audio_ms=request.client_model_audio_ms,
        )
        if not response_was_observed(conversation):
            raise ValueError("A rating requires an observed assistant response")
        submitted_at = utc_now()
        previous_rating = conversation.get("rating") or {}
        conversation.update(
            {
                "rating": {
                    "rating_id": previous_rating.get("rating_id")
                    or new_id("rating"),
                    "metrics": metrics,
                    "feedback": request.feedback,
                    "submitted_at": submitted_at,
                    "response_record_status": response_record_status(
                        conversation
                    ),
                    "turn_count_at_submission": len(
                        conversation.get("turns", [])
                    ),
                    "client_received_model_audio": conversation.get(
                        "client_observation", {}
                    ).get("received_model_audio"),
                },
                "evaluation_status": "completed",
                "evaluation_completed_at": submitted_at,
            }
        )

    try:
        store.mutate_conversation(conversation_id, update)
    except KeyError:
        raise _not_found("Conversation", conversation_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))
    saved = store.find_conversation(conversation_id)[2]
    return {
        "saved": True,
        "rating_id": saved["rating"].get("rating_id"),
        "response_record_status": response_record_status(saved),
    }


@app.put("/api/tasks/{task_id}/comparison")
def save_comparison(task_id: str, request: ComparisonRequest) -> dict[str, bool]:
    if request.preference not in {"first", "second", "same"}:
        raise HTTPException(status_code=422, detail="Invalid preference")
    try:
        store.expire_stale_sessions()
        session_id, task = store.find_task(task_id)
        if store.get(session_id).get("status") not in ACTIVE_SESSION_STATUSES:
            raise ValueError("Study session is no longer active")
        if not all(
            conversation.get("evaluation_status") == "completed"
            for conversation in task.get("conversations", [])
        ):
            raise ValueError("Rate both conversations before comparing them")
        store.mutate_task(
            task_id,
            lambda task: task.update(
                {
                    "comparison": {
                        "comparison_id": (task.get("comparison") or {}).get(
                            "comparison_id"
                        )
                        or new_id("comparison"),
                        "preference": request.preference,
                        "reasons": request.reasons,
                        "feedback": request.feedback,
                        "conversation_ids": [
                            item.get("conversation_id")
                            for item in sorted(
                                task.get("conversations", []),
                                key=lambda item: item.get("order", 0),
                            )
                        ],
                        "submitted_at": utc_now(),
                    }
                }
            ),
        )
    except KeyError:
        raise _not_found("Task", task_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))
    return {"saved": True}


def _finish_conversation(
    conversation_id: str,
    end_reason: str,
    *,
    timeout: bool = False,
    crash: bool = False,
    disconnect: bool = False,
    error: Optional[str] = None,
    client_received_model_audio: bool = False,
    client_completed_turn_count: int = 0,
    client_model_audio_ms: int = 0,
) -> None:
    def update(task: dict[str, Any], conversation: dict[str, Any]) -> None:
        _merge_client_observation(
            conversation,
            received_model_audio=client_received_model_audio,
            completed_turn_count=client_completed_turn_count,
            model_audio_ms=client_model_audio_ms,
        )
        if conversation.get("status") in TERMINAL_INTERACTION_STATUSES:
            if response_was_observed(conversation) and not conversation.get("rating"):
                conversation["evaluation_status"] = "pending"
                flags = conversation.setdefault("quality_review", {}).setdefault(
                    "automatic_flags", []
                )
                for stale_flag in ("no_observed_response", "no_completed_response"):
                    if stale_flag in flags:
                        flags.remove(stale_flag)
                if not conversation.get("turns") and "response_not_persisted" not in flags:
                    flags.append("response_not_persisted")
            return
        turn_count = len(conversation.get("turns", []))
        target_turns = target_turns_for(conversation, task.get("capability"))
        has_recorded_answer = turn_count > 0
        has_observed_answer = response_was_observed(conversation)
        if not has_observed_answer:
            status = "abandoned"
        elif crash:
            status = "failed"
        elif end_reason in {"user_finished", "time_limit"}:
            status = "interaction_completed"
        else:
            status = "abandoned"
        automatic_flags: list[str] = []
        if not has_observed_answer:
            automatic_flags.append("no_observed_response")
            automatic_flags.append("no_completed_response")
        elif not has_recorded_answer:
            automatic_flags.append("response_not_persisted")
        if turn_count < target_turns:
            automatic_flags.append("fewer_than_target_turns")
        if end_reason not in {"user_finished", "time_limit"}:
            automatic_flags.append("abnormal_end")
        if any(
            not meaningful_text(turn.get("user", {}).get("transcript"))
            for turn in conversation.get("turns", [])
        ):
            automatic_flags.append("missing_transcript")
        conversation.update(
            {
                "status": status,
                "ended_at": utc_now(),
                "end_reason": end_reason,
                "evaluation_status": (
                    "pending" if has_observed_answer else "not_submitted"
                ),
                "evaluation_completed_at": None,
                "quality_review": {
                    "status": "needs_review",
                    "reason": None,
                    "note": "",
                    "reviewer": None,
                    "reviewed_at": None,
                    "automatic_flags": automatic_flags,
                },
            }
        )
        if error:
            conversation.setdefault("errors", []).append(
                {"timestamp": utc_now(), "message": error[:2000]}
            )
        anomalies = conversation.setdefault("anomalies", {})
        anomalies["timeout"] = anomalies.get("timeout", False) or timeout
        anomalies["crash"] = anomalies.get("crash", False) or crash
        anomalies["disconnect"] = anomalies.get("disconnect", False) or disconnect

    store.mutate_conversation(conversation_id, update)


def _participant_model_event(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Keep routing/gate telemetry server-side while forwarding voice UX events."""
    event_type = payload.get("type")
    if event_type == "hello":
        ready: dict[str, Any] = {"type": "ready"}
        # HUMAN_EVAL_DEBUG_REMOVE_AFTER_PILOT_BEGIN
        # Temporary browser-console visibility for integration testing. This
        # exposes the blinded arm and must be disabled before real recruitment.
        if os.getenv("HUMAN_EVAL_DEBUG_MODEL_LOGS", "0") == "1":
            ready["debug"] = {
                "model": (
                    "minicpm_plus" if payload.get("probe_on") else "minicpm"
                ),
                "probe_on": payload.get("probe_on"),
                "tier": payload.get("tier"),
                "runtime_mode": payload.get("mode"),
                "protocol": payload.get("protocol"),
            }
        # HUMAN_EVAL_DEBUG_REMOVE_AFTER_PILOT_END
        return ready
    if event_type == "turn":
        return {"type": "turn"}
    allowed_fields = {
        "phase": ("v",),
        "audio": ("sr", "pcm"),
        "speech": ("on",),
        "duck": (),
        "resume": (),
        "interrupt": (),
    }
    if event_type not in allowed_fields:
        return None
    return {
        "type": event_type,
        **{key: payload.get(key) for key in allowed_fields[event_type]},
    }


async def _wait_for_model_drain(
    model_task: asyncio.Task[str], recorder: ConversationRecorder
) -> None:
    """Allow final upstream turn/bye events to persist after a stop request."""
    try:
        if model_task.done():
            await model_task
            return
        await asyncio.wait_for(
            asyncio.shield(model_task), timeout=FINISH_DRAIN_SECONDS
        )
    except asyncio.TimeoutError:
        recorder.record_backend_event(
            "finish_drain_timeout", {"timeout_s": FINISH_DRAIN_SECONDS}
        )
    except Exception as error:
        # The upstream demo sometimes sends its final turn and `bye`, then
        # closes without a valid WebSocket close frame. The useful data has
        # already been persisted, so this is an expected stop-path event.
        if not _is_upstream_close_error(error):
            raise
        recorder.record_backend_event(
            "upstream_closed_after_stop",
            {"error_type": type(error).__name__, "message": str(error)[:2000]},
        )


def _is_upstream_close_error(error: Exception) -> bool:
    return type(error).__name__ in {
        "ConnectionClosed",
        "ConnectionClosedError",
        "ConnectionClosedOK",
    }


@app.post("/api/conversations/{conversation_id}/finalize")
def finalize_conversation(
    conversation_id: str, request: FinalizeConversationRequest
) -> dict[str, Any]:
    _require_active_conversation(conversation_id)
    try:
        _finish_conversation(
            conversation_id,
            request.end_reason,
            timeout=request.timeout,
            crash=request.crash,
            disconnect=request.disconnect,
            error=request.error,
            client_received_model_audio=request.client_received_model_audio,
            client_completed_turn_count=request.client_completed_turn_count,
            client_model_audio_ms=request.client_model_audio_ms,
        )
    except KeyError:
        raise _not_found("Conversation", conversation_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))
    conversation = store.find_conversation(conversation_id)[2]
    return {
        "saved": True,
        "rating_allowed": response_was_observed(conversation),
        "turn_count": len(conversation.get("turns", [])),
        "status": conversation.get("status"),
        "response_record_status": response_record_status(conversation),
    }


@app.post("/api/study-sessions/{session_id}/complete")
def complete_session(session_id: str) -> dict[str, str]:
    store.expire_stale_sessions()
    try:
        session = store.get(session_id)
    except (KeyError, ValueError):
        raise _not_found("Session", session_id)
    conversations = [
        conversation
        for task in session["tasks"]
        for conversation in task["conversations"]
    ]
    if session.get("status") not in ACTIVE_SESSION_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Study session is no longer active ({session.get('status')})",
        )
    if not all(
        item.get("status") in TERMINAL_INTERACTION_STATUSES
        and item.get("evaluation_status") == "completed"
        and item.get("rating")
        for item in conversations
    ):
        raise HTTPException(
            status_code=409,
            detail="All interactions must end and all ratings must be submitted",
        )
    if not all(task.get("comparison") for task in session["tasks"]):
        raise HTTPException(status_code=409, detail="All task comparisons must be complete")

    completion_code = f"DONE-{session_id[-8:].upper()}"

    def mark_complete(target: dict[str, Any]) -> None:
        target.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "completion_code": completion_code,
            }
        )

    store.mutate(session_id, mark_complete)
    return {"completion_code": completion_code}


@app.get("/api/admin/export.jsonl")
def export_sessions(_admin: None = Depends(require_admin)) -> Response:
    store.expire_stale_sessions()
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in store.list_sessions())
    return Response(content=body, media_type="application/x-ndjson")


@app.get("/api/admin/export/{table}.jsonl")
def export_analysis_table(
    table: str, _admin: None = Depends(require_admin)
) -> Response:
    if table not in {"sessions", "tasks", "conversations", "turns"}:
        raise HTTPException(status_code=404, detail="Unknown analysis table")
    store.expire_stale_sessions()
    body = "".join(
        json.dumps(row, ensure_ascii=False) + "\n"
        for row in analysis_rows(store.list_sessions(), table)
    )
    return Response(content=body, media_type="application/x-ndjson")


@app.websocket("/api/conversations/{conversation_id}/stream")
async def conversation_stream(client: WebSocket, conversation_id: str) -> None:
    await client.accept()
    try:
        store.expire_stale_sessions()
        session_id, task, conversation = store.find_conversation(conversation_id)
        if store.get(session_id).get("status") not in ACTIVE_SESSION_STATUSES:
            await client.close(code=4409, reason="Study session is no longer active")
            return
    except (KeyError, ValueError):
        await client.close(code=4404, reason="Conversation not found")
        return

    recorder = ConversationRecorder(
        store, conversation_id, conversation["model"], task["capability"]
    )
    upstream_url = model_gateway.websocket_url(conversation["model"])
    finish_reason = "disconnect"

    try:
        import websockets

        await ensure_model_ready()
        async with websockets.connect(
            upstream_url, max_size=None, open_timeout=60, close_timeout=1
        ) as upstream:
            async def client_to_model() -> str:
                while True:
                    message = await client.receive()
                    if message.get("type") == "websocket.disconnect":
                        return "disconnect"
                    if message.get("bytes") is not None:
                        chunk = message["bytes"]
                        recorder.record_client_audio(chunk)
                        await upstream.send(chunk)
                        continue
                    text = message.get("text")
                    if text is None:
                        continue
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        payload = {}
                    if payload.get("type") == "finish_conversation":
                        recorder.record_backend_event(
                            "participant_finish_request", payload
                        )
                        _record_client_observation(
                            conversation_id,
                            received_model_audio=bool(
                                payload.get("client_received_model_audio")
                            ),
                            completed_turn_count=_nonnegative_int(
                                payload.get("client_completed_turn_count")
                            ),
                            model_audio_ms=_nonnegative_int(
                                payload.get("client_model_audio_ms")
                            ),
                        )
                        try:
                            await upstream.send(json.dumps({"type": "stop"}))
                        except Exception as error:
                            if not _is_upstream_close_error(error):
                                raise
                            recorder.record_backend_event(
                                "upstream_closed_before_stop",
                                {
                                    "error_type": type(error).__name__,
                                    "message": str(error)[:2000],
                                },
                            )
                        return (
                            "time_limit"
                            if payload.get("end_reason") == "time_limit"
                            else "user_finished"
                        )
                    recorder.record_client_event(payload)
                    await upstream.send(text)

            async def model_to_client() -> str:
                async for message in upstream:
                    if isinstance(message, bytes):
                        await client.send_bytes(message)
                        continue
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        await client.send_text(message)
                        continue
                    recorder.record_server_event(payload)
                    # HUMAN_EVAL_DEBUG_REMOVE_AFTER_PILOT_BEGIN
                    # Private server log; inspect with `modal app logs`.
                    if payload.get("type") == "hello":
                        print(
                            "[HUMAN_EVAL_DEBUG_REMOVE_AFTER_PILOT]",
                            json.dumps(
                                {
                                    "conversation_id": conversation_id,
                                    "model": conversation["model"],
                                    "probe_on": payload.get("probe_on"),
                                    "tier": payload.get("tier"),
                                    "runtime_mode": payload.get("mode"),
                                    "protocol": payload.get("protocol"),
                                }
                            ),
                            flush=True,
                        )
                    # HUMAN_EVAL_DEBUG_REMOVE_AFTER_PILOT_END
                    if payload.get("type") == "turn":
                        # Persist first: the browser may enable its Finish button
                        # as soon as it receives this turn-complete event.
                        await recorder.finalize_turn(payload)
                    participant_event = _participant_model_event(payload)
                    if participant_event:
                        await client.send_json(participant_event)
                return "upstream_closed"

            async def time_limit() -> str:
                await asyncio.sleep(CONVERSATION_LIMIT_SECONDS)
                await client.send_json({"type": "auto_finish", "reason": "time_limit"})
                return "time_limit"

            client_task = asyncio.create_task(client_to_model())
            model_task = asyncio.create_task(model_to_client())
            limit_task = asyncio.create_task(time_limit())
            tasks = {client_task, model_task, limit_task}
            done, _pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            if client_task in done:
                finish_reason = client_task.result()
                if finish_reason in {"user_finished", "time_limit"}:
                    await _wait_for_model_drain(model_task, recorder)
            elif limit_task in done:
                finish_reason = limit_task.result()
                recorder.record_backend_event("time_limit_stop_request")
                await upstream.send(json.dumps({"type": "stop"}))
                await _wait_for_model_drain(model_task, recorder)
            else:
                finish_reason = model_task.result()
            for pending_task in tasks:
                if pending_task.done():
                    continue
                pending_task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    except WebSocketDisconnect:
        finish_reason = "disconnect"
    except Exception as error:
        expected_stop_close = (
            finish_reason in {"user_finished", "time_limit"}
            and _is_upstream_close_error(error)
        )
        if expected_stop_close:
            recorder.record_backend_event(
                "upstream_closed_after_stop",
                {"error_type": type(error).__name__, "message": str(error)[:2000]},
            )
        else:
            finish_reason = "crash"
            recorder.record_backend_event(
                "backend_error",
                {"error_type": type(error).__name__, "message": str(error)[:2000]},
            )
            print(
                f"[model-error] {conversation_id}: {type(error).__name__}: {error}",
                flush=True,
            )
            try:
                await client.send_json(
                    {"type": "error", "message": "Model connection failed"}
                )
            except Exception:
                pass
            _finish_conversation(
                conversation_id,
                "crash",
                crash=True,
                error=f"{type(error).__name__}: {error}",
            )
            return
    finally:
        if finish_reason != "crash":
            try:
                recorder.record_backend_event(
                    "conversation_finished", {"reason": finish_reason}
                )
                _finish_conversation(
                    conversation_id,
                    finish_reason,
                    timeout=finish_reason == "time_limit",
                    disconnect=finish_reason in {"disconnect", "upstream_closed"},
                )
                finished = store.find_conversation(conversation_id)[2]
                await client.send_json(
                    {
                        "type": "conversation_finished",
                        "reason": finish_reason,
                        "rating_allowed": response_was_observed(finished),
                        "turn_count": len(finished.get("turns", [])),
                        "status": finished.get("status"),
                        "response_record_status": response_record_status(finished),
                    }
                )
            except (KeyError, ValueError):
                pass
            except Exception:
                # A participant disconnect can prevent delivery of the receipt;
                # persistence must still succeed independently.
                pass
        try:
            await client.close()
        except Exception:
            pass


# Keep the participant UI and API on one origin without exposing backend/data.
@app.get("/", include_in_schema=False)
def participant_ui() -> HTMLResponse:
    asset_version = quote(str(SCENARIOS.get("version", "unknown")), safe="")
    page = (HUMAN_EVAL_DIR / "index.html").read_text(encoding="utf-8")
    page = page.replace('href="styles.css"', f'href="styles.css?v={asset_version}"')
    page = page.replace('src="app.js"', f'src="app.js?v={asset_version}"')
    return HTMLResponse(page, headers=NO_STORE_HEADERS)


@app.get("/app.js", include_in_schema=False)
def participant_javascript() -> FileResponse:
    return FileResponse(
        HUMAN_EVAL_DIR / "app.js",
        media_type="text/javascript",
        headers=NO_STORE_HEADERS,
    )


@app.get("/styles.css", include_in_schema=False)
def participant_styles() -> FileResponse:
    return FileResponse(
        HUMAN_EVAL_DIR / "styles.css",
        media_type="text/css",
        headers=NO_STORE_HEADERS,
    )


@app.get("/scenarios.json", include_in_schema=False)
def participant_scenarios() -> FileResponse:
    return FileResponse(
        HUMAN_EVAL_DIR / "scenarios.json",
        media_type="application/json",
        headers=NO_STORE_HEADERS,
    )
