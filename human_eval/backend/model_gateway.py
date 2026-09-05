"""Adapter for the two blinded arms exposed by interactive_paper/demo_duplex.py.

Both arms use the same deployed Voice service. The baseline disables the probe;
MiniCPM+ enables the aggressive escalation gate. This module also converts the
demo's WebSocket events into one analysis-friendly turn record.
"""

from __future__ import annotations

import base64
import asyncio
import json
import os
import time
import wave
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .core import (
    DEFAULT_TIER,
    MODEL_MINICPM,
    MODEL_MINICPM_PLUS,
    JsonSessionStore,
    new_id,
    utc_now,
)


DEFAULT_DEMO_URL = "https://rhe9527--gate-duplex-voice.modal.run"
DEFAULT_DEMO_TOKEN = "62dc5cd9"


class DemoModelGateway:
    """Build upstream URLs without exposing model assignment to the browser."""

    def __init__(self) -> None:
        self.base_url = os.getenv("MINICPM_DEMO_URL", DEFAULT_DEMO_URL).rstrip("/")
        self.token = os.getenv("MINICPM_DEMO_TOKEN", DEFAULT_DEMO_TOKEN)

    def settings(self, model: str) -> dict[str, Any]:
        if model == MODEL_MINICPM:
            return {"probe_on": False, "tier": DEFAULT_TIER}
        if model == MODEL_MINICPM_PLUS:
            return {"probe_on": True, "tier": DEFAULT_TIER}
        raise ValueError(f"Unknown model arm: {model}")

    def websocket_url(self, model: str) -> str:
        settings = self.settings(model)
        scheme = "wss" if self.base_url.startswith("https://") else "ws"
        host = self.base_url.split("://", 1)[-1]
        query = urlencode(
            {"tier": settings["tier"], "probe_on": int(settings["probe_on"])}
        )
        return f"{scheme}://{host}/{self.token}/ws?{query}"

    def say_url(self) -> str:
        return f"{self.base_url}/{self.token}/say"

    def ready_url(self) -> str:
        return f"{self.base_url}/{self.token}/ready"

    async def wait_until_ready(self, timeout_seconds: int = 180) -> dict[str, Any]:
        """Wake the Modal GPU and wait for the model, matching demo_app.py."""
        import httpx

        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=30) as client:
            while time.monotonic() < deadline:
                try:
                    response = await client.get(self.ready_url())
                    if response.is_success:
                        payload = response.json()
                        if payload.get("ready"):
                            return payload
                except (httpx.HTTPError, ValueError) as error:
                    last_error = error
                await asyncio.sleep(4)
        detail = f": {last_error}" if last_error else ""
        raise TimeoutError(f"Model readiness timed out after {timeout_seconds}s{detail}")


def _milliseconds(start: str | None, end: str | None) -> int | None:
    if not start or not end:
        return None
    from datetime import datetime

    return max(0, int((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() * 1000))


def _nonnegative_milliseconds(value: Any) -> float | int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _reported_or_derived_milliseconds(
    reported: Any, start: str | None, end: str | None
) -> float | int | None:
    measured = _nonnegative_milliseconds(reported)
    return measured if measured is not None else _milliseconds(start, end)


def _write_pcm_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".wav.tmp")
    with wave.open(str(temporary), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    os.replace(temporary, path)


async def transcribe_wav(path: Path) -> tuple[str | None, str, int | None]:
    """Optional post-hoc ASR for baseline turns that lack demo uplink text."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "not_configured", None

    import httpx

    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=90) as client:
        with path.open("rb") as audio:
            response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                data={"model": os.getenv("HUMAN_EVAL_ASR_MODEL", "gpt-transcribe")},
                files={"file": (path.name, audio, "audio/wav")},
            )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if response.is_error:
        return None, f"error:{response.status_code}", elapsed_ms
    payload = response.json()
    return payload.get("text"), "complete", elapsed_ms


def transcript_collection_settings() -> dict[str, bool]:
    """Expose configuration state without ever exposing the API key."""
    upstream_full_transcripts = "gate-duplex" in os.getenv(
        "MINICPM_DEMO_URL", DEFAULT_DEMO_URL
    )
    return {
        "posthoc_asr_configured": bool(os.getenv("OPENAI_API_KEY")),
        "upstream_full_transcripts": upstream_full_transcripts,
        "transcript_collection_configured": (
            upstream_full_transcripts or bool(os.getenv("OPENAI_API_KEY"))
        ),
        "transcripts_required": os.getenv(
            "HUMAN_EVAL_REQUIRE_TRANSCRIPTS", "0"
        ).lower()
        in {"1", "true", "yes"},
    }


class ConversationRecorder:
    """Accumulate one WebSocket conversation and persist each completed turn."""

    def __init__(
        self,
        store: JsonSessionStore,
        conversation_id: str,
        model: str,
        capability: str,
    ) -> None:
        self.store = store
        self.conversation_id = conversation_id
        self.model = model
        self.capability = capability
        self.event_log_path = store.event_log_path(conversation_id)
        self.user_pcm = bytearray()
        self.pre_speech_pcm = bytearray()
        self.capturing_speech = False
        self.model_pcm = bytearray()
        self.current_turn: dict[str, Any] | None = None
        self.connected_at = utc_now()
        self.first_input_at: str | None = None
        self.speech_started_at: str | None = None
        self.speech_ended_at: str | None = None
        self.first_model_audio_at: str | None = None
        self.vu_count = 0
        self.rms_sum = 0.0
        self.rms_max = 0.0
        self.vad_threshold_sum = 0.0
        self.silence_before_eot_s: float | None = None
        self.speech_detected = False

    def record_backend_event(
        self, event_type: str, payload: dict[str, Any] | None = None
    ) -> None:
        """Persist useful lifecycle/telemetry events without duplicating audio."""
        safe_payload = {
            key: value
            for key, value in (payload or {}).items()
            if key not in {"pcm", "user_pcm16"}
        }
        self.store.append_event_log(
            self.event_log_path,
            {
                "timestamp": utc_now(),
                "event": event_type,
                "conversation_id": self.conversation_id,
                "model": self.model,
                "capability": self.capability,
                "payload": safe_payload,
            },
        )

    def record_client_audio(self, chunk: bytes) -> None:
        if not self.first_input_at:
            self.first_input_at = utc_now()
        if self.capturing_speech:
            self.user_pcm.extend(chunk)
        else:
            self.pre_speech_pcm.extend(chunk)
            self.pre_speech_pcm = self.pre_speech_pcm[-16000:]  # 0.5 s at 16 kHz PCM16

    def record_client_event(self, payload: dict[str, Any]) -> None:
        if payload.get("type") == "eot":
            self.speech_ended_at = self.speech_ended_at or utc_now()

    def record_server_event(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        now = utc_now()
        if event_type in {"hello", "gate", "turn", "error", "bye"}:
            self.record_backend_event(f"upstream_{event_type}", payload)
        if event_type == "hello":
            self.store.mutate_conversation(
                self.conversation_id,
                lambda _task, conversation: self._mark_started(
                    conversation, payload, now
                ),
            )
        elif event_type == "speech":
            if payload.get("on"):
                self.speech_detected = True
                self.speech_started_at = self.speech_started_at or now
                self.user_pcm.extend(self.pre_speech_pcm)
                self.pre_speech_pcm = bytearray()
                self.capturing_speech = True
            else:
                self.speech_ended_at = now
                self.capturing_speech = False
                if self.current_turn:
                    self.current_turn["timestamps"]["user_speech_ended_at"] = now
        elif event_type == "vu":
            rms = payload.get("rms")
            threshold = payload.get("thr")
            if isinstance(rms, (int, float)):
                self.vu_count += 1
                self.rms_sum += float(rms)
                self.rms_max = max(self.rms_max, float(rms))
                if isinstance(threshold, (int, float)):
                    self.vad_threshold_sum += float(threshold)
            if isinstance(payload.get("sil"), (int, float)):
                self.silence_before_eot_s = float(payload["sil"])
            self.speech_detected = self.speech_detected or bool(payload.get("speech"))
        elif event_type == "eot":
            self.speech_ended_at = self.speech_ended_at or now
            turn = self._ensure_turn()
            turn["timestamps"]["user_speech_ended_at"] = self.speech_ended_at
        elif event_type == "gate":
            turn = self._ensure_turn()
            turn["timestamps"]["gate_decision_at"] = now
            score_series = turn["gate"].get("score_series", [])
            turn["gate"] = {
                "eligible": self.model == MODEL_MINICPM_PLUS,
                "escalated": bool(payload.get("fired")),
                "threshold_tier": DEFAULT_TIER if self.model == MODEL_MINICPM_PLUS else None,
                "threshold": payload.get("thr"),
                "score": payload.get("score"),
                "score_series": score_series,
                "eot_read_ms": payload.get("eot_read_ms", payload.get("ms")),
                "act_score": payload.get("act"),
                "is_information_request": payload.get("is_info"),
                "probe_on": payload.get("probe_on"),
            }
        elif event_type == "score":
            turn = self.current_turn or self._ensure_turn()
            turn["gate"].setdefault("score_series", []).append(payload.get("v"))
        elif event_type == "audio":
            if not self.first_model_audio_at:
                self.first_model_audio_at = now
            try:
                self.model_pcm.extend(base64.b64decode(payload.get("pcm", "")))
            except (ValueError, TypeError):
                self._ensure_turn()["anomalies"]["output_audio_anomaly"] = True

    async def finalize_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        turn = self._ensure_turn()
        response_completed_at = utc_now()
        encoded_user_pcm = payload.get("user_pcm16")
        if encoded_user_pcm:
            try:
                # Duplex v1 snapshots the committed utterance at the model.
                # It is more complete than the browser-side rolling buffer.
                self.user_pcm = bytearray(base64.b64decode(encoded_user_pcm))
                self.speech_detected = bool(self.user_pcm)
                self.speech_started_at = self.speech_started_at or self.first_input_at
            except (ValueError, TypeError):
                turn["anomalies"]["input_audio_anomaly"] = True
        turn["timestamps"].update(
            {
                "user_speech_started_at": self.speech_started_at,
                "user_speech_ended_at": self.speech_ended_at,
                "first_model_audio_at": self.first_model_audio_at,
                "response_completed_at": response_completed_at,
            }
        )
        turn["gate"].update(
            {
                "eligible": self.model == MODEL_MINICPM_PLUS,
                "escalated": bool(payload.get("fired", payload.get("mode") == "escalated")),
                "threshold": payload.get("threshold", turn["gate"].get("threshold")),
                "score": payload.get("eot_score", turn["gate"].get("score")),
                "score_series": payload.get("scores", turn["gate"].get("score_series", [])),
                "eot_read_ms": payload.get("eot_read_ms", turn["gate"].get("eot_read_ms")),
                "act_score": payload.get("act_score", turn["gate"].get("act_score")),
                "is_information_request": payload.get(
                    "is_info", turn["gate"].get("is_information_request")
                ),
                "probe_on": payload.get("probe_on", turn["gate"].get("probe_on")),
            }
        )

        user_audio_path = self.store.audio_path(self.conversation_id, turn["turn_id"], "user")
        model_audio_path = self.store.audio_path(self.conversation_id, turn["turn_id"], "model")
        if self.user_pcm:
            _write_pcm_wav(user_audio_path, bytes(self.user_pcm), 16000)
        if self.model_pcm:
            _write_pcm_wav(model_audio_path, bytes(self.model_pcm), 24000)

        user_transcript = (payload.get("uplink_text") or "").strip() or None
        transcript_source = "upstream_asr" if user_transcript else None
        upstream_asr_error = payload.get("asr_error")
        transcript_status = (
            "complete"
            if user_transcript
            else f"error:{upstream_asr_error}"
            if upstream_asr_error
            else "missing"
        )
        posthoc_asr_ms = None
        if not user_transcript and self.user_pcm and os.getenv("OPENAI_API_KEY"):
            try:
                user_transcript, transcript_status, posthoc_asr_ms = await transcribe_wav(
                    user_audio_path
                )
                user_transcript = (user_transcript or "").strip() or None
                if user_transcript:
                    transcript_source = "posthoc_asr"
            except Exception as error:  # preserve the turn even if optional ASR fails
                transcript_status = f"error:{type(error).__name__}"

        model_transcript = (
            payload.get("answer") or payload.get("relay") or ""
        ).strip()
        turn["user"] = {
            "audio_path": str(user_audio_path) if self.user_pcm else None,
            "audio_bytes": len(self.user_pcm),
            "sample_rate": 16000,
            "transcript": user_transcript,
            "transcript_status": transcript_status,
            "transcript_source": transcript_source,
        }
        model_response = {
            "audio_path": str(model_audio_path) if self.model_pcm else None,
            "audio_bytes": len(self.model_pcm),
            "sample_rate": 24000,
            "transcript": model_transcript,
        }
        if payload.get("expert_answer") is not None:
            model_response["expert_transcript"] = payload.get("expert_answer")
        turn["model_response"] = model_response

        audio_quality = {
            "speech_detected": self.speech_detected,
            "input_duration_ms": int(len(self.user_pcm) / 2 / 16000 * 1000),
            "output_duration_ms": int(len(self.model_pcm) / 2 / 24000 * 1000),
        }
        if self.vu_count:
            audio_quality.update(
                {
                    "input_rms_mean": round(self.rms_sum / self.vu_count, 6),
                    "input_rms_max": round(self.rms_max, 6),
                    "vad_threshold_mean": round(
                        self.vad_threshold_sum / self.vu_count, 6
                    ),
                }
            )
        if self.silence_before_eot_s is not None:
            audio_quality["silence_before_eot_s"] = self.silence_before_eot_s
        turn["audio_quality"] = audio_quality

        # The duplex runtime reports these three durations from one clock,
        # starting at its actual end-of-turn decision. Prefer them over
        # timestamps observed after network transport at the eval backend.
        latency_candidates = {
            "speech_end_to_gate": _reported_or_derived_milliseconds(
                payload.get("gate_latency_ms"),
                turn["timestamps"].get("user_speech_ended_at"),
                turn["timestamps"].get("gate_decision_at"),
            ),
            "speech_end_to_first_audio": _reported_or_derived_milliseconds(
                payload.get("first_audio_ms"),
                turn["timestamps"].get("user_speech_ended_at"),
                self.first_model_audio_at,
            ),
            "speech_end_to_response_complete": _reported_or_derived_milliseconds(
                payload.get("response_complete_ms"),
                turn["timestamps"].get("user_speech_ended_at"),
                response_completed_at,
            ),
            "model_first_audio_ms": payload.get("first_audio_ms"),
            "expert_ms": (
                int(payload["expert_latency_s"] * 1000)
                if isinstance(payload.get("expert_latency_s"), (int, float))
                else None
            ),
            "stall_ms": payload.get("stall_ms"),
            "relay_ms": payload.get("relay_ms"),
            "posthoc_asr_ms": posthoc_asr_ms,
        }
        turn["latency_ms"] = {
            key: value for key, value in latency_candidates.items() if value is not None
        }
        turn["anomalies"].update(
            {
                "empty_response": (
                    not bool(model_transcript.strip())
                    or (
                        turn["gate"].get("escalated")
                        and not bool(payload.get("expert_answer"))
                    )
                ),
                "input_audio_anomaly": len(self.user_pcm) < 3200,
                "output_audio_anomaly": len(self.model_pcm) < 2400,
                "interrupted": bool(payload.get("interrupted")),
                "missing_transcript": not bool(user_transcript),
            }
        )
        actual_action = "escalate" if turn["gate"].get("escalated") else "local"
        turn["routing_review"] = {
            "status": (
                "unreviewed" if self.model == MODEL_MINICPM_PLUS else "not_applicable"
            ),
            "expected_action": None,
            "actual_action": actual_action,
            "correct": None,
            "note": "",
            "reviewer": None,
            "reviewed_at": None,
        }
        turn["raw_model_metrics"] = {
            key: payload.get(key)
            for key in (
                "mode",
                "answer_ms",
                "audio_s",
                "speech_out_s",
                "asr_s",
                "total_ms",
                "gate_latency_ms",
                "first_audio_ms",
                "response_complete_ms",
                "protocol",
                "turn_index",
                "act_score",
                "is_info",
                "asr_error",
                "expert_error",
                "probe_on",
            )
            if payload.get(key) is not None
        }

        def append(_task: dict[str, Any], conversation: dict[str, Any]) -> None:
            turn["turn_index"] = len(conversation.get("turns", [])) + 1
            conversation["turns"] = [
                existing
                for existing in conversation.get("turns", [])
                if existing.get("turn_id") != turn["turn_id"]
            ]
            conversation["turns"].append(turn)

        self.store.mutate_conversation(self.conversation_id, append)
        completed = turn
        self._reset_turn_buffers()
        return completed

    def _new_turn(self) -> dict[str, Any]:
        return {
            "turn_id": new_id("turn"),
            "turn_index": 0,
            "timestamps": {
                "input_stream_started_at": self.first_input_at,
                "user_speech_started_at": self.speech_started_at,
                "user_speech_ended_at": self.speech_ended_at,
                "gate_decision_at": None,
                "first_model_audio_at": None,
                "response_completed_at": None,
            },
            "user": {},
            "model_response": {},
            "gate": {
                "eligible": self.model == MODEL_MINICPM_PLUS,
                "escalated": False,
                "threshold_tier": DEFAULT_TIER if self.model == MODEL_MINICPM_PLUS else None,
                "threshold": None,
                "score": None,
                "score_series": [],
                "eot_read_ms": None,
                "act_score": None,
                "is_information_request": None,
                "probe_on": self.model == MODEL_MINICPM_PLUS,
            },
            "latency_ms": {},
            "audio_quality": {},
            "anomalies": {
                "timeout": False,
                "crash": False,
                "empty_response": False,
                "input_audio_anomaly": False,
                "output_audio_anomaly": False,
                "disconnect": False,
                "interrupted": False,
                "missing_transcript": False,
            },
            "raw_model_metrics": {},
        }

    def _mark_started(
        self,
        conversation: dict[str, Any],
        payload: dict[str, Any],
        now: str,
    ) -> None:
        """Start, or safely retry, a conversation that produced no answer."""
        retrying_empty_attempt = (
            conversation.get("status") in {"failed", "abandoned"}
            and not conversation.get("turns")
            and not conversation.get("rating")
        )
        if retrying_empty_attempt:
            conversation.update(
                {
                    "started_at": now,
                    "ended_at": None,
                    "end_reason": None,
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
                    "retry_count": int(conversation.get("retry_count", 0)) + 1,
                }
            )
        conversation.update(
            {
                "status": "in_progress",
                "started_at": conversation.get("started_at") or now,
                "threshold_tier": (
                    payload.get("tier")
                    if self.model == MODEL_MINICPM_PLUS
                    else None
                ),
                "model_runtime": {
                    "protocol": payload.get("protocol", "duplex_v1"),
                    "mode": payload.get("mode"),
                    "event_log_path": str(self.event_log_path),
                    "threshold_tier": payload.get("tier"),
                    "threshold": payload.get("thr"),
                    "probe_on": payload.get("probe_on"),
                    "connected_at": self.connected_at,
                },
            }
        )

    def _ensure_turn(self) -> dict[str, Any]:
        if self.current_turn is None:
            self.current_turn = self._new_turn()
        return self.current_turn

    def _reset_turn_buffers(self) -> None:
        self.user_pcm = bytearray()
        self.pre_speech_pcm = bytearray()
        self.capturing_speech = False
        self.model_pcm = bytearray()
        self.current_turn = None
        self.first_input_at = None
        self.speech_started_at = None
        self.speech_ended_at = None
        self.first_model_audio_at = None
        self.vu_count = 0
        self.rms_sum = 0.0
        self.rms_max = 0.0
        self.vad_threshold_sum = 0.0
        self.silence_before_eot_s = None
        self.speech_detected = False
