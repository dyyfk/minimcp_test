from __future__ import annotations

import asyncio
import base64
import json
import random
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from human_eval.backend.core import (
    JsonSessionStore,
    analysis_rows,
    create_assignment,
    public_session,
    recompute_summary,
)
from human_eval.backend.model_gateway import ConversationRecorder, DemoModelGateway


SCENARIOS_PATH = Path(__file__).resolve().parents[2] / "scenarios.json"


class AssignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))

    def test_participant_config_uses_four_focused_ratings(self) -> None:
        self.assertEqual(
            [question["id"] for question in self.scenarios["ratingQuestions"]],
            [
                "correctness",
                "helpfulness",
                "context_consistency",
                "conversation_naturalness",
            ],
        )

    def test_32_participant_block_is_balanced(self) -> None:
        rng = random.Random(7)
        sessions = []
        for index in range(32):
            sessions.append(
                create_assignment(self.scenarios, sessions, f"user-{index}", rng=rng)
            )

        pair_counts = Counter(row["assignment"]["pair_cell"] for row in sessions)
        self.assertEqual(pair_counts, {"s1_s2": 8, "s1_s3": 8, "s2_s3": 16})

        sequences = defaultdict(Counter)
        for session in sessions:
            for task in session["tasks"]:
                sequences[task["capability"]][task["sequence_cell"]] += 1
                self.assertEqual(
                    {item["model"] for item in task["conversations"]},
                    {"minicpm", "minicpm_plus"},
                )
        for counts in sequences.values():
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        capability_codes = {
            task["capability_key"]: task["capability"]
            for session in sessions
            for task in session["tasks"]
        }
        self.assertEqual(capability_codes["simple_guardrail"], "S1")
        self.assertEqual(capability_codes["realtime"], "S2")
        self.assertEqual(capability_codes["context_reasoning"], "S3")

    def test_arbitrary_participant_count_stays_close_to_target_ratio(self) -> None:
        rng = random.Random(11)
        sessions = []
        for index in range(11):
            sessions.append(
                create_assignment(self.scenarios, sessions, f"user-{index}", rng=rng)
            )
        counts = Counter(row["assignment"]["pair_cell"] for row in sessions)
        normalized = [counts["s1_s2"], counts["s1_s3"], counts["s2_s3"] / 2]
        self.assertLessEqual(max(normalized) - min(normalized), 1)

    def test_expired_or_failed_sessions_do_not_hold_quota(self) -> None:
        stale = create_assignment(self.scenarios, [], "stale", rng=random.Random(2))
        stale["assignment"]["reservation_expires_at"] = "2000-01-01T00:00:00+00:00"
        failed = create_assignment(self.scenarios, [], "failed", rng=random.Random(3))
        failed["status"] = "failed"

        expected = create_assignment(self.scenarios, [], "next", rng=random.Random(9))
        actual = create_assignment(
            self.scenarios, [stale, failed], "next", rng=random.Random(9)
        )
        self.assertEqual(actual["assignment"]["pair_cell"], expected["assignment"]["pair_cell"])
        self.assertEqual(
            [(task["capability"], task["sequence_cell"]) for task in actual["tasks"]],
            [(task["capability"], task["sequence_cell"]) for task in expected["tasks"]],
        )

    def test_public_assignment_hides_model_fields(self) -> None:
        session = create_assignment(self.scenarios, [], "user-1", rng=random.Random(3))
        public = public_session(session)
        serialized = json.dumps(public)
        for hidden in ("model", "probe_on", "threshold_tier", "sequence_cell"):
            self.assertNotIn(f'"{hidden}"', serialized)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))

    def test_session_and_summary_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            saves = []
            store = JsonSessionStore(directory, after_save=lambda: saves.append(True))
            session = create_assignment(self.scenarios, [], "user-1", rng=random.Random(1))
            conversation = session["tasks"][0]["conversations"][0]
            conversation["model"] = "minicpm_plus"
            conversation["turns"].append(
                {
                    "turn_id": "turn_test",
                    "gate": {"escalated": True},
                    "anomalies": {"empty_response": False},
                }
            )
            recompute_summary(session)
            store.create(session)
            loaded = store.get(session["session_id"])
            self.assertEqual(loaded["user_id"], "user-1")
            self.assertEqual(loaded["summary"]["turn_count"], 1)
            self.assertEqual(loaded["summary"]["escalation_count"], 1)
            self.assertNotIn(
                "inappropriate_escalation_conversation_count", loaded["summary"]
            )
            self.assertEqual(len(saves), 1)

    def test_expired_reservation_is_persisted_as_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JsonSessionStore(directory)
            session = create_assignment(
                self.scenarios, [], "expired-user", rng=random.Random(4)
            )
            conversation = session["tasks"][0]["conversations"][0]
            session["status"] = "in_progress"
            session["started_at"] = "1999-12-31T23:00:00+00:00"
            session["assignment"]["reservation_expires_at"] = (
                "2000-01-01T00:00:00+00:00"
            )
            conversation["status"] = "in_progress"
            conversation["started_at"] = "1999-12-31T23:30:00+00:00"
            store.save(session, renew_reservation=False)

            self.assertEqual(store.expire_stale_sessions(), [session["session_id"]])
            expired = store.get(session["session_id"])
            self.assertEqual(expired["status"], "expired")
            self.assertEqual(expired["expiration_reason"], "reservation_timeout")
            saved_conversation = expired["tasks"][0]["conversations"][0]
            self.assertEqual(saved_conversation["status"], "abandoned")
            self.assertEqual(saved_conversation["end_reason"], "reservation_expired")
            self.assertEqual(
                saved_conversation["evaluation_status"], "not_submitted"
            )
            self.assertIn(
                "reservation_expired",
                saved_conversation["quality_review"]["automatic_flags"],
            )
            row = analysis_rows([expired], "sessions")[0]
            self.assertEqual(row["reservation_status"], "expired")
            self.assertFalse(row["reservation_active"])

    def test_analysis_completion_requires_a_submitted_rating(self) -> None:
        session = create_assignment(
            self.scenarios, [], "rated-user", rng=random.Random(5)
        )
        conversation = session["tasks"][0]["conversations"][0]
        conversation["status"] = "interaction_completed"
        conversation["evaluation_status"] = "pending"
        recompute_summary(session)
        self.assertEqual(
            session["summary"]["interaction_ended_conversation_count"], 1
        )
        self.assertEqual(session["summary"]["completed_conversation_count"], 0)

        conversation["rating"] = {"metrics": {"overall": 4}, "submitted_at": "now"}
        conversation["evaluation_status"] = "completed"
        recompute_summary(session)
        self.assertEqual(
            session["summary"]["evaluation_completed_conversation_count"], 1
        )
        self.assertEqual(session["summary"]["completed_conversation_count"], 1)
        self.assertEqual(
            session["summary"]["analysis_complete_conversation_count"], 1
        )
        self.assertEqual(
            session["summary"]["telemetry_complete_rated_conversation_count"],
            0,
        )

        conversation["turns"] = [
            {
                "turn_id": "turn_rated",
                "user": {"transcript": "Question"},
                "gate": {},
                "anomalies": {},
                "routing_review": {},
            }
        ]
        recompute_summary(session)
        self.assertEqual(session["summary"]["completed_conversation_count"], 1)
        self.assertEqual(
            session["summary"]["analysis_complete_conversation_count"], 1
        )
        self.assertEqual(
            session["summary"]["telemetry_complete_rated_conversation_count"],
            1,
        )

    def test_task_rating_analysis_is_independent_from_turn_telemetry(self) -> None:
        session = create_assignment(
            self.scenarios, [], "rating-first", rng=random.Random(8)
        )
        task = session["tasks"][0]
        for conversation in task["conversations"]:
            conversation["status"] = "interaction_completed"
            conversation["evaluation_status"] = "completed"
            conversation["client_observation"] = {
                "received_model_audio": True,
                "completed_turn_count": 1,
                "model_audio_ms": 2000,
                "reported_at": "2026-09-04T00:00:00+00:00",
            }
            conversation["rating"] = {
                "metrics": {"overall": 4},
                "submitted_at": "2026-09-04T00:01:00+00:00",
                "response_record_status": "client_observed",
                "turn_count_at_submission": 0,
            }
        task["comparison"] = {
            "preference": "first",
            "reasons": ["More accurate"],
            "feedback": "",
            "submitted_at": "2026-09-04T00:02:00+00:00",
        }
        recompute_summary(session)

        task_row = analysis_rows([session], "tasks")[0]
        self.assertTrue(task_row["analysis_complete"])
        self.assertFalse(task_row["telemetry_complete"])
        self.assertEqual(task_row["analyzable_rating_count"], 2)
        self.assertEqual(task_row["recorded_rating_count"], 0)
        self.assertIn(task_row["preferred_model"], {"minicpm", "minicpm_plus"})

    def test_completed_task_is_analysis_ready_even_if_session_expires(self) -> None:
        session = create_assignment(
            self.scenarios, [], "partial-user", rng=random.Random(6)
        )
        completed_task = session["tasks"][0]
        for index, conversation in enumerate(completed_task["conversations"], 1):
            conversation["status"] = "interaction_completed"
            conversation["evaluation_status"] = "completed"
            conversation["quality_review"]["status"] = "valid"
            conversation["turns"] = [
                {
                    "turn_id": f"turn_partial_{index}",
                    "user": {"transcript": f"Question {index}"},
                    "gate": {},
                    "anomalies": {},
                    "routing_review": {},
                }
            ]
            conversation["rating"] = {
                "metrics": {"overall": 4},
                "submitted_at": "2026-09-04T00:00:00+00:00",
            }
        completed_task["comparison"] = {
            "preference": "first",
            "reasons": ["More accurate"],
            "feedback": "",
            "submitted_at": "2026-09-04T00:01:00+00:00",
        }
        session["status"] = "expired"
        recompute_summary(session)

        self.assertEqual(session["summary"]["analysis_complete_task_count"], 1)
        self.assertEqual(session["summary"]["ratings_complete_task_count"], 1)
        task_rows = analysis_rows([session], "tasks")
        self.assertTrue(task_rows[0]["analysis_complete"])
        self.assertTrue(task_rows[0]["quality_review_complete"])
        self.assertTrue(task_rows[0]["quality_valid"])
        self.assertFalse(task_rows[1]["analysis_complete"])
        conversation_rows = analysis_rows([session], "conversations")
        completed_rows = [
            row
            for row in conversation_rows
            if row["task_id"] == completed_task["task_id"]
        ]
        self.assertTrue(all(row["analysis_complete"] for row in completed_rows))
        self.assertTrue(
            all(row["task_analysis_complete"] for row in completed_rows)
        )

    def test_whitespace_transcript_is_missing(self) -> None:
        session = create_assignment(
            self.scenarios, [], "blank-transcript", rng=random.Random(7)
        )
        conversation = session["tasks"][0]["conversations"][0]
        conversation["turns"] = [
            {
                "turn_id": "turn_blank",
                "user": {"transcript": "\n", "transcript_status": "complete"},
                "gate": {},
                "anomalies": {"missing_transcript": False},
                "routing_review": {},
            }
        ]
        recompute_summary(session)

        turn = conversation["turns"][0]
        self.assertIsNone(turn["user"]["transcript"])
        self.assertEqual(turn["user"]["transcript_status"], "missing")
        self.assertTrue(turn["anomalies"]["missing_transcript"])
        self.assertEqual(session["summary"]["missing_user_transcript_count"], 1)


class ModelGatewayTests(unittest.TestCase):
    def test_model_arms_map_to_demo_probe_switch(self) -> None:
        gateway = DemoModelGateway()
        self.assertIn("gate-duplex-voice", gateway.base_url)
        self.assertIn("probe_on=0", gateway.websocket_url("minicpm"))
        self.assertIn("probe_on=1", gateway.websocket_url("minicpm_plus"))
        self.assertIn("tier=aggressive", gateway.websocket_url("minicpm_plus"))

    def test_turn_telemetry_and_audio_are_persisted(self) -> None:
        scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            store = JsonSessionStore(directory)
            session = create_assignment(scenarios, [], "user-1", rng=random.Random(2))
            task = session["tasks"][0]
            conversation = next(
                item for item in task["conversations"] if item["model"] == "minicpm_plus"
            )
            store.create(session)
            recorder = ConversationRecorder(
                store,
                conversation["conversation_id"],
                conversation["model"],
                task["capability"],
            )
            recorder.record_server_event(
                {
                    "type": "hello",
                    "protocol": "duplex_v1",
                    "tier": "aggressive",
                    "thr": 0.62,
                    "probe_on": True,
                }
            )
            recorder.record_client_audio(b"\x00\x00" * 2000)
            recorder.record_server_event({"type": "score", "v": 0.41})
            recorder.record_server_event(
                {
                    "type": "gate",
                    "score": 0.81,
                    "thr": 0.62,
                    "fired": True,
                    "eot_read_ms": 12.5,
                    "act": 0.73,
                    "is_info": True,
                    "probe_on": True,
                }
            )
            recorder.record_server_event(
                {
                    "type": "audio",
                    "pcm": base64.b64encode(b"\x00\x00" * 1600).decode(),
                }
            )
            turn_payload = {
                        "type": "turn",
                        "protocol": "duplex_v1",
                        "turn_index": 1,
                        "fired": True,
                        "mode": "escalated",
                        "eot_score": 0.81,
                        "threshold": 0.62,
                        "scores": [0.41],
                        "eot_read_ms": 12.5,
                        "act_score": 0.73,
                        "is_info": True,
                        "probe_on": True,
                        "user_pcm16": base64.b64encode(
                            b"\x01\x00" * 5000
                        ).decode(),
                        "uplink_text": "Test question",
                        "answer": "Test answer",
                        "expert_answer": "Verified answer",
                        "gate_latency_ms": 18,
                        "first_audio_ms": 300,
                        "response_complete_ms": 900,
                        "expert_latency_s": 1.2,
                    }
            recorder.record_server_event(turn_payload)
            asyncio.run(recorder.finalize_turn(turn_payload))
            persisted = store.get(session["session_id"])
            saved_conversation = next(
                item
                for saved_task in persisted["tasks"]
                for item in saved_task["conversations"]
                if item["conversation_id"] == conversation["conversation_id"]
            )
            turn = saved_conversation["turns"][0]
            self.assertEqual(saved_conversation["threshold_tier"], "aggressive")
            self.assertTrue(turn["gate"]["escalated"])
            self.assertEqual(turn["gate"]["score"], 0.81)
            self.assertEqual(turn["gate"]["act_score"], 0.73)
            self.assertTrue(turn["gate"]["is_information_request"])
            self.assertEqual(turn["user"]["transcript"], "Test question")
            self.assertEqual(turn["user"]["transcript_source"], "upstream_asr")
            self.assertEqual(turn["routing_review"]["status"], "unreviewed")
            self.assertEqual(turn["routing_review"]["actual_action"], "escalate")
            self.assertIsNone(turn["routing_review"]["correct"])
            self.assertIsNone(turn["timestamps"]["user_speech_ended_at"])
            self.assertEqual(turn["latency_ms"]["speech_end_to_gate"], 18)
            self.assertEqual(turn["latency_ms"]["speech_end_to_first_audio"], 300)
            self.assertEqual(
                turn["latency_ms"]["speech_end_to_response_complete"], 900
            )
            self.assertEqual(turn["gate"]["eot_read_ms"], 12.5)
            self.assertTrue(turn["audio_quality"]["speech_detected"])
            self.assertEqual(turn["user"]["audio_bytes"], 10000)
            self.assertTrue(Path(turn["user"]["audio_path"]).exists())
            self.assertTrue(Path(turn["model_response"]["audio_path"]).exists())
            log_files = list((Path(directory) / "logs").glob("*.jsonl"))
            self.assertEqual(len(log_files), 1)
            self.assertRegex(
                log_files[0].name,
                rf"^\d{{8}}T\d{{12}}Z_{conversation['conversation_id']}\.jsonl$",
            )
            log_text = log_files[0].read_text(encoding="utf-8")
            self.assertIn('"event": "upstream_hello"', log_text)
            self.assertIn('"event": "upstream_gate"', log_text)
            self.assertIn('"event": "upstream_turn"', log_text)
            self.assertNotIn("user_pcm16", log_text)

            turn_rows = analysis_rows([persisted], "turns")
            self.assertEqual(len(turn_rows), 1)
            self.assertEqual(turn_rows[0]["model"], "minicpm_plus")
            self.assertEqual(turn_rows[0]["user_transcript"], "Test question")
            self.assertEqual(turn_rows[0]["user_transcript_source"], "upstream_asr")
            self.assertEqual(turn_rows[0]["routing_review_status"], "unreviewed")
            self.assertEqual(turn_rows[0]["model_act_score"], 0.73)
            self.assertFalse(turn_rows[0]["speech_end_estimated"])
            self.assertEqual(turn_rows[0]["latency_speech_end_to_gate_ms"], 18)
            self.assertEqual(
                turn_rows[0]["latency_speech_end_to_first_audio_ms"], 300
            )

            conversation_rows = analysis_rows([persisted], "conversations")
            conversation_row = next(
                row
                for row in conversation_rows
                if row["conversation_id"] == conversation["conversation_id"]
            )
            self.assertEqual(conversation_row["escalation_rate"], 1)
            self.assertEqual(conversation_row["latency_gate_median_ms"], 18)
            self.assertEqual(conversation_row["latency_first_audio_median_ms"], 300)
            self.assertEqual(
                conversation_row["latency_response_complete_median_ms"], 900
            )

    def test_eot_read_cost_is_not_exported_as_gate_latency(self) -> None:
        scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
        session = create_assignment(
            scenarios, [], "read-cost-only", rng=random.Random(13)
        )
        conversation = session["tasks"][0]["conversations"][0]
        conversation["turns"] = [
            {
                "turn_id": "turn_read_cost_only",
                "timestamps": {
                    "user_speech_ended_at": None,
                    "gate_decision_at": "2026-09-04T00:00:00+00:00",
                },
                "latency_ms": {},
                "gate": {"eot_read_ms": 2.4},
                "user": {"transcript": "Question"},
                "model_response": {"transcript": "Answer"},
                "audio_quality": {},
                "anomalies": {},
                "routing_review": {},
                "raw_model_metrics": {},
            }
        ]

        turn_row = analysis_rows([session], "turns")[0]
        self.assertEqual(turn_row["eot_read_ms"], 2.4)
        self.assertNotIn("latency_speech_end_to_gate_ms", turn_row)

    def test_legacy_zero_gate_latency_is_exported_as_missing(self) -> None:
        scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
        session = create_assignment(
            scenarios, [], "legacy-latency", rng=random.Random(12)
        )
        conversation = session["tasks"][0]["conversations"][0]
        conversation["turns"] = [
            {
                "turn_id": "turn_legacy_zero",
                "timestamps": {
                    "user_speech_ended_at": None,
                    "gate_decision_at": "2026-09-04T00:00:00+00:00",
                },
                "latency_ms": {"speech_end_to_gate": 0},
                "gate": {"eot_read_ms": None},
                "user": {"transcript": "Question"},
                "model_response": {"transcript": "Answer"},
                "audio_quality": {},
                "anomalies": {},
                "routing_review": {},
                "raw_model_metrics": {},
            }
        ]

        turn_row = analysis_rows([session], "turns")[0]
        self.assertNotIn("latency_speech_end_to_gate_ms", turn_row)
        conversation_row = next(
            row
            for row in analysis_rows([session], "conversations")
            if row["conversation_id"] == conversation["conversation_id"]
        )
        self.assertIsNone(conversation_row["latency_gate_median_ms"])


if __name__ == "__main__":
    unittest.main()
