from __future__ import annotations

import asyncio
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import human_eval.backend.app as app_module
from human_eval.backend.core import JsonSessionStore


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        app_module.store = JsonSessionStore(self.temporary.name)
        self.client = TestClient(app_module.app)

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def test_complete_persistence_flow(self) -> None:
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Voice Assistant Experience Study", page.text)
        self.assertEqual(self.client.get("/backend/data/sessions/example.json").status_code, 404)

        response = self.client.post(
            "/api/study-sessions", json={"user_id": "participant-1"}
        )
        self.assertEqual(response.status_code, 200)
        assignment = response.json()
        self.assertNotIn('"model"', response.text)

        session_id = assignment["session_id"]
        resumed = self.client.post(
            "/api/study-sessions", json={"user_id": "participant-1"}
        )
        self.assertEqual(resumed.json()["session_id"], session_id)
        private = app_module.store.get(session_id)
        self.assertEqual(private["user_id"], "participant-1")
        self.assertEqual(
            {conversation["model"] for task in private["tasks"] for conversation in task["conversations"]},
            {"minicpm", "minicpm_plus"},
        )

        for task in assignment["tasks"]:
            for conversation in task["conversations"]:
                conversation_id = conversation["conversation_id"]
                for index in range(conversation["target_turns"]):
                    app_module.store.mutate_conversation(
                        conversation_id,
                        lambda _task, target, index=index: target["turns"].append(
                            {
                                "turn_id": f"turn_{conversation_id}_{index}",
                                "gate": {},
                                "user": {"transcript": f"Prompt {index + 1}"},
                                "anomalies": {},
                                "routing_review": {},
                            }
                        ),
                    )
                finalized = self.client.post(
                    f"/api/conversations/{conversation_id}/finalize",
                    json={"end_reason": "user_finished"},
                )
                self.assertEqual(finalized.status_code, 200)
                interaction = app_module.store.find_conversation(conversation_id)[2]
                self.assertEqual(interaction["status"], "interaction_completed")
                self.assertEqual(interaction["evaluation_status"], "pending")
                rating = self.client.put(
                    f"/api/conversations/{conversation_id}/rating",
                    json={"metrics": {"overall": 4}, "feedback": "Clear response"},
                )
                self.assertEqual(rating.status_code, 200)
                evaluated = app_module.store.find_conversation(conversation_id)[2]
                self.assertEqual(evaluated["evaluation_status"], "completed")
            comparison = self.client.put(
                f"/api/tasks/{task['task_id']}/comparison",
                json={
                    "preference": "first",
                    "reasons": ["More accurate"],
                    "feedback": "",
                },
            )
            self.assertEqual(comparison.status_code, 200)

        completed = self.client.post(f"/api/study-sessions/{session_id}/complete")
        self.assertEqual(completed.status_code, 200)
        self.assertTrue(completed.json()["completion_code"].startswith("DONE-"))

        completed_resume = self.client.post(
            "/api/study-sessions", json={"user_id": "participant-1"}
        )
        self.assertEqual(completed_resume.json()["session_id"], session_id)
        self.assertEqual(completed_resume.json()["status"], "completed")

        forced = self.client.post(
            "/api/study-sessions",
            json={"user_id": "participant-1", "force_new": True},
        )
        self.assertEqual(forced.status_code, 200)
        self.assertNotEqual(forced.json()["session_id"], session_id)
        self.assertEqual(forced.json()["user_id"], "participant-1")

        exported = self.client.get("/api/admin/export.jsonl")
        self.assertEqual(exported.status_code, 200)
        self.assertIn('"participant-1"', exported.text)

        conversation_export = self.client.get(
            "/api/admin/export/conversations.jsonl"
        )
        self.assertEqual(conversation_export.status_code, 200)
        self.assertEqual(len(conversation_export.text.strip().splitlines()), 8)
        self.assertIn('"rating_metrics"', conversation_export.text)

    def test_interaction_lifecycle_and_quality_review(self) -> None:
        assignment = self.client.post(
            "/api/study-sessions", json={"user_id": "completion-gate"}
        ).json()
        conversation = assignment["tasks"][0]["conversations"][0]
        conversation_id = conversation["conversation_id"]

        early_rating = self.client.put(
            f"/api/conversations/{conversation_id}/rating",
            json={"metrics": {"overall": 4}, "feedback": ""},
        )
        self.assertEqual(early_rating.status_code, 409)
        early_finish = self.client.post(
            f"/api/conversations/{conversation_id}/finalize",
            json={"end_reason": "user_finished"},
        )
        self.assertEqual(early_finish.status_code, 200)
        private = app_module.store.find_conversation(conversation_id)[2]
        self.assertEqual(private["status"], "interaction_completed")
        self.assertEqual(private["evaluation_status"], "pending")
        self.assertEqual(private["quality_review"]["status"], "needs_review")
        self.assertIn(
            "fewer_than_target_turns",
            private["quality_review"]["automatic_flags"],
        )

        reviewed = self.client.put(
            f"/api/admin/conversations/{conversation_id}/quality-review",
            json={
                "status": "valid",
                "reviewer": "researcher-1",
                "reason": "followed_task",
                "note": "All required prompts were used.",
            },
        )
        self.assertEqual(reviewed.status_code, 200)
        self.assertEqual(reviewed.json()["status"], "valid")

    def test_expired_session_is_not_resumed(self) -> None:
        first = self.client.post(
            "/api/study-sessions", json={"user_id": "returning-user"}
        ).json()
        private = app_module.store.get(first["session_id"])
        private["assignment"]["reservation_expires_at"] = (
            "2000-01-01T00:00:00+00:00"
        )
        app_module.store.save(private, renew_reservation=False)

        second = self.client.post(
            "/api/study-sessions", json={"user_id": "returning-user"}
        ).json()
        self.assertNotEqual(second["session_id"], first["session_id"])
        expired = app_module.store.get(first["session_id"])
        self.assertEqual(expired["status"], "expired")

    def test_participant_stream_events_hide_gate_telemetry(self) -> None:
        self.assertEqual(
            app_module._participant_model_event(
                {"type": "hello", "probe_on": True, "tier": "balanced", "thr": 0.62}
            ),
            {"type": "ready"},
        )
        self.assertEqual(
            app_module._participant_model_event(
                {"type": "turn", "fired": True, "eot_score": 0.9, "answer": "private"}
            ),
            {"type": "turn"},
        )
        self.assertIsNone(
            app_module._participant_model_event({"type": "score", "v": 0.9})
        )

    def test_temporary_debug_flag_exposes_model_config(self) -> None:
        with patch.dict(
            "os.environ", {"HUMAN_EVAL_DEBUG_MODEL_LOGS": "1"}, clear=False
        ):
            event = app_module._participant_model_event(
                {
                    "type": "hello",
                    "protocol": "duplex_v1",
                    "probe_on": True,
                    "tier": "aggressive",
                    "mode": "native duplex",
                }
            )
        self.assertEqual(event["debug"]["model"], "minicpm_plus")
        self.assertEqual(event["debug"]["tier"], "aggressive")

    def test_concurrent_readiness_requests_share_one_warmup(self) -> None:
        class FakeGateway:
            calls = 0

            async def wait_until_ready(self):
                self.calls += 1
                await asyncio.sleep(0.01)
                return {"ready": True, "busy": False, "load_s": 1.0}

        original_gateway = app_module.model_gateway
        fake_gateway = FakeGateway()
        app_module.model_gateway = fake_gateway
        app_module.model_warm_task = None
        try:
            async def run():
                return await asyncio.gather(
                    app_module.ensure_model_ready(),
                    app_module.ensure_model_ready(),
                )

            results = asyncio.run(run())
            self.assertTrue(all(result["ready"] for result in results))
            self.assertEqual(fake_gateway.calls, 1)
        finally:
            app_module.model_gateway = original_gateway
            app_module.model_warm_task = None

    def test_required_transcripts_block_legacy_upstream_without_asr(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "HUMAN_EVAL_REQUIRE_TRANSCRIPTS": "1",
                "MINICPM_DEMO_URL": "https://example.invalid/legacy",
            },
            clear=False,
        ):
            with patch.dict("os.environ", {}, clear=False):
                import os

                old_key = os.environ.pop("OPENAI_API_KEY", None)
                try:
                    response = self.client.get("/api/model/readiness")
                finally:
                    if old_key is not None:
                        os.environ["OPENAI_API_KEY"] = old_key
        self.assertEqual(response.status_code, 503)
        self.assertIn("post-hoc ASR", response.json()["detail"])

    def test_public_admin_routes_require_bearer_token(self) -> None:
        with patch.dict(
            "os.environ",
            {"HUMAN_EVAL_PUBLIC": "1", "HUMAN_EVAL_ADMIN_TOKEN": "test-secret"},
            clear=False,
        ):
            denied = self.client.get("/api/admin/export.jsonl")
            allowed = self.client.get(
                "/api/admin/export.jsonl",
                headers={"Authorization": "Bearer test-secret"},
            )
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(allowed.status_code, 200)

    def test_routing_review_uses_turn_level_expected_action(self) -> None:
        assignment = self.client.post(
            "/api/study-sessions", json={"user_id": "reviewer-test"}
        ).json()
        private = app_module.store.get(assignment["session_id"])
        plus = next(
            conversation
            for task in private["tasks"]
            for conversation in task["conversations"]
            if conversation["model"] == "minicpm_plus"
        )
        turn_id = "turn_review_test"

        def add_turn(_task, conversation):
            conversation["turns"].append(
                {
                    "turn_id": turn_id,
                    "gate": {"escalated": True},
                    "user": {"transcript": "Stop."},
                    "anomalies": {},
                }
            )

        app_module.store.mutate_conversation(plus["conversation_id"], add_turn)
        response = self.client.put(
            f"/api/admin/turns/{turn_id}/routing-review",
            json={
                "expected_action": "local",
                "reviewer": "researcher-1",
                "note": "Stop command should not use the expert.",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["correct"])
        saved = app_module.store.get(assignment["session_id"])
        self.assertEqual(saved["summary"]["routing_incorrect_turn_count"], 1)


if __name__ == "__main__":
    unittest.main()
