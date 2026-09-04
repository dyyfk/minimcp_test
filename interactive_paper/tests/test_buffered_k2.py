import unittest

from interactive_paper.src.buffered_k2 import BufferedK2Gate


class BufferedK2GateTest(unittest.TestCase):
    def test_nothing_escapes_before_decision(self):
        gate = BufferedK2Gate()
        self.assertEqual(gate.offer("a"), [])
        self.assertEqual(gate.offer("b"), [])
        self.assertEqual(gate.offer("c"), [])
        self.assertTrue(gate.ready)

    def test_local_releases_once_in_order_then_streams(self):
        gate = BufferedK2Gate()
        gate.offer("a")
        gate.offer("b")
        gate.offer("c")
        self.assertEqual(gate.decide(escalate=False), ["a", "b", "c"])
        self.assertEqual(gate.offer("d"), ["d"])
        self.assertEqual(gate.offer("e", end_of_turn=True), ["e"])
        self.assertEqual(gate.state, "local_done")

    def test_expert_discards_buffer_and_local_continuation(self):
        gate = BufferedK2Gate()
        gate.offer("wrong-a")
        gate.offer("wrong-b")
        gate.offer("wrong-c")
        self.assertEqual(gate.decide(escalate=True), [])
        self.assertEqual(gate.pending_count, 0)
        self.assertEqual(gate.offer("wrong-c", end_of_turn=True), [])
        self.assertEqual(gate.state, "expert")

    def test_short_answer_is_released_instead_of_stranded(self):
        gate = BufferedK2Gate()
        self.assertEqual(gate.offer("short", end_of_turn=True), ["short"])
        self.assertEqual(gate.state, "local_done")
        self.assertEqual(gate.pending_count, 0)

        gate = BufferedK2Gate()
        gate.offer("short-a")
        self.assertEqual(
            gate.offer("short-b", end_of_turn=True), ["short-a", "short-b"]
        )
        self.assertEqual(gate.state, "local_done")

    def test_three_chunk_answer_remains_eligible_for_k2(self):
        gate = BufferedK2Gate()
        gate.offer("a")
        gate.offer("b")
        self.assertEqual(gate.offer("c", end_of_turn=True), [])
        self.assertTrue(gate.ready)
        self.assertEqual(gate.decide(escalate=False), ["a", "b", "c"])

    def test_interruption_erases_pending_and_stale_continuation(self):
        gate = BufferedK2Gate()
        gate.offer("stale")
        gate.interrupt()
        self.assertEqual(gate.pending_count, 0)
        self.assertEqual(gate.offer("also-stale"), [])
        self.assertEqual(gate.state, "interrupted")

    def test_early_decision_and_double_decision_fail_closed(self):
        gate = BufferedK2Gate()
        gate.offer("a")
        with self.assertRaises(RuntimeError):
            gate.decide(escalate=False)
        gate.offer("b")
        with self.assertRaises(RuntimeError):
            gate.decide(escalate=False)
        gate.offer("c")
        gate.decide(escalate=False)
        with self.assertRaises(RuntimeError):
            gate.decide(escalate=True)


if __name__ == "__main__":
    unittest.main()
