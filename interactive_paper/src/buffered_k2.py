"""Delivery buffer for an opt-in, post-onset k2 gate.

This module deliberately knows nothing about models, scores, thresholds, or
WebSockets.  It owns the safety-critical part of the P36 intervention: local
answer chunks cannot escape before the caller makes the k2 decision.
"""
from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class BufferedK2Gate:
    """Hold local chunks until a k2 decision, then release or discard them."""

    # The readpoint dump defines k2 as two chunks *after* onset, so the
    # delivery buffer contains onset, k1, and k2 (three emitted chunks).
    target_chunks: int = 3
    state: str = field(default="buffering", init=False)
    _pending: List[Any] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if self.target_chunks < 1:
            raise ValueError("target_chunks must be positive")

    @property
    def ready(self):
        return self.state == "buffering" and len(self._pending) >= self.target_chunks

    @property
    def pending_count(self):
        return len(self._pending)

    def offer(self, chunk, *, end_of_turn=False):
        """Accept one generated local chunk and return chunks safe to emit.

        An answer that ends before k2 is released locally.  If the third
        chunk itself ends the answer, k2 is available and the caller must
        still decide; this method therefore keeps that chunk buffered.
        """
        if self.state == "buffering":
            self._pending.append(chunk)
            if end_of_turn and len(self._pending) < self.target_chunks:
                self.state = "local_done"
                return self._drain()
            return []
        if self.state == "local":
            if end_of_turn:
                self.state = "local_done"
            return [chunk]
        # Expert, interrupted, and completed turns never leak local output.
        return []

    def decide(self, *, escalate):
        """Resolve the k2 decision and return newly releasable local chunks."""
        if self.state != "buffering":
            raise RuntimeError(f"cannot decide while state={self.state}")
        if not self.ready:
            raise RuntimeError("k2 decision requested before enough chunks")
        if escalate:
            self._pending.clear()
            self.state = "expert"
            return []
        self.state = "local"
        return self._drain()

    def interrupt(self):
        """Cancel the pending turn so buffered content can never arrive late."""
        self._pending.clear()
        self.state = "interrupted"

    def _drain(self):
        out, self._pending = self._pending, []
        return out
