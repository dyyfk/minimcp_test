"""Causal streaming readout for the NVIDIA VoiceChat probe.

The offline calibration defines speaking onset as the first run of three
non-PAD agent tokens and reads an eight-frame L30 window beginning at that
onset.  This module reproduces that state machine online.  It deliberately
waits for the complete eight-frame window before returning a decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProbeDecision:
    score: float
    act_score: float
    is_information_request: bool
    fired: bool
    tier: str
    onset_frame: int
    read_frame: int


class NvidiaDuplexProbe:
    """Accumulate one L30 hidden vector and one agent token per 80 ms frame."""

    def __init__(self, artifact: dict[str, Any], tier: str = "balanced",
                 pad_id: int = 12):
        if tier not in artifact["fail"]["thresholds"]:
            raise ValueError(f"unknown tier: {tier}")
        self.artifact = artifact
        self.tier = tier
        self.pad_id = int(pad_id)
        self.window = int(artifact.get("k_eot", 8))
        self.fail_w = np.asarray(artifact["fail"]["w"], dtype=np.float32)
        self.fail_b = float(artifact["fail"]["b"])
        self.act_w = np.asarray(artifact["act"]["w"], dtype=np.float32)
        self.act_b = float(artifact["act"]["b"])
        self.act_tau = float(artifact["act"]["tau"])
        self.threshold = float(artifact["fail"]["thresholds"][tier])
        self.reset()

    def reset(self) -> None:
        self.frame = -1
        self._listen_sum: np.ndarray | None = None
        self._listen_count = 0
        self._candidate: list[tuple[int, np.ndarray]] = []
        self._onset: list[np.ndarray] = []
        self._onset_frame: int | None = None
        self._decision: ProbeDecision | None = None

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = float(np.clip(value, -40.0, 40.0))
        return float(1.0 / (1.0 + np.exp(-value)))

    def _add_listen(self, hidden: np.ndarray) -> None:
        if self._listen_sum is None:
            self._listen_sum = hidden.astype(np.float64, copy=True)
        else:
            self._listen_sum += hidden
        self._listen_count += 1

    def observe(self, hidden: np.ndarray, token_id: int) -> ProbeDecision | None:
        """Consume a frame and return the single decision when its read is ready."""
        if self._decision is not None:
            return None
        h = np.asarray(hidden, dtype=np.float32).reshape(-1)
        self.frame += 1

        if self._onset_frame is None:
            if int(token_id) == self.pad_id:
                for _, pending_h in self._candidate:
                    self._add_listen(pending_h)
                self._candidate.clear()
                self._add_listen(h)
                return None

            self._candidate.append((self.frame, h))
            if len(self._candidate) < 3:
                return None
            self._onset_frame = self._candidate[0][0]
            self._onset = [x for _, x in self._candidate]
            self._candidate.clear()
        else:
            self._onset.append(h)

        if len(self._onset) < self.window:
            return None

        onset = np.stack(self._onset[:self.window]).astype(np.float32)
        if self._listen_sum is None or self._listen_count == 0:
            user_mean = onset[0]
        else:
            user_mean = (self._listen_sum / self._listen_count).astype(np.float32)
        feature = np.concatenate([onset[-1], onset.mean(0), user_mean])
        if feature.shape != self.fail_w.shape:
            raise ValueError(
                f"probe feature shape {feature.shape} != weights {self.fail_w.shape}"
            )
        score = self._sigmoid(feature @ self.fail_w + self.fail_b)
        act_score = self._sigmoid(feature @ self.act_w + self.act_b)
        is_info = act_score >= self.act_tau
        self._decision = ProbeDecision(
            score=score,
            act_score=act_score,
            is_information_request=is_info,
            fired=is_info and score >= self.threshold,
            tier=self.tier,
            onset_frame=self._onset_frame,
            read_frame=self.frame,
        )
        return self._decision
