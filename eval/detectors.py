"""Failure-mode detectors for the agent improvement loop.

Three detectors:
  1. CritiqueCollapseDetector  — critic variance drops below threshold
  2. ScoreGamingDetector       — scores rise but spot-check quality does not
  3. PromptCoherenceChecker    — LLM rates prompt clarity every N iterations
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Any

import anthropic


class AnomalyKind(str, Enum):
    CRITIQUE_COLLAPSE = "critique_collapse"
    SCORE_GAMING = "score_gaming"
    PROMPT_DRIFT = "prompt_drift"


@dataclass
class Anomaly:
    kind: AnomalyKind
    iteration: int
    message: str
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# 1. Critique Collapse Detector
# ---------------------------------------------------------------------------

COLLAPSE_VARIANCE_THRESHOLD = 0.10   # std dev below this is suspicious
COLLAPSE_WINDOW = 3                  # consecutive iterations to confirm


class CritiqueCollapseDetector:
    """Flags when critic score variance is near-zero for several iterations.

    This indicates the critic has stopped discriminating between good and bad
    outputs — a classic failure of self-improvement loops.
    """

    def check(
        self,
        iteration: int,
        recent_scores: list[list[float]],
    ) -> Anomaly | None:
        """
        Args:
            iteration: current iteration number.
            recent_scores: one list of per-episode scores per recent iteration,
                           newest last. Needs at least COLLAPSE_WINDOW entries.
        """
        if len(recent_scores) < COLLAPSE_WINDOW:
            return None

        window = recent_scores[-COLLAPSE_WINDOW:]
        stdevs = []
        for scores in window:
            if len(scores) < 2:
                return None   # can't compute variance on a single episode
            stdevs.append(statistics.stdev(scores))

        if all(s < COLLAPSE_VARIANCE_THRESHOLD for s in stdevs):
            avg_stdev = sum(stdevs) / len(stdevs)
            return Anomaly(
                kind=AnomalyKind.CRITIQUE_COLLAPSE,
                iteration=iteration,
                message=(
                    f"Critique collapse detected: critic score std dev has been "
                    f"< {COLLAPSE_VARIANCE_THRESHOLD} for {COLLAPSE_WINDOW} consecutive iterations "
                    f"(avg stdev={avg_stdev:.3f}). Scores may no longer be meaningful."
                ),
                details={"stdevs": stdevs, "threshold": COLLAPSE_VARIANCE_THRESHOLD},
            )
        return None


# ---------------------------------------------------------------------------
# 2. Score Gaming Detector
# ---------------------------------------------------------------------------

GAMING_SCORE_RISE_THRESHOLD = 0.5   # score improvement over window
GAMING_WINDOW = 5                   # iterations to measure rise over


class ScoreGamingDetector:
    """Flags when automated scores rise sharply but spot-check scores do not.

    Score gaming (Goodhart's Law): the task agent learns to satisfy the
    critic's rubric without actually producing better summaries.

    In the absence of a human oracle, we use a second LLM call on a held-out
    prompt that the task agent has NOT been optimised against.
    """

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client or anthropic.Anthropic()

    SPOT_CHECK_PROMPT = (
        "Rate the following podcast summary holistically on a scale of 1-5 "
        "(1=poor, 5=excellent). Consider whether it is accurate, useful, and "
        "well-written. Return ONLY a JSON object: {\"score\": <int>, \"reason\": \"<one sentence>\"}."
    )

    def spot_check(self, transcript: str, summary: dict[str, Any]) -> tuple[float, str]:
        """Run a blind spot-check using a fixed prompt not exposed to the improver."""
        import json

        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=64,
            system=self.SPOT_CHECK_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Transcript (excerpt):\n{transcript[:3000]}\n\nSummary:\n{json.dumps(summary)}",
            }],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        parsed = json.loads(raw)
        return float(parsed["score"]), parsed.get("reason", "")

    def check(
        self,
        iteration: int,
        avg_scores: list[float],
        spot_check_scores: list[float],
    ) -> Anomaly | None:
        """Compare automated-score trend vs. spot-check trend.

        Args:
            avg_scores: automated critic avg scores per iteration (newest last).
            spot_check_scores: spot-check scores per iteration (newest last).
        """
        if len(avg_scores) < GAMING_WINDOW or len(spot_check_scores) < GAMING_WINDOW:
            return None

        auto_rise = avg_scores[-1] - avg_scores[-GAMING_WINDOW]
        spot_rise = spot_check_scores[-1] - spot_check_scores[-GAMING_WINDOW]

        if auto_rise >= GAMING_SCORE_RISE_THRESHOLD and spot_rise < 0:
            return Anomaly(
                kind=AnomalyKind.SCORE_GAMING,
                iteration=iteration,
                message=(
                    f"Score gaming suspected: automated scores rose +{auto_rise:.2f} "
                    f"over {GAMING_WINDOW} iterations while spot-check scores fell "
                    f"{spot_rise:.2f}. The prompt may be gaming the rubric."
                ),
                details={
                    "auto_rise": auto_rise,
                    "spot_rise": spot_rise,
                    "window": GAMING_WINDOW,
                },
            )
        return None


# ---------------------------------------------------------------------------
# 3. Prompt Coherence Checker
# ---------------------------------------------------------------------------

COHERENCE_CHECK_INTERVAL = 10   # check every N iterations
COHERENCE_LOW_THRESHOLD = 3.0   # flag if coherence score drops below this


class PromptCoherenceChecker:
    """Uses an LLM to rate prompt clarity every N iterations.

    Catches prompt drift: after many edits the prompt can become contradictory,
    repetitive, or incoherent even if scores are still rising.
    """

    COHERENCE_SYSTEM = (
        "You are a prompt quality evaluator. Rate the following AI system prompt "
        "on clarity and coherence (1-5). 1=contradictory/incoherent, "
        "5=clear, focused, and well-structured. "
        "Return ONLY JSON: {\"score\": <int>, \"issues\": [\"<issue>\", ...]}"
    )

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self._client = client or anthropic.Anthropic()

    def should_check(self, iteration: int) -> bool:
        return (iteration + 1) % COHERENCE_CHECK_INTERVAL == 0

    def check(self, iteration: int, system_prompt: str) -> Anomaly | None:
        """Score the prompt's coherence. Returns an Anomaly if score is low."""
        import json

        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=128,
            system=self.COHERENCE_SYSTEM,
            messages=[{"role": "user", "content": system_prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        parsed = json.loads(raw)
        score = float(parsed["score"])
        issues = parsed.get("issues", [])

        if score < COHERENCE_LOW_THRESHOLD:
            return Anomaly(
                kind=AnomalyKind.PROMPT_DRIFT,
                iteration=iteration,
                message=(
                    f"Prompt drift detected at iteration {iteration + 1}: "
                    f"coherence score={score:.1f}/5. Issues: {'; '.join(issues)}"
                ),
                details={"coherence_score": score, "issues": issues},
            )
        return None


# ---------------------------------------------------------------------------
# DetectorSuite: convenience wrapper used by the orchestrator
# ---------------------------------------------------------------------------

@dataclass
class DetectorSuite:
    """Aggregates all detectors. Call run() after each iteration."""

    collapse: CritiqueCollapseDetector
    gaming: ScoreGamingDetector
    coherence: PromptCoherenceChecker

    @classmethod
    def default(cls, client: anthropic.Anthropic | None = None) -> "DetectorSuite":
        c = client or anthropic.Anthropic()
        return cls(
            collapse=CritiqueCollapseDetector(),
            gaming=ScoreGamingDetector(client=c),
            coherence=PromptCoherenceChecker(client=c),
        )

    def run(
        self,
        iteration: int,
        recent_episode_scores: list[list[float]],
        avg_scores: list[float],
        spot_check_scores: list[float],
        system_prompt: str,
    ) -> list[Anomaly]:
        anomalies: list[Anomaly] = []

        a = self.collapse.check(iteration, recent_episode_scores)
        if a:
            anomalies.append(a)

        a = self.gaming.check(iteration, avg_scores, spot_check_scores)
        if a:
            anomalies.append(a)

        if self.coherence.should_check(iteration):
            a = self.coherence.check(iteration, system_prompt)
            if a:
                anomalies.append(a)

        return anomalies
