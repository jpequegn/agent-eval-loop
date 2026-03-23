"""Critic agent: evaluates TaskAgent output against a fixed rubric.

The critic's system prompt is intentionally frozen — scores must be comparable
across iterations. Never mutate CRITIC_SYSTEM_PROMPT.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

_COST_IN_PER_M = 15.00   # claude-opus-4-6 input
_COST_OUT_PER_M = 75.00  # claude-opus-4-6 output

DEFAULT_MODEL = "claude-opus-4-6"

# FROZEN — do not modify between iterations
CRITIC_SYSTEM_PROMPT = """You are a strict podcast summary evaluator. Score a given summary against the source transcript using this fixed rubric.

Score each dimension 1-5 (integers only):

- accuracy (1-5): Are the topics, quotes, and claims faithful to the transcript? 1=fabricated, 5=fully grounded.
- completeness (1-5): Are the most important ideas captured? 1=major gaps, 5=comprehensive.
- conciseness (1-5): Is the summary tight, without filler? 1=bloated/vague, 5=crisp and dense.
- actionability (1-5): Are the takeaways specific and useful to a listener? 1=generic platitudes, 5=concrete and actionable.

Return ONLY valid JSON with this exact schema:
{
  "scores": {
    "accuracy": <int>,
    "completeness": <int>,
    "conciseness": <int>,
    "actionability": <int>
  },
  "critiques": {
    "accuracy": "<one sentence explaining the score>",
    "completeness": "<one sentence explaining the score>",
    "conciseness": "<one sentence explaining the score>",
    "actionability": "<one sentence explaining the score>"
  },
  "overall_score": <float>   // arithmetic mean of the four scores, rounded to 2 decimal places
}"""


@dataclass
class CriticResult:
    scores: dict[str, int]          # accuracy, completeness, conciseness, actionability
    critiques: dict[str, str]       # per-dimension explanations
    overall_score: float            # mean of scores
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    model: str


class CriticAgent:
    """Evaluates TaskAgent output with a fixed, never-mutated rubric.

    Uses a stronger model (claude-opus-4-6) to ensure rubric enforcement is reliable.
    The system prompt is a module-level constant — never pass it as a constructor arg.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self._client = client or anthropic.Anthropic()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def run(self, transcript: str, task_output: dict[str, Any]) -> CriticResult:
        """Evaluate a task agent summary against the source transcript."""
        truncated = transcript[:60_000] if len(transcript) > 60_000 else transcript

        user_content = (
            f"TRANSCRIPT (may be truncated):\n{truncated}\n\n"
            f"SUMMARY TO EVALUATE:\n{json.dumps(task_output, indent=2)}"
        )

        t0 = time.monotonic()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system=CRITIC_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        raw = response.content[0].text.strip()
        parsed = _parse_json(raw)

        scores = parsed["scores"]
        critiques = parsed["critiques"]
        overall = parsed.get("overall_score") or round(sum(scores.values()) / len(scores), 2)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost_usd = (
            input_tokens / 1_000_000 * _COST_IN_PER_M
            + output_tokens / 1_000_000 * _COST_OUT_PER_M
        )

        return CriticResult(
            scores=scores,
            critiques=critiques,
            overall_score=overall,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            model=response.model,
        )


def _parse_json(text: str) -> dict[str, Any]:
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)
