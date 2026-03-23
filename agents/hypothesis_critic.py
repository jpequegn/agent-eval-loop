"""Hypothesis critic: evaluates research hypotheses on novelty, testability, groundedness.

Fixed rubric — never mutate CRITIC_SYSTEM_PROMPT. Same design constraint as CriticAgent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

_COST_IN_PER_M = 15.00   # claude-opus-4-6
_COST_OUT_PER_M = 75.00

DEFAULT_MODEL = "claude-opus-4-6"

# FROZEN — do not modify between iterations
CRITIC_SYSTEM_PROMPT = """You are a research hypothesis evaluator. Score each hypothesis on three dimensions.

Dimensions (1-5 integers):
- novelty (1-5): Is the hypothesis surprising and non-obvious? 1=trivially obvious, 5=genuinely novel insight.
- testability (1-5): Can it be falsified via data or experiment? 1=unfalsifiable/vague, 5=clear test exists.
- groundedness (1-5): Is it supported by the provided source summaries? 1=unsupported speculation, 5=strongly supported.

You will receive a list of hypotheses and the source summaries used to generate them.

Return ONLY valid JSON with this exact schema:
{
  "evaluations": [
    {
      "statement": "<the hypothesis statement>",
      "scores": {"novelty": <int>, "testability": <int>, "groundedness": <int>},
      "critiques": {
        "novelty": "<one sentence>",
        "testability": "<one sentence>",
        "groundedness": "<one sentence>"
      },
      "overall_score": <float>
    }
  ],
  "avg_overall_score": <float>
}"""


@dataclass
class HypothesisEvaluation:
    statement: str
    scores: dict[str, int]
    critiques: dict[str, str]
    overall_score: float


@dataclass
class HypothesisCriticResult:
    evaluations: list[HypothesisEvaluation]
    avg_overall_score: float
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    model: str

    @property
    def overall_score(self) -> float:
        """Alias for compatibility with IterationRecord interface."""
        return self.avg_overall_score


class HypothesisCriticAgent:
    """Evaluates hypotheses with a fixed, never-mutated rubric.

    Uses claude-opus-4-6 for reliable rubric enforcement.
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
    def run(
        self,
        research_output: dict[str, Any],
        source_summaries: list[dict[str, Any]],
        episode_ids: list[str],
    ) -> HypothesisCriticResult:
        """Evaluate hypotheses in research_output against the source summaries."""
        summaries_text = "\n\n".join(
            f"[{eid}]: {json.dumps(s, indent=2)}"
            for eid, s in zip(episode_ids, source_summaries)
        )
        user_content = (
            f"SOURCE SUMMARIES:\n{summaries_text}\n\n"
            f"HYPOTHESES TO EVALUATE:\n{json.dumps(research_output, indent=2)}"
        )

        t0 = time.monotonic()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=CRITIC_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        raw = response.content[0].text.strip()
        parsed = _parse_json(raw)

        evaluations = [
            HypothesisEvaluation(
                statement=e["statement"],
                scores=e["scores"],
                critiques=e["critiques"],
                overall_score=e.get("overall_score")
                    or round(sum(e["scores"].values()) / len(e["scores"]), 2),
            )
            for e in parsed["evaluations"]
        ]
        avg = parsed.get("avg_overall_score") or (
            sum(e.overall_score for e in evaluations) / len(evaluations)
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost_usd = (
            input_tokens / 1_000_000 * _COST_IN_PER_M
            + output_tokens / 1_000_000 * _COST_OUT_PER_M
        )

        return HypothesisCriticResult(
            evaluations=evaluations,
            avg_overall_score=round(avg, 2),
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
