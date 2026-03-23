"""Research agent: generates testable hypotheses about AI trends from podcast summaries.

Variant of TaskAgent for the Sakana-style hypothesis-generation loop.
Input:  a list of episode summaries (dicts produced by TaskAgent)
Output: 3 testable hypotheses as structured JSON
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

_COST_IN_PER_M = 3.00
_COST_OUT_PER_M = 15.00

DEFAULT_MODEL = "claude-sonnet-4-6"

DEFAULT_SYSTEM_PROMPT = """You are an AI research analyst specialising in emerging technology trends.

You will receive summaries of 3-10 recent podcast episodes covering AI, startups, and technology.

Your task: identify patterns across the episodes and propose exactly 3 testable research hypotheses about AI trends.

Each hypothesis must be:
- Novel: not a restatement of something obvious or already well-established
- Testable: falsifiable via data, experiment, or structured observation
- Grounded: supported by specific content from the provided summaries

Return ONLY valid JSON with this exact schema:
{
  "hypotheses": [
    {
      "statement": "<one clear sentence stating the hypothesis>",
      "rationale": "<2-3 sentences explaining why the summaries support this>",
      "test_method": "<how you would test or falsify this hypothesis>",
      "source_episodes": ["<episode_id>", ...]
    }
  ]
}

Propose exactly 3 hypotheses. Be specific and non-obvious."""


@dataclass
class ResearchResult:
    output: dict[str, Any]          # {"hypotheses": [...]}
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    model: str


class ResearchAgent:
    """Proposes testable hypotheses from a collection of episode summaries.

    The system_prompt is designed to be mutated by the ImproverAgent,
    identical design pattern to TaskAgent.
    """

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        model: str = DEFAULT_MODEL,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.model = model
        self._client = client or anthropic.Anthropic()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def run(self, summaries: list[dict[str, Any]], episode_ids: list[str]) -> ResearchResult:
        """Generate hypotheses from a batch of episode summaries.

        Args:
            summaries: list of task-agent output dicts (key_topics, main_argument, …)
            episode_ids: matching episode identifiers (same order as summaries)
        """
        if not summaries:
            raise ValueError("summaries must not be empty")
        if len(summaries) != len(episode_ids):
            raise ValueError("summaries and episode_ids must have the same length")

        formatted = "\n\n".join(
            f"[Episode: {eid}]\n{json.dumps(s, indent=2)}"
            for eid, s in zip(episode_ids, summaries)
        )

        t0 = time.monotonic()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": f"Episode summaries:\n\n{formatted}"}],
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        raw = response.content[0].text.strip()
        output = _parse_json(raw)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost_usd = (
            input_tokens / 1_000_000 * _COST_IN_PER_M
            + output_tokens / 1_000_000 * _COST_OUT_PER_M
        )

        return ResearchResult(
            output=output,
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
