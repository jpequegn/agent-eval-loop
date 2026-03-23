"""Task agent: summarizes a podcast episode transcript into structured JSON."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

# Input cost per million tokens (claude-sonnet-4-6, as of 2026-03)
_COST_IN_PER_M = 3.00
_COST_OUT_PER_M = 15.00

DEFAULT_MODEL = "claude-sonnet-4-6"

DEFAULT_SYSTEM_PROMPT = """You are an expert podcast analyst. Given a podcast transcript, produce a concise structured summary.

Return ONLY valid JSON with this exact schema:
{
  "key_topics": ["<topic>", ...],          // 3-7 main topics discussed
  "main_argument": "<string>",              // the central thesis or takeaway in 1-2 sentences
  "notable_quotes": ["<quote>", ...],       // 2-5 verbatim or near-verbatim quotes
  "actionable_takeaways": ["<item>", ...]   // 3-5 concrete things a listener can do/learn
}

Be specific and grounded in the actual content. Do not invent information."""


@dataclass
class TaskResult:
    output: dict[str, Any]
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    model: str


class TaskAgent:
    """Summarizes podcast transcripts into structured JSON.

    The system_prompt is designed to be replaced by the improver agent.
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
    def run(self, transcript: str) -> TaskResult:
        """Run the task agent on a transcript, returning structured output + metadata."""
        # Truncate very long transcripts to stay within context limits (~100k chars ≈ 25k tokens)
        truncated = transcript[:100_000] if len(transcript) > 100_000 else transcript

        t0 = time.monotonic()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": f"Transcript:\n\n{truncated}"}],
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

        return TaskResult(
            output=output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            model=response.model,
        )


def _parse_json(text: str) -> dict[str, Any]:
    """Parse JSON from model output, stripping markdown fences if present."""
    import json

    # Strip ```json ... ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    return json.loads(text)
