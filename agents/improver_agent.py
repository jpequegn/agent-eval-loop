"""Improver agent: rewrites the task agent's system prompt based on critic feedback.

Closes the meta-learning loop: critique scores → improved prompt → better summaries.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

_COST_IN_PER_M = 3.00   # claude-sonnet-4-6
_COST_OUT_PER_M = 15.00

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_PROMPT_LENGTH = 4000

IMPROVER_SYSTEM_PROMPT = """You are a prompt engineer specialising in improving AI task prompts.

You will receive:
1. The current system prompt used by a podcast summarisation agent
2. The last 1-3 critique outputs (scores + explanations per dimension)
3. The score history across iterations

Your job: rewrite the system prompt to address the weaknesses the critic identified, while preserving what is working.

Rules you MUST follow:
- The output prompt MUST instruct the agent to return ONLY valid JSON with this exact schema:
  {"key_topics": [...], "main_argument": "...", "notable_quotes": [...], "actionable_takeaways": [...]}
- Maximum length: 4000 characters
- Do not add instructions that conflict with JSON-only output
- Do not remove the JSON schema definition from the prompt
- Make targeted changes — do not rewrite everything at once

Return ONLY the new system prompt text. No explanation, no fencing, no preamble."""


@dataclass
class ImproverResult:
    new_prompt: str
    was_reverted: bool               # True if safety check triggered a revert
    revert_reason: str | None        # explanation when was_reverted=True
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    model: str


@dataclass
class IterationRecord:
    iteration: int
    system_prompt: str
    score: float
    critiques: list[dict[str, Any]] = field(default_factory=list)


class ImproverAgent:
    """Meta-agent that rewrites the TaskAgent system prompt based on critique history.

    Safety rule: if the score has declined for 2 consecutive iterations,
    revert to the best-known prompt instead of generating a new one.
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
    def run(self, history: list[IterationRecord]) -> ImproverResult:
        """Generate the next system prompt given the iteration history.

        Args:
            history: ordered list of past iterations (oldest first).
                     Must contain at least one record.

        Returns:
            ImproverResult with the new (or reverted) prompt.
        """
        if not history:
            raise ValueError("history must contain at least one IterationRecord")

        # Safety check: revert if score declined for 2 consecutive iterations
        revert_result = _check_revert(history)
        if revert_result is not None:
            return revert_result

        current = history[-1]
        recent_critiques = [h.critiques for h in history[-3:] if h.critiques]
        score_history = [{"iteration": h.iteration, "score": h.score} for h in history]

        user_content = (
            f"CURRENT SYSTEM PROMPT:\n{current.system_prompt}\n\n"
            f"RECENT CRITIQUES (last {len(recent_critiques)} iteration(s)):\n"
            f"{json.dumps(recent_critiques, indent=2)}\n\n"
            f"SCORE HISTORY:\n{json.dumps(score_history, indent=2)}\n\n"
            "Write an improved system prompt that addresses the weaknesses above."
        )

        t0 = time.monotonic()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=IMPROVER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        new_prompt = response.content[0].text.strip()
        new_prompt = _enforce_constraints(new_prompt, current.system_prompt)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost_usd = (
            input_tokens / 1_000_000 * _COST_IN_PER_M
            + output_tokens / 1_000_000 * _COST_OUT_PER_M
        )

        return ImproverResult(
            new_prompt=new_prompt,
            was_reverted=False,
            revert_reason=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            model=response.model,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_revert(history: list[IterationRecord]) -> ImproverResult | None:
    """Return a revert ImproverResult if scores declined 2 iterations in a row."""
    if len(history) < 3:
        return None

    last_three = history[-3:]
    scores = [r.score for r in last_three]
    if scores[1] < scores[0] and scores[2] < scores[1]:
        best = max(history, key=lambda r: r.score)
        reason = (
            f"Score declined 2 consecutive iterations "
            f"({scores[0]:.2f} → {scores[1]:.2f} → {scores[2]:.2f}). "
            f"Reverting to best prompt from iteration {best.iteration} "
            f"(score {best.score:.2f})."
        )
        return ImproverResult(
            new_prompt=best.system_prompt,
            was_reverted=True,
            revert_reason=reason,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            cost_usd=0.0,
            model="none (revert)",
        )
    return None


def _enforce_constraints(prompt: str, fallback: str) -> str:
    """Enforce max length. Falls back to the previous prompt if the new one is too long."""
    if len(prompt) > MAX_PROMPT_LENGTH:
        return prompt[:MAX_PROMPT_LENGTH]
    if not prompt.strip():
        return fallback
    return prompt
