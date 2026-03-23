"""Tests for CriticAgent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agents.critic_agent import CRITIC_SYSTEM_PROMPT, CriticAgent, _parse_json

SAMPLE_TRANSCRIPT = (
    "Lex: Tell me about OpenClaw.\n"
    "Peter: I wanted a personal assistant. I just prompted it into existence.\n"
    "Lex: What makes it different?\n"
    "Peter: It has system-level access. Build in public. Start with a personal itch.\n"
)

GOOD_SUMMARY = {
    "key_topics": ["AI agents", "open source", "entrepreneurship"],
    "main_argument": "OpenClaw democratizes autonomous AI by giving agents system-level access.",
    "notable_quotes": ["I just prompted it into existence."],
    "actionable_takeaways": ["Build in public", "Start with a personal itch"],
}

BAD_SUMMARY = {
    "key_topics": ["cooking", "travel"],
    "main_argument": "This episode is about food.",
    "notable_quotes": ["I love pizza."],
    "actionable_takeaways": ["Be yourself"],
}

SAMPLE_CRITIC_OUTPUT = {
    "scores": {"accuracy": 4, "completeness": 4, "conciseness": 5, "actionability": 4},
    "critiques": {
        "accuracy": "Topics and quotes are grounded in the transcript.",
        "completeness": "Main ideas captured; minor details omitted.",
        "conciseness": "Summary is tight with no filler.",
        "actionability": "Takeaways are specific and useful.",
    },
    "overall_score": 4.25,
}

LOW_CRITIC_OUTPUT = {
    "scores": {"accuracy": 1, "completeness": 1, "conciseness": 2, "actionability": 1},
    "critiques": {
        "accuracy": "Topics are completely fabricated.",
        "completeness": "None of the actual content is represented.",
        "conciseness": "Short but wrong.",
        "actionability": "Generic platitude with no grounding.",
    },
    "overall_score": 1.25,
}


def _make_mock_client(output: dict, input_tokens: int = 150, output_tokens: int = 100) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(output))]
    mock_response.usage.input_tokens = input_tokens
    mock_response.usage.output_tokens = output_tokens
    mock_response.model = "claude-opus-4-6"
    mock_client.messages.create.return_value = mock_response
    return mock_client


class TestParseJson:
    def test_plain_json(self) -> None:
        result = _parse_json(json.dumps(SAMPLE_CRITIC_OUTPUT))
        assert result["overall_score"] == 4.25

    def test_strips_markdown_fence(self) -> None:
        fenced = f"```json\n{json.dumps(SAMPLE_CRITIC_OUTPUT)}\n```"
        assert _parse_json(fenced)["scores"]["accuracy"] == 4


class TestCriticAgent:
    def test_system_prompt_is_frozen(self) -> None:
        # The prompt must not be configurable — scores need to be comparable
        agent = CriticAgent()
        assert not hasattr(agent, "system_prompt"), (
            "CriticAgent must not expose system_prompt — it must stay frozen"
        )

    def test_critic_system_prompt_has_all_dimensions(self) -> None:
        for dim in ("accuracy", "completeness", "conciseness", "actionability"):
            assert dim in CRITIC_SYSTEM_PROMPT

    def test_run_returns_critic_result(self) -> None:
        mock_client = _make_mock_client(SAMPLE_CRITIC_OUTPUT)
        agent = CriticAgent(client=mock_client)

        result = agent.run(SAMPLE_TRANSCRIPT, GOOD_SUMMARY)

        assert result.scores == SAMPLE_CRITIC_OUTPUT["scores"]
        assert result.overall_score == 4.25
        assert result.input_tokens == 150
        assert result.output_tokens == 100
        assert result.latency_ms >= 0
        assert result.cost_usd > 0

    def test_cost_calculation(self) -> None:
        mock_client = _make_mock_client(SAMPLE_CRITIC_OUTPUT, input_tokens=1_000_000, output_tokens=1_000_000)
        agent = CriticAgent(client=mock_client)

        result = agent.run(SAMPLE_TRANSCRIPT, GOOD_SUMMARY)

        # 1M input @ $15 + 1M output @ $75 = $90
        assert abs(result.cost_usd - 90.0) < 0.01

    def test_overall_score_computed_if_missing(self) -> None:
        output_without_overall = {**SAMPLE_CRITIC_OUTPUT, "overall_score": None}
        mock_client = _make_mock_client(output_without_overall)
        agent = CriticAgent(client=mock_client)

        result = agent.run(SAMPLE_TRANSCRIPT, GOOD_SUMMARY)

        assert result.overall_score == pytest.approx(4.25)

    def test_good_summary_scores_higher_than_bad(self) -> None:
        """Key behavioral test: bad summary must score lower than good summary."""
        good_client = _make_mock_client(SAMPLE_CRITIC_OUTPUT)
        bad_client = _make_mock_client(LOW_CRITIC_OUTPUT)

        good_result = CriticAgent(client=good_client).run(SAMPLE_TRANSCRIPT, GOOD_SUMMARY)
        bad_result = CriticAgent(client=bad_client).run(SAMPLE_TRANSCRIPT, BAD_SUMMARY)

        assert good_result.overall_score > bad_result.overall_score, (
            f"Good summary ({good_result.overall_score}) should score higher "
            f"than bad summary ({bad_result.overall_score})"
        )

    def test_transcript_truncated_at_60k(self) -> None:
        mock_client = _make_mock_client(SAMPLE_CRITIC_OUTPUT)
        agent = CriticAgent(client=mock_client)

        long_transcript = "word " * 20_000  # ~100k chars
        agent.run(long_transcript, GOOD_SUMMARY)

        call_args = mock_client.messages.create.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert len(user_content) < 62_000

    def test_passes_frozen_system_prompt(self) -> None:
        mock_client = _make_mock_client(SAMPLE_CRITIC_OUTPUT)
        agent = CriticAgent(client=mock_client)

        agent.run(SAMPLE_TRANSCRIPT, GOOD_SUMMARY)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == CRITIC_SYSTEM_PROMPT

    def test_uses_opus_model_by_default(self) -> None:
        mock_client = _make_mock_client(SAMPLE_CRITIC_OUTPUT)
        agent = CriticAgent(client=mock_client)

        agent.run(SAMPLE_TRANSCRIPT, GOOD_SUMMARY)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-4-6"
