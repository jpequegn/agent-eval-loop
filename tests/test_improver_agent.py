"""Tests for ImproverAgent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agents.improver_agent import (
    MAX_PROMPT_LENGTH,
    ImproverAgent,
    IterationRecord,
    _check_revert,
    _enforce_constraints,
)
from agents.task_agent import DEFAULT_SYSTEM_PROMPT

SAMPLE_CRITIQUES = [
    {
        "accuracy": "Topics are grounded but one quote is paraphrased.",
        "completeness": "Missing the entrepreneurship angle.",
        "conciseness": "Good.",
        "actionability": "Takeaways are too generic.",
    }
]

IMPROVED_PROMPT = (
    'You are an expert podcast analyst. Focus on specificity.\n\n'
    'Return ONLY valid JSON:\n'
    '{"key_topics": [...], "main_argument": "...", "notable_quotes": [...], "actionable_takeaways": [...]}'
)


def _record(iteration: int, score: float, prompt: str = DEFAULT_SYSTEM_PROMPT) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        system_prompt=prompt,
        score=score,
        critiques=SAMPLE_CRITIQUES,
    )


def _make_mock_client(response_text: str, input_tokens: int = 120, output_tokens: int = 80) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=response_text)]
    mock_response.usage.input_tokens = input_tokens
    mock_response.usage.output_tokens = output_tokens
    mock_response.model = "claude-sonnet-4-6"
    mock_client.messages.create.return_value = mock_response
    return mock_client


class TestEnforceConstraints:
    def test_passes_valid_prompt(self) -> None:
        assert _enforce_constraints("Valid prompt", "fallback") == "Valid prompt"

    def test_truncates_at_max_length(self) -> None:
        long = "x" * (MAX_PROMPT_LENGTH + 500)
        result = _enforce_constraints(long, "fallback")
        assert len(result) == MAX_PROMPT_LENGTH

    def test_empty_prompt_returns_fallback(self) -> None:
        assert _enforce_constraints("   ", "fallback") == "fallback"


class TestCheckRevert:
    def test_no_revert_with_fewer_than_3_iterations(self) -> None:
        history = [_record(0, 3.0), _record(1, 2.5)]
        assert _check_revert(history) is None

    def test_no_revert_when_scores_improving(self) -> None:
        history = [_record(0, 3.0), _record(1, 3.5), _record(2, 4.0)]
        assert _check_revert(history) is None

    def test_no_revert_when_only_one_decline(self) -> None:
        history = [_record(0, 4.0), _record(1, 3.5), _record(2, 3.8)]
        assert _check_revert(history) is None

    def test_revert_triggered_on_two_consecutive_declines(self) -> None:
        history = [_record(0, 4.0), _record(1, 3.5), _record(2, 3.0)]
        result = _check_revert(history)
        assert result is not None
        assert result.was_reverted is True
        assert "2 consecutive" in result.revert_reason

    def test_revert_selects_best_known_prompt(self) -> None:
        best_prompt = "best prompt ever"
        history = [
            _record(0, 4.5, prompt=best_prompt),
            _record(1, 3.5),
            _record(2, 3.0),
        ]
        result = _check_revert(history)
        assert result.new_prompt == best_prompt

    def test_revert_zero_cost(self) -> None:
        history = [_record(0, 4.0), _record(1, 3.5), _record(2, 3.0)]
        result = _check_revert(history)
        assert result.cost_usd == 0.0
        assert result.input_tokens == 0


class TestImproverAgent:
    def test_raises_on_empty_history(self) -> None:
        agent = ImproverAgent(client=MagicMock())
        with pytest.raises(ValueError, match="history must contain"):
            agent.run([])

    def test_run_returns_improver_result(self) -> None:
        mock_client = _make_mock_client(IMPROVED_PROMPT)
        agent = ImproverAgent(client=mock_client)
        history = [_record(0, 3.0), _record(1, 3.5)]

        result = agent.run(history)

        assert result.new_prompt == IMPROVED_PROMPT
        assert result.was_reverted is False
        assert result.revert_reason is None
        assert result.input_tokens == 120
        assert result.output_tokens == 80
        assert result.latency_ms >= 0

    def test_cost_calculation(self) -> None:
        mock_client = _make_mock_client(IMPROVED_PROMPT, input_tokens=1_000_000, output_tokens=1_000_000)
        agent = ImproverAgent(client=mock_client)

        result = agent.run([_record(0, 3.0)])

        # 1M in @ $3 + 1M out @ $15 = $18
        assert abs(result.cost_usd - 18.0) < 0.01

    def test_revert_skips_api_call(self) -> None:
        mock_client = _make_mock_client(IMPROVED_PROMPT)
        agent = ImproverAgent(client=mock_client)
        declining = [_record(0, 4.0), _record(1, 3.5), _record(2, 3.0)]

        result = agent.run(declining)

        assert result.was_reverted is True
        mock_client.messages.create.assert_not_called()

    def test_new_prompt_sent_to_api(self) -> None:
        mock_client = _make_mock_client(IMPROVED_PROMPT)
        agent = ImproverAgent(client=mock_client)
        history = [_record(0, 3.0)]

        agent.run(history)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "CURRENT SYSTEM PROMPT" in call_kwargs["messages"][0]["content"]
        assert "SCORE HISTORY" in call_kwargs["messages"][0]["content"]

    def test_prompt_truncated_if_over_max_length(self) -> None:
        oversized = "x" * (MAX_PROMPT_LENGTH + 1000)
        mock_client = _make_mock_client(oversized)
        agent = ImproverAgent(client=mock_client)

        result = agent.run([_record(0, 3.0)])

        assert len(result.new_prompt) == MAX_PROMPT_LENGTH

    def test_recent_critiques_included_in_context(self) -> None:
        mock_client = _make_mock_client(IMPROVED_PROMPT)
        agent = ImproverAgent(client=mock_client)
        history = [_record(i, 3.0 + i * 0.1) for i in range(5)]

        agent.run(history)

        content = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        # Only last 3 critiques should be included
        assert content.count("accuracy") <= 3
