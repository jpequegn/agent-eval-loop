"""Tests for TaskAgent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.task_agent import DEFAULT_SYSTEM_PROMPT, TaskAgent, _parse_json

SAMPLE_OUTPUT = {
    "key_topics": ["AI agents", "open source", "entrepreneurship"],
    "main_argument": "OpenClaw's success shows that autonomous AI agents with system access are the next frontier.",
    "notable_quotes": ["I just prompted it into existence."],
    "actionable_takeaways": ["Build in public", "Start with a personal itch"],
}

SAMPLE_TRANSCRIPT = (
    "Lex: Tell me about OpenClaw.\n"
    "Peter: I wanted a personal assistant since April. I just prompted it into existence.\n"
    "Lex: What makes it different?\n"
    "Peter: It has access to all your stuff and does things autonomously.\n"
)


def _make_mock_client(output_text: str, input_tokens: int = 100, output_tokens: int = 200) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=output_text)]
    mock_response.usage.input_tokens = input_tokens
    mock_response.usage.output_tokens = output_tokens
    mock_response.model = "claude-sonnet-4-6"
    mock_client.messages.create.return_value = mock_response
    return mock_client


class TestParseJson:
    def test_plain_json(self) -> None:
        result = _parse_json(json.dumps(SAMPLE_OUTPUT))
        assert result["key_topics"] == SAMPLE_OUTPUT["key_topics"]

    def test_strips_markdown_fence(self) -> None:
        fenced = f"```json\n{json.dumps(SAMPLE_OUTPUT)}\n```"
        result = _parse_json(fenced)
        assert result["main_argument"] == SAMPLE_OUTPUT["main_argument"]

    def test_strips_plain_fence(self) -> None:
        fenced = f"```\n{json.dumps(SAMPLE_OUTPUT)}\n```"
        result = _parse_json(fenced)
        assert "key_topics" in result

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_json("not json at all")


class TestTaskAgent:
    def test_default_system_prompt(self) -> None:
        agent = TaskAgent()
        assert agent.system_prompt == DEFAULT_SYSTEM_PROMPT
        assert "key_topics" in agent.system_prompt

    def test_custom_system_prompt(self) -> None:
        agent = TaskAgent(system_prompt="Custom prompt")
        assert agent.system_prompt == "Custom prompt"

    def test_run_returns_task_result(self) -> None:
        mock_client = _make_mock_client(json.dumps(SAMPLE_OUTPUT))
        agent = TaskAgent(client=mock_client)

        result = agent.run(SAMPLE_TRANSCRIPT)

        assert result.output == SAMPLE_OUTPUT
        assert result.input_tokens == 100
        assert result.output_tokens == 200
        assert result.model == "claude-sonnet-4-6"
        assert result.latency_ms >= 0
        assert result.cost_usd > 0

    def test_cost_calculation(self) -> None:
        mock_client = _make_mock_client(json.dumps(SAMPLE_OUTPUT), input_tokens=1_000_000, output_tokens=1_000_000)
        agent = TaskAgent(client=mock_client)

        result = agent.run(SAMPLE_TRANSCRIPT)

        # 1M input @ $3 + 1M output @ $15 = $18
        assert abs(result.cost_usd - 18.0) < 0.01

    def test_transcript_truncated_at_100k(self) -> None:
        mock_client = _make_mock_client(json.dumps(SAMPLE_OUTPUT))
        agent = TaskAgent(client=mock_client)

        long_transcript = "word " * 30_000  # ~150k chars
        agent.run(long_transcript)

        call_args = mock_client.messages.create.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert len(user_content) <= 100_100  # 100k transcript + "Transcript:\n\n" prefix

    def test_strips_markdown_fence_from_response(self) -> None:
        fenced = f"```json\n{json.dumps(SAMPLE_OUTPUT)}\n```"
        mock_client = _make_mock_client(fenced)
        agent = TaskAgent(client=mock_client)

        result = agent.run(SAMPLE_TRANSCRIPT)
        assert result.output["key_topics"] == SAMPLE_OUTPUT["key_topics"]

    def test_passes_system_prompt_to_api(self) -> None:
        mock_client = _make_mock_client(json.dumps(SAMPLE_OUTPUT))
        custom_prompt = "You are a custom analyst."
        agent = TaskAgent(system_prompt=custom_prompt, client=mock_client)

        agent.run(SAMPLE_TRANSCRIPT)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == custom_prompt

    def test_uses_configured_model(self) -> None:
        mock_client = _make_mock_client(json.dumps(SAMPLE_OUTPUT))
        agent = TaskAgent(model="claude-haiku-4-5-20251001", client=mock_client)

        agent.run(SAMPLE_TRANSCRIPT)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
