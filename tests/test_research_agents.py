"""Tests for ResearchAgent, HypothesisCriticAgent, and ResearchLoop."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agents.hypothesis_critic import (
    CRITIC_SYSTEM_PROMPT,
    HypothesisCriticAgent,
    HypothesisCriticResult,
)
from agents.improver_agent import ImproverResult
from agents.research_agent import DEFAULT_SYSTEM_PROMPT, ResearchAgent, ResearchResult
from loop.research_loop import ResearchLoop, ResearchLoopState, _build_history


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SUMMARIES = [
    {
        "key_topics": ["AI agents", "autonomy"],
        "main_argument": "Autonomous agents are replacing scripted tools.",
        "notable_quotes": ["I just prompted it into existence."],
        "actionable_takeaways": ["Build in public"],
    },
    {
        "key_topics": ["open source", "LLM"],
        "main_argument": "Open-source models are closing the gap on proprietary ones.",
        "notable_quotes": ["DeepSeek changed everything."],
        "actionable_takeaways": ["Use open weights for cost efficiency"],
    },
]
EPISODE_IDS = ["ep1", "ep2"]

SAMPLE_HYPOTHESES = {
    "hypotheses": [
        {
            "statement": "Autonomous AI agents will surpass scripted automation in enterprise by 2027.",
            "rationale": "Multiple episodes highlight rapid agent adoption.",
            "test_method": "Survey enterprise adoption rates annually.",
            "source_episodes": ["ep1"],
        },
        {
            "statement": "Open-source LLMs will match GPT-5 on coding benchmarks within 12 months.",
            "rationale": "DeepSeek and Llama show fast convergence.",
            "test_method": "Compare HumanEval scores quarterly.",
            "source_episodes": ["ep2"],
        },
        {
            "statement": "Prompt engineering will become a dedicated engineering discipline.",
            "rationale": "Growing tooling and courses signal professionalisation.",
            "test_method": "Track job postings on LinkedIn.",
            "source_episodes": ["ep1", "ep2"],
        },
    ]
}

SAMPLE_CRITIC_OUTPUT = {
    "evaluations": [
        {
            "statement": "Autonomous AI agents will surpass scripted automation in enterprise by 2027.",
            "scores": {"novelty": 3, "testability": 5, "groundedness": 4},
            "critiques": {
                "novelty": "Somewhat expected given current trends.",
                "testability": "Clear metric and timeline.",
                "groundedness": "Well supported by ep1.",
            },
            "overall_score": 4.0,
        },
        {
            "statement": "Open-source LLMs will match GPT-5 on coding benchmarks within 12 months.",
            "scores": {"novelty": 4, "testability": 5, "groundedness": 4},
            "critiques": {
                "novelty": "Not fully obvious given gap.",
                "testability": "Benchmark comparison is straightforward.",
                "groundedness": "Supported by DeepSeek reference.",
            },
            "overall_score": 4.33,
        },
        {
            "statement": "Prompt engineering will become a dedicated engineering discipline.",
            "scores": {"novelty": 2, "testability": 3, "groundedness": 3},
            "critiques": {
                "novelty": "Widely discussed already.",
                "testability": "Job postings are a proxy not a direct test.",
                "groundedness": "Loosely implied by sources.",
            },
            "overall_score": 2.67,
        },
    ],
    "avg_overall_score": 3.67,
}


def _mock_research_client(output: dict) -> MagicMock:
    c = MagicMock()
    r = MagicMock()
    r.content = [MagicMock(text=json.dumps(output))]
    r.usage.input_tokens = 200
    r.usage.output_tokens = 150
    r.model = "claude-sonnet-4-6"
    c.messages.create.return_value = r
    return c


def _mock_critic_client(output: dict) -> MagicMock:
    c = MagicMock()
    r = MagicMock()
    r.content = [MagicMock(text=json.dumps(output))]
    r.usage.input_tokens = 300
    r.usage.output_tokens = 200
    r.model = "claude-opus-4-6"
    c.messages.create.return_value = r
    return c


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------

class TestResearchAgent:
    def test_default_system_prompt_contains_schema(self) -> None:
        assert "hypotheses" in DEFAULT_SYSTEM_PROMPT
        assert "testable" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_custom_system_prompt(self) -> None:
        agent = ResearchAgent(system_prompt="Custom")
        assert agent.system_prompt == "Custom"

    def test_raises_on_empty_summaries(self) -> None:
        agent = ResearchAgent(client=MagicMock())
        with pytest.raises(ValueError, match="summaries must not be empty"):
            agent.run([], [])

    def test_raises_on_mismatched_lengths(self) -> None:
        agent = ResearchAgent(client=MagicMock())
        with pytest.raises(ValueError, match="same length"):
            agent.run(SAMPLE_SUMMARIES, ["only_one"])

    def test_run_returns_research_result(self) -> None:
        agent = ResearchAgent(client=_mock_research_client(SAMPLE_HYPOTHESES))
        result = agent.run(SAMPLE_SUMMARIES, EPISODE_IDS)
        assert isinstance(result, ResearchResult)
        assert "hypotheses" in result.output
        assert len(result.output["hypotheses"]) == 3

    def test_cost_calculation(self) -> None:
        agent = ResearchAgent(client=_mock_research_client(SAMPLE_HYPOTHESES))
        # mock returns 200 in + 150 out
        result = agent.run(SAMPLE_SUMMARIES, EPISODE_IDS)
        expected = 200 / 1e6 * 3.0 + 150 / 1e6 * 15.0
        assert abs(result.cost_usd - expected) < 1e-6

    def test_episode_ids_included_in_prompt(self) -> None:
        mock_client = _mock_research_client(SAMPLE_HYPOTHESES)
        agent = ResearchAgent(client=mock_client)
        agent.run(SAMPLE_SUMMARIES, EPISODE_IDS)
        content = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "ep1" in content and "ep2" in content

    def test_strips_markdown_fence_from_response(self) -> None:
        fenced = f"```json\n{json.dumps(SAMPLE_HYPOTHESES)}\n```"
        agent = ResearchAgent(client=_mock_research_client({}))
        agent._client.messages.create.return_value.content = [MagicMock(text=fenced)]
        result = agent.run(SAMPLE_SUMMARIES, EPISODE_IDS)
        assert "hypotheses" in result.output


# ---------------------------------------------------------------------------
# HypothesisCriticAgent
# ---------------------------------------------------------------------------

class TestHypothesisCriticAgent:
    def test_rubric_is_frozen(self) -> None:
        agent = HypothesisCriticAgent()
        assert not hasattr(agent, "system_prompt")

    def test_critic_prompt_has_all_dimensions(self) -> None:
        for dim in ("novelty", "testability", "groundedness"):
            assert dim in CRITIC_SYSTEM_PROMPT

    def test_run_returns_result(self) -> None:
        agent = HypothesisCriticAgent(client=_mock_critic_client(SAMPLE_CRITIC_OUTPUT))
        result = agent.run(SAMPLE_HYPOTHESES, SAMPLE_SUMMARIES, EPISODE_IDS)
        assert isinstance(result, HypothesisCriticResult)
        assert len(result.evaluations) == 3
        assert result.avg_overall_score == pytest.approx(3.67, abs=0.01)

    def test_overall_score_alias(self) -> None:
        agent = HypothesisCriticAgent(client=_mock_critic_client(SAMPLE_CRITIC_OUTPUT))
        result = agent.run(SAMPLE_HYPOTHESES, SAMPLE_SUMMARIES, EPISODE_IDS)
        assert result.overall_score == result.avg_overall_score

    def test_overall_score_computed_when_missing(self) -> None:
        output = {
            "evaluations": [{
                "statement": "H1",
                "scores": {"novelty": 3, "testability": 4, "groundedness": 5},
                "critiques": {"novelty": "ok", "testability": "ok", "groundedness": "ok"},
                "overall_score": None,
            }],
            "avg_overall_score": None,
        }
        agent = HypothesisCriticAgent(client=_mock_critic_client(output))
        result = agent.run({"hypotheses": []}, [], [])
        assert result.evaluations[0].overall_score == pytest.approx(4.0)

    def test_cost_calculation(self) -> None:
        agent = HypothesisCriticAgent(client=_mock_critic_client(SAMPLE_CRITIC_OUTPUT))
        result = agent.run(SAMPLE_HYPOTHESES, SAMPLE_SUMMARIES, EPISODE_IDS)
        # 300 in @ $15 + 200 out @ $75 = $0.019500
        expected = 300 / 1e6 * 15.0 + 200 / 1e6 * 75.0
        assert abs(result.cost_usd - expected) < 1e-6

    def test_uses_opus_by_default(self) -> None:
        mock_client = _mock_critic_client(SAMPLE_CRITIC_OUTPUT)
        agent = HypothesisCriticAgent(client=mock_client)
        agent.run(SAMPLE_HYPOTHESES, SAMPLE_SUMMARIES, EPISODE_IDS)
        assert mock_client.messages.create.call_args.kwargs["model"] == "claude-opus-4-6"


# ---------------------------------------------------------------------------
# ResearchLoop
# ---------------------------------------------------------------------------

TRANSCRIPT = "Lex: Tell me about agents.\nPeter: I prompted it into existence."
TRANSCRIPTS = [("ep1", TRANSCRIPT), ("ep2", TRANSCRIPT)]


def _make_research_loop(tmp_path: Path, n_iterations: int = 2) -> ResearchLoop:
    loop = ResearchLoop(
        n_iterations=n_iterations,
        n_episodes=2,
        db_path=str(tmp_path / "test.duckdb"),
        checkpoint_path=str(tmp_path / "research_ckpt.json"),
    )
    # Mock all agents
    task_mock = MagicMock()
    task_res = MagicMock()
    task_res.output = SAMPLE_SUMMARIES[0]
    task_res.cost_usd = 0.001
    task_mock.run.return_value = task_res

    research_mock = MagicMock()
    research_res = MagicMock()
    research_res.output = SAMPLE_HYPOTHESES
    research_res.cost_usd = 0.002
    research_res.latency_ms = 200
    research_mock.run.return_value = research_res
    research_mock.system_prompt = DEFAULT_SYSTEM_PROMPT

    critic_mock = MagicMock()
    critic_res = MagicMock(spec=HypothesisCriticResult)
    critic_res.evaluations = [MagicMock(scores={"novelty": 4, "testability": 4, "groundedness": 4}, critiques={}) for _ in range(3)]
    critic_res.avg_overall_score = 4.0
    critic_res.cost_usd = 0.003
    critic_res.latency_ms = 300
    critic_mock.run.return_value = critic_res

    improver_mock = MagicMock()
    improver_res = ImproverResult(
        new_prompt="Improved research prompt.",
        was_reverted=False, revert_reason=None,
        input_tokens=100, output_tokens=60, latency_ms=100, cost_usd=0.001,
        model="claude-sonnet-4-6",
    )
    improver_mock.run.return_value = improver_res

    loop._task_agent = task_mock
    loop._research_agent = research_mock
    loop._critic_agent = critic_mock
    loop._improver_agent = improver_mock
    return loop


class TestResearchLoop:
    def test_raises_on_empty_transcripts(self, tmp_path: Path) -> None:
        loop = _make_research_loop(tmp_path)
        with pytest.raises(ValueError, match="must not be empty"):
            loop.run([])

    def test_run_returns_state(self, tmp_path: Path) -> None:
        with _make_research_loop(tmp_path, n_iterations=2) as loop:
            state = loop.run(TRANSCRIPTS)
        assert isinstance(state, ResearchLoopState)
        assert len(state.iteration_summaries) == 2

    def test_task_agent_called_per_episode(self, tmp_path: Path) -> None:
        loop = _make_research_loop(tmp_path, n_iterations=1)
        with loop:
            loop.run(TRANSCRIPTS)
        # 1 iter × 2 episodes
        assert loop._task_agent.run.call_count == 2

    def test_research_agent_called_once_per_iteration(self, tmp_path: Path) -> None:
        loop = _make_research_loop(tmp_path, n_iterations=2)
        with loop:
            loop.run(TRANSCRIPTS)
        assert loop._research_agent.run.call_count == 2

    def test_prompt_updated_after_iteration(self, tmp_path: Path) -> None:
        loop = _make_research_loop(tmp_path, n_iterations=1)
        with loop:
            state = loop.run(TRANSCRIPTS)
        assert state.current_prompt == "Improved research prompt."

    def test_data_logged_to_duckdb(self, tmp_path: Path) -> None:
        import duckdb as ddb
        loop = _make_research_loop(tmp_path, n_iterations=1)
        with loop:
            loop.run(TRANSCRIPTS)
        conn = ddb.connect(str(tmp_path / "test.duckdb"), read_only=True)
        assert conn.execute("SELECT COUNT(*) FROM iterations").fetchone()[0] == 1
        conn.close()

    def test_checkpoint_deleted_on_completion(self, tmp_path: Path) -> None:
        loop = _make_research_loop(tmp_path, n_iterations=1)
        with loop:
            loop.run(TRANSCRIPTS)
        assert not (tmp_path / "research_ckpt.json").exists()
