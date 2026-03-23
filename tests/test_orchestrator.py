"""Tests for AgentLoop orchestrator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.critic_agent import CriticResult
from agents.improver_agent import ImproverResult, IterationRecord
from agents.task_agent import TaskResult
from loop.orchestrator import AgentLoop, EpisodeResult, IterationSummary, LoopState, _build_history


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

SAMPLE_TRANSCRIPT = "Lex: Tell me about OpenClaw.\nPeter: I just prompted it into existence."
SAMPLE_TASK_OUTPUT = {
    "key_topics": ["AI agents"],
    "main_argument": "OpenClaw proves autonomous agents are here.",
    "notable_quotes": ["I just prompted it into existence."],
    "actionable_takeaways": ["Build in public"],
}
SAMPLE_SCORES = {"accuracy": 4, "completeness": 4, "conciseness": 4, "actionability": 4}
SAMPLE_CRITIQUES_TEXT = {
    "accuracy": "Grounded.",
    "completeness": "Good.",
    "conciseness": "Tight.",
    "actionability": "Useful.",
}

EPISODES = [("ep1", SAMPLE_TRANSCRIPT), ("ep2", SAMPLE_TRANSCRIPT)]


def _task_result(score_hint: float = 1.0) -> TaskResult:
    return TaskResult(
        output=SAMPLE_TASK_OUTPUT,
        input_tokens=100,
        output_tokens=50,
        latency_ms=200,
        cost_usd=0.001 * score_hint,
        model="claude-sonnet-4-6",
    )


def _critic_result(score: float = 4.0) -> CriticResult:
    s = int(score)
    return CriticResult(
        scores={k: s for k in ("accuracy", "completeness", "conciseness", "actionability")},
        critiques=SAMPLE_CRITIQUES_TEXT,
        overall_score=score,
        input_tokens=150,
        output_tokens=80,
        latency_ms=300,
        cost_usd=0.003,
        model="claude-opus-4-6",
    )


def _improver_result(prompt: str = "Improved prompt. JSON: {...}") -> ImproverResult:
    return ImproverResult(
        new_prompt=prompt,
        was_reverted=False,
        revert_reason=None,
        input_tokens=100,
        output_tokens=60,
        latency_ms=150,
        cost_usd=0.001,
        model="claude-sonnet-4-6",
    )


def _make_loop(tmp_path: Path, n_iterations: int = 2, n_episodes: int = 2) -> AgentLoop:
    loop = AgentLoop(
        n_iterations=n_iterations,
        n_episodes=n_episodes,
        db_path=str(tmp_path / "test.duckdb"),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
    )
    loop._task_agent = MagicMock()
    loop._critic_agent = MagicMock()
    loop._improver_agent = MagicMock()

    loop._task_agent.run.return_value = _task_result()
    loop._critic_agent.run.return_value = _critic_result(4.0)
    loop._improver_agent.run.return_value = _improver_result()
    return loop


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAgentLoop:
    def test_raises_on_empty_transcripts(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path)
        with pytest.raises(ValueError, match="transcripts must not be empty"):
            loop.run([])

    def test_run_calls_all_agents(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, n_iterations=2, n_episodes=2)
        state = loop.run(EPISODES)

        assert loop._task_agent.run.call_count == 4   # 2 iter × 2 episodes
        assert loop._critic_agent.run.call_count == 4
        assert loop._improver_agent.run.call_count == 2

    def test_run_returns_loop_state(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, n_iterations=2, n_episodes=2)
        state = loop.run(EPISODES)

        assert isinstance(state, LoopState)
        assert len(state.iteration_summaries) == 2

    def test_avg_score_across_episodes(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, n_iterations=1, n_episodes=2)
        loop._critic_agent.run.return_value = _critic_result(3.5)

        state = loop.run(EPISODES)

        assert state.iteration_summaries[0].avg_score == pytest.approx(3.5)

    def test_episodes_capped_at_n_episodes(self, tmp_path: Path) -> None:
        many_episodes = [(f"ep{i}", SAMPLE_TRANSCRIPT) for i in range(10)]
        loop = _make_loop(tmp_path, n_iterations=1, n_episodes=3)

        state = loop.run(many_episodes)

        # 1 iteration × 3 episodes
        assert loop._task_agent.run.call_count == 3

    def test_prompt_updated_after_each_iteration(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, n_iterations=2)
        new_prompt = "Updated prompt with JSON schema."
        loop._improver_agent.run.return_value = _improver_result(prompt=new_prompt)

        state = loop.run(EPISODES)

        assert state.current_prompt == new_prompt
        assert loop._task_agent.system_prompt == new_prompt

    def test_data_logged_to_duckdb(self, tmp_path: Path) -> None:
        import duckdb as ddb
        loop = _make_loop(tmp_path, n_iterations=1, n_episodes=2)
        with loop:
            loop.run(EPISODES)

        conn = ddb.connect(str(tmp_path / "test.duckdb"), read_only=True)
        rows = conn.execute("SELECT COUNT(*) FROM iterations").fetchone()[0]
        run_rows = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        conn.close()

        assert rows == 2   # 1 iter × 2 episodes
        assert run_rows == 1

    def test_checkpoint_saved_after_each_iteration(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, n_iterations=2)
        loop.run(EPISODES)

        # Checkpoint should be deleted on successful completion
        assert not (tmp_path / "checkpoint.json").exists()

    def test_checkpoint_deleted_on_completion(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, n_iterations=1)
        loop.run(EPISODES)
        assert not (tmp_path / "checkpoint.json").exists()

    def test_resume_from_checkpoint(self, tmp_path: Path) -> None:
        # Write a checkpoint simulating 1 completed iteration
        checkpoint = {
            "run_id": "test-run-id",
            "current_prompt": "Checkpoint prompt.",
            "iteration_summaries": [
                {
                    "iteration": 0,
                    "system_prompt": "Old prompt.",
                    "avg_score": 3.5,
                    "total_cost_usd": 0.01,
                    "was_reverted": False,
                }
            ],
        }
        (tmp_path / "checkpoint.json").write_text(json.dumps(checkpoint))

        loop = _make_loop(tmp_path, n_iterations=2)
        state = loop.run(EPISODES)

        # Should only run 1 more iteration (iteration 1)
        assert loop._task_agent.run.call_count == 2  # 1 iter × 2 episodes
        assert len(state.iteration_summaries) == 2

    def test_revert_flag_propagated(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, n_iterations=1)
        revert_result = ImproverResult(
            new_prompt="reverted prompt",
            was_reverted=True,
            revert_reason="Score declined.",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            cost_usd=0.0,
            model="none",
        )
        loop._improver_agent.run.return_value = revert_result

        state = loop.run(EPISODES)

        assert state.iteration_summaries[0].was_reverted is True


class TestBuildHistory:
    def test_builds_iteration_records(self) -> None:
        ep = EpisodeResult("ep1", _task_result(), _critic_result(3.5))
        summaries = [
            IterationSummary(0, "prompt v1", [ep], avg_score=3.5, total_cost_usd=0.01),
            IterationSummary(1, "prompt v2", [ep], avg_score=4.0, total_cost_usd=0.01),
        ]
        records = _build_history(summaries)

        assert len(records) == 2
        assert records[0].score == 3.5
        assert records[1].score == 4.0
        assert records[0].system_prompt == "prompt v1"

    def test_critiques_included_in_records(self) -> None:
        ep = EpisodeResult("ep1", _task_result(), _critic_result(3.5))
        summaries = [IterationSummary(0, "p", [ep], avg_score=3.5, total_cost_usd=0.0)]
        records = _build_history(summaries)

        assert len(records[0].critiques) == 1
        assert "scores" in records[0].critiques[0]
