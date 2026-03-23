"""Tests for eval/analysis.py — convergence analysis and findings generation."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from eval.analysis import (
    IterationStats,
    cumulative_cost,
    detect_plateau,
    generate_findings,
    list_episodes,
    load_anomalies,
    load_episode_outputs,
    load_iteration_stats,
    prompt_evolution,
    side_by_side,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_stats(scores: list[float]) -> list[IterationStats]:
    return [
        IterationStats(
            iteration=i,
            avg_score=s,
            min_score=s - 0.1,
            max_score=s + 0.1,
            std_dev=0.05,
            total_cost_usd=0.01,
            system_prompt=f"prompt at iter {i}",
            prompt_length=len(f"prompt at iter {i}"),
        )
        for i, s in enumerate(scores)
    ]


def _make_db(tmp_path: Path) -> str:
    """Create an in-memory-style DuckDB file with test data."""
    db_path = str(tmp_path / "test.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE runs (
            run_id VARCHAR PRIMARY KEY,
            started_at TIMESTAMP DEFAULT now(),
            task_model VARCHAR,
            critic_model VARCHAR,
            improver_model VARCHAR,
            n_iterations INTEGER,
            n_episodes INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE iterations (
            run_id VARCHAR,
            iteration INTEGER,
            episode_id VARCHAR,
            system_prompt VARCHAR,
            task_output VARCHAR,
            score DOUBLE,
            cost_usd DOUBLE,
            PRIMARY KEY (run_id, iteration, episode_id)
        )
    """)
    conn.execute("""
        CREATE TABLE anomalies (
            run_id VARCHAR,
            iteration INTEGER,
            kind VARCHAR,
            message VARCHAR,
            details VARCHAR
        )
    """)

    run_id = "test-run-001"
    conn.execute(
        "INSERT INTO runs VALUES (?, now(), 'sonnet', 'opus', 'sonnet', 5, 3)",
        [run_id],
    )

    episodes = ["ep-a", "ep-b", "ep-c"]
    base_scores = [0.5, 0.6, 0.65, 0.70, 0.71]

    for iteration, base in enumerate(base_scores):
        prompt = f"System prompt v{iteration}: do the thing well"
        for ep_idx, ep in enumerate(episodes):
            score = base + ep_idx * 0.01
            output = json.dumps({"summary": f"output {ep} iter {iteration}"})
            conn.execute(
                "INSERT INTO iterations VALUES (?, ?, ?, ?, ?, ?, ?)",
                [run_id, iteration, ep, prompt, output, score, 0.005],
            )

    conn.execute(
        "INSERT INTO anomalies VALUES (?, ?, ?, ?, ?)",
        [run_id, 2, "score_gaming", "Possible score gaming detected", "{}"],
    )

    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# load_iteration_stats
# ---------------------------------------------------------------------------


class TestLoadIterationStats:
    def test_returns_one_stat_per_iteration(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        stats = load_iteration_stats(db_path, "test-run-001")
        assert len(stats) == 5

    def test_sorted_by_iteration(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        stats = load_iteration_stats(db_path, "test-run-001")
        assert [s.iteration for s in stats] == [0, 1, 2, 3, 4]

    def test_avg_score_computed(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        stats = load_iteration_stats(db_path, "test-run-001")
        # iteration 0: scores 0.50, 0.51, 0.52 → avg ≈ 0.51
        assert abs(stats[0].avg_score - 0.51) < 0.01

    def test_system_prompt_captured(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        stats = load_iteration_stats(db_path, "test-run-001")
        assert "System prompt v0" in stats[0].system_prompt

    def test_prompt_length_matches_prompt(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        stats = load_iteration_stats(db_path, "test-run-001")
        for s in stats:
            assert s.prompt_length == len(s.system_prompt)

    def test_empty_for_unknown_run(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        stats = load_iteration_stats(db_path, "no-such-run")
        assert stats == []


# ---------------------------------------------------------------------------
# load_anomalies
# ---------------------------------------------------------------------------


class TestLoadAnomalies:
    def test_returns_anomaly(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        anomalies = load_anomalies(db_path, "test-run-001")
        assert len(anomalies) == 1
        assert anomalies[0]["kind"] == "score_gaming"
        assert anomalies[0]["iteration"] == 2

    def test_empty_for_clean_run(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        # insert a second run with no anomalies
        conn = duckdb.connect(db_path)
        conn.execute("INSERT INTO runs VALUES ('clean-run', now(), 's', 'o', 's', 5, 3)")
        conn.close()
        assert load_anomalies(db_path, "clean-run") == []


# ---------------------------------------------------------------------------
# load_episode_outputs
# ---------------------------------------------------------------------------


class TestLoadEpisodeOutputs:
    def test_returns_one_row_per_iteration(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        rows = load_episode_outputs(db_path, "test-run-001", "ep-a")
        assert len(rows) == 5

    def test_row_structure(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        rows = load_episode_outputs(db_path, "test-run-001", "ep-a")
        iteration, output, score = rows[0]
        assert isinstance(iteration, int)
        assert isinstance(output, dict)
        assert isinstance(score, float)

    def test_output_deserialized(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        rows = load_episode_outputs(db_path, "test-run-001", "ep-a")
        assert "summary" in rows[0][1]


# ---------------------------------------------------------------------------
# list_episodes
# ---------------------------------------------------------------------------


class TestListEpisodes:
    def test_returns_all_episodes(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        episodes = list_episodes(db_path, "test-run-001")
        assert sorted(episodes) == ["ep-a", "ep-b", "ep-c"]

    def test_returns_empty_for_unknown_run(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        assert list_episodes(db_path, "ghost") == []


# ---------------------------------------------------------------------------
# detect_plateau
# ---------------------------------------------------------------------------


class TestDetectPlateau:
    def test_detects_plateau_when_scores_stall(self) -> None:
        # First 3 improve, then plateau for 5+ iterations
        scores = [0.3, 0.5, 0.7, 0.72, 0.72, 0.73, 0.71, 0.72, 0.72, 0.72]
        stats = _make_stats(scores)
        result = detect_plateau(stats)
        assert result.plateau_iteration is not None

    def test_no_plateau_when_still_improving(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        stats = _make_stats(scores)
        result = detect_plateau(stats)
        assert result.plateau_iteration is None

    def test_plateau_score_is_mean_of_window(self) -> None:
        # Need enough flat values so the sliding window lands fully in the flat region.
        # PLATEAU_WINDOW=5: range(len-5) must reach an all-flat window.
        # [0.3, 0.5, 0.6] + 7 flat → len=10, range(5) checks [0..4]; index 3 is fully flat.
        flat = [0.70] * 7
        scores = [0.3, 0.5, 0.6] + flat
        stats = _make_stats(scores)
        result = detect_plateau(stats)
        assert result.plateau_score is not None
        assert abs(result.plateau_score - 0.70) < 0.01

    def test_improvement_total_computed(self) -> None:
        scores = [0.4, 0.6, 0.8]
        stats = _make_stats(scores)
        result = detect_plateau(stats)
        assert abs(result.improvement_total - 0.4) < 0.001

    def test_improvement_rate_is_per_iteration(self) -> None:
        scores = [0.0, 0.5, 1.0]
        stats = _make_stats(scores)
        result = detect_plateau(stats)
        # total=1.0, iters-1=2 → rate=0.5
        assert abs(result.improvement_rate - 0.5) < 0.001

    def test_single_stat_returns_no_plateau(self) -> None:
        result = detect_plateau(_make_stats([0.5]))
        assert result.plateau_iteration is None
        assert result.plateau_score is None

    def test_empty_stats_returns_no_plateau(self) -> None:
        result = detect_plateau([])
        assert result.plateau_iteration is None


# ---------------------------------------------------------------------------
# side_by_side
# ---------------------------------------------------------------------------


class TestSideBySide:
    def test_returns_correct_iterations(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = side_by_side(db_path, "test-run-001", "ep-a", 0, 4)
        assert result.iteration_a == 0
        assert result.iteration_b == 4

    def test_scores_differ_between_iterations(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = side_by_side(db_path, "test-run-001", "ep-a", 0, 4)
        assert result.score_a != result.score_b

    def test_raises_on_missing_iteration_a(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        with pytest.raises(ValueError, match="No output for iteration 99"):
            side_by_side(db_path, "test-run-001", "ep-a", 99, 4)

    def test_raises_on_missing_iteration_b(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        with pytest.raises(ValueError, match="No output for iteration 99"):
            side_by_side(db_path, "test-run-001", "ep-a", 0, 99)

    def test_output_a_is_dict(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        result = side_by_side(db_path, "test-run-001", "ep-a", 0, 4)
        assert isinstance(result.output_a, dict)
        assert isinstance(result.output_b, dict)


# ---------------------------------------------------------------------------
# prompt_evolution
# ---------------------------------------------------------------------------


class TestPromptEvolution:
    def test_returns_prompts_at_checkpoints(self) -> None:
        stats = _make_stats([0.5] * 5)
        for i, s in enumerate(stats):
            s.system_prompt = f"prompt-{i}"
        result = prompt_evolution(stats, [0, 2, 4])
        assert result[0] == "prompt-0"
        assert result[2] == "prompt-2"
        assert result[4] == "prompt-4"

    def test_skips_missing_checkpoints(self) -> None:
        stats = _make_stats([0.5] * 3)
        result = prompt_evolution(stats, [0, 99])
        assert 99 not in result
        assert 0 in result

    def test_empty_checkpoints_returns_empty(self) -> None:
        stats = _make_stats([0.5, 0.6])
        assert prompt_evolution(stats, []) == {}


# ---------------------------------------------------------------------------
# cumulative_cost
# ---------------------------------------------------------------------------


class TestCumulativeCost:
    def test_sums_all_costs(self) -> None:
        stats = _make_stats([0.5, 0.6, 0.7])
        for s in stats:
            s.total_cost_usd = 0.10
        assert abs(cumulative_cost(stats) - 0.30) < 0.001

    def test_zero_cost(self) -> None:
        stats = _make_stats([0.5])
        stats[0].total_cost_usd = 0.0
        assert cumulative_cost(stats) == 0.0


# ---------------------------------------------------------------------------
# generate_findings
# ---------------------------------------------------------------------------


class TestGenerateFindings:
    def test_creates_file(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        output_path = str(tmp_path / "FINDINGS.md")
        path = generate_findings(db_path, "test-run-001", output_path)
        assert path.exists()

    def test_file_contains_run_id(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        output_path = str(tmp_path / "FINDINGS.md")
        path = generate_findings(db_path, "test-run-001", output_path)
        content = path.read_text()
        assert "test-run-001" in content

    def test_file_contains_score_table(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        output_path = str(tmp_path / "FINDINGS.md")
        path = generate_findings(db_path, "test-run-001", output_path)
        content = path.read_text()
        assert "| Iter |" in content

    def test_file_contains_prompt_evolution(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        output_path = str(tmp_path / "FINDINGS.md")
        path = generate_findings(db_path, "test-run-001", output_path)
        content = path.read_text()
        assert "Prompt Evolution" in content

    def test_file_contains_anomalies(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        output_path = str(tmp_path / "FINDINGS.md")
        path = generate_findings(db_path, "test-run-001", output_path)
        content = path.read_text()
        assert "score_gaming" in content

    def test_file_contains_side_by_side_section(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        output_path = str(tmp_path / "FINDINGS.md")
        path = generate_findings(db_path, "test-run-001", output_path)
        content = path.read_text()
        assert "Side-by-Side" in content

    def test_raises_on_unknown_run(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        output_path = str(tmp_path / "FINDINGS.md")
        with pytest.raises(ValueError, match="No data found"):
            generate_findings(db_path, "ghost-run", output_path)

    def test_default_output_path_used_when_not_specified(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        # Patch Path.write_text to avoid writing to cwd
        output_path = str(tmp_path / "FINDINGS.md")
        path = generate_findings(db_path, "test-run-001", output_path)
        assert path.name == "FINDINGS.md"
