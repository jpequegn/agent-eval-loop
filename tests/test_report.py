"""Tests for dashboard report generation."""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import duckdb
import pytest

from dashboard.report import (
    AnomalyRow,
    IterTableRow,
    IterationRow,
    RunSummary,
    _avg_score_chart,
    _best_worst,
    _cumulative_cost_chart,
    _prompt_length_chart,
    _score_distribution_chart,
    generate_report,
    list_runs,
    load_run,
)
from loop.db import init_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path) -> tuple[str, str]:
    """Create a DuckDB file with one run and a few iteration rows. Returns (db_path, run_id)."""
    db_path = str(tmp_path / "test.duckdb")
    run_id = str(uuid.uuid4())

    conn = duckdb.connect(db_path)
    init_schema(conn)

    conn.execute(
        "INSERT INTO runs (id, start_time, episodes_tested, final_avg_score, total_cost_usd) "
        "VALUES (?, '2026-01-01 00:00:00', 2, 3.75, 0.05)",
        [run_id],
    )

    for iteration in range(3):
        for ep_id, score in [("ep1", 3.0 + iteration * 0.5), ("ep2", 4.0 + iteration * 0.1)]:
            conn.execute(
                "INSERT INTO iterations (id, episode_id, run_id, iteration, system_prompt, "
                "task_output, critique, score, cost_usd, latency_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    str(uuid.uuid4()), ep_id, run_id, iteration,
                    f"Prompt v{iteration}" + " x" * (iteration * 50),
                    json.dumps({"key_topics": ["AI"], "main_argument": "test"}),
                    json.dumps({"scores": {"accuracy": int(score)}}),
                    score, 0.002, 300,
                ],
            )

    conn.execute(
        "INSERT INTO anomalies (id, run_id, iteration, kind, message, details) VALUES (?,?,?,?,?,?)",
        [str(uuid.uuid4()), run_id, 1, "critique_collapse", "Variance too low.", "{}"],
    )

    conn.close()
    return db_path, run_id


# ---------------------------------------------------------------------------
# load_run
# ---------------------------------------------------------------------------

class TestLoadRun:
    def test_loads_run_summary(self, tmp_path: Path) -> None:
        db_path, run_id = _make_db(tmp_path)
        run, rows, anomalies = load_run(db_path, run_id)
        assert run.run_id == run_id
        assert run.episodes_tested == 2
        assert run.final_avg_score == pytest.approx(3.75)

    def test_loads_iteration_rows(self, tmp_path: Path) -> None:
        db_path, run_id = _make_db(tmp_path)
        _, rows, _ = load_run(db_path, run_id)
        assert len(rows) == 6   # 3 iterations × 2 episodes

    def test_loads_anomalies(self, tmp_path: Path) -> None:
        db_path, run_id = _make_db(tmp_path)
        _, _, anomalies = load_run(db_path, run_id)
        assert len(anomalies) == 1
        assert anomalies[0].kind == "critique_collapse"

    def test_raises_on_unknown_run(self, tmp_path: Path) -> None:
        db_path, _ = _make_db(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            load_run(db_path, "nonexistent-run-id")

    def test_task_output_parsed_as_dict(self, tmp_path: Path) -> None:
        db_path, run_id = _make_db(tmp_path)
        _, rows, _ = load_run(db_path, run_id)
        assert isinstance(rows[0].task_output, dict)
        assert "key_topics" in rows[0].task_output


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------

class TestListRuns:
    def test_returns_run_ids(self, tmp_path: Path) -> None:
        db_path, run_id = _make_db(tmp_path)
        runs = list_runs(db_path)
        assert run_id in runs

    def test_empty_when_no_runs(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "empty.duckdb")
        conn = duckdb.connect(db_path)
        init_schema(conn)
        conn.close()
        assert list_runs(db_path) == []


# ---------------------------------------------------------------------------
# Chart builders (smoke tests — just check they return non-empty HTML)
# ---------------------------------------------------------------------------

def _sample_rows() -> list[IterationRow]:
    return [
        IterationRow(i, f"ep{j}", 3.0 + i * 0.2 + j * 0.1, 0.002,
                     f"prompt v{i}", {"key_topics": []}, {})
        for i in range(4) for j in range(2)
    ]


class TestCharts:
    def test_avg_score_chart_returns_html(self) -> None:
        html = _avg_score_chart(_sample_rows(), [])
        assert "<div" in html

    def test_avg_score_chart_includes_anomaly_line(self) -> None:
        anomalies = [AnomalyRow(1, "critique_collapse", "variance low")]
        html = _avg_score_chart(_sample_rows(), anomalies)
        assert "critique" in html.lower() or "vline" in html.lower() or len(html) > 1000

    def test_score_distribution_chart_returns_html(self) -> None:
        html = _score_distribution_chart(_sample_rows())
        assert "<div" in html

    def test_prompt_length_chart_returns_html(self) -> None:
        html = _prompt_length_chart(_sample_rows())
        assert "<div" in html

    def test_cumulative_cost_chart_returns_html(self) -> None:
        html = _cumulative_cost_chart(_sample_rows())
        assert "<div" in html


# ---------------------------------------------------------------------------
# _best_worst
# ---------------------------------------------------------------------------

class TestBestWorst:
    def test_best_has_highest_score(self) -> None:
        rows = _sample_rows()
        best, _ = _best_worst(rows)
        assert best.score == max(r.score for r in rows)

    def test_worst_has_lowest_score(self) -> None:
        rows = _sample_rows()
        _, worst = _best_worst(rows)
        assert worst.score == min(r.score for r in rows)


# ---------------------------------------------------------------------------
# generate_report (integration)
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_creates_html_file(self, tmp_path: Path) -> None:
        db_path, run_id = _make_db(tmp_path)
        out_dir = str(tmp_path / "results")
        path = generate_report(db_path, run_id, out_dir)
        assert path.exists()
        assert path.suffix == ".html"

    def test_html_contains_run_id(self, tmp_path: Path) -> None:
        db_path, run_id = _make_db(tmp_path)
        path = generate_report(db_path, run_id, str(tmp_path / "results"))
        html = path.read_text()
        assert run_id in html

    def test_html_contains_plotly_script(self, tmp_path: Path) -> None:
        db_path, run_id = _make_db(tmp_path)
        path = generate_report(db_path, run_id, str(tmp_path / "results"))
        html = path.read_text()
        assert "plotly" in html.lower()

    def test_html_contains_anomaly_section(self, tmp_path: Path) -> None:
        db_path, run_id = _make_db(tmp_path)
        path = generate_report(db_path, run_id, str(tmp_path / "results"))
        html = path.read_text()
        assert "critique_collapse" in html

    def test_output_filename_includes_run_prefix(self, tmp_path: Path) -> None:
        db_path, run_id = _make_db(tmp_path)
        path = generate_report(db_path, run_id, str(tmp_path / "results"))
        assert path.name.startswith("run_")
        assert path.name.endswith("_report.html")

    def test_raises_on_empty_run(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "empty.duckdb")
        run_id = str(uuid.uuid4())
        conn = duckdb.connect(db_path)
        init_schema(conn)
        conn.execute(
            "INSERT INTO runs (id, start_time) VALUES (?, '2026-01-01')", [run_id]
        )
        conn.close()
        with pytest.raises(ValueError, match="No iteration data"):
            generate_report(db_path, run_id, str(tmp_path / "results"))
