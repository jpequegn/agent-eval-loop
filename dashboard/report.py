"""Generate a static HTML report from a completed run stored in DuckDB."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
from jinja2 import BaseLoader, Environment

from loop.db import get_connection

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class IterationRow:
    iteration: int
    episode_id: str
    score: float
    cost_usd: float
    system_prompt: str
    task_output: dict
    critique: dict


@dataclass
class RunSummary:
    run_id: str
    start_time: str
    end_time: str | None
    episodes_tested: int
    final_avg_score: float | None
    total_cost_usd: float | None


@dataclass
class AnomalyRow:
    iteration: int
    kind: str
    message: str


def load_run(db_path: str, run_id: str) -> tuple[RunSummary, list[IterationRow], list[AnomalyRow]]:
    conn = get_connection(db_path)

    run = conn.execute(
        "SELECT id, start_time, end_time, episodes_tested, final_avg_score, total_cost_usd "
        "FROM runs WHERE id = ?",
        [run_id],
    ).fetchone()
    if not run:
        raise ValueError(f"Run {run_id!r} not found in {db_path}")

    run_summary = RunSummary(
        run_id=run[0],
        start_time=str(run[1]),
        end_time=str(run[2]) if run[2] else None,
        episodes_tested=run[3] or 0,
        final_avg_score=run[4],
        total_cost_usd=run[5],
    )

    rows = conn.execute(
        "SELECT iteration, episode_id, score, cost_usd, system_prompt, task_output, critique "
        "FROM iterations WHERE run_id = ? ORDER BY iteration, episode_id",
        [run_id],
    ).fetchall()

    iteration_rows = [
        IterationRow(
            iteration=r[0],
            episode_id=r[1],
            score=r[2] or 0.0,
            cost_usd=r[3] or 0.0,
            system_prompt=r[4] or "",
            task_output=json.loads(r[5]) if r[5] else {},
            critique=json.loads(r[6]) if r[6] else {},
        )
        for r in rows
    ]

    anomaly_rows_raw = conn.execute(
        "SELECT iteration, kind, message FROM anomalies WHERE run_id = ? ORDER BY iteration",
        [run_id],
    ).fetchall()
    anomaly_rows = [AnomalyRow(r[0], r[1], r[2]) for r in anomaly_rows_raw]

    conn.close()
    return run_summary, iteration_rows, anomaly_rows


def list_runs(db_path: str) -> list[str]:
    conn = get_connection(db_path)
    rows = conn.execute("SELECT id FROM runs ORDER BY start_time DESC").fetchall()
    conn.close()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _avg_score_chart(iteration_rows: list[IterationRow], anomalies: list[AnomalyRow]) -> str:
    """Line chart: avg score per iteration + anomaly markers."""
    by_iter: dict[int, list[float]] = {}
    for r in iteration_rows:
        by_iter.setdefault(r.iteration, []).append(r.score)

    iterations = sorted(by_iter)
    avgs = [sum(by_iter[i]) / len(by_iter[i]) for i in iterations]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[i + 1 for i in iterations],
        y=avgs,
        mode="lines+markers",
        name="Avg Score",
        line=dict(color="#1F4E8C", width=2),
        marker=dict(size=7),
    ))

    # Anomaly markers
    for a in anomalies:
        if a.iteration in by_iter:
            fig.add_vline(
                x=a.iteration + 1,
                line=dict(color="#DC3545", dash="dash", width=1),
                annotation_text=a.kind.replace("_", " "),
                annotation_position="top right",
                annotation_font=dict(size=10, color="#DC3545"),
            )

    fig.update_layout(
        title="Average Score per Iteration",
        xaxis_title="Iteration",
        yaxis_title="Score (1–5)",
        yaxis=dict(range=[0, 5.5]),
        template="plotly_dark",
        height=350,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _score_distribution_chart(iteration_rows: list[IterationRow]) -> str:
    """Box plots: score distribution per iteration."""
    by_iter: dict[int, list[float]] = {}
    for r in iteration_rows:
        by_iter.setdefault(r.iteration, []).append(r.score)

    fig = go.Figure()
    for i in sorted(by_iter):
        fig.add_trace(go.Box(
            y=by_iter[i],
            name=f"Iter {i + 1}",
            boxpoints="all",
            jitter=0.3,
            marker=dict(size=5),
        ))

    fig.update_layout(
        title="Score Distribution per Iteration",
        yaxis_title="Score (1–5)",
        yaxis=dict(range=[0, 5.5]),
        template="plotly_dark",
        height=350,
        showlegend=False,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _prompt_length_chart(iteration_rows: list[IterationRow]) -> str:
    """Line chart: system prompt character length per iteration."""
    seen: set[int] = set()
    points: list[tuple[int, int]] = []
    for r in sorted(iteration_rows, key=lambda x: x.iteration):
        if r.iteration not in seen:
            seen.add(r.iteration)
            points.append((r.iteration, len(r.system_prompt)))

    fig = go.Figure(go.Scatter(
        x=[p[0] + 1 for p in points],
        y=[p[1] for p in points],
        mode="lines+markers",
        line=dict(color="#28A745", width=2),
        marker=dict(size=7),
    ))
    fig.update_layout(
        title="System Prompt Length over Iterations",
        xaxis_title="Iteration",
        yaxis_title="Characters",
        template="plotly_dark",
        height=300,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _cumulative_cost_chart(iteration_rows: list[IterationRow]) -> str:
    """Line chart: cumulative cost over iterations."""
    by_iter: dict[int, float] = {}
    for r in iteration_rows:
        by_iter[r.iteration] = by_iter.get(r.iteration, 0.0) + r.cost_usd

    iterations = sorted(by_iter)
    cumulative = []
    total = 0.0
    for i in iterations:
        total += by_iter[i]
        cumulative.append(total)

    fig = go.Figure(go.Scatter(
        x=[i + 1 for i in iterations],
        y=cumulative,
        mode="lines+markers",
        fill="tozeroy",
        line=dict(color="#FFC107", width=2),
        marker=dict(size=7),
    ))
    fig.update_layout(
        title="Cumulative Cost ($)",
        xaxis_title="Iteration",
        yaxis_title="USD",
        template="plotly_dark",
        height=300,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _best_worst(iteration_rows: list[IterationRow]) -> tuple[IterationRow, IterationRow]:
    best = max(iteration_rows, key=lambda r: r.score)
    worst = min(iteration_rows, key=lambda r: r.score)
    return best, worst


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Run {{ run.run_id[:8] }} Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body { background:#121317; color:#E0E6F0; font-family:Inter,sans-serif; margin:0; padding:24px; }
  h1,h2,h3 { color:#E0E6F0; }
  .meta { background:#1E2130; border-radius:4px; padding:16px; margin-bottom:24px; }
  .meta span { margin-right:24px; color:#A3A9BF; }
  .meta strong { color:#E0E6F0; }
  .charts { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  .chart-full { grid-column:1/-1; }
  .chart { background:#1E2130; border-radius:4px; padding:12px; }
  .anomalies { background:#1E2130; border-radius:4px; padding:16px; margin:24px 0; }
  .anomaly { border-left:3px solid #DC3545; padding:8px 12px; margin:8px 0; background:#2A2F45; border-radius:2px; }
  .anomaly .kind { color:#DC3545; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
  .card { background:#1E2130; border-radius:4px; padding:16px; margin:8px 0; }
  .score-badge { display:inline-block; padding:2px 10px; border-radius:2px; font-weight:700; }
  .score-high { background:#28A745; color:#fff; }
  .score-low  { background:#DC3545; color:#fff; }
  pre { background:#121317; padding:12px; border-radius:4px; overflow-x:auto; font-size:12px; white-space:pre-wrap; }
  .prompt-box { background:#121317; padding:12px; border-radius:4px; font-size:12px; white-space:pre-wrap; max-height:200px; overflow-y:auto; }
  table { width:100%; border-collapse:collapse; }
  td,th { padding:8px 12px; border-bottom:1px solid #333A56; text-align:left; }
  th { color:#A3A9BF; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
</style>
</head>
<body>
<h1>Run Report</h1>
<div class="meta">
  <span><strong>Run ID:</strong> {{ run.run_id }}</span>
  <span><strong>Started:</strong> {{ run.start_time }}</span>
  {% if run.end_time %}<span><strong>Ended:</strong> {{ run.end_time }}</span>{% endif %}
  <span><strong>Episodes:</strong> {{ run.episodes_tested }}</span>
  {% if run.final_avg_score %}<span><strong>Final Score:</strong> {{ "%.2f"|format(run.final_avg_score) }}</span>{% endif %}
  {% if run.total_cost_usd %}<span><strong>Total Cost:</strong> ${{ "%.4f"|format(run.total_cost_usd) }}</span>{% endif %}
</div>

<div class="charts">
  <div class="chart chart-full">{{ avg_score_chart }}</div>
  <div class="chart">{{ score_dist_chart }}</div>
  <div class="chart">{{ prompt_len_chart }}</div>
  <div class="chart chart-full">{{ cost_chart }}</div>
</div>

{% if anomalies %}
<div class="anomalies">
  <h2>Detected Anomalies ({{ anomalies|length }})</h2>
  {% for a in anomalies %}
  <div class="anomaly">
    <div class="kind">{{ a.kind }} &mdash; Iteration {{ a.iteration + 1 }}</div>
    <div>{{ a.message }}</div>
  </div>
  {% endfor %}
</div>
{% endif %}

<h2>Best &amp; Worst Outputs</h2>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
  <div class="card">
    <h3>Best <span class="score-badge score-high">{{ "%.2f"|format(best.score) }}</span>
      &mdash; Episode {{ best.episode_id }}, Iter {{ best.iteration + 1 }}</h3>
    <pre>{{ best.task_output | tojson(indent=2) }}</pre>
  </div>
  <div class="card">
    <h3>Worst <span class="score-badge score-low">{{ "%.2f"|format(worst.score) }}</span>
      &mdash; Episode {{ worst.episode_id }}, Iter {{ worst.iteration + 1 }}</h3>
    <pre>{{ worst.task_output | tojson(indent=2) }}</pre>
  </div>
</div>

<h2>Iteration Summary</h2>
<table>
  <tr><th>Iter</th><th>Avg Score</th><th>Min</th><th>Max</th><th>Cost ($)</th><th>Prompt Length</th></tr>
  {% for row in iter_table %}
  <tr>
    <td>{{ row.iteration + 1 }}</td>
    <td>{{ "%.2f"|format(row.avg_score) }}</td>
    <td>{{ "%.2f"|format(row.min_score) }}</td>
    <td>{{ "%.2f"|format(row.max_score) }}</td>
    <td>{{ "%.4f"|format(row.cost) }}</td>
    <td>{{ row.prompt_len }}</td>
  </tr>
  {% endfor %}
</table>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


@dataclass
class IterTableRow:
    iteration: int
    avg_score: float
    min_score: float
    max_score: float
    cost: float
    prompt_len: int


def generate_report(db_path: str, run_id: str, output_dir: str = "results") -> Path:
    """Load run data and write a self-contained HTML report.

    Returns the path to the written file.
    """
    run, rows, anomalies = load_run(db_path, run_id)
    if not rows:
        raise ValueError(f"No iteration data found for run {run_id!r}")

    # Build iteration summary table
    by_iter: dict[int, list[IterationRow]] = {}
    for r in rows:
        by_iter.setdefault(r.iteration, []).append(r)

    iter_table = []
    for i in sorted(by_iter):
        iter_rows = by_iter[i]
        iter_table.append(IterTableRow(
            iteration=i,
            avg_score=sum(r.score for r in iter_rows) / len(iter_rows),
            min_score=min(r.score for r in iter_rows),
            max_score=max(r.score for r in iter_rows),
            cost=sum(r.cost_usd for r in iter_rows),
            prompt_len=len(iter_rows[0].system_prompt),
        ))

    best, worst = _best_worst(rows)

    env = Environment(loader=BaseLoader())
    env.filters["tojson"] = lambda v, indent=None: json.dumps(v, indent=indent)
    tmpl = env.from_string(_TEMPLATE)

    html = tmpl.render(
        run=run,
        avg_score_chart=_avg_score_chart(rows, anomalies),
        score_dist_chart=_score_distribution_chart(rows),
        prompt_len_chart=_prompt_length_chart(rows),
        cost_chart=_cumulative_cost_chart(rows),
        anomalies=anomalies,
        best=best,
        worst=worst,
        iter_table=iter_table,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_{run_id[:8]}_report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path
