"""Convergence analysis and experiment comparison utilities.

Used to answer the questions from issue #8:
- At what iteration does the score plateau?
- What did the prompt look like at iteration 1 vs 50 vs 100?
- Which failure modes triggered?
- Side-by-side output comparison across iterations.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loop.db import get_connection


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class IterationStats:
    iteration: int
    avg_score: float
    min_score: float
    max_score: float
    std_dev: float
    total_cost_usd: float
    system_prompt: str
    prompt_length: int


@dataclass
class PlateauResult:
    plateau_iteration: int | None   # None = never plateaued
    plateau_score: float | None
    improvement_total: float        # score gain from iter 0 to last
    improvement_rate: float         # avg gain per iteration


@dataclass
class SideBySide:
    episode_id: str
    iteration_a: int
    output_a: dict[str, Any]
    score_a: float
    iteration_b: int
    output_b: dict[str, Any]
    score_b: float


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def load_iteration_stats(db_path: str, run_id: str) -> list[IterationStats]:
    """Load per-iteration aggregated stats from DuckDB."""
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT
            iteration,
            AVG(score)      AS avg_score,
            MIN(score)      AS min_score,
            MAX(score)      AS max_score,
            STDDEV(score)   AS std_dev,
            SUM(cost_usd)   AS total_cost,
            ANY_VALUE(system_prompt) AS prompt
        FROM iterations
        WHERE run_id = ?
        GROUP BY iteration
        ORDER BY iteration
        """,
        [run_id],
    ).fetchall()
    conn.close()

    return [
        IterationStats(
            iteration=r[0],
            avg_score=r[1] or 0.0,
            min_score=r[2] or 0.0,
            max_score=r[3] or 0.0,
            std_dev=r[4] or 0.0,
            total_cost_usd=r[5] or 0.0,
            system_prompt=r[6] or "",
            prompt_length=len(r[6] or ""),
        )
        for r in rows
    ]


def load_anomalies(db_path: str, run_id: str) -> list[dict[str, Any]]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT iteration, kind, message FROM anomalies WHERE run_id = ? ORDER BY iteration",
        [run_id],
    ).fetchall()
    conn.close()
    return [{"iteration": r[0], "kind": r[1], "message": r[2]} for r in rows]


def load_episode_outputs(
    db_path: str, run_id: str, episode_id: str
) -> list[tuple[int, dict[str, Any], float]]:
    """Return (iteration, task_output, score) for a specific episode."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT iteration, task_output, score FROM iterations "
        "WHERE run_id = ? AND episode_id = ? ORDER BY iteration",
        [run_id, episode_id],
    ).fetchall()
    conn.close()
    return [(r[0], json.loads(r[1]) if r[1] else {}, r[2] or 0.0) for r in rows]


def list_episodes(db_path: str, run_id: str) -> list[str]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT DISTINCT episode_id FROM iterations WHERE run_id = ? ORDER BY episode_id",
        [run_id],
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


PLATEAU_WINDOW = 5          # look for stall over this many iterations
PLATEAU_THRESHOLD = 0.05    # less than this improvement = plateau


def detect_plateau(stats: list[IterationStats]) -> PlateauResult:
    """Find the iteration where score improvement effectively stalls."""
    if len(stats) < 2:
        return PlateauResult(
            plateau_iteration=None, plateau_score=None,
            improvement_total=0.0, improvement_rate=0.0,
        )

    scores = [s.avg_score for s in stats]
    first, last = scores[0], scores[-1]
    improvement_total = last - first
    improvement_rate = improvement_total / max(len(stats) - 1, 1)

    # Plateau: no net improvement over PLATEAU_WINDOW consecutive iterations
    plateau_iter = None
    plateau_score = None
    for i in range(len(scores) - PLATEAU_WINDOW):
        window = scores[i: i + PLATEAU_WINDOW]
        if max(window) - min(window) < PLATEAU_THRESHOLD:
            plateau_iter = stats[i].iteration
            plateau_score = statistics.mean(window)
            break

    return PlateauResult(
        plateau_iteration=plateau_iter,
        plateau_score=plateau_score,
        improvement_total=round(improvement_total, 3),
        improvement_rate=round(improvement_rate, 4),
    )


def side_by_side(
    db_path: str,
    run_id: str,
    episode_id: str,
    iter_a: int,
    iter_b: int,
) -> SideBySide:
    """Retrieve outputs at two different iterations for the same episode."""
    outputs = {it: (out, sc) for it, out, sc in load_episode_outputs(db_path, run_id, episode_id)}
    if iter_a not in outputs:
        raise ValueError(f"No output for iteration {iter_a}, episode {episode_id!r}")
    if iter_b not in outputs:
        raise ValueError(f"No output for iteration {iter_b}, episode {episode_id!r}")
    return SideBySide(
        episode_id=episode_id,
        iteration_a=iter_a, output_a=outputs[iter_a][0], score_a=outputs[iter_a][1],
        iteration_b=iter_b, output_b=outputs[iter_b][0], score_b=outputs[iter_b][1],
    )


def prompt_evolution(stats: list[IterationStats], checkpoints: list[int]) -> dict[int, str]:
    """Return the system prompt at specific iterations."""
    by_iter = {s.iteration: s.system_prompt for s in stats}
    return {cp: by_iter[cp] for cp in checkpoints if cp in by_iter}


def cumulative_cost(stats: list[IterationStats]) -> float:
    return sum(s.total_cost_usd for s in stats)


# ---------------------------------------------------------------------------
# Findings report generator
# ---------------------------------------------------------------------------


def generate_findings(db_path: str, run_id: str, output_path: str = "FINDINGS.md") -> Path:
    """Generate a FINDINGS.md from a completed run."""
    stats = load_iteration_stats(db_path, run_id)
    if not stats:
        raise ValueError(f"No data found for run {run_id!r}")

    anomalies = load_anomalies(db_path, run_id)
    plateau = detect_plateau(stats)
    total_cost = cumulative_cost(stats)
    episodes = list_episodes(db_path, run_id)
    n_iters = len(stats)

    # Prompt snapshots at 0, ~25%, ~50%, ~75%, last
    checkpoints = sorted({
        stats[0].iteration,
        stats[max(0, n_iters // 4)].iteration,
        stats[max(0, n_iters // 2)].iteration,
        stats[max(0, 3 * n_iters // 4)].iteration,
        stats[-1].iteration,
    })
    prompts = prompt_evolution(stats, checkpoints)

    # Side-by-side for first 3 episodes (first vs last iteration)
    comparisons: list[SideBySide] = []
    for ep in episodes[:3]:
        try:
            comparisons.append(side_by_side(db_path, run_id, ep, stats[0].iteration, stats[-1].iteration))
        except ValueError:
            pass

    lines = [
        "# Experiment Findings",
        "",
        f"**Run ID**: `{run_id}`  ",
        f"**Iterations completed**: {n_iters}  ",
        f"**Total cost**: ${total_cost:.4f}  ",
        f"**Score (iter 1)**: {stats[0].avg_score:.2f}  ",
        f"**Score (final)**: {stats[-1].avg_score:.2f}  ",
        f"**Total improvement**: {plateau.improvement_total:+.3f}  ",
        "",
        "---",
        "",
        "## 1. Convergence",
        "",
    ]

    if plateau.plateau_iteration is not None:
        lines += [
            f"Score **plateaued at iteration {plateau.plateau_iteration + 1}** "
            f"(avg score ≈ {plateau.plateau_score:.2f}).  ",
            f"After that point, the prompt changes produced no net improvement over "
            f"{PLATEAU_WINDOW} consecutive iterations.",
        ]
    else:
        lines += [
            "Score did **not plateau** within the run — still improving at the final iteration.",
            f"Avg improvement rate: {plateau.improvement_rate:+.4f} per iteration.",
        ]

    lines += ["", "### Score per iteration", "", "| Iter | Avg | Min | Max | Std Dev | Cost ($) |",
              "|------|-----|-----|-----|---------|----------|"]
    for s in stats:
        lines.append(
            f"| {s.iteration + 1} | {s.avg_score:.2f} | {s.min_score:.2f} | "
            f"{s.max_score:.2f} | {s.std_dev:.2f} | {s.total_cost_usd:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Prompt Evolution",
        "",
        f"Snapshots at iterations: {[c + 1 for c in checkpoints]}",
        "",
    ]
    for cp, prompt in prompts.items():
        lines += [
            f"### Iteration {cp + 1} ({len(prompt)} chars)",
            "```",
            prompt[:800] + ("…" if len(prompt) > 800 else ""),
            "```",
            "",
        ]

    lines += [
        "---",
        "",
        "## 3. Failure Modes Detected",
        "",
    ]
    if anomalies:
        for a in anomalies:
            lines.append(f"- **Iter {a['iteration'] + 1}** `{a['kind']}`: {a['message']}")
    else:
        lines.append("No anomalies detected during this run.")

    lines += [
        "",
        "---",
        "",
        "## 4. Side-by-Side: Iteration 1 vs Final",
        "",
    ]
    for cmp in comparisons:
        lines += [
            f"### Episode `{cmp.episode_id}`",
            "",
            f"| | Iteration {cmp.iteration_a + 1} (score {cmp.score_a:.2f}) "
            f"| Iteration {cmp.iteration_b + 1} (score {cmp.score_b:.2f}) |",
            "|---|---|---|",
            f"| Output | `{json.dumps(cmp.output_a)[:200]}…` | `{json.dumps(cmp.output_b)[:200]}…` |",
            "",
        ]

    lines += [
        "---",
        "",
        "## 5. Observations & Practical Limits",
        "",
        "> *Fill this in after reviewing the above data.*",
        "",
        "**What self-improvement loops do well:**",
        "- [ ] TODO",
        "",
        "**Where they break down:**",
        "- [ ] TODO",
        "",
        "**Practical limits observed:**",
        "- [ ] TODO",
        "",
    ]

    path = Path(output_path)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
