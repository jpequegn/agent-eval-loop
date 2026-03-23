"""AgentLoop: wires TaskAgent → CriticAgent → ImproverAgent into a logged iteration loop."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from agents.critic_agent import CriticAgent, CriticResult
from agents.improver_agent import ImproverAgent, IterationRecord
from agents.task_agent import DEFAULT_SYSTEM_PROMPT, TaskAgent, TaskResult
from eval.detectors import Anomaly, DetectorSuite
from loop.db import get_connection, init_schema, log_anomaly

console = Console()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    episode_id: str
    task_result: TaskResult
    critic_result: CriticResult


@dataclass
class IterationSummary:
    iteration: int
    system_prompt: str
    episode_results: list[EpisodeResult]
    avg_score: float
    total_cost_usd: float
    was_reverted: bool = False


@dataclass
class LoopState:
    """Persisted between iterations for checkpoint/resume."""
    run_id: str
    iteration_summaries: list[IterationSummary] = field(default_factory=list)
    current_prompt: str = DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


class AgentLoop:
    def __init__(
        self,
        n_iterations: int = 10,
        n_episodes: int = 5,
        task_model: str = "claude-sonnet-4-6",
        critic_model: str = "claude-opus-4-6",
        improver_model: str = "claude-sonnet-4-6",
        db_path: str = "eval_loop.duckdb",
        checkpoint_path: str = "loop_checkpoint.json",
        dry_run: bool = False,
        detect_mode: bool = False,
    ) -> None:
        self.n_iterations = n_iterations
        self.n_episodes = n_episodes
        self.db_path = db_path
        self.checkpoint_path = Path(checkpoint_path)
        self.dry_run = dry_run
        self.detect_mode = detect_mode

        self._task_agent = TaskAgent(model=task_model)
        self._critic_agent = CriticAgent(model=critic_model)
        self._improver_agent = ImproverAgent(model=improver_model)
        self._detectors = DetectorSuite.default() if detect_mode else None

        self._conn = get_connection(db_path)
        init_schema(self._conn)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AgentLoop":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def run(self, transcripts: list[tuple[str, str]]) -> LoopState:
        """Run the loop.

        Args:
            transcripts: list of (episode_id, transcript_text) pairs.
                         Sampled down to n_episodes per iteration.
        """
        if not transcripts:
            raise ValueError("transcripts must not be empty")

        episodes = transcripts[: self.n_episodes]
        state = self._load_checkpoint()

        if self.dry_run:
            return self._dry_run_estimate(episodes, state)

        run_id = state.run_id
        self._ensure_run_row(run_id)

        start_iter = len(state.iteration_summaries)
        if start_iter > 0:
            console.print(
                f"[yellow]Resuming from iteration {start_iter} "
                f"(checkpoint: {self.checkpoint_path})[/yellow]"
            )

        for i in range(start_iter, self.n_iterations):
            console.rule(f"[bold blue]Iteration {i + 1}/{self.n_iterations}")
            summary = self._run_iteration(i, state.current_prompt, episodes)
            state.iteration_summaries.append(summary)

            # Improve the prompt for next iteration
            history = _build_history(state.iteration_summaries)
            improver_result = self._improver_agent.run(history)

            if improver_result.was_reverted:
                console.print(f"[yellow]Revert: {improver_result.revert_reason}[/yellow]")
                summary.was_reverted = True

            state.current_prompt = improver_result.new_prompt
            self._task_agent.system_prompt = state.current_prompt

            self._log_iteration(run_id, summary)
            self._save_checkpoint(state)
            _print_progress_table(state.iteration_summaries)

            # Failure-mode detection
            if self._detectors:
                anomalies = self._detectors.run(
                    iteration=i,
                    recent_episode_scores=_extract_episode_scores(state.iteration_summaries),
                    avg_scores=[s.avg_score for s in state.iteration_summaries],
                    spot_check_scores=[],   # populated if spot-checks ran
                    system_prompt=state.current_prompt,
                )
                for anomaly in anomalies:
                    log_anomaly(
                        self._conn, run_id, i,
                        anomaly.kind.value, anomaly.message, anomaly.details,
                    )
                    console.print(f"[bold red]ANOMALY [{anomaly.kind.value}]: {anomaly.message}[/bold red]")
                    console.print("[yellow]Pausing — press Enter to continue or Ctrl-C to abort.[/yellow]")
                    input()

        self._close_run(run_id, state)
        console.print(f"\n[bold green]Run {run_id} complete.[/bold green]")
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_iteration(
        self,
        iteration: int,
        system_prompt: str,
        episodes: list[tuple[str, str]],
    ) -> IterationSummary:
        self._task_agent.system_prompt = system_prompt
        episode_results: list[EpisodeResult] = []

        for episode_id, transcript in episodes:
            console.print(f"  Episode [cyan]{episode_id}[/cyan]...", end=" ")

            task_result = self._task_agent.run(transcript)
            critic_result = self._critic_agent.run(transcript, task_result.output)

            episode_results.append(EpisodeResult(episode_id, task_result, critic_result))
            console.print(f"score={critic_result.overall_score:.2f} "
                          f"cost=${task_result.cost_usd + critic_result.cost_usd:.4f}")

        avg_score = sum(e.critic_result.overall_score for e in episode_results) / len(episode_results)
        total_cost = sum(
            e.task_result.cost_usd + e.critic_result.cost_usd for e in episode_results
        )

        return IterationSummary(
            iteration=iteration,
            system_prompt=system_prompt,
            episode_results=episode_results,
            avg_score=avg_score,
            total_cost_usd=total_cost,
        )

    def _log_iteration(self, run_id: str, summary: IterationSummary) -> None:
        for ep in summary.episode_results:
            self._conn.execute(
                """
                INSERT INTO iterations
                    (id, episode_id, run_id, iteration, system_prompt,
                     task_output, critique, score, cost_usd, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(uuid.uuid4()),
                    ep.episode_id,
                    run_id,
                    summary.iteration,
                    summary.system_prompt,
                    json.dumps(ep.task_result.output),
                    json.dumps({
                        "scores": ep.critic_result.scores,
                        "critiques": ep.critic_result.critiques,
                    }),
                    ep.critic_result.overall_score,
                    ep.task_result.cost_usd + ep.critic_result.cost_usd,
                    ep.task_result.latency_ms + ep.critic_result.latency_ms,
                ],
            )

    def _ensure_run_row(self, run_id: str) -> None:
        existing = self._conn.execute(
            "SELECT id FROM runs WHERE id = ?", [run_id]
        ).fetchone()
        if not existing:
            self._conn.execute(
                "INSERT INTO runs (id, start_time) VALUES (?, ?)",
                [run_id, datetime.now(timezone.utc)],
            )

    def _close_run(self, run_id: str, state: LoopState) -> None:
        summaries = state.iteration_summaries
        final_score = summaries[-1].avg_score if summaries else 0.0
        total_cost = sum(s.total_cost_usd for s in summaries)
        self._conn.execute(
            """
            UPDATE runs SET
                end_time = ?,
                episodes_tested = ?,
                final_avg_score = ?,
                total_cost_usd = ?
            WHERE id = ?
            """,
            [datetime.now(timezone.utc), self.n_episodes, final_score, total_cost, run_id],
        )
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

    def _dry_run_estimate(
        self,
        episodes: list[tuple[str, str]],
        state: LoopState,
    ) -> LoopState:
        """Run 3 iterations and extrapolate cost."""
        DRY_ITERATIONS = 3
        console.print(
            f"[bold yellow]Dry-run mode: running {DRY_ITERATIONS} iterations "
            f"then estimating full cost for {self.n_iterations} iterations.[/bold yellow]"
        )
        original_n = self.n_iterations
        self.n_iterations = DRY_ITERATIONS
        self.dry_run = False
        state = self.run(episodes)

        sampled_cost = sum(s.total_cost_usd for s in state.iteration_summaries)
        estimated_total = sampled_cost / DRY_ITERATIONS * original_n

        console.print(f"\n[bold]Cost estimate:[/bold]")
        console.print(f"  {DRY_ITERATIONS} iterations: ${sampled_cost:.4f}")
        console.print(f"  Projected {original_n} iterations: [bold]${estimated_total:.4f}[/bold]")
        return state

    def _load_checkpoint(self) -> LoopState:
        if self.checkpoint_path.exists():
            data = json.loads(self.checkpoint_path.read_text())
            state = LoopState(run_id=data["run_id"], current_prompt=data["current_prompt"])
            # Rebuild lightweight summaries (no full task/critic objects)
            for s in data.get("iteration_summaries", []):
                state.iteration_summaries.append(
                    IterationSummary(
                        iteration=s["iteration"],
                        system_prompt=s["system_prompt"],
                        episode_results=[],   # not checkpointed — just for history
                        avg_score=s["avg_score"],
                        total_cost_usd=s["total_cost_usd"],
                        was_reverted=s.get("was_reverted", False),
                    )
                )
            return state
        return LoopState(run_id=str(uuid.uuid4()))

    def _save_checkpoint(self, state: LoopState) -> None:
        data = {
            "run_id": state.run_id,
            "current_prompt": state.current_prompt,
            "iteration_summaries": [
                {
                    "iteration": s.iteration,
                    "system_prompt": s.system_prompt,
                    "avg_score": s.avg_score,
                    "total_cost_usd": s.total_cost_usd,
                    "was_reverted": s.was_reverted,
                }
                for s in state.iteration_summaries
            ],
        }
        self.checkpoint_path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_progress_table(summaries: list[IterationSummary]) -> None:
    table = Table(title="Progress", show_lines=False)
    table.add_column("Iter", justify="right", style="cyan")
    table.add_column("Avg Score", justify="right")
    table.add_column("Cost ($)", justify="right")
    table.add_column("Δ Score", justify="right")
    table.add_column("Note")

    for i, s in enumerate(summaries):
        delta = ""
        if i > 0:
            diff = s.avg_score - summaries[i - 1].avg_score
            delta = f"[green]+{diff:.2f}[/green]" if diff >= 0 else f"[red]{diff:.2f}[/red]"
        note = "[yellow]reverted[/yellow]" if s.was_reverted else ""
        table.add_row(str(s.iteration + 1), f"{s.avg_score:.2f}", f"{s.total_cost_usd:.4f}", delta, note)

    console.print(table)


def _build_history(summaries: list[IterationSummary]) -> list[IterationRecord]:
    records = []
    for s in summaries:
        critiques: list[dict[str, Any]] = []
        for ep in s.episode_results:
            critiques.append({
                "scores": ep.critic_result.scores,
                "critiques": ep.critic_result.critiques,
            })
        records.append(
            IterationRecord(
                iteration=s.iteration,
                system_prompt=s.system_prompt,
                score=s.avg_score,
                critiques=critiques,
            )
        )
    return records


def _extract_episode_scores(summaries: list[IterationSummary]) -> list[list[float]]:
    """Return per-iteration lists of per-episode critic scores."""
    return [
        [ep.critic_result.overall_score for ep in s.episode_results]
        for s in summaries
    ]
