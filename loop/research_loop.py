"""ResearchLoop: hypothesis-generation variant of AgentLoop.

Pipeline per iteration:
  1. TaskAgent (summarizer) runs on each transcript → summaries
  2. ResearchAgent synthesises summaries → 3 hypotheses
  3. HypothesisCriticAgent scores hypotheses (novelty, testability, groundedness)
  4. ImproverAgent rewrites the ResearchAgent system prompt
  5. Log to DuckDB, save checkpoint

This lets us compare convergence speed vs. the summarisation task (#8).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from agents.hypothesis_critic import HypothesisCriticAgent, HypothesisCriticResult
from agents.improver_agent import ImproverAgent, IterationRecord
from agents.research_agent import ResearchAgent, ResearchResult
from agents.task_agent import TaskAgent
from loop.db import get_connection, init_schema, log_anomaly
from eval.detectors import DetectorSuite

console = Console()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ResearchIterationSummary:
    iteration: int
    system_prompt: str
    hypotheses: list[dict[str, Any]]
    avg_score: float
    total_cost_usd: float
    was_reverted: bool = False


@dataclass
class ResearchLoopState:
    run_id: str
    iteration_summaries: list[ResearchIterationSummary] = field(default_factory=list)
    current_prompt: str = ""   # set in __post_init__

    def __post_init__(self) -> None:
        from agents.research_agent import DEFAULT_SYSTEM_PROMPT
        if not self.current_prompt:
            self.current_prompt = DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# ResearchLoop
# ---------------------------------------------------------------------------


class ResearchLoop:
    """Hypothesis-generation loop.  Wraps a TaskAgent (frozen prompt, for summarisation)
    and a ResearchAgent (mutable prompt, improved by ImproverAgent).
    """

    def __init__(
        self,
        n_iterations: int = 50,
        n_episodes: int = 5,
        task_model: str = "claude-sonnet-4-6",
        research_model: str = "claude-sonnet-4-6",
        critic_model: str = "claude-opus-4-6",
        improver_model: str = "claude-sonnet-4-6",
        db_path: str = "eval_loop.duckdb",
        checkpoint_path: str = "research_checkpoint.json",
        detect_mode: bool = False,
    ) -> None:
        self.n_iterations = n_iterations
        self.n_episodes = n_episodes
        self.db_path = db_path
        self.checkpoint_path = Path(checkpoint_path)
        self.detect_mode = detect_mode

        # Summariser: fixed prompt — not improved in this loop
        self._task_agent = TaskAgent(model=task_model)
        self._research_agent = ResearchAgent(model=research_model)
        self._critic_agent = HypothesisCriticAgent(model=critic_model)
        self._improver_agent = ImproverAgent(model=improver_model)
        self._detectors = DetectorSuite.default() if detect_mode else None

        self._conn = get_connection(db_path)
        init_schema(self._conn)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ResearchLoop":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def run(self, transcripts: list[tuple[str, str]]) -> ResearchLoopState:
        """Run the research loop.

        Args:
            transcripts: (episode_id, text) pairs. All episodes are summarised
                         each iteration; summaries are fed to the ResearchAgent.
        """
        if not transcripts:
            raise ValueError("transcripts must not be empty")

        episodes = transcripts[: self.n_episodes]
        state = self._load_checkpoint()

        run_id = state.run_id
        self._ensure_run_row(run_id)

        start_iter = len(state.iteration_summaries)
        if start_iter > 0:
            console.print(f"[yellow]Resuming research loop from iteration {start_iter}[/yellow]")

        for i in range(start_iter, self.n_iterations):
            console.rule(f"[bold cyan]Research Iteration {i + 1}/{self.n_iterations}")

            # Step 1: summarise all episodes (fixed TaskAgent)
            summaries: list[dict[str, Any]] = []
            ep_ids: list[str] = []
            summary_cost = 0.0
            for ep_id, transcript in episodes:
                tr = self._task_agent.run(transcript)
                summaries.append(tr.output)
                ep_ids.append(ep_id)
                summary_cost += tr.cost_usd

            # Step 2: generate hypotheses
            self._research_agent.system_prompt = state.current_prompt
            rr: ResearchResult = self._research_agent.run(summaries, ep_ids)

            # Step 3: critique
            cr: HypothesisCriticResult = self._critic_agent.run(rr.output, summaries, ep_ids)
            total_cost = summary_cost + rr.cost_usd + cr.cost_usd

            console.print(
                f"  avg score={cr.avg_overall_score:.2f}  "
                f"cost=${total_cost:.4f}  "
                f"hypotheses={len(rr.output.get('hypotheses', []))}"
            )

            summary = ResearchIterationSummary(
                iteration=i,
                system_prompt=state.current_prompt,
                hypotheses=rr.output.get("hypotheses", []),
                avg_score=cr.avg_overall_score,
                total_cost_usd=total_cost,
            )
            state.iteration_summaries.append(summary)

            # Step 4: improve the research agent prompt
            history = _build_history(state.iteration_summaries, cr)
            improver_result = self._improver_agent.run(history)
            if improver_result.was_reverted:
                console.print(f"[yellow]Revert: {improver_result.revert_reason}[/yellow]")
                summary.was_reverted = True
            state.current_prompt = improver_result.new_prompt

            self._log_iteration(run_id, i, summary, rr, cr)
            self._save_checkpoint(state)
            _print_progress_table(state.iteration_summaries)

            # Failure detection
            if self._detectors:
                anomalies = self._detectors.run(
                    iteration=i,
                    recent_episode_scores=[[cr.avg_overall_score]] * max(1, len(state.iteration_summaries)),
                    avg_scores=[s.avg_score for s in state.iteration_summaries],
                    spot_check_scores=[],
                    system_prompt=state.current_prompt,
                )
                for anomaly in anomalies:
                    log_anomaly(self._conn, run_id, i, anomaly.kind.value, anomaly.message, anomaly.details)
                    console.print(f"[bold red]ANOMALY: {anomaly.message}[/bold red]")
                    console.print("[yellow]Press Enter to continue or Ctrl-C to abort.[/yellow]")
                    input()

        self._close_run(run_id, state)
        console.print(f"\n[bold green]Research run {run_id} complete.[/bold green]")
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_iteration(
        self,
        run_id: str,
        iteration: int,
        summary: ResearchIterationSummary,
        rr: ResearchResult,
        cr: HypothesisCriticResult,
    ) -> None:
        self._conn.execute(
            """INSERT INTO iterations
               (id, episode_id, run_id, iteration, system_prompt,
                task_output, critique, score, cost_usd, latency_ms)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            [
                str(uuid.uuid4()),
                "research_batch",
                run_id,
                iteration,
                summary.system_prompt,
                json.dumps(rr.output),
                json.dumps([
                    {"scores": e.scores, "critiques": e.critiques}
                    for e in cr.evaluations
                ]),
                cr.avg_overall_score,
                summary.total_cost_usd,
                rr.latency_ms + cr.latency_ms,
            ],
        )

    def _ensure_run_row(self, run_id: str) -> None:
        if not self._conn.execute("SELECT id FROM runs WHERE id=?", [run_id]).fetchone():
            self._conn.execute(
                "INSERT INTO runs (id, start_time) VALUES (?,?)",
                [run_id, datetime.now(timezone.utc)],
            )

    def _close_run(self, run_id: str, state: ResearchLoopState) -> None:
        summaries = state.iteration_summaries
        self._conn.execute(
            "UPDATE runs SET end_time=?, final_avg_score=?, total_cost_usd=? WHERE id=?",
            [
                datetime.now(timezone.utc),
                summaries[-1].avg_score if summaries else 0.0,
                sum(s.total_cost_usd for s in summaries),
                run_id,
            ],
        )
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

    def _load_checkpoint(self) -> ResearchLoopState:
        if self.checkpoint_path.exists():
            data = json.loads(self.checkpoint_path.read_text())
            state = ResearchLoopState(run_id=data["run_id"], current_prompt=data["current_prompt"])
            for s in data.get("iteration_summaries", []):
                state.iteration_summaries.append(ResearchIterationSummary(
                    iteration=s["iteration"],
                    system_prompt=s["system_prompt"],
                    hypotheses=s.get("hypotheses", []),
                    avg_score=s["avg_score"],
                    total_cost_usd=s["total_cost_usd"],
                    was_reverted=s.get("was_reverted", False),
                ))
            return state
        return ResearchLoopState(run_id=str(uuid.uuid4()))

    def _save_checkpoint(self, state: ResearchLoopState) -> None:
        self.checkpoint_path.write_text(json.dumps({
            "run_id": state.run_id,
            "current_prompt": state.current_prompt,
            "iteration_summaries": [
                {
                    "iteration": s.iteration,
                    "system_prompt": s.system_prompt,
                    "hypotheses": s.hypotheses,
                    "avg_score": s.avg_score,
                    "total_cost_usd": s.total_cost_usd,
                    "was_reverted": s.was_reverted,
                }
                for s in state.iteration_summaries
            ],
        }, indent=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_history(
    summaries: list[ResearchIterationSummary],
    latest_critic: HypothesisCriticResult,
) -> list[IterationRecord]:
    records = []
    for s in summaries[:-1]:
        records.append(IterationRecord(iteration=s.iteration, system_prompt=s.system_prompt, score=s.avg_score))
    # Include critique detail for the latest iteration
    records.append(IterationRecord(
        iteration=summaries[-1].iteration,
        system_prompt=summaries[-1].system_prompt,
        score=summaries[-1].avg_score,
        critiques=[
            {"scores": e.scores, "critiques": e.critiques}
            for e in latest_critic.evaluations
        ],
    ))
    return records


def _print_progress_table(summaries: list[ResearchIterationSummary]) -> None:
    table = Table(title="Research Progress", show_lines=False)
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
