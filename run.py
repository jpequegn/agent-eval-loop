#!/usr/bin/env python
"""CLI entrypoint: python -m run (or python run.py)"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()

P3_TRANSCRIPTS = Path("/Users/julienpequegnot/Code/parakeet-podcast-processor/data/transcripts")

app = typer.Typer(
    help="Agent evaluation loop — improves a task prompt over N iterations.",
    invoke_without_command=True,
)


@app.command()
def main(
    iterations: int = typer.Option(10, "--iterations", "-i", help="Number of improvement iterations."),
    episodes: int = typer.Option(5, "--episodes", "-e", help="Episodes to test per iteration."),
    transcripts_dir: Path = typer.Option(P3_TRANSCRIPTS, "--transcripts", help="Directory of .txt transcripts."),
    db: str = typer.Option("eval_loop.duckdb", "--db", help="DuckDB file path."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run 3 iterations and estimate total cost."),
    detect_mode: bool = typer.Option(False, "--detect-mode", help="Enable failure-mode detectors; pause on anomaly."),
    task_model: str = typer.Option("claude-sonnet-4-6", "--task-model"),
    critic_model: str = typer.Option("claude-opus-4-6", "--critic-model"),
    improver_model: str = typer.Option("claude-sonnet-4-6", "--improver-model"),
) -> None:
    from loop.orchestrator import AgentLoop

    transcript_files = sorted(transcripts_dir.glob("*.txt"))
    if not transcript_files:
        typer.echo(f"No .txt transcript files found in {transcripts_dir}", err=True)
        raise typer.Exit(1)

    transcripts = [(f.stem, f.read_text(encoding="utf-8")) for f in transcript_files]
    typer.echo(f"Loaded {len(transcripts)} transcript(s) from {transcripts_dir}")

    loop = AgentLoop(
        n_iterations=iterations,
        n_episodes=episodes,
        task_model=task_model,
        critic_model=critic_model,
        improver_model=improver_model,
        db_path=db,
        dry_run=dry_run,
        detect_mode=detect_mode,
    )
    loop.run(transcripts)


@app.command("research")
def research_cmd(
    iterations: int = typer.Option(50, "--iterations", "-i"),
    episodes: int = typer.Option(5, "--episodes", "-e"),
    transcripts_dir: Path = typer.Option(P3_TRANSCRIPTS, "--transcripts"),
    db: str = typer.Option("eval_loop.duckdb", "--db"),
    detect_mode: bool = typer.Option(False, "--detect-mode"),
    task_model: str = typer.Option("claude-sonnet-4-6", "--task-model"),
    research_model: str = typer.Option("claude-sonnet-4-6", "--research-model"),
    critic_model: str = typer.Option("claude-opus-4-6", "--critic-model"),
    improver_model: str = typer.Option("claude-sonnet-4-6", "--improver-model"),
) -> None:
    """Run the hypothesis-generation research loop."""
    from loop.research_loop import ResearchLoop

    transcript_files = sorted(transcripts_dir.glob("*.txt"))
    if not transcript_files:
        typer.echo(f"No .txt transcripts found in {transcripts_dir}", err=True)
        raise typer.Exit(1)

    transcripts = [(f.stem, f.read_text(encoding="utf-8")) for f in transcript_files]
    typer.echo(f"Loaded {len(transcripts)} transcript(s)")

    with ResearchLoop(
        n_iterations=iterations,
        n_episodes=episodes,
        task_model=task_model,
        research_model=research_model,
        critic_model=critic_model,
        improver_model=improver_model,
        db_path=db,
        detect_mode=detect_mode,
    ) as loop:
        loop.run(transcripts)


@app.command("report")
def report_cmd(
    run_id: str = typer.Argument(..., help="Run ID to generate report for. Use 'latest' for most recent."),
    db: str = typer.Option("eval_loop.duckdb", "--db", help="DuckDB file path."),
    output_dir: str = typer.Option("results", "--output-dir", "-o", help="Directory to write HTML report."),
) -> None:
    """Generate a static HTML report for a completed run."""
    from dashboard.report import generate_report, list_runs

    if run_id == "latest":
        runs = list_runs(db)
        if not runs:
            typer.echo("No runs found.", err=True)
            raise typer.Exit(1)
        run_id = runs[0]
        typer.echo(f"Using latest run: {run_id}")

    path = generate_report(db, run_id, output_dir)
    typer.echo(f"Report written to: {path}")


if __name__ == "__main__":
    app()
