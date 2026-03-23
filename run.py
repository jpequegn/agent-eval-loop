#!/usr/bin/env python
"""CLI entrypoint: python -m run (or python run.py)"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()

P3_TRANSCRIPTS = Path("/Users/julienpequegnot/Code/parakeet-podcast-processor/data/transcripts")

app = typer.Typer(help="Agent evaluation loop — improves a task prompt over N iterations.")


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


if __name__ == "__main__":
    app()
