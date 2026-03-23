#!/usr/bin/env python
"""Smoke-test TaskAgent on available transcripts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

P3_TRANSCRIPTS = Path("/Users/julienpequegnot/Code/parakeet-podcast-processor/data/transcripts")
console = Console()


def main() -> None:
    from agents.task_agent import TaskAgent

    agent = TaskAgent()

    transcript_files = sorted(P3_TRANSCRIPTS.glob("*.txt"))
    if not transcript_files:
        console.print("[red]No transcript files found in P3 transcripts directory.[/red]")
        return

    # Test on up to 5 episodes
    files_to_test = transcript_files[:5]
    console.print(f"[bold blue]Testing TaskAgent on {len(files_to_test)} episode(s)...[/bold blue]\n")

    table = Table(title="TaskAgent Results", show_lines=True)
    table.add_column("File", style="cyan", max_width=30)
    table.add_column("Topics", max_width=40)
    table.add_column("Main Argument", max_width=50)
    table.add_column("Tokens", justify="right")
    table.add_column("Cost ($)", justify="right")
    table.add_column("Latency (ms)", justify="right")

    for path in files_to_test:
        transcript = path.read_text(encoding="utf-8")
        console.print(f"Processing [bold]{path.name}[/bold] ({len(transcript):,} chars)...")

        result = agent.run(transcript)

        topics = ", ".join(result.output.get("key_topics", [])[:3])
        main_arg = result.output.get("main_argument", "")[:100]
        total_tokens = result.input_tokens + result.output_tokens

        table.add_row(
            path.name,
            topics,
            main_arg,
            str(total_tokens),
            f"{result.cost_usd:.4f}",
            str(result.latency_ms),
        )

        console.print(f"  [green]Done[/green] — {total_tokens} tokens, ${result.cost_usd:.4f}, {result.latency_ms}ms")
        console.print(f"  Full output:\n{json.dumps(result.output, indent=2)}\n")

    console.print(table)


if __name__ == "__main__":
    main()
