#!/usr/bin/env python
"""Initialize project: create DuckDB schema and verify Anthropic SDK."""

import os
import sys

from dotenv import load_dotenv
from rich.console import Console

console = Console()

load_dotenv()


def setup_database() -> None:
    from loop.db import get_connection, init_schema

    console.print("[bold blue]Setting up DuckDB schema...[/bold blue]")
    conn = get_connection()
    init_schema(conn)
    conn.close()
    console.print("[green]DuckDB schema created.[/green]")


def verify_anthropic_sdk() -> None:
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.[/red]")
        sys.exit(1)

    console.print("[bold blue]Verifying Anthropic SDK...[/bold blue]")
    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": "Hi"}],
        )
        console.print(f"[green]SDK OK. Model responded: {response.model}[/green]")
    except anthropic.AuthenticationError:
        console.print("[yellow]SDK installed. API key authentication failed — update ANTHROPIC_API_KEY in .env with a valid key.[/yellow]")
    except Exception as e:
        console.print(f"[yellow]SDK installed but verification call failed: {e}[/yellow]")


if __name__ == "__main__":
    setup_database()
    verify_anthropic_sdk()
    console.print("\n[bold green]Setup complete.[/bold green]")
