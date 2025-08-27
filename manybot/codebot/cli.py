#!/usr/bin/env python3
"""
Codebot CLI - AI-Powered Development Assistant

Foundation for AI-powered development assistance through a unified CLI
with multiple execution modes, built on PydanticAI.
"""

import asyncio
from typing import List, Optional

import pyperclip
import typer
from typing_extensions import Annotated

from .logging import get_logger, setup_logging
from .workflow import ContextManager
from .workflows.summarize import SummarizeCommand

logger = get_logger(__name__)

# Create the main Typer app
app = typer.Typer(
    name="codebot",
    help="AI-Powered Development Assistant",
    no_args_is_help=True,
)


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        typer.echo("codebot 0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = False,
    version: Annotated[
        Optional[bool], typer.Option("--version", callback=version_callback, help="Show version")
    ] = None,
):
    """Codebot - AI-Powered Development Assistant"""
    setup_logging(verbose)


def handle_result(result, dry_run: bool = False, copy: bool = False):
    """Handle command output consistently"""
    if dry_run:
        typer.echo("DRY RUN - Changes would be:")
        for change in result.file_changes:
            typer.echo(change.preview())
    else:
        # Apply file changes
        for change in result.file_changes:
            change.apply()
            typer.echo(f"✓ {change.operation.upper()}: {change.filepath}")

    # Always show summary
    if result.summary:
        typer.echo(f"\n{result.summary}")

        # Copy summary to clipboard if requested
        if copy:
            pyperclip.copy(result.summary)
            typer.echo("Summary copied to clipboard")


@app.command()
def summarize(
    paths: List[str] = typer.Argument(help="Files or directories to analyze"),
    max_tokens: int = typer.Option(10000, "--max_tokens", help="Maximum tokens for summary"),
    role: Optional[str] = typer.Option(
        None, "--role", help="Role name (e.g., 'engineer') or path (e.g., 'roles/architect.md')"
    ),
    task: Optional[str] = typer.Option(
        None, "--task", help="Task name (e.g., 'summarize') or path (e.g., 'tasks/refactor.md')"
    ),
    copy: bool = typer.Option(False, "--copy", help="Copy summary to clipboard"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without making changes"),
):
    """Generate AI-powered code summaries"""

    # Set defaults for summarize command (simple names will be resolved by ContextManager)
    role_name = role or "engineer"
    task_name = task or "summarize"

    # Gather both prompt and execution contexts
    context_manager = ContextManager()
    prompt_context, execution_context = context_manager.gather_context(
        paths, role=role_name, task=task_name, dry_run=dry_run
    )

    summarizer = SummarizeCommand()

    # Run async function with current signature
    result = asyncio.run(summarizer.execute(prompt_context, execution_context, token_limit=max_tokens))
    handle_result(result, dry_run, copy)


def cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    app()
