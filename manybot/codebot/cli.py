#!/usr/bin/env python3
"""
Codebot CLI - AI-Powered Development Assistant

Foundation for AI-powered development assistance through a unified CLI
with multiple execution modes, built on PydanticAI.
"""

import asyncio
import logging
import sys
from typing import List, Optional

import typer
from typing_extensions import Annotated
from pathlib import Path

from .workflow import CommandOutput, ContextManager, ExecutionContext
from .workflows.summarize import SummarizeCommand, SummaryCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("codebot")

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
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


def handle_result(result, dry_run: bool = False):
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


@app.command()
def summarize(
    paths: List[str] = typer.Argument(default=[], help="Files or directories to analyze"),
    max_tokens: int = typer.Option(10000, "--max_tokens", help="Maximum tokens for summary"),
    role: Optional[str] = typer.Option(None, "--role", help="Role file to use (e.g., 'roles/architect.md')"),
    task: Optional[str] = typer.Option(None, "--task", help="Task file to use (e.g., 'tasks/refactor.md')"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without making changes"),
):
    """Generate AI-powered code summaries"""

    # Set defaults for summarize command
    role_file = role or "roles/engineer.md"
    task_file = task or "tasks/summarize.md"

    # Gather both prompt and execution contexts
    context_manager = ContextManager()
    prompt_context, execution_context = context_manager.gather_context(
        paths,
        role_file=role_file,
        task_file=task_file,
        dry_run=dry_run
    )

    # Use cache for efficiency
    cache = SummaryCache()

    try:
        import asyncio

        # Run async function
        result = asyncio.run(cache.get_or_create_summary(prompt_context, execution_context, max_tokens))
        handle_result(result, dry_run)

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None


@app.command()
def debug_tests(
    paths: Annotated[Optional[List[str]], typer.Argument(help="Paths to analyze")] = None,
):
    """Debug failing tests by analyzing test files and related code.

    Placeholder for future implementation.
    """
    typer.echo("🔧 Debug tests command coming soon!")
    if paths:
        typer.echo(f"Would analyze: {', '.join(paths)}")


@app.command()
def refactor(
    paths: Annotated[Optional[List[str]], typer.Argument(help="Paths to refactor")] = None,
    interactive: Annotated[bool, typer.Option("--interactive", "-i", help="Interactive mode")] = False,
):
    """Refactor code with AI assistance.

    Placeholder for future implementation.
    """
    typer.echo("🔧 Refactor command coming soon!")
    if paths:
        typer.echo(f"Would refactor: {', '.join(paths)}")
    if interactive:
        typer.echo("Interactive mode would be enabled")


@app.command()
def implement(
    paths: Annotated[Optional[List[str]], typer.Argument(help="Paths to implement in")] = None,
    pipeline: Annotated[bool, typer.Option("--pipeline", "-p", help="Pipeline mode with structured output")] = False,
):
    """Implement features with AI assistance.

    Placeholder for future implementation.
    """
    typer.echo("🔧 Implement command coming soon!")
    if paths:
        typer.echo(f"Would implement in: {', '.join(paths)}")
    if pipeline:
        typer.echo("Pipeline mode would be enabled")


def cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    app()
