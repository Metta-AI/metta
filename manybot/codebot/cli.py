#!/usr/bin/env python3
"""
Codebot CLI - AI-Powered Development Assistant

Foundation for AI-powered development assistance through a unified CLI
with multiple execution modes, built on PydanticAI.
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional
import logging

import typer
from typing_extensions import Annotated

from .workflow import ContextManager, CommandOutput
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
    version: Annotated[Optional[bool], typer.Option("--version", callback=version_callback, help="Show version")] = None,
):
    """Codebot - AI-Powered Development Assistant"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@app.command()
def summarize(
    paths: Annotated[Optional[List[str]], typer.Argument(help="Paths to analyze")] = None,
    token_limit: Annotated[int, typer.Option("--token-limit", "-t", help="Token limit for summary")] = 2000,
    output: Annotated[Optional[str], typer.Option("--output", "-o", help="Output file path")] = None,
    mode: Annotated[str, typer.Option("--mode", "-m", help="Execution mode")] = "oneshot",
    cached: Annotated[bool, typer.Option("--cached/--no-cache", help="Use cached results when available")] = True,
):
    """Create a structured summary of code optimized for AI consumption.

    The summarizer analyzes code to identify key components, patterns, and
    dependencies, producing a token-constrained summary perfect for providing
    context to other AI operations.

    Examples:
        codebot summarize                    # Summarize current directory
        codebot summarize src/ tests/        # Summarize specific paths
        codebot summarize -t 5000            # Higher token limit
        codebot summarize --no-cache         # Force regeneration
    """
    try:
        path_list = paths if paths else ["."]

        async def run_summarize():
            if cached:
                # Try cache first
                cache = SummaryCache()
                try:
                    summary_content = await cache.get_or_create_summary(path_list, token_limit)
                    typer.echo("✅ Summary generated successfully!")
                    typer.echo(f"📁 Output: {output or '.codebot/summaries/latest.md'}")
                    return
                except Exception as e:
                    logger.warning(f"Cache failed, running fresh: {e}")

            # Run fresh summarization
            context_manager = ContextManager()
            context = context_manager.gather_context(path_list)

            logger.info(f"Analyzing {len(context.files)} files ({context.token_count:,} input tokens)")

            # Create command instance following README pattern
            summarize_cmd = SummarizeCommand()
            result = await summarize_cmd.execute(context, token_limit=token_limit)

            # Handle custom output path if specified
            if output and result.file_changes:
                # Update the file path
                result.file_changes[0].filepath = output
                result.file_changes[0].apply()

            # Output results
            typer.echo("✅ " + result.summary)

            if result.file_changes:
                for change in result.file_changes:
                    typer.echo(f"📁 Created: {change.filepath}")

            # Show metadata
            if result.metadata:
                metadata = result.metadata
                typer.echo(f"📊 Components: {metadata.get('component_count', 'N/A')}")
                typer.echo(f"📦 Dependencies: {metadata.get('dependency_count', 'N/A')}")
                typer.echo(f"🔍 Patterns: {metadata.get('pattern_count', 'N/A')}")
                typer.echo(f"🎯 Token limit: {token_limit}")

        asyncio.run(run_summarize())

    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Detailed error:")
        raise typer.Exit(1)


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
