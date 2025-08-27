"""
CLI interface for codebot.

Provides command-line access to AI-powered development assistance tools.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import click

from .commands import run_summarize_command

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("codebot.cli")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """
    Codebot: AI-powered development assistance through a unified CLI.

    Provides structured AI operations for code analysis, testing, refactoring, and more.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        # Also ensure codebot module loggers are set to INFO
        logging.getLogger("codebot").setLevel(logging.INFO)
        logger.debug("Verbose logging enabled")


@main.command()
@click.argument("paths", nargs=-1, type=str)
@click.option("-t", "--token-limit", type=int, default=2000, help="Maximum tokens for the summary (default: 2000)")
@click.option(
    "-o", "--output", type=str, help="Output file path (default: <project-root>/.codebot/summaries/{cache-key}.md)"
)
@click.option("--apply", is_flag=True, help="Apply changes immediately (write summary file)")
@click.option("--preview", is_flag=True, help="Preview changes without applying them")
@click.option("--no-cache", is_flag=True, help="Bypass cache and generate fresh summary")
def summarize(
    paths: Tuple[str, ...], token_limit: int, output: Optional[str], apply: bool, preview: bool, no_cache: bool
) -> None:
    """
    Generate an AI-powered summary of the codebase.

    Analyzes the provided files and creates a structured summary optimized
    for AI consumption, including key components, patterns, and architecture.

    Examples:

        # Summarize current directory
        codebot summarize

        # Summarize specific files
        codebot summarize src/main.py src/utils.py

        # Summarize with custom token limit
        codebot summarize --token-limit 4000 src/

        # Preview the summary without writing files
        codebot summarize --preview src/
    """

    # Convert paths to list, use current directory if none provided
    path_list = list(paths) if paths else ["."]

    try:
        # Run the summarize command
        result = asyncio.run(
            run_summarize_command(paths=path_list, token_limit=token_limit, working_dir=Path.cwd(), no_cache=no_cache)
        )

        if not result.success:
            click.echo(f"Error: {result.summary}", err=True)
            sys.exit(1)

        # Show summary
        click.echo(f"✓ {result.summary}")

        # Show metadata if available
        if result.metadata:
            meta = result.metadata
            if "input_token_count" in meta:
                click.echo(f"  Input: ~{meta['input_token_count']:,} tokens", err=True)
            if "output_file" in meta:
                click.echo(f"  Output: {meta['output_file']}", err=True)
            if "cached" in meta and meta["cached"]:
                click.echo(f"  Cached: ✓ (key: {meta['cache_key'][:8]}...)", err=True)
            elif "cached" in meta and not meta["cached"]:
                click.echo(f"  Fresh: ✓ (key: {meta['cache_key'][:8]}...)", err=True)

        # Handle output options
        if preview:
            # Show preview of changes
            preview_text = result.preview_changes()
            click.echo("\nPreview of changes:")
            click.echo(preview_text)
        elif apply or not (preview):
            # Apply changes by default unless preview is requested
            modified_files = result.apply_changes()
            if modified_files:
                click.echo(f"  Written: {', '.join(modified_files)}", err=True)

        # Custom output path handling
        if output and result.file_changes:
            # Move the generated file to custom output path
            import shutil

            source_path = result.file_changes[0].filepath
            if Path(source_path).exists():
                shutil.move(source_path, output)
                click.echo(f"  Moved to: {output}", err=True)

    except Exception as e:
        logger.error(f"Error in summarize command: {e}")
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command()
def version() -> None:
    """Show version information."""
    from . import __version__

    click.echo(f"codebot {__version__}")


@main.command()
@click.argument("paths", nargs=-1, type=str)
def context(paths: Tuple[str, ...]) -> None:
    """
    Show the context that would be sent to AI commands.

    This is useful for debugging and understanding what information
    the AI will receive when analyzing your code.
    """

    # Import here to avoid circular imports
    try:
        # Add the codeclip directory to the path
        import os

        current_dir = os.path.dirname(__file__)
        codeclip_path = os.path.join(current_dir, "..", "codeclip", "codeclip")

        if codeclip_path not in sys.path:
            sys.path.insert(0, codeclip_path)

        from file import get_context

        path_list = list(paths) if paths else ["."]

        content, token_info = get_context(
            paths=path_list,
            extensions=(".py", ".ts", ".tsx", ".js", ".java", ".cpp", ".h"),
            include_git_diff=False,
            readmes_only=False,
        )

        # Show token info
        total_tokens = token_info.get("total_tokens", 0)
        total_files = token_info.get("total_files", 0)

        click.echo("Context Summary:", err=True)
        click.echo(f"  Files: {total_files}", err=True)
        click.echo(f"  Tokens: ~{total_tokens:,}", err=True)
        click.echo("", err=True)

        # Output the content
        click.echo(content)

    except ImportError as e:
        click.echo(f"Error: Cannot import codeclip: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
