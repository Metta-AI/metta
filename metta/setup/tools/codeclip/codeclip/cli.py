"""
Command-line interface for code context retrieval.

Provides a flexible way to extract context from codebases for LLM interactions.
"""

import logging
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import click

from .file import get_context
from .token_profiler import generate_flamegraph, profile_code_context

# Configure logging
logging.basicConfig(
    level=logging.INFO if os.getenv("LOGLEVEL", "info").lower() == "info" else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("codeclip.cli")


def copy_to_clipboard(content: str) -> None:
    """Copy content to clipboard on macOS."""
    try:
        process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        process.communicate(content.encode("utf-8"))
    except FileNotFoundError:
        click.echo("pbcopy not found - clipboard integration skipped", err=True)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("paths", nargs=-1, type=str)
@click.option("-p", "--pbcopy", is_flag=True, help="Copy to clipboard (macOS only)")
@click.option("-r", "--raw", is_flag=True, help="Output in raw format instead of XML")
@click.option("-e", "--extension", multiple=True, help="File extensions to include (e.g. -e .py -e .js)")
@click.option("--profile", is_flag=True, help="Profile token distribution instead of outputting content")
@click.option("--flamegraph", is_flag=True, help="Generate a flame graph HTML visualization of token distribution")
def cli(
    paths: Tuple[str, ...], pbcopy: bool, raw: bool, extension: Tuple[str, ...], profile: bool, flamegraph: bool
) -> None:
    """
    Provide codebase context to LLMs with smart defaults.
    - PATHS can be space-separated. Example: metta/rl tests/rl
    """
    # If no paths provided and no flags, show help
    if not paths and not any([profile, flamegraph, extension]):
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    # Use provided paths or default to current directory
    path_list = list(paths) if paths else ["."]
    logger.debug(f"Paths: {path_list}")

    # Load context documents from current directory
    try:
        if profile or flamegraph:
            logger.debug(f"Profiling code context for {path_list}")
            output_content, profile_data = profile_code_context(path_list, raw, extension)
            logger.debug("Profile complete")

            if flamegraph:
                # Create temp file in /tmp directory
                project_name = path_list[0].split("/")[-1] if path_list else "code"
                temp_file = tempfile.NamedTemporaryFile(
                    prefix=f"flamegraph_{project_name}_", suffix=".html", dir="/tmp", delete=False
                )
                flamegraph_path = temp_file.name
                temp_file.close()

                # Generate the flame graph
                logger.debug(f"Generating flame graph at: {flamegraph_path}")
                generate_flamegraph(profile_data, flamegraph_path)
                logger.debug(f"Flame graph generated at: {flamegraph_path}")
                click.echo(f"Flame graph generated at: {flamegraph_path}")

                # Open the file in Chrome
                try:
                    system = platform.system()
                    if system == "Darwin":  # macOS
                        subprocess.run(["open", "-a", "Google Chrome", flamegraph_path])
                    elif system == "Linux":
                        subprocess.run(["google-chrome", flamegraph_path])
                    elif system == "Windows":
                        subprocess.run(["chrome", flamegraph_path], shell=True)
                    else:
                        click.echo(f"Flame graph saved but couldn't auto-open browser on {system}.")
                except Exception as e:
                    click.echo(f"Flame graph saved but couldn't launch Chrome: {e}")

                # Return early when flamegraph is generated - don't output content
                return
        else:
            # Regular content output
            output_content = get_context(paths=path_list, raw=raw, extensions=extension)
    except Exception as e:
        click.echo(f"Error loading context: {e}", err=True)
        return

    if pbcopy:
        copy_to_clipboard(output_content)
        # Get detailed token breakdown
        _, profile_data = profile_code_context(path_list, raw, extension)

        # Handle case where no files were found
        if not profile_data:
            click.echo("No files found to copy", err=True)
            return

        total_tokens = profile_data["total_tokens"]
        total_files = profile_data["total_files"]

        # Build summary message
        summary_parts = [f"Copied ~{total_tokens:,} tokens from {total_files} files"]

        # Get top-level breakdown
        node_cache = profile_data["node_cache"]

        # If single path provided, show subdirectories/files
        if len(path_list) == 1:
            # Find the root node
            root_path = Path(path_list[0]).resolve()

            # Collect immediate children (both files and directories)
            children = []
            for path_str, node in node_cache.items():
                path = Path(path_str)
                try:
                    # Check if this is a direct child of root
                    relative = path.relative_to(root_path)
                    # Only include if it's a direct child (only one part in relative path)
                    if len(relative.parts) == 1:
                        children.append((path.name, node.total_tokens))
                except ValueError:
                    # Skip if paths aren't related
                    continue

            # Sort by tokens and take top 3
            children.sort(key=lambda x: x[1], reverse=True)
            if children:
                summary_parts.append("  Top items:")
                for name, tokens in children[:3]:
                    pct = (tokens / total_tokens * 100) if total_tokens else 0
                    summary_parts.append(f"    {name}: ~{tokens:,} tokens ({pct:.0f}%)")

        # If multiple paths, show breakdown by input path
        else:
            summary_parts.append("  By path:")
            for path_str in path_list:
                path = Path(path_str).resolve()
                node = node_cache.get(str(path))
                if node:
                    pct = (node.total_tokens / total_tokens * 100) if total_tokens else 0
                    summary_parts.append(f"    {path.name}: ~{node.total_tokens:,} tokens ({pct:.0f}%)")

        click.echo("\n".join(summary_parts), err=True)
    else:
        click.echo(output_content)


if __name__ == "__main__":
    cli()
