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
from typing import Annotated, Optional

import typer

from .file import get_context
from .token_profiler import generate_flamegraph, profile_code_context

# Configure logging
logging.basicConfig(
    level=logging.INFO if os.getenv("LOGLEVEL", "info").lower() == "info" else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("codeclip.cli")

app = typer.Typer(
    help="Provide codebase context to LLMs with smart defaults.",
    add_completion=False,
    pretty_exceptions_enable=False,
)


def copy_to_clipboard(content: str) -> None:
    """Copy content to clipboard on macOS."""
    try:
        process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        process.communicate(content.encode("utf-8"))
    except FileNotFoundError:
        typer.echo("pbcopy not found - clipboard integration skipped", err=True)


@app.command()
def main(
    paths: Annotated[
        Optional[list[str]], typer.Argument(help="Paths to include (space-separated). Example: metta/rl tests/rl")
    ] = None,
    stdout: Annotated[bool, typer.Option("-s", "--stdout", help="Output to stdout instead of clipboard")] = False,
    readmes: Annotated[
        bool, typer.Option("-r", "--readmes", help="Only include README.md files, including ancestor READMEs")
    ] = False,
    extension: Annotated[
        Optional[list[str]],
        typer.Option("-e", "--extension", help="File extensions to include (e.g. -e py -e js or -e .py -e .js)"),
    ] = None,
    ignore: Annotated[
        Optional[list[str]],
        typer.Option(
            "-i",
            "--ignore",
            help="Ignore patterns: directory names (-i cache) or paths (-i build/cache)",
        ),
    ] = None,
    profile: Annotated[
        bool, typer.Option("-p", "--profile", help="Show detailed token distribution analysis to stderr")
    ] = False,
    flamegraph: Annotated[
        bool, typer.Option("-f", "--flamegraph", help="Generate a flame graph HTML visualization of token distribution")
    ] = False,
    diff: Annotated[
        bool, typer.Option("-d", "--diff", help="Append git diff vs origin/main to the output as a single virtual file")
    ] = False,
) -> None:
    """Provide codebase context to LLMs with smart defaults."""
    # Use provided paths or default behavior
    # For diff mode with no paths, use empty list to get only diff
    # Otherwise default to current directory if no paths provided
    if diff and not paths and not readmes:
        path_list = []
    else:
        path_list = list(paths) if paths else ["."]
    logger.debug(f"Paths: {path_list}")

    # Normalize extensions to ensure they have a dot prefix
    normalized_extensions = tuple(ext if ext.startswith(".") else f".{ext}" for ext in extension) if extension else None
    ignore_dirs = tuple(ignore) if ignore else ()

    # Get content and profile data
    try:
        output_content = None
        token_info = None
        profile_data = None
        profile_report = None

        # If we need profile or flamegraph, use profile_code_context which includes get_context
        if profile or flamegraph:
            # This calls get_context internally and builds the full profile
            profile_report, profile_data = profile_code_context(
                paths=path_list,
                extensions=normalized_extensions,
                include_git_diff=diff,
                readmes_only=readmes,
                ignore_dirs=ignore_dirs,
            )
            # Extract the content from the profile data
            output_content = profile_data.get("context", "")
        else:
            # Just get content and basic token info
            output_content, token_info = get_context(
                paths=path_list,
                extensions=normalized_extensions,
                include_git_diff=diff,
                diff_base="origin/main",
                readmes_only=readmes,
                ignore_dirs=ignore_dirs,
            )
            profile_data = token_info

        # Generate flamegraph if requested
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
            typer.echo(f"Flame graph generated at: {flamegraph_path}", err=True)

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
                    typer.echo(f"Flame graph saved but couldn't auto-open browser on {system}.", err=True)
            except Exception as e:
                typer.echo(f"Flame graph saved but couldn't launch Chrome: {e}", err=True)
    except Exception as e:
        typer.echo(f"Error loading context: {e}", err=True)
        raise typer.Exit(1) from e

    # Output content to stdout or clipboard
    if stdout:
        # Output to stdout
        typer.echo(output_content)
    else:
        # Default behavior: copy to clipboard
        copy_to_clipboard(output_content)

    # Show summary when copying to clipboard (unless no files found)
    if not stdout and profile_data:
        total_tokens = profile_data["total_tokens"]
        total_files = profile_data["total_files"]

        if total_files == 0:
            typer.echo("No files found to copy", err=True)
            raise typer.Exit(1)

        # Build summary message
        action = "Copied"
        summary_parts = [f"{action} ~{total_tokens:,} tokens from {total_files} files"]

        # Build summary showing requested paths + auto-included READMEs
        if "documents" in profile_data and "file_token_counts" in profile_data:
            documents = profile_data["documents"]
            file_token_counts = profile_data["file_token_counts"]

            # Group tokens by requested path
            summary_items = {}  # path -> token count

            # First, identify which files belong to which requested path
            for doc in documents:
                # Special handling for GIT_DIFF
                if doc.source.startswith("GIT_DIFF:"):
                    summary_items["<diff>"] = file_token_counts.get(doc.source, 0)
                    continue

                doc_path = Path(doc.source)
                doc_path_resolved = doc_path.resolve()

                # Check if this file is under any requested path
                is_under_requested = False
                for orig_path in path_list:
                    req_path = Path(orig_path).resolve()
                    try:
                        doc_path_resolved.relative_to(req_path)
                        is_under_requested = True
                        break
                    except ValueError:
                        continue

                # Only show READMEs as individual entries if they're parent READMEs (not under requested paths)
                if doc_path.name == "README.md" and not is_under_requested:
                    # Show parent READMEs as individual entries
                    summary_items[doc.source] = file_token_counts.get(doc.source, 0)
                else:
                    # For other files, group by the requested path
                    matched = False
                    for orig_path in path_list:
                        req_path = Path(orig_path).resolve()
                        try:
                            # Check if doc is under this requested path
                            doc_path_resolved.relative_to(req_path)
                            # If the requested path is a file, show it directly
                            if req_path.is_file():
                                summary_items[str(req_path)] = file_token_counts.get(doc.source, 0)
                            else:
                                # If it's a directory, accumulate tokens
                                summary_items[str(req_path)] = summary_items.get(
                                    str(req_path), 0
                                ) + file_token_counts.get(doc.source, 0)
                            matched = True
                            break
                        except ValueError:
                            continue

                    if not matched and not is_under_requested:
                        # This is likely a parent README
                        summary_items[doc.source] = file_token_counts.get(doc.source, 0)

            # Sort by tokens and display
            sorted_items = sorted(summary_items.items(), key=lambda x: x[1], reverse=True)

            # Auto-zoom: if single path requested and it's a directory, show its contents instead
            if len(path_list) == 1:
                req_path = Path(path_list[0]).resolve()
                if req_path.is_dir():
                    # Show individual files within the directory instead of the directory itself
                    summary_parts.append("  Top items:")

                    # Collect and aggregate by immediate children (directories or files)
                    aggregated = {}
                    for doc in documents:
                        # Special handling for GIT_DIFF
                        if doc.source.startswith("GIT_DIFF:"):
                            aggregated["<diff>"] = file_token_counts.get(doc.source, 0)
                            continue

                        doc_path = Path(doc.source)
                        tokens = file_token_counts.get(doc.source, 0)
                        if tokens > 0:
                            try:
                                # Check if this file is under the requested directory
                                relative = doc_path.relative_to(req_path)
                                # Get the immediate child name (first part of relative path)
                                if len(relative.parts) > 0:
                                    immediate_child = relative.parts[0]
                                    display_path = str(Path(path_list[0]) / immediate_child)
                                    aggregated[display_path] = aggregated.get(display_path, 0) + tokens
                            except ValueError:
                                # Show READMEs that were auto-included with full path
                                if doc_path.name == "README.md":
                                    try:
                                        display_path = str(doc_path.relative_to(Path.cwd()))
                                    except ValueError:
                                        display_path = str(doc_path)
                                    aggregated[display_path] = aggregated.get(display_path, 0) + tokens

                    # Sort by tokens and show top 3
                    sorted_items = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
                    for display_name, tokens in sorted_items[:3]:
                        pct = (tokens / total_tokens * 100) if total_tokens else 0
                        summary_parts.append(f"    {display_name}: ~{tokens:,} tokens ({pct:.0f}%)")
                else:
                    # Single file requested - show it and any READMEs
                    summary_parts.append("  Files:")
                    for item_path, tokens in sorted_items:
                        pct = (tokens / total_tokens * 100) if total_tokens else 0
                        # Use relative path for display
                        item_path_obj = Path(item_path)
                        try:
                            display_path = item_path_obj.relative_to(Path.cwd())
                        except ValueError:
                            display_path = item_path_obj.name
                        summary_parts.append(f"    {display_path}: ~{tokens:,} tokens ({pct:.0f}%)")
            else:
                # Multiple paths - show them
                summary_parts.append("  By path:")
                for item_path, tokens in sorted_items[:10]:  # Show top 10
                    pct = (tokens / total_tokens * 100) if total_tokens else 0
                    # Special handling for diff marker
                    if item_path == "<diff>":
                        display_path = "<diff>"
                    else:
                        # Use relative path for display
                        item_path_obj = Path(item_path)
                        try:
                            # Try to make it relative to current working directory
                            display_path = item_path_obj.relative_to(Path.cwd())
                        except ValueError:
                            # If that fails, just use the name
                            display_path = item_path_obj.name
                    summary_parts.append(f"    {display_path}: ~{tokens:,} tokens ({pct:.0f}%)")
        # Get top-level breakdown
        elif "node_cache" in profile_data:
            # Full profile data available
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

        elif "top_level_summary" in profile_data:
            # Optimized top-level summary for single path
            top_level = profile_data["top_level_summary"]
            if top_level:
                # Sort by tokens descending
                sorted_items = sorted(top_level.items(), key=lambda x: x[1], reverse=True)
                if sorted_items:
                    summary_parts.append("  Top items:")
                    for name, tokens in sorted_items[:3]:
                        pct = (tokens / total_tokens * 100) if total_tokens else 0
                        summary_parts.append(f"    {name}: ~{tokens:,} tokens ({pct:.0f}%)")

        elif "path_summaries" in profile_data:
            # Quick summary data for multiple paths
            path_summaries = profile_data["path_summaries"]

            if len(path_list) > 1:
                summary_parts.append("  By path:")
                for path_str in path_list:
                    path = Path(path_str).resolve()
                    summary = path_summaries.get(str(path))
                    if summary:
                        tokens = summary["tokens"]
                        pct = (tokens / total_tokens * 100) if total_tokens else 0
                        summary_parts.append(f"    {path.name}: ~{tokens:,} tokens ({pct:.0f}%)")

        typer.echo("\n".join(summary_parts), err=True)

    # Show detailed profiling info if requested
    if profile and profile_data:
        typer.echo("\n" + "=" * 60, err=True)
        typer.echo("DETAILED TOKEN PROFILE", err=True)
        typer.echo("=" * 60 + "\n", err=True)
        typer.echo(profile_report, err=True)


if __name__ == "__main__":
    app()
