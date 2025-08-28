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

from .commands import (
    DebugTestsCommand,
    FixCommand,
    ImplementCommand,
    RefactorCommand,
    TestCommand,
    run_summarize_command,
)
from .git_layer import GitLayer, GitLayerError
from .models import ExecutionContext

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


@main.command("mcp-server")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def mcp_server(verbose: bool) -> None:
    """
    Start the MCP server for codebot.

    This command starts a Model Context Protocol (MCP) server that exposes
    codebot's AI-powered development tools to MCP clients like Claude Desktop.

    The server communicates via stdio and provides tools for:
    - summarize: Generate AI-powered code summaries
    - context: Show code context for AI commands

    Example usage in Claude Desktop config:
    {
        "mcpServers": {
            "codebot": {
                "command": "codebot",
                "args": ["mcp-server"]
            }
        }
    }
    """
    if verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("codebot").setLevel(logging.DEBUG)

    try:
        # Import and run the MCP server
        from .mcp_server import main as mcp_main

        asyncio.run(mcp_main())
    except KeyboardInterrupt:
        click.echo("MCP server stopped.", err=True)
    except Exception as e:
        logger.error(f"Error starting MCP server: {e}")
        click.echo(f"Error starting MCP server: {str(e)}", err=True)
        sys.exit(1)


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
        codeclip_path = os.path.join(current_dir, "..", "codeclip")

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


# Command-specific context modes
COMMAND_CONTEXT_MODES = {
    "RefactorCommand": "minimal",  # Just refactor the target file(s)
    "FixCommand": "minimal",  # Fix the specific file(s)
    "ImplementCommand": "targeted",  # May need related files for context
    "DebugTestsCommand": "targeted",  # Need test files + source files
    "TestCommand": "targeted",  # Need source to write tests for
    "SummarizeCommand": "full",  # Need full context for analysis
}


def _get_context_mode(command_name: str) -> str:
    """Get the appropriate context mode for a command."""
    return COMMAND_CONTEXT_MODES.get(command_name, "full")  # Default to full context


# Helper function for common command execution
async def _execute_command(command_class, paths: Tuple[str, ...], mode: Optional[str] = None, **kwargs) -> None:
    """Execute a command with the given paths and mode."""
    try:
        # Determine context mode based on command
        context_mode = _get_context_mode(command_class.__name__)

        path_list = list(paths) if paths else ["."]

        if context_mode == "minimal":
            # Direct file reading - no codeclip overhead or parent READMEs
            files = {}
            for path_str in path_list:
                try:
                    file_path = Path(path_str)
                    if file_path.exists():
                        files[str(file_path.resolve())] = file_path.read_text(encoding="utf-8")
                    else:
                        logger.warning(f"Path does not exist: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not read {path_str}: {e}")

            # Create minimal token_info structure
            token_info = {
                "total_tokens": 0,
                "total_files": len(files),
                "documents": [
                    type("Document", (), {"source": path, "content": content})() for path, content in files.items()
                ],
            }
            content = ""  # Not used when we have documents

        else:
            # Use codeclip with appropriate settings
            import os

            current_dir = os.path.dirname(__file__)
            codeclip_path = os.path.join(current_dir, "..", "codeclip")

            if codeclip_path not in sys.path:
                sys.path.insert(0, codeclip_path)

            from file import get_context

            if context_mode == "targeted":
                # Include git diff for better context
                content, token_info = get_context(
                    paths=path_list,
                    extensions=(".py", ".ts", ".tsx", ".js", ".java", ".cpp", ".h"),
                    include_git_diff=True,
                    readmes_only=False,
                )
            else:  # "full"
                # Current behavior - everything including parent READMEs
                content, token_info = get_context(
                    paths=path_list,
                    extensions=(".py", ".ts", ".tsx", ".js", ".java", ".cpp", ".h"),
                    include_git_diff=False,
                    readmes_only=False,
                )

        # Parse files from content (handle codeclip's document format)
        files = {}

        # Debug logging
        logger.debug(f"Token info keys: {token_info.keys()}")
        logger.debug(f"Content length: {len(content)}")

        # Use codeclip's token_info which already has parsed documents
        if "documents" in token_info:
            logger.debug(f"Found {len(token_info['documents'])} documents in token_info")
            for i, doc in enumerate(token_info["documents"]):
                logger.debug(
                    f"Document {i}: type={type(doc)}, attrs={dir(doc) if hasattr(doc, '__dict__') else 'no __dict__'}"
                )

                # Each document has 'source' (filepath) and 'content'
                if hasattr(doc, "source") and hasattr(doc, "content"):
                    files[doc.source] = doc.content
                    logger.debug(f"Added file: {doc.source} ({len(doc.content)} chars)")
                elif isinstance(doc, dict):
                    source = doc.get("source", "")
                    content = doc.get("content", "")
                    if source:
                        files[source] = content
                        logger.debug(f"Added file from dict: {source} ({len(content)} chars)")

        # If documents parsing didn't work, try to parse the raw content
        if not files and content.strip():
            logger.debug("No files from documents, trying to parse content directly")
            # The content looks like it has Document objects serialized
            # Let's try a different approach - look for the actual files
            if "git_utils.py" in content:
                # Include the full content as a single context for now
                files["combined_context"] = content
                logger.debug("Added combined context as fallback")

        # Create execution context
        context = ExecutionContext(
            files=files,
            working_directory=Path.cwd(),
            token_count=token_info.get("total_tokens", 0),
            metadata=token_info,
        )

        # Create and execute command
        command = command_class()
        result = await command.execute(context, mode=mode, **kwargs)

        if not result.success:
            click.echo(f"Error: {result.summary}", err=True)
            sys.exit(1)

        # Auto-commit AI changes if there are any file changes
        if result.file_changes and not kwargs.get("no_commit", False):
            try:
                from .git_layer import GitLayer

                git_layer = GitLayer()

                commit_hash = git_layer.commit_ai_changes(
                    command=command.name, changes=result.file_changes, description=result.summary
                )

                click.echo(f"🤖 Created AI commit: {commit_hash[:8]}")

            except Exception as e:
                # Don't fail the entire command if git commit fails
                logger.warning(f"Failed to create AI commit: {e}")
                click.echo(f"Warning: Could not create AI commit: {e}", err=True)

        # Show summary
        click.echo(f"✓ {result.summary}")

        # Show metadata if available
        if result.metadata:
            for key, value in result.metadata.items():
                if key not in ["error"]:
                    click.echo(f"  {key.replace('_', ' ').title()}: {value}", err=True)

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.argument("paths", nargs=-1, type=str)
@click.option(
    "-m", "--mode", type=click.Choice(["oneshot", "claudesdk", "interactive"]), help="Execution mode (default: oneshot)"
)
@click.option("--no-commit", is_flag=True, help="Skip automatic git commit")
def test(paths: Tuple[str, ...], mode: Optional[str], no_commit: bool) -> None:
    """
    Generate comprehensive tests for the provided code.

    Analyzes code structure and creates unit tests with good coverage,
    including edge cases and error scenarios.

    Examples:
        # Generate tests for current directory
        codebot test

        # Generate tests for specific files
        codebot test src/main.py src/utils.py

        # Use Claude SDK mode
        codebot test --mode claudesdk src/
    """
    asyncio.run(_execute_command(TestCommand, paths, mode, auto_commit=not no_commit))


@main.command()
@click.argument("paths", nargs=-1, type=str)
@click.option(
    "-m", "--mode", type=click.Choice(["oneshot", "claudesdk", "interactive"]), help="Execution mode (default: oneshot)"
)
@click.option("--no-commit", is_flag=True, help="Skip automatic git commit")
def refactor(paths: Tuple[str, ...], mode: Optional[str], no_commit: bool) -> None:
    """
    Refactor code to improve quality without changing behavior.

    Identifies code smells, removes duplication, improves structure,
    and applies refactoring techniques while preserving functionality.

    Examples:
        # Refactor current directory
        codebot refactor

        # Refactor specific files
        codebot refactor src/api.py

        # Interactive refactoring
        codebot refactor --mode interactive src/
    """
    asyncio.run(_execute_command(RefactorCommand, paths, mode, auto_commit=not no_commit))


@main.command()
@click.argument("paths", nargs=-1, type=str)
@click.option(
    "-m", "--mode", type=click.Choice(["oneshot", "claudesdk", "interactive"]), help="Execution mode (default: oneshot)"
)
@click.option("--no-commit", is_flag=True, help="Skip automatic git commit")
def fix(paths: Tuple[str, ...], mode: Optional[str], no_commit: bool) -> None:
    """
    Fix code issues, bugs, and problems.

    Identifies and fixes syntax errors, logic bugs, performance issues,
    security vulnerabilities, and other code problems.

    Examples:
        # Fix issues in current directory
        codebot fix

        # Fix specific files
        codebot fix src/buggy_module.py

        # Use Claude SDK for autonomous fixing
        codebot fix --mode claudesdk src/
    """
    asyncio.run(_execute_command(FixCommand, paths, mode, auto_commit=not no_commit))


@main.command()
@click.argument("paths", nargs=-1, type=str)
@click.option(
    "-m", "--mode", type=click.Choice(["oneshot", "claudesdk", "interactive"]), help="Execution mode (default: oneshot)"
)
@click.option("--no-commit", is_flag=True, help="Skip automatic git commit")
def implement(paths: Tuple[str, ...], mode: Optional[str], no_commit: bool) -> None:
    """
    Implement new features based on requirements and context.

    Analyzes existing code patterns and implements new functionality
    that fits the codebase architecture and conventions.

    Examples:
        # Implement feature based on context
        codebot implement

        # Implement with specific context files
        codebot implement src/models.py requirements.txt

        # Interactive implementation
        codebot implement --mode interactive
    """
    asyncio.run(_execute_command(ImplementCommand, paths, mode, auto_commit=not no_commit))


@main.command("debug-tests")
@click.option(
    "-m", "--mode", type=click.Choice(["oneshot", "claudesdk", "interactive"]), help="Execution mode (default: oneshot)"
)
@click.option("--no-commit", is_flag=True, help="Skip automatic git commit")
def debug_tests(mode: Optional[str], no_commit: bool) -> None:
    """
    Debug failing tests using error output from clipboard.

    Analyzes test failure output and fixes the issues causing tests to fail.
    Prefers fixing implementation code over modifying tests.

    Usage:
        1. Run your tests and copy the failure output
        2. Run: codebot debug-tests

    Examples:
        # Debug test failures (reads from clipboard)
        codebot debug-tests

        # Use Claude SDK for autonomous debugging
        codebot debug-tests --mode claudesdk
    """
    try:
        # Get clipboard content
        import subprocess

        clipboard_content = ""
        try:
            # Try different clipboard access methods
            if sys.platform == "darwin":  # macOS
                result = subprocess.run(["pbpaste"], capture_output=True, text=True)
                clipboard_content = result.stdout
            elif sys.platform.startswith("linux"):  # Linux
                result = subprocess.run(["xclip", "-selection", "clipboard", "-o"], capture_output=True, text=True)
                clipboard_content = result.stdout
            elif sys.platform == "win32":  # Windows
                result = subprocess.run(["powershell", "Get-Clipboard"], capture_output=True, text=True)
                clipboard_content = result.stdout
        except FileNotFoundError:
            click.echo("Error: Could not access clipboard. Make sure you have clipboard tools installed.", err=True)
            sys.exit(1)

        if not clipboard_content.strip():
            click.echo("Error: No content found in clipboard. Please copy test failure output first.", err=True)
            sys.exit(1)

        # Create execution context with clipboard content
        context = ExecutionContext(
            clipboard=clipboard_content,
            working_directory=Path.cwd(),
            files={},  # DebugTestsCommand will gather needed files
            token_count=len(clipboard_content.split()),
        )

        # Create and execute command
        async def run_debug():
            command = DebugTestsCommand()
            result = await command.execute(context, mode=mode, auto_commit=not no_commit)

            if not result.success:
                click.echo(f"Error: {result.summary}", err=True)
                sys.exit(1)

            click.echo(f"✓ {result.summary}")

            if result.metadata:
                for key, value in result.metadata.items():
                    if key not in ["error"]:
                        click.echo(f"  {key.replace('_', ' ').title()}: {value}", err=True)

        asyncio.run(run_debug())

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command("reset-to-human")
def reset_to_human() -> None:
    """
    Reset to the last human commit, undoing all AI-generated commits.

    This command performs a soft reset to the most recent commit that was
    not made by codebot, effectively undoing all AI-generated changes while
    keeping them in the working directory.

    Examples:
        # Reset to last human commit
        codebot reset-to-human
    """
    try:
        git_layer = GitLayer()

        # Get AI commits that will be reset
        ai_commits = git_layer.get_ai_commits_since_human()

        if not ai_commits:
            click.echo("No AI commits found to reset.")
            return

        click.echo(f"Found {len(ai_commits)} AI commits to reset:")
        for commit in ai_commits[:5]:  # Show first 5
            click.echo(f"  • {commit['hash'][:8]} - {commit['subject']}")

        if len(ai_commits) > 5:
            click.echo(f"  ... and {len(ai_commits) - 5} more")

        # Confirm with user
        if click.confirm("Reset to last human commit? (changes will be kept in working directory)"):
            reset_commit = git_layer.reset_to_human()
            if reset_commit:
                click.echo(f"✓ Reset to human commit {reset_commit[:8]}")
                click.echo("All AI-generated changes are now in your working directory.")
            else:
                click.echo("No human commits found to reset to.")
        else:
            click.echo("Reset cancelled.")

    except GitLayerError as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
