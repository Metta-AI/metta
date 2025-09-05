#!/usr/bin/env -S uv run
"""Generic tool runner using argparse for simplicity and directness."""

import argparse
import inspect
import json
import logging
import os
import signal
import sys
import warnings
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from typing_extensions import TypeVar

from metta.common.tool import Tool
from metta.common.util.log_config import init_logging
from metta.mettagrid.config import Config
from metta.mettagrid.util.module import load_symbol
from metta.rl.system_config import seed_everything

logger = logging.getLogger(__name__)
console = Console()

# Configure rich logging
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(console=console, rich_tracebacks=True)]
)


def init_mettagrid_system_environment() -> None:
    """Initialize environment variables for headless operation."""
    # Set CUDA launch blocking for better error messages in development
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    # Make sure Numpy uses only 1 thread (matches what happens in C++ side)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Make sure we're running headless to avoid any display issues
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("DISPLAY", "")

    # Suppress deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pygame.pkgdata")


T = TypeVar("T", bound=Config)


def parse_value(value_str: str) -> Any:
    """Parse a string value into appropriate Python type."""
    lower = value_str.lower()

    # Handle boolean values
    if lower in {"true", "false"}:
        return lower == "true"

    # Handle null/none
    if lower in {"none", "null"}:
        return None

    # Try to parse JSON containers
    if (value_str.startswith("{") and value_str.endswith("}")) or (
        value_str.startswith("[") and value_str.endswith("]")
    ):
        try:
            return json.loads(value_str)
        except Exception:
            pass

    # Try to parse as int
    try:
        return int(value_str)
    except ValueError:
        pass

    # Try to parse as float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Return as string
    return value_str


def parse_cli_args(cli_args: list[str]) -> dict[str, Any]:
    """Parse CLI arguments in key=value format, keeping dotted keys flat."""
    parsed = {}

    for arg in cli_args:
        # Strip leading dashes if present (support --key=value format)
        clean_arg = arg.lstrip("-")

        if "=" not in clean_arg:
            raise ValueError(f"Invalid argument format: {arg}. Expected key=value")

        key, value = clean_arg.split("=", 1)
        # Keep dotted keys intact for proper classification
        parsed[key] = parse_value(value)

    return parsed


def get_tool_fields(tool_class: type[Tool]) -> set[str]:
    """Get all field names from a Tool class and its parent classes."""
    fields = set()

    # Walk up the MRO to get fields from Pydantic models only
    for base in tool_class.__mro__:
        # Stop at Tool base class
        if base is Tool:
            break
        # Only include fields from Pydantic models
        if issubclass(base, BaseModel) and hasattr(base, "model_fields"):
            fields.update(base.model_fields.keys())

    return fields


def split_args_and_overrides(
    cli_args: dict[str, Any], make_tool_cfg: Any, tool_cfg: Tool
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """
    Split CLI arguments into function args and tool overrides.

    Rules:
    - Dotted keys (a.b.c) are always overrides
    - Exact parameter name matches are function args
    - Otherwise unknown

    Returns:
        - Function arguments (for make_tool_cfg)
        - Override arguments (for tool fields)
        - List of unknown arguments
    """
    func_args = {}
    overrides = {}
    unknown = []

    # Get function parameter names
    if inspect.isclass(make_tool_cfg) and issubclass(make_tool_cfg, Tool):
        # Tool class constructor
        func_params = set(inspect.signature(make_tool_cfg.__init__).parameters.keys()) - {"self"}
    else:
        # Function that returns a Tool
        func_params = set(inspect.signature(make_tool_cfg).parameters.keys())

    # Get tool field names
    tool_fields = get_tool_fields(type(tool_cfg))

    # Process each CLI argument
    for key, value in cli_args.items():
        if "." in key:
            # Dotted path => always an override
            base_key = key.split(".", 1)[0]
            if base_key in tool_fields:
                overrides[key] = value
            else:
                unknown.append(key)
            continue

        # Non-dotted: could be a function param or a top-level tool field
        if key in func_params:
            func_args[key] = value
        elif key in tool_fields:
            overrides[key] = value
        else:
            unknown.append(key)

    return func_args, overrides, unknown


def display_arg_classification(
    func_args: dict[str, Any], overrides: dict[str, Any], unknown: list[str], make_tool_cfg_path: str
) -> None:
    """Display a rich table showing how arguments were classified."""
    table = Table(title=f"Argument Classification for [bold cyan]{make_tool_cfg_path}[/bold cyan]")
    table.add_column("Argument", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Type", style="green")

    # Add function arguments
    for key, value in func_args.items():
        table.add_row(f"{key}", f"{value}", "Function Arg")

    # Add overrides
    for key, value in overrides.items():
        table.add_row(f"{key}", f"{value}", "Override")

    # Add unknown arguments
    for key in unknown:
        table.add_row(f"{key}", "N/A", "[red]Unknown[/red]")

    console.print(table)


def main():
    """Main entry point using argparse."""
    parser = argparse.ArgumentParser(
        description="Run a tool with automatic argument classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,  # avoid -v matching -verbose or similar
        epilog="""
Examples:
  %(prog)s experiments.recipes.arena.train run=test_123 trainer.total_timesteps=100000
  %(prog)s experiments.recipes.arena.play policy_uri=file://./checkpoints --verbose
  %(prog)s experiments.recipes.arena.train -- --trainer.epochs=10 --model.lr=0.001
  %(prog)s experiments.recipes.arena.train optim='{"lr":1e-3,"beta1":0.9}'

Rules:
  - Dotted keys (a.b.c) are always configuration overrides
  - Exact parameter names are function arguments
  - Values: true/false, null/none, JSON containers {...}/[...], or int/float/string

This tool automatically determines which arguments are meant for the tool
constructor/function vs which are configuration overrides based on introspection.

Use -- to force treating following arguments as key=value pairs if they start with dashes.
        """,
    )

    parser.add_argument(
        "make_tool_cfg_path", help="Path to the function to run (e.g., experiments.recipes.arena.train)"
    )
    parser.add_argument("args", nargs="*", help="Arguments in key=value format")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed argument classification")

    # Use parse_known_args to capture option-like tokens that aren't recognized
    known_args, unknown_args = parser.parse_known_args()

    # Combine positional args with any unknown option-like tokens
    # This allows --key=value to work without requiring --
    all_args = (known_args.args or []) + unknown_args

    # Initialize
    init_logging()
    init_mettagrid_system_environment()

    # Exit on ctrl+c with proper exit code
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(130))  # 130 = interrupted by Ctrl-C

    # Parse CLI arguments
    try:
        cli_args = parse_cli_args(all_args)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return 2  # Exit code 2 for usage errors

    console.print(f"\n[bold cyan]Loading tool:[/bold cyan] {known_args.make_tool_cfg_path}")

    # Load the tool configuration function/class
    try:
        make_tool_cfg = load_symbol(known_args.make_tool_cfg_path)
    except Exception as e:
        console.print(f"[red]Error loading {known_args.make_tool_cfg_path}:[/red] {e}")
        return 1

    # Create initial tool config to inspect its fields (without side effects)
    if inspect.isclass(make_tool_cfg) and issubclass(make_tool_cfg, Tool):
        # Tool class constructor - use model_construct to avoid validation/side effects
        try:
            # Pydantic v2 method - no validation or __init__
            temp_tool_cfg = make_tool_cfg.model_construct()
        except Exception:
            # Fallback if model_construct is unavailable
            temp_tool_cfg = make_tool_cfg()
    else:
        # Function that makes a tool - call with only defaults to minimize side effects
        sig = inspect.signature(make_tool_cfg)
        temp_args = {
            param_name: param.default
            for param_name, param in sig.parameters.items()
            if param.default is not inspect.Parameter.empty
        }
        try:
            temp_tool_cfg = make_tool_cfg(**temp_args)
        except Exception as e:
            console.print(f"[red]Error creating temporary tool instance:[/red] {e}")
            return 1

    # Split arguments into function args and overrides
    func_args, override_args, unknown_args = split_args_and_overrides(cli_args, make_tool_cfg, temp_tool_cfg)

    # Handle unknown arguments
    if unknown_args:
        console.print(f"\n[red]Error: Unknown arguments:[/red] {', '.join(unknown_args)}")
        console.print("\n[yellow]Available function parameters:[/yellow]")
        if inspect.isclass(make_tool_cfg) and issubclass(make_tool_cfg, Tool):
            sig = inspect.signature(make_tool_cfg.__init__)
        else:
            sig = inspect.signature(make_tool_cfg)
        for param in sig.parameters.values():
            if param.name != "self":
                console.print(f"  - {param.name}")

        console.print("\n[yellow]Available tool fields for overrides:[/yellow]")
        for field in sorted(get_tool_fields(type(temp_tool_cfg))):
            console.print(f"  - {field}")
        return 2  # Exit code 2 for usage errors

    # Display classification if verbose
    if known_args.verbose:
        display_arg_classification(func_args, override_args, unknown_args, known_args.make_tool_cfg_path)

    # Create the tool config object with function arguments
    try:
        if inspect.isclass(make_tool_cfg) and issubclass(make_tool_cfg, Tool):
            # Tool config constructor
            tool_cfg = make_tool_cfg(**func_args)
        else:
            # Function that makes a tool config
            tool_cfg = make_tool_cfg(**func_args)

            # Mark consumed args
            if hasattr(tool_cfg, "consumed_args"):
                tool_cfg.consumed_args.extend(func_args.keys())
    except Exception as e:
        console.print(f"[red]Error creating tool configuration:[/red] {e}")
        return 1

    if not isinstance(tool_cfg, Tool):
        console.print(
            f"[red]Error:[/red] {known_args.make_tool_cfg_path} must return a Tool instance, got {type(tool_cfg)}"
        )
        return 1

    # Apply overrides
    for key, value in override_args.items():
        try:
            tool_cfg = tool_cfg.override(key, value)
        except Exception as e:
            console.print(f"[red]Error applying override {key}={value}:[/red] {e}")
            return 1

    # Seed random number generators if system config is available
    if hasattr(tool_cfg, "system"):
        seed_everything(tool_cfg.system)

    # Execute the tool
    console.print("\n[bold green]Running tool...[/bold green]\n")

    try:
        result = tool_cfg.invoke(func_args, list(override_args.items()))
    except KeyboardInterrupt:
        return 130  # Interrupted by Ctrl-C
    except Exception:
        logger.exception("Tool invocation failed")
        return 1

    return result if result is not None else 0


def cli_entry():
    """Entry point for console scripts."""
    sys.exit(main())


if __name__ == "__main__":
    cli_entry()
