#!/usr/bin/env -S uv run
import inspect
import logging
import os
import signal
import warnings
from typing import Any, cast

import typer
from omegaconf import OmegaConf
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
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Set environment variables to run without display
    os.environ["GLFW_PLATFORM"] = "osmesa"  # Use OSMesa as the GLFW backend
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["DISPLAY"] = ""

    # Suppress deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pygame.pkgdata")


T = TypeVar("T", bound=Config)


def parse_cli_args(cli_args: list[str]) -> dict[str, Any]:
    """Parse CLI arguments in key=value format."""
    parsed = OmegaConf.to_container(OmegaConf.from_cli(cli_args))
    assert isinstance(parsed, dict)
    return cast(dict[str, Any], parsed)


def get_tool_fields(tool_class: type[Tool]) -> set[str]:
    """Get all field names from a Tool class and its parent classes."""
    fields = set()

    # Get fields from the tool class itself
    if hasattr(tool_class, "model_fields"):
        fields.update(tool_class.model_fields.keys())

    # Walk up the MRO to get fields from parent classes
    for base in tool_class.__mro__:
        if base is Tool or not issubclass(base, BaseModel):
            continue
        if hasattr(base, "model_fields"):
            fields.update(base.model_fields.keys())

    return fields


def split_args_and_overrides(
    cli_args: dict[str, Any], make_tool_cfg: Any, tool_cfg: Tool
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """
    Split CLI arguments into function args and tool overrides.

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
        func_params = set(inspect.signature(make_tool_cfg.__init__).parameters.keys())
        func_params.discard("self")
    else:
        # Function that returns a Tool
        func_params = set(inspect.signature(make_tool_cfg).parameters.keys())

    # Get tool field names (including nested fields)
    tool_fields = get_tool_fields(type(tool_cfg))

    # Process each CLI argument
    for key, value in cli_args.items():
        # Check if it's a nested field (contains dots)
        base_key = key.split(".")[0]

        if key in func_params or base_key in func_params:
            # Direct function parameter
            func_args[key] = value
        elif base_key in tool_fields or "." in key:
            # Tool field or nested field
            overrides[key] = value
        else:
            # Unknown argument
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


def main(
    make_tool_cfg_path: str = typer.Argument(  # noqa: B008
        ..., help="Path to the function to run (e.g., experiments.recipes.arena.train)"
    ),
    args: list[str] | None = typer.Argument(default=None, help="Arguments in key=value format"),  # noqa: B008
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed argument classification"),  # noqa: B008
):
    """Run a tool with automatic argument classification.

    This tool automatically determines which arguments are meant for the tool
    constructor/function vs which are configuration overrides.

    Examples:

        ./tools/run.py experiments.recipes.arena.train run=test_123 trainer.total_timesteps=100000

        ./tools/run.py experiments.recipes.arena.play policy_uri=file://./checkpoints
    """
    # Initialize
    init_logging()
    init_mettagrid_system_environment()

    # Exit on ctrl+c
    signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

    # Parse CLI arguments
    cli_args = parse_cli_args(args or [])

    console.print(f"\n[bold cyan]Loading tool:[/bold cyan] {make_tool_cfg_path}")

    # Load the tool configuration function/class
    try:
        make_tool_cfg = load_symbol(make_tool_cfg_path)
    except Exception as e:
        console.print(f"[red]Error loading {make_tool_cfg_path}:[/red] {e}")
        raise typer.Exit(1) from e

    # Create initial tool config to inspect its fields
    if inspect.isclass(make_tool_cfg) and issubclass(make_tool_cfg, Tool):
        # Tool class constructor - create with no args first
        temp_tool_cfg = make_tool_cfg()
    else:
        # Function that makes a tool - call with no args or defaults
        sig = inspect.signature(make_tool_cfg)
        temp_args = {}
        for param_name, param in sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                temp_args[param_name] = param.default
            elif param_name in cli_args:
                temp_args[param_name] = cli_args[param_name]
        temp_tool_cfg = make_tool_cfg(**temp_args)

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
        raise typer.Exit(1)

    # Display classification if verbose
    if verbose:
        display_arg_classification(func_args, override_args, unknown_args, make_tool_cfg_path)

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
        raise typer.Exit(1) from e

    if not isinstance(tool_cfg, Tool):
        console.print(f"[red]Error:[/red] {make_tool_cfg_path} must return a Tool instance, got {type(tool_cfg)}")
        raise typer.Exit(1)

    # Apply overrides
    for key, value in override_args.items():
        try:
            tool_cfg = tool_cfg.override(key, value)
        except Exception as e:
            console.print(f"[red]Error applying override {key}={value}:[/red] {e}")
            raise typer.Exit(1) from e

    # Seed random number generators
    seed_everything(tool_cfg.system)

    # Execute the tool
    console.print("\n[bold green]Running tool...[/bold green]\n")
    result = tool_cfg.invoke(func_args, list(override_args.items()))

    if result is not None:
        raise typer.Exit(result)


if __name__ == "__main__":
    typer.run(main)
