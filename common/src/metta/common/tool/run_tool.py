#!/usr/bin/env -S uv run
"""Runner that takes a function that creates a ToolConfig,
invokes the function, and then runs the tool defined by the config."""

import argparse
import copy
import functools
import inspect
import json
import logging
import os
import signal
import sys
import warnings
from typing import Any

from pydantic import BaseModel, TypeAdapter
from rich.console import Console
from typing_extensions import TypeVar

from metta.common.tool import Tool
from metta.common.util.log_config import init_logging
from metta.mettagrid.config import Config
from metta.mettagrid.util.module import load_symbol
from metta.rl.system_config import seed_everything

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Environment setup
# --------------------------------------------------------------------------------------


def init_mettagrid_system_environment() -> None:
    """Initialize environment variables for headless operation."""
    # Set CUDA launch blocking for better error messages in development
    # TODO (use env for prod/dev?)
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    os.environ.setdefault("GLFW_PLATFORM", "osmesa")  # Use OSMesa as the GLFW backend
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("DISPLAY", "")

    # Suppress deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pygame.pkgdata")


T = TypeVar("T", bound=Config)

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def parse_value(value_str: str) -> Any:
    """Parse a string value into appropriate Python type (minimal heuristics)."""
    lower = value_str.lower()

    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None

    # Try to parse JSON containers
    if (value_str.startswith("{") and value_str.endswith("}")) or (
        value_str.startswith("[") and value_str.endswith("]")
    ):
        if len(value_str) > 1_000_000:  # 1MB limit
            logger.warning(f"Skipping JSON parsing for oversized value ({len(value_str)} chars)")
            return value_str
        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            pass

    # Try numeric
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass

    return value_str


def parse_cli_args(cli_args: list[str]) -> dict[str, Any]:
    """Parse CLI arguments in key=value format, keeping dotted keys flat."""
    parsed: dict[str, Any] = {}
    for arg in cli_args:
        # Unlike earlier versions, we no longer lstrip('-'); args should be plain key=value
        if "=" not in arg:
            raise ValueError(f"Invalid argument format: {arg}. Expected key=value")
        key, value = arg.split("=", 1)
        parsed[key] = parse_value(value)
    return parsed


def deep_merge(dst: dict, src: dict) -> dict:
    """In-place deep merge of src into dst."""
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def nestify(flat: dict[str, Any]) -> dict[str, Any]:
    """Turn {'a.b.c': 1, 'x': 2} into {'a': {'b': {'c': 1}}, 'x': 2}."""
    out: dict[str, Any] = {}
    for k, v in flat.items():
        parts = k.split(".")
        node = functools.reduce(lambda acc, p: {p: acc}, reversed(parts), v)
        deep_merge(out, node)
    return out


def get_tool_fields(tool_class: type[Tool]) -> set[str]:
    """Get all field names from a Tool class and its parent classes."""
    fields = set()
    for base in tool_class.__mro__:
        if base is Tool:
            break
        if issubclass(base, BaseModel) and hasattr(base, "model_fields"):
            fields.update(base.model_fields.keys())
    return fields


def get_function_params(make_tool_cfg: Any) -> set[str]:
    """Get the parameters of a function or callable (not used for Tool classes)."""
    if inspect.isclass(make_tool_cfg) and issubclass(make_tool_cfg, Tool):
        # Important: do NOT read Tool.__init__ for params (it's usually **data).
        return set()
    else:
        return set(inspect.signature(make_tool_cfg).parameters.keys())


def classify_remaining_args(remaining_args: dict[str, Any], tool_fields: set[str]) -> tuple[dict[str, Any], list[str]]:
    """Classify remaining arguments as tool overrides or unknown arguments."""
    overrides: dict[str, Any] = {}
    unknown: list[str] = []

    for key, value in remaining_args.items():
        if "." in key:
            base_key = key.split(".", 1)[0]
            if base_key in tool_fields:
                overrides[key] = value
            else:
                unknown.append(key)
        elif key in tool_fields:
            overrides[key] = value
        else:
            unknown.append(key)

    return overrides, unknown


def type_parse(value: Any, annotation: Any) -> Any:
    """Type-aware coercion using Pydantic when a function annotation is present."""
    if annotation is inspect._empty:
        return value
    adapter = TypeAdapter(annotation)
    return adapter.validate_python(value)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main():
    """Main entry point using argparse."""
    parser = argparse.ArgumentParser(
        description="Run a tool with automatic argument classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=True,
        epilog="""
Examples:
  %(prog)s experiments.recipes.arena.train run=test_123 trainer.total_timesteps=100000
  %(prog)s experiments.recipes.arena.play policy_uri=file://./checkpoints --verbose
  %(prog)s experiments.recipes.arena.train optim='{"lr":1e-3,"beta1":0.9}'

Rules:
  - Dotted keys (a.b.c) are configuration paths and will be nested and validated.
  - Exact parameter names are function arguments for factory functions.
  - Values: true/false, null/none, JSON containers {...}/[...], or int/float/string.
  - Tool args are plain key=value tokens. If you need to pass flags to the runner, use them
    before `--`. Put tool args after `--` if there is any ambiguity.

This script automatically determines which arguments are meant for the tool
constructor/function vs configuration overrides based on introspection.
        """,
    )

    parser.add_argument(
        "make_tool_cfg_path", help="Path to the function or Tool class (e.g., experiments.recipes.arena.train)"
    )
    parser.add_argument("args", nargs="*", help="Arguments in key=value format")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed argument classification")

    # Parse known args; keep unknowns to validate separation between runner flags and tool args
    known_args, unknown_args = parser.parse_known_args()

    # Initialize logging and environment
    init_logging()
    init_mettagrid_system_environment()
    console = Console()

    # Enforce: unknown long options (starting with '-') are considered runner flags and not tool args.
    # Require users to separate with `--` if they want to pass after runner options.
    dash_unknowns = [a for a in unknown_args if a.startswith("-")]
    if dash_unknowns:
        console.print(
            "[red]Error:[/red] Unknown runner option(s): "
            + ", ".join(dash_unknowns)
            + "\nUse `--` to separate runner options from tool args, e.g.:\n"
            + f"  {os.path.basename(sys.argv[0])} {known_args.make_tool_cfg_path} -- trainer.total_timesteps=100000"
        )
        return 2
    all_args = (known_args.args or []) + unknown_args

    # Exit on ctrl+c with proper exit code
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(130))  # 130 = interrupted by Ctrl-C

    # Parse CLI arguments
    try:
        cli_args = parse_cli_args(all_args)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        return 2  # Exit code 2 for usage errors

    # Build nested payload from dotted paths for Pydantic validation
    nested_cli = nestify(cli_args)

    console.print(f"\n[bold cyan]Loading tool:[/bold cyan] {known_args.make_tool_cfg_path}")

    # Load the tool configuration function/class
    try:
        make_tool_cfg = load_symbol(known_args.make_tool_cfg_path)
    except Exception as e:
        console.print(f"[red]Error loading {known_args.make_tool_cfg_path}:[/red] {e}")
        return 1

    # ----------------------------------------------------------------------------------
    # Construct the Tool
    #   - If class subclassing Tool (Pydantic model): validate the entire nested payload
    #   - If function: bind parameters from nested_cli/cli_args using annotations
    # ----------------------------------------------------------------------------------
    func_args_for_invoke: dict[str, str] = {}  # what we pass to tool.invoke (as strings)
    try:
        if inspect.isclass(make_tool_cfg) and issubclass(make_tool_cfg, Tool):
            if known_args.verbose and nested_cli:
                cls_name = make_tool_cfg.__name__
                console.print(f"\n[cyan]Creating {cls_name} from nested CLI payload:[/cyan]")
                for k in sorted(nested_cli.keys()):
                    console.print(f"  {k} = {nested_cli[k]}")
            tool_cfg = make_tool_cfg.model_validate(nested_cli)
            remaining_args = {}  # all dotted/top-level consumed by model validation
        else:
            # Factory function that returns a Tool
            sig = inspect.signature(make_tool_cfg)
            func_kwargs: dict[str, Any] = {}
            consumed_keys: set[str] = set()

            if known_args.verbose and (cli_args or nested_cli):
                func_name = getattr(make_tool_cfg, "__name__", str(make_tool_cfg))
                console.print(f"\n[cyan]Creating {func_name}:[/cyan]")

            for name, p in sig.parameters.items():
                # Prefer nested group if provided (e.g., param 'trainer' and CLI has 'trainer.*')
                if name in nested_cli:
                    provided = nested_cli[name]

                    # If the parameter has a default dict or BaseModel, start from it and merge overrides.
                    base: Any | None = None
                    if p.default is not inspect._empty:
                        default_val = p.default
                        if isinstance(default_val, dict) and isinstance(provided, dict):
                            base = copy.deepcopy(default_val)
                            deep_merge(base, provided)
                        elif isinstance(default_val, BaseModel) and isinstance(provided, dict):
                            base = default_val.model_copy(update=provided, deep=True)

                    data = base if base is not None else provided

                    # If annotated as a Pydantic model class, validate against it.
                    ann = p.annotation
                    try:
                        if inspect.isclass(ann) and issubclass(ann, BaseModel):
                            val = ann.model_validate(data)
                        else:
                            val = type_parse(data, ann)
                    except Exception:
                        # Fall back to raw data; better to surface error downstream than to crash here.
                        val = data

                    func_kwargs[name] = val

                    # Determine which keys actually contributed to nested_cli[name]
                    # If name exists as a flat key in cli_args, mark it as consumed
                    if name in cli_args:
                        consumed_keys.add(name)

                    # Mark all dotted keys that start with this parameter name as consumed
                    for k in cli_args.keys():
                        if k.startswith(name + "."):
                            consumed_keys.add(k)

                    if known_args.verbose:
                        console.print(f"  {name}={val!r}")
                    continue

                # Check for direct parameter match in flat CLI args
                if name in cli_args:
                    val = type_parse(cli_args[name], p.annotation)
                    func_kwargs[name] = val
                    consumed_keys.add(name)
                    if known_args.verbose:
                        console.print(f"  {name}={val!r}")

            # Construct via function
            tool_cfg = make_tool_cfg(**func_kwargs)

            # Remaining args = anything not consumed as function params
            remaining_args = {k: v for k, v in cli_args.items() if k not in consumed_keys}

            # For invoke(), send just the function args as strings
            func_args_for_invoke = {k: str(v) for k, v in func_kwargs.items()}
    except TypeError as e:
        # Provide a nicer hint when someone passes an unbound method (missing self/cls)
        msg = str(e)
        hint = ""
        if ("missing" in msg and "positional argument" in msg) and (" self" in msg or " cls" in msg):
            hint = (
                "\n[yellow]Hint:[/yellow] It looks like an unbound method was passed. "
                "Pass the Tool subclass itself or a factory function that doesn't require 'self'/'cls'."
            )
        console.print(f"[red]Error creating tool configuration:[/red] {e}{hint}")
        return 1
    except Exception as e:
        console.print(f"[red]Error creating tool configuration:[/red] {e}")
        return 1

    if not isinstance(tool_cfg, Tool):
        console.print(
            f"[red]Error:[/red] {known_args.make_tool_cfg_path} must return a Tool instance, got {type(tool_cfg)}"
        )
        return 1

    # ----------------------------------------------------------------------------------
    # Overrides & Unknowns (post-construction)
    # ----------------------------------------------------------------------------------
    tool_fields = get_tool_fields(type(tool_cfg))
    override_args, unknown_args = classify_remaining_args(remaining_args, tool_fields)

    if unknown_args:
        console.print(f"\n[red]Error: Unknown arguments:[/red] {', '.join(unknown_args)}")
        # Only show function params list if the entrypoint is a function
        if not (inspect.isclass(make_tool_cfg) and issubclass(make_tool_cfg, Tool)):
            console.print("\n[yellow]Available function parameters:[/yellow]")
            for param in get_function_params(make_tool_cfg):
                console.print(f"  - {param}")
        console.print("\n[yellow]Available tool fields for overrides:[/yellow]")
        for field in sorted(tool_fields):
            console.print(f"  - {field}")
        return 2  # Exit code 2 for usage errors

    if override_args:
        if known_args.verbose:
            console.print("\n[cyan]Applying overrides:[/cyan]")
            for key, value in override_args.items():
                console.print(f"  {key}={value}")
        for key, value in override_args.items():
            try:
                tool_cfg = tool_cfg.override(key, value)
            except Exception as e:
                console.print(f"[red]Error applying override {key}={value}:[/red] {e}")
                return 1

    # ----------------------------------------------------------------------------------
    # Seed & Run
    # ----------------------------------------------------------------------------------
    if hasattr(tool_cfg, "system"):
        seed_everything(tool_cfg.system)

    console.print("\n[bold green]Running tool...[/bold green]\n")

    try:
        result = tool_cfg.invoke(func_args_for_invoke)
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
