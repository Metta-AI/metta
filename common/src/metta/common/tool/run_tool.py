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
import tempfile
import traceback
import warnings
from typing import Any

from pydantic import BaseModel, TypeAdapter
from rich.console import Console
from typing_extensions import TypeVar

from metta.common.tool import Tool
from metta.common.tool.recipe import Recipe
from metta.common.tool.tool_registry import get_tool_registry
from metta.common.util.log_config import init_logging
from metta.common.util.text_styles import bold, cyan, green, red, yellow
from metta.rl.system_config import seed_everything
from mettagrid.base_config import Config
from mettagrid.util.module import load_symbol

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Environment setup
# --------------------------------------------------------------------------------------


def init_mettagrid_system_environment() -> None:
    """Initialize environment variables for headless operation."""
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
# Output handling
# --------------------------------------------------------------------------------------


def output_info(message: str) -> None:
    if sys.stdout.isatty():
        print(message)
    else:
        logger.info(message.strip())


def output_error(message: str) -> None:
    if sys.stdout.isatty():
        print(message, file=sys.stderr)
    else:
        logger.error(message.strip())


def output_exception(message: str) -> None:
    """Emit an error message along with a traceback when available."""
    if sys.stdout.isatty():
        output_error(message)
        traceback.print_exc()
    else:
        logger.exception(message.strip())


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


def get_function_params(tool_maker: Any) -> set[str]:
    """Get the parameters of a function or callable (not used for Tool classes)."""
    if inspect.isclass(tool_maker) and issubclass(tool_maker, Tool):
        # Important: do NOT read Tool.__init__ for params (it's usually **data).
        return set()
    else:
        return set(inspect.signature(tool_maker).parameters.keys())


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


def get_pydantic_field_info(model_class: type[BaseModel], prefix: str = "") -> list[tuple[str, str, Any, bool]]:
    """Recursively get field information from a Pydantic model.
    Returns list of (path, type_str, default, required) tuples.
    """
    fields_info = []

    for field_name, field in model_class.model_fields.items():
        field_path = f"{prefix}.{field_name}" if prefix else field_name
        annotation = field.annotation

        # Get the origin type if it's a generic
        origin = getattr(annotation, "__origin__", None)

        # Handle Optional types
        if origin is type(None):
            actual_type = annotation
        elif hasattr(annotation, "__args__"):
            # For Optional[X], Union[X, None], etc.
            args = getattr(annotation, "__args__", ())
            non_none_types = [arg for arg in args if arg is not type(None)]
            if non_none_types:
                actual_type = non_none_types[0] if len(non_none_types) == 1 else annotation
            else:
                actual_type = annotation
        else:
            actual_type = annotation

        # Check if it's a nested Pydantic model
        try:
            if inspect.isclass(actual_type) and issubclass(actual_type, BaseModel):
                # Add the parent field first (before nested fields for better ordering)
                type_name = actual_type.__name__
                # Don't show the full default object representation for complex models
                if field.default is not None and not callable(field.default):
                    default_val = f"<{type_name} instance>"
                else:
                    default_val = field.default_factory if field.default_factory else None
                is_required = field.is_required() if hasattr(field, "is_required") else (default_val is None)
                fields_info.append((field_path, type_name, default_val, is_required))

                # Then recursively get nested fields
                nested_fields = get_pydantic_field_info(actual_type, field_path)
                fields_info.extend(nested_fields)
            else:
                # Regular field
                type_name = getattr(actual_type, "__name__", str(actual_type))
                default_val = field.default if field.default is not None else field.default_factory
                is_required = field.is_required() if hasattr(field, "is_required") else (default_val is None)
                fields_info.append((field_path, type_name, default_val, is_required))
        except (TypeError, AttributeError):
            # For complex types that can't be inspected
            type_name = str(annotation).replace("typing.", "")
            default_val = field.default if field.default is not None else field.default_factory
            is_required = field.is_required() if hasattr(field, "is_required") else (default_val is None)
            fields_info.append((field_path, type_name, default_val, is_required))

    return fields_info


def list_tool_arguments(tool_maker: Any, console: Console) -> None:
    """List all available arguments for a tool."""
    console.print("\n[bold cyan]Available Arguments[/bold cyan]\n")

    if inspect.isclass(tool_maker) and issubclass(tool_maker, Tool):
        console.print("[yellow]Tool Configuration Fields:[/yellow]\n")

        fields_info = get_pydantic_field_info(tool_maker)
        grouped = {}
        for path, type_str, default, required in sorted(fields_info):
            top_level = path.split(".")[0]
            if top_level not in grouped:
                grouped[top_level] = []
            grouped[top_level].append((path, type_str, default, required))

        for top_level, fields in grouped.items():
            for path, type_str, default, required in fields:
                if path == top_level:
                    console.print(f"  [bold]{path}[/bold]", end="")
                else:
                    depth = path.count(".")
                    indent = "    " * depth
                    field_name = path.split(".")[-1]
                    console.print(f"{indent}{field_name}", end="")

                console.print(f": [dim]{type_str}[/dim]", end="")
                if not required and default is not None:
                    if callable(default):
                        console.print(" [green](default: <factory>)[/green]", end="")
                    elif isinstance(default, BaseModel):
                        console.print(f" [green](default: <{type(default).__name__}>)[/green]", end="")
                    elif isinstance(default, str) and (default.startswith("<") and default.endswith(">")):
                        pass
                    elif isinstance(default, str) and len(str(default)) > 100:
                        console.print(f" [green](default: {str(default)[:100]}...)[/green]", end="")
                    else:
                        console.print(f" [green](default: {default})[/green]", end="")
                elif required:
                    console.print(" [red](required)[/red]", end="")

                console.print()

            if top_level != list(grouped.keys())[-1]:
                console.print()

    else:
        console.print("[yellow]Function Parameters:[/yellow]\n")

        sig = inspect.signature(tool_maker)
        for name, param in sig.parameters.items():
            console.print(f"  [bold]{name}[/bold]", end="")

            if param.annotation is not inspect._empty:
                ann_str = str(param.annotation).replace("typing.", "")
                console.print(f": [dim]{ann_str}[/dim]", end="")
            if param.default is not inspect._empty:
                if isinstance(param.default, BaseModel):
                    console.print(f" [green](default: {type(param.default).__name__})[/green]", end="")
                elif callable(param.default):
                    console.print(" [green](default: <function>)[/green]", end="")
                else:
                    console.print(f" [green](default: {param.default})[/green]", end="")
            else:
                console.print(" [red](required)[/red]", end="")

            console.print()

            if param.annotation is not inspect._empty:
                try:
                    if inspect.isclass(param.annotation) and issubclass(param.annotation, BaseModel):
                        nested_fields = get_pydantic_field_info(param.annotation, name)
                        if nested_fields:
                            for path, type_str, default, required in sorted(nested_fields):
                                depth = path.count(".")
                                indent = "    " * (depth + 1)
                                field_name = path.split(".")[-1]
                                console.print(f"{indent}{field_name}: [dim]{type_str}[/dim]", end="")
                                if not required and default is not None:
                                    if callable(default):
                                        console.print(" [green](default: <factory>)[/green]", end="")
                                    else:
                                        console.print(f" [green](default: {default})[/green]", end="")
                                console.print()
                except (TypeError, AttributeError):
                    pass

        console.print("\n[yellow]Returned Tool Fields:[/yellow]\n")

        try:
            with tempfile.TemporaryDirectory():
                try:
                    result = tool_maker()
                except TypeError:
                    sig = inspect.signature(tool_maker)
                    kwargs = {}
                    for name, param in sig.parameters.items():
                        if param.default is inspect._empty:
                            kwargs[name] = None
                    try:
                        result = tool_maker(**kwargs)
                    except Exception:
                        console.print("  [dim]Unable to determine Tool fields (function requires runtime values)[/dim]")
                        return

                if isinstance(result, Tool):
                    fields_info = get_pydantic_field_info(type(result))
                    for path, type_str, default, required in sorted(fields_info):
                        depth = path.count(".")
                        indent = "    " * depth
                        field_name = path.split(".")[-1] if "." in path else path
                        console.print(f"{indent}{field_name}: [dim]{type_str}[/dim]", end="")
                        if not required and default is not None:
                            if callable(default):
                                console.print(" [green](default: <factory>)[/green]", end="")
                            else:
                                console.print(f" [green](default: {default})[/green]", end="")
                        console.print()
        except Exception:
            console.print("  [dim]Unable to determine Tool fields (function requires runtime values)[/dim]")

    console.print("\n[dim]Use these arguments with key=value format, e.g.:[/dim]")
    console.print(
        f"[dim]  ./tools/run.py {tool_maker.__module__}.{tool_maker.__name__} run=test trainer.batch_size=1024[/dim]"
    )


# --------------------------------------------------------------------------------------
# Tool Resolution
# --------------------------------------------------------------------------------------


def list_all_recipes(console: Console) -> None:
    """List all available recipes and their tools."""
    console.print("\n[bold cyan]Available Recipes:[/bold cyan]\n")

    recipes = Recipe.discover_all()
    if not recipes:
        console.print("[yellow]No recipes found.[/yellow]")
        return

    for recipe in sorted(recipes, key=lambda r: r.module_name):
        short_name = recipe.module_name.replace("experiments.recipes.", "")
        tool_names = recipe.get_all_tool_names()

        if tool_names:
            console.print(f"[bold]{short_name}[/bold]")
            for tool_name in sorted(tool_names):
                console.print(f"  └─ {tool_name}")
            console.print()


def list_module_tools(module_path: str, console: Console) -> bool:
    """List all tools available in a module. Returns True if successful."""
    # Try to load recipe with experiments.recipes prefix if needed
    candidates = [module_path]
    if not module_path.startswith("experiments.recipes."):
        candidates.append(f"experiments.recipes.{module_path}")

    for module_name in candidates:
        recipe = Recipe.load(module_name)
        if not recipe:
            continue

        tool_names = recipe.get_all_tool_names()

        # Display results
        console.print(f"\n[bold]Available tools in {module_name}:[/bold]\n")
        short_mod = module_name.replace("experiments.recipes.", "")
        for name in sorted(tool_names):
            console.print(f"  {short_mod}.{name}")
        return True

    return False


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main():
    """Main entry point using argparse."""
    parser = argparse.ArgumentParser(
        description="Run a tool with automatic argument classification",
        add_help=False,  # Custom help handling for tool-specific help
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=True,
        epilog="""
Examples:
  %(prog)s arena.train run=test_123                      # Shorthand (recommended)
  %(prog)s experiments.recipes.arena.train run=test_123  # Full path (still works)
  %(prog)s train arena run=test_123                      # Two-token syntax
  %(prog)s arena.play \
    policy_uri=file://./train_dir/my_run/checkpoints/my_run:v12.pt --verbose
  %(prog)s arena.train optim='{"lr":1e-3,"beta1":0.9}'

Common tools:
  train           - Train a new policy
  play            - Interactive browser-based gameplay
  replay          - View recorded gameplay
  evaluate        - Run evaluation suite (aliases: eval, sim)
  evaluate_remote - Remote evaluation (aliases: eval_remote, sim_remote)

Recipe requirements:
  - Define mettagrid() -> MettaGridConfig for basic functionality
  - Define simulations() -> list[SimulationConfig] for custom evaluations
  - Or define explicit tool functions for full control

Advanced:
  %(prog)s arena.train -h                           # List all arguments
  %(prog)s arena.train --dry-run                    # Validate without running
  %(prog)s arena.train run=test trainer.lr=0.001    # Override nested config

This script automatically determines which arguments are meant for the tool
constructor/function vs configuration overrides based on introspection.
        """,
    )

    parser.add_argument(
        "tool_path",
        nargs="?",
        help=(
            "Path or shorthand to the tool maker (function or Tool class). Examples: "
            "'experiments.recipes.arena.train', 'arena.train', or two-part "
            "'train arena' (equivalent to 'arena.train')."
        ),
    )
    parser.add_argument("args", nargs="*", help="Arguments in key=value format")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed argument classification")
    parser.add_argument("--dry-run", action="store_true", help="Validate the args and exit")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List tools defined by the resolved recipe module and exit",
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show help and list all available arguments for the tool"
    )

    # Parse known args; keep unknowns to validate separation between runner flags and tool args
    known_args, unknown_args = parser.parse_known_args()
    console = Console()

    # If help is requested without a tool path, show general help
    if known_args.help and not known_args.tool_path:
        parser.print_help()
        return 0

    # Initialize logging and environment
    init_logging()
    init_mettagrid_system_environment()

    # Enforce: unknown long options (starting with '-') are considered runner flags and not tool args.
    # Require users to separate with `--` if they want to pass after runner options.
    dash_unknowns = [a for a in unknown_args if a.startswith("-")]
    if dash_unknowns:
        output_error(
            f"{red('Error:')} Unknown runner option(s): "
            + ", ".join(dash_unknowns)
            + "\nUse `--` to separate runner options from tool args, e.g.:\n"
            + f"  {os.path.basename(sys.argv[0])} {known_args.tool_path} -- trainer.total_timesteps=100000"
        )
        return 2
    # Support shorthand syntax for tool path:
    #  - Allow omitting 'experiments.recipes.' prefix, e.g. 'arena.train'
    #  - Allow two-part form 'train arena' as sugar for 'arena.train'

    # Exit on ctrl+c with proper exit code
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(130))  # 130 = interrupted by Ctrl-C

    tool_path = known_args.tool_path
    raw_positional_args: list[str] = list(known_args.args or [])

    # Handle --list early
    if known_args.list:
        # If no tool_path provided, list all recipes
        if not tool_path:
            list_all_recipes(console)
            return 0

        # Check if it's a bare tool name (like 'train', 'evaluate')
        # Get canonical tool name (handles aliases)
        registry = get_tool_registry()
        tool_name_map = registry._get_tool_name_map()
        canonical_tool = tool_name_map.get(tool_path, tool_path)

        # If it's a known tool type, list all recipes that support it
        if canonical_tool in registry.get_all_tools():
            console.print(f"\n[bold]Recipes supporting '{tool_path}':[/bold]\n")
            recipes = Recipe.discover_all()
            found_any = False

            for recipe in sorted(recipes, key=lambda r: r.module_name):
                tools = recipe.get_tools_for_canonical(canonical_tool)
                if tools:
                    short_name = recipe.module_name.replace("experiments.recipes.", "")
                    # Show all function names that provide this tool
                    for func_name, _ in tools:
                        console.print(f"  {short_name}.{func_name}")
                    found_any = True

            if not found_any:
                console.print(f"[yellow]No recipes found supporting '{tool_path}'[/yellow]")
            return 0

        # Otherwise try to list tools in specific recipe
        if list_module_tools(tool_path, console):
            return 0
        # If listing failed, continue to show error below

    # Try two-part form first if next arg looks like a module name (not key=value)
    registry = get_tool_registry()
    tool_maker = None
    args_consumed = 0

    if raw_positional_args and ("=" not in raw_positional_args[0]) and (not raw_positional_args[0].startswith("-")):
        # Try 'train arena' → 'arena.train'
        two_part_path = f"{raw_positional_args[0]}.{tool_path}"
        tool_maker = registry.resolve_and_load_tool(two_part_path)
        if tool_maker:
            args_consumed = 1

    # If two-part didn't work, try single form
    if not tool_maker:
        tool_maker = registry.resolve_and_load_tool(tool_path)

    # Rebuild the arg list to parse (skip consumed args)
    all_args = raw_positional_args[args_consumed:] + unknown_args

    # Parse CLI arguments
    try:
        cli_args = parse_cli_args(all_args)
    except ValueError as e:
        output_error(f"{red('Error:')} {e}")
        return 2  # Exit code 2 for usage errors

    # Build nested payload from dotted paths for Pydantic validation
    nested_cli = nestify(cli_args)

    if tool_maker is None:
        output_error(f"{red('Error:')} Could not find tool '{tool_path}'")

        # Try to provide helpful hints
        if "." in tool_path:
            module_part = tool_path.rsplit(".", 1)[0]
            # Try adding experiments.recipes prefix if not present
            if not module_part.startswith("experiments.recipes."):
                module_part = f"experiments.recipes.{module_part}"

            recipe = Recipe.load(module_part)
            if recipe:
                mg, sims = recipe.get_configs()
                if mg is not None or sims is not None:
                    registry = get_tool_registry()
                    output_info(f"\n{yellow('Hint:')} Recipe module exists but tool not found.")
                    output_info(f"Available inferred tools: {', '.join(registry.get_tool_display_names())}")

        return 1

    output_info(f"\n{bold(cyan('Loading tool:'))} {tool_maker.__module__}.{tool_maker.__name__}")

    # If help flag is set, list arguments and exit
    if known_args.help:
        list_tool_arguments(tool_maker, console)
        return 0

    # ----------------------------------------------------------------------------------
    # Construct the Tool
    #   - If class subclassing Tool (Pydantic model): validate the entire nested payload
    #   - If function: bind parameters from nested_cli/cli_args using annotations
    # ----------------------------------------------------------------------------------
    func_args_for_invoke: dict[str, str] = {}  # what we pass to tool.invoke (as strings)
    try:
        if inspect.isclass(tool_maker) and issubclass(tool_maker, Tool):
            if known_args.verbose and nested_cli:
                cls_name = tool_maker.__name__
                output_info(f"\n{cyan(f'Creating {cls_name} from nested CLI payload:')}")
                for k in sorted(nested_cli.keys()):
                    output_info(f"  {k} = {nested_cli[k]}")
            tool_cfg = tool_maker.model_validate(nested_cli)
            remaining_args = {}  # all dotted/top-level consumed by model validation
        else:
            # Factory function that returns a Tool
            sig = inspect.signature(tool_maker)
            func_kwargs: dict[str, Any] = {}
            consumed_keys: set[str] = set()

            if known_args.verbose and (cli_args or nested_cli):
                func_name = getattr(tool_maker, "__name__", str(tool_maker))
                output_info(f"\n{cyan(f'Creating {func_name}:')}")

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
                        output_info(f"  {name}={val!r}")
                    continue

                # Check for direct parameter match in flat CLI args
                if name in cli_args:
                    val = type_parse(cli_args[name], p.annotation)
                    func_kwargs[name] = val
                    consumed_keys.add(name)
                    if known_args.verbose:
                        output_info(f"  {name}={val!r}")

            # Construct via function
            tool_cfg = tool_maker(**func_kwargs)

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
                f"\n{yellow('Hint:')} It looks like an unbound method was passed. "
                "Pass the Tool subclass itself or a factory function that doesn't require 'self'/'cls'."
            )
        output_exception(f"{red('Error creating tool configuration:')} {e}{hint}")
        return 1
    except Exception as e:
        output_exception(f"{red('Error creating tool configuration:')} {e}")
        return 1

    if not isinstance(tool_cfg, Tool):
        output_error(f"{red('Error:')} {known_args.tool_path} must return a Tool instance, got {type(tool_cfg)}")
        return 1

    # ----------------------------------------------------------------------------------
    # Overrides & Unknowns (post-construction)
    # ----------------------------------------------------------------------------------
    tool_fields = get_tool_fields(type(tool_cfg))
    override_args, unknown_args = classify_remaining_args(remaining_args, tool_fields)

    if unknown_args:
        output_info(f"\n{red('Error: Unknown arguments:')} {', '.join(unknown_args)}")
        # Only show function params list if the entrypoint is a function
        if not (inspect.isclass(tool_maker) and issubclass(tool_maker, Tool)):
            output_info(f"\n{yellow('Available function parameters:')}")
            for param in get_function_params(tool_maker):
                output_info(f"  - {param}")
        output_info(f"\n{yellow('Available tool fields for overrides:')}")
        for field in sorted(tool_fields):
            output_info(f"  - {field}")
        return 2  # Exit code 2 for usage errors

    if override_args:
        if known_args.verbose:
            output_info(f"\n{cyan('Applying overrides:')}")
            for key, value in override_args.items():
                output_info(f"  {key}={value}")
        for key, value in override_args.items():
            try:
                tool_cfg = tool_cfg.override(key, value)
            except Exception as e:
                output_exception(f"{red('Error applying override')} {key}={value}: {e}")
                return 1

    # ----------------------------------------------------------------------------------
    # Dry run check - exit here if --dry-run flag is set
    # ----------------------------------------------------------------------------------
    if known_args.dry_run:
        output_info(f"\n{bold(green('✅ Configuration validation successful'))}")
        if known_args.verbose:
            output_info(f"Tool type: {type(tool_cfg).__name__}")
            output_info(f"Module: {tool_maker.__module__}.{tool_maker.__name__}")
        return 0

    # ----------------------------------------------------------------------------------
    # Seed & Run
    # ----------------------------------------------------------------------------------
    if hasattr(tool_cfg, "system"):
        seed_everything(tool_cfg.system)

    output_info(f"\n{bold(green('Running tool...'))}\n")

    try:
        result = tool_cfg.invoke(func_args_for_invoke)
    except KeyboardInterrupt:
        return 130  # Interrupted by Ctrl-C
    except Exception:
        output_exception(red("Tool invocation failed"))
        return 1

    return result if result is not None else 0


def cli_entry():
    """Entry point for console scripts."""
    sys.exit(main())


if __name__ == "__main__":
    cli_entry()
