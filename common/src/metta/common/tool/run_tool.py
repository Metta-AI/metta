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
from typing import Any, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter
from rich.console import Console
from typing_extensions import TypeVar

from metta.common.tool import Tool
from metta.common.util.log_config import init_logging
from metta.common.util.text_styles import bold, cyan, green, red, yellow
from metta.rl.system_config import seed_everything
from mettagrid.config import Config
from mettagrid.util.module import load_symbol

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Tool type mappings for verb validation
# --------------------------------------------------------------------------------------


# Mapping from verb to expected Tool type (lazy loading to avoid circular imports)
def get_tool_type_mapping() -> dict[str, Type[Tool]]:
    """Get mapping from verb to expected Tool type. Lazy loading to avoid circular imports.

    These map directly to metta/tools/*.py - no aliases.
    """
    from metta.tools.analyze import AnalysisTool
    from metta.tools.play import PlayTool
    from metta.tools.replay import ReplayTool
    from metta.tools.sim import SimTool
    from metta.tools.train import TrainTool

    return {
        "train": TrainTool,
        "sim": SimTool,
        "analyze": AnalysisTool,
        "play": PlayTool,
        "replay": ReplayTool,
    }


def get_tool_return_type(func: Any) -> Optional[Type[Tool]]:
    """Extract the Tool return type from a function's type annotation.

    Returns None if the function doesn't have a Tool return type annotation.
    """
    if not callable(func):
        return None

    try:
        sig = inspect.signature(func)
        return_annotation = sig.return_annotation

        # Handle direct Tool subclass
        if inspect.isclass(return_annotation) and issubclass(return_annotation, Tool):
            return return_annotation

        # Handle Optional[Tool] or Union[Tool, None]
        origin = get_origin(return_annotation)
        if origin is Union:
            args = get_args(return_annotation)
            for arg in args:
                if arg is not type(None) and inspect.isclass(arg) and issubclass(arg, Tool):
                    return arg

        return None
    except Exception:
        return None


def validate_tool_type(func_path: str, func: Any, expected_tool_type: Optional[Type[Tool]] = None) -> Type[Tool]:
    """Validate that a function returns the expected Tool type.

    Args:
        func_path: The path to the function (for error messages)
        func: The function to validate
        expected_tool_type: The expected Tool type (if specified by verb)

    Returns:
        The actual Tool return type

    Raises:
        TypeError: If validation fails
    """
    actual_type = get_tool_return_type(func)

    if actual_type is None:
        raise TypeError(
            f"{func_path} must have a return type annotation that is a Tool subclass. "
            f"Got: {inspect.signature(func).return_annotation}"
        )

    if expected_tool_type and not issubclass(actual_type, expected_tool_type):
        # Get the verb name for better error message
        tool_mapping = get_tool_type_mapping()
        verb = None
        for v, t in tool_mapping.items():
            if t == expected_tool_type:
                verb = v
                break

        raise TypeError(
            f"{func_path} returns {actual_type.__name__} but '{verb}' command expects {expected_tool_type.__name__}"
        )

    return actual_type


def wrap_config_to_tool(obj: Any, expected_tool_type: Optional[Type[Tool]], cli_args: dict[str, Any]) -> Any:
    """Wrap non-Tool return types into appropriate Tool instances.

    Handles:
    - TrainerConfig -> TrainTool
    - SimulationConfig -> PlayTool/ReplayTool/SimTool (based on expected type)
    - Sequence[SimulationConfig] -> SimTool
    - AnalysisConfig -> AnalysisTool
    - MettaGridConfig -> appropriate Tool (based on expected type)
    """
    from collections.abc import Sequence as SeqABC

    # If already a Tool, return as-is
    if isinstance(obj, Tool):
        return obj

    # Lazy imports to avoid circular dependencies
    from metta.rl.trainer_config import TrainerConfig
    from metta.sim.simulation_config import SimulationConfig
    from mettagrid.config.mettagrid_config import MettaGridConfig

    # Handle TrainerConfig -> TrainTool
    if isinstance(obj, TrainerConfig):
        from metta.tools.train import TrainTool

        return TrainTool(config=obj)

    # Handle SimulationConfig -> appropriate Tool
    if isinstance(obj, SimulationConfig):
        from metta.tools.play import PlayTool
        from metta.tools.replay import ReplayTool
        from metta.tools.sim import SimTool

        if expected_tool_type is ReplayTool:
            return ReplayTool(config=obj)
        elif expected_tool_type is PlayTool:
            return PlayTool(config=obj)
        elif expected_tool_type is SimTool:
            # SimTool needs policy_uri
            policy_uri = cli_args.get("policy_uri")
            if not policy_uri:
                output_error(f"{red('Error:')} evaluate/sim requires policy_uri parameter")
                sys.exit(1)
            return SimTool(config=[obj], policy_uri=policy_uri)
        # Default for simulation without expected type
        return PlayTool(config=obj)

    # Handle Sequence[SimulationConfig] -> SimTool
    if isinstance(obj, SeqABC) and obj and all(isinstance(s, SimulationConfig) for s in obj):
        from metta.tools.sim import SimTool

        policy_uri = cli_args.get("policy_uri")
        if not policy_uri:
            output_error(f"{red('Error:')} evaluate/sim requires policy_uri parameter")
            sys.exit(1)
        return SimTool(config=list(obj), policy_uri=policy_uri)

    # Handle AnalysisConfig -> AnalysisTool
    try:
        from metta.eval.analysis_config import AnalysisConfig

        if isinstance(obj, AnalysisConfig):
            from metta.tools.analyze import AnalysisTool

            return AnalysisTool(config=obj)
    except ImportError:
        pass  # AnalysisConfig might not exist

    # Handle MettaGridConfig -> appropriate Tool
    if isinstance(obj, MettaGridConfig):
        from metta.tools.play import PlayTool
        from metta.tools.replay import ReplayTool
        from metta.tools.sim import SimTool
        from metta.tools.train import TrainTool

        # Create a simulation config from the MettaGridConfig
        sim_cfg = SimulationConfig(env=obj, name=obj.label or "mettagrid")

        if expected_tool_type is TrainTool:
            # Create curriculum and trainer config
            from metta.cogworks.curriculum import single_task_curriculum
            from metta.rl.trainer_config import TrainerConfig

            curriculum_cfg = single_task_curriculum(obj)
            trainer_cfg = TrainerConfig(curriculum=curriculum_cfg)
            return TrainTool(config=trainer_cfg)
        elif expected_tool_type is ReplayTool:
            return ReplayTool(config=sim_cfg)
        elif expected_tool_type is PlayTool:
            return PlayTool(config=sim_cfg)
        elif expected_tool_type is SimTool:
            # SimTool needs policy_uri
            policy_uri = cli_args.get("policy_uri")
            if not policy_uri:
                output_error(f"{red('Error:')} evaluate/sim requires policy_uri parameter")
                sys.exit(1)
            return SimTool(config=[sim_cfg], policy_uri=policy_uri)
        # Default for MettaGridConfig without expected type
        return PlayTool(config=sim_cfg)

    # Return unchanged if no wrapper applies
    return obj


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
    """Get all field names from a Tool class.

    Pydantic v2 includes inherited fields in subclass.model_fields,
    so we can simply read them directly.
    """
    return set(getattr(tool_class, "model_fields", {}).keys())


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


def list_tool_arguments(make_tool_cfg: Any, console: Console) -> None:
    """List all available arguments for a tool."""
    console.print("\n[bold cyan]Available Arguments[/bold cyan]\n")

    if inspect.isclass(make_tool_cfg) and issubclass(make_tool_cfg, Tool):
        console.print("[yellow]Tool Configuration Fields:[/yellow]\n")

        fields_info = get_pydantic_field_info(make_tool_cfg)
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

        sig = inspect.signature(make_tool_cfg)
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
                    result = make_tool_cfg()
                except TypeError:
                    sig = inspect.signature(make_tool_cfg)
                    kwargs = {}
                    for name, param in sig.parameters.items():
                        if param.default is inspect._empty:
                            kwargs[name] = None
                    try:
                        result = make_tool_cfg(**kwargs)
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
        f"[dim]  ./tools/run.py {make_tool_cfg.__module__}.{make_tool_cfg.__name__} "
        f"run=test trainer.batch_size=1024[/dim]"
    )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def preprocess_recipe_path(path: str) -> tuple[str, Optional[Type[Tool]]]:
    """Convert recipe syntax to full module path and expected tool type.

    Handles several formats:
    1. Verb + recipe: "train arena" -> (experiments.recipes.arena.train, TrainTool)
    2. Direct function path: "arena.train_shaped" -> (experiments.recipes.arena.train_shaped, None)
    3. Subfolder: "train in_context_learning.ordered_chains" ->
       (experiments.recipes.in_context_learning.ordered_chains.train, TrainTool)
    4. Dotted path with function: "replay scratchpad.ci.replay_null" ->
       (experiments.recipes.scratchpad.ci.replay_null, ReplayTool)
    5. Single path needing prefix: "in_context_learning.ordered_chains.train" ->
       (in_context_learning.ordered_chains.train, None)

    Returns:
        A tuple of (module_path, expected_tool_type)
        - module_path: The full module path to the function
        - expected_tool_type: The expected Tool type if a verb was specified, None for direct paths

    Examples:
        train arena -> (experiments.recipes.arena.train, TrainTool)
        arena.train_shaped -> (experiments.recipes.arena.train_shaped, None)
        train in_context_learning.ordered_chains ->
            (experiments.recipes.in_context_learning.ordered_chains.train, TrainTool)
        sim navigation -> (experiments.recipes.navigation.sim, SimTool)
        play minimal -> (experiments.recipes.minimal.play, PlayTool)
        replay scratchpad.ci.replay_null -> (experiments.recipes.scratchpad.ci.replay_null, ReplayTool)
    """
    # Known tool names - these map directly to metta/tools/*.py
    # No aliases allowed - only actual tool names
    TOOL_MAPPINGS = {
        "train": "train",
        "sim": "sim",
        "analyze": "analyze",
        "play": "play",
        "replay": "replay",
    }

    parts = path.split()

    # Handle two-part syntax: "verb recipe_or_path"
    if len(parts) == 2 and parts[0] in TOOL_MAPPINGS:
        tool_name, recipe_path = parts
        tool_type = get_tool_type_mapping().get(tool_name)

        # Check if the recipe_path looks like it already has a function name at the end
        # e.g., "scratchpad.ci.replay_null" where replay_null is the function
        path_parts = recipe_path.split(".")
        if len(path_parts) > 1:
            last_part = path_parts[-1]
            # If the last part matches the tool name followed by underscore (e.g., replay_null for replay tool),
            # treat it as a specific function name to preserve
            if last_part.startswith(f"{tool_name}_"):
                # Looks like a function name for this tool, preserve it
                return f"experiments.recipes.{recipe_path}", tool_type

        # Otherwise, build the standard path with tool function
        return f"experiments.recipes.{recipe_path}.{TOOL_MAPPINGS[tool_name]}", tool_type

    # Handle single-path syntax (direct function path like "arena.train_shaped")
    # No verb specified, so no expected tool type - will validate based on return annotation
    # This path might need experiments.recipes prefix, which will be handled by the loader
    return path, None


def main():
    """Main entry point using argparse."""
    parser = argparse.ArgumentParser(
        description="Run a tool with automatic argument classification",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=True,
        epilog="""
Examples:
  %(prog)s train arena run=test_123 trainer.total_timesteps=100000
  %(prog)s evaluate navigation policy_uri=file://./checkpoints
  %(prog)s play arena policy_uri=file://./train_dir/my_run/checkpoints/my_run:v12.pt
  %(prog)s experiments.recipes.arena.train run=test_123  # Full path also works

Rules:
  - Short syntax: "train arena" -> experiments.recipes.arena.train
  - Dotted keys (a.b.c) are configuration paths and will be nested and validated.
  - Exact parameter names are function arguments for factory functions.
  - Values: true/false, null/none, JSON containers {...}/[...], or int/float/string.
  - Tool args are plain key=value tokens. If you need to pass flags to the runner, use them
    before `--`. Put tool args after `--` if there is any ambiguity.

This script automatically determines which arguments are meant for the tool
constructor/function vs configuration overrides based on introspection.
        """,
    )

    parser.add_argument("make_tool_cfg_path", help="Tool and recipe (e.g., 'train arena') or full path", nargs="+")
    parser.add_argument("args", nargs="*", help="Arguments in key=value format")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed argument classification")
    parser.add_argument("--dry-run", action="store_true", help="Validate the args and exit")
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show help and list all available arguments for the tool"
    )

    # Parse known args; keep unknowns to validate separation between runner flags and tool args
    known_args, unknown_args = parser.parse_known_args()
    console = Console()

    # If help is requested without a tool path, show general help
    if known_args.help and not known_args.make_tool_cfg_path:
        console.print("[bold]Tool Runner[/bold]\n")
        console.print("Usage: ./tools/run.py <tool_path> [arguments]\n")
        console.print("  tool_path: Path to the function or Tool class (e.g., experiments.recipes.arena.train)")
        console.print("  arguments: Arguments in key=value format\n")
        console.print("Options:")
        console.print("  -h, --help     Show help and list all available arguments for the tool")
        console.print("  -v, --verbose  Show detailed argument classification")
        console.print("  --dry-run      Validate the args and exit\n")
        console.print("Examples:")
        console.print("  ./tools/run.py experiments.recipes.arena.train -h")
        console.print("  ./tools/run.py experiments.recipes.arena.train run=test trainer.batch_size=1024")
        return 0

    # Handle the path as either a list (short syntax) or join it into a single string
    # Separate tool/recipe from arguments (which contain '=')
    path_parts = []
    extra_args = []
    for part in known_args.make_tool_cfg_path:
        if "=" in part:
            extra_args.append(part)
        else:
            # Once we hit an arg, rest are args too
            if extra_args:
                extra_args.append(part)
            else:
                path_parts.append(part)

    # Prepend extra args to the args list
    if extra_args:
        known_args.args = extra_args + (known_args.args or [])

    path_str = " ".join(path_parts)

    # Preprocess the path to handle short syntax and get expected tool type
    make_tool_cfg_path, expected_tool_type = preprocess_recipe_path(path_str)

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
            + f"  {os.path.basename(sys.argv[0])} {path_str} -- trainer.total_timesteps=100000"
        )
        return 2
    all_args = (known_args.args or []) + unknown_args

    # Exit on ctrl+c with proper exit code
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(130))  # 130 = interrupted by Ctrl-C

    # Parse CLI arguments
    try:
        cli_args = parse_cli_args(all_args)
    except ValueError as e:
        output_error(f"{red('Error:')} {e}")
        return 2  # Exit code 2 for usage errors

    # Build nested payload from dotted paths for Pydantic validation
    nested_cli = nestify(cli_args)

    output_info(f"\n{bold(cyan('Loading tool:'))} {make_tool_cfg_path}")

    # Load the tool configuration function/class
    # Try multiple fallback strategies:
    # 1. Try as-is
    # 2. If single-path input and not found, try with experiments.recipes prefix
    # 3. If it's a recipe function, try mettagrid_recipe as fallback
    make_tool_cfg = None
    original_error = None

    try:
        make_tool_cfg = load_symbol(make_tool_cfg_path)
    except (AttributeError, ImportError) as e:
        original_error = e

        # Strategy 1: If it doesn't start with experiments.recipes, try adding that prefix
        if not make_tool_cfg_path.startswith("experiments.recipes."):
            prefixed_path = f"experiments.recipes.{make_tool_cfg_path}"
            try:
                output_info(f"  {yellow('Trying with prefix:')} {prefixed_path}")
                make_tool_cfg = load_symbol(prefixed_path)
                make_tool_cfg_path = prefixed_path  # Update path for later use
            except Exception as prefix_error:
                if known_args.verbose:
                    output_info(f"  {yellow('Prefix failed:')} {prefix_error}")
                pass  # Continue to next fallback

        # Strategy 2: Check if this is a recipe function that might have a mettagrid fallback
        # Check if the path ends with a tool name (train, sim, play, replay, analyze)
        tool_names = {"train", "sim", "play", "replay", "analyze"}
        last_part = make_tool_cfg_path.split(".")[-1] if "." in make_tool_cfg_path else ""

        if make_tool_cfg is None and ".recipes." in make_tool_cfg_path and last_part in tool_names:
            # Don't fall back for analyze (it needs specific config)
            if last_part != "analyze":
                # Try loading mettagrid instead
                fallback_path = make_tool_cfg_path.rsplit(".", 1)[0] + ".mettagrid"
                try:
                    output_info(f"  {yellow('Trying fallback:')} {fallback_path}")
                    make_tool_cfg = load_symbol(fallback_path)
                    # Store which tool was originally requested so we can wrap appropriately
                    make_tool_cfg._requested_tool = last_part  # Tag for later wrapping
                except Exception:
                    pass  # Will report original error below

        # If all strategies failed, report the original error
        if make_tool_cfg is None:
            output_exception(f"{red('Error loading')} {make_tool_cfg_path}: {original_error}")
            return 1
    except Exception as e:
        output_exception(f"{red('Error loading')} {make_tool_cfg_path}: {e}")
        return 1

    # If help flag is set, list arguments and exit
    if known_args.help:
        list_tool_arguments(make_tool_cfg, console)
        return 0

    # ----------------------------------------------------------------------------------
    # Validate Tool type if this is a function returning a Tool
    # ----------------------------------------------------------------------------------
    if callable(make_tool_cfg) and not inspect.isclass(make_tool_cfg):
        # Skip validation for mettagrid fallback functions (they return MettaGridConfig, not Tool)
        if not hasattr(make_tool_cfg, "_requested_tool"):
            try:
                # Validate that the function returns an appropriate Tool type
                actual_tool_type = validate_tool_type(make_tool_cfg_path, make_tool_cfg, expected_tool_type)
                if known_args.verbose:
                    output_info(f"  {green('Function returns:')} {actual_tool_type.__name__}")
            except TypeError as e:
                output_error(f"{red('Error:')} {e}")
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
                output_info(f"\n{cyan(f'Creating {cls_name} from nested CLI payload:')}")
                for k in sorted(nested_cli.keys()):
                    output_info(f"  {k} = {nested_cli[k]}")
            tool_cfg = make_tool_cfg.model_validate(nested_cli)
            remaining_args = {}  # all dotted/top-level consumed by model validation
        else:
            # Factory function that returns a Tool
            sig = inspect.signature(make_tool_cfg)
            func_kwargs: dict[str, Any] = {}
            consumed_keys: set[str] = set()

            if known_args.verbose and (cli_args or nested_cli):
                func_name = getattr(make_tool_cfg, "__name__", str(make_tool_cfg))
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
            tool_cfg = make_tool_cfg(**func_kwargs)

            # Apply general wrapper for non-Tool return types if we have an expected type
            if expected_tool_type or not isinstance(tool_cfg, Tool):
                tool_cfg = wrap_config_to_tool(tool_cfg, expected_tool_type, cli_args)

            # Check if this was a mettagrid_recipe that needs wrapping
            if hasattr(make_tool_cfg, "_requested_tool"):
                # This was a fallback from mettagrid_recipe, wrap it in the appropriate tool
                from mettagrid.config.mettagrid_config import MettaGridConfig

                if isinstance(tool_cfg, MettaGridConfig):
                    requested_tool = make_tool_cfg._requested_tool

                    # Import and wrap based on the requested tool
                    if requested_tool == "train":
                        from metta.cogworks.curriculum import single_task_curriculum
                        from metta.rl.trainer_config import TrainerConfig
                        from metta.tools.train import TrainTool

                        # Create a curriculum from the MettaGridConfig
                        curriculum_cfg = single_task_curriculum(tool_cfg)
                        trainer_cfg = TrainerConfig(curriculum=curriculum_cfg)
                        tool_cfg = TrainTool(config=trainer_cfg)

                    elif requested_tool in ["play", "replay"]:
                        from metta.sim.simulation_config import SimulationConfig

                        sim_cfg = SimulationConfig(env=tool_cfg, name=tool_cfg.label or "mettagrid")

                        if requested_tool == "play":
                            from metta.tools.play import PlayTool

                            tool_cfg = PlayTool(config=sim_cfg)
                        else:  # replay
                            from metta.tools.replay import ReplayTool

                            tool_cfg = ReplayTool(config=sim_cfg)

                    elif requested_tool == "sim":
                        from metta.sim.simulation_config import SimulationConfig
                        from metta.tools.sim import SimTool

                        # Create evaluation simulations from the MettaGridConfig
                        sim_cfg = SimulationConfig(env=tool_cfg, name=tool_cfg.label or "mettagrid")
                        tool_cfg = SimTool(config=[sim_cfg])

                        # SimTool still needs policy_uri from CLI args
                        if "policy_uri" not in cli_args:
                            output_error(f"{red('Error:')} {requested_tool} requires policy_uri parameter")
                            return 1

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
        output_error(f"{red('Error:')} {make_tool_cfg_path} must return a Tool instance, got {type(tool_cfg)}")
        return 1

    # ----------------------------------------------------------------------------------
    # Overrides & Unknowns (post-construction)
    # ----------------------------------------------------------------------------------
    tool_fields = get_tool_fields(type(tool_cfg))
    override_args, unknown_args = classify_remaining_args(remaining_args, tool_fields)

    if unknown_args:
        output_info(f"\n{red('Error: Unknown arguments:')} {', '.join(unknown_args)}")
        # Only show function params list if the entrypoint is a function
        if not (inspect.isclass(make_tool_cfg) and issubclass(make_tool_cfg, Tool)):
            output_info(f"\n{yellow('Available function parameters:')}")
            for param in get_function_params(make_tool_cfg):
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
            output_info(f"Module: {make_tool_cfg_path}")
        return 0

    # ----------------------------------------------------------------------------------
    # Seed & Run
    # ----------------------------------------------------------------------------------
    if hasattr(tool_cfg, "system"):
        seed_everything(tool_cfg.system)

    output_info(f"\n{bold(green('Running tool...'))}\n")

    try:
        if known_args.dry_run:
            output_info(bold(green("Dry run: exiting")))
            result = 0
        else:
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
