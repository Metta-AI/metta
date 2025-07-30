#!/usr/bin/env python3
"""Generic experiment runner that handles CLI parsing and name extraction.

This module provides a reusable runner function that:
1. Extracts an optional name parameter as the first positional argument
2. Handles all the pydantic-settings CLI parsing
3. Creates and runs the experiment

Usage:
    from experiments.runner import runner
    from experiments.experiment import SingleJobExperiment, SingleJobExperimentConfig

    class MyExperimentConfig(SingleJobExperimentConfig):
        # Your config customizations
        pass

    if __name__ == "__main__":
        exit(runner(SingleJobExperiment, MyExperimentConfig))
"""

import sys
import os
from typing import Type, TypeVar
from pydantic_settings import BaseSettings, SettingsConfigDict

from experiments.experiment import Experiment, ExperimentConfig
from experiments.cli_formatter import format_help_with_defaults

T = TypeVar("T", bound=ExperimentConfig)
E = TypeVar("E", bound=Experiment)


def runner(
    experiment_class: Type[E],
    config_class: Type[T],
) -> int:
    """Run an experiment with standard CLI parsing and name handling.

    Args:
        experiment_class: The experiment class to instantiate and run
        config_class: The config class to use for parsing arguments
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Extract name from first positional argument if present
        name = None
        args_to_parse = sys.argv[1:]

        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            name = sys.argv[1]
            args_to_parse = sys.argv[2:]  # Remove the name from args to parse

        # Default name to config default or script name
        if name is None:
            # Try to get default from config class
            if hasattr(config_class, "name") and config_class.name:
                name = config_class.name
            else:
                # Fall back to script name
                script_name = os.path.basename(sys.argv[0])
                if script_name.endswith(".py"):
                    script_name = script_name[:-3]
                name = script_name

        # Get program name for help
        prog_name = os.path.basename(sys.argv[0])
        if prog_name.endswith(".py"):
            prog_name = prog_name[:-3]

        # Check if user wants custom help
        if (
            "--help" in args_to_parse
            or "-h" in args_to_parse
            or "--help-compact" in args_to_parse
        ):
            # Parse the arguments to get user overrides
            user_overrides = {}
            for i, arg in enumerate(args_to_parse):
                if (
                    arg.startswith("--")
                    and "=" in arg
                    and arg not in ["--help", "-h", "--help-compact"]
                ):
                    key, value = arg[2:].split("=", 1)
                    key = key.replace("-", "_")
                    # Try to parse the value
                    try:
                        # Try as boolean
                        if value.lower() in ["true", "false"]:
                            value = value.lower() == "true"
                        # Try as int
                        elif value.isdigit() or (
                            value.startswith("-") and value[1:].isdigit()
                        ):
                            value = int(value)
                        # Try as float
                        elif "." in value:
                            try:
                                value = float(value)
                            except ValueError:
                                pass  # Keep as string
                    except:
                        pass  # Keep as string
                    user_overrides[key] = value

            # Patch in the computed default for output_dir
            patched_defaults = {}

            # Check if the config class has an explicit default for output_dir
            has_explicit_default = False
            if (
                hasattr(config_class, "model_fields")
                and "output_dir" in config_class.model_fields
            ):
                field_info = config_class.model_fields["output_dir"]
                if field_info.default is not None and field_info.default != "":
                    # Use the explicit default from the config class
                    patched_defaults["output_dir"] = field_info.default
                    has_explicit_default = True

            if not has_explicit_default:
                try:
                    from metta.common.util.fs import get_repo_root

                    repo_root = get_repo_root()
                    patched_defaults["output_dir"] = str(
                        repo_root / "experiments" / "scratch"
                    )
                except:
                    patched_defaults["output_dir"] = "experiments/scratch"

            # Default is expanded, use --help-compact for collapsed view
            collapse = "--help-compact" in args_to_parse
            help_text = format_help_with_defaults(
                config_class,
                prog_name,
                has_positional_name=True,
                collapse=collapse,
                user_overrides=user_overrides,
                patched_defaults=patched_defaults,
            )
            print(help_text)
            return 0

        # Create a CLI wrapper that uses pydantic-settings for parsing
        class CLIWrapper(BaseSettings, config_class):
            model_config = SettingsConfigDict(
                cli_parse_args=args_to_parse,
                cli_prog_name=prog_name,
                cli_use_class_docs_for_groups=True,
                env_prefix="METTA_EXP_",
                env_nested_delimiter="__",
            )

        # Parse CLI args
        cli_config = CLIWrapper()

        # Create the actual config from parsed values
        config_dict = cli_config.model_dump()
        config_dict["name"] = name  # Override with our determined name

        # Handle string "None" -> actual None conversion
        if "output_dir" in config_dict and config_dict["output_dir"] == "None":
            config_dict["output_dir"] = None

        # Set default output_dir if not specified
        if "output_dir" not in config_dict or config_dict["output_dir"] is None:
            # Check if the config class has a non-None default for output_dir
            has_explicit_default = False
            if (
                hasattr(config_class, "model_fields")
                and "output_dir" in config_class.model_fields
            ):
                field_info = config_class.model_fields["output_dir"]
                if field_info.default is not None and field_info.default != "":
                    # Use the explicit default from the config class
                    config_dict["output_dir"] = field_info.default
                    has_explicit_default = True

            if not has_explicit_default:
                # Try to find repo root for a more stable path
                try:
                    from metta.common.util.fs import get_repo_root

                    repo_root = get_repo_root()
                    config_dict["output_dir"] = str(
                        repo_root / "experiments" / "scratch"
                    )
                except:
                    # Fallback to relative path if we can't find repo root
                    config_dict["output_dir"] = "experiments/scratch"

        config = config_class(**config_dict)

        # Create and run experiment
        experiment = experiment_class(config)
        notebook_path = experiment.run()

        # The notebook opening logic is handled by metta CLI
        # Just return success
        return 0

    except SystemExit:
        # This happens when pydantic-settings shows help or validation errors
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main():
    """Main entry point that dispatches to different experiment recipes.

    Usage: runner.py <recipe> [name] [options...]

    Examples:
        runner.py arena my_test_v1 --gpus 2
        runner.py arena --gpus 4  # Uses default name
    """
    if len(sys.argv) < 2:
        print("Usage: runner.py <recipe> [name] [options...]", file=sys.stderr)
        print("Available recipes: arena", file=sys.stderr)
        return 1

    # Extract recipe name
    recipe = sys.argv[1]

    # Remove recipe from argv so the rest of runner() works normally
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    try:
        if recipe == "arena":
            from experiments.recipes.arena_experiment import ArenaExperimentConfig
            from experiments.experiment import SingleJobExperiment

            return runner(
                SingleJobExperiment,
                ArenaExperimentConfig,
            )
        # Add new recipes here:
        # elif recipe == "your_recipe":
        #     from experiments.recipes.your_recipe import YourExperimentConfig
        #     return runner(
        #         SingleJobExperiment,  # or your custom experiment class
        #         YourExperimentConfig,
        #     )
        else:
            print(f"Unknown recipe: {recipe}", file=sys.stderr)
            print("Available recipes: arena", file=sys.stderr)
            return 1

    finally:
        # Restore original argv
        sys.argv = original_argv


if __name__ == "__main__":
    exit(main())
