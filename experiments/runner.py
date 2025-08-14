#!/usr/bin/env python3
"""Typer-based experiment runner with proper CLI structure."""

import sys
from typing import Type, TypeVar, Optional, List
from pathlib import Path

import typer
from typing_extensions import Annotated
from rich.console import Console
from pydantic import ValidationError

from experiments.experiment import Experiment, ExperimentConfig

T = TypeVar("T", bound=ExperimentConfig)
E = TypeVar("E", bound=Experiment)

console = Console()


def runner(
    experiment_class: Type[E],
    config_class: Type[T],
) -> int:
    """Run an experiment using typer.

    Args:
        experiment_class: The experiment class to instantiate
        config_class: The config class for parameters

    Returns:
        Exit code (0 for success)
    """
    # Get program name
    prog_name = Path(sys.argv[0]).stem

    # Create the app without subcommands
    app = typer.Typer(
        add_completion=False,
        pretty_exceptions_show_locals=False,
    )

    @app.command()
    def main(
        # Positional argument
        name: Annotated[Optional[str], typer.Argument(help="Experiment name")] = None,
        # Core parameters
        gpus: Annotated[int, typer.Option(help="Number of GPUs per node")] = 1,
        nodes: Annotated[int, typer.Option(help="Number of nodes")] = 1,
        spot: Annotated[bool, typer.Option(help="Use spot instances")] = True,
        # Launch control
        launch: Annotated[bool, typer.Option(help="Launch the job to Skypilot")] = True,
        git_check: Annotated[bool, typer.Option(help="Check git status")] = True,
        # Load previous jobs
        previous_job_ids: Annotated[
            Optional[List[str]], typer.Option(help="Previous job IDs to load")
        ] = None,
    ):
        """Run an experiment with the specified configuration."""

        # Typer has built-in help, we don't need custom help formatting

        # Determine name
        if name is None:
            if (
                hasattr(config_class, "model_fields")
                and "name" in config_class.model_fields
            ):
                field_info = config_class.model_fields["name"]
                name = field_info.default if field_info.default else prog_name
            else:
                name = prog_name

        # Build config - check if this is a SingleJobExperimentConfig
        from experiments.experiment import SingleJobExperimentConfig

        config_dict = {
            "name": name,
            "launch": launch,
        }

        if previous_job_ids:
            config_dict["previous_job_ids"] = previous_job_ids

        # If it's a SingleJobExperimentConfig, we need to set the nested configs
        if issubclass(config_class, SingleJobExperimentConfig):
            from experiments.skypilot_job_config import SkypilotJobConfig

            # Create skypilot config with CLI overrides
            config_dict["skypilot"] = SkypilotJobConfig(
                gpus=gpus,
                nodes=nodes,
                spot=spot,
                git_check=git_check,
            )

            # The config class should provide its own training config as a default
        else:
            # For base ExperimentConfig, just pass through the infrastructure values
            config_dict.update(
                {
                    "gpus": gpus,
                    "nodes": nodes,
                    "spot": spot,
                    "git_check": git_check,
                }
            )

        # Create and run experiment
        try:
            config = config_class(**config_dict)
            experiment = experiment_class(config)

            console.print(f"[bold blue]Running experiment: {name}[/bold blue]")
            notebook_path = experiment.run()

            if notebook_path:
                console.print(f"[green]✓[/green] Notebook: {notebook_path}")

            return 0

        except ValidationError as e:
            console.print("[red]Configuration error:[/red]", e)
            return 1
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            return 1

    # Run the app
    try:
        app()
        return 0
    except SystemExit as e:
        return e.code if e.code else 0
