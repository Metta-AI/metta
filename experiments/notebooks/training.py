"""Notebook-friendly wrappers for training functionality."""

from typing import Dict, Any, List, Optional

from experiments.launch import launch_training_run as core_launch


def launch_training(
    run_name: str,
    curriculum: str,
    gpus: int = 1,
    nodes: int = 1,
    no_spot: bool = False,
    skip_git_check: bool = False,
    additional_args: Optional[List[str]] = None,
    wandb_tags: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Launch a training run from a notebook.

    This is a convenience wrapper around experiments.launch.launch_training_run
    that provides a notebook-friendly interface.

    Args:
        run_name: Name for the training run
        curriculum: Path to curriculum config (e.g. "env/mettagrid/arena/basic")
        gpus: Number of GPUs per node
        nodes: Number of nodes
        no_spot: Whether to disable spot instances
        skip_git_check: Whether to skip git state validation (useful for local changes)
        additional_args: Additional command line arguments
        wandb_tags: Tags for wandb
        **kwargs: Additional keyword arguments passed as trainer.key=value

    Returns:
        Dictionary containing job_id, run_name, success, command, output

    Example:
        >>> result = launch_training(
        ...     run_name="my_experiment",
        ...     curriculum="env/mettagrid/arena/basic",
        ...     gpus=4,
        ...     skip_git_check=True,  # Allow uncommitted changes
        ...     wandb_tags=["arena", "test"],
        ...     learning_rate=0.001
        ... )
    """
    # Convert kwargs to additional args
    if kwargs:
        if additional_args is None:
            additional_args = []
        for key, value in kwargs.items():
            additional_args.append(f"trainer.{key}={value}")

    return core_launch(
        run_name=run_name,
        curriculum=curriculum,
        gpus=gpus,
        nodes=nodes,
        no_spot=no_spot,
        skip_git_check=skip_git_check,
        additional_args=additional_args,
        wandb_tags=wandb_tags,
    )


def launch_multiple_training_runs(
    base_run_name: str, curriculum: str, num_runs: int = 1, vary_seeds: bool = True, **kwargs
) -> List[Dict[str, Any]]:
    """Launch multiple training runs with optional seed variation.

    Args:
        base_run_name: Base name for runs (will append .1, .2, etc)
        curriculum: Path to curriculum config
        num_runs: Number of runs to launch
        vary_seeds: If True, each run gets a different seed
        **kwargs: Arguments passed to launch_training

    Returns:
        List of launch results
    """
    results = []

    for i in range(num_runs):
        run_name = f"{base_run_name}.{i + 1}" if num_runs > 1 else base_run_name

        if vary_seeds and i > 0:
            if "additional_args" not in kwargs:
                kwargs["additional_args"] = []
            kwargs["additional_args"].append(f"trainer.seed={42 + i}")

        result = launch_training(run_name=run_name, curriculum=curriculum, **kwargs)
        results.append(result)

        if not result["success"]:
            print(f"Warning: Run {i + 1} failed to launch, stopping remaining launches")
            break

    return results
