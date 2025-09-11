"""Simple train-and-eval adaptive experiment for PoC."""

from typing import Any

from metta.tools.adaptive import AdaptiveTool, SchedulerType, DispatcherType


def train_and_eval(
    run: str | None = None,  # Accept run parameter from dispatcher (unused)
    recipe_module: str = "experiments.recipes.arena",
    train_entrypoint: str = "train",
    eval_entrypoint: str = "evaluate",
    max_trials: int = 3,
    gpus_per_job: int = 1,
    experiment_id: str = "train_eval_poc",
    train_overrides: dict[str, Any] | None = None,  # Trainer overrides
    dispatcher_type: str = "skypilot",  # "local" or "skypilot"
) -> AdaptiveTool:
    """Create simple train-and-eval adaptive experiment for PoC.

    This experiment runs a sequence of training jobs followed by evaluation jobs.
    No hyperparameter optimization - just a simple proof of concept for the
    adaptive experiment infrastructure.

    Args:
        recipe_module: Module containing the training/eval functions
        train_entrypoint: Name of the training function
        eval_entrypoint: Name of the evaluation function
        max_trials: Maximum number of training runs
        gpus_per_job: Number of GPUs per training job
        experiment_id: Unique identifier for this experiment
        train_overrides: Additional overrides to apply to all training jobs
        dispatcher_type: Where to run jobs - "local" or "skypilot"

    Returns:
        Configured AdaptiveTool for the experiment

    Example:
        uv run ./tools/run.py experiments.recipes.adaptive.train_and_eval \\
            --args train_entrypoint=train eval_entrypoint=evaluate
    """

    # Return configured tool
    # Parse dispatcher type
    if dispatcher_type.lower() == "local":
        dispatcher_enum = DispatcherType.LOCAL
    elif dispatcher_type.lower() == "skypilot":
        dispatcher_enum = DispatcherType.SKYPILOT
    else:
        raise ValueError(f"Invalid dispatcher_type: {dispatcher_type}. Must be 'local' or 'skypilot'")

    # Build scheduler config
    scheduler_config = {
        "recipe_module": recipe_module,
        "train_entrypoint": train_entrypoint,
        "eval_entrypoint": eval_entrypoint,
        "max_trials": max_trials,
        "gpus_per_job": gpus_per_job,
        "experiment_id": experiment_id,
    }

    if train_overrides:
        scheduler_config["train_overrides"] = train_overrides

    return AdaptiveTool(
        scheduler_type=SchedulerType.TRAIN_AND_EVAL,
        scheduler_config=scheduler_config,
        dispatcher_type=dispatcher_enum,
        experiment_id=experiment_id,
    )
