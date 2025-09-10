"""Simple train-and-eval adaptive experiment for PoC."""

from metta.tools.adaptive import AdaptiveTool, SchedulerType


def train_and_eval(
    run: str | None = None,  # Accept run parameter from dispatcher (unused)
    recipe_module: str = "experiments.recipes.arena",
    train_entrypoint: str = "train",
    eval_entrypoint: str = "evaluate",
    max_trials: int = 3,
    gpus_per_job: int = 1,
    experiment_id: str = "train_eval_poc",
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

    Returns:
        Configured AdaptiveTool for the experiment

    Example:
        uv run ./tools/run.py experiments.recipes.adaptive.train_and_eval \\
            --args train_entrypoint=train eval_entrypoint=evaluate
    """

    # Return configured tool
    return AdaptiveTool(
        scheduler_type=SchedulerType.TRAIN_AND_EVAL,
        scheduler_config={
            "recipe_module": recipe_module,
            "train_entrypoint": train_entrypoint,
            "eval_entrypoint": eval_entrypoint,
            "max_trials": max_trials,
            "gpus_per_job": gpus_per_job,
            "experiment_id": experiment_id,
            "train_overrides": {
                "total_timesteps": 15000
            }
        },
        experiment_id=experiment_id,
    )
