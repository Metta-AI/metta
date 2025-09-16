"""Simple train-and-eval adaptive experiment for PoC."""

from typing import Any

from metta.adaptive.adaptive_config import AdaptiveConfig
from metta.tools.adaptive import AdaptiveTool, SchedulerType, DispatcherType
from metta.adaptive.schedulers.train_and_eval import TrainAndEvalConfig


def train_and_eval(
    run: str | None = None,  # Accept run parameter from dispatcher (unused)
    recipe_module: str = "experiments.recipes.arena",
    train_entrypoint: str = "train",
    eval_entrypoint: str = "evaluate",
    max_trials: int = 3,
    gpus: int = 4,
    experiment_id: str = "train_eval_poc",
    dispatcher_type: str = "skypilot",  # "local" or "skypilot"
    resume: bool = False,  # Resume from existing experiment
    total_timesteps: int = 2000000000,  # 2B default
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
        gpus: Number of GPUs per training job
        experiment_id: Unique identifier for this experiment
        train_overrides: Additional overrides to apply to all training jobs
        dispatcher_type: Where to run jobs - "local" or "skypilot"
        resume: Resume from existing experiment (skip initial fetch timeout)

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
        raise ValueError(
            f"Invalid dispatcher_type: {dispatcher_type}. Must be 'local' or 'skypilot'"
        )

    # Build typed scheduler config
    scheduler_config = TrainAndEvalConfig(
        recipe_module=recipe_module,
        train_entrypoint=train_entrypoint,
        eval_entrypoint=eval_entrypoint,
        max_trials=max_trials,
        gpus=gpus,
        experiment_id=experiment_id,
        train_overrides={
            "trainer.total_timesteps": total_timesteps,
        },
    )

    adaptive_config = AdaptiveConfig(max_parallel=4, resume=resume)

    return AdaptiveTool(
        scheduler_type=SchedulerType.TRAIN_AND_EVAL,
        scheduler_config=scheduler_config,
        config=adaptive_config,
        dispatcher_type=dispatcher_enum,
        experiment_id=experiment_id,
    )
