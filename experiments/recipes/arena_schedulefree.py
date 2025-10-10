"""Arena recipe with ScheduleFree AdamW optimizer for comparison testing."""

from metta.rl.trainer_config import OptimizerConfig, TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.tools.train import TrainTool

# Import everything from the base arena recipe
from experiments.recipes.arena import (
    make_curriculum,
    simulations,
)


def train(
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    """Train with ScheduleFree AdamW optimizer.

    This uses the same configuration as the base arena recipe but with
    ScheduleFree AdamW optimizer instead of regular Adam.
    """
    curriculum = make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    # Configure ScheduleFree AdamW optimizer
    optimizer_config = OptimizerConfig(
        type="adamw_schedulefree",
        learning_rate=0.001153637,  # Same as default
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0.01,  # Small weight decay for AdamW
        warmup_steps=1000,  # Warmup steps for ScheduleFree
    )

    trainer_config = TrainerConfig(
        optimizer=optimizer_config,
        total_timesteps=50_000_000_000,
    )

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        trainer=trainer_config,
        evaluator=EvaluatorConfig(simulations=simulations()),
    )


def train_shaped(rewards: bool = True, converters: bool = True) -> TrainTool:
    """Train with ScheduleFree AdamW optimizer on shaped rewards task.

    This provides easier training with reward shaping and converters enabled.
    """
    # Import and configure the shaped environment from base recipe
    from experiments.recipes.arena import train_shaped as base_train_shaped

    # Get the base shaped training tool
    base_tool = base_train_shaped(rewards=rewards, converters=converters)

    # Configure ScheduleFree AdamW optimizer (using native implementation)
    optimizer_config = OptimizerConfig(
        type="adamw_schedulefree",
        learning_rate=0.001153637,  # Same as default
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0.01,  # Small weight decay for AdamW
        warmup_steps=1000,  # Warmup steps for ScheduleFree
    )

    trainer_config = TrainerConfig(
        optimizer=optimizer_config,
        total_timesteps=50_000_000_000,
    )

    # Return a new TrainTool with the shaped environment but ScheduleFree optimizer
    return TrainTool(
        training_env=base_tool.training_env,
        trainer=trainer_config,
        evaluator=base_tool.evaluator,
    )
