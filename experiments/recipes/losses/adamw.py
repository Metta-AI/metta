"""Arena recipe with regular Adam optimizer for comparison testing."""

import metta.rl.trainer_config
import metta.rl.training
import metta.tools.train

# Import everything from the base arena recipe
import experiments.recipes.arena


def train(
    enable_detailed_slice_logging: bool = False,
) -> metta.tools.train.TrainTool:
    """Train with regular Adam optimizer.

    This uses the same configuration as ScheduleFree recipe but with
    regular Adam optimizer for comparison.
    """
    curriculum = experiments.recipes.arena.make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    # Configure regular Adam optimizer (matching default)
    optimizer_config = metta.rl.trainer_config.OptimizerConfig(
        type="adam",
        learning_rate=0.001153637,  # Same as default
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0,  # No weight decay for Adam
    )

    trainer_config = metta.rl.trainer_config.TrainerConfig(
        optimizer=optimizer_config,
        total_timesteps=50_000_000_000,
    )

    return metta.tools.train.TrainTool(
        training_env=metta.rl.training.TrainingEnvironmentConfig(curriculum=curriculum),
        trainer=trainer_config,
        evaluator=metta.rl.training.EvaluatorConfig(
            simulations=experiments.recipes.arena.simulations()
        ),
    )


def train_shaped(rewards: bool = True) -> metta.tools.train.TrainTool:
    """Train with regular Adam optimizer on shaped rewards task.

    This provides easier training with reward shaping and converters enabled.
    """

    # Get the base shaped training tool
    base_tool = experiments.recipes.arena.train_shaped(rewards=rewards)

    # Configure regular Adam optimizer
    optimizer_config = metta.rl.trainer_config.OptimizerConfig(
        type="adam",
        learning_rate=0.001153637,
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0,
    )

    trainer_config = metta.rl.trainer_config.TrainerConfig(
        optimizer=optimizer_config,
        total_timesteps=50_000_000_000,
    )

    # Return a new TrainTool with the shaped environment but regular Adam optimizer
    return metta.tools.train.TrainTool(
        training_env=base_tool.training_env,
        trainer=trainer_config,
        evaluator=base_tool.evaluator,
    )
