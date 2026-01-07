"""Arena recipe with regular Adam optimizer for comparison testing."""

from metta.agent.policies.vit import ViTDefaultConfig
from metta.rl.policy_assets import OptimizerConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.tools.train import TrainTool
from recipes.experiment.arena import (
    make_curriculum,
    simulations,
)

DEFAULT_LR = OptimizerConfig.model_fields["learning_rate"].default


def train(
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    """Train with regular Adam optimizer.

    This uses the same configuration as ScheduleFree recipe but with
    regular Adam optimizer for comparison.
    """
    curriculum = make_curriculum(enable_detailed_slice_logging=enable_detailed_slice_logging)

    # Configure regular Adam optimizer (matching default)
    optimizer_config = OptimizerConfig(
        type="adam",
        learning_rate=DEFAULT_LR,
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0,  # No weight decay for Adam
    )

    trainer_config = TrainerConfig(
        total_timesteps=50_000_000_000,
    )

    tt = TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        trainer=trainer_config,
        evaluator=EvaluatorConfig(simulations=simulations()),
        policy_architecture=ViTDefaultConfig(),
    )
    tt.policy_assets["primary"].optimizer = optimizer_config
    return tt


def train_shaped(rewards: bool = True) -> TrainTool:
    """Train with regular Adam optimizer on shaped rewards task.

    This provides easier training with reward shaping and converters enabled.
    """
    # Import and configure the shaped environment from base recipe
    from recipes.experiment.arena import train_shaped as base_train_shaped

    # Get the base shaped training tool
    base_tool = base_train_shaped(rewards=rewards)

    # Configure regular Adam optimizer
    optimizer_config = OptimizerConfig(
        type="adam",
        learning_rate=DEFAULT_LR,
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0,
    )

    trainer_config = TrainerConfig(
        total_timesteps=50_000_000_000,
    )

    # Return a new TrainTool with the shaped environment but regular Adam optimizer
    tt = TrainTool(
        training_env=base_tool.training_env,
        trainer=trainer_config,
        evaluator=base_tool.evaluator,
        policy_architecture=base_tool.policy_architecture,
    )
    tt.policy_assets["primary"].optimizer = optimizer_config
    return tt
