"""Arena recipe with regular Adam optimizer for comparison testing."""

from metta.rl.trainer_config import OptimizerConfig
from metta.tools.train import TrainTool
from recipes.prod.arena_basic_easy_shaped import BASELINE as ARENA_BASELINE

DEFAULT_LR = OptimizerConfig.model_fields["learning_rate"].default


def train(
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    """Train with regular Adam optimizer.

    This uses the same configuration as ScheduleFree recipe but with
    regular Adam optimizer for comparison.
    """
    tool = ARENA_BASELINE.model_copy(deep=True)

    # Configure regular Adam optimizer
    tool.trainer.optimizer = OptimizerConfig(
        type="adam",
        learning_rate=DEFAULT_LR,
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0,  # No weight decay for Adam
    )

    return tool


def train_shaped(rewards: bool = True) -> TrainTool:
    """Train with regular Adam optimizer on shaped rewards task.

    This provides easier training with reward shaping and converters enabled.
    """
    from recipes.experiment.arena import train_shaped as base_train_shaped

    baseline = ARENA_BASELINE.model_copy(deep=True)
    base_tool = base_train_shaped(rewards=rewards)
    baseline.training_env.curriculum = base_tool.training_env.curriculum

    baseline.trainer.optimizer = OptimizerConfig(
        type="adam",
        learning_rate=DEFAULT_LR,
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0,
    )

    return baseline
