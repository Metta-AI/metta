"""Arena recipe with GRPO (Group Relative Policy Optimization) for comparison testing."""

from metta.agent.policies.vit_grpo import ViTGRPOConfig
from metta.rl.loss.grpo import GRPOConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.trainer_config import OptimizerConfig
from metta.tools.train import TrainTool
from recipes.experiment.arena import train_shaped as base_train_shaped
from recipes.prod.arena_basic_easy_shaped import BASELINE as ARENA_BASELINE


def train(
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    """Train with GRPO loss (critic-free, group-based advantages).

    GRPO eliminates the value network and computes advantages by comparing
    each trajectory's return against the mean return of a group of sampled
    trajectories. This can be more sample efficient and stable than PPO
    in certain environments.
    """
    tool = ARENA_BASELINE.model_copy(deep=True)

    # Configure GRPO loss
    grpo_config = GRPOConfig(
        clip_coef=0.2,
        ent_coef=0.01,
        gamma=0.99,
        group_size=4,
        norm_adv=True,
        target_kl=None,
    )

    # Override losses to use GRPO instead of PPO
    tool.trainer.losses = LossesConfig(grpo=grpo_config)
    tool.policy_architecture = ViTGRPOConfig()

    return tool


def train_shaped(rewards: bool = True, converters: bool = True) -> TrainTool:
    """Train with GRPO loss on shaped rewards task.

    This provides easier training with reward shaping and converters enabled,
    using the critic-free GRPO algorithm.
    """
    baseline = ARENA_BASELINE.model_copy(deep=True)
    base_tool = base_train_shaped(rewards=rewards)
    baseline.training_env.curriculum = base_tool.training_env.curriculum

    grpo_config = GRPOConfig(
        clip_coef=0.2,
        ent_coef=0.01,
        gamma=0.99,
        group_size=4,
        norm_adv=True,
        target_kl=None,
    )

    baseline.trainer.losses = LossesConfig(grpo=grpo_config)
    baseline.trainer.optimizer = OptimizerConfig(
        type="adamw_schedulefree",
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0.01,
        warmup_steps=2000,
    )
    baseline.policy_architecture = ViTGRPOConfig()

    return baseline


def basic_easy_shaped() -> TrainTool:
    """Train with GRPO loss on basic easy shaped rewards task.

    This provides easier training with reward shaping and converters enabled,
    using the critic-free GRPO algorithm.
    """
    tool = ARENA_BASELINE.model_copy(deep=True)

    # Configure GRPO loss
    grpo_config = GRPOConfig(
        clip_coef=0.2,
        ent_coef=0.01,
        gamma=0.99,
        group_size=4,
        norm_adv=True,
        target_kl=None,
    )

    # Override losses to use GRPO instead of PPO
    tool.trainer.losses = LossesConfig(grpo=grpo_config)
    tool.policy_architecture = ViTGRPOConfig()

    return tool
