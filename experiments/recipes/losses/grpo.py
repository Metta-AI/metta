"""Arena recipe with GRPO (Group Relative Policy Optimization) for comparison testing."""

from metta.agent.policies.vit_grpo import ViTGRPOConfig
from metta.rl.loss import LossConfig
from metta.rl.loss.grpo import GRPOConfig
from metta.rl.trainer_config import OptimizerConfig, TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.tools.train import TrainTool
from experiments.recipes.arena import train_shaped as base_train_shaped

# Import everything from the base arena recipe
from experiments.recipes.arena import (
    make_curriculum,
    simulations,
)


def train(
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    """Train with GRPO loss (critic-free, group-based advantages).

    GRPO eliminates the value network and computes advantages by comparing
    each trajectory's return against the mean return of a group of sampled
    trajectories. This can be more sample efficient and stable than PPO
    in certain environments.
    """
    curriculum = make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    # Configure GRPO loss
    grpo_config = GRPOConfig(
        clip_coef=0.2,
        ent_coef=0.01,
        gamma=0.99,
        group_size=4,
        norm_adv=True,
        target_kl=None,
    )

    loss_config = LossConfig(
        loss_configs={"grpo": grpo_config},
    )

    # Configure optimizer
    optimizer_config = OptimizerConfig(
        type="adamw_schedulefree",
        learning_rate=0.001153637,
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0.01,
        warmup_steps=1000,
    )

    trainer_config = TrainerConfig(
        losses=loss_config,
        optimizer=optimizer_config,
        total_timesteps=50_000_000_000,
    )

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        trainer=trainer_config,
        evaluator=EvaluatorConfig(simulations=simulations()),
        policy_architecture=ViTGRPOConfig(),
    )


def train_shaped(rewards: bool = True, converters: bool = True) -> TrainTool:
    """Train with GRPO loss on shaped rewards task.

    This provides easier training with reward shaping and converters enabled,
    using the critic-free GRPO algorithm.
    """

    # Get the base shaped training tool
    base_tool = base_train_shaped(rewards=rewards, converters=converters)

    # Configure GRPO loss
    grpo_config = GRPOConfig(
        clip_coef=0.2,
        ent_coef=0.01,
        gamma=0.99,
        group_size=4,
        norm_adv=True,
        target_kl=None,
    )

    loss_config = LossConfig(
        loss_configs={"grpo": grpo_config},
    )

    # Configure optimizer
    optimizer_config = OptimizerConfig(
        type="adamw_schedulefree",
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0.01,
        warmup_steps=2000,
    )

    trainer_config = TrainerConfig(
        losses=loss_config,
        optimizer=optimizer_config,
        total_timesteps=50_000_000_000,
    )

    return TrainTool(
        training_env=base_tool.training_env,
        trainer=trainer_config,
        evaluator=base_tool.evaluator,
        policy_architecture=ViTGRPOConfig(),
    )
