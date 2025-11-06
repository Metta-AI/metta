"""Arena recipe with GRPO (Group Relative Policy Optimization) for comparison testing."""

import metta.agent.policies.vit_grpo
import metta.rl.loss
import metta.rl.loss.grpo
import metta.rl.trainer_config
import metta.rl.training
import metta.tools.train

# Import everything from the base arena recipe
import experiments.recipes.arena
import experiments.recipes.arena_basic_easy_shaped


def train(
    enable_detailed_slice_logging: bool = False,
) -> metta.tools.train.TrainTool:
    """Train with GRPO loss (critic-free, group-based advantages).

    GRPO eliminates the value network and computes advantages by comparing
    each trajectory's return against the mean return of a group of sampled
    trajectories. This can be more sample efficient and stable than PPO
    in certain environments.
    """
    curriculum = experiments.recipes.arena.make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    # Configure GRPO loss
    grpo_config = metta.rl.loss.grpo.GRPOConfig(
        clip_coef=0.2,
        ent_coef=0.01,
        gamma=0.99,
        group_size=4,
        norm_adv=True,
        target_kl=None,
    )

    loss_config = metta.rl.loss.LossConfig(
        loss_configs={"grpo": grpo_config},
    )

    # Configure optimizer
    optimizer_config = metta.rl.trainer_config.OptimizerConfig(
        type="adamw_schedulefree",
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0.01,
        warmup_steps=2000,
    )

    trainer_config = metta.rl.trainer_config.TrainerConfig(
        losses=loss_config,
        optimizer=optimizer_config,
        total_timesteps=50_000_000_000,
    )

    return metta.tools.train.TrainTool(
        training_env=metta.rl.training.TrainingEnvironmentConfig(curriculum=curriculum),
        trainer=trainer_config,
        evaluator=metta.rl.training.EvaluatorConfig(
            simulations=experiments.recipes.arena.simulations()
        ),
        policy_architecture=metta.agent.policies.vit_grpo.ViTGRPOConfig(),
    )


def train_shaped(
    rewards: bool = True, converters: bool = True
) -> metta.tools.train.TrainTool:
    """Train with GRPO loss on shaped rewards task.

    This provides easier training with reward shaping and converters enabled,
    using the critic-free GRPO algorithm.
    """

    # Get the base shaped training tool
    base_tool = experiments.recipes.arena.train_shaped(rewards=rewards)

    # Configure GRPO loss
    grpo_config = metta.rl.loss.grpo.GRPOConfig(
        clip_coef=0.2,
        ent_coef=0.01,
        gamma=0.99,
        group_size=4,
        norm_adv=True,
        target_kl=None,
    )

    loss_config = metta.rl.loss.LossConfig(
        loss_configs={"grpo": grpo_config},
    )

    # Configure optimizer
    optimizer_config = metta.rl.trainer_config.OptimizerConfig(
        type="adamw_schedulefree",
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0.01,
        warmup_steps=2000,
    )

    trainer_config = metta.rl.trainer_config.TrainerConfig(
        losses=loss_config,
        optimizer=optimizer_config,
        total_timesteps=50_000_000_000,
    )

    return metta.tools.train.TrainTool(
        training_env=base_tool.training_env,
        trainer=trainer_config,
        evaluator=base_tool.evaluator,
        policy_architecture=metta.agent.policies.vit_grpo.ViTGRPOConfig(),
    )


def basic_easy_shaped() -> metta.tools.train.TrainTool:
    """Train with GRPO loss on basic easy shaped rewards task.

    This provides easier training with reward shaping and converters enabled,
    using the critic-free GRPO algorithm.
    """

    # Get the base shaped training tool
    base_tool = experiments.recipes.arena_basic_easy_shaped.train()

    # Configure GRPO loss
    grpo_config = metta.rl.loss.grpo.GRPOConfig(
        clip_coef=0.2,
        ent_coef=0.01,
        gamma=0.99,
        group_size=4,
        norm_adv=True,
        target_kl=None,
    )

    loss_config = metta.rl.loss.LossConfig(
        loss_configs={"grpo": grpo_config},
    )

    # Configure optimizer
    optimizer_config = metta.rl.trainer_config.OptimizerConfig(
        type="adamw_schedulefree",
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0.01,
        warmup_steps=2000,
    )

    trainer_config = metta.rl.trainer_config.TrainerConfig(
        losses=loss_config,
        optimizer=optimizer_config,
        total_timesteps=50_000_000_000,
    )

    return metta.tools.train.TrainTool(
        training_env=base_tool.training_env,
        trainer=trainer_config,
        evaluator=base_tool.evaluator,
        policy_architecture=metta.agent.policies.vit_grpo.ViTGRPOConfig(),
    )
