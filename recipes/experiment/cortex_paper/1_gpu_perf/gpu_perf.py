"""Training recipe for cvc_random_maps with Cortex AgSA architecture.

Combines the cvc_random_maps curriculum with a configurable Cortex stack,
specifically for the GPU scaling experiments in the Cortex paper.
"""

from typing import Optional

from cortex.stacks import build_cortex_auto_config

from metta.agent.policies.cortex import CortexBaseConfig
from metta.agent.policy import PolicyArchitecture
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.tools.train import TrainTool
from recipes.experiment.cvc.cvc_random_maps import (
    make_random_maps_curriculum,
    make_training_eval_suite,
)


def train(
    num_cogs: int = 20,
    heart_buckets: bool = False,
    resource_buckets: bool = False,
    initial_inventory_buckets: bool = False,
    # Cortex configuration
    pattern: str = "AgSA",
    d_hidden: int = 128,
    num_layers: int = 3,
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    """
    Train on cvc_random_maps curriculum with Cortex architecture.

    This recipe combines the cvc_random_maps curriculum (which tests memory
    via procedurally generated maps) with the Cortex recurrent architecture.

    Args:
        num_cogs: Number of agents per environment (default: 20)
        heart_buckets: Enable bucketing over heart inventory rewards
        resource_buckets: Enable bucketing over resource stat rewards
        initial_inventory_buckets: Enable bucketing over agent's initial inventory
        pattern: Cortex pattern string (default: "AgSA")
            - "AgSA" = Associative memory + Axon + sLSTM + Axon
            - "AXMS" = Axon + Transformer-XL + mLSTM + sLSTM
            - See cortex.stacks.auto for available patterns
        d_hidden: Hidden dimension for Cortex stack (default: 128)
        num_layers: Number of Cortex layers (default: 3)
        policy_architecture: Optional override for policy architecture

    Returns:
        TrainTool configured for training

    Examples:
        # Train with default AgSA pattern:
        ./tools/run.py cortex_paper.gpu_scaling.train run=test

        # Train with custom pattern:
        ./tools/run.py cortex_paper.gpu_scaling.train \\
            pattern=AXMS d_hidden=256 num_layers=4 run=test

        # Use via SkyPilot:
        ./devops/skypilot/launch.py \\
            cortex_paper.gpu_scaling.train \\
            --gpus=4 \\
            run=gpu_scaling_test \\
            trainer.total_timesteps=20000000
    """
    # Build curriculum from cvc_random_maps
    curriculum = make_random_maps_curriculum(
        num_cogs=num_cogs,
        heart_buckets=heart_buckets,
        resource_buckets=resource_buckets,
        initial_inventory_buckets=initial_inventory_buckets,
    )

    # Build Cortex policy architecture if not provided
    if policy_architecture is None:
        stack_cfg = build_cortex_auto_config(
            d_hidden=d_hidden,
            num_layers=num_layers,
            pattern=pattern,
            post_norm=True,
        )
        policy_architecture = CortexBaseConfig(stack_cfg=stack_cfg)

    # Build training tool
    return TrainTool(
        trainer=TrainerConfig(),
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=make_training_eval_suite(num_cogs=num_cogs)),
        policy_architecture=policy_architecture,
    )


__all__ = ["train"]
