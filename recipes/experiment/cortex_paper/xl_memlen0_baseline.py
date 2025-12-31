"""Training recipe for cvc_random_maps with XL attention (mem_len=0) baseline.

This recipe tests whether CVC random maps require memory by using
Transformer-XL style attention with mem_len=0, meaning no state is
carried across timesteps. This is equivalent to standard causal
self-attention within each sequence/step.

Use this as a baseline to compare against memory-enabled architectures.
"""

from typing import Optional

from cortex.config import XLCellConfig
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
    # Cortex XL configuration
    d_hidden: int = 128,
    num_layers: int = 4,
    n_heads: int = 4,
    mem_len: int = 0,  # 0 = no memory across steps (baseline)
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    """
    Train on cvc_random_maps with XL attention (mem_len=0 by default).

    This is a memory-free baseline using Transformer-XL style attention
    but without cross-step memory. Use this to test whether an environment
    actually requires memory capabilities.

    Args:
        num_cogs: Number of agents per environment (default: 20)
        heart_buckets: Enable bucketing over heart inventory rewards
        resource_buckets: Enable bucketing over resource stat rewards
        initial_inventory_buckets: Enable bucketing over agent's initial inventory
        d_hidden: Hidden dimension for Cortex stack (default: 128)
        num_layers: Number of attention layers (default: 4)
        n_heads: Number of attention heads (default: 4)
        mem_len: Memory length for XL attention (default: 0 = no memory)
        policy_architecture: Optional override for policy architecture

    Returns:
        TrainTool configured for training

    Examples:
        # Train with mem_len=0 (no memory baseline):
        ./tools/run.py recipes.experiment.cortex_paper.xl_memlen0_baseline.train \\
            run=xl_memlen0_test trainer.total_timesteps=100000

        # Compare with memory-enabled version:
        ./tools/run.py recipes.experiment.cortex_paper.xl_memlen0_baseline.train \\
            mem_len=128 run=xl_memlen128_test trainer.total_timesteps=100000

        # Use via SkyPilot for 2B timesteps on 8x L4:
        ./devops/skypilot/launch.py \\
            recipes.experiment.cortex_paper.xl_memlen0_baseline.train \\
            --gpus=8 \\
            run=xl_memlen0_2b \\
            trainer.total_timesteps=2000000000
    """
    # Build curriculum from cvc_random_maps
    curriculum = make_random_maps_curriculum(
        num_cogs=num_cogs,
        heart_buckets=heart_buckets,
        resource_buckets=resource_buckets,
        initial_inventory_buckets=initial_inventory_buckets,
    )

    # Build Cortex policy architecture with XL attention (configurable mem_len)
    if policy_architecture is None:
        # Create XL cell config with specified mem_len
        xl_cell_override = XLCellConfig(
            mem_len=mem_len,
            n_heads=n_heads,
        )

        # Build stack with all-X pattern (XL attention layers only)
        # Use override_global_configs to apply mem_len to all XL cells
        stack_cfg = build_cortex_auto_config(
            d_hidden=d_hidden,
            num_layers=num_layers,
            pattern="X",  # XL attention only (no recurrent cells)
            post_norm=True,
            compile_blocks=True,
            override_global_configs=[xl_cell_override],
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
