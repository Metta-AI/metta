"""Training recipe for cvc_random_maps with configurable Cortex architecture.

This recipe enables testing different memory architectures on CVC random maps:
- XL attention with mem_len=0 (no memory baseline)
- XL attention with mem_len=128 (attention with memory)
- sLSTM (pattern="S") - recurrent memory
- AgSA (pattern="AgSA") - AGaLiTe + sLSTM + Axon
- Any other Cortex pattern

Use this to compare memory vs no-memory architectures.
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
    # Cortex configuration
    pattern: str = "X",  # Cortex pattern: X, S, M, L, AgSA, AXMS, etc.
    d_hidden: int = 128,
    num_layers: int = 4,
    n_heads: int = 4,
    mem_len: int = 0,  # Memory length for XL cells (0 = no memory)
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    """
    Train on cvc_random_maps with configurable Cortex architecture.

    Supports various memory architectures for comparing memory requirements.

    Args:
        num_cogs: Number of agents per environment (default: 20)
        heart_buckets: Enable bucketing over heart inventory rewards
        resource_buckets: Enable bucketing over resource stat rewards
        initial_inventory_buckets: Enable bucketing over agent's initial inventory
        pattern: Cortex pattern string (default: "X")
            - "X" = XL attention (use mem_len to control memory)
            - "S" = sLSTM (recurrent)
            - "M" = mLSTM (matrix LSTM)
            - "L" = standard LSTM
            - "Ag" = AGaLiTe attention
            - "AgSA" = AGaLiTe + sLSTM + Axon (proven combo)
            - "AXMS" = Axon + XL + mLSTM + sLSTM
        d_hidden: Hidden dimension for Cortex stack (default: 128)
        num_layers: Number of layers (default: 4)
        n_heads: Number of attention heads for XL/Ag cells (default: 4)
        mem_len: Memory length for XL attention cells (default: 0 = no memory)
        policy_architecture: Optional override for policy architecture

    Returns:
        TrainTool configured for training

    Examples:
        # XL with no memory (baseline):
        ./tools/run.py recipes.experiment.cortex_paper.memory_needed.cvc_random_maps_train.train \\
            pattern=X mem_len=0 run=xl_memlen0 trainer.total_timesteps=2000000000

        # XL with memory:
        ./tools/run.py recipes.experiment.cortex_paper.memory_needed.cvc_random_maps_train.train \\
            pattern=X mem_len=128 run=xl_memlen128 trainer.total_timesteps=2000000000

        # sLSTM only:
        ./tools/run.py recipes.experiment.cortex_paper.memory_needed.cvc_random_maps_train.train \\
            pattern=S run=slstm_only trainer.total_timesteps=2000000000

        # AgSA (proven architecture):
        ./tools/run.py recipes.experiment.cortex_paper.memory_needed.cvc_random_maps_train.train \\
            pattern=AgSA run=agsa trainer.total_timesteps=2000000000

        # Via SkyPilot:
        ./devops/skypilot/launch.py \\
            recipes.experiment.cortex_paper.memory_needed.cvc_random_maps_train.train \\
            --gpus=8 pattern=AgSA run=agsa_2b trainer.total_timesteps=2000000000
    """
    # Build curriculum from cvc_random_maps
    curriculum = make_random_maps_curriculum(
        num_cogs=num_cogs,
        heart_buckets=heart_buckets,
        resource_buckets=resource_buckets,
        initial_inventory_buckets=initial_inventory_buckets,
    )

    # Build Cortex policy architecture
    if policy_architecture is None:
        # Override configs for XL cells if pattern contains X
        override_configs = []
        if "X" in pattern.upper():
            xl_cell_override = XLCellConfig(
                mem_len=mem_len,
                n_heads=n_heads,
            )
            override_configs.append(xl_cell_override)

        stack_cfg = build_cortex_auto_config(
            d_hidden=d_hidden,
            num_layers=num_layers,
            pattern=pattern,
            post_norm=True,
            compile_blocks=True,
            override_global_configs=override_configs if override_configs else None,
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
