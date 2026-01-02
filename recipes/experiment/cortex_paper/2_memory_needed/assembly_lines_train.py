"""Training recipe for assembly_lines with configurable Cortex architecture.

This recipe enables testing different memory architectures on Assembly Lines:
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
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.tools.train import TrainTool
from recipes.experiment.assembly_lines import (
    curriculum_args,
    make_assembly_line_eval_suite,
    make_task_generator_cfg,
)


def train(
    curriculum_style: str = "all_room_sizes",
    # Cortex configuration
    pattern: str = "X",  # Cortex pattern: X, S, M, L, AgSA, AXMS, etc.
    d_hidden: int = 128,
    num_layers: int = 4,
    n_heads: int = 4,
    mem_len: int = 0,  # Memory length for XL cells (0 = no memory)
    policy_architecture: Optional[PolicyArchitecture] = None,
) -> TrainTool:
    """
    Train on assembly_lines with configurable Cortex architecture.

    Supports various memory architectures for comparing memory requirements.

    Args:
        curriculum_style: Curriculum difficulty preset (default: "all_room_sizes")
            Options: level_0, level_1, level_2, tiny, tiny_small, all_room_sizes,
                     longer_chains, terrain_1-4, full
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
        ./tools/run.py recipes.experiment.cortex_paper.2_memory_needed.assembly_lines_train.train \\
            pattern=X mem_len=0 run=assembly_xl_memlen0 trainer.total_timesteps=2000000000

        # XL with memory:
        ./tools/run.py recipes.experiment.cortex_paper.2_memory_needed.assembly_lines_train.train \\
            pattern=X mem_len=128 run=assembly_xl_memlen128 trainer.total_timesteps=2000000000

        # sLSTM only:
        ./tools/run.py recipes.experiment.cortex_paper.2_memory_needed.assembly_lines_train.train \\
            pattern=S run=assembly_slstm trainer.total_timesteps=2000000000

        # AgSA (proven architecture):
        ./tools/run.py recipes.experiment.cortex_paper.2_memory_needed.assembly_lines_train.train \\
            pattern=AgSA run=assembly_agsa trainer.total_timesteps=2000000000
    """
    # Build curriculum from assembly_lines
    task_generator_cfg = make_task_generator_cfg(**curriculum_args[curriculum_style])
    curriculum = CurriculumConfig(
        task_generator=task_generator_cfg,
        algorithm_config=LearningProgressConfig(),
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
        evaluator=EvaluatorConfig(simulations=make_assembly_line_eval_suite()),
        policy_architecture=policy_architecture,
    )


__all__ = ["train"]
