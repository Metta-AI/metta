"""Sequential learning recipe for continual learning experiments.

This module implements sequential task training where agents learn tasks
one after another, with periodic evaluation on all tasks to measure
forgetting and transfer effects.

Usage:
    # Minimal 3-task test
    uv run python tools/run.py recipes.experiment.sequential_learning.train_cogs_minimal

    # Full coordination sequence
    uv run python tools/run.py recipes.experiment.sequential_learning.train_cogs_coordination
"""

from typing import Optional, Sequence

from metta.cogworks.curriculum import CurriculumConfig
from metta.cogworks.curriculum.task_generator import SequentialTaskGenerator
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.tools.train import TrainTool
from mettagrid.config.mettagrid_config import MettaGridConfig
from recipes.experiment.assembly_lines import make_assembly_line_eval_env
from recipes.experiment.cogs_v_clips import make_training_env

# ========== Helper Functions ==========


def make_sequential_curriculum(
    task_sequence: list[MettaGridConfig],
    repeat_sequence: bool = True,
) -> CurriculumConfig:
    """Create a curriculum with sequential task generator.

    Args:
        task_sequence: Ordered list of tasks for continual learning
        repeat_sequence: Whether to cycle through tasks repeatedly

    Returns:
        CurriculumConfig configured for sequential learning
    """
    seq_generator_config = SequentialTaskGenerator.Config(
        task_sequence=task_sequence,
        repeat_sequence=repeat_sequence,
    )

    return CurriculumConfig(
        task_generator=seq_generator_config,
        num_active_tasks=1,  # One task active at a time
        max_task_id=len(task_sequence) * 1000,
    )


# ========== Base Training Function ==========


def train(
    task_sequence: list[MettaGridConfig],
    steps_per_task: int = 100_000_000,
    continual_eval_interval: int = 50_000,
    continual_eval_episodes: int = 10_000,
    eval_mode: str = "all",
) -> TrainTool:
    """Create a training tool for sequential learning.

    Args:
        task_sequence: Ordered list of tasks to train on
        steps_per_task: Training steps per task before moving to next
        continual_eval_interval: Evaluate every N epochs
        continual_eval_episodes: Episodes per task evaluation
        eval_mode: Which tasks to evaluate ("all", "past", "current", "future")

    Returns:
        TrainTool configured for sequential learning
    """
    curriculum = make_sequential_curriculum(task_sequence)

    trainer_cfg = TrainerConfig(
        total_timesteps=steps_per_task * len(task_sequence),
        bptt_horizon=64,  # Ensure sufficient horizon for sequence learning
    )

    training_env_cfg = TrainingEnvironmentConfig(
        curriculum=curriculum,
        maps_cache_size=30,
    )

    # Disable standard evaluation (we use ContinualLearningEvaluator instead)
    evaluator_cfg = EvaluatorConfig(
        epoch_interval=0,
        evaluate_local=False,
        evaluate_remote=False,
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=training_env_cfg,
        evaluator=evaluator_cfg,
    )


# ========== CoGs Mission Sequences ==========


def train_cogs_coordination(
    num_agents: int = 4,
    steps_per_task: int = 100_000_000,
) -> TrainTool:
    """5-task sequence from individual to coordination missions.

    Progression:
    1. oxygen_bottleneck - Basic survival
    2. distant_resources - Spatial planning
    3. quadrant_buildings - Quadrant coordination
    4. single_use_swarm - Sequential cooperation
    5. easy_hearts - Multi-agent coordination

    Tests transfer from single-agent to multi-agent coordination,
    forgetting of basic skills, and complexity scaling.
    """
    task_sequence = [
        make_training_env(mission="oxygen_bottleneck", num_cogs=num_agents),
        make_training_env(mission="distant_resources", num_cogs=num_agents),
        make_training_env(mission="quadrant_buildings", num_cogs=num_agents),
        make_training_env(mission="single_use_swarm", num_cogs=num_agents),
        make_training_env(mission="easy_hearts", num_cogs=num_agents),
    ]

    return train(
        task_sequence=task_sequence,
        steps_per_task=steps_per_task,
    )


def train_cogs_exploration(
    num_agents: int = 4,
    steps_per_task: int = 100_000_000,
) -> TrainTool:
    """4-task sequence emphasizing exploration and spatial reasoning.

    Progression:
    1. oxygen_bottleneck - Basic navigation
    2. open_world - Open exploration
    3. harvest - Resource gathering
    4. balanced_corners - Complex navigation

    Tests navigation transfer, exploration vs. exploitation,
    and skill composition.
    """
    task_sequence = [
        make_training_env(mission="oxygen_bottleneck", num_cogs=num_agents),
        make_training_env(mission="open_world", num_cogs=num_agents),
        make_training_env(mission="harvest", num_cogs=num_agents),
        make_training_env(mission="balanced_corners", num_cogs=num_agents),
    ]

    return train(
        task_sequence=task_sequence,
        steps_per_task=steps_per_task,
    )


def train_cogs_hearts(
    num_agents: int = 4,
    steps_per_task: int = 100_000_000,
) -> TrainTool:
    """3-task focused sequence on coordination progression.

    Progression:
    1. easy_hearts_hello_world - Introduction
    2. easy_hearts - Basic coordination
    3. easy_hearts_training_facility - Advanced coordination

    Tests skill building within related tasks with minimal forgetting.
    """
    task_sequence = [
        make_training_env(mission="easy_hearts_hello_world", num_cogs=num_agents),
        make_training_env(mission="easy_hearts", num_cogs=num_agents),
        make_training_env(mission="easy_hearts_training_facility", num_cogs=num_agents),
    ]

    return train(
        task_sequence=task_sequence,
        steps_per_task=steps_per_task,
    )


def train_cogs_minimal(
    num_agents: int = 4,
    steps_per_task: int = 100_000_000,
) -> TrainTool:
    """Minimal 3-task sequence for debugging and quick tests.

    Progression:
    1. oxygen_bottleneck - Basic survival
    2. easy_hearts - Simple coordination
    3. distant_resources - Resource gathering

    Training with 100M steps per task.
    """
    task_sequence = [
        make_training_env(mission="oxygen_bottleneck", num_cogs=num_agents),
        make_training_env(mission="easy_hearts", num_cogs=num_agents),
        make_training_env(mission="distant_resources", num_cogs=num_agents),
    ]

    return train(
        task_sequence=task_sequence,
        steps_per_task=steps_per_task,
    )


# ========== Assembly Line Sequences ==========


def train_assembly_complexity(
    steps_per_task: int = 100_000_000,
) -> TrainTool:
    """4-task sequence with increasing chain complexity.

    Progression:
    1. Chain 1, 1 sink - Direct delivery
    2. Chain 2, 1 sink - Transformation
    3. Chain 2, 2 sinks - Parallelization
    4. Chain 3, 2 sinks - Complex chains

    Tests transfer of chaining operations and hierarchical planning.
    """
    task_sequence = [
        make_assembly_line_eval_env(
            chain_lengths=[1],
            num_sinks=1,
            room_size=16,
            terrain="grass",
        ),
        make_assembly_line_eval_env(
            chain_lengths=[2],
            num_sinks=1,
            room_size=16,
            terrain="grass",
        ),
        make_assembly_line_eval_env(
            chain_lengths=[2],
            num_sinks=2,
            room_size=18,
            terrain="grass",
        ),
        make_assembly_line_eval_env(
            chain_lengths=[3],
            num_sinks=2,
            room_size=20,
            terrain="grass",
        ),
    ]

    return train(
        task_sequence=task_sequence,
        steps_per_task=steps_per_task,
    )


def train_assembly_parallelization(
    steps_per_task: int = 100_000_000,
) -> TrainTool:
    """3-task sequence focusing on parallel sink management.

    Progression:
    1. 1 sink - Sequential processing
    2. 2 sinks - Parallel processing
    3. 3 sinks - High parallelization

    Fixed chain length (2) to isolate parallelization learning.
    """
    task_sequence = [
        make_assembly_line_eval_env(
            chain_lengths=[2],
            num_sinks=1,
            room_size=16,
            terrain="grass",
        ),
        make_assembly_line_eval_env(
            chain_lengths=[2],
            num_sinks=2,
            room_size=18,
            terrain="grass",
        ),
        make_assembly_line_eval_env(
            chain_lengths=[2],
            num_sinks=3,
            room_size=20,
            terrain="grass",
        ),
    ]

    return train(
        task_sequence=task_sequence,
        steps_per_task=steps_per_task,
    )


def train_assembly_terrain(
    steps_per_task: int = 100_000_000,
) -> TrainTool:
    """3-task sequence with varying terrain complexity.

    Progression:
    1. Grass - Open terrain
    2. Mountain - Obstacles
    3. Desert - Mixed complexity

    Fixed task complexity to isolate terrain adaptation.
    """
    task_sequence = [
        make_assembly_line_eval_env(
            chain_lengths=[2],
            num_sinks=2,
            room_size=18,
            terrain="grass",
        ),
        make_assembly_line_eval_env(
            chain_lengths=[2],
            num_sinks=2,
            room_size=18,
            terrain="mountain",
        ),
        make_assembly_line_eval_env(
            chain_lengths=[2],
            num_sinks=2,
            room_size=18,
            terrain="desert",
        ),
    ]

    return train(
        task_sequence=task_sequence,
        steps_per_task=steps_per_task,
    )


def train_assembly_minimal(
    steps_per_task: int = 100_000_000,
) -> TrainTool:
    """Minimal 3-task assembly sequence for debugging.

    Progression:
    1. Simple: chain 1, 1 sink
    2. Medium: chain 2, 1 sink
    3. Complex: chain 2, 2 sinks

    Training with 100M steps per task.
    """
    task_sequence = [
        make_assembly_line_eval_env(
            chain_lengths=[1],
            num_sinks=1,
            room_size=14,
            terrain="grass",
        ),
        make_assembly_line_eval_env(
            chain_lengths=[2],
            num_sinks=1,
            room_size=16,
            terrain="grass",
        ),
        make_assembly_line_eval_env(
            chain_lengths=[2],
            num_sinks=2,
            room_size=18,
            terrain="grass",
        ),
    ]

    return train(
        task_sequence=task_sequence,
        steps_per_task=steps_per_task,
    )


# ========== Custom Task Sequences ==========


def train_custom(
    missions: Sequence[str],
    num_agents: int = 4,
    variants: Optional[Sequence[str]] = None,
    steps_per_task: int = 100_000_000,
) -> TrainTool:
    """Train on custom sequence of CoGs missions.

    Args:
        missions: List of mission names (e.g., ["easy_hearts", "harvest"])
        num_agents: Number of agents per mission
        variants: Optional variants to apply to all missions
        steps_per_task: Training steps per task

    Returns:
        TrainTool configured for the custom sequence

    Example:
        >>> train_custom(
        ...     missions=["oxygen_bottleneck", "easy_hearts", "harvest"],
        ...     num_agents=4,
        ...     steps_per_task=200_000,
        ... )
    """
    task_sequence = [make_training_env(mission=mission, num_cogs=num_agents, variants=variants) for mission in missions]

    return train(
        task_sequence=task_sequence,
        steps_per_task=steps_per_task,
    )
