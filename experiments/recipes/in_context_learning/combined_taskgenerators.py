"""
Combined Task Generators Recipe for In-Context Learning.

This recipe demonstrates the multi-task generator curriculum functionality
by combining assembly line and foraging task generators.
"""

import subprocess
import time
from typing import Optional

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.tools.train import TrainTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool

# Import the individual task generators
from experiments.recipes.in_context_learning.assembly_lines import (
    make_task_generator_cfg as make_assembly_cfg,
)
from experiments.recipes.in_context_learning.foraging import (
    make_task_generator_cfg as make_foraging_cfg,
)


def make_combined_curriculum_config(
    assembly_style: str = "multi_agent_easy",
    foraging_style: str = "multi_agent_multi_altars",
    num_active_tasks: int = 32,
    min_generator_proportion: float = 0.1,
    map_dir: Optional[str] = None,
) -> CurriculumConfig:
    """Create a curriculum config that combines assembly line and foraging generators.

    Args:
        assembly_style: Style for assembly line tasks (from assembly_lines.py curriculum_args)
        foraging_style: Style for foraging tasks (from foraging.py curriculum_args)
        num_active_tasks: Number of active tasks in the curriculum pool
        min_generator_proportion: Minimum proportion of tasks from each generator
        map_dir: Optional directory for pre-saved maps

    Returns:
        CurriculumConfig with multiple task generators
    """

    # Assembly line generator configurations
    assembly_args = {
        "single_agent_easy": {
            "num_agents": [1],
            "chain_lengths": [2, 3],
            "num_sinks": [0, 1],
            "room_sizes": ["tiny", "small"],
            "positions": [["Any"]],
        },
        "multi_agent_easy": {
            "num_agents": [1, 2],
            "chain_lengths": [2, 3],
            "num_sinks": [0, 1],
            "room_sizes": ["tiny", "small"],
            "positions": [["Any"], ["Any", "Any"]],
        },
        "multi_agent_hard": {
            "num_agents": [1, 2],
            "chain_lengths": [2, 3, 4, 5],
            "num_sinks": [0, 1, 2],
            "room_sizes": ["tiny", "small", "medium"],
            "positions": [["Any"], ["Any", "Any"]],
        },
    }

    # Foraging generator configurations
    foraging_args = {
        "single_agent_easy": {
            "num_agents": [1],
            "num_altars": [2, 5],
            "num_generators": [0, 1],
            "room_sizes": ["small", "medium"],
            "positions": [["Any"]],
        },
        "multi_agent_multi_altars": {
            "num_agents": [1, 2, 3],
            "num_altars": list(range(5, 20, 5)),
            "num_generators": [0],
            "room_sizes": ["small", "medium", "large"],
            "positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        },
        "multi_agent_with_generators": {
            "num_agents": [1, 2, 3],
            "num_altars": list(range(5, 15, 5)),
            "num_generators": list(range(1, 10, 2)),
            "room_sizes": ["small", "medium", "large"],
            "positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        },
    }

    # Create assembly line task generator config
    assembly_config = make_assembly_cfg(
        **assembly_args[assembly_style],
        map_dir=map_dir,
    )

    # Create foraging task generator config
    foraging_config = make_foraging_cfg(
        **foraging_args[foraging_style],
        map_dir=map_dir,
    )

    # Create curriculum config with multiple generators
    curriculum_config = CurriculumConfig(
        task_generators=[assembly_config, foraging_config],
        num_active_tasks=num_active_tasks,
        min_generator_proportion=min_generator_proportion,
        algorithm_config=LearningProgressConfig(
            use_bidirectional=True,
            num_active_tasks=num_active_tasks,
            rand_task_rate=0.1,
            exploration_bonus=0.1,
        ),
    )

    return curriculum_config


def train(
    assembly_style: str = "multi_agent_easy",
    foraging_style: str = "multi_agent_multi_altars",
    num_active_tasks: int = 32,
    min_generator_proportion: float = 0.1,
) -> TrainTool:
    """Train using combined assembly line and foraging task generators.

    Args:
        assembly_style: Style for assembly line tasks
        foraging_style: Style for foraging tasks
        num_active_tasks: Number of active tasks in curriculum pool
        min_generator_proportion: Minimum proportion from each generator

    Returns:
        TrainTool configured for combined curriculum training
    """

    # Create combined curriculum
    curriculum_config = make_combined_curriculum_config(
        assembly_style=assembly_style,
        foraging_style=foraging_style,
        num_active_tasks=num_active_tasks,
        min_generator_proportion=min_generator_proportion,
        map_dir=None,  # Disable pre-saved maps for training
    )

    # Create eval suite combining both task types
    def make_combined_eval_suite():
        from experiments.evals.in_context_learning.assembly_lines import (
            make_icl_assembler_resource_chain_eval_suite,
        )
        from experiments.evals.in_context_learning.foraging import (
            make_assembler_eval_suite,
        )

        # Combine evaluation suites from both task types
        assembly_evals = make_icl_assembler_resource_chain_eval_suite()
        foraging_evals = make_assembler_eval_suite()

        return assembly_evals + foraging_evals

    # Use the train_icl function but pass curriculum config directly
    return TrainTool(
        curriculum=curriculum_config,
        eval_suite_factory=make_combined_eval_suite,
        name=f"combined_taskgens_{assembly_style}_{foraging_style}",
        suite="in_context_learning",
    )


def play(
    assembly_style: str = "multi_agent_easy",
    foraging_style: str = "multi_agent_multi_altars",
) -> PlayTool:
    """Play using combined task generators.

    Args:
        assembly_style: Style for assembly line tasks
        foraging_style: Style for foraging tasks

    Returns:
        PlayTool for interactive play
    """

    curriculum_config = make_combined_curriculum_config(
        assembly_style=assembly_style,
        foraging_style=foraging_style,
        num_active_tasks=16,
    )

    return PlayTool(
        curriculum=curriculum_config,
        name="combined_taskgens_play",
        suite="in_context_learning",
    )


def replay(
    assembly_style: str = "multi_agent_easy",
    foraging_style: str = "multi_agent_multi_altars",
    policy_uri: Optional[str] = None,
) -> ReplayTool:
    """Replay using combined task generators.

    Args:
        assembly_style: Style for assembly line tasks
        foraging_style: Style for foraging tasks
        policy_uri: URI to policy for replay

    Returns:
        ReplayTool for policy replay
    """

    curriculum_config = make_combined_curriculum_config(
        assembly_style=assembly_style,
        foraging_style=foraging_style,
        num_active_tasks=16,
    )

    # Default policy if none provided
    if policy_uri is None:
        policy_uri = "s3://softmax-public/policies/icl_resource_chain_terrain_4.2.2025-09-24/icl_resource_chain_terrain_4.2.2025-09-24:v2370.pt"

    return ReplayTool(
        curriculum=curriculum_config,
        policy_uri=policy_uri,
        name="combined_taskgens_replay",
        suite="in_context_learning",
    )


def experiment():
    """Run experiments with different combinations of task generators."""

    combinations = [
        ("single_agent_easy", "single_agent_easy"),
        ("multi_agent_easy", "multi_agent_multi_altars"),
        ("multi_agent_hard", "multi_agent_with_generators"),
    ]

    for assembly_style, foraging_style in combinations:
        run_name = f"combined_taskgens_{assembly_style}_{foraging_style}.{time.strftime('%Y-%m-%d')}"

        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.in_context_learning.combined_taskgenerators.train",
                f"run={run_name}",
                f"assembly_style={assembly_style}",
                f"foraging_style={foraging_style}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )
        time.sleep(1)


def demonstrate_multi_generator_curriculum():
    """Demonstrate the multi-generator curriculum functionality."""

    print("=== Combined Task Generators Curriculum Demo ===\n")

    # Create curriculum
    curriculum_config = make_combined_curriculum_config(
        assembly_style="multi_agent_easy",
        foraging_style="multi_agent_multi_altars",
        num_active_tasks=20,
        min_generator_proportion=0.2,  # At least 20% from each
    )

    curriculum = curriculum_config.make()

    print("Created curriculum with:")
    print("  - Assembly line generator (multi_agent_easy)")
    print("  - Foraging generator (multi_agent_multi_altars)")
    print("  - Pool size: 20 tasks")
    print("  - Minimum 20% representation per generator")
    print()

    # Sample tasks and simulate performance
    assembly_count = 0
    foraging_count = 0

    for i in range(30):
        task = curriculum.get_task()
        env_cfg = task.get_env_cfg()

        # Determine task type
        if "chain" in env_cfg.label or "sinks" in env_cfg.label:
            assembly_count += 1
            # Simulate assembly lines getting better over time
            score = 0.3 + min(0.6, i / 30.0) + (i % 3) * 0.1  # Some randomness
        else:
            foraging_count += 1
            # Simulate foraging staying roughly constant
            score = 0.7 + (i % 5) * 0.05  # Some randomness

        # Clamp score
        score = max(0.0, min(1.0, score))

        # Update curriculum
        curriculum.update_task_performance(task._task_id, score)
        task.complete(score)

        if (i + 1) % 10 == 0:
            stats = curriculum.get_base_stats()
            print(f"After {i + 1} tasks:")
            print(f"  Assembly: {assembly_count}, Foraging: {foraging_count}")
            print(
                f"  Generator 0 mean score: {stats.get('generator_0_mean_score', 'N/A'):.3f}"
            )
            print(
                f"  Generator 1 mean score: {stats.get('generator_1_mean_score', 'N/A'):.3f}"
            )
            print(
                f"  Generator 0 pool: {stats.get('generator_0_pool_count', 'N/A'):.0f} ({stats.get('generator_0_pool_proportion', 0):.1%})"
            )
            print(
                f"  Generator 1 pool: {stats.get('generator_1_pool_count', 'N/A'):.0f} ({stats.get('generator_1_pool_proportion', 0):.1%})"
            )
            print(
                f"  Generator 0 total created: {stats.get('generator_0_task_count', 'N/A')}"
            )
            print(
                f"  Generator 1 total created: {stats.get('generator_1_task_count', 'N/A')}"
            )
            print()

    print("Demo completed! Assembly lines should show improving scores over time,")
    print("leading to more assembly line tasks being generated in later stages.")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_multi_generator_curriculum()
