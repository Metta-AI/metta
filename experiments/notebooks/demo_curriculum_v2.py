#!/usr/bin/env python3
"""
Demo script for the new curriculum system (v2).

This demonstrates the redesigned architecture with TaskSets and Curricula.
"""

import sys

sys.path.append(".")

from metta.mettagrid.config import builder
from experiments.notebooks.utils.curriculum_v2 import (
    WeightedTaskSet,
    BucketedTaskSet,
    RandomCurriculum,
    LearningProgressCurriculum,
)
from experiments.notebooks.utils.curriculum_builder_v2 import (
    TaskSetBuilder,
    BucketedTaskSetBuilder,
    CurriculumBuilder,
)


def demo_weighted_task_set():
    """Demo WeightedTaskSet with multiple configs."""
    print("=== WeightedTaskSet Demo ===")

    # Create different environment configs
    easy_config = builder.arena(num_agents=2, combat=False)
    hard_config = builder.arena(num_agents=8, combat=True)

    # Create weighted task set with overrides
    task_set = WeightedTaskSet(
        items=[(easy_config, 2.0), (hard_config, 1.0)],
        overrides={"game.max_steps": 500, "game.episode_truncates": False},
        seed=42,
    )

    # Generate multiple tasks (should be deterministic with same seed)
    # Note: With same seed, TaskSet will generate same task repeatedly
    print("Generated tasks from WeightedTaskSet (deterministic with seed=42):")
    config = task_set.get_task()
    is_combat = config.game.actions.attack is not None
    print(
        f"  Task: {config.game.num_agents} agents, combat={is_combat}, steps={config.game.max_steps}"
    )

    # To get variety, use different seeds or RandomCurriculum
    print("Different tasks with different seeds:")
    for i, seed in enumerate([10, 20, 30]):
        varied_task_set = WeightedTaskSet(
            items=[(easy_config, 2.0), (hard_config, 1.0)],
            overrides={"game.max_steps": 500},
            seed=seed,
        )
        config = varied_task_set.get_task()
        is_combat = config.game.actions.attack is not None
        print(f"  Seed {seed}: {config.game.num_agents} agents, combat={is_combat}")

    return task_set


def demo_bucketed_task_set():
    """Demo BucketedTaskSet with parameter ranges."""
    print("\\n=== BucketedTaskSet Demo ===")

    # Base configuration
    base_config = builder.arena(num_agents=4, combat=True)

    # Create bucketed task set
    buckets = {
        "game.num_agents": [2, 4, 8, 16],  # Discrete values
        "game.max_steps": {"range": [200, 1000]},  # Continuous range
        "game.agent.rewards.inventory.heart": [0.0, 1.0, 2.0],  # Reward values
    }

    task_set = BucketedTaskSet(
        base_config=base_config,
        buckets=buckets,
        overrides={"game.episode_truncates": False},
        seed=123,
    )

    print("Generated tasks from BucketedTaskSet with different seeds:")
    for i, seed in enumerate([123, 456, 789]):
        seeded_task_set = BucketedTaskSet(
            base_config=base_config,
            buckets=buckets,
            overrides={"game.episode_truncates": False},
            seed=seed,
        )
        config = seeded_task_set.get_task()
        heart_reward = config.game.agent.rewards.inventory.heart
        print(
            f"  Seed {seed}: {config.game.num_agents} agents, {config.game.max_steps} steps, heart_reward={heart_reward}"
        )

    return task_set


def demo_task_set_composition():
    """Demo composing TaskSets hierarchically."""
    print("\\n=== TaskSet Composition Demo ===")

    # Create base task sets
    easy_configs = [builder.arena(num_agents=n, combat=False) for n in [2, 4]]
    hard_configs = [builder.arena(num_agents=n, combat=True) for n in [8, 16]]

    easy_task_set = WeightedTaskSet([(cfg, 1.0) for cfg in easy_configs], seed=111)
    hard_task_set = WeightedTaskSet([(cfg, 1.0) for cfg in hard_configs], seed=222)

    # Compose them with different weights
    composed_task_set = WeightedTaskSet(
        items=[(easy_task_set, 3.0), (hard_task_set, 1.0)],  # 3:1 ratio easy:hard
        overrides=["game.episode_truncates: false", "game.max_steps: 750"],
        seed=333,
    )

    print("Generated tasks from composed TaskSet:")
    for i in range(5):
        config = composed_task_set.get_task()
        is_combat = config.game.actions.attack is not None
        print(f"  Task {i + 1}: {config.game.num_agents} agents, combat={is_combat}")

    return composed_task_set


def demo_curricula(task_set):
    """Demo different curriculum strategies."""
    print("\\n=== Curriculum Demo ===")

    # Random curriculum
    random_curr = RandomCurriculum(task_set, seed=456)
    print("Random curriculum tasks:")
    for i in range(3):
        task = random_curr.get_task()
        config = task.get_env_config()
        is_combat = config.game.actions.attack is not None
        print(f"  {task.task_id}: {config.game.num_agents} agents, combat={is_combat}")

    # Learning progress curriculum
    lp_curr = LearningProgressCurriculum(task_set, num_tasks=5, seed=789)
    print(
        f"\\nLearning progress curriculum (pre-generated {len(lp_curr.tasks)} tasks):"
    )
    for i in range(3):
        task = lp_curr.get_task()
        config = task.get_env_config()
        is_combat = config.game.actions.attack is not None
        print(f"  {task.task_id}: {config.game.num_agents} agents, combat={is_combat}")


def demo_builders():
    """Demo the fluent builder APIs."""
    print("\\n=== Builder API Demo ===")

    # Create configs
    config1 = builder.arena(num_agents=2, combat=False)
    config2 = builder.arena(num_agents=4, combat=True)

    # Build TaskSet with fluent API
    task_set = (
        TaskSetBuilder(seed=100)
        .add_config(config1, weight=2.0)
        .add_config(config2, weight=1.0)
        .add_override("game.max_steps", 600)
        .add_override("game.episode_truncates", False)
        .build()
    )

    print("TaskSet built with fluent API:")
    config = task_set.get_task()
    print(
        f"  Sample task: {config.game.num_agents} agents, {config.game.max_steps} steps"
    )

    # Build BucketedTaskSet with fluent API
    bucketed_set = (
        BucketedTaskSetBuilder(config1, seed=200)
        .add_bucket_values("game.num_agents", [2, 4, 8])
        .add_bucket_range("game.max_steps", 300, 900)
        .add_override("game.episode_truncates", False)
        .build()
    )

    print("BucketedTaskSet built with fluent API:")
    config = bucketed_set.get_task()
    print(
        f"  Sample task: {config.game.num_agents} agents, {config.game.max_steps} steps"
    )

    # Build Curriculum with fluent API
    curriculum = CurriculumBuilder(task_set).as_random(seed=300).build()

    task = curriculum.get_task()
    config = task.get_env_config()
    print("Curriculum built with fluent API:")
    print(f"  Sample task: {task.task_id}, {config.game.num_agents} agents")


def demo_deterministic_behavior():
    """Demo that TaskSets are deterministic with same seeds."""
    print("\\n=== Deterministic Behavior Demo ===")

    base_config = builder.arena(num_agents=4, combat=False)

    # Create two identical TaskSets with same seed
    buckets = {"game.num_agents": [2, 4, 8, 16]}

    task_set1 = BucketedTaskSet(base_config, buckets, seed=999)
    task_set2 = BucketedTaskSet(base_config, buckets, seed=999)

    print("Two TaskSets with same seed (999):")
    for i in range(3):
        config1 = task_set1.get_task()
        config2 = task_set2.get_task()
        agents1 = config1.game.num_agents
        agents2 = config2.game.num_agents
        match = "✓" if agents1 == agents2 else "✗"
        print(f"  Task {i + 1}: TaskSet1={agents1}, TaskSet2={agents2} {match}")

    # Test with different seeds
    task_set3 = BucketedTaskSet(base_config, buckets, seed=1000)
    print("\\nTaskSet with different seed (1000):")
    for i in range(3):
        config3 = task_set3.get_task()
        print(f"  Task {i + 1}: {config3.game.num_agents} agents")


if __name__ == "__main__":
    print("Curriculum System v2 Demo\\n")

    # Run all demos
    task_set = demo_weighted_task_set()
    bucketed_set = demo_bucketed_task_set()
    composed_set = demo_task_set_composition()

    demo_curricula(task_set)
    demo_builders()
    demo_deterministic_behavior()

    print("\\nDemo complete! The new curriculum system provides:")
    print("✓ Deterministic task generation from seeds")
    print("✓ Hierarchical composition of TaskSets")
    print("✓ Flexible parameter overrides and sampling")
    print("✓ Clean separation between task generation and curriculum strategy")
    print("✓ Fluent builder APIs for easy configuration")
