import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"# Mettabook - Curriculum System v2")
    return


@app.cell
def _(mo):
    mo.md(r"## Setup")
    return


@app.cell
def _():
    # Status check and basic imports
    import subprocess
    import os
    from datetime import datetime

    subprocess.run(
        ["metta", "status", "--components=core,system,aws,wandb", "--non-interactive"]
    )
    return subprocess, os, datetime


@app.cell
def _():
    import altair as alt
    import pandas as pd
    # Note: These utilities have been removed - curriculum builders are now in cogworks/curriculum
    # from experiments.notebooks.utils.metrics import fetch_metrics
    # from experiments.notebooks.utils.monitoring_marimo import monitor_training_statuses
    # from experiments.notebooks.utils.replays import show_replay

    # Curriculum system is now in cogworks/curriculum
    from cogworks.curriculum.task_set import (
        WeightedTaskSet,
        BuckettedTaskSet as BucketedTaskSet,
    )
    from cogworks.curriculum.curriculum import (
        RandomCurriculum,
        LearningProgressCurriculum,
        Task,
    )
    from cogworks.curriculum.builders import (
        TaskSetBuilder,
        BuckettedTaskSetBuilder as BucketedTaskSetBuilder,
        CurriculumBuilder,
        bucketed_task_set,
        random_curriculum,
    )
    from metta.mettagrid.config import builder

    print("Setup complete!")
    return (
        alt,
        pd,
        WeightedTaskSet,
        BucketedTaskSet,
        RandomCurriculum,
        LearningProgressCurriculum,
        Task,
        TaskSetBuilder,
        BucketedTaskSetBuilder,
        CurriculumBuilder,
        bucketed_task_set,
        random_curriculum,
        builder,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Create TaskSets

        TaskSets generate environment configurations deterministically from seeds.
        They can contain weighted lists of configs or other TaskSets, plus parameter overrides.
        """
    )
    return


@app.cell
def _(WeightedTaskSet, builder):
    # Example 1: WeightedTaskSet with environment configs

    # Create different environment configurations
    easy_config = builder.arena(num_agents=2, combat=False)
    medium_config = builder.arena(num_agents=4, combat=True)
    hard_config = builder.arena(num_agents=8, combat=True)

    # Create weighted TaskSet
    dehydration_task_set = WeightedTaskSet(
        items=[
            (easy_config, 3.0),  # 3x weight for easy
            (medium_config, 2.0),  # 2x weight for medium
            (hard_config, 1.0),  # 1x weight for hard
        ],
        overrides={
            "game.agent.rewards.inventory.heart": 2.0,  # Higher water rewards
            "game.agent.resource_limits.heart": 10,  # Lower water capacity
            "game.max_steps": 500,  # Shorter episodes
            "game.episode_truncates": False,
        },
        seed=42,
    )

    print("Created WeightedTaskSet with 3 configs and dehydration overrides")

    # Test task generation with different seeds
    print("Sample tasks with different seeds:")
    for seed in [42, 100, 200]:
        test_set = WeightedTaskSet(
            items=[(easy_config, 1.0), (hard_config, 1.0)], seed=seed
        )
        config = test_set.get_task()
        is_combat = config.game.actions.attack is not None
        print(f"  Seed {seed}: {config.game.num_agents} agents, combat={is_combat}")

    return dehydration_task_set, easy_config, medium_config, hard_config


@app.cell
def _(BucketedTaskSet, builder):
    # Example 2: BucketedTaskSet for parameter sweeps

    # Base configuration for bucketing
    base_config = builder.arena(num_agents=4, combat=True)

    # Define parameter buckets
    buckets = {
        # Discrete values for agent count
        "game.num_agents": [2, 4, 8, 16],
        # Continuous range for episode length
        "game.max_steps": {"range": [300, 800]},
        # Discrete reward values
        "game.agent.rewards.inventory.heart": [0.0, 1.0, 2.0, 3.0],
        "game.agent.rewards.inventory.ore_red": [0.0, 0.5, 1.0],
        # Resource limits
        "game.agent.resource_limits.heart": {"range": [5, 20]},
    }

    # Create bucketed TaskSet
    bucketed_dehydration = BucketedTaskSet(
        base_config=base_config,
        buckets=buckets,
        overrides={"game.episode_truncates": False},
        seed=123,
    )

    print("Created BucketedTaskSet with 5 parameter buckets")
    print("This generates tasks by sampling from each bucket independently")

    # Show variety with different seeds
    print("Sample bucketed tasks:")
    for seed in [123, 456, 789]:
        test_set = BucketedTaskSet(base_config, buckets, seed=seed)
        config = test_set.get_task()
        heart_reward = config.game.agent.rewards.inventory.heart
        ore_reward = config.game.agent.rewards.inventory.ore_red
        print(
            f"  Seed {seed}: {config.game.num_agents} agents, heart={heart_reward}, ore={ore_reward}"
        )

    return bucketed_dehydration, buckets


@app.cell
def _(WeightedTaskSet, dehydration_task_set, bucketed_dehydration):
    # Example 3: Hierarchical TaskSet composition

    # Compose TaskSets hierarchically
    mixed_task_set = WeightedTaskSet(
        items=[
            (dehydration_task_set, 2.0),  # 2x weight for dehydration scenarios
            (bucketed_dehydration, 1.0),  # 1x weight for bucketed exploration
        ],
        overrides=["game.track_movement_metrics: true"],  # String format overrides
        seed=999,
    )

    print("Created hierarchical TaskSet composing WeightedTaskSet + BucketedTaskSet")
    print("This allows combining different task generation strategies")

    # Test the composed TaskSet
    config = mixed_task_set.get_task()
    print(
        f"Sample from mixed TaskSet: {config.game.num_agents} agents, movement_tracking={config.game.track_movement_metrics}"
    )

    return (mixed_task_set,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Create Curricula

        Curricula decide which tasks to return using different strategies.
        They wrap TaskSets and implement selection logic.
        """
    )
    return


@app.cell
def _(RandomCurriculum, LearningProgressCurriculum, dehydration_task_set):
    # Example 1: RandomCurriculum - generates new random seeds

    random_curriculum = RandomCurriculum(dehydration_task_set, seed=42)

    print("Created RandomCurriculum:")
    print("This generates a new random seed for each get_task() call")

    # Get multiple tasks - each will have different characteristics
    for i in range(3):
        task = random_curriculum.get_task()
        config = task.get_env_config()
        is_combat = config.game.actions.attack is not None
        heart_reward = config.game.agent.rewards.inventory.heart
        print(
            f"  {task.task_id}: {config.game.num_agents} agents, combat={is_combat}, heart_reward={heart_reward}"
        )

    return (random_curriculum,)


@app.cell
def _(LearningProgressCurriculum, bucketed_dehydration):
    # Example 2: LearningProgressCurriculum - pre-generates tasks

    lp_curriculum = LearningProgressCurriculum(
        bucketed_dehydration, num_tasks=8, seed=789
    )

    print(
        f"Created LearningProgressCurriculum with {len(lp_curriculum.tasks)} pre-generated tasks:"
    )
    for task in lp_curriculum.tasks[:3]:  # Show first 3
        config = task.get_env_config()
        heart_reward = config.game.agent.rewards.inventory.heart
        print(
            f"  {task.task_id}: {config.game.num_agents} agents, heart_reward={heart_reward}"
        )

    # Get tasks from curriculum (uses learning progress selection)
    print("Tasks selected by learning progress:")
    for i in range(3):
        task = lp_curriculum.get_task()
        config = task.get_env_config()
        print(f"  Selected: {task.task_id}")

    return (lp_curriculum,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Builder APIs

        Use fluent builder APIs for convenient TaskSet and Curriculum construction.
        """
    )
    return


@app.cell
def _(TaskSetBuilder, BucketedTaskSetBuilder, CurriculumBuilder, builder):
    # Example 1: TaskSetBuilder fluent API

    config1 = builder.arena(num_agents=2, combat=False)
    config2 = builder.arena(num_agents=8, combat=True)

    built_task_set = (
        TaskSetBuilder(seed=100)
        .add_config(config1, weight=3.0)
        .add_config(config2, weight=1.0)
        .add_override("game.max_steps", 600)
        .add_override("game.agent.rewards.inventory.heart", 1.5)
        .add_overrides(
            ["game.episode_truncates: false", "game.track_movement_metrics: true"]
        )
        .build()
    )

    print("Built TaskSet with fluent API:")
    sample_config = built_task_set.get_task()
    print(
        f"  Sample: {sample_config.game.num_agents} agents, {sample_config.game.max_steps} steps"
    )

    return built_task_set, config1, config2


@app.cell
def _(BucketedTaskSetBuilder, CurriculumBuilder, config1, built_task_set):
    # Example 2: BucketedTaskSetBuilder fluent API

    built_bucketed = (
        BucketedTaskSetBuilder(config1, seed=200)
        .add_bucket_values("game.num_agents", [2, 4, 8, 16])
        .add_bucket_range("game.max_steps", 400, 1000)
        .add_bucket_values("game.agent.rewards.inventory.heart", [0.5, 1.0, 2.0])
        .add_override("game.episode_truncates", False)
        .build()
    )

    print("Built BucketedTaskSet with fluent API:")
    sample_config = built_bucketed.get_task()
    heart_reward = sample_config.game.agent.rewards.inventory.heart
    print(
        f"  Sample: {sample_config.game.num_agents} agents, heart_reward={heart_reward}"
    )

    # Example 3: CurriculumBuilder fluent API

    built_curriculum = CurriculumBuilder(built_task_set).as_random(seed=300).build()

    print("Built Curriculum with fluent API:")
    task = built_curriculum.get_task()
    print(f"  Sample: {task.task_id}")

    return built_bucketed, built_curriculum


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Task Determinism Demo

        TaskSets with the same seed always generate the same tasks.
        This ensures reproducible experiments and debugging.
        """
    )
    return


@app.cell
def _(BucketedTaskSet, builder, buckets):
    # Demonstrate deterministic behavior

    base_config = builder.arena(num_agents=4, combat=False)

    # Create identical TaskSets with same seed
    task_set_a = BucketedTaskSet(base_config, buckets, seed=555)
    task_set_b = BucketedTaskSet(base_config, buckets, seed=555)

    print("Two identical TaskSets with seed=555:")
    for i in range(3):
        config_a = task_set_a.get_task()
        config_b = task_set_b.get_task()

        agents_match = config_a.game.num_agents == config_b.game.num_agents
        heart_match = (
            config_a.game.agent.rewards.inventory.heart
            == config_b.game.agent.rewards.inventory.heart
        )

        match_symbol = "✓" if agents_match and heart_match else "✗"
        print(
            f"  Task {i + 1}: A={config_a.game.num_agents} agents, B={config_b.game.num_agents} agents {match_symbol}"
        )

    # Show different seed produces different tasks
    task_set_c = BucketedTaskSet(base_config, buckets, seed=777)
    print("TaskSet with different seed=777:")
    for i in range(3):
        config_c = task_set_c.get_task()
        print(f"  Task {i + 1}: {config_c.game.num_agents} agents")

    return task_set_a, task_set_b, task_set_c


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Integration with Training

        Save curricula and use with Metta's training system.
        """
    )
    return


@app.cell
def _(random_curriculum, subprocess, os, datetime):
    # Example: Using curriculum with training

    # For integration with Metta's training system, we'd need to adapt the curriculum
    # to work with the existing curriculum interface. This could involve:
    # 1. Creating a wrapper that implements the old Curriculum interface
    # 2. Modifying the training system to use the new TaskSet/Curriculum API

    print("Integration with training system:")
    print("✓ TaskSets provide deterministic task generation")
    print("✓ Curricula manage task selection strategy")
    print("✓ Tasks wrap environment configurations")
    print("✓ Seeds ensure reproducible experiments")

    # Training command would look like:
    run_name = f"{os.environ.get('USER', 'user')}.curriculum-v2.{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    print("\\nTraining with new curriculum system:")
    print(f"  Run name: {run_name}")
    print("  Command: uv run ./tools/train.py \\")
    print(f"    run={run_name} \\")
    print("    curriculum=custom_v2_curriculum \\")
    print("    trainer.num_workers=4")

    return (run_name,)


@app.cell
def _(mo):
    mo.md(r"## Monitor Training Jobs")
    return


@app.cell
def _(monitor_training_statuses):
    # Monitor Training (placeholder - same as before)
    run_names = [
        "user.curriculum-v2.example.1",
        "user.curriculum-v2.example.2",
    ]

    print(f"Would monitor curriculum v2 runs: {run_names}")
    print("(Actual monitoring skipped - no real runs exist)")
    return (run_names,)


@app.cell
def _(mo):
    mo.md(r"## Fetch Metrics")
    return


@app.cell
def _(fetch_metrics, run_names):
    # Fetch metrics (placeholder - same as before)
    print(f"Would fetch metrics for curriculum v2 runs: {run_names}")
    print("(Actual fetching skipped - no real runs exist)")
    return


@app.cell
def _(mo):
    mo.md(r"## Analyze Metrics")
    return


@app.cell
def _(alt, pd):
    # Analysis setup (placeholder - same as before)
    print("Analysis tools ready for curriculum v2 metrics")
    print("✓ Altair for interactive charts")
    print("✓ Pandas for data processing")
    print("✓ Can analyze task selection patterns, learning progress, etc.")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## View Replays
    
        Display replay viewer for curriculum v2 training runs.
        """
    )
    return


@app.cell
def _(show_replay):
    # Show replay (placeholder - same as before)
    print("Replay viewer ready for curriculum v2 runs")
    print("Can visualize how different TaskSet configurations affect agent behavior")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary - Curriculum System v2
        
        The new curriculum architecture provides:
        
        ### **TaskSet** - Deterministic Task Generation
        - ✅ **WeightedTaskSet**: Sample from weighted lists of configs/TaskSets
        - ✅ **BucketedTaskSet**: Generate tasks by sampling parameter buckets
        - ✅ **Hierarchical composition**: TaskSets can contain other TaskSets  
        - ✅ **Deterministic**: Same seed always produces same tasks
        - ✅ **Flexible overrides**: Apply parameter changes to generated configs
        
        ### **Curriculum** - Task Selection Strategy
        - ✅ **RandomCurriculum**: Generate random seeds for variety
        - ✅ **LearningProgressCurriculum**: Pre-generate tasks, use learning progress
        - ✅ **Clean separation**: TaskSet handles generation, Curriculum handles selection
        
        ### **Task** - Simple Wrapper
        - ✅ **Deterministic**: Always wraps a specific EnvConfig
        - ✅ **Identifiable**: Has task_id for tracking
        - ✅ **Simple**: Just provides get_env_config() method
        
        ### **Key Benefits**
        - 🎯 **Reproducible**: Seeds ensure identical experiments
        - 🔧 **Composable**: Mix and match different generation strategies
        - 🎛️ **Flexible**: Easy parameter sweeps and overrides  
        - 🚀 **Performant**: Generate configs on-demand, no pre-computation
        - 📈 **Scalable**: Handle thousands of task variations efficiently
        
        This replaces the old curriculum system with better separation of concerns
        and more flexible task generation capabilities.
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
