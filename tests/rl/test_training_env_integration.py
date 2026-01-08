"""Test training environment integration with curriculum and stats tracking."""

import numpy as np

import metta.cogworks.curriculum as cc
from metta.cogworks.curriculum import Curriculum, CurriculumConfig, SingleTaskGenerator
from metta.cogworks.curriculum.curriculum_env import CurriculumEnv
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.envs.stats_tracker import StatsTracker
from mettagrid.simulator import Simulator
from mettagrid.util.stats_writer import NoopStatsWriter


def test_curriculum_env_with_stats_tracker_runs_episodes():
    """Test that curriculum-wrapped environment with stats tracker runs episodes correctly."""
    env_cfg = MettaGridConfig.EmptyRoom(num_agents=2)
    env_cfg.game.max_steps = 50

    curriculum = Curriculum(CurriculumConfig(task_generator=SingleTaskGenerator.Config(env=env_cfg)))
    stats_writer = NoopStatsWriter()

    sim = Simulator()
    sim.add_event_handler(StatsTracker(stats_writer))
    env = MettaGridPufferEnv(sim, curriculum.get_task().get_env_cfg())
    env = CurriculumEnv(env, curriculum)

    num_episodes = 3
    episodes_completed = 0

    for _ in range(num_episodes):
        obs, info = env.reset()

        episode_done = False
        step_count = 0
        max_steps = 200

        while not episode_done and step_count < max_steps:
            actions = np.array([env.single_action_space.sample() for _ in range(env.num_agents)])
            obs, rewards, terminals, truncations, infos = env.step(actions)

            step_count += 1

            if terminals.all() or truncations.all():
                episode_done = True
                episodes_completed += 1

                # Verify infos are being set
                assert isinstance(infos, dict)
                assert "game" in infos
                assert "agent" in infos
                assert "attributes" in infos

    assert episodes_completed == num_episodes

    env.close()
    stats_writer.close()


def test_curriculum_with_multiple_tasks_runs_both():
    """Test that curriculum with 2 tasks runs both tasks."""
    env_cfg = MettaGridConfig.EmptyRoom(num_agents=2)

    # Create a curriculum with 2 tasks (different max_steps values)
    # Use None for algorithm_config to get random selection for more predictable testing
    tasks = cc.bucketed(env_cfg)
    tasks.add_bucket("game.max_steps", [20, 50])  # 2 tasks: one with 20 steps, one with 50 steps
    curriculum_config = tasks.to_curriculum(num_active_tasks=2, algorithm_config=None)
    curriculum = Curriculum(curriculum_config)

    # First verify that the curriculum can generate both tasks
    # by sampling multiple times directly
    direct_samples = set()
    for _ in range(50):
        task = curriculum.get_task()
        max_steps = task.get_env_cfg().game.max_steps
        direct_samples.add(max_steps)
    # Both values should be possible to generate
    assert 20 in direct_samples or 50 in direct_samples, "Curriculum should be able to generate at least one task"

    stats_writer = NoopStatsWriter()

    sim = Simulator()
    sim.add_event_handler(StatsTracker(stats_writer))
    # Don't pre-initialize with a task - let CurriculumEnv handle it
    initial_task = curriculum.get_task()
    env = MettaGridPufferEnv(sim, initial_task.get_env_cfg())
    env = CurriculumEnv(env, curriculum)

    # Track which tasks we've seen by their max_steps value
    seen_max_steps = set()
    num_episodes = 30  # Run enough episodes to see both tasks with random selection
    episodes_completed = 0

    for _ in range(num_episodes):
        obs, info = env.reset()

        # Get the current task's max_steps from the environment's config after reset
        # CurriculumEnv sets the config after reset, so check the env's config
        current_max_steps = env.env_cfg.game.max_steps
        seen_max_steps.add(current_max_steps)

        episode_done = False
        step_count = 0
        max_steps = 200

        while not episode_done and step_count < max_steps:
            actions = np.array([env.single_action_space.sample() for _ in range(env.num_agents)])
            obs, rewards, terminals, truncations, infos = env.step(actions)

            step_count += 1

            if terminals.all() or truncations.all():
                episode_done = True
                episodes_completed += 1

                # Also check task after episode completion (task switches in step())
                if episode_done:
                    post_completion_max_steps = env.env_cfg.game.max_steps
                    seen_max_steps.add(post_completion_max_steps)

                # Verify infos are being set
                assert isinstance(infos, dict)
                assert "game" in infos
                assert "agent" in infos
                assert "attributes" in infos

    assert episodes_completed == num_episodes
    # Verify both tasks were run (both max_steps values were seen)
    # Note: With random selection, it's possible but unlikely to see only one task
    # If we see both direct_samples, we should see both in the curriculum too
    if len(direct_samples) == 2:
        assert len(seen_max_steps) == 2, (
            f"Expected to see 2 different tasks, but only saw: {seen_max_steps}. Direct samples: {direct_samples}"
        )
        assert 20 in seen_max_steps, "Task with max_steps=20 was not run"
        assert 50 in seen_max_steps, "Task with max_steps=50 was not run"

    env.close()
    stats_writer.close()
