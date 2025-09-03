"""Common fixtures for curriculum tests."""

import random

import numpy as np
import pytest

import metta.cogworks.curriculum as cc
import metta.mettagrid.config.envs as eb
from metta.cogworks.curriculum import (
    CurriculumConfig,
    SingleTaskGeneratorConfig,
    TaskGeneratorSetConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import (
    LearningProgressAlgorithm,
    LearningProgressConfig,
)
from metta.cogworks.curriculum.task_generator import ValueRange


@pytest.fixture(scope="function")
def random_seed():
    """Set random seeds for reproducible tests."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    yield seed


@pytest.fixture
def arena_env():
    """Create a basic arena environment configuration."""
    return eb.make_arena(num_agents=4)


@pytest.fixture
def navigation_env():
    """Create a basic navigation environment configuration."""
    return eb.make_navigation(num_agents=4)


@pytest.fixture
def single_task_generator_config(arena_env):
    """Create a single task generator configuration."""
    return SingleTaskGeneratorConfig(env=arena_env)


@pytest.fixture
def bucketed_task_generator_config(arena_env):
    """Create a bucketed task generator configuration with sample buckets."""
    config = cc.bucketed(arena_env)
    config.add_bucket("game.map_builder.width", [10, 20, 30])
    config.add_bucket("game.map_builder.height", [10, 20, 30])
    config.add_bucket("game.agent.rewards.inventory.ore_red", [0, ValueRange.vr(0, 1.0)])
    return config


@pytest.fixture
def task_generator_set_config(arena_env):
    """Create a task generator set configuration."""
    single1 = SingleTaskGeneratorConfig(env=arena_env)
    single2 = SingleTaskGeneratorConfig(env=arena_env.model_copy())
    return TaskGeneratorSetConfig(task_generators=[single1, single2], weights=[0.5, 0.5])


@pytest.fixture
def curriculum_config(single_task_generator_config):
    """Create a basic curriculum configuration."""
    return CurriculumConfig(
        task_generator=single_task_generator_config,
        max_task_id=1000,
        num_active_tasks=50,
        new_task_rate=0.05,
    )


@pytest.fixture
def learning_progress_config():
    """Create a learning progress algorithm configuration."""
    return LearningProgressConfig(
        ema_timescale=0.001,
        pool_size=10,
        sample_size=5,
        max_samples=10,
        exploration_bonus=0.1,
    )


@pytest.fixture
def learning_progress_algorithm(learning_progress_config):
    """Create a learning progress algorithm instance."""
    return LearningProgressAlgorithm(num_tasks=5, hypers=learning_progress_config)


@pytest.fixture
def mock_task_generator():
    """Create a mock task generator for testing."""

    class MockTaskGenerator:
        def get_task(self, task_id):
            return {"task_id": task_id}

    return MockTaskGenerator()


@pytest.fixture
def production_curriculum_config(arena_env):
    """Create a production-like curriculum configuration similar to arena.py."""
    arena_tasks = cc.bucketed(arena_env)

    # Add various bucket types like in production
    arena_tasks.add_bucket("game.map_builder.width", [10, 20, 30, 40])
    arena_tasks.add_bucket("game.map_builder.height", [10, 20, 30, 40])
    arena_tasks.add_bucket("game.agent.rewards.inventory.ore_red", [0, 0.1, 0.5, 1.0])
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 50, 100])

    # Add initial items like in production
    for obj in ["mine_red", "generator_red", "altar"]:
        arena_tasks.add_bucket(f"game.objects.{obj}.initial_resource_count", [0, 1])

    return arena_tasks.to_curriculum()


@pytest.fixture
def production_navigation_curriculum(navigation_env):
    """Create a production-like navigation curriculum similar to navigation.py."""
    nav_tasks = cc.bucketed(navigation_env)

    # Dense reward tasks
    dense_tasks = cc.bucketed(navigation_env)
    dense_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.5, 1.0])
    dense_tasks.add_bucket("game.agent.rewards.inventory.heart_max", [1, 2])

    # Maps like in production
    maps = ["terrain_maps_nohearts"]
    for size in ["large", "medium", "small"]:
        for terrain in ["balanced", "maze", "sparse", "dense"]:
            maps.append(f"varied_terrain/{terrain}_{size}")

    dense_tasks.add_bucket("game.map_builder.instance_map.dir", maps)
    dense_tasks.add_bucket("game.map_builder.instance_map.objects.altar", [ValueRange.vr(3, 50)])

    # Sparse tasks
    sparse_env = navigation_env.model_copy()
    sparse_env.game.map_builder = eb.make_navigation(num_agents=4).game.map_builder
    sparse_tasks = cc.bucketed(sparse_env)
    sparse_tasks.add_bucket("game.map_builder.width", [ValueRange.vr(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.height", [ValueRange.vr(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.objects.altar", [ValueRange.vr(1, 10)])

    nav_tasks = cc.merge([dense_tasks, sparse_tasks])
    return nav_tasks.to_curriculum()


@pytest.fixture
def performance_test_environment():
    """Create an environment configuration optimized for performance testing."""
    env = eb.make_arena(num_agents=2)
    env.game.map_builder.width = 20
    env.game.map_builder.height = 20
    # The rewards.inventory is a dict, so we need to set it properly
    if not hasattr(env.game.agent.rewards, "inventory"):
        env.game.agent.rewards.inventory = {}
    env.game.agent.rewards.inventory["ore_red"] = 0.5
    return env


@pytest.fixture
def curriculum_with_algorithm(arena_env, learning_progress_config):
    """Create a curriculum configuration with learning progress algorithm."""
    task_gen_config = SingleTaskGeneratorConfig(env=arena_env)
    return CurriculumConfig(
        task_generator=task_gen_config,
        num_active_tasks=4,
        algorithm_config=learning_progress_config,
    )


@pytest.fixture
def curriculum_without_algorithm(arena_env):
    """Create a curriculum configuration without algorithm (backward compatibility)."""
    task_gen_config = SingleTaskGeneratorConfig(env=arena_env)
    return CurriculumConfig(
        task_generator=task_gen_config,
        num_active_tasks=4,
        algorithm_config=None,
    )
