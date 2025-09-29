"""Common fixtures for curriculum tests."""

import pytest

import metta.cogworks.curriculum as cc
from metta.cogworks.curriculum import CurriculumConfig, SingleTaskGenerator, Span
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from mettagrid.builder.envs import make_arena, make_navigation


@pytest.fixture(scope="function")
def arena_env():
    """Create a basic arena environment configuration."""
    return make_arena(num_agents=4)


@pytest.fixture(scope="function")
def navigation_env():
    """Create a basic navigation environment configuration."""
    return make_navigation(num_agents=4)


@pytest.fixture(scope="function")
def curriculum_config(arena_env):
    """Create a basic curriculum configuration."""
    return CurriculumConfig(
        task_generator=SingleTaskGenerator.Config(env=arena_env),
        max_task_id=1000,
        num_active_tasks=50,
    )


@pytest.fixture(scope="function")
def learning_progress_algorithm():
    """Create a learning progress algorithm configuration."""
    return LearningProgressConfig(
        ema_timescale=0.001,
        exploration_bonus=0.1,
        max_memory_tasks=1000,
        max_slice_axes=3,
        enable_detailed_slice_logging=False,
        use_shared_memory=False,  # Disable shared memory for tests
    )


@pytest.fixture(scope="function")
def production_curriculum_config(arena_env):
    """Create a production-like curriculum configuration similar to arena.py."""
    arena_tasks = cc.bucketed(arena_env)

    # Add reward buckets for all resources
    for item in arena_env.game.resource_names:
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}", [0, Span(0, 1.0)])
        arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    # Add map size buckets
    arena_tasks.add_bucket("game.map_builder.width", [10, 20, 30, 40, 50])
    arena_tasks.add_bucket("game.map_builder.height", [10, 20, 30, 40, 50])

    # Add attack cost bucket
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    return arena_tasks.to_curriculum()


@pytest.fixture(scope="function")
def production_navigation_curriculum(navigation_env):
    """Create a production-like navigation curriculum similar to navigation.py."""
    nav_tasks = cc.bucketed(navigation_env)

    dense_tasks = cc.bucketed(navigation_env)
    dense_tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.5, 1.0])
    dense_tasks.add_bucket("game.agent.rewards.inventory_max.heart", [1, 2])

    maps = ["terrain_maps_nohearts"]
    for size in ["large", "medium", "small"]:
        for terrain in ["balanced", "maze", "sparse", "dense", "cylinder-world"]:
            maps.append(f"varied_terrain/{terrain}_{size}")

    dense_tasks.add_bucket("game.map_builder.instance_map.dir", maps)
    dense_tasks.add_bucket("game.map_builder.instance_map.objects.altar", [Span(3, 50)])

    # Sparse tasks
    sparse_env = navigation_env.model_copy()
    sparse_env.game.map_builder = make_navigation(num_agents=4).game.map_builder
    sparse_tasks = cc.bucketed(sparse_env)
    sparse_tasks.add_bucket("game.map_builder.width", [Span(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.height", [Span(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.objects.altar", [Span(1, 10)])

    nav_tasks = cc.merge([dense_tasks, sparse_tasks])
    return nav_tasks.to_curriculum()


@pytest.fixture(scope="function")
def single_task_generator_config(arena_env):
    """Create a single task generator configuration."""
    return SingleTaskGenerator.Config(env=arena_env)


@pytest.fixture(scope="function")
def bucketed_task_generator_config(arena_env):
    """Create a bucketed task generator configuration."""
    return cc.bucketed(arena_env)


@pytest.fixture(scope="function")
def task_generator_set_config(arena_env):
    """Create a task generator set configuration."""
    return cc.multi_task(arena_env)


@pytest.fixture(scope="function")
def curriculum_with_algorithm(arena_env, learning_progress_algorithm):
    """Create a curriculum with learning progress algorithm."""
    return CurriculumConfig(
        task_generator=SingleTaskGenerator.Config(env=arena_env),
        algorithm_config=learning_progress_algorithm,
    )


@pytest.fixture(scope="function")
def curriculum_without_algorithm(arena_env):
    """Create a curriculum without algorithm."""
    return CurriculumConfig(
        task_generator=SingleTaskGenerator.Config(env=arena_env),
    )


@pytest.fixture(scope="function")
def mock_task_generator():
    """Create a mock task generator for testing."""

    class MockTaskGenerator:
        def get_task(self, task_id):
            return {"task_id": task_id, "env_config": {"mock": True}}

    return MockTaskGenerator()


@pytest.fixture(scope="function")
def random_seed():
    """Provide a random seed for deterministic testing."""
    return 42


@pytest.fixture(scope="function")
def performance_test_environment():
    """Create a minimal environment for performance testing."""
    return make_arena(num_agents=4)
