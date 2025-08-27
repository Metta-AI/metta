#!/usr/bin/env python3
"""Test the integration of curriculum algorithms into the main curriculum system."""

from metta.cogworks.curriculum.curriculum import (
    CurriculumConfig,
    DiscreteRandomConfig,
    DiscreteRandomCurriculum,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressAlgorithm, LearningProgressConfig
from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig
from metta.mettagrid.config.envs import make_arena


def test_curriculum_integration_with_discrete_random_algorithm():
    """Test that curriculum works with discrete random algorithm."""
    print("Testing curriculum with discrete random algorithm...")

    # Create a proper arena environment configuration
    arena_env = make_arena(num_agents=4)

    # Create task generator configuration
    task_gen_config = SingleTaskGeneratorConfig(env=arena_env)

    # Create curriculum config with discrete random algorithm
    curriculum_config = CurriculumConfig(
        task_generator=task_gen_config,
        num_active_tasks=4,
        algorithm_config=DiscreteRandomConfig(),
    )

    # Create curriculum
    curriculum = curriculum_config.make()

    # Test that algorithm is initialized
    assert curriculum._algorithm is not None, "Algorithm should be initialized"
    assert isinstance(curriculum._algorithm, DiscreteRandomCurriculum), "Should be DiscreteRandomCurriculum"
    print("✅ Discrete random algorithm initialization works")

    # Test task creation and selection
    task = curriculum.get_task()
    assert task is not None, "Should get a task"
    assert task.get_env_cfg() is not None, "Task should have environment config"
    print("✅ Task creation and selection works")

    # Test algorithm statistics
    stats = curriculum.stats()
    assert "num_active_tasks" in stats, "Should include basic statistics"
    print("✅ Discrete random algorithm statistics integration works")

    print("🎉 Curriculum with discrete random algorithm test passed!")


def test_curriculum_integration_without_algorithm():
    """Test that curriculum works without algorithm (backward compatibility)."""
    print("Testing curriculum without algorithm (backward compatibility)...")

    # Create a proper arena environment configuration
    arena_env = make_arena(num_agents=4)

    # Create task generator configuration
    task_gen_config = SingleTaskGeneratorConfig(env=arena_env)

    # Create curriculum config without algorithm
    curriculum_config = CurriculumConfig(
        task_generator=task_gen_config,
        num_active_tasks=4,
        algorithm_config=None,  # No algorithm
    )

    # Create curriculum
    curriculum = curriculum_config.make()

    # Test that algorithm is not initialized
    assert curriculum._algorithm is None, "Algorithm should not be initialized"
    print("✅ No algorithm initialization works")

    # Test task creation and selection
    task = curriculum.get_task()
    assert task is not None, "Should get a task"
    assert task.get_env_cfg() is not None, "Task should have environment config"
    print("✅ Task creation and selection works without algorithm")

    # Test statistics
    stats = curriculum.stats()
    assert "num_active_tasks" in stats, "Should include basic statistics"
    print("✅ Basic statistics work without algorithm")

    print("🎉 Curriculum without algorithm test passed!")


def test_curriculum_integration_with_learning_progress_algorithm():
    """Test full curriculum integration with learning progress algorithm."""
    print("Testing curriculum with learning progress integration...")

    # Create a proper arena environment configuration
    arena_env = make_arena(num_agents=4)

    # Create task generator configuration
    task_gen_config = SingleTaskGeneratorConfig(env=arena_env)

    # Create curriculum config with learning progress algorithm
    lp_config = LearningProgressConfig(
        ema_timescale=0.1,
        pool_size=10,
        sample_size=5,
        max_samples=10,
    )

    curriculum_config = CurriculumConfig(
        task_generator=task_gen_config,
        num_active_tasks=4,
        algorithm_config=lp_config,
    )

    # Create curriculum
    curriculum = curriculum_config.make()

    # Test that learning progress algorithm is initialized
    assert curriculum._algorithm is not None, "Learning progress algorithm should be initialized"
    assert isinstance(curriculum._algorithm, LearningProgressAlgorithm), "Should be LearningProgressAlgorithm"
    print("✅ Learning progress algorithm initialization works")

    # Test task creation and selection
    task = curriculum.get_task()
    assert task is not None, "Should get a task"
    assert task.get_env_cfg() is not None, "Task should have environment config"
    print("✅ Task creation and selection works with learning progress")

    # Complete the task and update performance
    task.complete(0.8)
    curriculum.update_task_performance(task._task_id, 0.8)

    # Test algorithm statistics
    stats = curriculum.stats()
    assert "num_active_tasks" in stats, "Should include basic statistics"
    assert "algorithm/" in str(stats), "Stats should include algorithm information"
    print("✅ Learning progress algorithm statistics and performance updates work")

    print("🎉 Curriculum with learning progress integration test passed!")


if __name__ == "__main__":
    test_curriculum_integration_with_discrete_random_algorithm()
    test_curriculum_integration_without_algorithm()
    test_curriculum_integration_with_learning_progress_algorithm()
