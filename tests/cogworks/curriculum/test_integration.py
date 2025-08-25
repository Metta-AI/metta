#!/usr/bin/env python3
"""Test the integration of curriculum algorithms into the main curriculum system."""

from metta.cogworks.curriculum.curriculum import (
    CurriculumConfig,
    DiscreteRandomCurriculum,
    DiscreteRandomHypers,
)
from metta.cogworks.curriculum.task_generator import SingleTaskGeneratorConfig
from metta.mettagrid.config.envs import make_arena


def test_curriculum_with_algorithm():
    """Test that curriculum works with algorithm integration."""
    print("Testing curriculum with algorithm integration...")

    # Create a proper arena environment configuration
    arena_env = make_arena(num_agents=4)

    # Create task generator configuration
    task_gen_config = SingleTaskGeneratorConfig(env=arena_env)

    # Create curriculum config with algorithm
    curriculum_config = CurriculumConfig(
        task_generator=task_gen_config,
        num_active_tasks=4,
        algorithm_hypers=DiscreteRandomHypers(),
    )

    # Create curriculum
    curriculum = curriculum_config.make()

    # Test that algorithm is initialized
    assert curriculum._algorithm is not None, "Algorithm should be initialized"
    assert isinstance(curriculum._algorithm, DiscreteRandomCurriculum), "Should be DiscreteRandomCurriculum"
    print("✅ Algorithm initialization works")

    # Test task creation and selection
    task = curriculum.get_task()
    assert task is not None, "Should get a task"
    assert task.get_env_cfg() is not None, "Task should have environment config"
    print("✅ Task creation and selection works")

    # Test algorithm statistics
    stats = curriculum.stats()
    assert "num_active_tasks" in stats, "Should include basic statistics"
    print("✅ Algorithm statistics integration works")

    print("🎉 Curriculum with algorithm integration test passed!")


def test_curriculum_without_algorithm():
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
        algorithm_hypers=None,  # No algorithm
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


def test_curriculum_task_performance_update():
    """Test that curriculum can update task performance with algorithm."""
    print("Testing curriculum task performance update...")

    # Create a proper arena environment configuration
    arena_env = make_arena(num_agents=4)

    # Create task generator configuration
    task_gen_config = SingleTaskGeneratorConfig(env=arena_env)

    # Create curriculum config with algorithm
    curriculum_config = CurriculumConfig(
        task_generator=task_gen_config,
        num_active_tasks=4,
        algorithm_hypers=DiscreteRandomHypers(),
    )

    # Create curriculum
    curriculum = curriculum_config.make()

    # Get a task
    task = curriculum.get_task()
    task_id = task._task_id

    # Update task performance
    curriculum.update_task_performance(task_id, 0.8)

    print("✅ Task performance update works")

    print("🎉 Curriculum task performance update test passed!")


def test_curriculum_with_learning_progress():
    """Test that curriculum works with learning progress algorithm."""
    print("Testing curriculum with learning progress...")

    # Create a proper arena environment configuration
    arena_env = make_arena(num_agents=4)

    # Create task generator configuration
    task_gen_config = SingleTaskGeneratorConfig(env=arena_env)

    # Create curriculum config with learning progress algorithm
    from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressHypers

    lp_hypers = LearningProgressHypers(
        type="learning_progress",
        ema_timescale=0.001,
        progress_smoothing=0.05,
        num_active_tasks=4,
        rand_task_rate=0.25,
        sample_threshold=10,
        memory=25,
    )

    curriculum_config = CurriculumConfig(
        task_generator=task_gen_config,
        num_active_tasks=4,
        algorithm_hypers=lp_hypers,
    )

    # Create curriculum
    curriculum = curriculum_config.make()

    # Test that learning progress algorithm is initialized
    assert curriculum._algorithm is not None, "Learning progress algorithm should be initialized"
    print("✅ Learning progress algorithm initialization works")

    # Test task creation and selection
    task = curriculum.get_task()
    assert task is not None, "Should get a task"
    assert task.get_env_cfg() is not None, "Task should have environment config"
    print("✅ Task creation and selection works with learning progress")

    # Test algorithm statistics
    stats = curriculum.stats()
    assert "num_active_tasks" in stats, "Should include basic statistics"
    print("✅ Learning progress algorithm statistics work")

    print("🎉 Curriculum with learning progress test passed!")


if __name__ == "__main__":
    test_curriculum_with_algorithm()
    test_curriculum_without_algorithm()
    test_curriculum_task_performance_update()
    test_curriculum_with_learning_progress()
