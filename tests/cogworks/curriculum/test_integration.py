#!/usr/bin/env python3
"""Test the integration of curriculum algorithms into the main curriculum system."""

import numpy as np

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

    # Test task creation and selection still works
    task = curriculum.get_task()
    assert task is not None, "Should get a task"
    assert task.get_env_cfg() is not None, "Task should have environment config"
    print("✅ Task creation and selection works without algorithm")

    # Test statistics still work
    stats = curriculum.stats()
    assert "num_active_tasks" in stats, "Should include basic statistics"
    print("✅ Basic statistics work without algorithm")

    print("🎉 Curriculum without algorithm test passed!")


def test_curriculum_task_performance_update():
    """Test that curriculum can update task performance with algorithms."""
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

    # Complete the task
    task.complete(0.8)

    # Update curriculum with task performance
    curriculum.update_task_performance(task_id, 0.8)

    # Verify the task was completed
    assert task._num_completions == 1, "Task should be marked as completed"
    assert task._mean_score == 0.8, "Task should have correct mean score"
    print("✅ Task completion and performance update works")

    print("🎉 Curriculum task performance update test passed!")


def test_curriculum_with_learning_progress():
    """Test that curriculum works with learning progress algorithm."""
    print("Testing curriculum with learning progress algorithm...")

    from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressHypers

    # Create a proper arena environment configuration
    arena_env = make_arena(num_agents=4)

    # Create task generator configuration
    task_gen_config = SingleTaskGeneratorConfig(env=arena_env)

    # Create learning progress algorithm hyperparameters
    lp_hypers = LearningProgressHypers(
        ema_timescale=0.01,
        progress_smoothing=0.05,
        num_active_tasks=4,
        rand_task_rate=0.25,
        sample_threshold=5,
        memory=10,
    )

    # Create curriculum config with learning progress algorithm
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

    # Get a task and complete it
    task = curriculum.get_task()
    task_id = task._task_id
    task.complete(0.7)

    # Update curriculum with task performance
    curriculum.update_task_performance(task_id, 0.7)

    # Get statistics
    stats = curriculum.stats()
    assert "num_active_tasks" in stats, "Should include basic statistics"
    print("✅ Learning progress algorithm statistics work")

    print("🎉 Curriculum with learning progress algorithm test passed!")


def test_algorithm_framework():
    """Test the curriculum algorithm framework."""
    print("Testing curriculum algorithm framework...")

    # Test DiscreteRandomHypers
    hypers = DiscreteRandomHypers()
    assert hypers.algorithm_type() == "discrete_random", "Should return correct algorithm type"
    print("✅ DiscreteRandomHypers works")

    # Test DiscreteRandomCurriculum
    algorithm = DiscreteRandomCurriculum(num_tasks=5, hypers=hypers)
    assert algorithm.num_tasks == 5, "Should have correct number of tasks"
    assert len(algorithm.weights) == 5, "Should have correct number of weights"
    assert len(algorithm.probabilities) == 5, "Should have correct number of probabilities"
    print("✅ DiscreteRandomCurriculum initialization works")

    # Test weight updates (should do nothing for discrete random)
    algorithm.update(0, 0.8)
    algorithm.update(1, 0.6)
    assert np.allclose(algorithm.weights, 1.0), "Weights should remain unchanged"
    print("✅ DiscreteRandomCurriculum weight updates work")

    # Test sampling
    sample_idx = algorithm.sample_idx()
    assert 0 <= sample_idx < 5, "Should sample valid index"
    print("✅ DiscreteRandomCurriculum sampling works")

    # Test statistics
    stats = algorithm.stats()
    assert isinstance(stats, dict), "Should return statistics dictionary"
    print("✅ DiscreteRandomCurriculum statistics work")

    print("🎉 Curriculum algorithm framework test passed!")


if __name__ == "__main__":
    test_curriculum_with_algorithm()
    test_curriculum_without_algorithm()
    test_curriculum_task_performance_update()
    test_curriculum_with_learning_progress()
    test_algorithm_framework()
    print("\n🎉 All integration tests passed!")
