#!/usr/bin/env python3
"""Test the integration of curriculum algorithms into the main curriculum system."""

import numpy as np

from metta.mettagrid.mettagrid_config import EnvConfig

from .curriculum import (
    CurriculumConfig,
    DiscreteRandomCurriculum,
    DiscreteRandomHypers,
)
from .task_generator import SingleTaskGeneratorConfig


def test_curriculum_with_algorithm():
    """Test that curriculum works with algorithm integration."""
    print("Testing curriculum with algorithm integration...")

    # Create a simple mock environment configuration
    mock_env_config = EnvConfig.model_validate(
        {
            "game": {
                "level_map": {"grid": np.array([]), "labels": []},
                "actions": {"attack": {"consumed_resources": {"laser": 1}}},
                "agent": {"rewards": {}},
            }
        }
    )

    # Create task generator configuration
    task_gen_config = SingleTaskGeneratorConfig(env_config=mock_env_config)

    # Create curriculum config with algorithm
    curriculum_config = CurriculumConfig(
        task_generator_config=task_gen_config,
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
    print("✅ Task creation and selection works")

    # Test algorithm statistics (DiscreteRandomCurriculum returns empty stats)
    stats = curriculum.stats()
    assert "num_active_tasks" in stats, "Should include basic statistics"
    print("✅ Algorithm statistics integration works")

    print("🎉 Curriculum with algorithm integration test passed!")


def test_curriculum_without_algorithm():
    """Test that curriculum works without algorithm (backward compatibility)."""
    print("Testing curriculum without algorithm (backward compatibility)...")

    # Create a simple mock environment configuration
    mock_env_config = EnvConfig.model_validate(
        {
            "game": {
                "level_map": {"grid": np.array([]), "labels": []},
                "actions": {"attack": {"consumed_resources": {"laser": 1}}},
                "agent": {"rewards": {}},
            }
        }
    )

    # Create task generator configuration
    task_gen_config = SingleTaskGeneratorConfig(env_config=mock_env_config)

    # Create curriculum config without algorithm
    curriculum_config = CurriculumConfig(
        task_generator_config=task_gen_config,
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
    print("✅ Task creation and selection works without algorithm")

    # Test statistics still work
    stats = curriculum.stats()
    assert "num_active_tasks" in stats, "Should include basic statistics"
    print("✅ Basic statistics work without algorithm")

    print("🎉 Curriculum without algorithm test passed!")


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
    test_algorithm_framework()
    print("\n🎉 All integration tests passed!")
