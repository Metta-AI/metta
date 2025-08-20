#!/usr/bin/env python3
"""Comprehensive test for learning progress curriculum integration."""

import numpy as np

from .curriculum import Curriculum, CurriculumConfig, CurriculumTask
from .learning_progress import (
    LearningProgressCurriculum,
    LearningProgressCurriculumConfig,
    LearningProgressCurriculumTask,
)
from .learning_progress_algorithm import LearningProgressAlgorithm, LearningProgressHypers


def test_learning_progress_inheritance():
    """Test that LearningProgressCurriculum properly inherits from Curriculum."""
    print("Testing learning progress curriculum inheritance...")

    # Test inheritance hierarchy
    assert issubclass(LearningProgressCurriculum, Curriculum), (
        "LearningProgressCurriculum should inherit from Curriculum"
    )
    assert issubclass(LearningProgressCurriculumTask, CurriculumTask), (
        "LearningProgressCurriculumTask should inherit from CurriculumTask"
    )
    assert issubclass(LearningProgressCurriculumConfig, CurriculumConfig), (
        "LearningProgressCurriculumConfig should inherit from CurriculumConfig"
    )
    print("✅ Inheritance hierarchy correct")

    print("🎉 Learning progress inheritance test passed!")


def test_method_overrides():
    """Test that learning progress classes properly override base class methods."""
    print("Testing method overrides...")

    # Test that LearningProgressCurriculum overrides key methods
    curriculum_methods = ["_create_task", "_choose_task", "_evict_task", "stats"]
    for method in curriculum_methods:
        assert hasattr(LearningProgressCurriculum, method), f"LearningProgressCurriculum should have {method} method"
        # Check that it's not the same as the base class method
        base_method = getattr(Curriculum, method)
        derived_method = getattr(LearningProgressCurriculum, method)
        assert derived_method != base_method, f"LearningProgressCurriculum should override {method}"
    print("✅ Curriculum method overrides correct")

    # Test that LearningProgressCurriculumTask overrides key methods
    task_methods = ["complete"]
    for method in task_methods:
        assert hasattr(LearningProgressCurriculumTask, method), (
            f"LearningProgressCurriculumTask should have {method} method"
        )
        # Check that it's not the same as the base class method
        base_method = getattr(CurriculumTask, method)
        derived_method = getattr(LearningProgressCurriculumTask, method)
        assert derived_method != base_method, f"LearningProgressCurriculumTask should override {method}"
    print("✅ Task method overrides correct")

    # Test that LearningProgressCurriculumTask has learning progress specific methods
    lp_methods = ["get_learning_progress", "get_success_rate", "_update_learning_progress"]
    for method in lp_methods:
        assert hasattr(LearningProgressCurriculumTask, method), (
            f"LearningProgressCurriculumTask should have {method} method"
        )
    print("✅ Learning progress specific methods present")

    print("🎉 Method overrides test passed!")


def test_config_inheritance():
    """Test that LearningProgressCurriculumConfig properly inherits and extends CurriculumConfig."""
    print("Testing configuration inheritance...")

    # Test that it has all base class fields
    base_fields = {"task_generator_config", "max_task_id", "num_active_tasks", "new_task_rate"}
    for field in base_fields:
        assert field in LearningProgressCurriculumConfig.model_fields, f"Should have base field {field}"
    print("✅ Has all base configuration fields")

    # Test that it has learning progress specific fields
    lp_fields = {"ema_timescale", "progress_smoothing", "rand_task_rate", "memory"}
    for field in lp_fields:
        assert field in LearningProgressCurriculumConfig.model_fields, f"Should have learning progress field {field}"
    print("✅ Has all learning progress configuration fields")

    # Test that it overrides the make method
    base_make = CurriculumConfig.make
    derived_make = LearningProgressCurriculumConfig.make
    assert derived_make != base_make, "Should override make method"
    print("✅ Configuration make method overridden")

    print("🎉 Configuration inheritance test passed!")


def test_learning_progress_algorithm():
    """Test learning progress algorithm functionality."""
    print("Testing learning progress algorithm...")

    # Create hyperparameters
    hypers = LearningProgressHypers(
        ema_timescale=0.01,
        progress_smoothing=0.05,
        num_active_tasks=4,
        rand_task_rate=0.25,
        sample_threshold=5,
        memory=10,
    )

    # Create algorithm
    algorithm = LearningProgressAlgorithm(num_tasks=6, hypers=hypers)

    # Test basic functionality
    assert algorithm.num_tasks == 6
    assert len(algorithm.weights) == 6
    assert len(algorithm.probabilities) == 6

    # Test initial state
    assert np.allclose(algorithm.weights, 1.0)
    assert np.allclose(algorithm.probabilities, 1.0 / 6)

    # Test weight updates
    algorithm.update(0, 0.8)
    algorithm.update(1, 0.6)
    algorithm.update(2, 0.9)

    # Verify weights changed
    assert not np.allclose(algorithm.weights, 1.0)

    # Test sampling
    sample_idx = algorithm.sample_idx()
    assert 0 <= sample_idx < 6

    # Test statistics
    stats = algorithm.stats()
    assert isinstance(stats, dict)

    print("✅ Learning progress algorithm test passed!")


def test_curriculum_algorithm_classes():
    """Test that the curriculum algorithm classes are properly defined."""
    print("Testing curriculum algorithm classes...")

    # Import the classes (now integrated into curriculum.py)
    from .curriculum import (
        CurriculumAlgorithm,
        CurriculumAlgorithmHypers,
        DiscreteRandomCurriculum,
        DiscreteRandomHypers,
    )

    # Test that classes exist and have expected methods
    assert hasattr(CurriculumAlgorithm, "update")
    assert hasattr(CurriculumAlgorithm, "sample_idx")
    assert hasattr(CurriculumAlgorithm, "stats")
    assert hasattr(CurriculumAlgorithmHypers, "create")
    assert hasattr(DiscreteRandomCurriculum, "_update_weights")
    assert hasattr(DiscreteRandomHypers, "algorithm_type")

    print("✅ Curriculum algorithm classes are properly defined!")


if __name__ == "__main__":
    test_learning_progress_inheritance()
    test_method_overrides()
    test_config_inheritance()
    test_learning_progress_algorithm()
    test_curriculum_algorithm_classes()
    print("\n🎉 All comprehensive tests passed!")
