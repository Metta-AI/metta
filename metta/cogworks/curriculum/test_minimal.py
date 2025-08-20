#!/usr/bin/env python3
"""Minimal test for learning progress curriculum."""

from learning_progress_minimal import (
    LearningProgressCurriculum,
    LearningProgressCurriculumConfig,
    LearningProgressCurriculumTask,
)


def test_minimal():
    """Test minimal learning progress implementation."""
    print("Testing minimal learning progress curriculum...")

    # Create config
    config = LearningProgressCurriculumConfig(
        ema_timescale=0.01,
        progress_smoothing=0.05,
        rand_task_rate=0.25,
        memory=10,
    )

    # Create curriculum
    curriculum = LearningProgressCurriculum(config, seed=42)

    # Test task creation
    task = curriculum.get_task()
    assert isinstance(task, LearningProgressCurriculumTask)
    print("✅ Task creation works")

    # Test task completion
    task.complete(0.5)
    lp = task.get_learning_progress()
    assert isinstance(lp, float)
    assert lp >= 0.0
    print("✅ Task completion works")

    # Test multiple completions
    for score in [0.6, 0.7, 0.8]:
        task.complete(score)

    lp = task.get_learning_progress()
    print(f"✅ Learning progress: {lp:.3f}")

    # Test statistics
    stats = curriculum.stats()
    assert isinstance(stats, dict)
    print(f"✅ Statistics: {stats}")

    print("🎉 Minimal learning progress test passed!")


if __name__ == "__main__":
    test_minimal()
