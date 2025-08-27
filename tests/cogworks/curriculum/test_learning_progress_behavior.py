"""
Behavioral tests for learning progress curriculum algorithm.

These tests verify that the learning progress algorithm behaves correctly
in terms of LP score calculation and task sampling.
"""

import random

import numpy as np

from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressAlgorithm, LearningProgressHypers


class MockTaskGenerator:
    """Mock task generator for testing."""

    def get_task(self, task_id):
        return {"task_id": task_id}


def setup_learning_progress_algorithm_with_tasks(num_tasks: int = 2, pool_size: int = 10):
    """Helper function to set up a learning progress algorithm with tasks."""
    hypers = LearningProgressHypers(
        ema_timescale=0.001,
        pool_size=pool_size,
        sample_size=5,
        max_samples=10,
    )
    algorithm = LearningProgressAlgorithm(num_tasks=num_tasks, hypers=hypers)

    # Add tasks to the pool
    rng = random.Random(42)
    task_generator = MockTaskGenerator()

    tasks = []
    for _i in range(num_tasks):
        task = algorithm.get_task_from_pool(task_generator, rng)
        tasks.append(task)

    return algorithm, tasks


def test_learning_progress_favors_fast_learning():
    """Test that fast learning (higher slope) has higher learning progress scores than slow learning (lower slope)."""
    print("Testing that fast learning has higher LP scores than slow learning...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Set up algorithm with tasks
    algorithm, tasks = setup_learning_progress_algorithm_with_tasks(num_tasks=2)
    task1_id = tasks[0]._task_id
    task2_id = tasks[1]._task_id

    # Simulate fast learning for task 1 (higher slope)
    for i in range(20):
        algorithm.update_task_performance(task1_id, 0.1 + i * 0.04)  # Fast improvement: slope = 0.04

    # Simulate slow learning for task 2 (lower slope)
    for i in range(20):
        algorithm.update_task_performance(task2_id, 0.1 + i * 0.01)  # Slow improvement: slope = 0.01

    # Get LP scores for both tasks
    lp_score_1 = algorithm._get_task_lp_score(task1_id)
    lp_score_2 = algorithm._get_task_lp_score(task2_id)

    print(f"Task 1 (fast learning, slope=0.04) LP score: {lp_score_1}")
    print(f"Task 2 (slow learning, slope=0.01) LP score: {lp_score_2}")
    print(f"Difference: {lp_score_1 - lp_score_2}")

    # Test that fast learning has higher LP score
    assert lp_score_1 > lp_score_2, f"Fast learning should have higher LP score. Fast: {lp_score_1}, Slow: {lp_score_2}"
    print("✅ Fast learning has higher LP score than slow learning")


def test_learning_progress_favors_changing_performance():
    """Test that changing performance has higher learning progress scores than consistent performance."""
    print("Testing that changing performance has higher LP scores...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Set up algorithm with tasks
    algorithm, tasks = setup_learning_progress_algorithm_with_tasks(num_tasks=2)
    task1_id = tasks[0]._task_id
    task2_id = tasks[1]._task_id

    # Simulate truly consistent performance for task 1 (should have LP score close to zero)
    for _i in range(20):
        algorithm.update_task_performance(task1_id, 0.5)  # Exactly the same value every time

    # Simulate dramatic changing performance for task 2 (should have high LP score)
    for _i in range(20):
        if _i % 3 == 0:
            algorithm.update_task_performance(task2_id, 0.9)  # High performance
        elif _i % 3 == 1:
            algorithm.update_task_performance(task2_id, 0.1)  # Low performance
        else:
            algorithm.update_task_performance(task2_id, 0.5)  # Medium performance

    # Get LP scores for both tasks
    lp_score_1 = algorithm._get_task_lp_score(task1_id)
    lp_score_2 = algorithm._get_task_lp_score(task2_id)

    print(f"Task 1 (consistent) LP score: {lp_score_1}")
    print(f"Task 2 (changing) LP score: {lp_score_2}")
    print(f"Difference: {lp_score_2 - lp_score_1}")

    # Test that consistent performance has LP score close to zero
    assert lp_score_1 < 0.1, f"Consistent performance should have LP score close to zero. Got: {lp_score_1}"

    # Test that changing performance has higher LP score
    assert lp_score_2 > lp_score_1, (
        f"Changing performance should have higher LP score. Changing: {lp_score_2}, Consistent: {lp_score_1}"
    )
    print("✅ Changing performance has higher LP score than consistent performance")
    print("✅ Consistent performance has LP score close to zero")


def test_learning_progress_sampling_favors_high_lp_tasks():
    """Test that tasks with higher learning progress scores are sampled more frequently."""
    print("Testing learning progress sampling frequency...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Set up algorithm with tasks
    algorithm, tasks = setup_learning_progress_algorithm_with_tasks(num_tasks=2)
    task1_id = tasks[0]._task_id
    task2_id = tasks[1]._task_id

    # Train both tasks with different patterns
    print("Training task 1 (pattern A)...")
    for i in range(20):
        algorithm.update_task_performance(task1_id, 0.1 + i * 0.01)  # Pattern A - slower

    print("Training task 2 (pattern B)...")
    for i in range(20):
        algorithm.update_task_performance(task2_id, 0.1 + i * 0.05)  # Pattern B - much faster

    # Get LP scores
    lp_score_1 = algorithm._get_task_lp_score(task1_id)
    lp_score_2 = algorithm._get_task_lp_score(task2_id)

    print(f"Task 1 (pattern A) LP score: {lp_score_1}")
    print(f"Task 2 (pattern B) LP score: {lp_score_2}")
    print(f"Difference: {lp_score_2 - lp_score_1}")

    # Determine which task has higher LP score
    if lp_score_1 > lp_score_2:
        higher_lp_task = task1_id
        lower_lp_task = task2_id
        print(f"Task 1 has higher LP score ({lp_score_1}) than task 2 ({lp_score_2})")
    else:
        higher_lp_task = task2_id
        lower_lp_task = task1_id
        print(f"Task 2 has higher LP score ({lp_score_2}) than task 1 ({lp_score_1})")

    # Sample many times to get stable statistics
    selections = []
    for _ in range(1000):  # Increased sample size for more stable statistics
        selected_task_id = algorithm._sample_from_pool()
        selections.append(selected_task_id)

    higher_count = sum(1 for task_id in selections if task_id == higher_lp_task)
    lower_count = sum(1 for task_id in selections if task_id == lower_lp_task)

    print(f"Task {higher_lp_task} (higher LP) selected: {higher_count} times")
    print(f"Task {lower_lp_task} (lower LP) selected: {lower_count} times")

    # Calculate selection ratio
    selection_ratio = higher_count / (higher_count + lower_count)
    print(f"Higher LP task ({higher_lp_task}) selection ratio: {selection_ratio:.3f}")

    # More robust assertion: higher LP task should be selected more often
    assert higher_count > lower_count, (
        f"Task with higher LP score ({higher_lp_task}) should be selected more often. "
        f"Higher: {higher_count}, Lower: {lower_count}"
    )

    # More robust assertion: selection ratio should be significantly above 0.5
    assert selection_ratio > 0.55, (
        f"Task with higher LP score should be selected with ratio > 0.55. Got: {selection_ratio:.3f}"
    )
    print("✅ Higher LP score task is sampled more frequently")


def test_learning_progress_sampling_distribution():
    """Test that the sampling distribution favors tasks with higher learning progress."""
    print("Testing sampling distribution...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Set up algorithm with tasks
    algorithm, tasks = setup_learning_progress_algorithm_with_tasks(num_tasks=2)
    task1_id = tasks[0]._task_id
    task2_id = tasks[1]._task_id

    # Train task 1 to have higher LP score (fast learning)
    for i in range(15):
        algorithm.update_task_performance(task1_id, 0.1 + i * 0.08)  # Much faster improvement

    # Train task 2 to have lower LP score (consistent)
    for _i in range(15):
        algorithm.update_task_performance(task2_id, 0.5)  # No change

    # Get LP scores for debugging
    lp_score_1 = algorithm._get_task_lp_score(task1_id)
    lp_score_2 = algorithm._get_task_lp_score(task2_id)
    print(f"Task 1 LP score: {lp_score_1}")
    print(f"Task 2 LP score: {lp_score_2}")
    print(f"LP difference: {lp_score_1 - lp_score_2}")

    # Test sampling distribution with larger sample size for stability
    selections = []
    for _ in range(1000):  # Increased sample size
        selected_task_id = algorithm._sample_from_pool()
        selections.append(selected_task_id)

    task_1_count = sum(1 for task_id in selections if task_id == task1_id)
    task_2_count = sum(1 for task_id in selections if task_id == task2_id)

    print(f"Task 1 (high LP) selected: {task_1_count} times")
    print(f"Task 2 (low LP) selected: {task_2_count} times")

    # More robust assertion: check that the higher LP task is selected more often
    # with a reasonable margin for stochasticity
    if lp_score_1 > lp_score_2:
        expected_higher = task_1_count
        expected_lower = task_2_count
    else:
        expected_higher = task_2_count
        expected_lower = task_1_count

    # Allow for some stochasticity but ensure the trend is correct
    assert expected_higher > expected_lower, (
        f"High LP task should be selected more often. Task 1: {task_1_count}, Task 2: {task_2_count}"
    )

    # Check that the selection ratio is reasonable (allowing for exploration bonus)
    selection_ratio = expected_higher / (expected_higher + expected_lower)
    assert selection_ratio > 0.52, f"High LP task should have selection ratio > 0.52. Got: {selection_ratio:.3f}"
    print("✅ High LP task is selected more frequently")


def test_learning_progress_unified_pool_integration():
    """Test that the unified pool system works correctly with learning progress."""
    print("Testing unified pool integration...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Set up algorithm with tasks
    algorithm, tasks = setup_learning_progress_algorithm_with_tasks(num_tasks=3, pool_size=3)

    # Test that we got the expected number of tasks
    assert len(tasks) == 3, f"Should get 3 tasks, got {len(tasks)}"
    assert len(algorithm._task_memory) == 3, f"Pool should be full. Size: {len(algorithm._task_memory)}"

    # Test that all tasks are CurriculumTask instances
    from metta.cogworks.curriculum.curriculum import CurriculumTask

    for task in tasks:
        assert isinstance(task, CurriculumTask), "Should return CurriculumTask"

    print("✅ Unified pool integration works correctly")


if __name__ == "__main__":
    test_learning_progress_favors_fast_learning()
    test_learning_progress_favors_changing_performance()
    test_learning_progress_sampling_favors_high_lp_tasks()
    test_learning_progress_sampling_distribution()
    test_learning_progress_unified_pool_integration()
    print("\n🎉 All learning progress behavior tests passed!")
