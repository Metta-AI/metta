#!/usr/bin/env python3
"""Progressive curriculum scenario tests.

Tests for Progressive curriculum behavior under controlled conditions:
1. Monotonic linear signal -> should advance through tasks correctly
2. Always 0 signal -> should stay on first task
3. Random signal -> should still progress with right parameters
"""


import numpy as np
import pytest

from metta.mettagrid.curriculum.progressive import ProgressiveMultiTaskCurriculum

from .conftest import (
    MonotonicLinearScores,
    RandomScores,
    ZeroScores,
    create_mock_curricula,
    fake_curriculum_from_config_path,
    run_curriculum_simulation,
)


class TestProgressiveCurriculumScenarios:
    """Test the specific Progressive Curriculum scenarios."""

    def test_scenario_1_monotonic_linear_advances_correctly(self, monkeypatch):
        """
        Scenario 1: Monotonic linear signal should advance through tasks with right timing.

        Expected: Should spend time on each task in order, advancing when signal increases.
        """
        print("\n=== PROGRESSIVE SCENARIO 1: Monotonic Linear Signal ===")

        # Patch curriculum_from_config_path
        monkeypatch.setattr(
            "metta.mettagrid.curriculum.random.curriculum_from_config_path",
            fake_curriculum_from_config_path
        )

        tasks = ["task_1", "task_2", "task_3"]
        task_weights = create_mock_curricula(tasks)

        # Create a monotonic linear score generator
        score_gen = MonotonicLinearScores(increment=0.1)

        curriculum = ProgressiveMultiTaskCurriculum(
            tasks=task_weights,
            performance_threshold=0.8,  # Will reach after ~8 steps per task
            progression_rate=0.3,  # Advance 30% per threshold crossing
            smoothing=0.1,  # Fast adaptation to signal
            blending_smoothness=0.2,  # Moderate transitions
            progression_mode="perf",  # Progress based on performance
        )

        results = run_curriculum_simulation(curriculum, score_gen, 100)

        # Analyze progression pattern
        weight_history = results["weight_history"]
        score_history = results["score_history"]
        assert len(weight_history) == 100, f"Should have 100 weight snapshots, got {len(weight_history)}"

        # Debug: Print first 20 scores to understand progression
        print(f"First 20 scores: {score_history[:20]}")

        # Early: should focus on task_1
        early_weights = weight_history[5]
        print(f"Early weights (step 5): {early_weights}")
        assert early_weights["task_1"] > 0.8, f"Should start strongly focused on task_1, got {early_weights}"
        assert early_weights.get("task_2", 0) < 0.2, f"Task 2 should have minimal weight early, got {early_weights}"
        assert early_weights.get("task_3", 0) < 0.1, f"Task 3 should have minimal weight early, got {early_weights}"

        # Check progression more gradually
        mid_weights = weight_history[50]
        print(f"Mid weights (step 50): {mid_weights}")
        progress = results["curriculum_stats"].get("progress", 0)
        print(f"Progress at end: {progress}")

        # Task 1 should have reduced weight
        assert mid_weights["task_1"] < 0.7, f"Task 1 weight should decrease by middle, got {mid_weights}"
        # Later tasks should have meaningful weight
        later_tasks_weight = mid_weights.get("task_2", 0) + mid_weights.get("task_3", 0)
        assert later_tasks_weight > 0.3, f"Should show clear progression by middle, got {mid_weights}"

        # Late: should show further progression
        late_weights = weight_history[80]
        print(f"Late weights (step 80): {late_weights}")
        assert late_weights.get("task_3", 0) > 0.05, f"Task 3 should have some weight by late stage, got {late_weights}"

        # Final state analysis
        final_weights = results["final_weights"]
        task_counts = results["task_counts"]
        curriculum_stats = results["curriculum_stats"]
        print(f"Final weights: {final_weights}")
        print(f"Task counts: {task_counts}")
        print(f"Curriculum stats: {curriculum_stats}")

        # Progress should be meaningful
        progress = curriculum_stats.get("progress", 0)
        assert progress > 0.3, f"Should show significant progress with monotonic signal, got {progress}"

        # Should have sampled all tasks at least once
        assert len(task_counts) == 3, f"Should have tried all 3 tasks, got {task_counts.keys()}"
        assert all(count > 0 for count in task_counts.values()), (
            f"All tasks should be sampled at least once, got {task_counts}"
        )

        print("✓ PASSED: Monotonic signal correctly advances through tasks with expected timing")

    def test_scenario_2_zero_signal_stays_on_first(self, monkeypatch):
        """
        Scenario 2: Always 0 signal should keep curriculum on first task.

        Expected: Should stay overwhelmingly on first task throughout training.
        """
        print("\n=== PROGRESSIVE SCENARIO 2: Zero Signal ===")

        # Patch curriculum_from_config_path
        monkeypatch.setattr(
            "metta.mettagrid.curriculum.random.curriculum_from_config_path",
            fake_curriculum_from_config_path
        )

        tasks = ["task_1", "task_2", "task_3"]
        task_weights = create_mock_curricula(tasks)

        # Create a zero score generator
        score_gen = ZeroScores()

        curriculum = ProgressiveMultiTaskCurriculum(
            tasks=task_weights,
            performance_threshold=0.5,
            progression_rate=0.1,
            smoothing=0.1,
            progression_mode="perf",
        )

        results = run_curriculum_simulation(curriculum, score_gen, 100)

        # Analyze the results
        weight_history = results["weight_history"]
        final_weights = results["final_weights"]
        task_counts = results["task_counts"]
        curriculum_stats = results["curriculum_stats"]

        print(f"Final weights: {final_weights}")
        print(f"Task counts: {task_counts}")
        print(f"Curriculum stats: {curriculum_stats}")

        # Check weights throughout training - should remain on task_1
        assert len(weight_history) == 100, f"Should have 100 weight snapshots, got {len(weight_history)}"

        # With progress=0, the gating mechanism gives higher weight to early tasks
        early_weights = weight_history[10]
        print(f"Early weights (step 10): {early_weights}")
        assert early_weights["task_1"] > 0.5, f"Should favor task_1 early, got {early_weights}"

        # Middle weights - should still favor task_1
        mid_weights = weight_history[50]
        print(f"Mid weights (step 50): {mid_weights}")
        assert mid_weights["task_1"] > 0.5, f"Should still favor task_1 at midpoint, got {mid_weights}"

        # Late weights - should still favor task_1
        late_weights = weight_history[80]
        print(f"Late weights (step 80): {late_weights}")
        assert late_weights["task_1"] > 0.5, f"Should still favor task_1 late in training, got {late_weights}"

        # Weights should remain constant throughout (no progress)
        weight_variance = np.var([early_weights["task_1"], mid_weights["task_1"], late_weights["task_1"]])
        assert weight_variance < 0.01, f"Weights should remain stable with zero progress, variance: {weight_variance}"

        # Final weights should favor task_1
        assert final_weights["task_1"] > 0.5, f"Should favor task_1, got {final_weights['task_1']}"
        assert final_weights["task_1"] > final_weights.get("task_2", 0), "Task 1 should have more weight than task 2"
        assert final_weights["task_1"] > final_weights.get("task_3", 0), "Task 1 should have more weight than task 3"

        # Task 1 should dominate selection counts
        total_selections = sum(task_counts.values())
        task_1_count = sum(count for task, count in task_counts.items() if "task_1" in task)
        task_1_ratio = task_1_count / total_selections if total_selections > 0 else 0
        assert task_1_ratio > 0.5, f"Should spend majority time on task_1 with zero scores, got {task_1_ratio:.2f}"

        # Progress should be zero
        progress = curriculum_stats.get("progress", 0)
        assert progress == 0.0, (
            f"Progress should be exactly 0 with zero signal (never crosses threshold), got {progress}"
        )

        print("✓ PASSED: Zero signal correctly stays on first task throughout training")

    def test_scenario_3_random_signal_still_progresses(self, monkeypatch):
        """
        Scenario 3: Random signal should still progress with right parameters.

        Expected: Due to randomness occasionally exceeding threshold, should eventually progress.
        """
        print("\n=== PROGRESSIVE SCENARIO 3: Random Signal ===")

        # Patch curriculum_from_config_path
        monkeypatch.setattr(
            "metta.mettagrid.curriculum.random.curriculum_from_config_path",
            fake_curriculum_from_config_path
        )

        tasks = ["task_1", "task_2", "task_3"]
        task_weights = create_mock_curricula(tasks)

        # Create a random score generator with seed for reproducibility
        score_gen = RandomScores(seed=42, min_val=0.0, max_val=1.0)

        curriculum = ProgressiveMultiTaskCurriculum(
            tasks=task_weights,
            performance_threshold=0.4,  # Low threshold - random will occasionally exceed
            progression_rate=0.05,  # Slower progression
            smoothing=0.5,  # Heavy smoothing to handle noise
            progression_mode="perf",
        )

        results = run_curriculum_simulation(curriculum, score_gen, 300)  # Longer simulation

        # Analyze results
        weight_history = results["weight_history"]
        task_counts = results["task_counts"]
        final_weights = results["final_weights"]
        curriculum_stats = results["curriculum_stats"]
        progress = curriculum_stats.get("progress", 0)

        print(f"Task counts: {task_counts}")
        print(f"Final weights: {final_weights}")
        print(f"Final progress: {progress}")
        print(f"Curriculum stats: {curriculum_stats}")

        # Check weight evolution - should show progression over time
        assert len(weight_history) == 300, f"Should have 300 weight snapshots, got {len(weight_history)}"

        # With heavy smoothing and low threshold, progression can happen quickly
        very_early_weights = weight_history[5]
        print(f"Very early weights (step 5): {very_early_weights}")

        # By step 20, with random scores and heavy smoothing, weights may already be distributed
        early_weights = weight_history[20]
        print(f"Early weights (step 20): {early_weights}")
        # Just verify all tasks have some weight (exploration happening)
        assert all(early_weights.get(task, 0) > 0.01 for task in tasks), (
            f"All tasks should have some weight by step 20, got {early_weights}"
        )

        # Check progression is happening
        mid_early_weights = weight_history[50]
        print(f"Mid-early weights (step 50): {mid_early_weights}")
        # Progress should be advancing
        assert progress > 0, "Should show progress after 300 steps with random signal"

        # Middle: should show clear progression
        mid_weights = weight_history[150]
        print(f"Mid weights (step 150): {mid_weights}")
        assert mid_weights["task_1"] < 0.9, f"Task 1 weight should decrease by middle, got {mid_weights}"
        assert mid_weights.get("task_2", 0) > 0.05, f"Task 2 should have meaningful weight by middle, got {mid_weights}"

        # Late: should show further progression
        late_weights = weight_history[250]
        print(f"Late weights (step 250): {late_weights}")
        task_2_and_3_weight = late_weights.get("task_2", 0) + late_weights.get("task_3", 0)
        assert task_2_and_3_weight > 0.1, (
            f"Later tasks should have combined weight > 0.1 by late stage, got {late_weights}"
        )

        # Final analysis
        # Should have tried all tasks due to randomness and progression
        assert len(task_counts) == 3, f"Should have tried all 3 tasks with 300 steps, got {task_counts.keys()}"
        assert all(count > 0 for count in task_counts.values()), (
            f"All tasks should be sampled at least once, got {task_counts}"
        )

        # Progress should be meaningful
        assert progress > 0.1, f"Should show meaningful progression from random signal, got {progress}"

        # Task distribution should show progression
        total_selections = sum(task_counts.values())
        task_1_count = sum(count for task, count in task_counts.items() if "task_1" in task)
        task_2_count = sum(count for task, count in task_counts.items() if "task_2" in task)
        task_3_count = sum(count for task, count in task_counts.items() if "task_3" in task)

        task_1_ratio = task_1_count / total_selections
        task_2_ratio = task_2_count / total_selections
        task_3_ratio = task_3_count / total_selections

        print(f"Task ratios - 1: {task_1_ratio:.2f}, 2: {task_2_ratio:.2f}, 3: {task_3_ratio:.2f}")

        # With full progression, later tasks should dominate
        assert task_3_ratio > task_1_ratio, "Task 3 should be sampled more than task 1 with full progression"
        assert task_1_ratio < 0.5, f"Task 1 should not dominate with full progression, got {task_1_ratio:.2f}"
        assert task_3_ratio > 0.3, (
            f"Task 3 should have significant samples with full progression, got {task_3_ratio:.2f}"
        )

        print("✓ PASSED: Random signal enables gradual progression with appropriate parameters")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])