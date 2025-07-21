#!/usr/bin/env python3
"""Tests for Prioritize Regressed Curriculum specific scenarios.

This curriculum prioritizes tasks where performance has regressed from peak.
Weight = max_reward / average_reward, so high weight means we've done better before.
"""


import numpy as np
import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.curriculum.prioritize_regressed import PrioritizeRegressedCurriculum

from .conftest import (
    ConditionalLinearScores,
    IndependentLinearScores,
    create_mock_curricula,
    run_curriculum_simulation,
)


class TestPrioritizeRegressedCurriculumScenarios:
    """Test the specific Prioritize Regressed Curriculum scenarios."""

    def test_scenario_6_all_linear_scaling_equal_distribution(self, monkeypatch):
        """
        Scenario 6: All tasks have linear scaling rewards.

        Expected: With the same linear progression, max/avg ratio should be similar for all tasks,
        leading to approximately equal distribution over time.
        """
        print("\n=== PRIORITIZE REGRESSED SCENARIO 6: All Linear Scaling ===")

        # Patch curriculum_from_config_path
        def mock_curriculum_from_config_path(path, env_overrides=None):
            default_cfg = OmegaConf.create({"game": {"num_agents": 1, "map": {"width": 10, "height": 10}}})
            cfg = OmegaConf.merge(default_cfg, env_overrides or {})
            return SingleTaskCurriculum(path, cfg)

        monkeypatch.setattr(
            "metta.mettagrid.curriculum.random.curriculum_from_config_path", mock_curriculum_from_config_path
        )

        tasks = ["task_1", "task_2", "task_3"]
        task_weights = create_mock_curricula(tasks)

        # Each task gets its own independent linear progression
        score_gen = IndependentLinearScores(increment=0.05)

        curriculum = PrioritizeRegressedCurriculum(
            tasks=task_weights,
            moving_avg_decay_rate=0.1,  # Moderate smoothing
        )

        # Initialize all tasks with a small score to avoid the 0/0 issue
        for task in tasks:
            curriculum.get_task()
            curriculum.complete_task(task, 0.01)

        results = run_curriculum_simulation(curriculum, score_gen, 200)

        # Analyze results
        weight_history = results["weight_history"]
        final_weights = results["final_weights"]
        task_counts = results["task_counts"]

        print(f"Task counts: {task_counts}")
        print(f"Final weights: {final_weights}")

        # Check weight evolution
        assert len(weight_history) == 200, f"Should have 200 weight snapshots, got {len(weight_history)}"

        # Early: should start with equal weights
        early_weights = weight_history[10]
        print(f"Early weights (step 10): {early_weights}")

        # Middle: weights should remain relatively balanced
        mid_weights = weight_history[100]
        print(f"Mid weights (step 100): {mid_weights}")

        # Late: weights should still be relatively balanced
        late_weights = weight_history[180]
        print(f"Late weights (step 180): {late_weights}")

        # Final analysis: all tasks should have similar weights
        # With same linear progression, max/avg ratios should be similar
        weights_list = list(final_weights.values())
        weight_variance = np.var(weights_list)
        print(f"Final weight variance: {weight_variance:.6f}")

        # Weights should be relatively equal (low variance)
        assert weight_variance < 0.01, (
            f"Weights should be relatively equal with same linear progression, variance: {weight_variance}"
        )

        # Task counts should be relatively balanced
        total_samples = sum(task_counts.values())
        task_count_values = []
        for task in tasks:
            task_count = sum(count for t, count in task_counts.items() if task in t)
            task_count_values.append(task_count)

        task_ratios = [count / total_samples for count in task_count_values]
        print(f"Task sampling ratios: {[f'{r:.3f}' for r in task_ratios]}")

        # All tasks should be sampled roughly equally (33% each)
        for ratio in task_ratios:
            assert 0.2 < ratio < 0.5, f"Task sampling should be relatively balanced, got ratio: {ratio}"

        print("✓ PASSED: All linear scaling tasks maintain equal distribution")

    def test_scenario_7_one_impossible_task_gets_lowest_weight(self, monkeypatch):
        """
        Scenario 7: One task is impossible (always returns 0), others have linear scaling.

        Expected: The impossible task has max/avg = 0/0, resulting in weight = epsilon.
        Learnable tasks that improve over time will have max > avg, giving them higher weight.
        The curriculum should focus on learnable tasks, especially those showing regression.
        """
        print("\n=== PRIORITIZE REGRESSED SCENARIO 7: One Impossible Task ===")

        # Patch curriculum_from_config_path
        def mock_curriculum_from_config_path(path, env_overrides=None):
            default_cfg = OmegaConf.create({"game": {"num_agents": 1, "map": {"width": 10, "height": 10}}})
            cfg = OmegaConf.merge(default_cfg, env_overrides or {})
            return SingleTaskCurriculum(path, cfg)

        monkeypatch.setattr(
            "metta.mettagrid.curriculum.random.curriculum_from_config_path", mock_curriculum_from_config_path
        )

        tasks = ["impossible", "learnable_1", "learnable_2"]
        task_weights = create_mock_curricula(tasks)

        # Only learnable tasks give increasing signals
        learnable_tasks = {"learnable_1", "learnable_2"}
        score_gen = ConditionalLinearScores(linear_tasks=learnable_tasks, increment=0.1)

        curriculum = PrioritizeRegressedCurriculum(
            tasks=task_weights,
            moving_avg_decay_rate=0.05,  # Slower adaptation
        )

        # Initialize all tasks to ensure they all get sampled at least once
        for task in tasks:
            curriculum.get_task()
            score = score_gen.get_score(task)
            curriculum.complete_task(task, score)

        results = run_curriculum_simulation(curriculum, score_gen, 300)

        # Analyze results
        weight_history = results["weight_history"]
        final_weights = results["final_weights"]
        task_counts = results["task_counts"]

        print(f"Task counts: {task_counts}")
        print(f"Final weights: {final_weights}")

        # Check weight evolution
        assert len(weight_history) == 300, f"Should have 300 weight snapshots, got {len(weight_history)}"

        # Early: should start with equal weights
        early_weights = weight_history[10]
        print(f"Early weights (step 10): {early_weights}")

        # After some samples, learnable tasks should dominate
        mid_weights = weight_history[100]
        print(f"Mid weights (step 100): {mid_weights}")

        # Late: learnable tasks should strongly dominate
        late_weights = weight_history[250]
        print(f"Late weights (step 250): {late_weights}")

        # Final analysis: impossible task should have minimal weight
        # because it has max/avg = 0/0 (no peak performance to regress from)
        max_learnable_weight = max(final_weights["learnable_1"], final_weights["learnable_2"])
        assert final_weights["impossible"] < max_learnable_weight, (
            f"Impossible task should have lower weight than learnable tasks: "
            f"{final_weights['impossible']} vs {max_learnable_weight}"
        )

        # The impossible task weight should be close to epsilon relative to learnable tasks
        assert final_weights["impossible"] < 0.01, (
            f"Impossible task should have minimal normalized weight, got {final_weights['impossible']}"
        )

        # Task counts - impossible task should be sampled least
        total_samples = sum(task_counts.values())
        impossible_count = sum(count for task, count in task_counts.items() if "impossible" in task)
        learnable_1_count = sum(count for task, count in task_counts.items() if "learnable_1" in task)
        learnable_2_count = sum(count for task, count in task_counts.items() if "learnable_2" in task)

        impossible_ratio = impossible_count / total_samples if total_samples > 0 else 0
        learnable_1_ratio = learnable_1_count / total_samples if total_samples > 0 else 0
        learnable_2_ratio = learnable_2_count / total_samples if total_samples > 0 else 0

        print(
            f"Sampling ratios - Impossible: {impossible_ratio:.3f}, "
            f"Learnable_1: {learnable_1_ratio:.3f}, Learnable_2: {learnable_2_ratio:.3f}"
        )

        # Impossible task should be sampled less than learnable tasks
        # because it has no peak performance to regress from (max/avg = 0/0)
        learnable_count = learnable_1_count + learnable_2_count
        assert impossible_count < learnable_count, (
            f"Impossible task should be sampled less than learnable tasks combined: "
            f"{impossible_count} vs {learnable_count}"
        )

        # Impossible task should get minimal samples
        assert impossible_ratio < 0.1, f"Impossible task should get minimal samples, got {impossible_ratio:.3f}"

        # Learnable tasks should dominate
        learnable_ratio = learnable_1_ratio + learnable_2_ratio
        assert learnable_ratio > 0.9, f"Learnable tasks should dominate sampling, got {learnable_ratio:.3f}"

        print("✓ PASSED: Prioritize regressed curriculum correctly avoids impossible task (no regression possible)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])