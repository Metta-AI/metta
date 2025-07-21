#!/usr/bin/env python3
"""Consolidated curriculum scenario tests.

This file combines all scenario-based tests for different curriculum algorithms:
- Progressive curriculum scenarios
- Learning progress curriculum scenarios  
- Prioritize regressed curriculum scenarios
"""

import numpy as np
import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.curriculum.learning_progress import LearningProgressCurriculum
from metta.mettagrid.curriculum.prioritize_regressed import PrioritizeRegressedCurriculum
from metta.mettagrid.curriculum.progressive import ProgressiveMultiTaskCurriculum

from .scenario_helpers import (
    ConditionalLinearScores,
    IndependentLinearScores,
    MonotonicLinearScores,
    RandomScores,
    ThresholdDependentScores,
    ZeroScores,
    create_mock_curricula,
    fake_curriculum_from_config_path,
    run_curriculum_simulation,
)


class TestProgressiveCurriculumScenarios:
    """Test Progressive Curriculum behavior under controlled conditions."""

    def test_monotonic_linear_advances_correctly(self, monkeypatch):
        """Monotonic linear signal should advance through tasks with right timing."""
        print("\n=== PROGRESSIVE SCENARIO: Monotonic Linear Signal ===")

        monkeypatch.setattr(
            "metta.mettagrid.curriculum.random.curriculum_from_config_path",
            fake_curriculum_from_config_path
        )

        tasks = ["task_1", "task_2", "task_3"]
        task_weights = create_mock_curricula(tasks)
        score_gen = MonotonicLinearScores(increment=0.1)

        curriculum = ProgressiveMultiTaskCurriculum(
            tasks=task_weights,
            performance_threshold=0.8,
            progression_rate=0.3,
            smoothing=0.1,
            blending_smoothness=0.2,
            progression_mode="perf",
        )

        results = run_curriculum_simulation(curriculum, score_gen, 100)

        # Verify progression
        weight_history = results["weight_history"]
        final_weights = results["final_weights"]
        progress = results["curriculum_stats"].get("progress", 0)

        # Should start focused on task_1
        assert weight_history[5]["task_1"] > 0.8
        # Should show progression by end
        assert progress > 0.3
        # All tasks should be sampled
        assert len(results["task_counts"]) == 3

        print("✓ PASSED: Monotonic signal correctly advances through tasks")

    def test_zero_signal_stays_on_first(self, monkeypatch):
        """Always 0 signal should keep curriculum on first task."""
        print("\n=== PROGRESSIVE SCENARIO: Zero Signal ===")

        monkeypatch.setattr(
            "metta.mettagrid.curriculum.random.curriculum_from_config_path",
            fake_curriculum_from_config_path
        )

        tasks = ["task_1", "task_2", "task_3"]
        task_weights = create_mock_curricula(tasks)
        score_gen = ZeroScores()

        curriculum = ProgressiveMultiTaskCurriculum(
            tasks=task_weights,
            performance_threshold=0.5,
            progression_rate=0.1,
            smoothing=0.1,
            progression_mode="perf",
        )

        results = run_curriculum_simulation(curriculum, score_gen, 100)

        # Should stay on task_1
        final_weights = results["final_weights"]
        assert final_weights["task_1"] > 0.5
        
        # Progress should be zero
        progress = results["curriculum_stats"].get("progress", 0)
        assert progress == 0.0

        print("✓ PASSED: Zero signal correctly stays on first task")

    def test_random_signal_still_progresses(self, monkeypatch):
        """Random signal should still progress with right parameters."""
        print("\n=== PROGRESSIVE SCENARIO: Random Signal ===")

        monkeypatch.setattr(
            "metta.mettagrid.curriculum.random.curriculum_from_config_path",
            fake_curriculum_from_config_path
        )

        tasks = ["task_1", "task_2", "task_3"]
        task_weights = create_mock_curricula(tasks)
        score_gen = RandomScores(seed=42, min_val=0.0, max_val=1.0)

        curriculum = ProgressiveMultiTaskCurriculum(
            tasks=task_weights,
            performance_threshold=0.4,
            progression_rate=0.05,
            smoothing=0.5,
            progression_mode="perf",
        )

        results = run_curriculum_simulation(curriculum, score_gen, 300)

        # Should show progression
        progress = results["curriculum_stats"].get("progress", 0)
        assert progress > 0.1
        
        # All tasks should be sampled
        assert len(results["task_counts"]) == 3

        print("✓ PASSED: Random signal enables gradual progression")


class TestLearningProgressCurriculumScenarios:
    """Test Learning Progress curriculum behavior under controlled conditions."""

    def test_mixed_impossible_learnable_tasks(self, monkeypatch):
        """Mixed impossible and learnable tasks - should weight learnable evenly, ignore impossible."""
        print("\n=== LEARNING PROGRESS SCENARIO: Mixed Impossible/Learnable ===")

        monkeypatch.setattr(
            "metta.mettagrid.curriculum.random.curriculum_from_config_path",
            fake_curriculum_from_config_path
        )

        tasks = ["impossible_1", "impossible_2", "learnable_1", "learnable_2"]
        task_weights = create_mock_curricula(tasks)
        learnable_tasks = {"learnable_1", "learnable_2"}
        score_gen = ConditionalLinearScores(linear_tasks=learnable_tasks, increment=0.1)

        curriculum = LearningProgressCurriculum(
            tasks=task_weights,
            ema_timescale=0.02,
            sample_threshold=5,
            memory=15,
            num_active_tasks=4,
            rand_task_rate=0.1,
        )

        # Initialize all tasks
        for task_name in tasks:
            task = curriculum.get_task()
            score = score_gen.get_score(task_name)
            curriculum.complete_task(task_name, score)

        results = run_curriculum_simulation(curriculum, score_gen, 600)

        # Calculate final weight groups
        final_weights = results["final_weights"]
        impossible_weight = final_weights.get("impossible_1", 0) + final_weights.get("impossible_2", 0)
        learnable_weight = final_weights.get("learnable_1", 0) + final_weights.get("learnable_2", 0)

        # Learnable tasks should dominate
        assert learnable_weight > impossible_weight * 1.5
        assert learnable_weight > 0.55

        print("✓ PASSED: Learning progress correctly identifies and prefers learnable tasks")

    def test_threshold_dependent_progression(self, monkeypatch):
        """Primary task reaches milestone, then secondary becomes learnable."""
        print("\n=== LEARNING PROGRESS SCENARIO: Threshold Dependency ===")

        monkeypatch.setattr(
            "metta.mettagrid.curriculum.random.curriculum_from_config_path",
            fake_curriculum_from_config_path
        )

        tasks = ["primary", "secondary"]
        task_weights = create_mock_curricula(tasks)
        threshold = 0.5
        score_gen = ThresholdDependentScores(
            primary_task="primary", secondary_task="secondary", threshold=threshold, increment=0.1
        )

        curriculum = LearningProgressCurriculum(
            tasks=task_weights,
            ema_timescale=0.03,
            sample_threshold=5,
            memory=20,
            rand_task_rate=0.5,
        )

        # Initialize tasks
        for task_name in tasks:
            curriculum.get_task()
            score = score_gen.get_score(task_name)
            curriculum.complete_task(task_name, score)

        results = run_curriculum_simulation(curriculum, score_gen, 150)

        # Verify both tasks were sampled
        assert len(results["task_counts"]) == 2
        
        # Primary should be heavily sampled
        primary_count = sum(count for task, count in results["task_counts"].items() if "primary" in task)
        assert primary_count > 60

        print("✓ PASSED: Learning progress correctly responds to threshold-dependent task dynamics")


class TestPrioritizeRegressedCurriculumScenarios:
    """Test Prioritize Regressed curriculum behavior under controlled conditions."""

    def test_all_linear_scaling_equal_distribution(self, monkeypatch):
        """All tasks have linear scaling - should have equal distribution."""
        print("\n=== PRIORITIZE REGRESSED SCENARIO: All Linear Scaling ===")

        def mock_curriculum_from_config_path(path, env_overrides=None):
            default_cfg = OmegaConf.create({"game": {"num_agents": 1, "map": {"width": 10, "height": 10}}})
            cfg = OmegaConf.merge(default_cfg, env_overrides or {})
            return SingleTaskCurriculum(path, cfg)

        monkeypatch.setattr(
            "metta.mettagrid.curriculum.random.curriculum_from_config_path", 
            mock_curriculum_from_config_path
        )

        tasks = ["task_1", "task_2", "task_3"]
        task_weights = create_mock_curricula(tasks)
        score_gen = IndependentLinearScores(increment=0.05)

        curriculum = PrioritizeRegressedCurriculum(
            tasks=task_weights,
            moving_avg_decay_rate=0.1,
        )

        # Initialize tasks
        for task in tasks:
            curriculum.get_task()
            curriculum.complete_task(task, 0.01)

        results = run_curriculum_simulation(curriculum, score_gen, 200)

        # All tasks should have similar weights
        final_weights = results["final_weights"]
        weights_list = list(final_weights.values())
        weight_variance = np.var(weights_list)
        assert weight_variance < 0.01

        print("✓ PASSED: All linear scaling tasks maintain equal distribution")

    def test_one_impossible_task_gets_lowest_weight(self, monkeypatch):
        """One impossible task (always 0) should get minimal weight."""
        print("\n=== PRIORITIZE REGRESSED SCENARIO: One Impossible Task ===")

        def mock_curriculum_from_config_path(path, env_overrides=None):
            default_cfg = OmegaConf.create({"game": {"num_agents": 1, "map": {"width": 10, "height": 10}}})
            cfg = OmegaConf.merge(default_cfg, env_overrides or {})
            return SingleTaskCurriculum(path, cfg)

        monkeypatch.setattr(
            "metta.mettagrid.curriculum.random.curriculum_from_config_path",
            mock_curriculum_from_config_path
        )

        tasks = ["impossible", "learnable_1", "learnable_2"]
        task_weights = create_mock_curricula(tasks)
        learnable_tasks = {"learnable_1", "learnable_2"}
        score_gen = ConditionalLinearScores(linear_tasks=learnable_tasks, increment=0.1)

        curriculum = PrioritizeRegressedCurriculum(
            tasks=task_weights,
            moving_avg_decay_rate=0.05,
        )

        # Initialize tasks
        for task in tasks:
            curriculum.get_task()
            score = score_gen.get_score(task)
            curriculum.complete_task(task, score)

        results = run_curriculum_simulation(curriculum, score_gen, 300)

        # Impossible task should have minimal weight
        final_weights = results["final_weights"]
        max_learnable_weight = max(final_weights["learnable_1"], final_weights["learnable_2"])
        assert final_weights["impossible"] < max_learnable_weight
        assert final_weights["impossible"] < 0.01

        # Impossible task should be sampled least
        task_counts = results["task_counts"]
        impossible_count = sum(count for task, count in task_counts.items() if "impossible" in task)
        learnable_count = sum(count for task, count in task_counts.items() if "learnable" in task)
        assert impossible_count < learnable_count

        print("✓ PASSED: Prioritize regressed correctly avoids impossible task")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])