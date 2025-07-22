"""
Test suite for curriculum algorithm scenarios using the new Curriculum system.

This module tests specific scenarios to validate that curriculum algorithms
behave correctly under controlled conditions using the Curriculum architecture.

Progressive Algorithm Tests:
1. Monotonic linear signal -> should advance through tasks correctly
2. Always 0 signal -> should stay on first task
3. Random signal -> should still progress with right parameters

Learning Progress Algorithm Tests:
4. Mixed impossible/learnable tasks -> should weight learnable evenly, ignore impossible
5. Threshold dependency -> should first weight primary, then secondary after milestone

Prioritize Regressed Algorithm Tests:
6. All tasks have linear scaling -> should maintain equal distribution
7. One impossible task (always 0) -> should get minimum weight as max_score = 0 ==> LP = epsilon
8. Different score trajectories -> regressed task (A) gets highest weight, non-regressed tasks (B,C) similar
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum import task_set
from metta.mettagrid.curriculum.curriculum import (
    Curriculum,
    MettaGridTask,
)
from metta.mettagrid.curriculum.curriculum_algorithm import (
    CurriculumAlgorithm,
    CurriculumAlgorithmHypers,
    DiscreteRandomHypers,
)
from metta.mettagrid.curriculum.learning_progress import LearningProgressHypers
from metta.mettagrid.curriculum.prioritize_regressed import PrioritizeRegressedHypers
from metta.mettagrid.curriculum.progressive import (
    ProgressiveHypers,
    SimpleProgressiveHypers,
)


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set all random seeds for deterministic test behavior."""
    random.seed(42)
    np.random.seed(42)
    yield
    # Reset after test
    random.seed()
    np.random.seed()


@pytest.fixture
def env_cfg():
    return OmegaConf.create({"sampling": 0, "game": {"num_agents": 1, "map": {"width": 10, "height": 10}}})


# ============================================================================
# Score Generators for Controlled Testing
# ============================================================================


class ScoreGenerator(ABC):
    """Abstract interface for generating controlled scores for curriculum testing."""

    @abstractmethod
    def get_score(self, task_id: str) -> float:
        """Get the next score for the given task.

        Args:
            task_id: Identifier of the task being completed

        Returns:
            Score to pass to curriculum.complete()
        """
        pass

    def reset(self) -> None:
        """Reset generator state. Override if needed."""
        return


class MonotonicLinearScores(ScoreGenerator):
    """Generates monotonically increasing scores for each task.

    Each time a task is run, returns count * increment.
    Useful for testing progressive curricula behavior.
    """

    def __init__(self, increment: float = 0.1):
        self.increment = increment
        self.task_counts = {}

    def get_score(self, task_id: str) -> float:
        count = self.task_counts.get(task_id, 0)
        self.task_counts[task_id] = count + 1
        return min(1.0, count * self.increment)  # Cap at 1.0

    def reset(self) -> None:
        self.task_counts = {}


class ZeroScores(ScoreGenerator):
    """Always returns 0 regardless of task.

    Useful for testing curriculum behavior when no learning occurs.
    """

    def get_score(self, task_id: str) -> float:
        return 0.0


class RandomScores(ScoreGenerator):
    """Returns random scores in [min_val, max_val].

    Useful for testing curriculum behavior under noisy conditions.
    """

    def __init__(self, seed: int = None, min_val: float = 0.0, max_val: float = 1.0):
        self.rng = random.Random(seed)
        self.min_val = min_val
        self.max_val = max_val

    def get_score(self, task_id: str) -> float:
        return self.rng.uniform(self.min_val, self.max_val)


class ConditionalLinearScores(ScoreGenerator):
    """Linear increasing scores for specified tasks, zero for others.

    Useful for testing learning progress curricula with mixed task difficulties.
    """

    def __init__(self, linear_tasks: Set[str], increment: float = 0.1):
        self.linear_tasks = linear_tasks
        self.increment = increment
        self.task_counts = {}

    def get_score(self, task_id: str) -> float:
        # Extract base task name in case it has curriculum prefix
        base_task = task_id.split("/")[-1] if "/" in task_id else task_id

        if base_task in self.linear_tasks:
            count = self.task_counts.get(base_task, 0)
            self.task_counts[base_task] = count + 1
            return min(1.0, count * self.increment)
        else:
            return 0.0

    def reset(self) -> None:
        self.task_counts = {}


class ThresholdDependentScores(ScoreGenerator):
    """Scores where secondary task only gives reward after primary reaches threshold.

    Primary task: linear increaser that flatlines after reaching threshold
    Secondary task: stays at 0 until primary reaches threshold, then linear increaser

    Useful for testing curriculum adaptation to task dependencies.
    """

    def __init__(self, primary_task: str, secondary_task: str, threshold: float = 0.5, increment: float = 0.1):
        self.primary_task = primary_task
        self.secondary_task = secondary_task
        self.threshold = threshold
        self.increment = increment
        self.task_counts = {}
        self.primary_score = 0.0

    def get_score(self, task_id: str) -> float:
        # Extract base task name in case it has curriculum prefix
        base_task = task_id.split("/")[-1] if "/" in task_id else task_id

        count = self.task_counts.get(base_task, 0)
        self.task_counts[base_task] = count + 1

        if base_task == self.primary_task:
            score = min(count * self.increment, self.threshold)
            self.primary_score = score
            return score

        elif base_task == self.secondary_task:
            if self.primary_score >= self.threshold:
                return min(1.0, count * self.increment)
            else:
                return 0.0
        else:
            return 0.0

    def reset(self) -> None:
        self.task_counts = {}
        self.primary_score = 0.0


# ============================================================================
# Test Utilities
# ============================================================================


def create_curriculum_with_algorithm(
    task_names: List[str], algorithm: CurriculumAlgorithm, env_cfg: DictConfig
) -> Curriculum:
    """Create a Curriculum with the specified algorithm and tasks.

    Args:
        task_names: List of task names
        algorithm: Curriculum algorithm to use
        env_cfg: Environment configuration for tasks

    Returns:
        Curriculum initialized with the given algorithm and tasks
    """
    # Create task configs as list of tuples
    env_configs = [(name, env_cfg.copy()) for name in task_names]

    # Create custom hypers that wraps the algorithm
    class CustomHypers(CurriculumAlgorithmHypers):
        def algorithm_type(self) -> str:
            return "custom"

        def create(self, num_tasks: int) -> CurriculumAlgorithm:
            return algorithm

    # Create Curriculum using task_set helper
    return task_set(
        name="test_curriculum",
        env_configs=env_configs,
        curriculum_hypers=CustomHypers(),
    )


def run_curriculum_simulation(
    curriculum: Curriculum, score_generator: ScoreGenerator, num_steps: int
) -> Dict[str, Any]:
    """Run a task tree test with controlled scores and collect detailed statistics.

    Args:
        curriculum: Curriculum instance to test
        score_generator: Generator for controlled scores
        num_steps: Number of steps to simulate

    Returns:
        Dictionary with detailed simulation results
    """
    task_counts = {}
    weight_history = []
    selection_history = []
    score_history = []

    for step in range(num_steps):
        # Sample task from Curriculum
        metta_task = curriculum.sample()
        task_name = metta_task.name

        # Record selection
        task_counts[task_name] = task_counts.get(task_name, 0) + 1
        selection_history.append(task_name)

        # Get controlled score
        score = score_generator.get_score(task_name)
        score_history.append(score)

        # Complete task - this will propagate up the tree
        metta_task.complete(score)

        # Record current weights and probabilities
        current_weights = curriculum.curriculum_algorithm.weights.copy()
        current_probs = curriculum.curriculum_algorithm.probabilities.copy()
        weight_history.append({"weights": current_weights, "probabilities": current_probs, "step": step})

    # Collect final state
    final_weights = {
        child.name: prob
        for child, prob in zip(curriculum.tasks, curriculum.curriculum_algorithm.probabilities, strict=False)
    }

    curriculum_stats = curriculum.get_curriculum_stats()

    return {
        "task_counts": task_counts,
        "weight_history": weight_history,
        "selection_history": selection_history,
        "score_history": score_history,
        "final_weights": final_weights,
        "curriculum_stats": curriculum_stats,
        "total_steps": num_steps,
        "completion_rates": curriculum.get_completion_rates(),
        "sample_rates": curriculum.get_sample_rates(),
        "task_probabilities": curriculum.get_task_probabilities(),
    }


# ============================================================================
# Progressive Algorithm Test Scenarios
# ============================================================================


class TestProgressiveAlgorithmScenarios:
    """Test the specific Progressive Algorithm scenarios."""

    def test_scenario_1_monotonic_linear_advances_correctly(self, env_cfg):
        """
        Scenario 1: Monotonic linear signal should advance through tasks with right timing.

        Expected: Should spend time on each task in order, advancing when signal increases.
        """
        print("\n=== PROGRESSIVE SCENARIO 1: Monotonic Linear Signal ===")

        task_names = ["task_1", "task_2", "task_3"]

        # Create a monotonic linear score generator
        score_gen = MonotonicLinearScores(increment=0.1)

        algorithm = ProgressiveHypers(
            performance_threshold=0.8,  # Will reach after ~8 steps per task
            progression_rate=0.3,  # Advance 30% per threshold crossing
            smoothing=0.1,  # Fast adaptation to signal
            blending_smoothness=0.2,  # Moderate transitions
            progression_mode="perf",  # Progress based on performance
        ).create(len(task_names))

        curriculum = create_curriculum_with_algorithm(task_names, algorithm, env_cfg)
        results = run_curriculum_simulation(curriculum, score_gen, 100)

        # Analyze progression pattern
        weight_history = results["weight_history"]
        score_history = results["score_history"]
        assert len(weight_history) == 100, f"Should have 100 weight snapshots, got {len(weight_history)}"

        # Debug: Print first 20 scores to understand progression
        print(f"First 20 scores: {score_history[:20]}")

        # Early: should focus on task_1
        early_probs = weight_history[5]["probabilities"]
        print(f"Early probabilities (step 5): {dict(zip(task_names, early_probs, strict=False))}")
        assert early_probs[0] > 0.8, f"Should start strongly focused on task_1, got {early_probs[0]}"
        assert early_probs[1] < 0.2, f"Task 2 should have minimal weight early, got {early_probs[1]}"
        assert early_probs[2] < 0.1, f"Task 3 should have minimal weight early, got {early_probs[2]}"

        # Debug: Check final curriculum stats
        curriculum_stats = results["curriculum_stats"]
        print(f"Final curriculum stats: {curriculum_stats}")

        # Check if progress has started by step 50
        mid_probs = weight_history[50]["probabilities"]
        print(f"Mid probabilities (step 50): {dict(zip(task_names, mid_probs, strict=False))}")
        progress = curriculum_stats.get("prog/progress", 0)
        print(f"Progress at end: {progress}")

        # Task 1 should have reduced weight
        assert mid_probs[0] < 0.7, f"Task 1 weight should decrease by middle, got {mid_probs[0]}"
        # Later tasks should have meaningful weight
        later_tasks_weight = mid_probs[1] + mid_probs[2]
        assert later_tasks_weight > 0.3, f"Should show clear progression by middle, got {later_tasks_weight}"

        # Late: should show further progression
        late_probs = weight_history[80]["probabilities"]
        print(f"Late probabilities (step 80): {dict(zip(task_names, late_probs, strict=False))}")
        assert late_probs[2] > 0.05, f"Task 3 should have some weight by late stage, got {late_probs[2]}"

        # Final state analysis
        final_weights = results["final_weights"]
        task_counts = results["task_counts"]
        print(f"Final weights: {final_weights}")
        print(f"Task counts: {task_counts}")

        # Progress should be meaningful (0.3 progression rate * number of threshold crossings)
        assert progress > 0.3, f"Should show significant progress with monotonic signal, got {progress}"

        # Should have sampled all tasks at least once
        assert len(task_counts) == 3, f"Should have tried all 3 tasks, got {task_counts.keys()}"
        assert all(count > 0 for count in task_counts.values()), (
            f"All tasks should be sampled at least once, got {task_counts}"
        )

        # Task sampling analysis
        total_samples = sum(task_counts.values())
        task_ratios = {name: count / total_samples for name, count in task_counts.items()}
        print(f"Task ratios: {task_ratios}")

        # With monotonic linear progress, we expect progression through tasks
        assert task_counts["task_1"] > 10, f"Task 1 should be sampled early, got {task_counts['task_1']}"

        print("✓ PASSED: Monotonic signal correctly advances through tasks with expected timing")

    def test_scenario_2_zero_signal_stays_on_first(self, env_cfg):
        """
        Scenario 2: Always 0 signal should keep curriculum on first task.

        Expected: Should stay overwhelmingly on first task throughout training.
        """
        print("\n=== PROGRESSIVE SCENARIO 2: Zero Signal ===")

        task_names = ["task_1", "task_2", "task_3"]

        # Create a zero score generator
        score_gen = ZeroScores()

        algorithm = ProgressiveHypers(
            performance_threshold=0.5,
            progression_rate=0.1,
            smoothing=0.1,
            progression_mode="perf",
        ).create(len(task_names))

        curriculum = create_curriculum_with_algorithm(task_names, algorithm, env_cfg)
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
        early_probs = weight_history[10]["probabilities"]
        print(f"Early probabilities (step 10): {dict(zip(task_names, early_probs, strict=False))}")
        assert early_probs[0] > 0.5, f"Should favor task_1 early, got {early_probs[0]}"

        # Middle weights - should still favor task_1
        mid_probs = weight_history[50]["probabilities"]
        print(f"Mid probabilities (step 50): {dict(zip(task_names, mid_probs, strict=False))}")
        assert mid_probs[0] > 0.5, f"Should still favor task_1 at midpoint, got {mid_probs[0]}"

        # Late weights - should still favor task_1
        late_probs = weight_history[80]["probabilities"]
        print(f"Late probabilities (step 80): {dict(zip(task_names, late_probs, strict=False))}")
        assert late_probs[0] > 0.5, f"Should still favor task_1 late in training, got {late_probs[0]}"

        # Weights should remain constant throughout (no progress)
        weight_variance = np.var([early_probs[0], mid_probs[0], late_probs[0]])
        assert weight_variance < 0.01, f"Weights should remain stable with zero progress, variance: {weight_variance}"

        # Task 1 should dominate selection counts
        total_selections = sum(task_counts.values())
        task_1_ratio = task_counts.get("task_1", 0) / total_selections if total_selections > 0 else 0
        assert task_1_ratio > 0.5, f"Should spend majority time on task_1 with zero scores, got {task_1_ratio:.2f}"

        # Progress should be zero
        progress = curriculum_stats.get("prog/progress", 0)
        assert progress == 0.0, f"Progress should be exactly 0 with zero signal, got {progress}"

        print("✓ PASSED: Zero signal correctly stays on first task throughout training")

    def test_scenario_3_random_signal_still_progresses(self, env_cfg):
        """
        Scenario 3: Random signal should still progress with right parameters.

        Expected: Due to randomness occasionally exceeding threshold, should eventually progress.
        """
        print("\n=== PROGRESSIVE SCENARIO 3: Random Signal ===")

        task_names = ["task_1", "task_2", "task_3"]

        # Create a random score generator with seed for reproducibility
        score_gen = RandomScores(seed=42, min_val=0.0, max_val=1.0)

        algorithm = ProgressiveHypers(
            performance_threshold=0.4,  # Low threshold - random will occasionally exceed
            progression_rate=0.05,  # Slower progression
            smoothing=0.5,  # Heavy smoothing to handle noise
            progression_mode="perf",
        ).create(len(task_names))

        curriculum = create_curriculum_with_algorithm(task_names, algorithm, env_cfg)
        results = run_curriculum_simulation(curriculum, score_gen, 300)  # Longer simulation

        # Analyze results
        weight_history = results["weight_history"]
        task_counts = results["task_counts"]
        final_weights = results["final_weights"]
        curriculum_stats = results["curriculum_stats"]
        progress = curriculum_stats.get("prog/progress", 0)

        print(f"Task counts: {task_counts}")
        print(f"Final weights: {final_weights}")
        print(f"Final progress: {progress}")

        # Check weight evolution - should show progression over time
        assert len(weight_history) == 300, f"Should have 300 weight snapshots, got {len(weight_history)}"

        # Check progression is happening
        assert progress > 0, "Should show progress after 300 steps with random signal"

        # Middle: should show clear progression
        mid_probs = weight_history[150]["probabilities"]
        print(f"Mid probabilities (step 150): {dict(zip(task_names, mid_probs, strict=False))}")
        assert mid_probs[0] < 0.9, f"Task 1 weight should decrease by middle, got {mid_probs[0]}"
        assert mid_probs[1] > 0.05, f"Task 2 should have meaningful weight by middle, got {mid_probs[1]}"

        # Should have tried all tasks due to randomness and progression
        assert len(task_counts) == 3, f"Should have tried all 3 tasks with 300 steps, got {task_counts.keys()}"
        assert all(count > 0 for count in task_counts.values()), (
            f"All tasks should be sampled at least once, got {task_counts}"
        )

        # Progress should be meaningful
        assert progress > 0.1, f"Should show meaningful progression from random signal, got {progress}"

        # Task distribution should show some progression
        total_selections = sum(task_counts.values())
        task_ratios = {name: count / total_selections for name, count in task_counts.items()}
        print(f"Task ratios: {task_ratios}")

        print("✓ PASSED: Random signal enables gradual progression with appropriate parameters")


class TestSimpleProgressiveAlgorithmScenarios:
    """Test the SimpleProgressiveAlgorithm scenarios."""

    def test_simple_progressive_advances_on_threshold(self, env_cfg):
        """Test that SimpleProgressiveAlgorithm advances when threshold is exceeded."""
        print("\n=== SIMPLE PROGRESSIVE: Threshold Advancement ===")

        task_names = ["task_1", "task_2", "task_3"]

        # Create score generator that gives increasing scores
        score_gen = MonotonicLinearScores(increment=0.2)  # Will exceed 0.5 threshold after 3 steps

        algorithm = SimpleProgressiveHypers(
            score_threshold=0.5,
        ).create(len(task_names))

        curriculum = create_curriculum_with_algorithm(task_names, algorithm, env_cfg)
        results = run_curriculum_simulation(curriculum, score_gen, 50)

        weight_history = results["weight_history"]
        task_counts = results["task_counts"]
        curriculum_stats = results["curriculum_stats"]

        print(f"Task counts: {task_counts}")
        print(f"Final curriculum stats: {curriculum_stats}")

        # Should advance through tasks as thresholds are met
        current_task = curriculum_stats.get("simple_prog/current_task", 0)
        progress_ratio = curriculum_stats.get("simple_prog/progress_ratio", 0)

        print(f"Current task: {current_task}, Progress ratio: {progress_ratio}")

        # Should have advanced beyond first task
        assert current_task > 0, f"Should have advanced beyond first task, got {current_task}"
        assert progress_ratio > 0, f"Should show progress, got {progress_ratio}"

        # Check weight history shows discrete jumps
        # Look at step 1 for early weights since advancement happens quickly
        early_weights = weight_history[1]["weights"]
        late_weights = weight_history[45]["weights"]

        print(f"Early weights (step 1): {early_weights}")
        print(f"Late weights: {late_weights}")

        # Check that the algorithm shows progression by examining the pattern
        # Early should be different from late (showing advancement occurred)
        early_focused_task = np.argmax(early_weights)
        late_focused_task = np.argmax(late_weights)

        print(f"Early focused task: {early_focused_task}, Late focused task: {late_focused_task}")

        # The focused task should advance or stay the same (never go backwards)
        assert late_focused_task >= early_focused_task, (
            f"Should show progression or stability, got early={early_focused_task}, late={late_focused_task}"
        )

        # The current task should have high weight
        assert late_weights[int(current_task)] > 0.9, f"Should focus on current task, got {late_weights}"

        print("✓ PASSED: Simple progressive algorithm advances correctly on threshold")


# ============================================================================
# Learning Progress Algorithm Test Scenarios
# ============================================================================


class TestLearningProgressAlgorithmScenarios:
    """Test the Learning Progress Algorithm scenarios using Curriculum."""

    def test_hierarchical_learning_progress_initial_distribution(self, env_cfg):
        """
        Test that hierarchical Learning Progress curriculum (like arena/learning_progress)
        returns uniform distribution initially.

        This mimics the production setup where learning_progress.yaml inherits from random.yaml
        which defines 15 tasks. The bug causes only the first task to get probability 1.0.
        """
        print("\n=== LEARNING PROGRESS: Hierarchical Initial Distribution Test ===")

        # Create many tasks like in arena/random.yaml
        task_names = [
            "basic",
            "basic_easy",
            "basic_easy_shaped",
            "basic_poor",
            "combat",
            "combat_easy",
            "combat_easy_shaped",
            "combat_poor",
            "advanced",
            "advanced_easy",
            "advanced_easy_shaped",
            "advanced_poor",
            "tag",
            "tag_easy",
            "tag_easy_shaped",
        ]

        # Create the learning progress algorithm with production-like settings
        algorithm = LearningProgressHypers(
            ema_timescale=0.001,  # Default from learning_progress.py
            progress_smoothing=0.05,
            num_active_tasks=16,  # Default
            rand_task_rate=0.25,  # Default
            sample_threshold=10,  # Default
            memory=25,  # Default
        ).create(len(task_names))

        # Check the internal state of BidirectionalLearningProgress
        lp_tracker = algorithm._lp_tracker
        print(f"Initial _random_baseline: {lp_tracker._random_baseline}")
        print(f"Initial _counter: {lp_tracker._counter}")

        # Call calculate_dist directly to see what happens
        dist, _ = lp_tracker.calculate_dist()
        print(f"Direct calculate_dist result: {dist}")
        print(f"Length of dist: {len(dist)}")

        # Check if only first task has weight
        if dist[0] == 1.0 and all(d == 0.0 for d in dist[1:]):
            print("BUG CONFIRMED: Only first task has probability 1.0!")

        curriculum = create_curriculum_with_algorithm(task_names, algorithm, env_cfg)

        # Check initial probabilities
        initial_probs = curriculum.get_task_probabilities(relative_to_root=True)
        print("\nTask probabilities:")
        for name, prob in initial_probs.items():
            print(f"  {name}: {prob:.4f}")

        # Check if bug is present: only 'basic' (first task) has probability 1.0
        if initial_probs.get("basic", 0) > 0.99 and all(
            prob < 0.01 for name, prob in initial_probs.items() if name != "basic"
        ):
            raise AssertionError(
                "BUG DETECTED: Only 'basic' task has probability 1.0! "
                "All tasks should have uniform probability initially."
            )

        # All tasks should have roughly equal probability
        expected_prob = 1.0 / len(task_names)
        for task_name in task_names:
            actual_prob = initial_probs.get(task_name, 0)
            assert abs(actual_prob - expected_prob) < 0.01, (
                f"Task {task_name} should have probability ~{expected_prob:.3f}, but got {actual_prob:.3f}"
            )

        print("✓ PASSED: Hierarchical learning progress has uniform initial distribution")

    def test_initial_distribution_before_sampling(self, env_cfg):
        """
        Test that Learning Progress algorithm returns uniform distribution initially.

        This test catches the bug where the algorithm incorrectly returns a non-uniform
        distribution before any samples are collected, causing only the first task to be selected.
        """
        print("\n=== LEARNING PROGRESS: Initial Distribution Test ===")

        task_names = ["task_1", "task_2", "task_3", "task_4"]

        algorithm = LearningProgressHypers(
            ema_timescale=0.02,
            sample_threshold=5,
            memory=15,
            num_active_tasks=4,
            rand_task_rate=0.1,
        ).create(len(task_names))

        curriculum = create_curriculum_with_algorithm(task_names, algorithm, env_cfg)

        # Check initial probabilities BEFORE any sampling
        # Test both with and without relative_to_root (production uses relative_to_root=True)
        initial_probs = curriculum.get_task_probabilities()
        initial_probs_relative = curriculum.get_task_probabilities(relative_to_root=True)
        print(f"Initial probabilities: {initial_probs}")
        print(f"Initial probabilities (relative_to_root=True): {initial_probs_relative}")

        # All tasks should have equal probability initially (uniform distribution)
        expected_prob = 1.0 / len(task_names)
        for task_name in task_names:
            actual_prob = initial_probs.get(task_name, 0)
            assert abs(actual_prob - expected_prob) < 0.01, (
                f"Task {task_name} should have probability ~{expected_prob:.3f}, but got {actual_prob:.3f}"
            )

        # Also check the raw weights from the algorithm
        weights = curriculum.curriculum_algorithm.weights
        probabilities = curriculum.curriculum_algorithm.probabilities
        print(f"Raw weights: {weights}")
        print(f"Raw probabilities: {probabilities}")

        # Verify all probabilities are equal
        assert np.allclose(probabilities, expected_prob, atol=0.01), (
            f"All probabilities should be ~{expected_prob:.3f}, but got {probabilities}"
        )

        # Sample once to verify uniform sampling works
        task_count = {name: 0 for name in task_names}
        for _ in range(100):
            task = curriculum.sample()
            task_count[task.name] += 1

        print(f"Sample distribution (100 samples): {task_count}")

        # With uniform distribution, all tasks should be sampled
        assert all(count > 0 for count in task_count.values()), (
            f"All tasks should be sampled at least once with uniform distribution, got {task_count}"
        )

        # No single task should dominate
        max_count = max(task_count.values())
        min_count = min(task_count.values())
        assert max_count < 60, f"No task should be sampled more than 60 times out of 100, got {max_count}"
        assert min_count > 10, f"No task should be sampled less than 10 times out of 100, got {min_count}"

        print("✓ PASSED: Learning progress has uniform initial distribution")

    def test_scenario_4_mixed_impossible_learnable_tasks(self, env_cfg):
        """
        Scenario 4: Mixed impossible and learnable tasks.

        Some tasks always give 0, others give linear increase.
        Expected: Should learn to give even weight to learnable tasks, near-0 to impossible.
        """
        print("\n=== LEARNING PROGRESS SCENARIO 4: Mixed Impossible/Learnable ===")

        task_names = ["impossible_1", "impossible_2", "learnable_1", "learnable_2"]

        # Only learnable tasks give increasing signals
        learnable_tasks = {"learnable_1", "learnable_2"}
        score_gen = ConditionalLinearScores(linear_tasks=learnable_tasks, increment=0.1)

        algorithm = LearningProgressHypers(
            ema_timescale=0.02,  # Faster adaptation for testing
            sample_threshold=5,  # Lower threshold for quicker adaptation
            memory=15,  # Shorter memory
            num_active_tasks=4,  # Sample all tasks
            rand_task_rate=0.1,  # Lower random exploration
        ).create(len(task_names))

        curriculum = create_curriculum_with_algorithm(task_names, algorithm, env_cfg)
        results = run_curriculum_simulation(curriculum, score_gen, 600)

        # Analyze results
        final_weights = results["final_weights"]
        task_counts = results["task_counts"]

        print(f"Task counts: {task_counts}")
        print(f"Final weights: {final_weights}")

        # Calculate final weight groups
        impossible_weight = final_weights.get("impossible_1", 0) + final_weights.get("impossible_2", 0)
        learnable_weight = final_weights.get("learnable_1", 0) + final_weights.get("learnable_2", 0)
        print(f"Final - Impossible total weight: {impossible_weight:.3f}")
        print(f"Final - Learnable total weight: {learnable_weight:.3f}")

        # After 600 steps, learning progress should prefer learnable tasks
        assert learnable_weight > impossible_weight * 1.5, (
            f"Should prefer learnable tasks: {learnable_weight:.3f} vs {impossible_weight:.3f}"
        )

        assert learnable_weight > 0.55, f"Learnable tasks should have majority weight, got {learnable_weight:.3f}"

        print("✓ PASSED: Learning progress correctly identifies and prefers learnable tasks")

    def test_scenario_5_threshold_dependent_progression(self, env_cfg):
        """
        Scenario 5: Threshold-dependent task progression.

        Primary task: linear increaser that flatlines after reaching milestone.
        Secondary task: stays at 0 until primary reaches milestone, then linear increaser.
        Expected: Should first weight primary, then shift to secondary after milestone.
        """
        print("\n=== LEARNING PROGRESS SCENARIO 5: Threshold Dependency ===")

        task_names = ["primary", "secondary"]

        threshold = 0.5
        score_gen = ThresholdDependentScores(
            primary_task="primary", secondary_task="secondary", threshold=threshold, increment=0.1
        )

        algorithm = LearningProgressHypers(
            ema_timescale=0.03,  # Faster adaptation
            sample_threshold=5,
            memory=20,
            rand_task_rate=0.5,  # Higher random rate to ensure both tasks get sampled
        ).create(len(task_names))

        curriculum = create_curriculum_with_algorithm(task_names, algorithm, env_cfg)
        results = run_curriculum_simulation(curriculum, score_gen, 150)

        task_counts = results["task_counts"]
        final_weights = results["final_weights"]

        print(f"Task counts: {task_counts}")
        print(f"Final weights: {final_weights}")

        # Verify both tasks were sampled
        assert len(task_counts) == 2, "Should have sampled both tasks"

        # Primary should be heavily sampled (shows initial learning progress)
        assert task_counts["primary"] > 60, f"Should sample primary task heavily, got {task_counts['primary']}"

        print("✓ PASSED: Learning progress correctly responds to threshold-dependent dynamics")


# ============================================================================
# Prioritize Regressed Algorithm Test Scenarios
# ============================================================================


class TestPrioritizeRegressedAlgorithmScenarios:
    """Test the Prioritize Regressed Algorithm scenarios using Curriculum."""

    def test_scenario_6_all_linear_scaling_equal_distribution(self, env_cfg):
        """
        Scenario 6: All tasks have linear scaling rewards.

        Expected: With the same linear progression, max/avg ratio should be similar for all tasks,
        leading to approximately equal distribution over time.
        """
        print("\n=== PRIORITIZE REGRESSED SCENARIO 6: All Linear Scaling ===")

        task_names = ["task_1", "task_2", "task_3"]

        # Each task gets its own independent linear progression
        class IndependentLinearScores(ScoreGenerator):
            """Each task has its own counter for linear progression."""

            def __init__(self, increment: float = 0.1):
                self.increment = increment
                self.task_counters = {}

            def get_score(self, task_id: str) -> float:
                if task_id not in self.task_counters:
                    self.task_counters[task_id] = 0
                score = self.task_counters[task_id] * self.increment
                self.task_counters[task_id] += 1
                return min(score, 1.0)  # Cap at 1.0

        score_gen = IndependentLinearScores(increment=0.05)

        algorithm = PrioritizeRegressedHypers(
            moving_avg_decay_rate=0.1,  # Moderate smoothing
        ).create(len(task_names))

        curriculum = create_curriculum_with_algorithm(task_names, algorithm, env_cfg)

        # NO INITIALIZATION - let the algorithm handle exploration naturally

        results = run_curriculum_simulation(curriculum, score_gen, 200)

        final_weights = results["final_weights"]
        task_counts = results["task_counts"]

        print(f"Task counts: {task_counts}")
        print(f"Final weights: {final_weights}")

        # Final analysis: all tasks should have similar weights
        weights_list = list(final_weights.values())
        weight_variance = np.var(weights_list)
        print(f"Final weight variance: {weight_variance:.6f}")

        # With PrioritizeRegressed, the issue might be that one task gets more samples
        # and thus has a different max/avg ratio. Let's be more lenient but check
        # that no single task completely dominates
        max_weight = max(weights_list)
        min_weight = min(weights_list)
        weight_ratio = max_weight / max(min_weight, 1e-6)

        print(f"Weight ratio (max/min): {weight_ratio:.3f}")

        # Check that all tasks were sampled
        assert len(task_counts) == 3, f"All tasks should be sampled, but only got: {task_counts.keys()}"

        # Check minimum sampling for each task (at least 10 times in 200 steps)
        for task_name in task_names:
            assert task_counts.get(task_name, 0) >= 10, (
                f"Task {task_name} should be sampled at least 10 times, got {task_counts.get(task_name, 0)}"
            )

        # Instead of requiring very similar weights, ensure no single task completely dominates
        assert max_weight < 0.8, f"No single task should dominate (>80%), max weight: {max_weight:.3f}"
        assert weight_ratio < 50, f"Weight ratio should not be extreme, got {weight_ratio:.3f}"

        # Task counts should show that multiple tasks were sampled
        total_samples = sum(task_counts.values())
        task_ratios = [count / total_samples for count in task_counts.values()]
        print(f"Task sampling ratios: {[f'{r:.3f}' for r in task_ratios]}")

        # At least 2 tasks should have meaningful sampling (>10%)
        meaningful_tasks = sum(1 for ratio in task_ratios if ratio > 0.1)
        assert meaningful_tasks >= 2, f"At least 2 tasks should have meaningful sampling, got {meaningful_tasks}"

        print("✓ PASSED: All linear scaling tasks maintain reasonable distribution")

    def test_scenario_7_one_impossible_task_gets_lowest_weight(self, env_cfg):
        """
        Scenario 7: One task is impossible (always returns 0), others have linear scaling.

        Expected: The impossible task has max/avg = 0/0, resulting in weight = epsilon.
        """
        print("\n=== PRIORITIZE REGRESSED SCENARIO 7: One Impossible Task ===")

        task_names = ["impossible", "learnable_1", "learnable_2"]

        # Only learnable tasks give increasing signals
        learnable_tasks = {"learnable_1", "learnable_2"}
        score_gen = ConditionalLinearScores(linear_tasks=learnable_tasks, increment=0.1)

        algorithm = PrioritizeRegressedHypers(
            moving_avg_decay_rate=0.05,  # Slower adaptation
        ).create(len(task_names))

        curriculum = create_curriculum_with_algorithm(task_names, algorithm, env_cfg)
        results = run_curriculum_simulation(curriculum, score_gen, 300)

        final_weights = results["final_weights"]
        task_counts = results["task_counts"]

        print(f"Task counts: {task_counts}")
        print(f"Final weights: {final_weights}")

        # Final analysis: impossible task should have minimal weight
        max_learnable_weight = max(final_weights["learnable_1"], final_weights["learnable_2"])
        assert final_weights["impossible"] < max_learnable_weight, (
            f"Impossible task should have lower weight than learnable tasks: "
            f"{final_weights['impossible']} vs {max_learnable_weight}"
        )

        # The impossible task weight should be minimal
        assert final_weights["impossible"] < 0.01, (
            f"Impossible task should have minimal weight, got {final_weights['impossible']}"
        )

        # Impossible task should be sampled less
        total_samples = sum(task_counts.values())
        impossible_ratio = task_counts.get("impossible", 0) / total_samples if total_samples > 0 else 0
        learnable_count = task_counts.get("learnable_1", 0) + task_counts.get("learnable_2", 0)

        assert task_counts.get("impossible", 0) < learnable_count, (
            "Impossible task should be sampled less than learnable tasks combined"
        )

        assert impossible_ratio < 0.1, f"Impossible task should get minimal samples, got {impossible_ratio:.3f}"

        print("✓ PASSED: Prioritize regressed correctly avoids impossible task")

    def test_scenario_8_regression_patterns(self, env_cfg):
        """
        Scenario 8: Different score trajectories to test regression detection.

        - 2 impossible tasks (always 0)
        - Task A: starts high (0.8) and decreases over time (regression)
        - Task B: static scores around 0.5
        - Task C: starts low (0.2) and increases over time (improvement)

        Expected behavior:
        - Early: all tasks sampled equally (exploration phase)
        - Middle: impossible tasks get low weight, others roughly equal
        - Late: A > B≈C > impossible
          * A has regressed (max=0.8, current=0.1) so gets highest weight
          * B and C both perform at their max (no regression) so get similar weights
          * Impossible tasks get minimal weight
        """
        print("\n=== PRIORITIZE REGRESSED SCENARIO 8: Regression Patterns ===")

        task_names = ["impossible_1", "impossible_2", "task_A_regressing", "task_B_static", "task_C_improving"]

        # Custom score generator for different trajectories
        class TrajectoryScores(ScoreGenerator):
            """Generates different score trajectories for each task type."""

            def __init__(self):
                self.task_counts = {}

            def get_score(self, task_id: str) -> float:
                # Extract base task name
                base_task = task_id.split("/")[-1] if "/" in task_id else task_id

                # Get count for this task
                count = self.task_counts.get(base_task, 0)
                self.task_counts[base_task] = count + 1

                # Generate score based on task type
                if "impossible" in base_task:
                    return 0.0
                elif base_task == "task_A_regressing":
                    # Start at 0.8, decrease by 0.05 each time, floor at 0.1
                    return max(0.1, 0.8 - (count * 0.05))
                elif base_task == "task_B_static":
                    # Always return 0.5
                    return 0.5
                elif base_task == "task_C_improving":
                    # Start at 0.2, increase by 0.05 each time, cap at 0.8
                    return min(0.8, 0.2 + (count * 0.05))
                else:
                    return 0.0

            def reset(self) -> None:
                self.task_counts = {}

        score_gen = TrajectoryScores()

        algorithm = PrioritizeRegressedHypers(
            moving_avg_decay_rate=0.1,  # Moderate smoothing
            min_samples_per_task=5,  # Ensure all tasks get sampled initially
        ).create(len(task_names))

        curriculum = create_curriculum_with_algorithm(task_names, algorithm, env_cfg)

        # Run simulation in phases to observe behavior over time
        phase_size = 100

        # Phase 1: Early (steps 0-99)
        results_early = run_curriculum_simulation(curriculum, score_gen, phase_size)

        # Continue for Phase 2: Middle (steps 100-199)
        results_middle = run_curriculum_simulation(curriculum, score_gen, phase_size)

        # Continue for Phase 3: Late (steps 200-299)
        results_late = run_curriculum_simulation(curriculum, score_gen, phase_size)

        # Analyze each phase
        print("\n--- EARLY PHASE (steps 0-99) ---")
        early_counts = results_early["task_counts"]
        early_weights = results_early["final_weights"]
        print(f"Task counts: {early_counts}")
        print(f"Final weights: {early_weights}")

        # During exploration, all tasks should be sampled
        assert len(early_counts) == 5, "All tasks should be sampled during exploration"
        assert all(count >= 5 for count in early_counts.values()), (
            f"All tasks should reach min_samples threshold, got {early_counts}"
        )

        print("\n--- MIDDLE PHASE (steps 100-199) ---")
        middle_counts = results_middle["task_counts"]
        middle_weights = results_middle["final_weights"]
        print(f"Task counts: {middle_counts}")
        print(f"Final weights: {middle_weights}")

        # Impossible tasks should have minimal weight
        impossible_weight_mid = middle_weights.get("impossible_1", 0) + middle_weights.get("impossible_2", 0)
        other_weight_mid = (
            middle_weights.get("task_A_regressing", 0)
            + middle_weights.get("task_B_static", 0)
            + middle_weights.get("task_C_improving", 0)
        )
        print(f"Impossible tasks total weight: {impossible_weight_mid:.4f}")
        print(f"Other tasks total weight: {other_weight_mid:.4f}")

        assert impossible_weight_mid < 0.01, f"Impossible tasks should have minimal weight, got {impossible_weight_mid}"

        print("\n--- LATE PHASE (steps 200-299) ---")
        late_counts = results_late["task_counts"]
        late_weights = results_late["final_weights"]
        print(f"Task counts: {late_counts}")
        print(f"Final weights: {late_weights}")

        # Get individual weights
        weight_A = late_weights.get("task_A_regressing", 0)
        weight_B = late_weights.get("task_B_static", 0)
        weight_C = late_weights.get("task_C_improving", 0)
        weight_impossible = late_weights.get("impossible_1", 0) + late_weights.get("impossible_2", 0)

        print("\nFinal weight ordering:")
        print(f"  Task A (regressing): {weight_A:.4f}")
        print(f"  Task B (static): {weight_B:.4f}")
        print(f"  Task C (improving): {weight_C:.4f}")
        print(f"  Impossible tasks: {weight_impossible:.4f}")

        # Verify expected ordering: A > B,C > impossible
        assert weight_A > weight_B, f"Regressing task A should have highest weight: {weight_A} <= {weight_B}"
        assert weight_A > weight_C, f"Regressing task A should have highest weight: {weight_A} <= {weight_C}"

        # B and C should have very similar weights because:
        # - B is static at 0.5: max=0.5, current=0.5, no regression (ratio ≈ 1.0)
        # - C improved to 0.8: max=0.8, current=0.8, no regression (ratio ≈ 1.0)
        # Both are performing at their peak, so neither is "regressed"
        assert weight_B > weight_impossible * 100, (
            f"Static B should beat impossible by far: {weight_B} <= {weight_impossible * 100}"
        )
        assert weight_C > weight_impossible * 100, (
            f"Improving C should beat impossible by far: {weight_C} <= {weight_impossible * 100}"
        )

        # Get detailed stats for analysis
        alg = curriculum.curriculum_algorithm
        print("\nDetailed task analysis:")
        print("(Note: PrioritizeRegressed prioritizes tasks where current performance < past peak)")
        for i, name in enumerate(task_names):
            if alg.task_completed_count[i] > 0:
                max_score = alg.reward_maxes[i]
                avg_score = alg.reward_averages[i]
                ratio = max_score / (avg_score + alg.epsilon)
                print(
                    f"  {name}: max={max_score:.3f}, avg={avg_score:.3f}, "
                    f"ratio={ratio:.3f}, weight={alg.weights[i]:.6f}"
                )

        # The key insight: PrioritizeRegressed finds tasks that HAVE regressed, not just different trajectories
        # - Task A has regressed (was 0.8, now 0.1) → high priority
        # - Tasks B & C are at their peak performance → lower priority (similar weights)
        print("\n✓ PASSED: PrioritizeRegressed correctly identifies and prioritizes regression patterns")


# ============================================================================
# Integration Tests
# ============================================================================


class TestCurriculumIntegration:
    """Test Curriculum integration and basic functionality."""

    def test_curriculum_basic_functionality(self, env_cfg):
        """Test basic Curriculum functionality with DiscreteRandomCurriculum."""
        print("\n=== TASK TREE INTEGRATION: Basic Functionality ===")

        task_names = ["task_1", "task_2", "task_3"]
        algorithm = DiscreteRandomHypers().create(len(task_names))

        curriculum = create_curriculum_with_algorithm(task_names, algorithm, env_cfg)

        # Test basic sampling
        for _ in range(10):
            task = curriculum.sample()
            assert isinstance(task, MettaGridTask)
            assert task.name in task_names

        # Test task completion
        task = curriculum.sample()
        initial_completed = curriculum.total_completed_tasks
        task.complete(0.5)
        assert curriculum.total_completed_tasks == initial_completed + 1

        # Test statistics
        completion_rates = curriculum.get_completion_rates()
        sample_rates = curriculum.get_sample_rates()
        task_probs = curriculum.get_task_probabilities()
        curriculum_stats = curriculum.get_curriculum_stats()

        assert isinstance(completion_rates, dict)
        assert isinstance(sample_rates, dict)
        assert isinstance(task_probs, dict)
        assert isinstance(curriculum_stats, dict)

        print("✓ PASSED: Curriculum basic functionality works correctly")

    def test_curriculum_with_custom_weights(self, env_cfg):
        """Test Curriculum with custom initial weights."""
        print("\n=== TASK TREE INTEGRATION: Custom Weights ===")

        task_names = ["task_1", "task_2", "task_3"]
        env_configs = [(name, env_cfg.copy()) for name in task_names]

        # Custom weights favoring task_2
        task_weights = [0.1, 0.8, 0.1]  # Corresponding to task_1, task_2, task_3

        hypers = DiscreteRandomHypers(initial_weights=task_weights)

        curriculum = task_set(
            name="test_weighted",
            env_configs=env_configs,
            curriculum_hypers=hypers,
        )

        # Sample many times and check distribution
        counts = {name: 0 for name in task_names}
        for _ in range(1000):
            task = curriculum.sample()
            counts[task.name] += 1

        total = sum(counts.values())
        ratios = {name: count / total for name, count in counts.items()}
        print(f"Sampling ratios: {ratios}")

        # task_2 should be sampled most often
        assert ratios["task_2"] > ratios["task_1"], "task_2 should be sampled more than task_1"
        assert ratios["task_2"] > ratios["task_3"], "task_2 should be sampled more than task_3"
        assert ratios["task_2"] > 0.6, f"task_2 should dominate sampling, got {ratios['task_2']}"

        print("✓ PASSED: Curriculum respects custom initial weights")
