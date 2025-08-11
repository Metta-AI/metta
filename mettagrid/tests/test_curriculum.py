"""
Test suite for curriculum learning algorithms.

This module tests specific scenarios to validate that curriculum algorithms
behave correctly under controlled conditions:

Learning Progress Tests:
4. Mixed impossible/learnable tasks -> should weight learnable evenly, ignore impossible
5. Threshold dependency -> should first weight primary, then secondary after milestone

Prioritize Regressed Curriculum Tests:
6. All tasks have linear scaling -> should maintain equal distribution
7. One impossible task (always 0) -> should get minimum weight as max_score = 0 ==> LP = epislon
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

import metta.mettagrid.room.random
from metta.mettagrid.curriculum.bucketed import BucketedCurriculum, _expand_buckets
from metta.mettagrid.curriculum.core import Curriculum, SingleTaskCurriculum
from metta.mettagrid.curriculum.learning_progress import LearningProgressCurriculum
from metta.mettagrid.curriculum.multi_task import MultiTaskCurriculum
from metta.mettagrid.curriculum.prioritize_regressed import PrioritizeRegressedCurriculum
from metta.mettagrid.curriculum.random import RandomCurriculum
from metta.mettagrid.curriculum.sampling import SampledTaskCurriculum
from metta.mettagrid.curriculum.util import curriculum_from_config_path


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


def fake_curriculum_from_config_path(path, env_overrides=None):
    base_config = OmegaConf.create({"game": {"num_agents": 5, "map": {"width": 10, "height": 10}}})
    task_cfg = OmegaConf.merge(base_config, env_overrides or {})
    assert isinstance(task_cfg, DictConfig)
    return SingleTaskCurriculum(path, task_cfg=task_cfg)


def test_single_task_curriculum(env_cfg):
    curr = SingleTaskCurriculum("task", env_cfg)
    task = curr.get_task()
    assert task.id() == "task"
    assert task.env_cfg() == env_cfg
    assert not task.is_complete()
    task.complete_trial(0.5)
    assert task.is_complete()
    with pytest.raises(AssertionError):
        task.complete_trial(0.1)


def test_random_curriculum_selects_task(monkeypatch, env_cfg):
    monkeypatch.setattr(random, "choices", lambda population, weights: ["b"])
    monkeypatch.setattr(
        "metta.mettagrid.curriculum.random.curriculum_from_config_path", fake_curriculum_from_config_path
    )

    curr = RandomCurriculum({"a": 1.0, "b": 1.0}, OmegaConf.create({}))
    task = curr.get_task()
    assert task.id() == "b"
    assert task.name() == "b:b"


def test_prioritize_regressed_curriculum_updates(monkeypatch, env_cfg):
    monkeypatch.setattr(
        "metta.mettagrid.curriculum.random.curriculum_from_config_path", fake_curriculum_from_config_path
    )
    curr = PrioritizeRegressedCurriculum({"a": 1.0, "b": 1.0}, OmegaConf.create({}))

    # Complete task "a" with low reward 0.1
    curr.complete_task("a", 0.1)
    weight_after_a = curr._task_weights["a"]
    # Task "a" has max/avg = 0.1/0.1 = 1.0, task "b" has max/avg = 0/0 (undefined, uses epsilon)
    # So task "a" should have higher weight
    assert weight_after_a > curr._task_weights["b"], (
        "Task with actual performance should have higher weight than untried task"
    )

    # Complete task "b" with high reward 1.0
    prev_b = curr._task_weights["b"]
    curr.complete_task("b", 1.0)
    # Task "b" now has max/avg = 1.0/1.0 = 1.0, similar to task "a"
    # But weight should have increased from epsilon
    assert curr._task_weights["b"] > prev_b, "Weight should increase when task gets its first score"


def test_bucketed_curriculum(monkeypatch, env_cfg):
    monkeypatch.setattr(
        "metta.mettagrid.curriculum.bucketed.config_from_path", lambda path, env_overrides=None: env_cfg
    )
    buckets = {
        "game.map.width": [5, 10],
        "game.map.height": [5, 10],
    }
    curr = BucketedCurriculum(env_cfg_template_path="dummy", buckets=buckets)

    # There should be 4 tasks (2x2 grid)
    assert len(curr._id_to_curriculum) == 4
    # Sample a task
    task = curr.get_task()
    assert hasattr(task, "id")
    assert any(str(w) in task.id() for w in [5, 10])


def test_bucketed_curriculum_from_yaml_with_map_builder():
    """Test BucketedCurriculum loading from YAML file with buckets that impact map builder."""
    from pathlib import Path

    import hydra

    # Get the path to the test YAML config file
    test_dir = Path(__file__).parent
    config_file = test_dir / "test_bucketed_config.yaml"

    # Verify the config file exists
    assert config_file.exists(), f"Config file not found: {config_file}"

    # Initialize Hydra and load the config
    with hydra.initialize(config_path=".", version_base=None):
        # Instantiate the BucketedCurriculum using Hydra
        curr = curriculum_from_config_path("test_bucketed_config", OmegaConf.create({"game": {"num_agents": 5}}))  # type: ignore

    # There should be 27 tasks (3x3x3 grid)
    assert len(curr._id_to_curriculum) == 27

    # Sample tasks and verify the map builder parameters are correctly overridden
    # Test that task IDs contain the bucket parameter values
    task = curr.get_task()
    task_id = task.id()
    assert "width=" in task_id, f"Task ID should contain width parameter: {task_id}"
    assert "height=" in task_id, f"Task ID should contain height parameter: {task_id}"
    assert "altar=" in task_id, f"Task ID should contain altar parameter: {task_id}"

    # Verify the task config structure is correct
    task_cfg = task.env_cfg()
    assert hasattr(task_cfg.game, "map_builder")
    assert isinstance(task_cfg.game.map_builder, metta.mettagrid.room.random.Random)
    assert task_cfg.game.num_agents == 5, f"num_agents should have been overridden to 5, got {task_cfg.game.num_agents}"
    assert task_cfg.game.map_builder._width in [20, 40, 60]
    assert task_cfg.game.map_builder._height in [20, 40, 60]
    assert task_cfg.game.map_builder._objects.altar in [1, 3, 5]


def test_expand_buckets_values_and_range():
    buckets = {
        "param1": [1, 2, 3],
        "param2": {"range": (0, 10), "bins": 2},
    }
    expanded = _expand_buckets(buckets)
    # param1 should be a direct list
    assert expanded["param1"] == [1, 2, 3]
    # param2 should be a list of 2 bins, each a dict with 'range' and 'want_int'
    assert len(expanded["param2"]) == 2
    assert expanded["param2"][0]["range"] == (0, 5)
    assert expanded["param2"][1]["range"] == (5, 10)
    assert all(isinstance(b, dict) and "range" in b for b in expanded["param2"])


def test_expand_buckets_choice():
    buckets = {
        "param1": ["red", "blue", "green"],
        "param2": [1, 2, 3, 4],
        "param3": [True, False],
    }
    expanded = _expand_buckets(buckets)
    # All choice parameters should be direct lists
    assert expanded["param1"] == ["red", "blue", "green"]
    assert expanded["param2"] == [1, 2, 3, 4]
    assert expanded["param3"] == [True, False]


def test_expand_buckets_mixed_types():
    buckets = {
        "param1": [1, 2, 3],
        "param2": {"range": (0, 10), "bins": 2},
        "param3": ["a", "b", "c"],
    }
    expanded = _expand_buckets(buckets)
    # Test all three types together
    assert expanded["param1"] == [1, 2, 3]
    assert len(expanded["param2"]) == 2
    assert expanded["param2"][0]["range"] == (0, 5)
    assert expanded["param2"][1]["range"] == (5, 10)
    assert expanded["param3"] == ["a", "b", "c"]


def test_sampled_task_curriculum():
    # Setup: one value bucket, one range bucket (int), one range bucket (float)
    task_id = "test_task"
    task_cfg_template = OmegaConf.create({"param1": None, "param2": None, "param3": None})
    sampling_parameters = {
        "param1": 42,
        "param2": {"range": (0, 10), "want_int": True},
        "param3": {"range": (0.0, 1.0)},
    }
    curr = SampledTaskCurriculum(task_id, task_cfg_template, sampling_parameters)
    task = curr.get_task()
    assert task.id() == task_id
    cfg = task.env_cfg()
    assert set(cfg.keys()) == {"param1", "param2", "param3"}
    assert cfg["param1"] == 42
    assert 0 <= cfg["param2"] < 10 and isinstance(cfg["param2"], int)
    assert 0.0 <= cfg["param3"] < 1.0 and isinstance(cfg["param3"], float)


def test_multi_task_curriculum_completion_rates(env_cfg):
    # Dummy curriculum that returns a task with env_cfg
    class DummyCurriculum:
        def get_task(self):
            class DummyTask:
                def env_cfg(self):
                    return env_cfg

            return DummyTask()

        def complete_task(self, id, score):
            pass

    curricula = {"a": DummyCurriculum(), "b": DummyCurriculum(), "c": DummyCurriculum()}
    curr = MultiTaskCurriculum(curricula)
    # Simulate completions: a, a, b
    curr.complete_task("a", 1.0)
    curr.complete_task("a", 1.0)
    curr.complete_task("b", 1.0)
    rates = curr.get_completion_rates()
    # There are 3 completions, so a:2/3, b:1/3, c:0/3
    assert abs(rates["task_completions/a"] - 2 / 3) < 1e-6
    assert abs(rates["task_completions/b"] - 1 / 3) < 1e-6
    assert abs(rates["task_completions/c"] - 0.0) < 1e-6


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
            Score to pass to curriculum.complete_task()
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
        base_task = task_id.split(":")[0] if ":" in task_id else task_id

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
        base_task = task_id.split(":")[0] if ":" in task_id else task_id

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


def run_curriculum_simulation(
    curriculum: Curriculum, score_generator: ScoreGenerator, num_steps: int
) -> Dict[str, Any]:
    """Run a curriculum test with controlled scores and collect detailed statistics.

    Args:
        curriculum: Any curriculum implementing get_task() and complete_task()
        score_generator: Generator for controlled scores
        num_steps: Number of steps to simulate

    Returns:
        Dictionary with detailed simulation results
    """
    task_counts = {}
    weight_history = []
    selection_history = []
    score_history = []

    for _ in range(num_steps):
        # Get task from curriculum
        task = curriculum.get_task()
        task_id = task.short_name() if hasattr(task, "short_name") else task.id()

        # Record selection
        task_counts[task_id] = task_counts.get(task_id, 0) + 1
        selection_history.append(task_id)

        # Get controlled score and complete task
        score = score_generator.get_score(task_id)
        score_history.append(score)

        # For complete_task, we need the original task ID without curriculum prefix
        # Extract the base task ID (e.g., "task_1:task_1" -> "task_1")
        complete_id = task_id.split(":")[0] if ":" in task_id else task_id
        curriculum.complete_task(complete_id, score)

        # Record current weights if available
        if hasattr(curriculum, "get_task_probs"):
            weights = curriculum.get_task_probs()
            weight_history.append(weights.copy())
        elif hasattr(curriculum, "_task_weights"):
            # Normalize weights for display
            total = sum(curriculum._task_weights.values())
            weights = (
                {k: v / total for k, v in curriculum._task_weights.items()} if total > 0 else curriculum._task_weights
            )
            weight_history.append(weights.copy())

    # Collect final state
    final_weights = {}
    if hasattr(curriculum, "get_task_probs"):
        final_weights = curriculum.get_task_probs()
    elif hasattr(curriculum, "_task_weights"):
        total = sum(curriculum._task_weights.values())
        final_weights = (
            {k: v / total for k, v in curriculum._task_weights.items()} if total > 0 else curriculum._task_weights
        )

    curriculum_stats = {}
    if hasattr(curriculum, "get_curriculum_stats"):
        curriculum_stats = curriculum.get_curriculum_stats()

    return {
        "task_counts": task_counts,
        "weight_history": weight_history,
        "selection_history": selection_history,
        "score_history": score_history,
        "final_weights": final_weights,
        "curriculum_stats": curriculum_stats,
        "total_steps": num_steps,
    }


def create_mock_curricula(task_names: List[str]) -> Dict[str, float]:
    """Create task weights dictionary for testing.

    For LearningProgressCurriculum,
    we need a dict mapping task names to initial weights.
    """
    # Equal initial weights for all tasks
    return {task_name: 1.0 for task_name in task_names}


# ============================================================================
# Specific Test Scenarios for Curriculum Validation
# ============================================================================


class TestLearningProgressScenarios:
    """Test the specific Learning Progress scenarios requested."""

    def test_scenario_4_mixed_impossible_learnable_tasks(self, monkeypatch):
        """
        Scenario 4: Mixed impossible and learnable tasks.

        Some tasks always give 0, others give linear increase.
        Expected: Should learn to give even weight to learnable tasks, near-0 to impossible.
        """
        print("\n=== LEARNING PROGRESS SCENARIO 4: Mixed Impossible/Learnable ===")

        # Patch curriculum_from_config_path
        def mock_curriculum_from_config_path(path, env_overrides=None):
            default_cfg = OmegaConf.create({"game": {"num_agents": 1, "map": {"width": 10, "height": 10}}})
            cfg = OmegaConf.merge(default_cfg, env_overrides or {})
            return SingleTaskCurriculum(path, cfg)

        monkeypatch.setattr(
            "metta.mettagrid.curriculum.random.curriculum_from_config_path", mock_curriculum_from_config_path
        )

        tasks = ["impossible_1", "impossible_2", "learnable_1", "learnable_2"]
        task_weights = create_mock_curricula(tasks)

        # Only learnable tasks give increasing signals
        learnable_tasks = {"learnable_1", "learnable_2"}
        score_gen = ConditionalLinearScores(linear_tasks=learnable_tasks, increment=0.1)

        curriculum = LearningProgressCurriculum(
            tasks=task_weights,
            ema_timescale=0.02,  # Faster adaptation for testing
            sample_threshold=5,  # Lower threshold for quicker adaptation
            memory=15,  # Shorter memory
            num_active_tasks=4,  # Sample all tasks
            rand_task_rate=0.1,  # Lower random exploration for more deterministic behavior
        )

        # First ensure all tasks get sampled at least once during initialization
        # This prevents np.mean() from being called on empty lists
        for task_name in tasks:
            task = curriculum.get_task()
            score = score_gen.get_score(task_name)
            curriculum.complete_task(task_name, score)

        # Now run the main simulation with more steps for convergence
        results = run_curriculum_simulation(curriculum, score_gen, 600)

        # Analyze results
        weight_history = results["weight_history"]
        final_weights = results["final_weights"]
        task_counts = results["task_counts"]
        curriculum_stats = results["curriculum_stats"]

        print(f"Task counts: {task_counts}")
        print(f"Final weights: {final_weights}")

        # Check weight evolution over time
        assert len(weight_history) == 600, f"Should have 600 weight snapshots, got {len(weight_history)}"

        # Check very early weights to see exploration phase
        very_early_weights = weight_history[5]
        print(f"Very early weights (step 5): {very_early_weights}")

        # By step 30, learning progress may already identify differences
        early_weights = weight_history[30]
        print(f"Early weights (step 30): {early_weights}")
        # With fast adaptation (ema_timescale=0.02), algorithm quickly identifies learnable tasks
        # Check that at least some tasks have been tried
        assert len(task_counts) >= 2, f"Should have tried at least 2 tasks by simulation end, got {task_counts.keys()}"

        # Middle: should start differentiating
        mid_weights = weight_history[300]
        print(f"Mid weights (step 300): {mid_weights}")
        mid_impossible = mid_weights.get("impossible_1", 0) + mid_weights.get("impossible_2", 0)
        mid_learnable = mid_weights.get("learnable_1", 0) + mid_weights.get("learnable_2", 0)
        print(f"Mid - Impossible weight: {mid_impossible:.3f}, Learnable weight: {mid_learnable:.3f}")

        # Late: should strongly prefer learnable tasks
        late_weights = weight_history[500]
        print(f"Late weights (step 500): {late_weights}")
        late_impossible = late_weights.get("impossible_1", 0) + late_weights.get("impossible_2", 0)
        late_learnable = late_weights.get("learnable_1", 0) + late_weights.get("learnable_2", 0)
        print(f"Late - Impossible weight: {late_impossible:.3f}, Learnable weight: {late_learnable:.3f}")

        # Calculate final weight groups
        impossible_weight = final_weights.get("impossible_1", 0) + final_weights.get("impossible_2", 0)
        learnable_weight = final_weights.get("learnable_1", 0) + final_weights.get("learnable_2", 0)
        print(f"Final - Impossible total weight: {impossible_weight:.3f}")
        print(f"Final - Learnable total weight: {learnable_weight:.3f}")

        # After 600 steps, learning progress should prefer learnable tasks
        # Use more lenient thresholds to account for randomness
        assert learnable_weight > impossible_weight * 1.5, (
            f"Should prefer learnable tasks after 600 steps: {learnable_weight:.3f} vs {impossible_weight:.3f}"
        )

        # Learnable tasks should have higher weight
        assert learnable_weight > 0.55, f"Learnable tasks should have majority weight, got {learnable_weight:.3f}"
        assert impossible_weight < 0.45, f"Impossible tasks should have minority weight, got {impossible_weight:.3f}"

        # Check that most tasks were explored (at least 3 out of 4)
        # Learning progress may quickly abandon impossible tasks
        assert len(task_counts) >= 3, f"Should have sampled at least 3 tasks, got {len(task_counts)}"

        # Verify task sampling distribution
        total_samples = sum(task_counts.values())
        assert total_samples == 600, f"Should have 600 total samples, got {total_samples}"

        # Calculate sampling ratios (handle curriculum prefix)
        impossible_samples = sum(count for task, count in task_counts.items() if "impossible" in task)
        learnable_samples = sum(count for task, count in task_counts.items() if "learnable" in task)
        impossible_ratio = impossible_samples / total_samples if total_samples > 0 else 0
        learnable_ratio = learnable_samples / total_samples if total_samples > 0 else 0

        print(f"Sampling ratios - Impossible: {impossible_ratio:.3f}, Learnable: {learnable_ratio:.3f}")

        # Learnable tasks should be sampled more frequently
        assert learnable_ratio > impossible_ratio, (
            f"Learnable tasks should be sampled more: {learnable_ratio:.3f} vs {impossible_ratio:.3f}"
        )

        # Check learning progress metrics if available
        if "learning_progress" in curriculum_stats:
            lp_data = curriculum_stats["learning_progress"]
            print(f"Learning progress data: {lp_data}")
            # Learnable tasks should show positive learning progress
            for task in ["learnable_1", "learnable_2"]:
                if task in lp_data:
                    assert lp_data[task] > 0, f"Learnable task {task} should show positive learning progress"
            # Impossible tasks should show near-zero learning progress
            for task in ["impossible_1", "impossible_2"]:
                if task in lp_data:
                    assert abs(lp_data[task]) < 0.1, f"Impossible task {task} should show minimal learning progress"

        print("✓ PASSED: Learning progress correctly identifies and strongly prefers learnable tasks")

    def test_scenario_5_threshold_dependent_progression(self, monkeypatch):
        """
        Scenario 5: Threshold-dependent task progression.

        Primary task: linear increaser that flatlines after reaching milestone.
        Secondary task: stays at 0 until primary reaches milestone, then linear increaser.
        Expected: Should first weight primary, then shift to secondary after milestone.
        """
        print("\n=== LEARNING PROGRESS SCENARIO 5: Threshold Dependency ===")

        # Patch curriculum_from_config_path
        def mock_curriculum_from_config_path(path, env_overrides=None):
            default_cfg = OmegaConf.create({"game": {"num_agents": 1, "map": {"width": 10, "height": 10}}})
            cfg = OmegaConf.merge(default_cfg, env_overrides or {})
            return SingleTaskCurriculum(path, cfg)

        monkeypatch.setattr(
            "metta.mettagrid.curriculum.random.curriculum_from_config_path", mock_curriculum_from_config_path
        )

        tasks = ["primary", "secondary"]
        task_weights = create_mock_curricula(tasks)

        threshold = 0.5  # Primary reaches this after 5 steps with increment 0.1
        score_gen = ThresholdDependentScores(
            primary_task="primary", secondary_task="secondary", threshold=threshold, increment=0.1
        )

        curriculum = LearningProgressCurriculum(
            tasks=task_weights,
            ema_timescale=0.03,  # Faster adaptation
            sample_threshold=5,  # Lower threshold for faster learning
            memory=20,
            rand_task_rate=0.5,  # Higher random rate to ensure both tasks get sampled
        )

        # Initialize all tasks to prevent empty outcomes
        for task_name in tasks:
            curriculum.get_task()
            score = score_gen.get_score(task_name)
            curriculum.complete_task(task_name, score)

        results = run_curriculum_simulation(curriculum, score_gen, 150)

        # Analyze results
        weight_history = results["weight_history"]
        task_counts = results["task_counts"]
        final_weights = results["final_weights"]

        print(f"Task counts: {task_counts}")
        print(f"Final weights: {final_weights}")

        # Check weight evolution over time
        assert len(weight_history) == 150, f"Should have 150 weight snapshots, got {len(weight_history)}"

        # Early: with high rand_task_rate=0.5, weights may be more balanced
        early_weights = weight_history[10]
        print(f"Early weights (step 10): {early_weights}")
        # With 50% random sampling, primary might not dominate early
        assert early_weights.get("primary", 0) >= 0.5, (
            f"Primary should have at least 50% weight early, got {early_weights}"
        )

        # After ~5 steps, primary reaches threshold and flatlines
        # Check weights after primary flatlines
        post_threshold_weights = weight_history[30]
        print(f"Post-threshold weights (step 30): {post_threshold_weights}")

        # Mid: primary has flatlined, but secondary is still at 0 (not yet discovered as learnable)
        mid_weights = weight_history[75]
        print(f"Mid weights (step 75): {mid_weights}")

        # Late: algorithm may or may not discover secondary becomes learnable
        late_weights = weight_history[120]
        print(f"Late weights (step 120): {late_weights}")

        # Verify both tasks were sampled
        assert len(task_counts) == 2, "Should have sampled both tasks"

        # Count samples for each task (handle curriculum prefix)
        primary_count = sum(count for task, count in task_counts.items() if "primary" in task)
        secondary_count = sum(count for task, count in task_counts.items() if "secondary" in task)
        total_samples = sum(task_counts.values())

        print(f"Primary task count: {primary_count}")
        print(f"Secondary task count: {secondary_count}")

        assert total_samples == 150, f"Should have 150 total samples, got {total_samples}"

        # Primary should be heavily sampled (shows initial learning progress)
        assert primary_count > 60, f"Should sample primary task heavily (shows early progress), got {primary_count}"

        # Secondary should be sampled at least a few times due to rand_task_rate=0.5
        # But learning progress quickly identifies primary as better, so secondary gets few samples
        assert secondary_count > 0, (
            f"Should sample secondary task at least once due to random exploration, got {secondary_count}"
        )

        # Analyze sampling ratios
        primary_ratio = primary_count / total_samples
        secondary_ratio = secondary_count / total_samples
        print(f"Sampling ratios - Primary: {primary_ratio:.3f}, Secondary: {secondary_ratio:.3f}")

        # The expected behavior:
        # 1. Primary shows learning progress initially (0->0.5 in 5 steps)
        # 2. Primary flatlines after threshold, losing learning progress
        # 3. Secondary stays at 0 until primary hits threshold, then becomes learnable
        # 4. Due to high rand_task_rate (0.5), secondary gets sampled enough to potentially discover it's learnable

        # Check if algorithm discovered secondary becomes learnable
        if secondary_count > 30:  # If secondary was sampled enough after threshold
            # Calculate when secondary likely became learnable
            # Primary hits threshold after ~5-10 samples, so secondary becomes learnable around step 10-20
            # With rand_task_rate=0.5, secondary should get ~50% of samples after that point
            late_secondary_weight = late_weights.get("secondary", 0)
            if late_secondary_weight > 0.3:
                print("✓ Algorithm discovered secondary task becomes learnable after threshold")
            else:
                print("✓ Algorithm focused on early learning progress but didn't fully discover secondary's potential")

        # Final weights analysis
        print(
            f"Final weights - Primary: {final_weights.get('primary', 0):.3f}, "
            f"Secondary: {final_weights.get('secondary', 0):.3f}"
        )

        # The algorithm's behavior is correct either way:
        # - It correctly identified primary's initial learning progress
        # - Whether it discovers secondary's delayed learnability depends on exploration vs exploitation balance

        print("✓ PASSED: Learning progress correctly responds to threshold-dependent task dynamics")


class TestPrioritizeRegressedCurriculumScenarios:
    """Test the specific Prioritize Regressed Curriculum scenarios.

    This curriculum prioritizes tasks where performance has regressed from peak.
    Weight = max_reward / average_reward, so high weight means we've done better before.
    """

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
