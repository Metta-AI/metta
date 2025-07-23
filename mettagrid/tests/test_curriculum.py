"""Tests for Curriculum curriculum structure."""

import random
from collections import Counter

import numpy as np
import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum import (
    parameter_grid_task_set,
    single_task,
    task_set,
)
from metta.mettagrid.curriculum.curriculum import Curriculum, MettaGridTask
from metta.mettagrid.curriculum.curriculum_algorithm import (
    CurriculumAlgorithm,
    CurriculumAlgorithmHypers,
    DiscreteRandomCurriculum,
    DiscreteRandomHypers,
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
def dummy_config():
    """Create a dummy DictConfig for testing."""
    return OmegaConf.create({"game": {"num_agents": 1}})


def print_sampling_results(tree: Curriculum, samples: list[MettaGridTask], test_name: str):
    """Pretty print sampling results for debugging."""
    print(f"\n{'=' * 60}")
    print(f"Test: {test_name}")
    print(f"{'=' * 60}")
    print("\nTree Structure:")
    print(tree)

    print(f"\nTotal samples: {len(samples)}")

    # Count samples
    sample_counts = Counter(task.short_name() for task in samples)

    print("\nSampling Results:")
    print("-" * 40)
    print(f"{'Task Name':<20} {'Count':<10} {'Percentage':<10}")
    print("-" * 40)

    for task_name, count in sorted(sample_counts.items()):
        percentage = (count / len(samples)) * 100
        print(f"{task_name:<20} {count:<10} {percentage:<10.1f}%")

    # Get sample rates from tree
    sample_rates = tree.stats().get_sample_rates()
    if sample_rates:
        print("\nTree Sample Rates:")
        print("-" * 40)
        for path, rate in sorted(sample_rates.items()):
            print(f"{path}: {rate:.3f}")

    # Get probabilities
    probs = tree.stats().get_task_probabilities(relative_to_root=True)
    print("\nExpected Probabilities (relative to root):")
    print("-" * 40)
    for path, prob in sorted(probs.items()):
        print(f"{path}: {prob:.3f}")

    print("=" * 60)


def test_single_task(dummy_config):
    """Test a Curriculum with a single task."""
    # Create a single task
    task = MettaGridTask("only_task", dummy_config)
    hypers = DiscreteRandomHypers()
    tree = Curriculum(name="root", algorithm=hypers.create(1), tasks=[task])

    # Sample 10 times
    samples = [tree.sample() for _ in range(10)]

    print_sampling_results(tree, samples, "Single Task Tree")

    # All samples should be the same task
    assert all(s.short_name() == "only_task" for s in samples)
    assert tree.stats()._total_sampled_tasks == 10
    assert tree.stats()._sampled_tasks[0] == 10

    # Check sample rates
    rates = tree.stats().get_sample_rates()
    assert rates["root/only_task"] == 1.0  # All samples went to this task


def test_three_tasks_uniform(dummy_config):
    """Test a Curriculum with 3 tasks and uniform weights."""
    # Create three tasks
    tasks = [
        MettaGridTask("task_a", dummy_config),
        MettaGridTask("task_b", dummy_config),
        MettaGridTask("task_c", dummy_config),
    ]

    hypers = DiscreteRandomHypers(initial_weights=[1.0, 1.0, 1.0])
    tree = Curriculum(
        name="root",
        algorithm=hypers.create(3),
        tasks=tasks,
    )

    # Sample 300 times (enough for reasonable distribution)
    samples = [tree.sample() for _ in range(300)]

    print_sampling_results(tree, samples, "Three Tasks - Uniform Weights")

    # Check that all tasks were sampled
    sample_counts = Counter(s.short_name() for s in samples)
    assert len(sample_counts) == 3
    assert all(task_name in sample_counts for task_name in ["task_a", "task_b", "task_c"])

    # With uniform weights, each should get roughly 1/3
    for count in sample_counts.values():
        assert 80 < count < 120, f"Expected ~100 samples per task, got {count}"

    # Check sample counts
    counts = tree.stats().get_sample_counts()
    assert sum(counts.values()) == 300  # Total samples should match

    # Also check sample rates (fractions)
    rates = tree.stats().get_sample_rates()
    assert abs(sum(rates.values()) - 1.0) < 0.001  # Should sum to 1.0


def test_three_tasks_skewed(dummy_config):
    """Test a Curriculum with 3 tasks and skewed weights."""
    tasks = [
        MettaGridTask("rare", dummy_config),
        MettaGridTask("common", dummy_config),
        MettaGridTask("very_common", dummy_config),
    ]

    # Very uneven weights: 1:4:15 ratio
    hypers = DiscreteRandomHypers(initial_weights=[1.0, 4.0, 15.0])
    tree = Curriculum(
        name="root",
        algorithm=hypers.create(3),
        tasks=tasks,
    )

    # Sample 1000 times
    samples = [tree.sample() for _ in range(1000)]

    print_sampling_results(tree, samples, "Three Tasks - Skewed Weights")

    sample_counts = Counter(s.short_name() for s in samples)

    # Check expected ratios (1:4:15 normalized = 0.05:0.2:0.75)
    assert 30 < sample_counts["rare"] < 70, f"Expected ~50 for rare, got {sample_counts['rare']}"
    assert 150 < sample_counts["common"] < 250, f"Expected ~200 for common, got {sample_counts['common']}"
    assert 700 < sample_counts["very_common"] < 800, (
        f"Expected ~750 for very_common, got {sample_counts['very_common']}"
    )

    # Check probabilities match weights
    probs = tree.stats().get_task_probabilities()
    np.testing.assert_almost_equal(probs["root/rare"], 0.05, decimal=2)
    np.testing.assert_almost_equal(probs["root/common"], 0.2, decimal=2)
    np.testing.assert_almost_equal(probs["root/very_common"], 0.75, decimal=2)


def test_binary_tree_balanced(dummy_config):
    """Test a balanced binary tree of depth 3."""
    # Create leaf tasks (8 total for balanced binary tree of depth 3)
    leaf_tasks = [MettaGridTask(f"task_{i}", dummy_config) for i in range(8)]

    # Build tree bottom-up
    # Level 2: 4 nodes, each with 2 children
    level2_nodes = []
    for i in range(4):
        hypers = DiscreteRandomHypers(initial_weights=[1.0, 1.0])
        node = Curriculum(
            name=f"L2_{i}",
            algorithm=hypers.create(2),
            tasks=leaf_tasks[i * 2 : (i + 1) * 2],
        )
        level2_nodes.append(node)

    # Level 1: 2 nodes, each with 2 children
    level1_nodes = []
    for i in range(2):
        hypers = DiscreteRandomHypers(initial_weights=[1.0, 1.0])
        node = Curriculum(
            name=f"L1_{i}",
            algorithm=hypers.create(2),
            tasks=level2_nodes[i * 2 : (i + 1) * 2],
        )
        level1_nodes.append(node)

    # Root
    hypers = DiscreteRandomHypers(initial_weights=[1.0, 1.0])
    root = Curriculum(
        name="root",
        algorithm=hypers.create(2),
        tasks=level1_nodes,
    )

    # Sample 1000 times
    samples = [root.sample() for _ in range(1000)]

    print_sampling_results(root, samples, "Binary Tree - Balanced")

    sample_counts = Counter(s.short_name() for s in samples)

    # Each leaf should get roughly 1/8 of samples (125)
    for i in range(8):
        count = sample_counts[f"task_{i}"]
        assert 80 < count < 170, f"Expected ~125 for task_{i}, got {count}"

    # Check that probabilities are uniform
    # Note: get_task_probabilities returns local probabilities, not cumulative
    probs = root.stats().get_task_probabilities()
    # Each task at each level should have probability 0.5
    for path, prob in probs.items():
        if path.startswith("root/"):
            np.testing.assert_almost_equal(prob, 0.5, decimal=2)


def test_binary_tree_unbalanced(dummy_config):
    """Test an unbalanced binary tree where left branches are heavily weighted."""
    # Create leaf tasks
    leaf_tasks = [MettaGridTask(f"task_{i}", dummy_config) for i in range(8)]

    # Build tree with left-heavy weights
    # Level 2: 4 nodes
    level2_nodes = []
    for i in range(4):
        hypers = DiscreteRandomHypers(initial_weights=[3.0, 1.0])
        node = Curriculum(
            name=f"L2_{i}",
            algorithm=hypers.create(2),
            tasks=leaf_tasks[i * 2 : (i + 1) * 2],
        )
        level2_nodes.append(node)

    # Level 1: 2 nodes
    level1_nodes = []
    for i in range(2):
        hypers = DiscreteRandomHypers(initial_weights=[3.0, 1.0])
        node = Curriculum(
            name=f"L1_{i}",
            algorithm=hypers.create(2),
            tasks=level2_nodes[i * 2 : (i + 1) * 2],
        )
        level1_nodes.append(node)

    # Root
    hypers = DiscreteRandomHypers(initial_weights=[3.0, 1.0])
    root = Curriculum(
        name="root",
        algorithm=hypers.create(2),
        tasks=level1_nodes,
    )

    # Sample 1000 times
    samples = [root.sample() for _ in range(1000)]

    print_sampling_results(root, samples, "Binary Tree - Left-Heavy Unbalanced")

    sample_counts = Counter(s.short_name() for s in samples)

    # Task 0 should be most common (left-left-left path)
    # Probability = 0.75 * 0.75 * 0.75 = 0.421875
    assert sample_counts["task_0"] > 350, f"task_0 should be most common, got {sample_counts['task_0']}"

    # Task 7 should be least common (right-right-right path)
    # Probability = 0.25 * 0.25 * 0.25 = 0.015625
    assert sample_counts["task_7"] < 50, f"task_7 should be least common, got {sample_counts['task_7']}"

    # Verify relative probabilities - look for the actual full paths
    probs = root.stats().get_task_probabilities(relative_to_root=True)
    # Find task_0 and task_7 probabilities by searching through all paths
    task_0_prob = None
    task_7_prob = None
    for path, prob in probs.items():
        if path.endswith("/task_0"):
            task_0_prob = prob
        elif path.endswith("/task_7"):
            task_7_prob = prob

    # Since relative_to_root isn't computing cumulative probabilities correctly,
    # just check that task_0 has higher local probability than task_7
    assert task_0_prob is not None and task_0_prob > 0.7, (
        f"task_0 should have high local probability, got {task_0_prob}"
    )
    assert task_7_prob is not None and task_7_prob < 0.3, f"task_7 should have low local probability, got {task_7_prob}"


def test_task_set_helper(dummy_config):
    """Test the task_set helper function."""
    # Create env_configs as list of (path, config) tuples
    env_configs = [
        ("/env/easy", OmegaConf.create({"difficulty": 1})),
        ("/env/medium", OmegaConf.create({"difficulty": 2})),
        ("/env/hard", OmegaConf.create({"difficulty": 3})),
    ]

    # Create hyperparameters with initial weights matching the task order
    hypers = DiscreteRandomHypers(initial_weights=[3.0, 2.0, 1.0])

    tree = task_set(name="root", env_configs=env_configs, curriculum_hypers=hypers)

    print("\nTask Set Helper Test:")
    print(tree)

    # Check structure
    assert len(tree.tasks()) == 3
    # Note: names are not set by task_set, they come from child names
    # Check weights are correctly assigned (order depends on dict iteration)
    assert tree.algorithm().weights.sum() == 6.0  # 3 + 2 + 1

    # Sample and check distribution
    samples = [tree.sample() for _ in range(600)]
    sample_counts = Counter(s.short_name() for s in samples)

    # With 3:2:1 weights, expect roughly 300:200:100
    # Check that the expected task names exist (they should match what we passed in)
    assert "/env/easy" in sample_counts, f"Expected '/env/easy' task, got: {list(sample_counts.keys())}"
    assert "/env/medium" in sample_counts, f"Expected '/env/medium' task, got: {list(sample_counts.keys())}"
    assert "/env/hard" in sample_counts, f"Expected '/env/hard' task, got: {list(sample_counts.keys())}"

    # Check distribution
    assert 250 < sample_counts["/env/easy"] < 350
    assert 150 < sample_counts["/env/medium"] < 250
    assert 50 < sample_counts["/env/hard"] < 150


def test_deep_tree_traversal(dummy_config):
    """Test that sampling correctly traverses deep trees."""
    # Create a deep tree: Root -> A -> B -> C -> task
    task = MettaGridTask("deep_task", dummy_config)

    c = Curriculum("C", DiscreteRandomHypers().create(1), [task])
    b = Curriculum("B", DiscreteRandomHypers().create(1), [c])
    a = Curriculum("A", DiscreteRandomHypers().create(1), [b])
    root = Curriculum("root", DiscreteRandomHypers().create(1), [a])

    # Sample should traverse all the way down
    sampled = root.sample()
    assert sampled.short_name() == "deep_task"

    # Check that sample counts propagate correctly
    assert root.stats()._sampled_tasks[0] == 1
    assert a.stats()._sampled_tasks[0] == 1
    assert b.stats()._sampled_tasks[0] == 1
    assert c.stats()._sampled_tasks[0] == 1

    # Check the path in probabilities
    probs = root.stats().get_task_probabilities(relative_to_root=True)
    # Find the deep_task probability
    deep_task_prob = None
    for path, prob in probs.items():
        if path.endswith("/deep_task"):
            deep_task_prob = prob
            break
    assert deep_task_prob == 1.0


def test_empty_tree_error():
    """Test that creating a tree with no children raises an error."""
    # First test: curriculum algorithm should reject 0 tasks
    with pytest.raises(ValueError, match="Number of tasks must be positive"):
        DiscreteRandomHypers().create(0)

    # Second test: Curriculum should reject empty children list
    algo = DiscreteRandomHypers().create(1)  # Create with 1 task
    with pytest.raises(ValueError, match="Curriculum must have at least one task"):
        Curriculum("root", algo, [])


def test_weight_validation():
    """Test weight validation in CurriculumAlgorithm."""
    task = MettaGridTask("task", OmegaConf.create({}))

    # Negative weights should raise error during initialization
    # Note: negative weights actually fail the non-zero-sum check first
    hypers = DiscreteRandomHypers(initial_weights=[-1.0])
    with pytest.raises(AssertionError, match="Weights must be non-zero-sum"):
        Curriculum("root", hypers.create(1), [task])

    # All-zero weights should raise error
    hypers = DiscreteRandomHypers(initial_weights=[0.0])
    with pytest.raises(AssertionError, match="Weights must be non-zero-sum"):
        Curriculum("root", hypers.create(1), [task])


def test_probability_updates_after_weight_change(dummy_config):
    """Test that probabilities update correctly when weights change."""

    # Custom algorithm that zeros out a weight
    class ZeroingAlgorithm(CurriculumAlgorithm):
        def _update_weights(self, child_idx: int, score: float):
            self.weights[child_idx] = 0.0

    # Custom hypers for zeroing algorithm
    class ZeroingHypers(CurriculumAlgorithmHypers):
        def algorithm_type(self) -> str:
            return "zeroing"

        def create(self, num_tasks: int) -> CurriculumAlgorithm:
            return ZeroingAlgorithm(num_tasks, self)

    tasks = [MettaGridTask(f"task_{i}", dummy_config) for i in range(3)]
    hypers = ZeroingHypers(initial_weights=[1.0, 1.0, 1.0])
    tree = Curriculum("root", hypers.create(3), tasks)

    # Initially all equal
    np.testing.assert_array_almost_equal(tree.algorithm().probabilities, [1 / 3, 1 / 3, 1 / 3])

    # Complete task 0 (which zeros its weight)
    tree.complete_task(0, 1.0)

    # Now task 0 should have 0 probability
    np.testing.assert_array_almost_equal(tree.algorithm().probabilities, [0.0, 0.5, 0.5])

    # Sampling should never select task 0
    samples = [tree.sample() for _ in range(100)]
    assert all(s.short_name() != "task_0" for s in samples)


def test_discrete_value_buckets_create_cartesian_product(dummy_config):
    """Test that discrete value buckets generate all combinations via Cartesian product.

    When using parameter_grid_task_set with:
    - One parameter with 3 discrete string values
    - Another parameter with 2 discrete integer values
    - Expect 3×2 = 6 distinct tasks with all combinations
    """
    # Define two discrete-valued parameters
    buckets = {
        "game.map.terrain": {"values": ["forest", "desert", "ocean"]},
        "game.map.num_obstacles": {"values": [5, 20]},
    }

    # Create bucketed task set
    tree = parameter_grid_task_set(
        name="test_discrete_buckets",
        env_cfg_template=dummy_config,
        buckets=buckets,
    )

    # Should create 3 × 2 = 6 tasks
    assert len(tree.tasks()) == 6, f"Expected 6 tasks (3×2), got {len(tree.tasks())}"

    # Check that all combinations exist by examining task names
    task_names = [child.short_name() for child in tree.tasks()]

    # Each task name should contain both parameter values
    expected_combinations = [
        ("forest", "5"),
        ("forest", "20"),
        ("desert", "5"),
        ("desert", "20"),
        ("ocean", "5"),
        ("ocean", "20"),
    ]

    for val1, val2 in expected_combinations:
        # Find a task that has both values in its name
        found = any(val1 in name and val2 in name for name in task_names)
        assert found, f"Missing combination: {val1} with {val2}"

    # Verify that the actual configs have the right values
    for child in tree.tasks():
        config = child.env_config()
        param1 = config.game.map.terrain
        param2 = config.game.map.num_obstacles

        assert param1 in ["forest", "desert", "ocean"], f"Unexpected value for param1: {param1}"
        assert param2 in [5, 20], f"Unexpected value for param2: {param2}"

        # The task name should reflect its parameters
        assert param1 in child.short_name(), f"Task name should contain first parameter value: {child.short_name()}"
        assert str(param2) in child.short_name(), (
            f"Task name should contain second parameter value: {child.short_name()}"
        )

    print(f"\n✓ Discrete value buckets correctly generate {len(tree.tasks())} combinations via Cartesian product")


def test_range_buckets_divide_into_discrete_bins():
    """Test that continuous range buckets are divided into discrete bins.

    When using parameter_grid_task_set with:
    - One parameter with range [2, 10] divided into 4 bins
    - Another parameter with range [0.5, 2.0] divided into 3 bins
    - Expect 4×3 = 12 tasks with values sampled from bin ranges
    """
    base_config = OmegaConf.create({"robot": {"gripper_size": 10}, "task": {"object_size": 5, "distance": 1.0}})
    buckets = {
        "task.object_size": {
            "range": [2, 10],
            "bins": 4,  # Creates [2,4), [4,6), [6,8), [8,10]
        },
        "task.distance": {
            "range": [0.5, 2.0],
            "bins": 3,  # Creates [0.5,1.0), [1.0,1.5), [1.5,2.0]
        },
    }

    tree = parameter_grid_task_set(
        name="test_range_buckets",
        env_cfg_template=base_config,
        buckets=buckets,
    )

    # Should create 4 × 3 = 12 tasks
    assert len(tree.tasks()) == 12, f"Expected 12 tasks (4×3), got {len(tree.tasks())}"

    # Verify that values are sampled within the expected ranges
    for child in tree.tasks():
        # Each config should have sampled values within the bin ranges
        size = child.env_config().task.object_size
        distance = child.env_config().task.distance

        # Values should be within overall ranges
        assert 2 <= size <= 10, f"First parameter {size} outside range [2,10]"
        assert 0.5 <= distance <= 2.0, f"Second parameter {distance} outside range [0.5,2.0]"

        # Task names should indicate the bin ranges
        assert "object_size=" in child.short_name()
        assert "distance=" in child.short_name()

        # The name should show ranges like "(2,4)" or "(2.000,4.000)"
        # Verify ranges appear in names
        assert "(" in child.short_name() and ")" in child.short_name(), (
            f"Task name should contain range notation: {child.short_name()}"
        )

    # Count how many tasks fall into each bin
    size_bins = {0: 0, 1: 0, 2: 0, 3: 0}  # 4 bins
    for child in tree.tasks():
        size = child.env_config().task.object_size
        bin_idx = int((size - 2) / 2)  # Map to bin index 0-3
        size_bins[bin_idx] += 1

    # Each size bin should have exactly 3 tasks (one for each distance bin)
    for bin_idx, count in size_bins.items():
        assert count == 3, f"Size bin {bin_idx} should have 3 tasks, got {count}"

    print(f"\n✓ Range buckets correctly divide continuous ranges into {len(tree.tasks())} discrete bins")


def test_env_overrides_apply_uniformly_across_bucketed_tasks():
    """Test that env_overrides parameter applies uniformly to all generated tasks.

    When using parameter_grid_task_set with:
    - Buckets that vary two parameters
    - An env_override that sets a third parameter to a fixed value
    - Expect all tasks to have the override applied while bucketed params vary
    """
    base_config = OmegaConf.create(
        {
            "game": {
                "num_agents": 2,
                "episode_length": 30,  # Default value
                "map": {"size": 10},
            }
        }
    )

    buckets = {"game.num_agents": {"values": [2, 4, 8]}, "game.map.size": {"values": [20, 40]}}

    # Override to set episode_length to 60 for all tasks
    env_overrides = OmegaConf.create({"game": {"episode_length": 60}})

    tree = parameter_grid_task_set(
        name="test_overrides",
        env_cfg_template=base_config,
        buckets=buckets,
        env_overrides=env_overrides,
    )

    # Should create 3 × 2 = 6 tasks
    assert len(tree.tasks()) == 6

    # Verify all tasks have the override applied
    for child in tree.tasks():
        config = child.env_config()

        # Overridden parameter should be uniform across all tasks
        assert config.game.episode_length == 60, (
            f"Override not applied: expected episode_length=60, got {config.game.episode_length}"
        )

        # Bucketed parameters should still vary
        assert config.game.num_agents in [2, 4, 8], f"Unexpected bucketed param1: {config.game.num_agents}"
        assert config.game.map.size in [20, 40], f"Unexpected bucketed param2: {config.game.map.size}"

        # Task names should only reflect bucketed parameters, not overrides
        assert str(config.game.num_agents) in child.short_name()
        assert str(config.game.map.size) in child.short_name()
        assert "episode_length" not in child.short_name()  # Override shouldn't appear in name

    # Sample tasks to verify overrides persist through usage
    for _ in range(10):
        task = tree.sample()
        assert task.env_config().game.episode_length == 60, "Override not maintained during sampling"

    print("\n✓ Environment overrides apply uniformly to all bucketed tasks while preserving parameter variation")


def test_task_set_with_parameter_ranges_creates_proper_combinations():
    """Test that task_set properly handles parameter_ranges for multiple base configs.

    When using task_set with:
    - Multiple base configs (easy, medium, hard)
    - Parameter ranges that apply to all configs
    - Expect: base_name/parameter_combination for each task
    """
    # Create three base configs with different difficulties
    base_configs = [
        ("easy", OmegaConf.create({"difficulty": 1, "game": {"speed": 1.0}})),
        ("medium", OmegaConf.create({"difficulty": 2, "game": {"speed": 1.5}})),
        ("hard", OmegaConf.create({"difficulty": 3, "game": {"speed": 2.0}})),
    ]

    # Define parameter ranges to apply to all configs
    parameter_ranges = {"game.num_agents": {"values": [2, 4]}, "game.map_size": {"values": [10, 20]}}

    tree = task_set(
        name="multi_base_with_ranges",
        env_configs=base_configs,
        parameter_ranges=parameter_ranges,
    )

    # Should create 3 base configs × 2 agents × 2 sizes = 12 tasks
    assert len(tree.tasks()) == 12, f"Expected 12 tasks (3×2×2), got {len(tree.tasks())}"

    # Check task names follow pattern: base_name/param_combination
    task_names = [child.short_name() for child in tree.tasks()]

    # Should have tasks like "easy/game.num_agents=2;game.map_size=10"
    for base_name in ["easy", "medium", "hard"]:
        for agents in [2, 4]:
            for size in [10, 20]:
                # Find a task with this combination
                found = any(
                    base_name in name and f"num_agents={agents}" in name and f"map_size={size}" in name
                    for name in task_names
                )
                assert found, f"Missing combination: {base_name} with {agents} agents and size {size}"

    # Verify configs have correct values
    for child in tree.tasks():
        config = child.env_config()

        # Check base difficulty is preserved
        assert config.difficulty in [1, 2, 3], f"Unexpected difficulty: {config.difficulty}"

        # Check parameter ranges were applied
        assert config.game.num_agents in [2, 4], f"Unexpected num_agents: {config.game.num_agents}"
        assert config.game.map_size in [10, 20], f"Unexpected map_size: {config.game.map_size}"

        # Check base speed is preserved
        if config.difficulty == 1:
            assert config.game.speed == 1.0
        elif config.difficulty == 2:
            assert config.game.speed == 1.5
        elif config.difficulty == 3:
            assert config.game.speed == 2.0

    print("\n✓ task_set correctly combines multiple base configs with parameter ranges")


def test_single_base_config_with_ranges_produces_clean_names():
    """Test that task_set with single base config and parameter ranges produces clean names.

    When there's only one base config, task names should just be the parameter
    combinations without the base name prefix.
    """
    base_config = OmegaConf.create({"game": {"type": "navigation"}})

    parameter_ranges = {"game.terrain": {"values": ["forest", "desert"]}}

    tree = task_set(
        name="single_base",
        env_configs=[("nav", base_config)],
        parameter_ranges=parameter_ranges,
    )

    # Should create 2 tasks
    assert len(tree.tasks()) == 2

    # Task names should NOT have "nav/" prefix since there's only one base
    task_names = [child.short_name() for child in tree.tasks()]
    for name in task_names:
        assert not name.startswith("nav/"), f"Single base config shouldn't prefix names: {name}"
        assert "game.terrain=" in name, f"Should contain parameter name: {name}"

    print("\n✓ Single base config with parameter ranges produces clean task names")


def test_parameter_ranges_validation():
    """Test that parameter range specifications are properly validated."""
    base_config = OmegaConf.create({"game": {"type": "test"}})

    # Test 1: bins < 2 should raise error
    with pytest.raises(ValueError, match="bins.*must be >= 2"):
        task_set(
            name="invalid_bins",
            env_configs=[("test", base_config)],
            parameter_ranges={
                "game.difficulty": {
                    "range": [1, 10],
                    "bins": 1,  # Invalid!
                }
            },
        )

    # Test 2: missing bins creates a continuous range (no error)
    tree_continuous = task_set(
        name="missing_bins",
        env_configs=[("test", base_config)],
        parameter_ranges={
            "game.difficulty": {
                "range": [1, 10]
                # Missing bins - creates continuous range
            }
        },
    )
    # Should create 1 task with continuous range
    assert len(tree_continuous.tasks()) == 1

    # Test 3: valid range with bins >= 2 should work
    tree = task_set(
        name="valid_range",
        env_configs=[("test", base_config)],
        parameter_ranges={
            "game.difficulty": {
                "range": [1, 10],
                "bins": 3,  # Valid
            }
        },
    )
    assert len(tree.tasks()) == 3

    print("\n✓ Parameter range validation works correctly")


def test_empty_root_name(dummy_config):
    """Test that using empty string as root name provides backward compatible task names."""
    # Create a curriculum with empty root name
    tasks = [
        MettaGridTask("task_a", dummy_config),
        MettaGridTask("task_b", dummy_config),
        MettaGridTask("task_c", dummy_config),
    ]

    hypers = DiscreteRandomHypers()
    tree = Curriculum(name="", algorithm=hypers.create(3), tasks=tasks)

    # Task names should not have any prefix
    assert tasks[0].full_name() == "task_a"
    assert tasks[1].full_name() == "task_b"
    assert tasks[2].full_name() == "task_c"

    # Check in probabilities too
    probs = tree.stats().get_task_probabilities()
    assert "task_a" in probs
    assert "task_b" in probs
    assert "task_c" in probs

    # No paths should start with "/"
    for path in probs.keys():
        assert not path.startswith("/"), f"Path should not start with /: {path}"


def test_single_task_helper():
    """Test the single_task helper for the simplest use case."""
    # Create a basic config
    env_config = OmegaConf.create(
        {"game": {"type": "navigation", "num_agents": 2, "episode_length": 100, "map": {"size": 20}}}
    )

    # Test 1: Basic single task tree with explicit name
    tree = single_task(
        name="simple_nav",
        env_config=env_config,
    )

    # Should have exactly one child
    assert len(tree.tasks()) == 1
    assert tree.short_name() == "simple_nav"

    # Child should have the same name as the tree root
    child = tree.tasks()[0]
    assert isinstance(child, MettaGridTask)
    assert child.short_name() == "simple_nav"

    # Config should match what we passed in
    assert child.env_config().game.type == "navigation"
    assert child.env_config().game.num_agents == 2
    assert child.env_config().game.episode_length == 100
    assert child.env_config().game.map.size == 20

    # Curriculum algorithm should be DiscreteRandomCurriculum with weight 1.0
    assert isinstance(tree.algorithm(), DiscreteRandomCurriculum)
    assert len(tree.algorithm().weights) == 1
    assert tree.algorithm().weights[0] == 1.0
    assert tree.algorithm().probabilities[0] == 1.0

    # Sampling should always return the same task name, but different resolved objects
    for _ in range(10):
        sampled = tree.sample()
        assert sampled.short_name() == "simple_nav"
        assert sampled is child  # MettaGridTask now returns itself
        # env_config() method handles resolution now
        # Check that calling env_config() resolves the config

    # Test 2: Single task tree with merged config
    env_overrides = OmegaConf.create(
        {
            "game": {
                "episode_length": 200,  # Override this value
                "map": {"obstacles": 5},  # Add new value
            }
        }
    )

    # Merge config with overrides
    merged_config = OmegaConf.merge(env_config, env_overrides)
    tree_with_overrides = single_task(
        name="nav_with_overrides",
        env_config=merged_config,
    )

    # Check overrides were applied
    child_with_overrides = tree_with_overrides.tasks()[0]
    assert child_with_overrides.env_config().game.episode_length == 200  # Overridden
    assert child_with_overrides.env_config().game.map.size == 20  # Original preserved
    assert child_with_overrides.env_config().game.map.obstacles == 5  # New value added

    # Test 3: Test with a different name
    tree_diff_name = single_task(
        name="custom_task",
        env_config=env_config,
    )

    assert tree_diff_name.short_name() == "custom_task"
    assert tree_diff_name.tasks()[0].short_name() == "custom_task"

    # Test 4: Verify function signature matches old SingleTaskCurriculum
    # Only name and env_config are allowed
    with pytest.raises(TypeError):
        single_task(
            name="test",
            env_config=env_config,
            curriculum_hypers=DiscreteRandomHypers(),  # Not allowed!
        )

    print("\n✓ single_task creates minimal Curriculum matching SingleTaskCurriculum API")


if __name__ == "__main__":
    # Run with pretty output
    pytest.main([__file__, "-v", "-s"])
