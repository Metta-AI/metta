"""Test suite for the GeneticBuckettedCurriculum."""

import random
from typing import Dict

import pytest
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.genetic import GeneticBuckettedCurriculum


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set all random seeds for deterministic test behavior."""
    random.seed(42)
    yield
    random.seed()


@pytest.fixture
def env_cfg():
    """Create a dummy environment configuration."""
    return DictConfig({
        "game": {
            "map": {"width": 10, "height": 10},
            "objects": {"altar": 1, "generator_red": 1},
            "num_agents": 4,
        }
    })


@pytest.fixture
def buckets():
    """Create test buckets with both discrete and continuous parameters."""
    return {
        "game.map.width": {"range": [5, 20]},
        "game.map.height": {"range": [5, 20]},
        "game.objects.altar": {"values": [1, 2, 3]},
        "game.objects.generator_red": {"range": [0.5, 2.5]},
    }


def test_genetic_curriculum_initialization(monkeypatch, env_cfg, buckets):
    """Test that the genetic curriculum initializes correctly."""
    monkeypatch.setattr(
        "metta.mettagrid.curriculum.genetic.config_from_path",
        lambda path, env_overrides=None: env_cfg
    )
    
    curr = GeneticBuckettedCurriculum(
        "dummy_path",
        buckets=buckets,
        population_size=10,
        replacement_rate=0.2,
        mutation_rate=0.5,
    )
    
    # Check population size
    assert len(curr._id_to_curriculum) == 10
    assert len(curr._id_to_params) == 10
    
    # Check that all tasks have valid parameters
    for task_id, params in curr._id_to_params.items():
        assert 5 <= params["game.map.width"] <= 20
        assert 5 <= params["game.map.height"] <= 20
        assert params["game.objects.altar"] in [1, 2, 3]
        assert 0.5 <= params["game.objects.generator_red"] <= 2.5
        
        # Check task ID format
        assert "game.map.width=" in task_id
        assert "game.map.height=" in task_id


def test_get_task(monkeypatch, env_cfg, buckets):
    """Test getting tasks from the curriculum."""
    monkeypatch.setattr(
        "metta.mettagrid.curriculum.genetic.config_from_path",
        lambda path, env_overrides=None: env_cfg
    )
    
    curr = GeneticBuckettedCurriculum(
        "dummy_path",
        buckets=buckets,
        population_size=5,
    )
    
    # Get multiple tasks
    tasks_seen = set()
    for _ in range(20):
        task = curr.get_task()
        assert hasattr(task, "id")
        assert hasattr(task, "env_cfg")
        tasks_seen.add(task.id())
    
    # Should see multiple different tasks from the population
    assert len(tasks_seen) > 1
    assert len(tasks_seen) <= 5  # Cannot exceed population size


def test_evolution(monkeypatch, env_cfg, buckets):
    """Test that the population evolves after task completion."""
    monkeypatch.setattr(
        "metta.mettagrid.curriculum.genetic.config_from_path",
        lambda path, env_overrides=None: env_cfg
    )
    
    curr = GeneticBuckettedCurriculum(
        "dummy_path",
        buckets=buckets,
        population_size=10,
        replacement_rate=0.3,  # Replace 30% of population
        mutation_rate=0.5,
    )
    
    # Record initial population
    initial_task_ids = set(curr._id_to_curriculum.keys())
    assert len(initial_task_ids) == 10
    
    # Complete some tasks to trigger evolution
    for i, task_id in enumerate(list(initial_task_ids)[:5]):
        # Give different scores to create weight differences
        score = 0.1 + i * 0.2
        curr.complete_task(task_id, score)
    
    # Check that population has evolved
    final_task_ids = set(curr._id_to_curriculum.keys())
    assert len(final_task_ids) == 10  # Population size maintained
    
    # Some tasks should be new (but not necessarily all due to randomness)
    new_tasks = final_task_ids - initial_task_ids
    assert len(new_tasks) >= 1  # At least some evolution occurred


def test_mutation(monkeypatch, env_cfg, buckets):
    """Test the mutation operator."""
    monkeypatch.setattr(
        "metta.mettagrid.curriculum.genetic.config_from_path",
        lambda path, env_overrides=None: env_cfg
    )
    
    curr = GeneticBuckettedCurriculum(
        "dummy_path",
        buckets=buckets,
        population_size=5,
    )
    
    # Create a fake task score list
    task_scores = [(task_id, 1.0) for task_id in curr._id_to_curriculum.keys()]
    
    # Test mutation multiple times
    parent_id = list(curr._id_to_curriculum.keys())[0]
    parent_params = curr._id_to_params[parent_id]
    
    mutations_seen = set()
    for _ in range(20):
        mutated = curr._mutate(task_scores)
        
        # Should have same keys
        assert set(mutated.keys()) == set(parent_params.keys())
        
        # Should differ in at least one parameter
        differences = sum(1 for k in mutated if mutated[k] != parent_params[k])
        if differences > 0:
            mutations_seen.add(tuple(sorted(mutated.items())))
    
    # Should see some variety in mutations
    assert len(mutations_seen) > 1


def test_crossover(monkeypatch, env_cfg, buckets):
    """Test the crossover operator."""
    monkeypatch.setattr(
        "metta.mettagrid.curriculum.genetic.config_from_path",
        lambda path, env_overrides=None: env_cfg
    )
    
    curr = GeneticBuckettedCurriculum(
        "dummy_path",
        buckets=buckets,
        population_size=5,
    )
    
    # Create a fake task score list
    task_scores = [(task_id, 1.0) for task_id in curr._id_to_curriculum.keys()]
    
    # Test crossover multiple times
    crossovers_seen = set()
    for _ in range(20):
        child = curr._crossover(task_scores)
        
        # Should have all parameter keys
        assert set(child.keys()) == set(buckets.keys())
        
        # Each parameter should be valid
        assert 5 <= child["game.map.width"] <= 20
        assert 5 <= child["game.map.height"] <= 20
        assert child["game.objects.altar"] in [1, 2, 3]
        assert 0.5 <= child["game.objects.generator_red"] <= 2.5
        
        crossovers_seen.add(tuple(sorted(child.items())))
    
    # Should see variety in crossover results
    assert len(crossovers_seen) > 1


def test_weighted_selection(monkeypatch, env_cfg, buckets):
    """Test that parent selection is proportional to weight."""
    monkeypatch.setattr(
        "metta.mettagrid.curriculum.genetic.config_from_path",
        lambda path, env_overrides=None: env_cfg
    )
    
    curr = GeneticBuckettedCurriculum(
        "dummy_path",
        buckets=buckets,
        population_size=3,
    )
    
    # Create task scores with very different weights
    task_ids = list(curr._id_to_curriculum.keys())
    task_scores = [
        (task_ids[0], 0.1),   # Low weight
        (task_ids[1], 1.0),   # Medium weight
        (task_ids[2], 10.0),  # High weight
    ]
    
    # Sample many times and count selections
    selections = {task_id: 0 for task_id, _ in task_scores}
    for _ in range(1000):
        selected = curr._select_parent(task_scores)
        selections[selected] += 1
    
    # High weight task should be selected most often
    assert selections[task_ids[2]] > selections[task_ids[1]]
    assert selections[task_ids[1]] > selections[task_ids[0]]


def test_parameter_constraints(monkeypatch, env_cfg, buckets):
    """Test that parameters stay within their defined constraints."""
    monkeypatch.setattr(
        "metta.mettagrid.curriculum.genetic.config_from_path",
        lambda path, env_overrides=None: env_cfg
    )
    
    curr = GeneticBuckettedCurriculum(
        "dummy_path",
        buckets=buckets,
        population_size=50,
        replacement_rate=0.5,
        mutation_rate=0.7,
    )
    
    # Complete many tasks to trigger multiple evolutions
    for i in range(100):
        task = curr.get_task()
        curr.complete_task(task.id(), random.random())
    
    # Check all tasks still have valid parameters
    for params in curr._id_to_params.values():
        assert 5 <= params["game.map.width"] <= 20
        assert 5 <= params["game.map.height"] <= 20
        assert params["game.objects.altar"] in [1, 2, 3]
        assert 0.5 <= params["game.objects.generator_red"] <= 2.5


def test_integer_parameter_handling(monkeypatch, env_cfg):
    """Test that integer parameters remain integers."""
    monkeypatch.setattr(
        "metta.mettagrid.curriculum.genetic.config_from_path",
        lambda path, env_overrides=None: env_cfg
    )
    
    buckets = {
        "game.map.width": {"range": [5, 20]},  # Should be integer
        "game.objects.scale": {"range": [0.5, 2.5]},  # Should be float
    }
    
    curr = GeneticBuckettedCurriculum(
        "dummy_path",
        buckets=buckets,
        population_size=20,
    )
    
    # Check all parameters have correct types
    for params in curr._id_to_params.values():
        assert isinstance(params["game.map.width"], int)
        assert isinstance(params["game.objects.scale"], float)