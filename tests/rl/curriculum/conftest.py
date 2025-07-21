"""Shared fixtures and mock classes for curriculum tests."""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set

import numpy as np
import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, Task


class MockCurriculum(Curriculum):
    """Simple mock curriculum for testing basic functionality."""

    def __init__(self):
        self.task_count = 0
        self.completed_tasks = []

    def get_task(self) -> Task:
        self.task_count += 1
        task_id = f"task_{self.task_count}"
        env_cfg = OmegaConf.create({
            "game": {
                "num_agents": 1,
                "max_steps": 100,
                "width": 10,
                "height": 10
            }
        })
        return Task(task_id, self, env_cfg)

    def complete_task(self, id: str, score: float):
        self.completed_tasks.append((id, score))

    def get_task_probs(self) -> dict[str, float]:
        return {"mock_task": 1.0}

    def get_completion_rates(self) -> dict[str, float]:
        if not self.completed_tasks:
            return {"mock_task": 0.0}
        return {"mock_task": len(self.completed_tasks) / self.task_count}

    def stats(self) -> dict:
        return {
            "total_tasks": self.task_count,
            "completed": len(self.completed_tasks)
        }


class StatefulCurriculum(Curriculum):
    """Curriculum that tracks state and provides comprehensive stats."""

    def __init__(self):
        self.task_count = 0
        self.completed_tasks = []
        self.task_probs = {"easy": 0.7, "hard": 0.3}

    def get_task(self) -> Task:
        self.task_count += 1
        # Alternate between easy and hard tasks
        difficulty = "easy" if self.task_count % 3 != 0 else "hard"
        task_id = f"{difficulty}_task_{self.task_count}"

        env_cfg = OmegaConf.create({
            "game": {
                "difficulty": difficulty,
                "width": 10 if difficulty == "easy" else 20,
                "height": 10 if difficulty == "easy" else 20,
                "num_agents": 2,
                "max_steps": 100
            }
        })
        return Task(task_id, self, env_cfg)

    def complete_task(self, id: str, score: float):
        self.completed_tasks.append((id, score))
        # Update task probabilities based on performance
        if "easy" in id and score > 0.8:
            self.task_probs["easy"] = max(0.3, self.task_probs["easy"] - 0.05)
            self.task_probs["hard"] = min(0.7, self.task_probs["hard"] + 0.05)

    def get_completion_rates(self) -> dict[str, float]:
        easy_tasks = [t for t, _ in self.completed_tasks if "easy" in t]
        hard_tasks = [t for t, _ in self.completed_tasks if "hard" in t]
        total = len(self.completed_tasks)
        return {
            "task_completions/easy": len(easy_tasks) / total if total > 0 else 0,
            "task_completions/hard": len(hard_tasks) / total if total > 0 else 0,
        }

    def get_task_probs(self) -> dict[str, float]:
        return self.task_probs

    def stats(self) -> dict:
        avg_score = sum(s for _, s in self.completed_tasks) / len(self.completed_tasks) if self.completed_tasks else 0
        return {
            "total_tasks": self.task_count,
            "completed_tasks": len(self.completed_tasks),
            "average_score": avg_score,
            "learning_progress": 0.002,
        }


# ============================================================================
# Score Generators for Controlled Testing
# ============================================================================


class ScoreGenerator(ABC):
    """Abstract interface for generating controlled scores for curriculum testing."""

    @abstractmethod
    def get_score(self, task_id: str) -> float:
        """Get the next score for the given task."""
        pass

    def reset(self) -> None:
        """Reset generator state. Override if needed."""
        return


class MonotonicLinearScores(ScoreGenerator):
    """Generates monotonically increasing scores for each task."""

    def __init__(self, increment: float = 0.1):
        self.increment = increment
        self.task_counts = {}

    def get_score(self, task_id: str) -> float:
        count = self.task_counts.get(task_id, 0)
        self.task_counts[task_id] = count + 1
        return min(1.0, count * self.increment)

    def reset(self) -> None:
        self.task_counts = {}


class ZeroScores(ScoreGenerator):
    """Always returns 0 regardless of task."""

    def get_score(self, task_id: str) -> float:
        return 0.0


class RandomScores(ScoreGenerator):
    """Returns random scores in [min_val, max_val]."""

    def __init__(self, seed: int = None, min_val: float = 0.0, max_val: float = 1.0):
        self.rng = random.Random(seed)
        self.min_val = min_val
        self.max_val = max_val

    def get_score(self, task_id: str) -> float:
        return self.rng.uniform(self.min_val, self.max_val)


class ConditionalLinearScores(ScoreGenerator):
    """Linear increasing scores for specified tasks, zero for others."""

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
    """Scores where secondary task only gives reward after primary reaches threshold."""

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
    """Run a curriculum test with controlled scores and collect detailed statistics."""
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

        # For complete_task, extract the base task ID
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
    if hasattr(curriculum, "stats"):
        curriculum_stats = curriculum.stats()

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
    """Create task weights dictionary for testing."""
    return {task_name: 1.0 for task_name in task_names}


def fake_curriculum_from_config_path(path, env_overrides=None):
    """Mock curriculum loading function."""
    from metta.mettagrid.curriculum.core import SingleTaskCurriculum
    
    base_config = OmegaConf.create({
        "game": {
            "num_agents": 5,
            "map": {"width": 10, "height": 10}
        }
    })
    task_cfg = OmegaConf.merge(base_config, env_overrides or {})
    assert isinstance(task_cfg, OmegaConf)
    return SingleTaskCurriculum(path, task_cfg=task_cfg)


@pytest.fixture
def free_port():
    """Get a free port for testing."""
    import socket
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


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
    """Standard environment configuration for tests."""
    return OmegaConf.create({
        "sampling": 0,
        "game": {
            "num_agents": 1,
            "map": {"width": 10, "height": 10}
        }
    })
