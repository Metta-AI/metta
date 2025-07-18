"""
Curriculum system for adaptive task selection in MettaGrid.

The curriculum is organized as a tree structure with two node types:
* MettaGridTask: Leaf nodes representing specific environment configurations
* Curriculum: Internal nodes that manage collections of tasks or sub-curricula

Key concepts:
- Tasks are sampled probabilistically based on weights managed by curriculum algorithms
- Each Curriculum node can use a different algorithm (random, learning progress, etc.)
- Performance scores propagate up the tree, allowing each level to adapt its sampling
- Weights are automatically normalized and must remain non-negative

Example structure:
    Root Curriculum (ProgressiveAlgorithm)
    ├── Navigation Curriculum (LearningProgressAlgorithm)
    │   ├── MettaGridTask: easy_maze
    │   └── MettaGridTask: hard_maze
    └── Combat Curriculum (DiscreteRandomAlgorithm)
        ├── MettaGridTask: 1v1_arena
        └── MettaGridTask: team_battle
"""

import copy
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.curriculum_algorithm import CurriculumAlgorithm
from metta.mettagrid.curriculum.curriculum_stats import CurriculumStats


# Task Interface
class Task:
    """Base class for tasks in a curriculum tree."""

    #
    # Task State
    #

    _parent: Optional["Curriculum"] = None
    _id: int = 0
    _name: str = ""

    def __init__(self, name: str):
        self._name = name

    #
    # Primary API
    #

    def complete(self, score: float):
        """Completes the task and updates the parent with the score."""
        if self._parent is not None:
            self._parent.complete_task(self._id, score)

    #
    # Accessors
    #

    def short_name(self) -> str:
        return self._name

    def full_name(self) -> str:
        if self._parent is None:
            return self._name
        parent_name = self._parent.full_name()
        if parent_name == "":
            return self._name
        else:
            return f"{parent_name}/{self._name}"

    def parent(self) -> Optional["Curriculum"]:
        return self._parent

    def root(self) -> "Curriculum":
        if self._parent is None:
            return self
        else:
            return self._parent.root()

    #
    # Internal methods
    #

    def set_parent(self, parent: "Curriculum", child_idx: int):
        """Set parent and id for this task."""
        self._parent = parent
        self._id = child_idx

    def is_leaf(self) -> bool:
        return True


class MettaGridTask(Task):
    """
    A task that represents a specific environment configuration in MettaGrid.
    """

    def __init__(self, name: str, env_config: DictConfig):
        super().__init__(name)
        self._unresolved_env_config = env_config

    def env_config(self) -> DictConfig:
        """
        Provide a new env config with all OmegaConf-included randomness resolved.
        """
        resolved_env_config = copy.copy(self._unresolved_env_config)
        OmegaConf.resolve(resolved_env_config)
        assert resolved_env_config is not None, "Env config is None"
        return resolved_env_config


class Curriculum(Task):
    """
    A curriculum is a system for selecting tasks.
    """

    #
    # Init
    #

    def __init__(
        self,
        name: str,
        algorithm: CurriculumAlgorithm,
        tasks: list[Task],
    ):
        super().__init__(name)
        self._tasks = tasks
        num_tasks = len(tasks)
        self._algorithm = algorithm
        if num_tasks == 0:
            raise ValueError("Curriculum must have at least one task")
        self._stats = CurriculumStats(self, num_tasks)

        # Set parent references
        for i, task in enumerate(tasks):
            task.set_parent(self, i)

    def is_leaf(self) -> bool:
        return False

    #
    # Primary API
    # 1) sample() called by the trainer to generate a new task to train on
    # 2) complete_task() called, via sample().complete(), by the trainer
    # to update the curriculum with the score of the task
    #

    def sample(self) -> MettaGridTask:
        task_idx = self._algorithm.sample_idx()
        selected_task = self._tasks[task_idx]
        self._stats.record_sample(task_idx)
        if isinstance(selected_task, MettaGridTask):
            return selected_task
        else:
            return selected_task.sample()

    def complete_task(self, task_idx: int, score: float):
        self._stats.record_completion(task_idx)
        self._algorithm.update(task_idx, score)
        if self._parent is not None:
            self._parent.complete_task(self._id, score)

    #
    # Accessors
    #

    def algorithm(self) -> CurriculumAlgorithm:
        return self._algorithm

    def stats(self) -> CurriculumStats:
        return self._stats

    def tasks(self) -> list[Task]:
        return self._tasks

    #
    # Debugging
    #

    def __repr__(self) -> str:
        """Return a tree representation showing structure, weights, and algorithms."""
        return self._tree_repr(indent=0)

    def _tree_repr(self, indent: int = 0, prefix: str = "") -> str:
        """Recursive helper to build tree representation."""
        indent_str = "  " * indent
        lines = []

        # Current node info
        algo_name = type(self._algorithm).__name__
        lines.append(f"{indent_str}{prefix}Curriculum({algo_name})")

        # Show weights and probabilities for children
        for i, task in enumerate(self._tasks):
            weight = self._algorithm.weights[i]
            prob = self._algorithm.probabilities[i]
            task_name = self._tasks[i].full_name()

            # Prefix for last child vs others
            is_last = i == len(self._tasks) - 1
            branch = "└─" if is_last else "├─"
            continuation = "  " if is_last else "│ "

            # Child info
            task_info = f"{indent_str}{branch} [{task_name}] w={weight:.3f} p={prob:.3f}"

            if isinstance(task, Curriculum):
                # Recursive case
                lines.append(task_info)
                subtree = task._tree_repr(indent + 1, prefix=continuation)
                lines.append(subtree)
            else:
                # Leaf case (MettaGridTask)
                lines.append(f"{task_info} -> {task.short_name()}")

        return "\n".join(lines)
