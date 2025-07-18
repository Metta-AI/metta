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

import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.curriculum_algorithm import CurriculumAlgorithm


# Task Interface
class Task:
    """Base class for tasks in a curriculum tree."""

    # Task State
    parent: Optional["Curriculum"] = None
    id: int = 0
    name: str = ""

    # Public API
    def sample(self) -> "Task":
        """Returns a potentially new instance of the task, which could include new diversity."""
        return self

    def complete(self, score: float):
        """Completes the task and updates the parent with the score."""
        if self.parent is not None:
            self.parent.complete_task(self.id, score)

    # Internal methods
    def _adopt(self, parent: "Curriculum", child_idx: int):
        """Set parent and id for this task."""
        self.parent = parent
        self.id = child_idx


class MettaGridTask(Task):
    def __init__(self, name: str, env_config: DictConfig):
        self.name = name
        self.env_config = env_config
        self.env_config_is_resolved = False

    def sample(self) -> "MettaGridTask":
        """
        Returns a new MettaGridTask with the env_config resolved, so that we can re-sample
        any randomness currently owned by OmegaConf.

        Note that no state (e.g. completion count) is stored on the returned task.
        The returned_task must still forward its completions to the same parent.
        """
        assert not self.env_config_is_resolved, "Env config is already resolved"
        resolved_copy = copy.copy(self)
        resolved_copy.env_config = OmegaConf.create(self.env_config)
        OmegaConf.resolve(resolved_copy.env_config)
        assert resolved_copy.env_config is not None, "Env config is None"
        resolved_copy.env_config_is_resolved = True
        return resolved_copy


class Curriculum(Task):
    def __init__(
        self,
        name: str,
        curriculum_algorithm: CurriculumAlgorithm,
        tasks: list[Task],
    ):
        self.name = name
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.curriculum_algorithm = curriculum_algorithm
        self.completed_tasks = np.zeros(self.num_tasks, dtype=np.int32)
        self.sampled_tasks = np.zeros(self.num_tasks, dtype=np.int32)
        self.total_completed_tasks = 0
        self.total_sampled_tasks = 0
        if self.num_tasks == 0:
            raise ValueError("Curriculum must have at least one task")

        # Set parent references
        for i, task in enumerate(tasks):
            task._adopt(self, i)

    def sample(self) -> MettaGridTask:
        task_idx = self.curriculum_algorithm.sample_idx()
        selected_task = self.tasks[task_idx]
        self.sampled_tasks[task_idx] += 1
        self.total_sampled_tasks += 1
        if isinstance(selected_task, MettaGridTask):
            return selected_task
        else:
            return selected_task.sample()

    def full_name(self, task_idx: int) -> str:
        name = self.tasks[task_idx].name
        return f"{self.name}/{name}"

    def complete_task(self, task_idx: int, score: float):
        self.completed_tasks[task_idx] += 1
        self.total_completed_tasks += 1
        self.curriculum_algorithm.update(task_idx, score)
        if self.parent is not None:
            self.parent.complete_task(self.id, score)

    def get_completion_rates(self) -> dict[str, int]:
        if self.total_completed_tasks != 0:
            return self._completion_dict_with_prefix("task_completions/")
        else:
            return dict()

    def get_sample_rates(self) -> dict[str, int]:
        if self.total_sampled_tasks != 0:
            return self._sample_dict_with_prefix("task_samples/")
        else:
            return dict()

    def get_task_probabilities(self, relative_to_root: bool = False) -> dict[str, float]:
        return self._probability_dict_with_prefix(relative_to_root=relative_to_root)

    def get_curriculum_stats(self) -> dict[str, float]:
        # TODO: make recursive and include stats from children
        return self.curriculum_algorithm.stats()

    def _completion_dict_with_prefix(self, prefix: str = "") -> dict[str, int]:
        result = dict()
        for task_idx in range(self.num_tasks):
            task = self.tasks[task_idx]
            task_path = f"{prefix}{task.name}"
            result[task_path] = self.completed_tasks[task_idx]
            if isinstance(task, Curriculum):
                result.update(task._completion_dict_with_prefix(f"{task_path}/"))
        return result

    def _sample_dict_with_prefix(self, prefix: str = "") -> dict[str, int]:
        result = dict()
        for task_idx in range(self.num_tasks):
            task = self.tasks[task_idx]
            task_path = f"{prefix}{task.name}"
            result[task_path] = self.sampled_tasks[task_idx]
            if isinstance(task, Curriculum):
                result.update(task._sample_dict_with_prefix(f"{task_path}/"))
        return result

    def _probability_dict_with_prefix(
        self, prefix: str = "", relative_to_root: bool = False, base_prob: float = 1.0
    ) -> dict[str, float]:
        """
        If relative_to_root is True, then the probabiltiies for each chld are for the full path from root to child.
        If relative_to_root is False, then the probabiltiies are for conditional on having reached this node.
        If base_prob is provided, then the probabilities are multiplied by this value.
        """
        probs = {}
        for task_idx in range(self.num_tasks):
            task = self.tasks[task_idx]
            task_prob = self.curriculum_algorithm.probabilities[task_idx]
            if relative_to_root:
                task_prob = task_prob * base_prob
            task_path = f"{prefix}{task.name}"
            probs[task_path] = task_prob
            if isinstance(task, Curriculum):
                probs.update(
                    task._probability_dict_with_prefix(
                        f"{task_path}/",
                        relative_to_root,
                        self.curriculum_algorithm.probabilities[task_idx] * base_prob,
                    )
                )

        return probs

    def __repr__(self) -> str:
        """Return a tree representation showing structure, weights, and algorithms."""
        return self._tree_repr(indent=0)

    def _tree_repr(self, indent: int = 0, prefix: str = "") -> str:
        """Recursive helper to build tree representation."""
        indent_str = "  " * indent
        lines = []

        # Current node info
        algo_name = type(self.curriculum_algorithm).__name__
        lines.append(f"{indent_str}{prefix}Curriculum({algo_name})")

        # Show weights and probabilities for children
        for i, task in enumerate(self.tasks):
            weight = self.curriculum_algorithm.weights[i]
            prob = self.curriculum_algorithm.probabilities[i]
            task_name = self.full_name(i)

            # Prefix for last child vs others
            is_last = i == len(self.tasks) - 1
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
                lines.append(f"{task_info} -> {task.name}")

        return "\n".join(lines)
