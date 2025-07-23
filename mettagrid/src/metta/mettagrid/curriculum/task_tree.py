"""
TaskTrees are trees where nodes have two possible types:
* MettaGridTask: A specific environment configuration for MettaGrid
* TaskTree: The root of a task tree whose children are either MettaGridTask or TaskTrees

All leaves are MettaGridTasks; all parents are TaskTrees.

A TaskTree has an associated curriculum algorithm, which is used to update the weights of the children of the TaskTree.
Each TaskTree node can has its own curriculum algorithm.

Scores are propagated up the tree so that each rebalances
the weights for its children based on the score input at the leaf.

Sample queries are propagated down the tree so that each node can sample from its children. All TaskTrees sample
based on a weighted random distribution; curricula are responsible for rebalancing the weights. Weights are
automatically normalized to sum to 1 but should always be non-negative.
"""

import copy
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.curriculum_algorithm import CurriculumAlgorithm


# TaskTreeNode is the base class for nodes in a task graph
class TaskTreeNode(ABC):
    parent: Optional["TaskTree"] = None
    child_index: Optional[int] = None
    name: str

    @abstractmethod
    def sample(self) -> "MettaGridTask":
        """Sample a task from the node, either directly or by sampling a from a child."""
        pass

    def set_as_child(self, parent: "TaskTree", index: int):
        """Set the parent and child index of the node. Used by TaskTree.__init__ as
        graphs are constructed bottom-up."""
        self.parent = parent
        self.child_index = index


class MettaGridTask(TaskTreeNode):
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

    def complete(self, score: float):
        if self.parent is not None:
            assert self.child_index is not None, "Child index must be set when completing task with parent"
            self.parent.complete(self.child_index, score)

    def short_name(self) -> str:
        """Return a short version of the task name for logging."""
        # Use the full name as short name for now
        # In the future, this could extract just the last component
        return self.name


class TaskTree(TaskTreeNode):
    def __init__(
        self,
        name: str,
        curriculum_algorithm: CurriculumAlgorithm,
        children: list[TaskTreeNode],
    ):
        self.name = name
        self.children = children
        self.num_children = len(children)
        self.curriculum_algorithm = curriculum_algorithm
        self.completed_tasks = np.zeros(self.num_children, dtype=np.int32)
        self.sampled_tasks = np.zeros(self.num_children, dtype=np.int32)
        self.total_completed_tasks = 0
        self.total_sampled_tasks = 0
        if self.num_children == 0:
            raise ValueError("TaskTree must have at least one child")

        # Set parent references
        for i, child in enumerate(children):
            child.set_as_child(self, i)

    def sample(self) -> MettaGridTask:
        child_idx = self.curriculum_algorithm.sample_idx()
        selected_child = self.children[child_idx]
        self.sampled_tasks[child_idx] += 1
        self.total_sampled_tasks += 1
        return selected_child.sample()

    def full_name(self, child_idx: int) -> str:
        name = self.children[child_idx].name
        return f"{self.name}/{name}"

    def complete(self, child_idx: int, score: float, name: Optional[str] = None):
        self.completed_tasks[child_idx] += 1
        self.total_completed_tasks += 1
        self.curriculum_algorithm.update(child_idx, score)
        if self.parent is not None:
            self.parent.complete(self.child_index, score)

    def get_total_completions(self) -> dict[str, int]:
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

    def get_algorithm_stats(self) -> dict[str, float]:
        # TODO: make recursive and include stats from children
        return self.curriculum_algorithm.stats()

    def _completion_dict_with_prefix(self, prefix: str = "") -> dict[str, int]:
        result = dict()
        for child_idx in range(self.num_children):
            child = self.children[child_idx]
            child_path = f"{prefix}{child.name}"
            result[child_path] = self.completed_tasks[child_idx]
            if isinstance(child, TaskTree):
                result.update(child._completion_dict_with_prefix(f"{child_path}/"))
        return result

    def _sample_dict_with_prefix(self, prefix: str = "") -> dict[str, int]:
        result = dict()
        for child_idx in range(self.num_children):
            child = self.children[child_idx]
            child_path = f"{prefix}{child.name}"
            result[child_path] = self.sampled_tasks[child_idx]
            if isinstance(child, TaskTree):
                result.update(child._sample_dict_with_prefix(f"{child_path}/"))
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
        for child_idx in range(self.num_children):
            child = self.children[child_idx]
            child_prob = self.curriculum_algorithm.probabilities[child_idx]
            if relative_to_root:
                child_prob = child_prob * base_prob
            child_path = f"{prefix}{child.name}"
            probs[child_path] = child_prob
            if isinstance(child, TaskTree):
                probs.update(
                    child._probability_dict_with_prefix(
                        f"{child_path}/",
                        relative_to_root,
                        self.curriculum_algorithm.probabilities[child_idx] * base_prob,
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
        lines.append(f"{indent_str}{prefix}TaskTree({algo_name})")

        # Show weights and probabilities for children
        for i, child in enumerate(self.children):
            weight = self.curriculum_algorithm.weights[i]
            prob = self.curriculum_algorithm.probabilities[i]
            child_name = self.full_name(i)

            # Prefix for last child vs others
            is_last = i == len(self.children) - 1
            branch = "└─" if is_last else "├─"
            continuation = "  " if is_last else "│ "

            # Child info
            child_info = f"{indent_str}{branch} [{child_name}] w={weight:.3f} p={prob:.3f}"

            if isinstance(child, TaskTree):
                # Recursive case
                lines.append(child_info)
                subtree = child._tree_repr(indent + 1, prefix=continuation)
                lines.append(subtree)
            else:
                # Leaf case (MettaGridTask)
                lines.append(f"{child_info} -> {child.name}")

        return "\n".join(lines)


#
# Helper functions for task set generation
#


def _expand_buckets(buckets: dict[str, dict[str, Any]]) -> dict[str, list[Any]]:
    """
    Expand bucket specifications into lists of values.

    Args:
        buckets: Dict mapping parameter paths to bucket specifications

    Returns:
        Dict mapping parameter paths to lists of values
    """

    buckets_unpacked = {}
    for parameter, bucket_spec in buckets.items():
        if "values" in bucket_spec:
            buckets_unpacked[parameter] = bucket_spec["values"]
        elif "range" in bucket_spec:
            lo, hi = bucket_spec["range"]
            if "bins" not in bucket_spec:
                raise ValueError(f"'bins' is required for range specification in parameter '{parameter}'")
            n = int(bucket_spec["bins"])
            if n < 2:
                raise ValueError(
                    f"'bins' must be >= 2 for parameter '{parameter}'. For a single value, use env_overrides instead."
                )
            step = (hi - lo) / n
            want_int = isinstance(lo, int) and isinstance(hi, int)

            binned_ranges = []
            for i in range(n):
                lo_i, hi_i = lo + i * step, lo + (i + 1) * step
                binned_ranges.append({"range": (lo_i, hi_i), "want_int": want_int})

            buckets_unpacked[parameter] = binned_ranges
        else:
            raise ValueError(f"Invalid bucket spec: {bucket_spec}")
    return buckets_unpacked


def _sample_from_bucket_value(value: Any) -> Any:
    """Sample a concrete value from a bucket value specification."""
    import numpy as np

    if isinstance(value, dict) and "range" in value:
        lo, hi = value["range"]
        sampled = np.random.uniform(lo, hi)
        if value.get("want_int", False):
            sampled = int(sampled)
        return sampled
    return value


def _get_bucket_id(parameters: list[str], values: list[Any]) -> str:
    """Generate a unique ID for a parameter combination."""
    id_parts = []
    for param, value in zip(parameters, values, strict=False):
        # Use full parameter path to ensure uniqueness

        # Format value for ID
        if isinstance(value, dict) and "range" in value:
            lo, hi = value["range"]
            if isinstance(lo, float) or isinstance(hi, float):
                value_str = f"({lo:.3f},{hi:.3f})"
            else:
                value_str = f"({lo},{hi})"
        elif isinstance(value, float):
            value_str = f"{value:.3f}"
        else:
            value_str = str(value)

        id_parts.append(f"{param}={value_str}")

    return ";".join(id_parts)


def _get_short_name(full_name: str) -> str:
    """Extract the short name from a full path.

    For example:
    - "/env/easy" -> "easy"
    - "path/to/navigation.yaml" -> "navigation"
    """
    # Get the last component
    name = full_name.split("/")[-1] if "/" in full_name else full_name

    # Remove file extension if present
    if "." in name:
        name = name.split(".")[0]

    return name
