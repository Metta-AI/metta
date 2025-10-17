"""Lightweight task dependency graph utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, Generic, Iterable, List, MutableMapping, Optional, Set, TypeVar

__all__ = ["TaskGraph", "TaskGraphNode"]

T = TypeVar("T")


@dataclass(slots=True)
class TaskGraphNode(Generic[T]):
    """Representation of a task node stored within :class:`TaskGraph`."""

    task_id: str
    payload: Optional[T] = None


class TaskGraph(Generic[T]):
    """Directed acyclic graph structure for modelling task dependencies."""

    def __init__(self) -> None:
        self._nodes: Dict[str, TaskGraphNode[T]] = {}
        self._dependencies: MutableMapping[str, Set[str]] = defaultdict(set)
        self._dependents: MutableMapping[str, Set[str]] = defaultdict(set)

    def add_task(
        self,
        task_id: str,
        *,
        payload: Optional[T] = None,
        dependencies: Optional[Iterable[str]] = None,
    ) -> None:
        """Add or update a task node and its dependency edges."""

        if task_id not in self._nodes:
            self._nodes[task_id] = TaskGraphNode(task_id=task_id, payload=payload)
        elif payload is not None:
            self._nodes[task_id] = TaskGraphNode(task_id=task_id, payload=payload)

        new_dependencies = set(dependencies or [])

        # Remove edges that no longer exist.
        for removed in self._dependencies.get(task_id, set()) - new_dependencies:
            self._dependents[removed].discard(task_id)

        # Ensure dependency nodes exist and update adjacency maps.
        dep_set = self._dependencies[task_id]
        dep_set.clear()
        for dependency in new_dependencies:
            if dependency not in self._nodes:
                self._nodes[dependency] = TaskGraphNode(task_id=dependency)
            dep_set.add(dependency)
            self._dependents[dependency].add(task_id)

        # Guarantee all nodes appear in adjacency tables.
        self._dependents.setdefault(task_id, set())

    def dependencies_of(self, task_id: str) -> Set[str]:
        """Return the dependency set for *task_id* (copy)."""

        return set(self._dependencies.get(task_id, set()))

    def dependents_of(self, task_id: str) -> Set[str]:
        """Return tasks that depend on *task_id* (copy)."""

        return set(self._dependents.get(task_id, set()))

    def get_payload(self, task_id: str) -> Optional[T]:
        """Return the payload associated with *task_id*."""

        node = self._nodes.get(task_id)
        if node is None:
            raise KeyError(f"Task '{task_id}' is not present in the graph")
        return node.payload

    def ready_tasks(self, completed: Optional[Set[str]] = None) -> List[str]:
        """Return tasks whose dependencies are satisfied by *completed*."""

        completed_set = set(completed or set())
        ready: List[str] = []
        for task_id, dependencies in self._dependencies.items():
            if task_id in completed_set:
                continue
            if dependencies.issubset(completed_set):
                ready.append(task_id)
        # Include isolated nodes with no dependencies.
        for task_id in self._nodes:
            if task_id in completed_set:
                continue
            if task_id not in self._dependencies or not self._dependencies[task_id]:
                if all(task_id != existing for existing in ready):
                    ready.append(task_id)
        return sorted(set(ready))

    def topological_order(self) -> List[str]:
        """Return a deterministic topological ordering.

        Raises:
            ValueError: if the graph contains a cycle.
        """

        in_degree: Dict[str, int] = {task_id: len(deps) for task_id, deps in self._dependencies.items()}
        for task_id in self._nodes:
            in_degree.setdefault(task_id, 0)

        queue: List[str] = []
        for task_id, degree in in_degree.items():
            if degree == 0:
                heappush(queue, task_id)

        order: List[str] = []
        while queue:
            current = heappop(queue)
            order.append(current)
            for dependent in sorted(self._dependents.get(current, set())):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    heappush(queue, dependent)

        if len(order) != len(self._nodes):
            raise ValueError("Task graph contains a cycle")

        return order

    def __contains__(self, task_id: str) -> bool:  # pragma: no cover - trivial
        return task_id in self._nodes

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._nodes)
