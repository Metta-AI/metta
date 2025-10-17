"""Lightweight task dependency graph utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, Generic, Iterable, List, MutableMapping, Optional, Sequence, Set, Tuple, TypeVar

__all__ = [
    "TaskGraph",
    "TaskGraphNode",
    "LearningTaskGraph",
    "LearningTaskData",
    "LearningSimulationResult",
    "LearningSimulationStep",
]

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


@dataclass(slots=True)
class LearningTaskData:
    difficulty: float
    competence: float = 0.0


@dataclass(slots=True)
class LearningSimulationStep:
    task_id: str
    learned: bool
    competence_gain: float
    competence_after: float
    prereq_competence: Dict[str, float]


@dataclass(slots=True)
class LearningSimulationResult:
    steps: List[LearningSimulationStep]
    final_competence: Dict[str, float]
    total_competence: float


class LearningTaskGraph(TaskGraph[LearningTaskData]):
    """Task graph with simple competence dynamics."""

    def __init__(
        self,
        *,
        base_learning_rate: float = 0.15,
        decay_rate: float = 0.02,
        prereq_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if base_learning_rate <= 0:
            raise ValueError("base_learning_rate must be positive")
        if decay_rate < 0:
            raise ValueError("decay_rate must be non-negative")
        if not 0 <= prereq_threshold <= 1:
            raise ValueError("prereq_threshold must be within [0, 1]")
        self.base_learning_rate = base_learning_rate
        self.decay_rate = decay_rate
        self.prereq_threshold = prereq_threshold

    def add_task(
        self,
        task_id: str,
        *,
        difficulty: float,
        dependencies: Optional[Iterable[str]] = None,
        initial_competence: float = 0.0,
    ) -> None:
        if not 0 <= difficulty <= 1:
            raise ValueError("difficulty must be within [0, 1]")
        if not 0 <= initial_competence <= 1:
            raise ValueError("initial_competence must be within [0, 1]")
        payload = LearningTaskData(difficulty=difficulty, competence=initial_competence)
        super().add_task(task_id, payload=payload, dependencies=dependencies)

    def get_competence(self, task_id: str) -> float:
        payload = self._get_payload(task_id)
        return payload.competence

    def set_competence(self, task_id: str, value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError("competence must be within [0, 1]")
        payload = self._get_payload(task_id)
        payload.competence = value

    def reset_competence(self) -> None:
        for node in self._nodes.values():
            if node.payload is not None:
                node.payload.competence = 0.0

    def record_experience(self, task_id: str, experience: float = 1.0) -> bool:
        if experience <= 0:
            raise ValueError("experience must be positive")

        payload = self._get_payload(task_id)
        prereqs = self.dependencies_of(task_id)
        prereq_competence = 1.0 if not prereqs else min(self.get_competence(dep) for dep in prereqs)

        learned = False
        if prereq_competence >= self.prereq_threshold:
            learning_rate = self.base_learning_rate * (1.0 - payload.difficulty)
            gain = learning_rate * prereq_competence * experience * (1.0 - payload.competence)
            if gain > 0:
                payload.competence = min(1.0, payload.competence + gain)
                learned = True

        self._apply_decay(task_id, experience)
        return learned

    def simulate_schedule(
        self,
        schedule: Sequence[str],
        *,
        experience: float = 1.0,
    ) -> LearningSimulationResult:
        snapshot = self._snapshot_competence()
        steps: List[LearningSimulationStep] = []

        for task_id in schedule:
            prereqs = self.dependencies_of(task_id)
            prereq_comp = {dep: self.get_competence(dep) for dep in prereqs}
            before = self.get_competence(task_id)
            learned = self.record_experience(task_id, experience)
            after = self.get_competence(task_id)
            steps.append(
                LearningSimulationStep(
                    task_id=task_id,
                    learned=learned,
                    competence_gain=max(0.0, after - before),
                    competence_after=after,
                    prereq_competence=prereq_comp,
                )
            )

        final_state = {task_id: self.get_competence(task_id) for task_id in self._nodes}
        total_comp = sum(final_state.values())
        self._restore_competence(snapshot)
        return LearningSimulationResult(steps=steps, final_competence=final_state, total_competence=total_comp)

    def compute_optimal_learning_order(self) -> List[str]:
        return self._greedy_optimal_order(list(self._nodes.keys()))

    def compute_regret(
        self,
        schedule: Sequence[str],
        *,
        experience: float = 1.0,
    ) -> Tuple[float, LearningSimulationResult, LearningSimulationResult]:
        optimal_order = self._greedy_optimal_order(schedule)
        optimal_result = self.simulate_schedule(optimal_order, experience=experience)
        actual_result = self.simulate_schedule(schedule, experience=experience)
        regret = max(0.0, optimal_result.total_competence - actual_result.total_competence)
        return regret, optimal_result, actual_result

    def _apply_decay(self, focused_task: str, experience: float) -> None:
        if self.decay_rate == 0:
            return
        decay_factor = max(0.0, 1.0 - self.decay_rate * experience)
        for task_id, node in self._nodes.items():
            if task_id == focused_task or node.payload is None:
                continue
            node.payload.competence = max(0.0, node.payload.competence * decay_factor)

    def _snapshot_competence(self) -> Dict[str, float]:
        return {task_id: node.payload.competence for task_id, node in self._nodes.items() if node.payload is not None}

    def _restore_competence(self, snapshot: Dict[str, float]) -> None:
        for task_id, competence in snapshot.items():
            self._get_payload(task_id).competence = competence

    def _get_payload(self, task_id: str) -> LearningTaskData:
        node = self._nodes.get(task_id)
        if node is None or node.payload is None:
            raise KeyError(f"Task '{task_id}' is not present in the graph")
        return node.payload

    def _greedy_optimal_order(self, multiset: Sequence[str]) -> List[str]:
        remaining = defaultdict(int)
        for task_id in multiset:
            if task_id not in self._nodes:
                raise KeyError(f"Task '{task_id}' is not present in the graph")
            remaining[task_id] += 1

        order: List[str] = []
        state = self._snapshot_competence()

        for _ in range(len(multiset)):
            best_task = None
            best_gain = -1.0
            for task_id, count in remaining.items():
                if count <= 0:
                    continue
                payload = self._get_payload(task_id)
                prereqs = self.dependencies_of(task_id)
                prereq_comp = 1.0 if not prereqs else min(state[dep] for dep in prereqs)
                if prereq_comp < self.prereq_threshold:
                    gain = 0.0
                else:
                    gain = self.base_learning_rate * (1.0 - payload.difficulty) * prereq_comp * (
                        1.0 - state[task_id]
                    )
                if gain > best_gain:
                    best_gain = gain
                    best_task = task_id

            if best_task is None:
                best_task = next(task for task, count in remaining.items() if count > 0)

            order.append(best_task)
            remaining[best_task] -= 1
            state = self._simulate_state_step(state, best_task)

        return order

    def _simulate_state_step(self, state: Dict[str, float], task_id: str, experience: float = 1.0) -> Dict[str, float]:
        new_state = dict(state)
        payload = self._get_payload(task_id)
        prereqs = self.dependencies_of(task_id)
        prereq_comp = 1.0 if not prereqs else min(new_state[dep] for dep in prereqs)
        if prereq_comp >= self.prereq_threshold:
            gain = self.base_learning_rate * (1.0 - payload.difficulty) * prereq_comp * experience * (
                1.0 - new_state[task_id]
            )
            if gain > 0:
                new_state[task_id] = min(1.0, new_state[task_id] + gain)

        if self.decay_rate:
            decay = max(0.0, 1.0 - self.decay_rate * experience)
            for tid in new_state:
                if tid == task_id:
                    continue
                new_state[tid] = max(0.0, new_state[tid] * decay)

        return new_state
