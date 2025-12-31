from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

Workspace = dict[str, Any]


@dataclass(slots=True)
class Node:
    name: str
    fn: Callable[[Any, Workspace], Mapping[str, Any] | None]
    deps: tuple[str, ...] = ()
    enabled: Callable[[Any, Workspace], bool] | None = None


class TrainingGraph:
    def __init__(self, nodes: list[Node]) -> None:
        self._nodes = {node.name: node for node in nodes}
        self._order = self._resolve_order(nodes)

    def run(self, context: Any, workspace: Workspace) -> None:
        for node in self._order:
            if node.enabled is not None and not node.enabled(context, workspace):
                continue
            outputs = node.fn(context, workspace)
            if outputs:
                workspace.update(outputs)

    def _resolve_order(self, nodes: list[Node]) -> list[Node]:
        order: list[Node] = []
        in_degree: dict[str, int] = {node.name: 0 for node in nodes}
        edges: dict[str, list[str]] = {node.name: [] for node in nodes}
        index: dict[str, int] = {node.name: idx for idx, node in enumerate(nodes)}

        for node in nodes:
            for dep in node.deps:
                if dep not in self._nodes:
                    raise KeyError(f"Unknown dependency '{dep}' for node '{node.name}'")
                edges[dep].append(node.name)
                in_degree[node.name] += 1

        queue: deque[str] = deque(sorted((name for name, deg in in_degree.items() if deg == 0), key=index.__getitem__))

        while queue:
            name = queue.popleft()
            order.append(self._nodes[name])
            for nxt in edges[name]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(nodes):
            remaining = [name for name, deg in in_degree.items() if deg > 0]
            raise RuntimeError(f"Cycle detected in training graph: {remaining}")

        return order
