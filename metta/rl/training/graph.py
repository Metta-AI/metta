from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Literal

Phase = Literal["rollout", "train", "both"]
Workspace = dict[str, Any]


@dataclass(slots=True)
class Node:
    name: str
    phase: Phase
    fn: Callable[[Any, Workspace], Mapping[str, Any] | None]
    next: tuple[str, ...] = ()
    enabled: Callable[[Any, Workspace], bool] | None = None


class TrainingGraph:
    def __init__(self, nodes: list[Node], *, roots: Iterable[str] | None = None) -> None:
        self._nodes = {node.name: node for node in nodes}
        self._order = self._resolve_order(nodes, roots=roots)

    def run(self, phase: Phase, context: Any, workspace: Workspace) -> None:
        for node in self._order:
            if node.phase != "both" and node.phase != phase:
                continue
            if node.enabled is not None and not node.enabled(context, workspace):
                continue
            outputs = node.fn(context, workspace)
            if outputs:
                workspace.update(outputs)

    def _resolve_order(self, nodes: list[Node], *, roots: Iterable[str] | None) -> list[Node]:
        has_edges = any(node.next for node in nodes)
        if not has_edges and roots is None:
            return list(nodes)

        inbound: set[str] = set()
        for node in nodes:
            inbound.update(node.next)

        if roots is None:
            root_list = [node.name for node in nodes if node.name not in inbound]
            if not root_list and nodes:
                root_list = [nodes[0].name]
        else:
            root_list = list(roots)

        order: list[Node] = []
        seen: set[str] = set()
        queue: deque[str] = deque(root_list)

        while queue:
            name = queue.popleft()
            if name in seen:
                continue
            node = self._nodes.get(name)
            if node is None:
                raise KeyError(f"Unknown node in graph traversal: {name}")
            seen.add(name)
            order.append(node)
            queue.extend(node.next)

        for node in nodes:
            if node.name not in seen:
                order.append(node)
        return order
