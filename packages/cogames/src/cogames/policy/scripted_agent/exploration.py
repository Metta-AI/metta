"""
Exploration strategies for scripted agents.

This module provides various exploration strategies that can be selected
via hyperparameters to control how agents explore unknown areas of the map.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Protocol

if TYPE_CHECKING:
    from mettagrid.simulator import Action

    from .types import SimpleAgentState

# ---- Strategy Interface -----------------------------------------------------


class Explorer(Protocol):
    """Protocol for exploration strategies."""

    def choose_action(self, s: SimpleAgentState) -> Action:
        """Return the next exploration action for the given agent state."""
        ...


# ---- Ops Adapter (decouples from BaselineAgentPolicyImpl) -------------------


@dataclass
class ExplorerOps:
    """
    Lightweight adapter exposing only what exploration strategies need.
    This avoids importing or depending on your policy class.
    """

    actions: any  # policy_env_info.actions (must have noop.Noop() and move.Move(dir))
    obs_hr: int
    obs_wr: int
    move_towards: Callable[[SimpleAgentState, tuple[int, int]], Action]
    move_towards_adj: Callable[[SimpleAgentState, tuple[int, int]], Action]  # reach_adjacent=True
    try_random_direction: Callable[[SimpleAgentState], Optional[Action]]
    is_traversable: Callable[[SimpleAgentState, int, int, type], bool]  # (s, r, c, CellType) -> bool
    CellType: type  # pass the enum class itself


# ---- Shared helpers ---------------------------------------------------------

_DIRS = {"north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1)}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _has_seen(s, r, c) -> bool:
    return hasattr(s, "seen") and 0 <= r < s.map_height and 0 <= c < s.map_width and s.seen[r][c]


def _is_free(s, ops: ExplorerOps, r, c) -> bool:
    H, W = s.map_height, s.map_width
    if not (0 <= r < H and 0 <= c < W):
        return False
    return s.occupancy[r][c] == ops.CellType.FREE.value


# ---- Directional Explorer (fallback) ---------------------------------------


class DirectionalExplorer(Explorer):
    """
    Simple directional exploration that picks a random direction and persists in it.
    Tries to continue in current direction; if blocked, picks a new random direction.
    """

    def __init__(self, ops: ExplorerOps):
        self.ops = ops

    def choose_action(self, s: SimpleAgentState) -> Action:
        # Try to continue in current direction
        if s.exploration_target and isinstance(s.exploration_target, str):
            dr, dc = _DIRS.get(s.exploration_target, (0, 0))
            if dr or dc:
                act = self.ops.move_towards(s, (s.row + dr, s.col + dc))
                if act != self.ops.actions.noop.Noop():
                    return act
                # Blocked - clear current direction and pick new random one
                s.cached_path = None
                s.cached_path_target = None
                s.exploration_target = None

        # Pick a random valid direction and set exploration_target so it persists
        directions = ["north", "south", "east", "west"]
        random.shuffle(directions)
        for direction in directions:
            dr, dc = _DIRS[direction]
            nr, nc = s.row + dr, s.col + dc
            if _is_free(s, self.ops, nr, nc):
                # Set exploration_target so direction persists
                s.exploration_target = direction
                return self.ops.move_towards(s, (nr, nc))

        # No valid direction found
        return self.ops.actions.noop.Noop()


# ---- STC Explorer (Spanning-Tree Coverage) ---------------------------------


class STCExplorer(Explorer):
    """
    Classic coverage: build a DFS spanning tree over discovered FREE cells.
    - Prefer stepping into any FREE neighbor that hasn't been 'covered' yet
    - Otherwise backtrack along parent pointers
    We store minimal state on 's' lazily: s._stc_parent, s._stc_covered
    """

    def __init__(self, ops: ExplorerOps):
        self.ops = ops

    def _ensure_state(self, s: SimpleAgentState):
        if not hasattr(s, "_stc_parent"):
            s._stc_parent = {}  # (r,c) -> (pr,pc)
        if not hasattr(s, "_stc_covered"):
            s._stc_covered = set()  # set[(r,c)]

    def _uncovered_neighbors(self, s: SimpleAgentState, r: int, c: int) -> list[tuple[int, int]]:
        nbs = []
        for nr, nc in _neighbors4(r, c):
            if _is_free(s, self.ops, nr, nc) and (nr, nc) not in s._stc_covered:
                nbs.append((nr, nc))
        return nbs

    def _backtrack_target(self, s: SimpleAgentState, start: tuple[int, int]) -> Optional[tuple[int, int]]:
        # climb parents until you find a node with an uncovered neighbor
        cur = start
        visited_guard = set()
        while cur and cur not in visited_guard:
            visited_guard.add(cur)
            r, c = cur
            if self._uncovered_neighbors(s, r, c):
                return cur
            cur = s._stc_parent.get(cur)
        return None

    def choose_action(self, s: SimpleAgentState) -> Action:
        self._ensure_state(s)
        cur = (s.row, s.col)
        s._stc_covered.add(cur)

        # link parent for newly reached nodes (only once)
        if cur not in s._stc_parent:
            # choose a parent as the closest previously covered neighbor, if any
            for nr, nc in _neighbors4(*cur):
                if _is_free(s, self.ops, nr, nc) and (nr, nc) in s._stc_covered:
                    s._stc_parent[cur] = (nr, nc)
                    break
            else:
                s._stc_parent[cur] = None  # root

        # 1) try to go deeper (any uncovered FREE neighbor)
        nbs = self._uncovered_neighbors(s, *cur)
        if nbs:
            # heuristic: pick the neighbor with most unseen around (encourages progress)
            def score(p):
                pr, pc = p
                return sum(1 for rr, cc in _neighbors4(pr, pc) if not _has_seen(s, rr, cc))

            nxt = max(nbs, key=score)
            # set parent of nxt if not set
            if nxt not in s._stc_parent:
                s._stc_parent[nxt] = cur
            return self.ops.move_towards(s, nxt)

        # 2) backtrack toward the nearest ancestor with uncovered neighbors
        target = self._backtrack_target(s, cur)
        if target is None:
            # nothing left to cover nearby; random nudge
            rnd = self.ops.try_random_direction(s)
            return rnd if rnd else self.ops.actions.noop.Noop()
        if target == cur:
            # we are already at branching node; pick neighbor now uncovered?
            nbs = self._uncovered_neighbors(s, *cur)
            if nbs:
                nxt = nbs[0]
                if nxt not in s._stc_parent:
                    s._stc_parent[nxt] = cur
                return self.ops.move_towards(s, nxt)
            # else fall through to random nudge
            rnd = self.ops.try_random_direction(s)
            return rnd if rnd else self.ops.actions.noop.Noop()

        return self.ops.move_towards(s, target)


# ---- Factory ----------------------------------------------------------------


def get_explorer(name: str, ops: ExplorerOps) -> Explorer:
    """
    Factory by string name. Add more strategies here as you implement them.

    Args:
        name: Strategy name - one of:
            - "stc", "spanning_tree", "coverage", "stc_explorer": Spanning-tree coverage
            - "directional", "random", "directional_explorer": Simple directional exploration (default)
        ops: ExplorerOps adapter providing movement primitives

    Returns:
        Explorer instance for the requested strategy
    """
    key = (name or "").lower()
    if key == "stc":
        return STCExplorer(ops)
    if key == "directional":
        return DirectionalExplorer(ops)
    # default to directional
    return DirectionalExplorer(ops)
