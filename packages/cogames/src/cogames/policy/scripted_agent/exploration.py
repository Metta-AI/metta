"""
Exploration strategies for scripted agents.

This module provides various exploration strategies that can be selected
via hyperparameters to control how agents explore unknown areas of the map.
"""

from __future__ import annotations

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
_RIGHT_OF = {"north": "east", "east": "south", "south": "west", "west": "north"}
_LEFT_OF = {"north": "west", "west": "south", "south": "east", "east": "north"}


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
    Simple directional exploration that mirrors the original behavior.
    Keeps logic minimal: try continuing in a direction; else pick a random free step.
    """

    def __init__(self, ops: ExplorerOps):
        self.ops = ops

    def choose_action(self, s: SimpleAgentState) -> Action:
        if s.exploration_target and isinstance(s.exploration_target, str):
            dr, dc = _DIRS.get(s.exploration_target, (0, 0))
            if dr or dc:
                act = self.ops.move_towards(s, (s.row + dr, s.col + dc))
                if act != self.ops.actions.noop.Noop():
                    return act
                s.cached_path = None
                s.cached_path_target = None
                s.exploration_target = None

        rnd = self.ops.try_random_direction(s)
        return rnd if rnd else self.ops.actions.noop.Noop()


# ---- Frontier Explorer (uses seen[] if present) ----------------------------


class FrontierExplorer(Explorer):
    """
    Utility-guided frontier exploration:
    - A frontier is any UNSEEN cell with a FREE neighbor.
    - Score = 10 * (unseen_neighbor_count) - 1 * distance - 2 * crowd_penalty
    - Caches a target_position := the FREE neighbor adjacent to the frontier.
    """

    def __init__(self, ops: ExplorerOps):
        self.ops = ops

    def _collect_frontiers(self, s: SimpleAgentState) -> list[tuple[int, int, int, int]]:
        """
        Returns a list of tuples (fr_r, fr_c, adj_r, adj_c) where (fr_r,fr_c) is UNSEEN
        and (adj_r,adj_c) is an adjacent FREE cell we can aim for.
        Scans a local window around the agent for speed.
        """
        H, W = s.map_height, s.map_width
        pad = 4
        r0 = max(0, s.row - self.ops.obs_hr - pad)
        r1 = min(H, s.row + self.ops.obs_hr + pad + 1)
        c0 = max(0, s.col - self.ops.obs_wr - pad)
        c1 = min(W, s.col + self.ops.obs_wr + pad + 1)

        frontiers: list[tuple[int, int, int, int]] = []
        # Unknown means "not seen" if seen[][] exists; otherwise we gracefully skip
        uses_seen = hasattr(s, "seen")
        if not uses_seen:
            return frontiers  # no unknown notion: let factory default to other strategies

        for r in range(r0, r1):
            for c in range(c0, c1):
                if s.seen[r][c]:
                    continue  # already known
                # needs at least one FREE approach cell
                for ar, ac in _neighbors4(r, c):
                    if _is_free(s, self.ops, ar, ac):
                        frontiers.append((r, c, ar, ac))
                        break
        return frontiers

    def _score_frontier(self, s: SimpleAgentState, fr: tuple[int, int, int, int]) -> float:
        fr_r, fr_c, ar, ac = fr
        dist = _manhattan((s.row, s.col), (ar, ac))
        info = 0
        H, W = s.map_height, s.map_width
        for nr, nc in _neighbors4(fr_r, fr_c):
            if 0 <= nr < H and 0 <= nc < W and not _has_seen(s, nr, nc):
                info += 1
        crowd_pen = 5 if (ar, ac) in getattr(s, "agent_occupancy", set()) else 0
        return 10 * info - 1.0 * dist - 2.0 * crowd_pen

    def choose_action(self, s: SimpleAgentState) -> Action:
        if s.target_position:
            act = self.ops.move_towards(s, s.target_position)
            if act != self.ops.actions.noop.Noop():
                return act
            s.cached_path = None
            s.cached_path_target = None
            s.exploration_target = None
            s.target_position = None

        frontiers = self._collect_frontiers(s)
        if not frontiers:
            rnd = self.ops.try_random_direction(s)
            return rnd if rnd else self.ops.actions.noop.Noop()

        best = max(frontiers, key=lambda fr: self._score_frontier(s, fr))
        _, _, ar, ac = best
        s.target_position = (ar, ac)
        return self.ops.move_towards(s, s.target_position)


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


# ---- Right-Hand Wall Follower ----------------------------------------------


class RightHandWallFollower(Explorer):
    """
    Right-hand rule using a lightweight heading.
    - Maintains s._wf_heading in {"north","east","south","west"} (lazy init to "east")
    - Chooses next move by checking (right, forward, left, back) for FREE cells
    - Uses move_towards to respect collisions and your cached BFS checks
    """

    def __init__(self, ops: ExplorerOps):
        self.ops = ops

    def _ensure_heading(self, s: SimpleAgentState):
        if not hasattr(s, "_wf_heading") or s._wf_heading not in _DIRS:
            s._wf_heading = "east"

    def _step_towards_dir(self, s: SimpleAgentState, dir_name: str) -> Action:
        dr, dc = _DIRS[dir_name]
        return self.ops.move_towards(s, (s.row + dr, s.col + dc))

    def choose_action(self, s: SimpleAgentState) -> Action:
        self._ensure_heading(s)
        h = s._wf_heading
        right = _RIGHT_OF[h]
        left = _LEFT_OF[h]
        forward = h
        back = _RIGHT_OF[right]  # opposite

        # preference: right -> forward -> left -> back
        for choice in (right, forward, left, back):
            dr, dc = _DIRS[choice]
            nr, nc = s.row + dr, s.col + dc
            if _is_free(s, self.ops, nr, nc):
                s._wf_heading = choice
                act = self._step_towards_dir(s, choice)
                if act != self.ops.actions.noop.Noop():
                    return act
                # if blocked by dynamic agent, try next choice
        # fallback
        rnd = self.ops.try_random_direction(s)
        return rnd if rnd else self.ops.actions.noop.Noop()


# ---- Factory ----------------------------------------------------------------


def get_explorer(name: str, ops: ExplorerOps) -> Explorer:
    """
    Factory by string name. Add more strategies here as you implement them.

    Args:
        name: Strategy name - one of:
            - "frontier" or "frontier_explorer": Utility-guided frontier exploration (default)
            - "stc", "spanning_tree", "coverage": Spanning-tree coverage
            - "wall", "right_hand", "wall_follower": Right-hand wall following
            - "directional", "random": Simple directional exploration
        ops: ExplorerOps adapter providing movement primitives

    Returns:
        Explorer instance for the requested strategy
    """
    key = (name or "").lower()
    if key in ("frontier", "frontier_explorer"):
        return FrontierExplorer(ops)
    if key in ("stc", "spanning_tree", "coverage", "stc_explorer"):
        return STCExplorer(ops)
    if key in ("wall", "right_hand", "wall_follower"):
        return RightHandWallFollower(ops)
    if key in ("directional", "random", "directional_explorer"):
        return DirectionalExplorer(ops)
    # default
    return FrontierExplorer(ops)
