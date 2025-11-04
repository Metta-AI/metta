"""
CoordinatingAgent - Multi-agent coordination for assemblers and extractors.

Extends UnclippingAgent with:
- Smart mouth selection for assemblers and extractors (agents spread around stations)
- Collision avoidance via random unsticking when blocked
- Commitment to chosen mouths to prevent oscillation

This agent has all capabilities:
- Core: resource gathering, assembling, delivery (from SimpleBaselineAgent)
- Unclipping: detect and restore clipped extractors (from UnclippingAgent)
- Coordination: multi-agent collision avoidance (this class)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from .simple_baseline_agent import CellType
from .unclipping_agent import UnclippingAgent, UnclippingAgentState

if TYPE_CHECKING:
    from cogames.cogs_vs_clips.env import MettaGridEnv
    from cogames.cogs_vs_clips.observation import MettaGridObservation


class CoordinatingAgent(UnclippingAgent):
    """
    Multi-agent coordination via smart mouth selection.

    Extends UnclippingAgent (which extends SimpleBaselineAgent), so has:
    - Core gathering/assembly/delivery from SimpleBaselineAgent
    - Unclipping capability from UnclippingAgent
    - Coordination via smart mouth selection (this class)

    When within 2 cells of assembler or extractor, picks an available adjacent
    cell (mouth) from observations to avoid collisions. Agents commit to their
    chosen mouth to prevent oscillation.
    """

    def __init__(self, env: MettaGridEnv):
        super().__init__(env)

    def _random_move_if_stuck(self, s: UnclippingAgentState, action: int) -> int:
        """If action is NOOP (blocked), 30% chance to take a random move to unstick."""
        if action != self._NOOP:
            return action

        # 30% chance to take random move when stuck
        import random

        if random.random() < 0.3:
            valid_moves = []
            for move_action, (dr, dc) in [
                (self._MOVE_N, (-1, 0)),
                (self._MOVE_S, (1, 0)),
                (self._MOVE_E, (0, 1)),
                (self._MOVE_W, (0, -1)),
            ]:
                nr, nc = s.row + dr, s.col + dc
                if self._is_within_bounds(s, nr, nc) and self._is_passable(s, nr, nc):
                    valid_moves.append(move_action)

            if valid_moves:
                return random.choice(valid_moves)

        return self._NOOP

    def _do_assemble(self, s: UnclippingAgentState) -> int:
        """Override to add smart mouth selection when approaching assembler."""
        # Change glyph to heart if needed
        if s.current_glyph != "heart":
            vibe_action = self._change_vibe_actions["heart"]
            s.current_glyph = "heart"
            return vibe_action

        # Explore until we find assembler
        explore_action = self._explore_until(
            s, condition=lambda: self._stations["assembler"] is not None, reason="Need assembler"
        )
        if explore_action is not None:
            return explore_action

        # We know where assembler is
        assembler = self._stations["assembler"]
        ar, ac = assembler

        # Check if we're adjacent
        dr = abs(s.row - ar)
        dc = abs(s.col - ac)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        if is_adjacent:
            return self._move_into_cell(s, assembler)

        # Not adjacent yet - apply mouth coordination
        distance = abs(s.row - ar) + abs(s.col - ac)

        # Check if we have a committed mouth for this assembler
        committed_mouth = getattr(s, "committed_assembler_mouth", None)
        committed_target = getattr(s, "committed_assembler_target", None)

        # Clear commitment if target changed (shouldn't happen for assembler but for safety)
        if committed_target is not None and committed_target != assembler:
            committed_mouth = None
            committed_target = None
            s.committed_assembler_mouth = None
            s.committed_assembler_target = None

        # Mouth coordination: activate when distance <= 2, persist until used
        if distance <= 2:
            # If no commitment yet, find and commit to a mouth
            if committed_mouth is None:
                available_mouth = self._find_available_mouth(s, assembler)
                if available_mouth is not None:
                    # Commit to this mouth
                    s.committed_assembler_mouth = available_mouth
                    s.committed_assembler_target = assembler
                    committed_mouth = available_mouth
            else:
                # No available mouth, use default pathfinding
                return self._move_towards(s, assembler, reach_adjacent=True)

            # We have a committed mouth - move to it
            # Check if we're at the mouth
            if (s.row, s.col) == committed_mouth:
                # At mouth! Clear commitment and use assembler
                s.committed_assembler_mouth = None
                s.committed_assembler_target = None
                return self._move_into_cell(s, assembler)

            # Not at mouth yet - move toward it
            action = self._move_towards(s, committed_mouth, reach_adjacent=False)
            # If pathfinding fails, clear commitment and use default
            if action == self._NOOP:
                s.committed_assembler_mouth = None
                s.committed_assembler_target = None
                action = self._move_towards(s, assembler, reach_adjacent=True)
            # Add random motion to unstick from collisions
            return self._random_move_if_stuck(s, action)

        # If we have a commitment but moved too far, keep trying (allows paths that temporarily increase distance)
        if committed_mouth is not None and distance <= 4:
            # Check if at mouth
            if (s.row, s.col) == committed_mouth:
                s.committed_assembler_mouth = None
                s.committed_assembler_target = None
                return self._move_into_cell(s, assembler)

            # Move toward committed mouth
            action = self._move_towards(s, committed_mouth, reach_adjacent=False)
            if action == self._NOOP:
                # Can't reach mouth, clear and use default
                s.committed_assembler_mouth = None
                s.committed_assembler_target = None
                action = self._move_towards(s, assembler, reach_adjacent=True)
            # Add random motion to unstick from collisions
            return self._random_move_if_stuck(s, action)

        # Too far or no coordination - clear any commitment and use default
        s.committed_assembler_mouth = None
        s.committed_assembler_target = None
        action = self._move_towards(s, assembler, reach_adjacent=True)
        return self._random_move_if_stuck(s, action)

    def _do_gather(self, s: UnclippingAgentState) -> int:
        """Override to add smart mouth selection when approaching extractors."""
        # Use parent's logic for waiting, deficits check, and exploration
        if s.pending_use_resource is not None:
            return super()._do_gather(s)

        deficits = self._calculate_deficits(s)
        if all(d <= 0 for d in deficits.values()):
            return super()._do_gather(s)

        # Explore until we find an extractor (using parent's explore_until)
        explore_action = self._explore_until(
            s,
            condition=lambda: self._find_any_needed_extractor(s) is not None,
            reason=f"Need extractors for: {', '.join(k for k, v in deficits.items() if v > 0)}",
        )
        if explore_action is not None:
            return explore_action

        # Found an extractor
        result = self._find_any_needed_extractor(s)
        if result is None:
            return self._explore(s)

        extractor, resource_type = result
        s.target_resource = resource_type
        s.exploration_target = None  # Clear exploration target

        # Get extractor position
        er, ec = extractor.position

        # Check if we're adjacent
        dr = abs(s.row - er)
        dc = abs(s.col - ec)
        is_adjacent = (dr == 1 and dc == 0) or (dr == 0 and dc == 1)

        if is_adjacent:
            # Check if we still need this resource
            current_deficits = self._calculate_deficits(s)
            if current_deficits[resource_type] <= 0:
                # Don't use it - we have enough. Move away and find next extractor.
                return self._explore(s)

            # Check if extractor is ready
            if extractor.cooldown_remaining > 0 or extractor.converting:
                s.waiting_at_extractor = extractor.position
                s.wait_steps += 1
                return self._NOOP

            if extractor.remaining_uses == 0 or extractor.clipped:
                s.waiting_at_extractor = None
                s.wait_steps = 0
                return self._NOOP

            # Use the extractor
            old_amount = getattr(s, resource_type, 0)
            s.pending_use_resource = resource_type
            s.pending_use_amount = old_amount
            return self._move_into_cell(s, extractor.position)

        # Not adjacent yet - apply mouth coordination if close enough
        distance = abs(s.row - er) + abs(s.col - ec)

        # Check if we have a committed mouth for this extractor
        committed_mouth = getattr(s, "committed_mouth", None)
        committed_target = getattr(s, "committed_target", None)

        # Clear commitment if target changed
        if committed_target is not None and committed_target != extractor.position:
            committed_mouth = None
            committed_target = None
            s.committed_mouth = None
            s.committed_target = None

        # Mouth coordination: activate when distance <= 2, persist until used
        if distance <= 2:
            # If no commitment yet, find and commit to a mouth
            if committed_mouth is None:
                available_mouth = self._find_available_mouth(s, extractor.position)
                if available_mouth is not None:
                    # Commit to this mouth
                    s.committed_mouth = available_mouth
                    s.committed_target = extractor.position
                    committed_mouth = available_mouth
            else:
                # No available mouth, use default pathfinding
                return self._move_towards(s, extractor.position, reach_adjacent=True)

            # We have a committed mouth - move to it
            # Check if we're at the mouth
            if (s.row, s.col) == committed_mouth:
                # At mouth! Clear commitment and use extractor
                s.committed_mouth = None
                s.committed_target = None
                return self._move_into_cell(s, extractor.position)

            # Not at mouth yet - move toward it
            action = self._move_towards(s, committed_mouth, reach_adjacent=False)
            # If pathfinding fails, clear commitment and use default
            if action == self._NOOP:
                s.committed_mouth = None
                s.committed_target = None
                action = self._move_towards(s, extractor.position, reach_adjacent=True)
            # Add random motion to unstick from collisions
            return self._random_move_if_stuck(s, action)

        # If we have a commitment but moved too far, keep trying (allows paths that temporarily increase distance)
        if committed_mouth is not None and distance <= 4:
            # Check if at mouth
            if (s.row, s.col) == committed_mouth:
                s.committed_mouth = None
                s.committed_target = None
                return self._move_into_cell(s, extractor.position)

            # Move toward committed mouth
            action = self._move_towards(s, committed_mouth, reach_adjacent=False)
            if action == self._NOOP:
                # Can't reach mouth, clear and use default
                s.committed_mouth = None
                s.committed_target = None
                action = self._move_towards(s, extractor.position, reach_adjacent=True)
            # Add random motion to unstick from collisions
            return self._random_move_if_stuck(s, action)

        # Too far or no coordination - clear any commitment and use default
        s.committed_mouth = None
        s.committed_target = None
        action = self._move_towards(s, extractor.position, reach_adjacent=True)
        return self._random_move_if_stuck(s, action)

    def _find_available_mouth(self, s: UnclippingAgentState, target_pos: tuple[int, int]) -> tuple[int, int] | None:
        """
        Find an available adjacent cell (mouth) around a target (extractor/assembler/etc).

        Returns the position of a free adjacent cell, or None if can't determine.
        """
        tr, tc = target_pos

        # Four possible mouths (N, S, E, W)
        mouths = [
            (tr - 1, tc),  # North
            (tr + 1, tc),  # South
            (tr, tc - 1),  # West
            (tr, tc + 1),  # East
        ]

        # Check which mouths are free
        free_mouths = []

        for mouth_r, mouth_c in mouths:
            # Check if this position is in bounds
            if not (0 <= mouth_r < s.map_height and 0 <= mouth_c < s.map_width):
                continue

            # Check if it's passable (not a wall/obstacle)
            if not self._is_passable(s, mouth_r, mouth_c):
                continue

            # Check if there's an agent at this position
            occupied = False
            try:
                for _obj_id, obj in self._env.c_env.grid_objects().items():
                    obj_r = obj.get("r")
                    obj_c = obj.get("c")
                    obj_agent_id = obj.get("agent_id")

                    # Check if there's an agent at this mouth position (not us)
                    if obj_r == mouth_r and obj_c == mouth_c and obj_agent_id is not None:
                        if obj_agent_id != s.agent_id:
                            occupied = True
                            break
            except Exception:
                # If we can't check, assume it might be occupied
                pass

            if not occupied:
                free_mouths.append((mouth_r, mouth_c))

        if not free_mouths:
            return None

        # Pick the closest free mouth to our current position
        def distance(pos: tuple[int, int]) -> int:
            return abs(pos[0] - s.row) + abs(pos[1] - s.col)

        return min(free_mouths, key=distance)


# ============================================================================
# Policy Wrapper Classes
# ============================================================================


class CoordinatingAgentPolicy:
    """Per-agent policy wrapper."""

    def __init__(self, impl: CoordinatingAgent, agent_id: int):
        self._impl = impl
        self._agent_id = agent_id

    def step(self, obs: MettaGridObservation) -> int:
        """Compute action for this agent."""
        return self._impl.step(self._agent_id, obs)


class CoordinatingPolicy:
    """Policy wrapper for CoordinatingAgent with per-agent views."""

    def __init__(self, env: MettaGridEnv | None = None, device=None):
        self._env = env
        self._device = device  # Not used for scripted agents but needed for interface
        self._impl = CoordinatingAgent(env) if env is not None else None
        self._agent_policies: Dict[int, CoordinatingAgentPolicy] = {}

    def reset(self, obs, info):
        """Reset policy state."""
        # Get environment from info if not provided at init
        if self._env is None:
            self._env = info.get("env")

        # Initialize implementation if needed
        if self._impl is None:
            if self._env is None:
                raise RuntimeError("CoordinatingPolicy needs env - provide during __init__ or via info['env']")
            self._impl = CoordinatingAgent(self._env)

        # Reset agent states
        self._impl._agent_states.clear()
        self._agent_policies.clear()

    def agent_policy(self, agent_id: int) -> CoordinatingAgentPolicy:
        """Get policy for a specific agent."""
        if agent_id not in self._agent_policies:
            if self._impl is None:
                raise RuntimeError("Policy not initialized - call reset() first")

            # Initialize agent state if needed
            if agent_id not in self._impl._agent_states:
                self._impl._agent_states[agent_id] = UnclippingAgentState(
                    agent_id=agent_id,
                    map_height=self._impl._map_h,
                    map_width=self._impl._map_w,
                    occupancy=[[CellType.FREE.value] * self._impl._map_w for _ in range(self._impl._map_h)],
                )

            self._agent_policies[agent_id] = CoordinatingAgentPolicy(self._impl, agent_id)

        return self._agent_policies[agent_id]

    def agent_state(self, agent_id: int = 0) -> UnclippingAgentState:
        """Get state for an agent (for debugging/inspection)."""
        if agent_id not in self._impl._agent_states:
            self._impl._agent_states[agent_id] = UnclippingAgentState(
                agent_id=agent_id,
                map_height=self._impl._map_h,
                map_width=self._impl._map_w,
                occupancy=[[CellType.FREE.value] * self._impl._map_w for _ in range(self._impl._map_h)],
            )
        return self._impl._agent_states[agent_id]
