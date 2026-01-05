"""
Scrambler role for CoGsGuard.

Scramblers find enemy-aligned supply depots and scramble them to take control.
With scrambler gear, they get +200 HP.

Strategy:
- Find ALL chargers on the map
- Prioritize scrambling enemy (clips) aligned chargers
- Systematically work through all chargers to take them over
- Check energy before moving to targets
- Retry failed scramble actions up to MAX_RETRIES times
"""

from __future__ import annotations

from typing import Optional

from cogames.policy.scripted_agent.utils import is_adjacent
from mettagrid.simulator import Action

from .policy import DEBUG, CogsguardAgentPolicyImpl
from .types import CogsguardAgentState, Role, StructureType

# Maximum number of times to retry a failed scramble action
MAX_RETRIES = 3


class ScramblerAgentPolicyImpl(CogsguardAgentPolicyImpl):
    """Scrambler agent: scramble enemy supply depots to take control."""

    ROLE = Role.SCRAMBLER

    def execute_role(self, s: CogsguardAgentState) -> Action:
        """Execute scrambler behavior: find and scramble ALL enemy depots.

        Energy-aware behavior:
        - Check if we have enough energy before attempting to move to targets
        - If energy is low, go recharge at the nexus
        - Retry failed scramble actions up to MAX_RETRIES times
        - If gear is lost, go back to base to re-equip
        - If gear acquisition fails repeatedly, get hearts first (gear may require hearts)
        """
        if DEBUG and s.step_count % 100 == 0:
            num_chargers = len(s.get_structures_by_type(StructureType.CHARGER))
            clips_chargers = len([c for c in s.get_structures_by_type(StructureType.CHARGER) if c.alignment == "clips"])
            num_worked = len(s.worked_chargers)
            print(
                f"[A{s.agent_id}] SCRAMBLER: step={s.step_count} heart={s.heart} energy={s.energy} gear={s.scrambler} "
                f"chargers={num_chargers} clips={clips_chargers} scrambled={num_worked}"
            )

        # === Resource check: need both gear AND heart to scramble ===
        has_gear = s.scrambler >= 1
        has_heart = s.heart >= 1

        # If we don't have gear, try to get it
        if not has_gear:
            return self._handle_no_gear(s)

        # If we have gear but no heart, go get hearts
        if not has_heart:
            if DEBUG and s.step_count % 10 == 0:
                print(f"[A{s.agent_id}] SCRAMBLER: Have gear but no heart, getting hearts first")
            return self._get_hearts(s)

        # Check if last action succeeded (for retry logic)
        # Actions can fail due to insufficient energy - agents auto-regen so just retry
        if s._pending_action_type == "scramble":
            if s.check_action_success():
                if DEBUG:
                    print(f"[A{s.agent_id}] SCRAMBLER: Previous scramble succeeded!")
            elif s.should_retry_action(MAX_RETRIES):
                retry_count = s.increment_retry()
                if DEBUG:
                    print(
                        f"[A{s.agent_id}] SCRAMBLER: Scramble failed, retrying ({retry_count}/{MAX_RETRIES}) "
                        f"at {s._pending_action_target}"
                    )
                # Retry the same action - agent will have auto-regenerated some energy
                if s._pending_action_target and is_adjacent((s.row, s.col), s._pending_action_target):
                    return self._use_object_at(s, s._pending_action_target)
            else:
                if DEBUG:
                    print(f"[A{s.agent_id}] SCRAMBLER: Scramble failed after {MAX_RETRIES} retries, moving on")
                s.clear_pending_action()

        # Find the best enemy depot to scramble (prioritize closest enemy charger)
        target_depot = self._find_best_target(s)

        if target_depot is None:
            # No known enemy depots, explore to find more chargers
            if DEBUG:
                chargers = s.get_structures_by_type(StructureType.CHARGER)
                print(f"[A{s.agent_id}] SCRAMBLER: No targets (total chargers={len(chargers)}), exploring")
            return self._explore_for_chargers(s)

        # Navigate to depot
        # Note: moves require energy. If move fails due to low energy,
        # action failure detection will catch it and we'll retry next step
        # (agents auto-regen energy every step, and regen full near aligned buildings)
        dist = abs(target_depot[0] - s.row) + abs(target_depot[1] - s.col)
        if not is_adjacent((s.row, s.col), target_depot):
            if DEBUG and s.step_count % 10 == 0:
                print(f"[A{s.agent_id}] SCRAMBLER: Moving to charger at {target_depot} (dist={dist})")
            return self._move_towards(s, target_depot, reach_adjacent=True)

        # Scramble the depot by bumping it
        # Mark this charger as worked
        s.worked_chargers[target_depot] = s.step_count

        # Start tracking this scramble attempt
        s.start_action_attempt("scramble", target_depot)

        if DEBUG:
            charger = s.get_structure_at(target_depot)
            alignment = charger.alignment if charger else "unknown"
            print(
                f"[A{s.agent_id}] SCRAMBLER: SCRAMBLING charger at {target_depot} "
                f"(alignment={alignment}, heart={s.heart}, energy={s.energy})!"
            )
        return self._use_object_at(s, target_depot)

    def _handle_no_gear(self, s: CogsguardAgentState) -> Action:
        """Handle behavior when scrambler doesn't have gear.

        Strategy: Go to gear station and wait there until gear is available.
        Can't do much without gear, so just wait.
        """
        station_pos = s.stations.get("scrambler_station")

        # If we don't know where the station is, explore to find it
        if station_pos is None:
            if DEBUG:
                print(f"[A{s.agent_id}] SCRAMBLER_NO_GEAR: Station unknown, exploring")
            return self._explore(s)

        # Go to gear station
        if not is_adjacent((s.row, s.col), station_pos):
            if DEBUG and s.step_count % 10 == 0:
                print(f"[A{s.agent_id}] SCRAMBLER_NO_GEAR: Moving to station at {station_pos}")
            return self._move_towards(s, station_pos, reach_adjacent=True)

        # At station - keep trying to get gear
        if DEBUG and s.step_count % 10 == 0:
            print(f"[A{s.agent_id}] SCRAMBLER_NO_GEAR: At station, waiting for gear")
        return self._use_object_at(s, station_pos)

    def _get_hearts(self, s: CogsguardAgentState) -> Action:
        """Get hearts from chest (primary source for hearts).

        The chest can produce hearts from resources:
        1. First tries to withdraw existing hearts from cogs commons (get_heart handler)
        2. If no hearts available, converts 1 of each element into 1 heart (make_heart handler)

        So as long as miners deposit resources, scramblers can get hearts.
        If we've been trying to get hearts for too long, go explore instead.
        """
        # If we've waited more than 20 steps for hearts, go explore instead
        # This prevents getting stuck when commons is out of resources
        if s.step_count - s._heart_wait_start > 20:
            if DEBUG:
                print(f"[A{s.agent_id}] SCRAMBLER: Waited 30+ steps for hearts, exploring instead")
            s._heart_wait_start = s.step_count  # Reset timer
            return self._explore_for_chargers(s)

        # Try chest first - it's the primary heart source
        chest_pos = s.stations.get("chest")
        if chest_pos is not None:
            if DEBUG and s.step_count % 10 == 0:
                adj = is_adjacent((s.row, s.col), chest_pos)
                print(f"[A{s.agent_id}] SCRAMBLER: Getting hearts from chest at {chest_pos}, adjacent={adj}")
            if not is_adjacent((s.row, s.col), chest_pos):
                return self._move_towards(s, chest_pos, reach_adjacent=True)
            return self._use_object_at(s, chest_pos)

        # Try assembler as fallback (may have heart AOE or deposit function)
        assembler_pos = s.stations.get("assembler")
        if assembler_pos is not None:
            if DEBUG:
                print(f"[A{s.agent_id}] SCRAMBLER: No chest found, trying assembler at {assembler_pos}")
            if not is_adjacent((s.row, s.col), assembler_pos):
                return self._move_towards(s, assembler_pos, reach_adjacent=True)
            return self._use_object_at(s, assembler_pos)

        # Neither found - explore to find them
        if DEBUG:
            print(f"[A{s.agent_id}] SCRAMBLER: No chest/assembler found, exploring")
        return self._explore(s)

    def _find_best_target(self, s: CogsguardAgentState) -> Optional[tuple[int, int]]:
        """Find the best charger to scramble - prioritize enemy (clips) aligned ones.

        Skips chargers that were recently worked on to ensure we visit multiple chargers.
        """
        # Get all known chargers from structures map
        chargers = s.get_structures_by_type(StructureType.CHARGER)

        # How long to ignore a charger after working on it (steps)
        cooldown = 50

        # Collect chargers and sort by distance, skipping recently worked ones
        enemy_chargers: list[tuple[int, tuple[int, int]]] = []
        neutral_chargers: list[tuple[int, tuple[int, int]]] = []
        any_chargers: list[tuple[int, tuple[int, int]]] = []

        if DEBUG and s.step_count % 20 == 1:
            print(f"[A{s.agent_id}] FIND_TARGET: {len(chargers)} chargers in structures map")
            for ch in chargers:
                print(f"  - {ch.position}: alignment={ch.alignment}, clipped={ch.clipped}")

        for charger in chargers:
            pos = charger.position
            dist = abs(pos[0] - s.row) + abs(pos[1] - s.col)

            if DEBUG and s.step_count % 20 == 1:
                print(f"  LOOP charger@{pos}: alignment='{charger.alignment}' clipped={charger.clipped} dist={dist}")

            # Skip recently worked chargers (only if actually worked before)
            last_worked = s.worked_chargers.get(pos, 0)
            if last_worked > 0 and s.step_count - last_worked < cooldown:
                if DEBUG and s.step_count % 20 == 1:
                    print(f"    SKIP: on cooldown (worked {s.step_count - last_worked} steps ago)")
                continue

            # Skip cogs-aligned chargers (already ours)
            if charger.alignment == "cogs":
                if DEBUG and s.step_count % 20 == 1:
                    print("    SKIP: cogs-aligned (ours)")
                continue

            # Check alignment - prioritize clips (enemy) chargers
            if charger.alignment == "clips" or charger.clipped:
                if DEBUG and s.step_count % 20 == 1:
                    print("    ADD to enemy_chargers")
                enemy_chargers.append((dist, pos))
            elif charger.alignment is None or charger.alignment == "neutral":
                neutral_chargers.append((dist, pos))
            else:
                any_chargers.append((dist, pos))

        if DEBUG and s.step_count % 20 == 1:
            print(f"  enemy_chargers={enemy_chargers} neutral={neutral_chargers} any={any_chargers}")

        # First try enemy chargers (sorted by distance)
        if enemy_chargers:
            enemy_chargers.sort()
            if DEBUG:
                print(f"[A{s.agent_id}] FIND_TARGET: Returning enemy charger at {enemy_chargers[0][1]}")
            return enemy_chargers[0][1]

        # Then try neutral chargers
        if neutral_chargers:
            neutral_chargers.sort()
            if DEBUG:
                print(f"[A{s.agent_id}] FIND_TARGET: Returning neutral charger at {neutral_chargers[0][1]}")
            return neutral_chargers[0][1]

        # Then try any non-cogs charger
        if any_chargers:
            any_chargers.sort()
            if DEBUG:
                print(f"[A{s.agent_id}] FIND_TARGET: Returning any charger at {any_chargers[0][1]}")
            return any_chargers[0][1]

        # Fall back to legacy supply_depots list
        for depot_pos, alignment in s.supply_depots:
            if alignment != "cogs":
                last_worked = s.worked_chargers.get(depot_pos, 0)
                if s.step_count - last_worked >= cooldown:
                    return depot_pos

        # Last resort: try charger from stations dict (if not on cooldown)
        charger_pos = s.stations.get("charger")
        if charger_pos:
            last_worked = s.worked_chargers.get(charger_pos, 0)
            if s.step_count - last_worked >= cooldown:
                return charger_pos

        return None

    def _explore_for_chargers(self, s: CogsguardAgentState) -> Action:
        """Explore aggressively to find more chargers spread around the map."""
        # Move in a direction based on agent ID and step count to spread out
        # Chargers are spread around the map, so cover different areas
        directions = ["north", "south", "east", "west"]
        # Cycle through directions, spending 20 steps in each direction
        dir_idx = (s.agent_id + s.step_count // 20) % 4
        direction = directions[dir_idx]

        dr, dc = self._move_deltas[direction]
        next_r, next_c = s.row + dr, s.col + dc

        # Check if we can move in that direction
        from cogames.policy.scripted_agent.pathfinding import is_traversable
        from cogames.policy.scripted_agent.types import CellType

        if is_traversable(s, next_r, next_c, CellType):  # type: ignore[arg-type]
            return self._actions.move.Move(direction)  # type: ignore[arg-type]

        # Fall back to regular exploration if blocked
        return self._explore(s)

    def _go_get_gear(self, s: CogsguardAgentState) -> Action:
        """Navigate to scrambler station to get new gear.

        This is called when the scrambler loses its gear and needs to re-equip.
        Delegates to _handle_no_gear which has proper retry/fallback logic.
        """
        return self._handle_no_gear(s)
