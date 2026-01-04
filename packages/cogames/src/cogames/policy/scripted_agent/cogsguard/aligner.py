"""
Aligner role for CoGsGuard.

Aligners find supply depots and align them to the cogs commons to take control.
With aligner gear, they get +20 influence capacity.

Strategy:
- Find ALL chargers on the map
- Prioritize aligning neutral and enemy (clips) chargers
- Systematically work through all chargers to take them over
- Check energy before moving to targets
- Retry failed align actions up to MAX_RETRIES times
"""

from __future__ import annotations

from typing import Optional

from cogames.policy.scripted_agent.utils import is_adjacent
from mettagrid.simulator import Action

from .policy import DEBUG, CogsguardAgentPolicyImpl
from .types import CogsguardAgentState, Role, StructureType

# Maximum number of times to retry a failed align action
MAX_RETRIES = 3


class AlignerAgentPolicyImpl(CogsguardAgentPolicyImpl):
    """Aligner agent: align ALL supply depots to cogs."""

    ROLE = Role.ALIGNER

    def execute_role(self, s: CogsguardAgentState) -> Action:
        """Execute aligner behavior: find and align ALL supply depots.

        Energy-aware behavior:
        - Check if we have enough energy before attempting to move to targets
        - If energy is low, go recharge at the nexus
        - Retry failed align actions up to MAX_RETRIES times
        - Require both gear AND heart before attempting to align
        - If gear acquisition fails repeatedly, get hearts first
        """
        if DEBUG and s.step_count % 50 == 0:
            num_chargers = len(s.get_structures_by_type(StructureType.CHARGER))
            num_worked = len(s.worked_chargers)
            print(
                f"[A{s.agent_id}] ALIGNER: step={s.step_count} influence={s.influence} "
                f"heart={s.heart} energy={s.energy} gear={s.aligner} chargers_known={num_chargers} worked={num_worked}"
            )

        # === Resource check: need both gear AND heart to align ===
        has_gear = s.aligner >= 1
        has_heart = s.heart >= 1

        # If we don't have gear, try to get it
        if not has_gear:
            return self._handle_no_gear(s)

        # If we have gear but no heart, go get hearts
        if not has_heart:
            if DEBUG and s.step_count % 10 == 0:
                print(f"[A{s.agent_id}] ALIGNER: Have gear but no heart, getting resources first")
            return self._get_resources(s, need_influence=False, need_heart=True)

        # Check if last action succeeded (for retry logic)
        # Actions can fail due to insufficient energy - agents auto-regen so just retry
        if s._pending_action_type == "align":
            if s.check_action_success():
                if DEBUG:
                    print(f"[A{s.agent_id}] ALIGNER: Previous align succeeded!")
            elif s.should_retry_action(MAX_RETRIES):
                retry_count = s.increment_retry()
                if DEBUG:
                    print(
                        f"[A{s.agent_id}] ALIGNER: Align failed, retrying ({retry_count}/{MAX_RETRIES}) "
                        f"at {s._pending_action_target}"
                    )
                # Retry the same action - agent will have auto-regenerated some energy
                if s._pending_action_target and is_adjacent((s.row, s.col), s._pending_action_target):
                    return self._use_object_at(s, s._pending_action_target)
            else:
                if DEBUG:
                    print(f"[A{s.agent_id}] ALIGNER: Align failed after {MAX_RETRIES} retries, moving on")
                s.clear_pending_action()

        # Find the best depot to align (prioritize closest non-cogs charger)
        target_depot = self._find_best_target(s)

        if target_depot is None:
            if DEBUG and s.step_count % 50 == 0:
                print(f"[A{s.agent_id}] ALIGNER: No targets, exploring for chargers")
            return self._explore_for_chargers(s)

        # Navigate to depot
        # Note: moves require energy. If move fails due to low energy,
        # action failure detection will catch it and we'll retry next step
        # (agents auto-regen energy every step, and regen full near aligned buildings)
        if not is_adjacent((s.row, s.col), target_depot):
            if DEBUG and s.step_count % 20 == 0:
                print(f"[A{s.agent_id}] ALIGNER: Moving to charger at {target_depot}")
            return self._move_towards(s, target_depot, reach_adjacent=True)

        # Align the depot by bumping it
        # Mark this charger as worked for a while (align multiple times then move on)
        last_worked = s.worked_chargers.get(target_depot, 0)
        times_worked = s.step_count - last_worked if last_worked > 0 else 0
        s.worked_chargers[target_depot] = s.step_count

        # Start tracking this align attempt
        s.start_action_attempt("align", target_depot)

        if DEBUG and times_worked < 5:
            print(f"[A{s.agent_id}] ALIGNER: ALIGNING charger at {target_depot} (energy={s.energy}, heart={s.heart})!")
        return self._use_object_at(s, target_depot)

    def _handle_no_gear(self, s: CogsguardAgentState) -> Action:
        """Handle behavior when aligner doesn't have gear.

        Strategy: Go to gear station and wait there until gear is available.
        Can't do much without gear, so just wait.
        """
        station_pos = s.stations.get("aligner_station")

        # If we don't know where the station is, explore to find it
        if station_pos is None:
            if DEBUG:
                print(f"[A{s.agent_id}] ALIGNER_NO_GEAR: Station unknown, exploring")
            return self._explore(s)

        # Go to gear station
        if not is_adjacent((s.row, s.col), station_pos):
            if DEBUG and s.step_count % 10 == 0:
                print(f"[A{s.agent_id}] ALIGNER_NO_GEAR: Moving to station at {station_pos}")
            return self._move_towards(s, station_pos, reach_adjacent=True)

        # At station - keep trying to get gear
        if DEBUG and s.step_count % 10 == 0:
            print(f"[A{s.agent_id}] ALIGNER_NO_GEAR: At station, waiting for gear")
        return self._use_object_at(s, station_pos)

    def _get_resources(self, s: CogsguardAgentState, need_influence: bool, need_heart: bool) -> Action:
        """Get hearts from the chest (primary source).

        The chest can produce hearts from resources:
        1. First tries to withdraw existing hearts from cogs commons (get_heart handler)
        2. If no hearts available, converts 1 of each element into 1 heart (make_heart handler)

        So as long as miners deposit resources, aligners can get hearts.
        If we've been trying to get hearts for too long, go explore instead.
        """
        if need_heart:
            # If we've waited more than 20 steps for hearts, go explore instead
            if s.step_count - s._heart_wait_start > 20:
                if DEBUG:
                    print(f"[A{s.agent_id}] ALIGNER: Waited 20+ steps for hearts, exploring instead")
                s._heart_wait_start = s.step_count  # Reset timer
                return self._explore_for_chargers(s)

            # Try chest first - it's the primary heart source
            chest_pos = s.stations.get("chest")
            if chest_pos is not None:
                if DEBUG and s.step_count % 10 == 0:
                    adj = is_adjacent((s.row, s.col), chest_pos)
                    print(f"[A{s.agent_id}] ALIGNER: Getting hearts from chest at {chest_pos}, adjacent={adj}")
                if not is_adjacent((s.row, s.col), chest_pos):
                    return self._move_towards(s, chest_pos, reach_adjacent=True)
                return self._use_object_at(s, chest_pos)

            # Try assembler as fallback (may have heart AOE or deposit function)
            assembler_pos = s.stations.get("assembler")
            if assembler_pos is not None:
                if DEBUG:
                    print(f"[A{s.agent_id}] ALIGNER: No chest found, trying assembler at {assembler_pos}")
                if not is_adjacent((s.row, s.col), assembler_pos):
                    return self._move_towards(s, assembler_pos, reach_adjacent=True)
                return self._use_object_at(s, assembler_pos)

            # Neither found - explore to find them
            if DEBUG:
                print(f"[A{s.agent_id}] ALIGNER: No chest/assembler found, exploring")
            return self._explore(s)

        # Just need influence - wait for AOE regeneration near assembler
        assembler_pos = s.stations.get("assembler")
        if assembler_pos is None:
            return self._explore(s)
        if not is_adjacent((s.row, s.col), assembler_pos):
            return self._move_towards(s, assembler_pos, reach_adjacent=True)
        return self._actions.noop.Noop()

    def _find_best_target(self, s: CogsguardAgentState) -> Optional[tuple[int, int]]:
        """Find the closest un-aligned charger to align.

        Prioritizes by distance - aligns the closest charger that isn't already cogs-aligned.
        Skips chargers that were recently worked on to ensure we visit multiple chargers.
        """
        # Get all known chargers from structures map
        chargers = s.get_structures_by_type(StructureType.CHARGER)

        # How long to ignore a charger after working on it (steps)
        cooldown = 50

        # Collect all un-aligned chargers (not cogs) and sort by distance
        unaligned_chargers: list[tuple[int, tuple[int, int]]] = []

        for charger in chargers:
            pos = charger.position
            dist = abs(pos[0] - s.row) + abs(pos[1] - s.col)

            # Skip recently worked chargers
            last_worked = s.worked_chargers.get(pos, 0)
            if s.step_count - last_worked < cooldown:
                continue

            # Skip already cogs-aligned chargers
            if charger.alignment == "cogs":
                continue

            # Add all un-aligned chargers (neutral, clips, or unknown)
            unaligned_chargers.append((dist, pos))

        # Sort by distance and return closest
        if unaligned_chargers:
            unaligned_chargers.sort()
            if DEBUG and s.step_count % 20 == 0:
                count = len(unaligned_chargers)
                closest = unaligned_chargers[0][1]
                print(f"[A{s.agent_id}] ALIGNER: Found {count} un-aligned chargers, closest at {closest}")
            return unaligned_chargers[0][1]

        # Check legacy supply_depots list
        legacy_targets: list[tuple[int, tuple[int, int]]] = []
        for depot_pos, alignment in s.supply_depots:
            if alignment != "cogs":
                last_worked = s.worked_chargers.get(depot_pos, 0)
                if s.step_count - last_worked >= cooldown:
                    dist = abs(depot_pos[0] - s.row) + abs(depot_pos[1] - s.col)
                    legacy_targets.append((dist, depot_pos))

        if legacy_targets:
            legacy_targets.sort()
            return legacy_targets[0][1]

        # Last resort: try charger from stations dict
        charger_pos = s.stations.get("charger")
        if charger_pos:
            last_worked = s.worked_chargers.get(charger_pos, 0)
            if s.step_count - last_worked >= cooldown:
                return charger_pos

        return None

    def _explore_for_chargers(self, s: CogsguardAgentState) -> Action:
        """Explore aggressively to find more chargers spread around the map."""
        # Move in a direction based on agent ID and step count to spread out
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
