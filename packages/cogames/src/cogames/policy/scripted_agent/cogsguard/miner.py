"""
Miner role for CoGsGuard.

Miners gather resources from extractors and deposit at aligned buildings.
With miner gear, they get +40 cargo capacity and extract 10 resources at a time.

Strategy:
- Extractors are in map corners, aligned buildings provide deposit points
- Miners should quickly head to corners to find extractors
- Once extractors are known, alternate between mining and depositing
- Deposit to nearest aligned building (assembler or cogs-aligned chargers)
- If gear is lost, collect resources without gear and check for gear on each dropoff
- Retry failed mine actions up to MAX_RETRIES times
- HP-aware: Never venture further than can safely return to healing territory
  (HP drains at -1/step outside aligned building AOE, losing all HP = lose gear)
"""

from __future__ import annotations

from cogames.policy.scripted_agent.utils import is_adjacent
from mettagrid.simulator import Action

from .policy import DEBUG, CogsguardAgentPolicyImpl
from .types import CogsguardAgentState, Role, StructureInfo, StructureType

# Extractors are typically in map corners - explore these areas first
# Map is 200x200, center is ~100,100
CORNER_OFFSETS = [
    (-10, -10),  # NW corner direction
    (-10, 10),  # NE corner direction
    (10, -10),  # SW corner direction
    (10, 10),  # SE corner direction
]

# Maximum number of times to retry a failed mine action
MAX_RETRIES = 3

# HP management constants
# Assembler AOE range that provides healing (+100 HP to aligned agents)
HEALING_AOE_RANGE = 10
# Base HP drain per step outside healing AOE (from regen_amounts)
HP_DRAIN_BASE = 1
# Additional HP drain per step when near enemy buildings (from attack_aoe)
HP_DRAIN_ENEMY_AOE = 1
# Enemy AOE range
ENEMY_AOE_RANGE = 10
# Safety margin - return home with this much HP buffer
HP_SAFETY_MARGIN = 5


class MinerAgentPolicyImpl(CogsguardAgentPolicyImpl):
    """Miner agent: gather resources and deposit at nearest aligned building."""

    ROLE = Role.MINER

    def _get_nearest_aligned_depot(self, s: CogsguardAgentState) -> tuple[int, int] | None:
        """Find the nearest aligned building that accepts deposits.

        Returns position of nearest cogs-aligned building (assembler or charger).
        These buildings have healing AOE and accept resource deposits.
        """
        candidates: list[tuple[int, tuple[int, int]]] = []

        # Assembler is always aligned to cogs
        assembler_pos = s.stations.get("assembler")
        if assembler_pos:
            dist = abs(assembler_pos[0] - s.row) + abs(assembler_pos[1] - s.col)
            candidates.append((dist, assembler_pos))

        # Check for cogs-aligned chargers
        chargers = s.get_structures_by_type(StructureType.CHARGER)
        for charger in chargers:
            if charger.alignment == "cogs":
                dist = abs(charger.position[0] - s.row) + abs(charger.position[1] - s.col)
                candidates.append((dist, charger.position))

        if not candidates:
            return None

        # Return nearest
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _get_distance_to_nearest_healing(self, s: CogsguardAgentState) -> int:
        """Get distance to nearest aligned building's healing AOE.

        Returns steps needed to reach healing territory.
        """
        depot_pos = self._get_nearest_aligned_depot(s)
        if depot_pos is None:
            return 100  # Unknown - be conservative

        dist = abs(depot_pos[0] - s.row) + abs(depot_pos[1] - s.col)
        return max(0, dist - HEALING_AOE_RANGE)

    def execute_role(self, s: CogsguardAgentState) -> Action:
        """Execute miner behavior: continuously gather resources and deposit at assembler.

        If gear is lost, collect resources without gear and check for gear on each dropoff.
        Retry failed mine actions up to MAX_RETRIES times.
        HP-aware: Return to healing territory before HP gets too low to avoid losing gear.
        """
        # Use dynamic properties - cargo_capacity updates if gear is lost
        total_cargo = s.total_cargo
        cargo_capacity = s.cargo_capacity
        has_gear = s.miner > 0

        if DEBUG and s.step_count <= 50:
            num_extractors = sum(len(exts) for exts in s.extractors.values())
            mode = "deposit" if total_cargo >= cargo_capacity - 2 else "gather"
            gear_status = "GEAR" if has_gear else "NO_GEAR"
            steps_to_heal = self._get_steps_to_healing(s)
            drain_rate = self._get_hp_drain_rate(s)
            nearest_depot = self._get_nearest_aligned_depot(s)
            print(
                f"[A{s.agent_id}] MINER step={s.step_count}: pos=({s.row},{s.col}) "
                f"cargo={total_cargo}/{cargo_capacity} mode={mode} ext={num_extractors} "
                f"{gear_status} energy={s.energy} hp={s.hp} steps_to_heal={steps_to_heal} "
                f"drain={drain_rate}/step depot={nearest_depot}"
            )

        # === HP check - highest priority ===
        # If HP is getting low, head back to healing territory immediately
        if self._should_return_for_healing(s):
            if DEBUG:
                steps = self._get_steps_to_healing(s)
                print(f"[A{s.agent_id}] MINER: HP LOW ({s.hp}), returning to heal! Steps to safety: {steps}")
            return self._return_to_healing(s)

        # Check if last action succeeded (for retry logic)
        # Actions can fail due to insufficient energy - agents auto-regen so just retry
        if s._pending_action_type == "mine":
            if s.check_action_success():
                if DEBUG:
                    print(f"[A{s.agent_id}] MINER: Previous mine succeeded!")
            elif s.should_retry_action(MAX_RETRIES):
                retry_count = s.increment_retry()
                if DEBUG:
                    print(
                        f"[A{s.agent_id}] MINER: Mine failed, retrying ({retry_count}/{MAX_RETRIES}) "
                        f"at {s._pending_action_target}"
                    )
                # Retry the same action - agent will have auto-regenerated some energy
                if s._pending_action_target and is_adjacent((s.row, s.col), s._pending_action_target):
                    return self._use_object_at(s, s._pending_action_target)
            else:
                if DEBUG:
                    print(f"[A{s.agent_id}] MINER: Mine failed after {MAX_RETRIES} retries, moving on")
                s.clear_pending_action()

        # === Gear re-acquisition logic ===
        if not has_gear:
            return self._handle_no_gear(s, total_cargo, cargo_capacity)

        # === Normal mining loop (has gear) ===
        # Simple loop: gather until full, deposit, repeat
        if total_cargo >= cargo_capacity - 2:
            return self._do_deposit(s)

        # Otherwise gather resources
        return self._do_gather(s)

    def _get_steps_to_healing(self, s: CogsguardAgentState) -> int:
        """Calculate steps needed to reach healing territory (nearest aligned building AOE).

        Returns the number of steps to get within any aligned building's healing AOE.
        If no aligned buildings known, returns a large number to be conservative.
        """
        return self._get_distance_to_nearest_healing(s)

    def _should_return_for_healing(self, s: CogsguardAgentState) -> bool:
        """Check if miner should return to healing territory based on HP.

        Returns True if HP is low enough that we need to head back now to survive.
        Accounts for faster HP drain when near enemy buildings.
        """
        steps_to_healing = self._get_steps_to_healing(s)
        current_drain = self._get_hp_drain_rate(s)

        # Use current drain rate (which may be elevated if near enemies)
        # to calculate HP needed to survive the trip
        hp_needed = (steps_to_healing * current_drain) + HP_SAFETY_MARGIN

        return s.hp <= hp_needed

    def _get_hp_drain_rate(self, s: CogsguardAgentState) -> int:
        """Calculate current HP drain rate based on proximity to enemy buildings.

        Base drain is HP_DRAIN_BASE per step outside healing AOE.
        Additional HP_DRAIN_ENEMY_AOE per step when near enemy chargers.
        """
        drain_rate = HP_DRAIN_BASE

        # Check if near any enemy chargers
        chargers = s.get_structures_by_type(StructureType.CHARGER)
        for charger in chargers:
            if charger.alignment == "cogs":
                continue  # Friendly charger
            dist = abs(charger.position[0] - s.row) + abs(charger.position[1] - s.col)
            if dist <= ENEMY_AOE_RANGE:
                drain_rate += HP_DRAIN_ENEMY_AOE
                if DEBUG:
                    print(f"[A{s.agent_id}] MINER: Near enemy charger at {charger.position}, drain_rate={drain_rate}")
                break  # Only count once even if near multiple enemies

        return drain_rate

    def _return_to_healing(self, s: CogsguardAgentState) -> Action:
        """Return to nearest aligned building to heal. Deposit any cargo while there."""
        depot_pos = self._get_nearest_aligned_depot(s)

        if depot_pos is None:
            # Don't know where any aligned building is - explore to find one
            if DEBUG:
                print(f"[A{s.agent_id}] MINER_HEAL: No aligned building known, exploring!")
            return self._explore(s)

        # Check if we're already in healing range
        dist_to_depot = abs(depot_pos[0] - s.row) + abs(depot_pos[1] - s.col)
        if dist_to_depot <= HEALING_AOE_RANGE:
            # We're in healing range - if we have cargo, deposit it
            if s.total_cargo > 0 and is_adjacent((s.row, s.col), depot_pos):
                if DEBUG:
                    print(f"[A{s.agent_id}] MINER_HEAL: In range, depositing cargo={s.total_cargo}")
                return self._use_object_at(s, depot_pos)
            # Otherwise just wait here to heal (or move closer to deposit)
            if s.total_cargo > 0:
                return self._move_towards(s, depot_pos, reach_adjacent=True)
            # No cargo, just noop to heal
            if DEBUG and s.step_count % 10 == 0:
                print(f"[A{s.agent_id}] MINER_HEAL: Healing at HP={s.hp}")
            return self._actions.noop.Noop()

        # Move towards nearest aligned building
        if DEBUG and s.step_count % 10 == 0:
            print(f"[A{s.agent_id}] MINER_HEAL: Moving to aligned building at {depot_pos}, dist={dist_to_depot}")
        return self._move_towards(s, depot_pos, reach_adjacent=True)

    def _handle_no_gear(self, s: CogsguardAgentState, total_cargo: int, cargo_capacity: int) -> Action:
        """Handle behavior when miner doesn't have gear.

        Strategy: Collect resources even without gear (reduced capacity/extraction rate).
        Check for gear availability on each dropoff at the assembler.
        """
        # If cargo is full (even with small capacity), deposit and check for gear
        if total_cargo >= cargo_capacity - 1:
            return self._do_deposit_and_check_gear(s)

        # Otherwise, continue gathering resources (at reduced rate without gear)
        if DEBUG and s.step_count % 20 == 0:
            print(f"[A{s.agent_id}] MINER_NO_GEAR: Gathering without gear, cargo={total_cargo}/{cargo_capacity}")
        return self._do_gather(s)

    def _do_deposit_and_check_gear(self, s: CogsguardAgentState) -> Action:
        """Deposit resources at nearest aligned building, then check gear station.

        After depositing, immediately try to get gear before continuing to mine.
        """
        depot_pos = self._get_nearest_aligned_depot(s)
        station_pos = s.stations.get("miner_station")

        # If we still have cargo, deposit first at nearest aligned building
        if s.total_cargo > 0:
            if depot_pos is None:
                if DEBUG:
                    print(f"[A{s.agent_id}] MINER_NO_GEAR: No aligned building known, exploring")
                return self._explore(s)

            if not is_adjacent((s.row, s.col), depot_pos):
                if DEBUG and s.step_count % 20 == 0:
                    print(f"[A{s.agent_id}] MINER_NO_GEAR: Moving to deposit at {depot_pos}")
                return self._move_towards(s, depot_pos, reach_adjacent=True)

            # At depot - deposit
            if DEBUG:
                print(f"[A{s.agent_id}] MINER_NO_GEAR: Depositing cargo={s.total_cargo} at {depot_pos}")
            return self._use_object_at(s, depot_pos)

        # Cargo deposited, now try to get gear
        if station_pos is None:
            if DEBUG:
                print(f"[A{s.agent_id}] MINER_NO_GEAR: Station unknown, exploring")
            return self._explore(s)

        if not is_adjacent((s.row, s.col), station_pos):
            if DEBUG and s.step_count % 20 == 0:
                print(f"[A{s.agent_id}] MINER_NO_GEAR: Checking gear station at {station_pos}")
            return self._move_towards(s, station_pos, reach_adjacent=True)

        # At station - try to get gear (will fail if commons lacks resources, that's ok)
        if DEBUG:
            print(f"[A{s.agent_id}] MINER_NO_GEAR: Attempting to get gear")
        return self._use_object_at(s, station_pos)

    def _do_gather(self, s: CogsguardAgentState) -> Action:
        """Gather resources from nearest extractor.

        Tracks mining attempts for retry logic.
        Note: moves require energy. If move fails due to low energy,
        action failure detection will catch it and we'll retry next step
        (agents auto-regen energy every step, and regen full near aligned buildings)
        """
        # Use structures map for most up-to-date extractor info
        # Prefer extractors that are safe (not near clips chargers)
        extractor = self._get_safe_extractor(s)

        if extractor is None:
            # No usable extractors known - explore to find more
            total_known = len(s.get_usable_extractors())
            total_extractors = len(s.get_structures_by_type(StructureType.EXTRACTOR))
            if total_extractors > 0 and DEBUG:
                print(
                    f"[A{s.agent_id}] GATHER: {total_extractors} extractors known, "
                    f"{total_known} usable, exploring for more"
                )
            return self._explore_for_extractors(s)

        # Navigate to extractor
        ext_pos = extractor.position
        agent_pos = (s.row, s.col)
        adjacent = is_adjacent(agent_pos, ext_pos)
        if DEBUG and s.step_count <= 60:
            print(f"[A{s.agent_id}] GATHER: agent@{agent_pos} ext@{ext_pos} adjacent={adjacent}")
        if not adjacent:
            if DEBUG and s.step_count <= 50:
                print(f"[A{s.agent_id}] GATHER: Moving to extractor at {extractor.position}")
            return self._move_towards(s, extractor.position, reach_adjacent=True)

        # At extractor - get CURRENT state from structures map (updated from observation)
        current_extractor = s.get_structure_at(extractor.position)
        # Debug: log when extractor is depleted
        if DEBUG and current_extractor and current_extractor.inventory_amount == 0:
            print(f"[A{s.agent_id}] GATHER: EXTRACTOR_EMPTY at {extractor.position}! Will switch.")
        if current_extractor is None or not current_extractor.is_usable_extractor():
            # Find another extractor
            other = s.get_nearest_usable_extractor(exclude=extractor.position)
            if other is not None:
                return self._move_towards(s, other.position, reach_adjacent=True)
            # No usable extractors - explore to find more
            if DEBUG:
                total_ext = len(s.get_structures_by_type(StructureType.EXTRACTOR))
                usable_ext = len(s.get_usable_extractors())
                print(f"[A{s.agent_id}] GATHER: No usable extractors! total={total_ext}, usable={usable_ext}")
            return self._explore_for_extractors(s)

        # Check if another agent is blocking the extractor
        if extractor.position in s.agent_occupancy:
            if DEBUG:
                print(f"[A{s.agent_id}] GATHER: Extractor at {extractor.position} blocked by another agent")
            # Try another extractor
            other = s.get_nearest_usable_extractor(exclude=extractor.position)
            if other is not None:
                if DEBUG:
                    print(f"[A{s.agent_id}] GATHER: Switching to extractor at {other.position}")
                return self._move_towards(s, other.position, reach_adjacent=True)
            # Wait a bit - other agent should move soon
            return self._actions.noop.Noop()

        # At extractor - check cooldown
        if current_extractor.cooldown_remaining > 0:
            if DEBUG and s.step_count <= 50:
                print(f"[A{s.agent_id}] GATHER: Extractor on cooldown={current_extractor.cooldown_remaining}")
            # Try another extractor while this one cools down
            other = s.get_nearest_usable_extractor(exclude=extractor.position)
            if other is not None and other.cooldown_remaining == 0:
                return self._move_towards(s, other.position, reach_adjacent=True)
            # Wait for cooldown - noop
            return self._actions.noop.Noop()

        # Start tracking this mine attempt
        s.start_action_attempt("mine", current_extractor.position)

        # Extract!
        if DEBUG and s.step_count <= 60:
            print(f"[A{s.agent_id}] GATHER: MINING at {current_extractor.position} (energy={s.energy})!")
        return self._use_object_at(s, current_extractor.position)

    def _explore_for_extractors(self, s: CogsguardAgentState) -> Action:
        """Explore towards map corners where extractors are located."""
        # Track exploration progress - rotate through corners quickly
        # Each miner starts at a different corner, rotates every 50 steps
        miner_idx = s.agent_id % 4  # Spread across all 4 corners
        # Rotate corner based on step count - faster rotation (every 50 steps) to find all resources
        corner_rotation = s.step_count // 50
        corner_idx = (miner_idx + corner_rotation) % len(CORNER_OFFSETS)
        dr, dc = CORNER_OFFSETS[corner_idx]

        # Target a point away from center in the corner direction
        target_r = max(10, min(s.map_height - 10, s.row + dr))
        target_c = max(10, min(s.map_width - 10, s.col + dc))

        # If we've reached our corner area, switch to regular exploration
        at_corner = ((dr < 0 and s.row < 95) or (dr > 0 and s.row > 105)) and (
            (dc < 0 and s.col < 95) or (dc > 0 and s.col > 105)
        )

        if at_corner:
            # In corner area, explore locally to find extractors
            return self._explore(s)

        # Move towards corner
        if DEBUG and s.step_count <= 20:
            print(f"[A{s.agent_id}] MINER: exploring towards corner {corner_idx}, target=({target_r},{target_c})")
        return self._move_towards(s, (target_r, target_c), reach_adjacent=False)

    def _get_safe_extractor(self, s: CogsguardAgentState) -> "StructureInfo | None":
        """Get extractor prioritized by distance to aligned stations.

        Prioritizes extractors nearest to aligned buildings (assembler/cogs chargers)
        for shorter, safer mining routes.

        Considers:
        1. Distance from extractor to nearest aligned station (primary sort)
        2. Distance from clips chargers (enemy AOE damage)
        3. HP-based range limit - only select extractors we can reach and return from safely
        """
        # AOE range is 10 - stay further to be safe from enemies
        danger_range = 12

        # Get all clips chargers
        chargers = s.get_structures_by_type(StructureType.CHARGER)
        danger_zones = [c.position for c in chargers if c.alignment != "cogs"]

        # Calculate max safe operating distance based on HP
        # We need HP to get to extractor AND back to healing zone
        max_safe_dist = self._get_max_safe_distance(s)

        # Get usable extractors
        extractors = s.get_usable_extractors()

        # Sort by distance to aligned station and prefer safe ones within HP range
        safe_extractors: list[tuple[int, int, StructureInfo]] = []  # (dist_to_depot, dist_to_ext, ext)
        risky_extractors: list[tuple[int, int, StructureInfo]] = []

        # Get nearest aligned depot for round-trip calculations
        nearest_depot = self._get_nearest_aligned_depot(s)

        for ext in extractors:
            dist_to_ext = abs(ext.position[0] - s.row) + abs(ext.position[1] - s.col)

            # Calculate distance from extractor to nearest aligned building
            if nearest_depot:
                dist_ext_to_depot = abs(ext.position[0] - nearest_depot[0]) + abs(ext.position[1] - nearest_depot[1])
                round_trip = dist_to_ext + max(0, dist_ext_to_depot - HEALING_AOE_RANGE)
            else:
                # Unknown depot - use conservative estimate
                dist_ext_to_depot = 100  # Large number to deprioritize
                round_trip = dist_to_ext * 2

            # Skip extractors that are too far for our current HP
            if round_trip > max_safe_dist:
                if DEBUG:
                    print(
                        f"[A{s.agent_id}] MINER: Skipping extractor at {ext.position}, "
                        f"round_trip={round_trip} > max_safe={max_safe_dist}"
                    )
                continue

            # Check if extractor is in danger zone (enemy AOE)
            is_safe = True
            for danger_pos in danger_zones:
                danger_dist = abs(ext.position[0] - danger_pos[0]) + abs(ext.position[1] - danger_pos[1])
                if danger_dist < danger_range:
                    is_safe = False
                    break

            # Store with dist_to_depot as primary sort key, dist_to_ext as tiebreaker
            if is_safe:
                safe_extractors.append((dist_ext_to_depot, dist_to_ext, ext))
            else:
                risky_extractors.append((dist_ext_to_depot, dist_to_ext, ext))

        # Prefer safe extractors, sorted by distance to aligned station
        if safe_extractors:
            safe_extractors.sort(key=lambda x: (x[0], x[1]))  # Primary: depot dist, Secondary: agent dist
            return safe_extractors[0][2]

        # Fall back to risky extractors if no safe ones
        if risky_extractors:
            risky_extractors.sort(key=lambda x: (x[0], x[1]))
            return risky_extractors[0][2]

        return None

    def _get_max_safe_distance(self, s: CogsguardAgentState) -> int:
        """Calculate max round-trip distance based on current HP.

        Returns the maximum total distance (to target + back to healing) that's safe.
        Uses current drain rate which may be elevated if near enemy buildings.
        """
        # Reserve HP_SAFETY_MARGIN for emergencies
        available_hp = max(0, s.hp - HP_SAFETY_MARGIN)

        # Use current drain rate (may be faster near enemies)
        drain_rate = self._get_hp_drain_rate(s)

        # Max steps we can take before HP runs out
        max_steps = available_hp // drain_rate if drain_rate > 0 else available_hp

        return max_steps

    def _do_deposit(self, s: CogsguardAgentState) -> Action:
        """Deposit resources at the nearest aligned building.

        Energy-aware: checks if we have enough energy to reach the depot.
        """
        depot_pos = self._get_nearest_aligned_depot(s)

        if depot_pos is None:
            if DEBUG:
                print(f"[A{s.agent_id}] DEPOSIT: No aligned building found, exploring")
            return self._explore(s)

        # Check if we have enough energy to reach the depot
        dist = abs(depot_pos[0] - s.row) + abs(depot_pos[1] - s.col)
        if not s.has_enough_energy_for_moves(dist + 2):  # +2 for safety margin
            if DEBUG:
                print(
                    f"[A{s.agent_id}] DEPOSIT: Not enough energy ({s.energy}) to reach "
                    f"depot at dist={dist}, but going anyway (AOE will recharge)"
                )
            # Note: we need to recharge BUT we're carrying cargo, so go to depot anyway
            # since aligned buildings provide energy AOE - recharging and depositing are the same trip!
            pass  # Continue to move to depot - we'll recharge there

        if not is_adjacent((s.row, s.col), depot_pos):
            if DEBUG and s.step_count % 20 == 0:
                print(f"[A{s.agent_id}] DEPOSIT: Moving to depot at {depot_pos}")
            return self._move_towards(s, depot_pos, reach_adjacent=True)

        # Track resources deposited (for gear re-acquisition logic)
        cargo_to_deposit = s.total_cargo
        s._resources_deposited_since_gear_attempt += cargo_to_deposit
        if DEBUG:
            print(
                f"[A{s.agent_id}] DEPOSIT: At depot {depot_pos}, cargo={cargo_to_deposit}, "
                f"total_deposited={s._resources_deposited_since_gear_attempt}"
            )

        return self._use_object_at(s, depot_pos)
