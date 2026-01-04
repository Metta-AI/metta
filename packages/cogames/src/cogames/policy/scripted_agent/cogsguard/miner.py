"""
Miner role for CoGsGuard.

Miners gather resources from extractors and deposit at the assembler (cogs nexus).
With miner gear, they get +40 cargo capacity and extract 10 resources at a time.

Strategy:
- Extractors are in map corners, assembler is in center
- Miners should quickly head to corners to find extractors
- Once extractors are known, alternate between mining and depositing
"""

from __future__ import annotations

from cogames.policy.scripted_agent.utils import is_adjacent
from mettagrid.simulator import Action

from .policy import DEBUG, CogsguardAgentPolicyImpl
from .types import CogsguardAgentState, Role, StructureType

# Extractors are typically in map corners - explore these areas first
# Map is 200x200, center is ~100,100
CORNER_OFFSETS = [
    (-10, -10),  # NW corner direction
    (-10, 10),  # NE corner direction
    (10, -10),  # SW corner direction
    (10, 10),  # SE corner direction
]


class MinerAgentPolicyImpl(CogsguardAgentPolicyImpl):
    """Miner agent: gather resources and deposit at the cogs assembler."""

    ROLE = Role.MINER

    def execute_role(self, s: CogsguardAgentState) -> Action:
        """Execute miner behavior: gather resources, deposit at assembler."""
        total_cargo = s.carbon + s.oxygen + s.germanium + s.silicon
        # Cargo capacity: 44 with gear (4 base + 40 from miner gear), 4 without
        cargo_capacity = 44 if s.miner > 0 else 4

        if DEBUG and s.step_count <= 50:
            num_extractors = sum(len(exts) for exts in s.extractors.values())
            mode = "deposit" if total_cargo >= cargo_capacity - 2 else "gather"
            has_gear = "GEAR" if s.miner > 0 else "NO_GEAR"
            print(
                f"[A{s.agent_id}] MINER step={s.step_count}: pos=({s.row},{s.col}) "
                f"cargo={total_cargo}/{cargo_capacity} mode={mode} ext={num_extractors} {has_gear}"
            )

        # If mining without gear, periodically check if we can get gear now
        # (commons might have resources after other miners deposited)
        if s.miner == 0 and total_cargo == 0 and s.step_count % 100 == 0:
            station_name = s.get_gear_station_name()
            station_pos = s.stations.get(station_name)
            if station_pos is not None:
                if DEBUG:
                    print(f"[A{s.agent_id}] MINER: Attempting to get gear again")
                return self._move_towards(s, station_pos, reach_adjacent=True)

        # If cargo is full or nearly full, go deposit
        # Use cargo_capacity - 2 threshold to avoid oscillating at boundary
        if total_cargo >= cargo_capacity - 2:
            return self._do_deposit(s)

        # Otherwise gather resources
        return self._do_gather(s)

    def _do_gather(self, s: CogsguardAgentState) -> Action:
        """Gather resources from nearest extractor."""
        # Use structures map for most up-to-date extractor info
        extractor = s.get_nearest_usable_extractor()

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
        if not is_adjacent((s.row, s.col), extractor.position):
            if DEBUG and s.step_count <= 50:
                print(f"[A{s.agent_id}] GATHER: Moving to extractor at {extractor.position}")
            return self._move_towards(s, extractor.position, reach_adjacent=True)

        # At extractor - get CURRENT state from structures map (updated from observation)
        current_extractor = s.get_structure_at(extractor.position)
        if current_extractor is None or not current_extractor.is_usable_extractor():
            if DEBUG:
                uses = current_extractor.remaining_uses if current_extractor else "?"
                print(f"[A{s.agent_id}] GATHER: Extractor at {extractor.position} not usable (uses={uses})")
            # Find another extractor
            other = s.get_nearest_usable_extractor(exclude=extractor.position)
            if other is not None:
                return self._move_towards(s, other.position, reach_adjacent=True)
            return self._explore_for_extractors(s)

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

        # Extract!
        if DEBUG and s.step_count <= 60:
            pos = current_extractor.position
            cd, uses = current_extractor.cooldown_remaining, current_extractor.remaining_uses
            print(f"[A{s.agent_id}] GATHER: Extracting from {pos}, cooldown={cd}, uses={uses}")
        return self._use_object_at(s, current_extractor.position)

    def _explore_for_extractors(self, s: CogsguardAgentState) -> Action:
        """Explore towards map corners where extractors are located."""
        # Track exploration progress - rotate through corners over time
        # Miners are agents 0, 4, 8 - spread them across corners initially
        miner_idx = s.agent_id // 4
        # Rotate corner based on step count to explore new areas when old ones are depleted
        corner_rotation = s.step_count // 200  # Change corner every 200 steps
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

    def _do_deposit(self, s: CogsguardAgentState) -> Action:
        """Deposit resources at the cogs assembler (main nexus)."""
        depot_pos = s.stations.get("assembler")

        if depot_pos is None:
            if DEBUG:
                print(f"[A{s.agent_id}] DEPOSIT: No assembler found, exploring")
            return self._explore(s)

        if not is_adjacent((s.row, s.col), depot_pos):
            if DEBUG and s.step_count % 20 == 0:
                print(f"[A{s.agent_id}] DEPOSIT: Moving to assembler at {depot_pos}")
            return self._move_towards(s, depot_pos, reach_adjacent=True)

        if DEBUG:
            total_cargo = s.carbon + s.oxygen + s.germanium + s.silicon
            print(f"[A{s.agent_id}] DEPOSIT: At assembler, cargo={total_cargo}")
        return self._use_object_at(s, depot_pos)
