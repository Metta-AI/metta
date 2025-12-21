"""State tracking for harvest policy.

Handles observation processing, position tracking, and object discovery.
"""
from .types import HarvestState, ChargerInfo


class StateTracker:
    """Tracks agent state and processes observations."""

    def __init__(self, obs_hr: int, obs_wr: int, tag_names: dict, logger):
        """Initialize state tracker.

        Args:
            obs_hr: Observation half-height radius
            obs_wr: Observation half-width radius
            tag_names: Mapping of tag IDs to names
            logger: Logger instance
        """
        self._obs_hr = obs_hr
        self._obs_wr = obs_wr
        self._tag_names = tag_names
        self._logger = logger

    def update_position_history(self, state: HarvestState):
        """Update position history for stuck detection.

        Keeps last 10 positions.

        Args:
            state: Current agent state (will be modified)
        """
        current_pos = (state.row, state.col)
        state.position_history.append(current_pos)

        # Keep only last 10 positions
        if len(state.position_history) > 10:
            state.position_history.pop(0)

        # Count how many recent positions are same as current
        same_position_count = sum(1 for pos in state.position_history if pos == current_pos)

        # If stuck at same position for 5+ steps, update consecutive_failed_moves
        if same_position_count >= 5:
            # Use max to avoid overwriting higher counts
            state.consecutive_failed_moves = max(state.consecutive_failed_moves, same_position_count)

    def discover_objects_in_observation(self, state: HarvestState):
        """Process observation to discover chargers and extractors.

        Args:
            state: Current agent state (will be modified)
        """
        if state.current_obs is None or not state.current_obs.tokens:
            return

        # Process each token in observation
        for tok in state.current_obs.tokens:
            if tok.feature.name != "tag":
                continue

            tag_name = self._tag_names.get(tok.value, "").lower()

            # Convert observation position to world position
            obs_r, obs_c = tok.location
            world_r = obs_r - self._obs_hr + state.row
            world_c = obs_c - self._obs_wr + state.col

            # Check bounds
            if not (0 <= world_r < state.map_height and 0 <= world_c < state.map_width):
                continue

            world_pos = (world_r, world_c)

            # Discover chargers
            if "charger" in tag_name:
                if world_pos not in state.discovered_chargers:
                    state.discovered_chargers.add(world_pos)
                    self._logger.info(f"  ★ DISCOVERED charger at {world_pos}")

                    # Create charger info entry
                    if world_pos not in state.charger_info:
                        state.charger_info[world_pos] = ChargerInfo(position=world_pos)

            # Discover extractors
            for resource_type in ["carbon", "oxygen", "germanium", "silicon"]:
                if f"{resource_type}_extractor" in tag_name:
                    extractor_key = f"{resource_type}_extractors"
                    if extractor_key not in state.discovered_extractors:
                        state.discovered_extractors[extractor_key] = set()

                    if world_pos not in state.discovered_extractors[extractor_key]:
                        state.discovered_extractors[extractor_key].add(world_pos)
                        self._logger.info(f"  ★ DISCOVERED {resource_type} extractor at {world_pos}")

                        # Track that we found this resource type
                        state.found_resource_types.add(resource_type)

            # Discover assembler
            if "assembler" in tag_name:
                if world_pos not in state.discovered_assemblers:
                    state.discovered_assemblers.add(world_pos)
                    self._logger.info(f"  ★ DISCOVERED assembler at {world_pos}")

            # Discover chest
            if "chest" in tag_name:
                if world_pos not in state.discovered_chests:
                    state.discovered_chests.add(world_pos)
                    self._logger.info(f"  ★ DISCOVERED chest at {world_pos}")

    def update_explored_cells(self, state: HarvestState):
        """Mark cells in current observation as explored.

        Args:
            state: Current agent state (will be modified)
        """
        if state.current_obs is None:
            return

        # Mark all cells in observation as explored
        for obs_r in range(2 * self._obs_hr + 1):
            for obs_c in range(2 * self._obs_wr + 1):
                world_r = obs_r - self._obs_hr + state.row
                world_c = obs_c - self._obs_wr + state.col

                # Check bounds
                if 0 <= world_r < state.map_height and 0 <= world_c < state.map_width:
                    state.explored_cells.add((world_r, world_c))
