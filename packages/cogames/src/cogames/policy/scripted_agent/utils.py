"""
Utility functions for scripted agents.

Pure/stateless helper functions that can be reused across different agents.
"""

from __future__ import annotations

from typing import Any, Union

from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

from .types import ObjectState, ParsedObservation, SimpleAgentState


def manhattan_distance(pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def is_adjacent(pos1: tuple[int, int], pos2: tuple[int, int]) -> bool:
    """Check if two positions are adjacent (4-way cardinal directions)."""
    dr = abs(pos1[0] - pos2[0])
    dc = abs(pos1[1] - pos2[1])
    return (dr == 1 and dc == 0) or (dr == 0 and dc == 1)


def is_within_bounds(pos: tuple[int, int], map_height: int, map_width: int) -> bool:
    """Check if a position is within map bounds."""
    r, c = pos
    return 0 <= r < map_height and 0 <= c < map_width


def get_cardinal_neighbors(pos: tuple[int, int]) -> list[tuple[int, int]]:
    """Get all 4-way cardinal neighbor positions."""
    r, c = pos
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def is_wall(obj_name: str) -> bool:
    """Check if an object name represents a wall or obstacle."""
    return "wall" in obj_name or "#" in obj_name or obj_name in {"wall", "obstacle"}


def is_floor(obj_name: str) -> bool:
    """Check if an object name represents floor (passable empty space)."""
    # Environment returns empty string for empty cells
    return obj_name in {"floor", ""}


def is_station(obj_name: str, station: str) -> bool:
    """Check if an object name contains a specific station type."""
    return station in obj_name


def position_to_direction(from_pos: tuple[int, int], to_pos: tuple[int, int]) -> str | None:
    """
    Convert adjacent positions to a cardinal direction name.

    Returns: "north", "south", "east", "west", or None if not adjacent.
    """
    dr = to_pos[0] - from_pos[0]
    dc = to_pos[1] - from_pos[1]

    if dr == -1 and dc == 0:
        return "north"
    elif dr == 1 and dc == 0:
        return "south"
    elif dr == 0 and dc == 1:
        return "east"
    elif dr == 0 and dc == -1:
        return "west"
    return None


def process_feature_at_position(
    position_features: dict[tuple[int, int], dict[str, Union[int, list[int], dict[str, int]]]],
    pos: tuple[int, int],
    feature_name: str,
    value: int,
    *,
    spatial_feature_names: set[str],
    agent_feature_key_by_name: dict[str, str],
    protocol_input_prefix: str,
    protocol_output_prefix: str,
) -> None:
    """Process a single observation feature and add it to position_features."""
    if pos not in position_features:
        position_features[pos] = {}

    # Handle spatial features (tag, cooldown, etc.)
    if feature_name in spatial_feature_names:
        # Tag: collect all tags as a list (objects can have multiple tags)
        if feature_name == "tag":
            tags = position_features[pos].setdefault("tags", [])
            if isinstance(tags, list):
                tags.append(value)
            return
        # Other spatial features are single values
        position_features[pos][feature_name] = value
        return

    # Handle agent features (agent:group -> agent_group, etc.)
    agent_feature_key = agent_feature_key_by_name.get(feature_name)
    if agent_feature_key is not None:
        position_features[pos][agent_feature_key] = value
        return

    # Handle protocol features (recipes)
    if feature_name.startswith(protocol_input_prefix):
        resource = feature_name[len(protocol_input_prefix) :]
        inputs = position_features[pos].setdefault("protocol_inputs", {})
        if isinstance(inputs, dict):
            inputs[resource] = value
        return

    if feature_name.startswith(protocol_output_prefix):
        resource = feature_name[len(protocol_output_prefix) :]
        outputs = position_features[pos].setdefault("protocol_outputs", {})
        if isinstance(outputs, dict):
            outputs[resource] = value
        return


def create_object_state(
    features: dict[str, Union[int, list[int], dict[str, int]]],
    *,
    tag_names: dict[int, str],
) -> ObjectState:
    """Create an ObjectState from collected features.

    Note: Objects can have multiple tags (e.g., "wall" + "green" vibe).
    We use the first tag as the primary object name.
    """
    # Get tags list (now stored as "tags" instead of "tag")
    tags_value = features.get("tags", [])
    if isinstance(tags_value, list):
        tag_ids = list(tags_value)
    elif isinstance(tags_value, int):
        tag_ids = [tags_value]
    else:
        tag_ids = []

    # Use first tag as primary object name
    if tag_ids:
        primary_tag_id = tag_ids[0]
        obj_name = tag_names.get(primary_tag_id, f"unknown_tag_{primary_tag_id}")
    else:
        obj_name = "unknown"

    # Helper to safely extract int values
    def get_int(key: str, default: int) -> int:
        val = features.get(key, default)
        return int(val) if isinstance(val, int) else default

    # Helper to safely extract dict values
    def get_dict(key: str) -> dict[str, int]:
        val = features.get(key, {})
        return dict(val) if isinstance(val, dict) else {}

    return ObjectState(
        name=obj_name,
        cooldown_remaining=get_int("cooldown_remaining", 0),
        clipped=get_int("clipped", 0),
        remaining_uses=get_int("remaining_uses", 999),
        protocol_inputs=get_dict("protocol_inputs"),
        protocol_outputs=get_dict("protocol_outputs"),
        agent_group=get_int("agent_group", -1),
        agent_frozen=get_int("agent_frozen", 0),
    )


def read_inventory_from_obs(
    state: SimpleAgentState,
    obs: AgentObservation,
    *,
    obs_hr: int,
    obs_wr: int,
) -> None:
    """Read inventory from observation tokens at center cell and update state."""
    inv = {}
    center_r, center_c = obs_hr, obs_wr
    for tok in obs.tokens:
        if tok.location == (center_r, center_c):
            feature_name = tok.feature.name
            if feature_name.startswith("inv:"):
                resource_name = feature_name[4:]  # Remove "inv:" prefix
                inv[resource_name] = tok.value

    state.energy = inv.get("energy", 0)
    state.carbon = inv.get("carbon", 0)
    state.oxygen = inv.get("oxygen", 0)
    state.germanium = inv.get("germanium", 0)
    state.silicon = inv.get("silicon", 0)
    state.hearts = inv.get("heart", 0)
    state.decoder = inv.get("decoder", 0)
    state.modulator = inv.get("modulator", 0)
    state.resonator = inv.get("resonator", 0)
    state.scrambler = inv.get("scrambler", 0)


def parse_observation(
    state: SimpleAgentState,
    obs: AgentObservation,
    *,
    obs_hr: int,
    obs_wr: int,
    spatial_feature_names: set[str],
    agent_feature_key_by_name: dict[str, str],
    protocol_input_prefix: str,
    protocol_output_prefix: str,
    tag_names: dict[int, str],
    debug: bool = False,
) -> ParsedObservation:
    """Parse token-based observation into structured format.

    AgentObservation with tokens (ObservationToken list)
    - Inventory is obtained via agent.inventory (not parsed here)
    - Only spatial features are parsed from observations

    Converts egocentric spatial coordinates to world coordinates using agent position.
    Agent position (agent_row, agent_col) comes from simulation.grid_objects().
    """
    # First pass: collect all spatial features by position
    position_features: dict[tuple[int, int], dict[str, Union[int, list[int], dict[str, int]]]] = {}

    for tok in obs.tokens:
        obs_r, obs_c = tok.location
        feature_name = tok.feature.name
        value = tok.value

        # Skip center location - that's inventory/global obs, obtained via agent.inventory
        if obs_r == obs_hr and obs_c == obs_wr:
            continue

        # Convert observation-relative coords to world coords
        if state.row >= 0 and state.col >= 0:
            r = obs_r - obs_hr + state.row
            c = obs_c - obs_wr + state.col

            if 0 <= r < state.map_height and 0 <= c < state.map_width:
                process_feature_at_position(
                    position_features,
                    (r, c),
                    feature_name,
                    value,
                    spatial_feature_names=spatial_feature_names,
                    agent_feature_key_by_name=agent_feature_key_by_name,
                    protocol_input_prefix=protocol_input_prefix,
                    protocol_output_prefix=protocol_output_prefix,
                )

    # Second pass: create ObjectState for each position with tags
    nearby_objects = {
        pos: create_object_state(features, tag_names=tag_names)
        for pos, features in position_features.items()
        if "tags" in features  # Note: stored as "tags" (plural) to support multiple tags per object
    }

    return ParsedObservation(
        row=state.row,
        col=state.col,
        energy=0,  # Inventory obtained via agent.inventory
        carbon=0,
        oxygen=0,
        germanium=0,
        silicon=0,
        hearts=0,
        decoder=0,
        modulator=0,
        resonator=0,
        scrambler=0,
        nearby_objects=nearby_objects,
    )


def change_vibe_action(
    vibe_name: str,
    *,
    actions: Any,  # PolicyEnvInterface.actions
) -> Action:
    """
    Return a safe vibe-change action.
    Guard against disabled or single-vibe configurations before issuing the action.
    """
    from mettagrid.config.vibes import VIBE_BY_NAME

    change_vibe_cfg = getattr(actions, "change_vibe", None)
    if change_vibe_cfg is None:
        return actions.noop.Noop()
    if not getattr(change_vibe_cfg, "enabled", True):
        return actions.noop.Noop()
    num_vibes = len(getattr(change_vibe_cfg, "vibes", []))
    if num_vibes <= 1:
        return actions.noop.Noop()
    # Raise loudly if the requested vibe isn't registered instead of silently
    # falling back to noop; otherwise config issues become very hard to spot.
    vibe = VIBE_BY_NAME.get(vibe_name)
    if vibe is None:
        raise Exception(f"No valid vibes called {vibe_name}")
    return actions.change_vibe.ChangeVibe(vibe)


def update_agent_position(
    state: SimpleAgentState,
    *,
    move_deltas: dict[str, tuple[int, int]],
) -> None:
    """Update agent position based on last action.

    Position is tracked relative to origin (starting position), using only movement deltas.
    No dependency on simulation.grid_objects().

    IMPORTANT: When using objects (extractors, stations), the agent "moves into" them but doesn't
    actually change position. We detect this by checking the using_object_this_step flag.
    """
    # If last action was a move and we're not using an object, update position
    # We assume the move succeeded unless we were using an object
    if state.last_action and state.last_action.name.startswith("move_") and not state.using_object_this_step:
        # Extract direction from action name (e.g., "move_north" -> "north")
        direction = state.last_action.name[5:]  # Remove "move_" prefix
        if direction in move_deltas:
            dr, dc = move_deltas[direction]
            state.row += dr
            state.col += dc
    # Clear the flag for next step
    state.using_object_this_step = False


def use_object_at(
    state: SimpleAgentState,
    target_pos: tuple[int, int],
    *,
    actions: Any,  # PolicyEnvInterface.actions
    move_deltas: dict[str, tuple[int, int]],
    using_for: str = "",
) -> Action:
    """Use an object by moving into its cell. Sets a flag so position tracking knows not to update.

    This is the generic "move into to use" action for extractors, assemblers, chests, chargers, etc.
    The 'using_for' parameter is used for tracking what we're using (e.g., 'extractor', 'assembler').
    """
    action = move_into_cell(state, target_pos, actions=actions, move_deltas=move_deltas)

    # Mark that we're using an object so position tracking doesn't update
    state.using_object_this_step = True

    return action


def move_into_cell(
    state: SimpleAgentState,
    target: tuple[int, int],
    *,
    actions: Any,  # PolicyEnvInterface.actions
    move_deltas: dict[str, tuple[int, int]],
) -> Action:
    """Return the action that attempts to step into the target cell.

    Checks for agent occupancy before moving to avoid collisions.
    """

    tr, tc = target
    if state.row == tr and state.col == tc:
        return actions.noop.Noop()
    dr = tr - state.row
    dc = tc - state.col

    # Check if another agent is at the target position
    if (tr, tc) in state.agent_occupancy:
        # Another agent is blocking the target, wait or try alternative
        # For a simple fallback, return noop (caller can handle random direction if needed)
        return actions.noop.Noop()

    if dr == -1:
        return actions.move.Move("north")
    if dr == 1:
        return actions.move.Move("south")
    if dc == 1:
        return actions.move.Move("east")
    if dc == -1:
        return actions.move.Move("west")
    # Fallback to noop if offsets unexpected
    return actions.noop.Noop()
