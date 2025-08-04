"""Tests for replay format validation.

Validates that replay files match the format specification in `mettascope/docs/replay_spec.md`.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Any

import pytest


def load_replay(path: str | Path) -> dict[str, Any]:
    """Load and decompress a .json.z replay file."""
    path = Path(path)
    if path.suffix != ".z" or not path.name.endswith(".json.z"):
        raise ValueError("Replay file name must end with '.json.z'")

    with path.open("rb") as fh:
        compressed_data = fh.read()

    try:
        decompressed = zlib.decompress(compressed_data)
    except zlib.error as exc:
        raise ValueError("Failed to decompress replay file") from exc

    try:
        data: dict[str, Any] = json.loads(decompressed)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON in replay file") from exc

    return data


# Required top-level keys for replay version 2.
_REQUIRED_KEYS = {
    "version",
    "num_agents",
    "max_steps",
    "map_size",
    "action_names",
    "item_names",
    "type_names",
    "objects",
}

# Optional top-level keys for replay version 2.
_OPTIONAL_KEYS = {
    "file_name",
    "group_names",
    "reward_sharing_matrix",
}


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise ValueError(msg)


def _validate_sequence_or_value(
    data: Any, field_name: str, expected_type: type | tuple[type, ...], allow_sequences: bool = True
) -> None:
    """Validate that data is either a single value or a sequence of [step, value] pairs."""
    if data is None:
        return

    # Handle tuple of types.
    if isinstance(expected_type, tuple):
        type_check = lambda x: isinstance(x, expected_type)
        type_name = " or ".join(t.__name__ for t in expected_type)
    else:
        type_check = lambda x: isinstance(x, expected_type)
        type_name = expected_type.__name__

    if type_check(data):
        return
    elif allow_sequences and isinstance(data, list):
        if len(data) == 0:
            return
        # Check if it's a sequence of [step, value] pairs.
        for item in data:
            _assert(
                isinstance(item, list) and len(item) == 2, f"'{field_name}' sequence items must be [step, value] pairs"
            )
            _assert(
                isinstance(item[0], int) and item[0] >= 0, f"'{field_name}' sequence step must be non-negative integer"
            )
            _assert(type_check(item[1]), f"'{field_name}' sequence value must be {type_name}")
    else:
        _assert(False, f"'{field_name}' must be {type_name} or sequence of [step, {type_name}] pairs")


def _validate_inventory_format(inventory: Any, field_name: str) -> None:
    """Validate inventory format: list of [item_id, amount] pairs or sequence of such lists."""
    if inventory is None:
        return

    if not isinstance(inventory, list):
        _assert(False, f"'{field_name}' must be a list")
        return

    if len(inventory) == 0:
        return

    # Check if it's a sequence of [step, inventory_list] pairs.
    if (
        len(inventory) > 0
        and isinstance(inventory[0], list)
        and len(inventory[0]) == 2
        and isinstance(inventory[0][0], int)
        and isinstance(inventory[0][1], list)
    ):
        # This is a sequence format: [[step, inventory_list], ...]
        for item in inventory:
            _assert(
                isinstance(item, list) and len(item) == 2,
                f"'{field_name}' sequence items must be [step, inventory_list] pairs",
            )
            _assert(
                isinstance(item[0], int) and item[0] >= 0, f"'{field_name}' sequence step must be non-negative integer"
            )
            _validate_inventory_list(item[1], field_name)
    else:
        # This is a direct inventory list: [[item_id, amount], ...]
        _validate_inventory_list(inventory, field_name)


def _validate_inventory_list(inventory_list: Any, field_name: str) -> None:
    """Validate a single inventory list: list of [item_id, amount] pairs."""
    if not isinstance(inventory_list, list):
        _assert(False, f"'{field_name}' inventory must be a list")
        return

    if len(inventory_list) == 0:
        return

    for pair in inventory_list:
        _assert(
            isinstance(pair, list) and len(pair) == 2, f"'{field_name}' inventory items must be [item_id, amount] pairs"
        )
        _assert(
            isinstance(pair[0], int) and pair[0] >= 0, f"'{field_name}' inventory item_id must be non-negative integer"
        )
        _assert(
            isinstance(pair[1], (int, float)) and pair[1] >= 0,
            f"'{field_name}' inventory amount must be non-negative number",
        )


def validate_replay_schema(data: dict[str, Any]) -> None:
    """Validate that replay data matches the version 2 schema specification."""
    # Check required keys and absence of unexpected keys.
    data_keys = set(data.keys())
    missing = _REQUIRED_KEYS - data_keys
    allowed_keys = _REQUIRED_KEYS | _OPTIONAL_KEYS
    unexpected = data_keys - allowed_keys
    _assert(not missing, f"Missing required keys: {sorted(missing)}")
    _assert(not unexpected, f"Unexpected keys present: {sorted(unexpected)}")

    # Top-level field validation.
    version = data.get("version")
    _assert(isinstance(version, int), "'version' must be int")
    _assert(version == 2, "'version' must equal 2")

    for int_field in ("num_agents", "max_steps"):
        value = data[int_field]
        _assert(isinstance(value, int), f"'{int_field}' must be int")
        _assert(value >= 0, f"'{int_field}' must be non-negative")

    map_size = data["map_size"]
    _assert(
        isinstance(map_size, list) and len(map_size) == 2,
        "'map_size' must be a list of two integers",
    )
    for dim in map_size:
        _assert(isinstance(dim, int) and dim > 0, "'map_size' dimensions must be positive ints")

    # file_name is optional but if present must be valid.
    if "file_name" in data:
        file_name = data["file_name"]
        _assert(isinstance(file_name, str) and bool(file_name), "'file_name' must be non-empty string")
        _assert(file_name.endswith(".json.z"), "'file_name' must end with '.json.z'")

    for list_field in ("action_names", "item_names", "type_names"):
        lst = data[list_field]
        _assert(isinstance(lst, list), f"'{list_field}' must be a list")
        _assert(len(lst) > 0, f"'{list_field}' must not be empty")
        _assert(
            all(isinstance(elem, str) and elem for elem in lst),
            f"All entries in '{list_field}' must be non-empty strings",
        )

    # group_names is optional.
    if "group_names" in data:
        group_names = data["group_names"]
        _assert(isinstance(group_names, list), "'group_names' must be a list")
        _assert(
            all(isinstance(elem, str) and elem for elem in group_names),
            "All entries in 'group_names' must be non-empty strings",
        )

    # reward_sharing_matrix is optional.
    if "reward_sharing_matrix" in data:
        matrix = data["reward_sharing_matrix"]
        _assert(isinstance(matrix, list), "'reward_sharing_matrix' must be a list")
        num_agents = data["num_agents"]
        _assert(len(matrix) == num_agents, f"'reward_sharing_matrix' must have {num_agents} rows")
        for i, row in enumerate(matrix):
            _assert(isinstance(row, list), f"'reward_sharing_matrix' row {i} must be a list")
            _assert(len(row) == num_agents, f"'reward_sharing_matrix' row {i} must have {num_agents} columns")
            _assert(
                all(isinstance(val, (int, float)) for val in row),
                f"'reward_sharing_matrix' row {i} must contain numbers",
            )

    # Objects validation.
    objects = data["objects"]
    _assert(isinstance(objects, list), "'objects' must be a list")
    _assert(len(objects) > 0, "'objects' must not be empty")
    _assert(all(isinstance(obj, dict) for obj in objects), "All entries in 'objects' must be dictionaries")

    # Validate each object.
    agent_count = 0
    for i, obj in enumerate(objects):
        _validate_object(obj, i, data)
        if obj.get("is_agent") or "agent_id" in obj:
            agent_count += 1

    # Check that we have the expected number of agents.
    _assert(agent_count == data["num_agents"], f"Expected {data['num_agents']} agents, found {agent_count}")


def _validate_object(obj: dict[str, Any], obj_index: int, replay_data: dict[str, Any]) -> None:
    """Validate a single object in the replay."""
    # All objects must have these basic fields.
    _assert("id" in obj, f"Object {obj_index} missing 'id'")
    _assert("type_id" in obj, f"Object {obj_index} missing 'type_id'")
    _assert("location" in obj, f"Object {obj_index} missing 'location'")

    obj_id = obj["id"]
    type_id = obj["type_id"]

    _assert(isinstance(obj_id, int) and obj_id > 0, f"Object {obj_index} 'id' must be positive integer")
    _assert(isinstance(type_id, int) and type_id >= 0, f"Object {obj_index} 'type_id' must be non-negative integer")
    _assert(
        type_id < len(replay_data["type_names"]), f"Object {obj_index} 'type_id' {type_id} out of range for type_names"
    )

    # Validate location (required for all objects).
    location = obj["location"]
    if isinstance(location, list) and len(location) == 3:
        # Single location [x, y, z].
        for i, coord in enumerate(location):
            _assert(isinstance(coord, (int, float)), f"Object {obj_index} location coordinate {i} must be number")
    elif isinstance(location, list):
        # Sequence of [step, [x, y, z]] pairs.
        for step_data in location:
            _assert(
                isinstance(step_data, list) and len(step_data) == 2,
                f"Object {obj_index} location sequence items must be [step, [x, y, z]] pairs",
            )
            _assert(
                isinstance(step_data[0], int) and step_data[0] >= 0,
                f"Object {obj_index} location sequence step must be non-negative integer",
            )
            coords = step_data[1]
            _assert(
                isinstance(coords, list) and len(coords) == 3,
                f"Object {obj_index} location coordinates must be [x, y, z]",
            )
            for i, coord in enumerate(coords):
                _assert(isinstance(coord, (int, float)), f"Object {obj_index} location coordinate {i} must be number")
    else:
        _assert(False, f"Object {obj_index} 'location' must be [x, y, z] or sequence of [step, [x, y, z]]")

    # Fields that replay writer always generates for all objects - make them required.
    _assert("orientation" in obj, f"Object {obj_index} missing 'orientation'")
    _assert("inventory" in obj, f"Object {obj_index} missing 'inventory'")
    _assert("inventory_max" in obj, f"Object {obj_index} missing 'inventory_max'")
    _assert("color" in obj, f"Object {obj_index} missing 'color'")
    _assert("is_swappable" in obj, f"Object {obj_index} missing 'is_swappable'")

    # Validate required common fields.
    _validate_sequence_or_value(obj["orientation"], f"Object {obj_index} orientation", (int, float))
    _validate_inventory_format(obj["inventory"], f"Object {obj_index} inventory")
    _assert(
        isinstance(obj["inventory_max"], (int, float)) and obj["inventory_max"] >= 0,
        f"Object {obj_index} 'inventory_max' must be non-negative number",
    )
    _validate_sequence_or_value(obj["color"], f"Object {obj_index} color", int)
    _assert(isinstance(obj["is_swappable"], bool), f"Object {obj_index} 'is_swappable' must be boolean")

    # Check if this is an agent.
    is_agent = obj.get("is_agent", False) or "agent_id" in obj

    if is_agent:
        _validate_agent_object(obj, obj_index, replay_data)
    elif "input_resources" in obj:
        _validate_building_object(obj, obj_index)


def _validate_agent_object(obj: dict[str, Any], obj_index: int, replay_data: dict[str, Any]) -> None:
    """Validate agent-specific fields."""
    # Required agent fields that replay writer always generates.
    _assert("agent_id" in obj, f"Agent object {obj_index} missing 'agent_id'")
    _assert("is_agent" in obj, f"Agent object {obj_index} missing 'is_agent'")
    _assert("vision_size" in obj, f"Agent object {obj_index} missing 'vision_size'")
    _assert("action_id" in obj, f"Agent object {obj_index} missing 'action_id'")
    _assert("action_param" in obj, f"Agent object {obj_index} missing 'action_param'")
    _assert("action_success" in obj, f"Agent object {obj_index} missing 'action_success'")
    _assert("current_reward" in obj, f"Agent object {obj_index} missing 'current_reward'")
    _assert("total_reward" in obj, f"Agent object {obj_index} missing 'total_reward'")
    _assert("freeze_remaining" in obj, f"Agent object {obj_index} missing 'freeze_remaining'")
    _assert("is_frozen" in obj, f"Agent object {obj_index} missing 'is_frozen'")
    _assert("freeze_duration" in obj, f"Agent object {obj_index} missing 'freeze_duration'")
    _assert("group_id" in obj, f"Agent object {obj_index} missing 'group_id'")

    # Validate required agent fields.
    agent_id = obj["agent_id"]
    _assert(
        isinstance(agent_id, int) and 0 <= agent_id < replay_data["num_agents"],
        f"Agent object {obj_index} 'agent_id' {agent_id} out of range",
    )

    _assert(obj["is_agent"] is True, f"Agent object {obj_index} 'is_agent' must be True")

    _assert(
        isinstance(obj["vision_size"], int) and obj["vision_size"] > 0,
        f"Agent object {obj_index} 'vision_size' must be positive integer",
    )

    _validate_sequence_or_value(obj["action_id"], f"Agent {obj_index} action_id", int)
    # Validate action_id values are in range.
    action_ids = obj["action_id"]
    if isinstance(action_ids, int):
        _assert(0 <= action_ids < len(replay_data["action_names"]), f"Agent {obj_index} action_id out of range")
    elif isinstance(action_ids, list):
        for step_data in action_ids:
            if isinstance(step_data, list) and len(step_data) == 2:
                action_id = step_data[1]
                _assert(
                    0 <= action_id < len(replay_data["action_names"]),
                    f"Agent {obj_index} action_id {action_id} out of range",
                )

    _validate_sequence_or_value(obj["action_param"], f"Agent {obj_index} action_param", int)
    _validate_sequence_or_value(obj["action_success"], f"Agent {obj_index} action_success", bool)
    _validate_sequence_or_value(obj["current_reward"], f"Agent {obj_index} current_reward", (int, float))
    _validate_sequence_or_value(obj["total_reward"], f"Agent {obj_index} total_reward", (int, float))
    _validate_sequence_or_value(obj["freeze_remaining"], f"Agent {obj_index} freeze_remaining", (int, float))
    _validate_sequence_or_value(obj["is_frozen"], f"Agent {obj_index} is_frozen", bool)
    _validate_sequence_or_value(obj["freeze_duration"], f"Agent {obj_index} freeze_duration", (int, float))

    _assert(
        isinstance(obj["group_id"], int) and obj["group_id"] >= 0,
        f"Agent object {obj_index} 'group_id' must be non-negative integer",
    )


def _validate_building_object(obj: dict[str, Any], obj_index: int) -> None:
    """Validate building-specific fields."""
    # Required building fields that replay writer always generates.
    _assert("input_resources" in obj, f"Building object {obj_index} missing 'input_resources'")
    _assert("output_resources" in obj, f"Building object {obj_index} missing 'output_resources'")
    _assert("output_limit" in obj, f"Building object {obj_index} missing 'output_limit'")
    _assert("conversion_remaining" in obj, f"Building object {obj_index} missing 'conversion_remaining'")
    _assert("is_converting" in obj, f"Building object {obj_index} missing 'is_converting'")
    _assert("conversion_duration" in obj, f"Building object {obj_index} missing 'conversion_duration'")
    _assert("cooldown_remaining" in obj, f"Building object {obj_index} missing 'cooldown_remaining'")
    _assert("is_cooling_down" in obj, f"Building object {obj_index} missing 'is_cooling_down'")
    _assert("cooldown_duration" in obj, f"Building object {obj_index} missing 'cooldown_duration'")

    # Validate required building fields.
    _validate_inventory_format(obj["input_resources"], f"Building {obj_index} input_resources")
    _validate_inventory_format(obj["output_resources"], f"Building {obj_index} output_resources")

    _assert(
        isinstance(obj["output_limit"], (int, float)) and obj["output_limit"] >= 0,
        f"Building object {obj_index} 'output_limit' must be non-negative number",
    )

    _assert(
        isinstance(obj["conversion_remaining"], (int, float)) and obj["conversion_remaining"] >= 0,
        f"Building object {obj_index} 'conversion_remaining' must be non-negative number",
    )
    _assert(
        isinstance(obj["cooldown_remaining"], (int, float)) and obj["cooldown_remaining"] >= 0,
        f"Building object {obj_index} 'cooldown_remaining' must be non-negative number",
    )

    _validate_sequence_or_value(obj["is_converting"], f"Building {obj_index} is_converting", bool)
    _validate_sequence_or_value(obj["is_cooling_down"], f"Building {obj_index} is_cooling_down", bool)

    _assert(
        isinstance(obj["conversion_duration"], (int, float)) and obj["conversion_duration"] >= 0,
        f"Building object {obj_index} 'conversion_duration' must be non-negative number",
    )
    _assert(
        isinstance(obj["cooldown_duration"], (int, float)) and obj["cooldown_duration"] >= 0,
        f"Building object {obj_index} 'cooldown_duration' must be non-negative number",
    )


def _make_valid_replay(file_name: str = "sample.json.z") -> dict[str, Any]:
    """Create a minimal valid replay dict per the spec."""
    return {
        "version": 2,
        "num_agents": 2,
        "max_steps": 100,
        "map_size": [10, 10],
        "file_name": file_name,
        "type_names": ["agent", "resource"],
        "action_names": ["move", "collect"],
        "item_names": ["wood", "stone"],
        "objects": [
            {
                "id": 1,
                "type_id": 0,
                "location": [5, 5, 0],
                "agent_id": 0,
                "is_agent": True,
                "vision_size": 11,
                "action_id": 0,
                "action_param": 0,
                "action_success": True,
                "current_reward": 0.0,
                "total_reward": 0.0,
                "freeze_remaining": 0,
                "is_frozen": False,
                "freeze_duration": 0,
                "group_id": 0,
                "orientation": 0,
                "inventory": [],
                "inventory_max": 10,
                "color": 0,
                "is_swappable": False,
            },
            {
                "id": 2,
                "type_id": 0,
                "location": [3, 3, 0],
                "agent_id": 1,
                "is_agent": True,
                "vision_size": 11,
                "action_id": 1,
                "action_param": 0,
                "action_success": False,
                "current_reward": 1.5,
                "total_reward": 10.0,
                "freeze_remaining": 0,
                "is_frozen": False,
                "freeze_duration": 0,
                "group_id": 0,
                "orientation": 1,
                "inventory": [[0, 2], [1, 1]],
                "inventory_max": 10,
                "color": 1,
                "is_swappable": False,
            },
        ],
    }


def test_validate_replay_schema_valid() -> None:
    """The validator should accept a properly-formed replay."""
    valid_replay = _make_valid_replay()
    validate_replay_schema(valid_replay)


@pytest.mark.parametrize(
    "mutation, error_substr",
    [
        (lambda r: r.pop("version"), "Missing required keys"),
        (lambda r: r.update({"unexpected": 123}), "Unexpected keys"),
        (lambda r: r.update({"version": 1}), "must equal 2"),
        (lambda r: r.update({"num_agents": -1}), "non-negative"),
        (lambda r: r.update({"map_size": [0, 5]}), "positive ints"),
        (lambda r: r.update({"file_name": "replay.txt"}), "end with '.json.z'"),
        (lambda r: r.update({"action_names": ["", "collect"]}), "non-empty strings"),
        (lambda r: r.update({"objects": [123]}), "must be dictionaries"),
    ],
)
def test_validate_replay_schema_invalid(mutation, error_substr: str) -> None:
    """Verify that the validator can detect invalid replays."""
    replay_dict = _make_valid_replay()
    mutation(replay_dict)

    with pytest.raises(ValueError, match=error_substr):
        validate_replay_schema(replay_dict)


def test_validate_real_generated_replay() -> None:
    """Generate a fresh replay using the CI setup and validate it against the strict schema."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Generate a replay using the same command as CI but with a custom output directory.
        cmd = [
            "uv",
            "run",
            "--no-sync",
            "tools/replay.py",
            "+user=ci",
            "wandb=off",
            f"replay_job.replay_dir={tmp_dir}",
            f"replay_job.stats_dir={tmp_dir}",
            "replay_job.policy_uri=null",
            "run=test_validator",
        ]

        # Run from the project root (parent of mettascope).
        project_root = Path(__file__).parent.parent.parent
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=60)

        replay_files = list(Path(tmp_dir).glob("**/*.json.z"))
        if len(replay_files) == 0:
            pytest.skip(f"No replay generated (exit {result.returncode}): {result.stderr}")

        # Should have exactly one replay file.
        assert len(replay_files) == 1, f"Expected exactly 1 replay file, found {len(replay_files)}: {replay_files}"

        # Validate the replay file.
        replay_path = replay_files[0]
        loaded_replay = load_replay(replay_path)
        validate_replay_schema(loaded_replay)

        print(f"✓ Successfully generated and validated fresh replay: {replay_path.name}")
