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


def _require_fields(obj: dict[str, Any], fields: list[str], obj_name: str) -> None:
    """Assert that all required fields are present."""
    missing = [f for f in fields if f not in obj]
    if missing:
        raise ValueError(f"{obj_name} missing required fields: {missing}")


def _validate_type(value: Any, expected_type: type | tuple[type, ...], field_name: str) -> None:
    """Validate that value has the expected type."""
    if not isinstance(value, expected_type):
        type_name = (
            expected_type.__name__
            if isinstance(expected_type, type)
            else " or ".join(t.__name__ for t in expected_type)
        )
        raise ValueError(f"'{field_name}' must be {type_name}, got {type(value).__name__}")


def _validate_positive_int(value: Any, field_name: str) -> None:
    """Validate that value is a positive integer."""
    _validate_type(value, int, field_name)
    if value <= 0:
        raise ValueError(f"'{field_name}' must be positive, got {value}")


def _validate_non_negative_number(value: Any, field_name: str) -> None:
    """Validate that value is a non-negative number."""
    _validate_type(value, (int, float), field_name)
    if value < 0:
        raise ValueError(f"'{field_name}' must be non-negative, got {value}")


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
    _assert(data.get("version") == 2, f"'version' must equal 2, got {data.get('version')}")

    _validate_positive_int(data["num_agents"], "num_agents")
    _validate_non_negative_number(data["max_steps"], "max_steps")

    map_size = data["map_size"]
    _validate_type(map_size, list, "map_size")
    _assert(len(map_size) == 2, "'map_size' must have exactly 2 dimensions")
    for i, dim in enumerate(map_size):
        _validate_positive_int(dim, f"map_size[{i}]")

    # Optional file_name validation.
    if "file_name" in data:
        file_name = data["file_name"]
        _validate_type(file_name, str, "file_name")
        _assert(file_name and file_name.endswith(".json.z"), "'file_name' must be non-empty and end with '.json.z'")

    # String list validation.
    for field in ["action_names", "item_names", "type_names"]:
        lst = data[field]
        _validate_type(lst, list, field)
        _assert(len(lst) > 0, f"'{field}' must not be empty")
        _assert(all(isinstance(s, str) and s for s in lst), f"'{field}' must contain non-empty strings")

    # Optional string lists.
    if "group_names" in data:
        group_names = data["group_names"]
        _validate_type(group_names, list, "group_names")
        _assert(all(isinstance(s, str) and s for s in group_names), "'group_names' must contain non-empty strings")

    # Optional reward sharing matrix.
    if "reward_sharing_matrix" in data:
        matrix = data["reward_sharing_matrix"]
        _validate_type(matrix, list, "reward_sharing_matrix")
        num_agents = data["num_agents"]
        _assert(len(matrix) == num_agents, f"'reward_sharing_matrix' must have {num_agents} rows")
        for i, row in enumerate(matrix):
            _validate_type(row, list, f"reward_sharing_matrix[{i}]")
            _assert(len(row) == num_agents, f"'reward_sharing_matrix[{i}]' must have {num_agents} columns")
            _assert(all(isinstance(v, (int, float)) for v in row), f"'reward_sharing_matrix[{i}]' must contain numbers")

    # Objects validation.
    objects = data["objects"]
    _validate_type(objects, list, "objects")
    _assert(len(objects) > 0, "'objects' must not be empty")
    _assert(all(isinstance(obj, dict) for obj in objects), "'objects' must contain dictionaries")

    # Validate each object and count agents.
    agent_count = 0
    for i, obj in enumerate(objects):
        _validate_object(obj, i, data)
        if obj.get("is_agent") or "agent_id" in obj:
            agent_count += 1

    _assert(agent_count == data["num_agents"], f"Expected {data['num_agents']} agents, found {agent_count}")


def _validate_object(obj: dict[str, Any], obj_index: int, replay_data: dict[str, Any]) -> None:
    """Validate a single object in the replay."""
    obj_name = f"Object {obj_index}"

    # All objects have these required fields.
    _require_fields(
        obj,
        ["id", "type_id", "location", "orientation", "inventory", "inventory_max", "color", "is_swappable"],
        obj_name,
    )

    # Validate basic object fields.
    _validate_positive_int(obj["id"], f"{obj_name}.id")

    type_id = obj["type_id"]
    _validate_non_negative_number(type_id, f"{obj_name}.type_id")
    _assert(type_id < len(replay_data["type_names"]), f"{obj_name}.type_id {type_id} out of range")

    # Validate location format.
    _validate_location(obj["location"], obj_name)

    # Validate common fields.
    _validate_sequence_or_value(obj["orientation"], f"{obj_name}.orientation", (int, float))
    _validate_inventory_format(obj["inventory"], f"{obj_name}.inventory")
    _validate_non_negative_number(obj["inventory_max"], f"{obj_name}.inventory_max")
    _validate_sequence_or_value(obj["color"], f"{obj_name}.color", int)
    _validate_type(obj["is_swappable"], bool, f"{obj_name}.is_swappable")

    # Validate specific object types.
    if obj.get("is_agent") or "agent_id" in obj:
        _validate_agent_fields(obj, obj_name, replay_data)
    elif "input_resources" in obj:
        _validate_building_fields(obj, obj_name)


def _validate_location(location: Any, obj_name: str) -> None:
    """Validate location field format."""
    field_name = f"{obj_name}.location"

    if isinstance(location, list) and len(location) == 3:
        # Single location [x, y, z].
        for i, coord in enumerate(location):
            _validate_type(coord, (int, float), f"{field_name}[{i}]")
    elif isinstance(location, list):
        # Sequence of [step, [x, y, z]] pairs.
        for step_data in location:
            _assert(
                isinstance(step_data, list) and len(step_data) == 2,
                f"{field_name} sequence items must be [step, [x, y, z]]",
            )
            _validate_non_negative_number(step_data[0], f"{field_name} step")
            coords = step_data[1]
            _assert(isinstance(coords, list) and len(coords) == 3, f"{field_name} coordinates must be [x, y, z]")
            for i, coord in enumerate(coords):
                _validate_type(coord, (int, float), f"{field_name} coord[{i}]")
    else:
        _assert(False, f"{field_name} must be [x, y, z] or sequence of [step, [x, y, z]]")


def _validate_agent_fields(obj: dict[str, Any], obj_name: str, replay_data: dict[str, Any]) -> None:
    """Validate all agent-specific fields."""
    agent_fields = [
        "agent_id",
        "is_agent",
        "vision_size",
        "action_id",
        "action_param",
        "action_success",
        "current_reward",
        "total_reward",
        "freeze_remaining",
        "is_frozen",
        "freeze_duration",
        "group_id",
    ]
    _require_fields(obj, agent_fields, obj_name)

    # Validate agent_id range.
    agent_id = obj["agent_id"]
    _validate_non_negative_number(agent_id, f"{obj_name}.agent_id")
    _assert(agent_id < replay_data["num_agents"], f"{obj_name}.agent_id {agent_id} out of range")

    # Validate specific agent fields.
    _assert(obj["is_agent"] is True, f"{obj_name}.is_agent must be True")
    _validate_positive_int(obj["vision_size"], f"{obj_name}.vision_size")
    _validate_non_negative_number(obj["group_id"], f"{obj_name}.group_id")

    # Validate sequence fields.
    _validate_sequence_or_value(obj["action_id"], f"{obj_name}.action_id", int)
    _validate_sequence_or_value(obj["action_param"], f"{obj_name}.action_param", int)
    _validate_sequence_or_value(obj["action_success"], f"{obj_name}.action_success", bool)
    _validate_sequence_or_value(obj["current_reward"], f"{obj_name}.current_reward", (int, float))
    _validate_sequence_or_value(obj["total_reward"], f"{obj_name}.total_reward", (int, float))
    _validate_sequence_or_value(obj["freeze_remaining"], f"{obj_name}.freeze_remaining", (int, float))
    _validate_sequence_or_value(obj["is_frozen"], f"{obj_name}.is_frozen", bool)
    _validate_sequence_or_value(obj["freeze_duration"], f"{obj_name}.freeze_duration", (int, float))

    # Validate action_id values are in range.
    _validate_action_id_range(obj["action_id"], obj_name, replay_data["action_names"])


def _validate_action_id_range(action_ids: Any, obj_name: str, action_names: list[str]) -> None:
    """Validate that action_id values are within the valid range."""
    if isinstance(action_ids, int):
        _assert(0 <= action_ids < len(action_names), f"{obj_name}.action_id {action_ids} out of range")
    elif isinstance(action_ids, list):
        for step_data in action_ids:
            if isinstance(step_data, list) and len(step_data) == 2:
                action_id = step_data[1]
                _assert(0 <= action_id < len(action_names), f"{obj_name}.action_id {action_id} out of range")


def _validate_building_fields(obj: dict[str, Any], obj_name: str) -> None:
    """Validate all building-specific fields."""
    building_fields = [
        "input_resources",
        "output_resources",
        "output_limit",
        "conversion_remaining",
        "is_converting",
        "conversion_duration",
        "cooldown_remaining",
        "is_cooling_down",
        "cooldown_duration",
    ]
    _require_fields(obj, building_fields, obj_name)

    # Validate inventory resources.
    _validate_inventory_format(obj["input_resources"], f"{obj_name}.input_resources")
    _validate_inventory_format(obj["output_resources"], f"{obj_name}.output_resources")

    # Validate numeric fields.
    for field in [
        "output_limit",
        "conversion_remaining",
        "cooldown_remaining",
        "conversion_duration",
        "cooldown_duration",
    ]:
        _validate_non_negative_number(obj[field], f"{obj_name}.{field}")

    # Validate boolean sequence fields.
    _validate_sequence_or_value(obj["is_converting"], f"{obj_name}.is_converting", bool)
    _validate_sequence_or_value(obj["is_cooling_down"], f"{obj_name}.is_cooling_down", bool)


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
