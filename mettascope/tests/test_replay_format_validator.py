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
    assert value > 0, f"'{field_name}' must be positive, got {value}"


def _validate_non_negative_number(value: Any, field_name: str) -> None:
    """Validate that value is a non-negative number."""
    _validate_type(value, (int, float), field_name)
    assert value >= 0, f"'{field_name}' must be non-negative, got {value}"


def _validate_string_list(lst: Any, field_name: str, allow_empty: bool = False) -> None:
    """Validate that value is a list of non-empty strings."""
    _validate_type(lst, list, field_name)
    if not allow_empty:
        assert len(lst) > 0, f"'{field_name}' must not be empty"
    assert all(isinstance(s, str) and s for s in lst), f"'{field_name}' must contain non-empty strings"


def _validate_static_value(value: Any, field_name: str, expected_type: type | tuple[type, ...]) -> None:
    """Validate that value is a static value of the expected type (never a time series)."""
    _validate_type(value, expected_type, field_name)


def _validate_time_series(data: Any, field_name: str, expected_type: type | tuple[type, ...]) -> None:
    """Validate time series values: either single values (never changed) or arrays of [step, value] pairs."""
    if data is None:
        return

    # Check if it's a single value (field never changed during replay)
    if isinstance(data, expected_type):
        return

    # Check if it's a time series array (field changed during replay)
    if isinstance(data, list):
        if len(data) == 0:
            return
        # Validate time series of [step, value] pairs
        for item in data:
            assert isinstance(item, list) and len(item) == 2, (
                f"'{field_name}' time series items must be [step, value] pairs"
            )
            assert isinstance(item[0], int) and item[0] >= 0, f"'{field_name}' time series step must be non-negative"
            assert isinstance(item[1], expected_type), (
                f"'{field_name}' time series value must be {_get_type_name(expected_type)}"
            )

        # First entry should be step 0
        assert data[0][0] == 0, f"'{field_name}' time series must start with step 0"
        return

    # Neither single value nor valid time series
    type_name = _get_type_name(expected_type)
    assert False, f"'{field_name}' must be {type_name} or time series of [step, {type_name}] pairs"


def _get_type_name(expected_type: type | tuple[type, ...]) -> str:
    """Get a readable name for a type or tuple of types."""
    return expected_type.__name__ if isinstance(expected_type, type) else " or ".join(t.__name__ for t in expected_type)


def _validate_inventory_format(inventory: Any, field_name: str) -> None:
    """Validate inventory format: single inventory list or time series of [step, inventory_list] pairs."""
    if inventory is None:
        return

    _validate_type(inventory, list, field_name)
    if len(inventory) == 0:
        return

    # Check if it's a single inventory list (never changed during replay)
    # Single inventory format: [[item_id, amount], [item_id, amount], ...]
    if len(inventory) > 0 and all(
        isinstance(item, list) and len(item) == 2 and isinstance(item[0], int) and isinstance(item[1], (int, float))
        for item in inventory
    ):
        _validate_inventory_list(inventory, field_name)
        return

    # Check if it's a time series format: [[step, inventory_list], ...]
    for item in inventory:
        assert isinstance(item, list) and len(item) == 2, (
            f"'{field_name}' time series items must be [step, inventory_list] pairs"
        )
        assert isinstance(item[0], int) and item[0] >= 0, f"'{field_name}' time series step must be non-negative"
        _validate_inventory_list(item[1], field_name)


def _validate_inventory_list(inventory_list: Any, field_name: str) -> None:
    """Validate a single inventory list: list of [item_id, amount] pairs."""
    _validate_type(inventory_list, list, field_name)

    for pair in inventory_list:
        assert isinstance(pair, list) and len(pair) == 2, f"'{field_name}' must contain [item_id, amount] pairs"
        assert isinstance(pair[0], int) and pair[0] >= 0, f"'{field_name}' item_id must be non-negative integer"
        assert isinstance(pair[1], (int, float)) and pair[1] >= 0, f"'{field_name}' amount must be non-negative number"


def validate_replay_schema(data: dict[str, Any]) -> None:
    """Validate that replay data matches the version 2 schema specification."""
    # Check required keys and absence of unexpected keys.
    data_keys = set(data.keys())
    missing = _REQUIRED_KEYS - data_keys
    allowed_keys = _REQUIRED_KEYS | _OPTIONAL_KEYS
    unexpected = data_keys - allowed_keys
    assert not missing, f"Missing required keys: {sorted(missing)}"
    assert not unexpected, f"Unexpected keys present: {sorted(unexpected)}"

    # Top-level field validation.
    assert data.get("version") == 2, f"'version' must equal 2, got {data.get('version')}"

    _validate_positive_int(data["num_agents"], "num_agents")
    _validate_non_negative_number(data["max_steps"], "max_steps")

    # Validate map_size.
    map_size = data["map_size"]
    _validate_type(map_size, list, "map_size")
    assert len(map_size) == 2, "'map_size' must have exactly 2 dimensions"
    for i, dim in enumerate(map_size):
        _validate_positive_int(dim, f"map_size[{i}]")

    # Optional file_name validation.
    if "file_name" in data:
        file_name = data["file_name"]
        _validate_type(file_name, str, "file_name")
        assert file_name and file_name.endswith(".json.z"), "'file_name' must be non-empty and end with '.json.z'"

    # Required string lists.
    for field in ["action_names", "item_names", "type_names"]:
        _validate_string_list(data[field], field)

    # Optional string lists.
    if "group_names" in data:
        _validate_string_list(data["group_names"], "group_names", allow_empty=True)

    # Optional reward sharing matrix.
    if "reward_sharing_matrix" in data:
        matrix = data["reward_sharing_matrix"]
        _validate_type(matrix, list, "reward_sharing_matrix")
        num_agents = data["num_agents"]
        assert len(matrix) == num_agents, f"'reward_sharing_matrix' must have {num_agents} rows"
        for i, row in enumerate(matrix):
            _validate_type(row, list, f"reward_sharing_matrix[{i}]")
            assert len(row) == num_agents, f"'reward_sharing_matrix[{i}]' must have {num_agents} columns"
            assert all(isinstance(v, (int, float)) for v in row), f"'reward_sharing_matrix[{i}]' must contain numbers"

    # Objects validation.
    objects = data["objects"]
    _validate_type(objects, list, "objects")
    assert len(objects) > 0, "'objects' must not be empty"
    assert all(isinstance(obj, dict) for obj in objects), "'objects' must contain dictionaries"

    # Validate each object and count agents.
    agent_count = 0
    for i, obj in enumerate(objects):
        _validate_object(obj, i, data)
        if obj.get("is_agent") or "agent_id" in obj:
            agent_count += 1

    assert agent_count == data["num_agents"], f"Expected {data['num_agents']} agents, found {agent_count}"


def _validate_object(obj: dict[str, Any], obj_index: int, replay_data: dict[str, Any]) -> None:
    """Validate a single object in the replay."""
    obj_name = f"Object {obj_index}"

    # All objects have these required fields.
    required_fields = [
        "id",
        "type_id",
        "position",
        "orientation",
        "inventory",
        "inventory_max",
        "color",
        "is_swappable",
    ]
    _require_fields(obj, required_fields, obj_name)

    # Validate static fields.
    _validate_static_value(obj["id"], f"{obj_name}.id", int)
    _validate_positive_int(obj["id"], f"{obj_name}.id")

    type_id = obj["type_id"]
    _validate_static_value(type_id, f"{obj_name}.type_id", int)
    _validate_non_negative_number(type_id, f"{obj_name}.type_id")
    assert type_id < len(replay_data["type_names"]), f"{obj_name}.type_id {type_id} out of range"

    _validate_static_value(obj["is_swappable"], f"{obj_name}.is_swappable", bool)

    # Validate dynamic fields (always time series).
    _validate_position(obj["position"], obj_name)
    _validate_time_series(obj["orientation"], f"{obj_name}.orientation", (int, float))
    _validate_inventory_format(obj["inventory"], f"{obj_name}.inventory")
    _validate_time_series(obj["inventory_max"], f"{obj_name}.inventory_max", (int, float))
    _validate_time_series(obj["color"], f"{obj_name}.color", int)

    # Validate specific object types.
    if obj.get("is_agent") or "agent_id" in obj:
        _validate_agent_fields(obj, obj_name, replay_data)
    elif "input_resources" in obj:
        _validate_building_fields(obj, obj_name)


def _validate_position(position: Any, obj_name: str) -> None:
    """Validate position field format: single [x, y, z] or time series of [step, [x, y, z]] pairs."""
    field_name = f"{obj_name}.position"

    # Check if it's a single position (never changed during replay)
    if isinstance(position, list) and len(position) == 3:
        for i, coord in enumerate(position):
            _validate_type(coord, (int, float), f"{field_name}[{i}]")
        return

    # Check if it's a time series array (position changed during replay)
    _validate_type(position, list, field_name)
    assert len(position) > 0, f"{field_name} must have at least one entry"

    # Validate time series of [step, [x, y, z]] pairs
    for step_data in position:
        assert isinstance(step_data, list) and len(step_data) == 2, (
            f"{field_name} items must be [step, [x, y, z]] pairs"
        )
        assert isinstance(step_data[0], int) and step_data[0] >= 0, f"{field_name} step must be non-negative"
        coords = step_data[1]
        assert isinstance(coords, list) and len(coords) == 3, f"{field_name} coordinates must be [x, y, z]"
        for i, coord in enumerate(coords):
            _validate_type(coord, (int, float), f"{field_name} coord[{i}]")

    # Must start with step 0
    assert position[0][0] == 0, f"{field_name} must start with step 0"


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

    # Validate static agent fields.
    agent_id = obj["agent_id"]
    _validate_static_value(agent_id, f"{obj_name}.agent_id", int)
    _validate_non_negative_number(agent_id, f"{obj_name}.agent_id")
    assert agent_id < replay_data["num_agents"], f"{obj_name}.agent_id {agent_id} out of range"

    _validate_static_value(obj["is_agent"], f"{obj_name}.is_agent", bool)
    assert obj["is_agent"] is True, f"{obj_name}.is_agent must be True"

    _validate_static_value(obj["vision_size"], f"{obj_name}.vision_size", int)
    _validate_positive_int(obj["vision_size"], f"{obj_name}.vision_size")

    _validate_static_value(obj["group_id"], f"{obj_name}.group_id", int)
    _validate_non_negative_number(obj["group_id"], f"{obj_name}.group_id")

    # Validate dynamic agent fields (always time series).
    _validate_time_series(obj["action_id"], f"{obj_name}.action_id", int)
    _validate_time_series(obj["action_param"], f"{obj_name}.action_param", int)
    _validate_time_series(obj["action_success"], f"{obj_name}.action_success", bool)
    _validate_time_series(obj["current_reward"], f"{obj_name}.current_reward", (int, float))
    _validate_time_series(obj["total_reward"], f"{obj_name}.total_reward", (int, float))
    _validate_time_series(obj["freeze_remaining"], f"{obj_name}.freeze_remaining", (int, float))
    _validate_time_series(obj["is_frozen"], f"{obj_name}.is_frozen", bool)
    _validate_time_series(obj["freeze_duration"], f"{obj_name}.freeze_duration", (int, float))

    # Validate action_id values are in range.
    _validate_action_id_range(obj["action_id"], obj_name, replay_data["action_names"])


def _validate_action_id_range(action_ids: Any, obj_name: str, action_names: list[str]) -> None:
    """Validate that action_id values are within the valid range."""
    if isinstance(action_ids, list):
        for step_data in action_ids:
            if isinstance(step_data, list) and len(step_data) == 2:
                action_id = step_data[1]
                assert 0 <= action_id < len(action_names), f"{obj_name}.action_id {action_id} out of range"


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

    # Validate static building fields.
    _validate_static_value(obj["output_limit"], f"{obj_name}.output_limit", (int, float))
    _validate_non_negative_number(obj["output_limit"], f"{obj_name}.output_limit")

    _validate_static_value(obj["conversion_duration"], f"{obj_name}.conversion_duration", (int, float))
    _validate_non_negative_number(obj["conversion_duration"], f"{obj_name}.conversion_duration")

    _validate_static_value(obj["cooldown_duration"], f"{obj_name}.cooldown_duration", (int, float))
    _validate_non_negative_number(obj["cooldown_duration"], f"{obj_name}.cooldown_duration")

    # Validate dynamic building fields (always time series).
    _validate_inventory_format(obj["input_resources"], f"{obj_name}.input_resources")
    _validate_inventory_format(obj["output_resources"], f"{obj_name}.output_resources")
    _validate_time_series(obj["conversion_remaining"], f"{obj_name}.conversion_remaining", (int, float))
    _validate_time_series(obj["is_converting"], f"{obj_name}.is_converting", bool)
    _validate_time_series(obj["cooldown_remaining"], f"{obj_name}.cooldown_remaining", (int, float))
    _validate_time_series(obj["is_cooling_down"], f"{obj_name}.is_cooling_down", bool)


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
                # Static fields
                "id": 1,
                "type_id": 0,
                "agent_id": 0,
                "is_agent": True,
                "vision_size": 11,
                "group_id": 0,
                "is_swappable": False,
                # Time series fields (some single values, some arrays)
                "position": [5, 5, 0],  # Never moved
                "action_id": 0,  # Never changed action
                "action_param": 0,  # Never changed param
                "action_success": True,  # Never failed
                "current_reward": 0.0,  # Never got reward
                "total_reward": 0.0,  # Never got reward
                "freeze_remaining": 0,  # Never frozen
                "is_frozen": False,  # Never frozen
                "freeze_duration": 0,  # Never frozen
                "orientation": 0,  # Never rotated
                "inventory": [],  # Empty inventory that never changed
                "inventory_max": 10,  # Single value
                "color": 0,  # Never changed color
            },
            {
                # Static fields
                "id": 2,
                "type_id": 0,
                "agent_id": 1,
                "is_agent": True,
                "vision_size": 11,
                "group_id": 0,
                "is_swappable": False,
                # Time series fields (mix of single values and arrays)
                "position": [[0, [3, 3, 0]], [5, [4, 3, 0]]],  # Moved at step 5
                "action_id": [[0, 1], [10, 0]],  # Changed action at step 10
                "action_param": 0,  # Never changed param
                "action_success": [[0, False], [10, True]],  # Success changed at step 10
                "current_reward": 1.5,  # Single reward value
                "total_reward": [[0, 0.0], [10, 1.5]],  # Total changed at step 10
                "freeze_remaining": 0,  # Never frozen
                "is_frozen": False,  # Never frozen
                "freeze_duration": 0,  # Never frozen
                "orientation": 1,  # Never rotated
                "inventory": [[0, []], [20, [[0, 2], [1, 1]]]],  # Got items at step 20
                "inventory_max": 10,  # Single value
                "color": 1,  # Never changed color
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
        (lambda r: r.update({"unexpected": 123}), "Unexpected keys present"),
        (lambda r: r.update({"version": 1}), "'version' must equal 2"),
        (lambda r: r.update({"num_agents": -1}), "'num_agents' must be positive"),
        (lambda r: r.update({"map_size": [0, 5]}), "'map_size\\[0\\]' must be positive"),
        (lambda r: r.update({"file_name": "replay.txt"}), "'file_name' must be non-empty and end with '\\.json\\.z'"),
        (lambda r: r.update({"action_names": ["", "collect"]}), "'action_names' must contain non-empty strings"),
        (lambda r: r.update({"objects": [123]}), "'objects' must contain dictionaries"),
    ],
)
def test_validate_replay_schema_invalid(mutation, error_substr: str) -> None:
    """Verify that the validator can detect invalid replays."""
    replay_dict = _make_valid_replay()
    mutation(replay_dict)

    with pytest.raises(AssertionError, match=error_substr):
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
