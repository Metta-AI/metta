#!/usr/bin/env python3
"""
Replay format validator for MettaScope replay files.
Validates replay files against the specification version 2.
"""

import argparse
import json
import sys
import zlib
from typing import Any, Dict, List

# Global lists to store errors and warnings
errors = []
warnings = []
verbose = False


def log(message: str):
    """Log a message if verbose mode is enabled."""
    if verbose:
        print(f"[INFO] {message}")


def add_error(error: str):
    """Add an error message."""
    errors.append(error)
    if verbose:
        print(f"[ERROR] {error}")


def add_warning(warning: str):
    """Add a warning message."""
    warnings.append(warning)
    if verbose:
        print(f"[WARNING] {warning}")


def load_replay_file(file_path: str) -> Dict[str, Any]:
    """Load and parse a replay file (compressed or plain JSON)."""
    try:
        if file_path.endswith(".z"):
            log(f"Loading compressed file: {file_path}")
            with open(file_path, "rb") as file:
                compressed_data = file.read()
            log("Decompressing data...")
            decompressed_data = zlib.decompress(compressed_data)
            log("Parsing JSON...")
            return json.loads(decompressed_data)
        else:
            log(f"Loading plain JSON file: {file_path}")
            with open(file_path, "r") as file:
                return json.load(file)
    except zlib.error as e:
        add_error(f"Failed to decompress file: {e}")
        return None
    except json.JSONDecodeError as e:
        add_error(f"Invalid JSON format: {e}")
        return None
    except FileNotFoundError as e:
        add_error(f"File not found: {e}")
        return None
    except Exception as e:
        add_error(f"Unexpected error: {e}")
        return None


def validate_version(data: Dict[str, Any]) -> bool:
    """Validate the version field."""
    if "version" not in data:
        add_error("Missing required field: 'version'")
        return False

    version = data["version"]
    if version != 2:
        add_error(f"Unsupported version: {version}. This validator supports version 2 only.")
        return False
    return True


def validate_constants(data: Dict[str, Any]):
    """Validate required constants."""
    required_constants = ["num_agents", "max_steps", "map_size"]

    for field in required_constants:
        if field not in data:
            add_error(f"Missing required constant: '{field}'")
        else:
            value = data[field]
            if field == "num_agents":
                if not isinstance(value, int) or value < 0:
                    add_error(f"'{field}' must be a non-negative integer, got: {value}")
            elif field == "max_steps":
                if not isinstance(value, int) or value <= 0:
                    add_error(f"'{field}' must be a positive integer, got: {value}")
            elif field == "map_size":
                if not isinstance(value, list) or len(value) != 2:
                    add_error(f"'{field}' must be a list of 2 integers, got: {value}")
                elif not all(isinstance(x, int) and x > 0 for x in value):
                    add_error(f"'{field}' values must be positive integers, got: {value}")


def validate_mapping_arrays(data: Dict[str, Any]):
    """Validate the mapping arrays."""
    mapping_arrays = ["type_names", "action_names", "item_names", "group_names"]

    # Arrays that cannot contain dots
    no_dot_arrays = ["type_names", "action_names", "item_names"]

    for array_name in mapping_arrays:
        if array_name not in data:
            add_error(f"Missing required mapping array: '{array_name}'")
        else:
            array = data[array_name]
            if not isinstance(array, list):
                add_error(f"'{array_name}' must be a list, got: {type(array).__name__}")
            elif not all(isinstance(x, str) for x in array):
                add_error(f"'{array_name}' must contain only strings")
            elif len(array) == 0:
                add_warning(f"'{array_name}' is empty")
            else:
                # Check for dots in specific arrays
                if array_name in no_dot_arrays:
                    for i, name in enumerate(array):
                        if "." in name:
                            add_error(f"'{array_name}[{i}]' contains invalid character '.': '{name}'")


def is_time_series(value: Any) -> bool:
    """Check if a value is a time series."""
    if not isinstance(value, list) or len(value) == 0:
        return False
    first_item = value[0]
    return isinstance(first_item, list) and len(first_item) == 2 and isinstance(first_item[0], (int, float))


def validate_time_series_value(
    field: str,
    value: Any,
    index: int,
    obj_prefix: str,
    map_size: List[int],
    action_names: List[str],
    item_names: List[str],
    group_names: List[str],
):
    """Validate a value in a time series based on the field type."""
    if field == "position":
        if not isinstance(value, list) or len(value) != 2:
            add_error(f"{obj_prefix}: Position value at index {index} must be [x, y]")
        elif not all(isinstance(x, (int, float)) for x in value):
            add_error(f"{obj_prefix}: Position coordinates must be numbers")
        else:
            x, y = value
            if x < 0 or x >= map_size[0]:
                add_error(
                    f"{obj_prefix}: Position x={x} at time series index {index} is out of map bounds (0-{map_size[0] - 1})"
                )
            if y < 0 or y >= map_size[1]:
                add_error(
                    f"{obj_prefix}: Position y={y} at time series index {index} is out of map bounds (0-{map_size[1] - 1})"
                )

    elif field == "inventory":
        if not isinstance(value, list):
            add_error(f"{obj_prefix}: Inventory value at index {index} must be a list")
        else:
            for item_id in value:
                if not isinstance(item_id, int):
                    add_error(f"{obj_prefix}: Inventory item IDs must be integers")
                elif item_id < 0 or item_id >= len(item_names):
                    add_error(f"{obj_prefix}: Inventory item ID {item_id} is out of range")

    elif field == "action_id":
        if not isinstance(value, int):
            add_error(f"{obj_prefix}: action_id at index {index} must be an integer")
        elif value < 0 or value >= len(action_names):
            add_error(f"{obj_prefix}: action_id {value} at index {index} is out of range")

    elif field == "group_id":
        if not isinstance(value, int):
            add_error(f"{obj_prefix}: group_id at index {index} must be an integer")
        elif value < 0 or value >= len(group_names):
            add_error(f"{obj_prefix}: group_id {value} at index {index} is out of range")

    elif field in [
        "rotation",
        "layer",
        "action_parameter",
        "frozen_progress",
        "production_progress",
        "cooldown_progress",
    ]:
        if not isinstance(value, (int, float)):
            add_error(f"{obj_prefix}: '{field}' at index {index} must be a number")

    elif field in ["action_success", "frozen"]:
        if not isinstance(value, bool):
            add_error(f"{obj_prefix}: '{field}' at index {index} must be a boolean")

    elif field in ["total_reward", "current_reward"]:
        if not isinstance(value, (int, float)):
            add_error(f"{obj_prefix}: '{field}' at index {index} must be a number")


def validate_time_series(
    field: str,
    series: List[List[Any]],
    obj_prefix: str,
    map_size: List[int],
    max_steps: int,
    action_names: List[str],
    item_names: List[str],
    group_names: List[str],
):
    """Validate a time series field."""
    if len(series) == 0:
        add_error(f"{obj_prefix}: Time series for '{field}' is empty")
        return

    prev_step = -1
    for i, entry in enumerate(series):
        if not isinstance(entry, list) or len(entry) != 2:
            add_error(f"{obj_prefix}: Time series entry {i} for '{field}' must be [step, value]")
            continue

        step, value = entry

        # Validate step
        if not isinstance(step, (int, float)):
            add_error(f"{obj_prefix}: Step in time series entry {i} for '{field}' must be a number")
        elif step < 0:
            add_error(f"{obj_prefix}: Step in time series entry {i} for '{field}' must be non-negative")
        elif step > max_steps:
            add_error(
                f"{obj_prefix}: Step {step} in time series entry {i} for '{field}' exceeds max_steps ({max_steps})"
            )
        elif step <= prev_step:
            add_error(f"{obj_prefix}: Steps in time series for '{field}' must be in ascending order")

        prev_step = step

        # Validate value based on field type
        validate_time_series_value(field, value, i, obj_prefix, map_size, action_names, item_names, group_names)


def validate_constant(
    field: str,
    value: Any,
    obj_prefix: str,
    type_names: List[str],
    action_names: List[str],
    item_names: List[str],
    group_names: List[str],
    map_size: List[int],
):
    """Validate a constant field value."""
    if field in [
        "type_id",
        "id",
        "agent_id",
        "inventory_max",
        "recipe_max",
        "frozen_time",
        "production_time",
        "cooldown_time",
    ]:
        if not isinstance(value, int):
            add_error(f"{obj_prefix}: '{field}' must be an integer")
        elif value < 0:
            add_error(f"{obj_prefix}: '{field}' must be non-negative")

    elif field in ["recipe_input", "recipe_output"]:
        if not isinstance(value, list):
            add_error(f"{obj_prefix}: '{field}' must be a list")
        else:
            for item_id in value:
                if not isinstance(item_id, int):
                    add_error(f"{obj_prefix}: Item IDs in '{field}' must be integers")
                elif item_id < 0 or item_id >= len(item_names):
                    add_error(f"{obj_prefix}: Item ID {item_id} in '{field}' is out of range")

    elif field == "position":
        # Validate constant position
        if not isinstance(value, list) or len(value) != 2:
            add_error(f"{obj_prefix}: Position must be [x, y]")
        else:
            x, y = value
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                add_error(f"{obj_prefix}: Position coordinates must be numbers")
            else:
                if x < 0 or x >= map_size[0]:
                    add_error(f"{obj_prefix}: Position x={x} is out of map bounds (0-{map_size[0] - 1})")
                if y < 0 or y >= map_size[1]:
                    add_error(f"{obj_prefix}: Position y={y} is out of map bounds (0-{map_size[1] - 1})")


def get_expected_object_fields() -> set:
    """Get the set of all expected object fields."""
    return {
        # Common fields
        "type_id",
        "id",
        "position",
        "layer",
        "rotation",
        "inventory",
        "inventory_max",
        # Agent-specific fields
        "agent_id",
        "action_id",
        "action_parameter",
        "action_success",
        "total_reward",
        "current_reward",
        "frozen",
        "frozen_progress",
        "frozen_time",
        "group_id",
        # Object-specific fields
        "recipe_input",
        "recipe_output",
        "recipe_max",
        "production_progress",
        "production_time",
        "cooldown_progress",
        "cooldown_time",
        # Allow underscore-prefixed fields for metadata
        "_comment",
    }


def validate_field(
    field: str,
    value: Any,
    obj_prefix: str,
    type_names: List[str],
    action_names: List[str],
    item_names: List[str],
    group_names: List[str],
    map_size: List[int],
    max_steps: int,
):
    """Validate a single field value."""
    # Skip validation for underscore-prefixed fields (metadata)
    if field.startswith("_"):
        return

    if is_time_series(value):
        validate_time_series(field, value, obj_prefix, map_size, max_steps, action_names, item_names, group_names)
    else:
        validate_constant(field, value, obj_prefix, type_names, action_names, item_names, group_names, map_size)


def validate_object(
    obj: Dict[str, Any],
    index: int,
    type_names: List[str],
    action_names: List[str],
    item_names: List[str],
    group_names: List[str],
    map_size: List[int],
    max_steps: int,
    agent_ids: set,
    object_ids: set,
):
    """Validate a single object."""
    obj_prefix = f"Object at index {index}"

    # Check required fields
    if "type_id" not in obj:
        add_error(f"{obj_prefix}: Missing required field 'type_id'")
        return

    if "id" not in obj:
        add_error(f"{obj_prefix}: Missing required field 'id'")
        return

    # Validate type_id
    type_id = obj["type_id"]
    if not isinstance(type_id, int):
        add_error(f"{obj_prefix}: 'type_id' must be an integer, got: {type(type_id).__name__}")
    elif type_id < 0 or type_id >= len(type_names):
        add_error(f"{obj_prefix}: 'type_id' {type_id} is out of range (0-{len(type_names) - 1})")

    # Validate id
    obj_id = obj["id"]
    if not isinstance(obj_id, int):
        add_error(f"{obj_prefix}: 'id' must be an integer, got: {type(obj_id).__name__}")
    else:
        if obj_id in object_ids:
            add_error(f"{obj_prefix}: Duplicate object id: {obj_id}")
        object_ids.add(obj_id)

    # Check if this is an agent
    if "agent_id" in obj:
        agent_id = obj["agent_id"]
        if not isinstance(agent_id, int):
            add_error(f"{obj_prefix}: 'agent_id' must be an integer, got: {type(agent_id).__name__}")
        else:
            if agent_id in agent_ids:
                add_error(f"{obj_prefix}: Duplicate agent_id: {agent_id}")
            agent_ids.add(agent_id)

    # Check for unexpected fields
    expected_fields = get_expected_object_fields()
    for field in obj:
        if field not in expected_fields and not field.startswith("_"):
            add_warning(f"{obj_prefix}: Unexpected field '{field}'")

    # Validate fields based on their type
    for field, value in obj.items():
        validate_field(field, value, obj_prefix, type_names, action_names, item_names, group_names, map_size, max_steps)


def validate_objects(data: Dict[str, Any]):
    """Validate the objects array."""
    if "objects" not in data:
        add_error("Missing required field: 'objects'")
        return

    objects = data["objects"]
    if not isinstance(objects, list):
        add_error(f"'objects' must be a list, got: {type(objects).__name__}")
        return

    # Get mapping arrays for validation
    type_names = data.get("type_names", [])
    action_names = data.get("action_names", [])
    item_names = data.get("item_names", [])
    group_names = data.get("group_names", [])
    map_size = data.get("map_size", [0, 0])
    max_steps = data.get("max_steps", 0)

    # Track agent IDs
    agent_ids = set()
    object_ids = set()

    for i, obj in enumerate(objects):
        if not isinstance(obj, dict):
            add_error(f"Object at index {i} must be a dictionary, got: {type(obj).__name__}")
            continue

        # Validate object
        validate_object(
            obj, i, type_names, action_names, item_names, group_names, map_size, max_steps, agent_ids, object_ids
        )

    # Validate num_agents matches actual agent count
    if "num_agents" in data and len(agent_ids) != data["num_agents"]:
        add_error(f"num_agents ({data['num_agents']}) doesn't match actual agent count ({len(agent_ids)})")


def validate_reward_sharing_matrix(data: Dict[str, Any]):
    """Validate the reward sharing matrix."""
    if "reward_sharing_matrix" not in data:
        add_warning("Missing optional field: 'reward_sharing_matrix'")
        return

    matrix = data["reward_sharing_matrix"]
    num_agents = data.get("num_agents", 0)

    if not isinstance(matrix, list):
        add_error("'reward_sharing_matrix' must be a list of lists")
        return

    if len(matrix) != num_agents:
        add_error(f"'reward_sharing_matrix' rows ({len(matrix)}) must match num_agents ({num_agents})")

    for i, row in enumerate(matrix):
        if not isinstance(row, list):
            add_error(f"Row {i} in 'reward_sharing_matrix' must be a list")
            continue

        if len(row) != num_agents:
            add_error(f"Row {i} in 'reward_sharing_matrix' has {len(row)} columns, expected {num_agents}")

        for j, value in enumerate(row):
            if not isinstance(value, (int, float)):
                add_error(f"Value at [{i}][{j}] in 'reward_sharing_matrix' must be a number")
            elif value < 0 or value > 1:
                add_error(f"Value at [{i}][{j}] in 'reward_sharing_matrix' must be between 0 and 1")

            # Validate diagonal is 0 (agents don't share with themselves)
            if i == j and value != 0:
                add_warning(f"Diagonal value at [{i}][{j}] in 'reward_sharing_matrix' should be 0")


def validate_replay(data: Dict[str, Any]):
    """Validate the replay data structure."""
    if not validate_version(data):
        return

    validate_constants(data)
    validate_mapping_arrays(data)
    validate_objects(data)
    validate_reward_sharing_matrix(data)

    # Check for extra fields at the root level
    expected_root_fields = {
        "version",
        "num_agents",
        "max_steps",
        "map_size",
        "type_names",
        "action_names",
        "item_names",
        "group_names",
        "objects",
        "reward_sharing_matrix",
        "_comment",
    }
    for field in data:
        if field not in expected_root_fields:
            add_warning(f"Unexpected field at root level: '{field}'")


def validate_file(file_path: str) -> bool:
    """Validate a replay file. Returns True if valid, False otherwise."""
    global errors, warnings
    errors = []
    warnings = []

    log("Validating replay data...")
    data = load_replay_file(file_path)
    if data is None:
        return False

    validate_replay(data)
    return len(errors) == 0


def main():
    """Main entry point for the validator."""
    global verbose

    parser = argparse.ArgumentParser(description="Validate MettaScope replay files against the specification.")
    parser.add_argument("file", help="Path to the replay file to validate (.json.z or .json)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    verbose = args.verbose

    # Validate the file
    is_valid = validate_file(args.file)

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    if len(errors) > 0:
        print(f"\nERRORS ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")

    if len(warnings) > 0:
        print(f"\nWARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")

    if is_valid:
        print("\n✓ Validation PASSED")
        return 0
    else:
        print("\n✗ Validation FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
