# metta/util/debug.py

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np
from omegaconf import DictConfig, OmegaConf

# Track objects we've already seen to prevent infinite recursion
_seen_objects: Set[int] = set()
_max_recursion_depth = 10  # Limit recursion depth


def save_args_for_c(
    args: Dict[str, Any], base_filename: str = "c_test_args", output_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Save Python arguments in formats suitable for C testing.

    Args:
        args: Dictionary of arguments to save. Each key-value pair will be saved.
        base_filename: Base name for the output files (without extension).
        output_dir: Directory to save files. If None, uses current directory.

    Returns:
        dict: Paths to the saved files
    """
    if output_dir is None:
        output_dir = "."

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # 1. Save the complete args as pickle for Python testing
    pickle_path = output_path / f"{base_filename}.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(args, f)
    saved_files["pickle"] = str(pickle_path)

    # 2. Save individual arguments in C-friendly formats
    for arg_name, arg_value in args.items():
        # Convert DictConfig to dict if needed
        if isinstance(arg_value, DictConfig):
            arg_value = OmegaConf.to_container(arg_value, resolve=True)
            arg_value = filter_test_data(arg_value)  # Filter unnecessary fields

        # Handle different types of arguments
        if isinstance(arg_value, np.ndarray):
            # For numpy arrays like maps
            file_path = output_path / f"{base_filename}_{arg_name}.txt"

            # Determine best format based on content
            if np.issubdtype(arg_value.dtype, np.number):
                # For numeric arrays
                np.savetxt(file_path, arg_value, fmt="%.6g", delimiter=",")
            else:
                # For string/mixed arrays (like your map)
                np.savetxt(file_path, arg_value, fmt="%s", delimiter=",")

            saved_files[arg_name] = str(file_path)
        elif isinstance(arg_value, (dict, list)) or hasattr(arg_value, "__dict__"):
            # For complex objects like configs
            try:
                file_path = output_path / f"{base_filename}_{arg_name}.json"
                with open(file_path, "w") as f:
                    json.dump(arg_value, f, indent=2, cls=CustomEncoder)
                saved_files[arg_name] = str(file_path)
            except Exception as e:
                print(f"Warning: Could not save {arg_name} as JSON: {str(e)}")
                # Fallback to simplified format
                file_path = output_path / f"{base_filename}_{arg_name}_simplified.txt"

                # Reset the seen objects set for each new argument
                global _seen_objects
                _seen_objects = set()

                simplified = _simplify_for_c(arg_value)
                with open(file_path, "w") as f:
                    for key, value in simplified.items():
                        f.write(f"{key}={value}\n")
                saved_files[f"{arg_name}_simplified"] = str(file_path)
        else:
            # For simple types
            file_path = output_path / f"{base_filename}_{arg_name}.txt"
            with open(file_path, "w") as f:
                f.write(str(arg_value))
            saved_files[arg_name] = str(file_path)

    print(f"Arguments saved for C testing in {output_dir}")
    for name, path in saved_files.items():
        print(f"  - {name}: {path}")

    return saved_files


def save_mettagrid_args(
    env_cfg: DictConfig,
    env_map: np.ndarray,
    base_filename: str = "mettagrid_test_args",
    output_dir: Optional[str] = "./test_data",
) -> Dict[str, str]:
    """
    Specialized function to save MettaGrid constructor arguments for C testing.

    Args:
        env_cfg: Environment configuration
        env_map: Environment map data
        base_filename: Base name for the output files
        output_dir: Directory to save files

    Returns:
        dict: Paths to the saved files
    """
    # Convert DictConfig to plain Python dict
    env_cfg_dict = OmegaConf.to_container(env_cfg, resolve=True)

    # Filter out unnecessary metadata
    env_cfg_dict = filter_test_data(env_cfg_dict)

    # Create args with the converted dict
    args = {"env_cfg": env_cfg_dict, "env_map": env_map}

    return save_args_for_c(args, base_filename, output_dir)


def filter_test_data(data: Any) -> Any:
    """
    Filter out unnecessary fields from test data to make it more compact.

    Args:
        data: Dictionary or other data structure to filter

    Returns:
        Filtered data with unnecessary fields removed
    """
    if isinstance(data, dict):
        # Remove metadata that's not needed for C tests
        filtered = {}
        for k, v in data.items():
            # Skip keys that start with underscore or are known metadata
            if k.startswith("_"):
                continue

            # Recursively filter nested dictionaries
            if isinstance(v, dict):
                filtered[k] = filter_test_data(v)
            elif isinstance(v, list):
                filtered[k] = [filter_test_data(item) for item in v]
            else:
                filtered[k] = v
        return filtered
    elif isinstance(data, list):
        return [filter_test_data(item) for item in data]
    else:
        return data


class CustomEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling non-serializable objects."""

    def default(self, obj):
        try:
            # Handle OmegaConf objects explicitly
            if isinstance(obj, DictConfig):
                return filter_test_data(OmegaConf.to_container(obj, resolve=True))

            # Try to convert to a dictionary
            elif hasattr(obj, "to_dict") and callable(obj.to_dict):
                return obj.to_dict()
            elif hasattr(obj, "__dict__"):
                # Filter out private attributes
                return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}

            # For numpy arrays
            elif isinstance(obj, np.ndarray):
                return obj.tolist()

            # For numpy data types
            elif np.issubdtype(type(obj), np.integer):
                return int(obj)
            elif np.issubdtype(type(obj), np.floating):
                return float(obj)

            # For sets
            elif isinstance(obj, set):
                return list(obj)

            # For other types, convert to string
            else:
                # Try to see if object is directly JSON serializable first
                try:
                    json.dumps(obj)
                    return obj
                except TypeError:
                    return str(obj)
        except Exception as e:
            return f"<{type(obj).__name__}: {str(e)}>"


def _simplify_for_c(
    obj: Any, prefix: str = "", result: Optional[Dict[str, Any]] = None, depth: int = 0
) -> Dict[str, Any]:
    """
    Recursively flatten and simplify a complex object for C usage.
    Returns a dictionary with string keys and simple value types.

    Args:
        obj: The object to simplify
        prefix: Prefix for keys in the resulting dictionary
        result: Dictionary to accumulate results (used in recursion)
        depth: Current recursion depth

    Returns:
        dict: Simplified representation with flattened keys
    """
    if result is None:
        result = {}

    # Guard against infinite recursion
    if depth > _max_recursion_depth:
        result[prefix] = "<MAX_DEPTH_EXCEEDED>"
        return result

    # Check for circular references by object id
    obj_id = id(obj)
    if obj_id in _seen_objects:
        result[prefix] = "<CIRCULAR_REFERENCE>"
        return result

    # Keep track of objects we've seen
    if isinstance(obj, (dict, list, tuple)) or hasattr(obj, "__dict__"):
        _seen_objects.add(obj_id)

    # Handle different types
    if isinstance(obj, DictConfig):
        # Convert DictConfig to dict first
        dict_obj = OmegaConf.to_container(obj, resolve=True)
        dict_obj = filter_test_data(dict_obj)
        for key, value in dict_obj.items():
            new_key = f"{prefix}.{key}" if prefix else key
            _simplify_value(value, new_key, result, depth + 1)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key.startswith("_"):  # Skip private attributes
                continue
            new_key = f"{prefix}.{key}" if prefix else key
            _simplify_value(value, new_key, result, depth + 1)
    elif hasattr(obj, "__dict__"):
        # Handle objects by processing their __dict__
        try:
            for key, value in obj.__dict__.items():
                if not key.startswith("_"):  # Skip private attributes
                    new_key = f"{prefix}.{key}" if prefix else key
                    _simplify_value(value, new_key, result, depth + 1)
        except Exception as e:
            result[prefix] = f"<ERROR: {str(e)}>"
    else:
        # For non-dictionaries, just store the value
        result[prefix] = _format_value(obj)

    return result


def _simplify_value(value: Any, key: str, result: Dict[str, Any], depth: int) -> None:
    """
    Safely process a value for simplification, handling potential recursion.

    Args:
        value: The value to process
        key: The key for this value
        result: Dictionary to store the processed value
        depth: Current recursion depth
    """
    if isinstance(value, DictConfig):
        _simplify_for_c(value, key, result, depth)
    elif isinstance(value, dict) or hasattr(value, "__dict__"):
        _simplify_for_c(value, key, result, depth)
    else:
        result[key] = _format_value(value)


def _format_value(value: Any) -> Any:
    """
    Format a value for C-friendly output.

    Args:
        value: The value to format

    Returns:
        Formatted value suitable for C
    """
    try:
        if value is None:
            return "NULL"
        elif isinstance(value, (list, tuple, set)):
            # Convert sequences to comma-separated strings
            return ",".join(str(x) for x in value)
        elif isinstance(value, (int, float, bool, str)):
            # Basic types
            return value
        elif isinstance(value, np.ndarray):
            # Numpy arrays
            if value.size <= 100:  # Only convert small arrays to avoid huge strings
                return ",".join(str(x) for x in value.flatten())
            else:
                return f"<ndarray:shape={value.shape},dtype={value.dtype}>"
        else:
            # Other objects
            return f"<{type(value).__name__}>"
    except Exception as e:
        return f"<ERROR:{str(e)}>"
