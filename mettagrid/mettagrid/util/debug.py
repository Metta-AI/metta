import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np
from omegaconf import DictConfig, OmegaConf

# Track objects we've already seen to prevent infinite recursion
_seen_objects: Set[int] = set()
_max_recursion_depth = 10  # Limit recursion depth


def save_array_slice(
    array: np.ndarray, indices: tuple, file_path: Path, max_preview_elements: int = 60, append: bool = False
) -> None:
    """
    Save a slice of an array to a file, with a preview of values.

    Args:
        array: The NumPy array to slice
        indices: Tuple of indices to select the slice (use None for ':' in that dimension)
                 e.g., (0, None, 5) would represent array[0, :, 5]
        file_path: Path to the file to save the slice information
        max_preview_elements: Maximum number of elements to include in the preview
        append: Whether to append to the file (True) or overwrite it (False)
    """
    # Convert None indices to slice(None) for proper indexing
    idx = tuple(slice(None) if i is None else i for i in indices)

    # Check that the indices are valid for the array
    if len(idx) > array.ndim:
        raise ValueError(f"Too many indices ({len(idx)}) for array with {array.ndim} dimensions")

    # Create a string representation of the indices for display
    idx_str = []
    for _i, index in enumerate(idx):
        if index == slice(None):
            idx_str.append(":")
        else:
            idx_str.append(str(index))

    # If we have fewer indices than dimensions, add ":" for the remaining dims
    idx_str.extend([":" for _ in range(array.ndim - len(idx))])

    # Get the slice
    try:
        array_slice = array[idx]

        # Open the file in append or write mode
        mode = "a" if append else "w"
        with open(file_path, mode) as f:
            # If we're appending, add a section header
            if append:
                f.write("\n# Slice preview\n")

            # Write the slice information
            f.write(f"slice_indices: [{', '.join(idx_str)}]\n")
            f.write(f"slice_shape: {array_slice.shape}\n")

            # Check if the slice is a scalar (0-dim array)
            if array_slice.ndim == 0:
                f.write(f"slice_value: {array_slice.item()}\n")
                return

            # Format the slice preview based on its size
            f.write("slice_preview: [")

            # Handle empty array
            if array_slice.size == 0:
                f.write("]\n")
                return

            # Flatten the array for preview if it's multi-dimensional
            flat_slice = array_slice.flatten() if array_slice.ndim > 1 else array_slice

            # Preview logic
            if flat_slice.size <= max_preview_elements:
                # If small enough, show the entire slice
                slice_str = ", ".join(str(val) for val in flat_slice)
                f.write(f"{slice_str}]\n")
            else:
                # Show first and last parts with ellipsis in the middle
                half_elements = max_preview_elements // 2
                first_slice = ", ".join(str(val) for val in flat_slice[:half_elements])
                last_slice = ", ".join(str(val) for val in flat_slice[-half_elements:])
                f.write(f"{first_slice}, ... , {last_slice}]\n")

    except Exception as e:
        # Handle errors gracefully
        with open(file_path, mode) as f:
            if append:
                f.write("\n# Slice preview\n")
            f.write(f"Error getting slice at {idx_str}: {str(e)}\n")


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
    import datetime

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

    # Get current date and time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 2. Save individual arguments in C-friendly formats
    for arg_name, arg_value in args.items():
        # Convert DictConfig to dict if needed
        if isinstance(arg_value, DictConfig):
            arg_value = OmegaConf.to_container(arg_value, resolve=True)
            arg_value = filter_test_data(arg_value)  # Filter unnecessary fields

        # Handle different types of arguments
        if isinstance(arg_value, np.ndarray):
            # For numpy arrays
            file_path = output_path / f"{base_filename}_{arg_name}.npz"

            # Save arrays of any dimension using numpy's compressed format
            np.savez_compressed(file_path, array=arg_value)
            saved_files[arg_name] = str(file_path)

            # Save shape, dtype and preview information in a separate text file for easier parsing in C
            info_path = output_path / f"{base_filename}_{arg_name}_info.txt"
            with open(info_path, "w") as f:
                f.write(f"# Generated: {timestamp}\n")
                f.write(f"shape: {arg_value.shape}\n")
                f.write(f"dtype: {arg_value.dtype}\n")
                f.write(f"ndim: {arg_value.ndim}\n")
                f.write(f"size: {arg_value.size}\n")

                # Add min/max for numeric arrays
                if arg_value.size > 0 and np.issubdtype(arg_value.dtype, np.number):
                    f.write(f"min: {np.min(arg_value)}\n")
                    f.write(f"max: {np.max(arg_value)}\n")
                else:
                    f.write("min: N/A\n")
                    f.write("max: N/A\n")

                # Create a preview of the flattened array values
                f.write("preview: [")
                if arg_value.size == 0:
                    f.write("]\n")
                else:
                    # Flatten the array for preview
                    flat = arg_value.flatten()
                    if flat.size <= 60:
                        # If the array is small, show all values
                        values_str = ", ".join(str(val) for val in flat)
                        f.write(f"{values_str}]\n")
                    else:
                        # Show first 30 and last 30 values with ellipsis in the middle
                        first_values = ", ".join(str(val) for val in flat[:30])
                        last_values = ", ".join(str(val) for val in flat[-30:])
                        f.write(f"{first_values}, ... , {last_values}]\n")

            # Add slices for multi-dimensional arrays
            if arg_value.ndim > 1:
                # For multi-dimensional arrays, save slices
                # First, save the slice of the last dimension at index 0 for all other dimensions
                zeros_idx = tuple(0 for _ in range(arg_value.ndim - 1))
                save_array_slice(arg_value, zeros_idx, info_path, append=True)

                # If it's a 3D+ array, add another slice from the middle of the array if possible
                if arg_value.ndim >= 3:
                    try:
                        # Try to get indices from the middle of each dimension (except the last)
                        mid_idx = tuple(min(5, max(0, s // 2)) for s in arg_value.shape[:-1])
                        save_array_slice(arg_value, mid_idx, info_path, append=True)
                    except Exception as e:
                        print(f"Warning: Could not save middle slice for {arg_name}: {str(e)}")

            saved_files[f"{arg_name}_info"] = str(info_path)

            # For 1D and 2D arrays, also save in text format for human readability
            if arg_value.ndim <= 2:
                txt_path = output_path / f"{base_filename}_{arg_name}.txt"
                try:
                    if np.issubdtype(arg_value.dtype, np.number):
                        np.savetxt(txt_path, arg_value, fmt="%.6g", delimiter=",")
                    else:
                        np.savetxt(txt_path, arg_value, fmt="%s", delimiter=",")
                    saved_files[f"{arg_name}_txt"] = str(txt_path)
                except Exception as e:
                    print(f"Warning: Could not save {arg_name} as text: {str(e)}")
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
                    f.write(f"# Generated: {timestamp}\n")
                    for key, value in simplified.items():
                        f.write(f"{key}={value}\n")
                saved_files[f"{arg_name}_simplified"] = str(file_path)
        else:
            # For simple types
            file_path = output_path / f"{base_filename}_{arg_name}.txt"
            with open(file_path, "w") as f:
                f.write(f"# Generated: {timestamp}\n")
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

            # For numpy scalar types
            elif isinstance(obj, np.number):
                return obj.item()  # Converts any numpy number to its Python equivalent

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


def save_step_results(
    observations: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    truncations: np.ndarray,
    infos: Dict[str, Any],
    base_filename: str = "step_results",
    output_dir: Optional[str] = "./test_data",
    step_count: Optional[int] = None,
) -> Dict[str, str]:
    """
    Save the results from the step function for testing.

    Args:
        observations: Observation array from step function
        rewards: Rewards array from step function
        terminals: Terminal states array from step function
        truncations: Truncation states array from step function
        infos: Additional information dictionary from step function
        base_filename: Base name for the output files
        output_dir: Directory to save files
        step_count: Optional step counter for auto-incrementing filenames

    Returns:
        dict: Paths to the saved files
    """

    # Create args with all the step results
    args = {
        "observations": observations,
        "rewards": rewards,
        "terminals": terminals,
        "truncations": truncations,
        "infos": infos,
    }

    # Add numeric suffix to filename if step_count is provided
    filename = base_filename
    if step_count is not None:
        filename = f"{base_filename}_{step_count:03d}"

    return save_args_for_c(args, filename, output_dir)
