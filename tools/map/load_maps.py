import json
from typing import Any, List


def load_maps(file_path: str) -> List[Any]:
    """
    Loads a list of map objects from a JSON file.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A list of map objects.
    """
    try:
        with open(file_path, "r") as f:
            maps_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}.")
        return []

    if not isinstance(maps_data, list):
        print("Error: JSON data is not a list of maps.")
        return []

    return maps_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load maps from a JSON file.")
    parser.add_argument("file_path", type=str, help="Path to the JSON file containing map data.")
    args = parser.parse_args()

    maps = load_maps(args.file_path)
    if maps:
        print(f"Successfully loaded {len(maps)} maps.")
        print("First map object:")
        print(maps[0])
