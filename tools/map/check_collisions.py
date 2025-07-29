import argparse
import json
from collections import defaultdict


def find_hash_collisions(file_path: str):
    """
    Reads a JSON file containing map data and checks for hash collisions.
    """
    try:
        with open(file_path, "r") as f:
            maps_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}.")
        return

    if not isinstance(maps_data, list):
        print("Error: JSON data is not a list of maps.")
        return

    hashes = defaultdict(list)
    for map_item in maps_data:
        if "hash" in map_item and "map_id" in map_item:
            hashes[map_item["hash"]].append(map_item["map_id"])
        else:
            print("Warning: Found a map entry without a 'hash' or 'map_id'.")

    collisions_found = False
    for map_hash, map_ids in hashes.items():
        if len(map_ids) > 1:
            collisions_found = True
            print(f"Collision detected for hash: {map_hash}")
            print(f"  Map IDs with this hash: {map_ids}")

    if not collisions_found:
        print("No hash collisions were found.")


def main():
    """
    Main function to parse arguments and run the collision check.
    """
    parser = argparse.ArgumentParser(description="Check for hash collisions in a map data file.")
    parser.add_argument("file_path", type=str, help="Path to the JSON file containing map data.")
    args = parser.parse_args()
    find_hash_collisions(args.file_path)


if __name__ == "__main__":
    main()
