import argparse
import hashlib
import json
import os

import numpy as np

from metta.map.mapgen import MapGen
from metta.map.utils.ascii_grid import grid_to_lines


def generate_random_scene_params():
    """Generates random parameters for the Random scene."""
    objects = {
        "altar": np.random.randint(1, 10),
        "key": np.random.randint(0, 3),
        "door": np.random.randint(0, 3),
        "chest": np.random.randint(0, 2),
    }
    # only include objects with count > 0
    objects = {k: v for k, v in objects.items() if v > 0}
    return {
        "agents": np.random.randint(1, 5),
        "objects": objects,
    }


def generate_maze_scene_params():
    """Generates random parameters for the Maze scene."""
    return {
        "algorithm": np.random.choice(["kruskal", "dfs"]),
        "room_size": np.random.randint(2, 5),
        "wall_size": 1,
    }


SCENE_GENERATORS = {
    "random": {
        "params_fn": generate_random_scene_params,
        "class_path": "metta.map.scenes.random.Random",
    },
    "maze": {
        "params_fn": generate_maze_scene_params,
        "class_path": "metta.map.scenes.maze.Maze",
    },
}


def create_map(width: int, height: int, scene_type: str) -> tuple[np.ndarray, dict]:
    """
    Generates a single map using MapGen with a specified scene type.
    """
    generator_info = SCENE_GENERATORS[scene_type]
    params = generator_info["params_fn"]()

    root_config = {
        "type": generator_info["class_path"],
        "params": params,
    }

    map_generator = MapGen(
        width=width,
        height=height,
        root=root_config,
    )
    level = map_generator.build()
    return level.grid, root_config


def main():
    """
    Main function to generate and save maps.
    """
    parser = argparse.ArgumentParser(description="Generate and save random maps.")
    parser.add_argument("--num_maps", type=int, default=10, help="Number of maps to generate.")
    parser.add_argument("--output_dir", type=str, default="pregenerated_maps", help="Directory to save the maps.")
    parser.add_argument("--width", type=int, default=32, help="Width of the maps.")
    parser.add_argument("--height", type=int, default=32, help="Height of the maps.")
    parser.add_argument("--save", action="store_true", help="Save the generated maps to files.")
    parser.add_argument(
        "--scene-type",
        type=str,
        choices=list(SCENE_GENERATORS.keys()),
        default=None,
        help="Only generate maps of a specific scene type.",
    )

    args = parser.parse_args()

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)

    scene_types = list(SCENE_GENERATORS.keys())

    all_maps_data = []
    output_path = os.path.join(args.output_dir, "maps.json")
    if args.save and os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                all_maps_data = json.load(f)
            if not isinstance(all_maps_data, list):
                print(f"Warning: Existing file {output_path} is not a JSON list. Starting fresh.")
                all_maps_data = []
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {output_path}. Starting fresh.")
            all_maps_data = []

    start_map_id = all_maps_data[-1]["map_id"] if all_maps_data else 0

    for i in range(args.num_maps):
        scene_type = args.scene_type if args.scene_type else np.random.choice(scene_types)
        print(f"Generating map {i + 1}/{args.num_maps} using '{scene_type}' scene...")

        grid, config = create_map(args.width, args.height, scene_type)

        print(f"Generated with config: {config}")

        map_lines = grid_to_lines(grid, border=True)
        map_ascii = "\n".join(map_lines)

        print(map_ascii)

        map_hash = hashlib.sha256(map_ascii.encode()).hexdigest()
        map_id = start_map_id + i + 1
        map_data = {
            "map_id": map_id,
            "hash": map_hash,
            "map": map_lines,
            "config": config,
        }
        all_maps_data.append(map_data)

    if args.save:
        with open(output_path, "w") as f:
            json.dump(all_maps_data, f, indent=2)
        print(f"Saved all maps to {output_path}")


if __name__ == "__main__":
    main()
