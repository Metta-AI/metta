import argparse
import os

import numpy as np

from metta.map.mapgen import MapGen
from metta.map.utils.ascii_grid import print_grid


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

    args = parser.parse_args()

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)

    scene_types = list(SCENE_GENERATORS.keys())

    for i in range(args.num_maps):
        scene_type = np.random.choice(scene_types)
        print(f"Generating map {i + 1}/{args.num_maps} using '{scene_type}' scene...")

        grid, config = create_map(args.width, args.height, scene_type)

        print(f"Generated with config: {config}")

        print_grid(grid)

        if args.save:
            output_path = os.path.join(args.output_dir, f"map_{i}_{scene_type}.npy")
            np.save(output_path, grid)
            print(f"Saved map to {output_path}")


if __name__ == "__main__":
    main()
