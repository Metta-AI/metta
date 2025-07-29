import argparse
import hashlib
import json
import os

import numpy as np
from tqdm import tqdm

from metta.map.mapgen import MapGen
from metta.map.utils.ascii_grid import grid_to_lines
from tools.map.scene_params import SCENE_GENERATORS


def generate_map(rng, scene_name, width, height):
    try:
        if scene_name == "random":
            scene_name = rng.choice(list(SCENE_GENERATORS.keys()))
        scene_generator = SCENE_GENERATORS[scene_name]
        params = scene_generator["params_fn"](rng)
        map_config = {
            "root": {
                "type": scene_generator["class_path"],
                "params": params,
            },
            "width": width,
            "height": height,
        }
        return MapGen(**map_config)
    except:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random maps.")
    parser.add_argument("--num_maps", type=int, default=10)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--scene", type=str, default="random")
    parser.add_argument("--output_dir", type=str, default="pregenerated_maps")
    args = parser.parse_args()

    rng = np.random.default_rng()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "maps.json")

    maps_data = []
    map_id = 0
    with tqdm(total=args.num_maps, desc="Generating maps") as pbar:
        while len(maps_data) < args.num_maps:
            map_gen = generate_map(rng, args.scene, args.width, args.height)
            if not map_gen:
                continue
            level = map_gen.build()
            map_lines = grid_to_lines(level.grid, border=True)
            map_hash = hashlib.sha256("\n".join(map_lines).encode()).hexdigest()
            maps_data.append(
                {
                    "map_id": map_id,
                    "hash": map_hash,
                    "map": map_lines,
                    "config": map_gen.root,
                }
            )
            map_id += 1
            pbar.update(1)

    with open(output_path, "w") as f:
        json.dump(maps_data, f, indent=2)
    print(f"Saved {len(maps_data)} maps to {output_path}")
