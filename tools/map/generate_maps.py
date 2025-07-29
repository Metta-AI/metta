import argparse
import hashlib
import json
import logging
import os

import numpy as np
from tqdm import tqdm

from metta.map.mapgen import MapGen
from metta.map.utils.ascii_grid import grid_to_lines
from tools.map.scene_params import SCENE_GENERATORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)


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
    except Exception as e:
        logger.warning(f"Failed to create map generator for scene '{scene_name}': {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random maps.")
    parser.add_argument("--num-maps", type=int, default=10)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--scene", type=str, default="random")
    parser.add_argument("--output-dir", type=str, default="pregenerated_maps")
    args = parser.parse_args()

    rng = np.random.default_rng()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "maps.json")
    partial_save_path = os.path.join(args.output_dir, "maps.json.partial")

    maps_data = []
    map_id = 0
    with tqdm(total=args.num_maps, desc="Generating maps") as pbar:
        while len(maps_data) < args.num_maps:
            map_gen = generate_map(rng, args.scene, args.width, args.height)
            if not map_gen:
                continue

            try:
                level = map_gen.build()
            except Exception as e:
                # Log the error and continue to the next map
                logger.warning(f"Failed to build map: {e}")
                continue

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

            # Periodic save to prevent data loss on long runs
            if map_id > 0 and map_id % 1000 == 0:
                # Atomically write to a single partial file to prevent corruption
                temp_save_path = partial_save_path + ".tmp"
                with open(temp_save_path, "w") as f:
                    json.dump(maps_data, f, indent=2, cls=NpEncoder)
                os.rename(temp_save_path, partial_save_path)
                logger.info(f"Periodic save: {len(maps_data)} maps to {partial_save_path}")

    with open(output_path, "w") as f:
        json.dump(maps_data, f, indent=2, cls=NpEncoder)

    # Clean up the partial save file upon successful completion
    if os.path.exists(partial_save_path):
        os.remove(partial_save_path)

    print(f"Saved {len(maps_data)} maps to {output_path}")
