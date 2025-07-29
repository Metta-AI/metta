from typing import Any, Dict

# Parameter generation functions for each scene


def generate_random_scene_params(rng) -> dict:
    objects = {
        "wall": rng.integers(0, 10),
        "floor": rng.integers(0, 10),
    }
    agents = rng.integers(1, 5)
    return {"objects": objects, "agents": agents}


def generate_maze_scene_params(rng) -> dict:
    return {
        "algorithm": rng.choice(["kruskal", "dfs"]),
        "room_size": ("uniform", 1, 3),
        "wall_size": ("uniform", 1, 2),
    }


def generate_bsp_scene_params(rng) -> dict:
    min_ratio = rng.uniform(0.1, 0.5)
    return {
        "rooms": rng.integers(2, 10),
        "min_room_size": rng.integers(3, 6),
        "min_room_size_ratio": min_ratio,
        "max_room_size_ratio": rng.uniform(min_ratio + 0.1, 0.9),
        "skip_corridors": rng.choice([True, False]),
    }


_PATTERNS = [
    """
##########
#........#
#........#
#..####..#
#..#..#..#
#..#..#..#
#..####..#
#........#
#........#
##########
""",
    """
..#..
.###.
#####
.###.
..#..
""",
    "....\n.##.\n.##.\n....",
    "#####\n#...#\n#...#\n#####",
]


def generate_convchain_scene_params(rng) -> dict:
    return {
        "pattern": rng.choice(_PATTERNS),
        "pattern_size": rng.integers(2, 4),
        "iterations": rng.integers(1, 3),
        "temperature": rng.uniform(0.5, 2.0),
        "periodic_input": rng.choice([True, False]),
        "symmetry": rng.choice(["all", "horizontal", "none"]),
    }


def generate_inline_ascii_scene_params(rng) -> dict:
    return {
        "data": rng.choice(_PATTERNS),
        "row": rng.integers(0, 5),
        "column": rng.integers(0, 5),
    }


def generate_layout_scene_params(rng) -> dict:
    num_areas = rng.integers(1, 4)
    areas = []
    for i in range(num_areas):
        area = {
            "width": rng.integers(5, 15),
            "height": rng.integers(5, 15),
            "placement": "center",
            "tag": f"area_{i}",
        }
        areas.append(area)
    return {"areas": areas}


def generate_make_connected_scene_params(rng) -> dict:
    return {}


def generate_mean_distance_scene_params(rng) -> dict:
    return {
        "mean_distance": rng.uniform(1.0, 10.0),
        "objects": {
            "altar": rng.integers(1, 5),
            "key": rng.integers(0, 2),
        },
    }


def generate_multi_left_and_right_scene_params(rng) -> dict:
    return {
        "rows": rng.integers(1, 3),
        "columns": rng.integers(1, 3),
        "altar_ratio": rng.uniform(0.5, 1.0),
        "total_altars": rng.integers(1, 10),
    }


def generate_nop_scene_params(rng) -> dict:
    return {}


def generate_radial_maze_scene_params(rng) -> dict:
    return {
        "arms": rng.integers(4, 12),
        "arm_width": rng.integers(1, 5),
        "arm_length": rng.integers(10, 20) if rng.choice([True, False]) else None,
    }


def generate_random_objects_scene_params(rng) -> dict:
    return {
        "object_ranges": {
            "wall": ("uniform", 0.0, 0.2),
            "empty": ("uniform", 0.0, 0.1),
        }
    }


def generate_remove_agents_scene_params(rng) -> dict:
    return {}


def generate_room_grid_scene_params(rng) -> dict:
    if rng.choice([True, False]):
        return {
            "rows": rng.integers(1, 5),
            "columns": rng.integers(1, 5),
            "border_width": rng.integers(1, 5),
            "border_object": "wall",
        }
    else:
        # generate a layout
        rows = rng.integers(1, 4)
        cols = rng.integers(1, 4)
        layout = [[f"tag_{r}_{c}" for c in range(cols)] for r in range(rows)]
        return {
            "layout": layout,
            "border_width": rng.integers(1, 5),
            "border_object": "wall",
        }


def generate_wfc_scene_params(rng) -> dict:
    return {
        "pattern": rng.choice(_PATTERNS),
        "pattern_size": rng.integers(2, 4),
        "next_node_heuristic": rng.choice(["scanline", "mrv", "entropy"]),
        "periodic_input": rng.choice([True, False]),
        "symmetry": rng.choice(["all", "x", "y", "xy", "none"]),
        "attempts": 10,
    }


SCENE_GENERATORS: Dict[str, Dict[str, Any]] = {
    "maze": {
        "params_fn": generate_maze_scene_params,
        "class_path": "metta.map.scenes.maze.Maze",
    },
    "bsp": {
        "params_fn": generate_bsp_scene_params,
        "class_path": "metta.map.scenes.bsp.BSP",
    },
    "convchain": {
        "params_fn": generate_convchain_scene_params,
        "class_path": "metta.map.scenes.convchain.ConvChain",
    },
    "inline_ascii": {
        "params_fn": generate_inline_ascii_scene_params,
        "class_path": "metta.map.scenes.inline_ascii.InlineAscii",
    },
    "layout": {
        "params_fn": generate_layout_scene_params,
        "class_path": "metta.map.scenes.layout.Layout",
    },
    "mean_distance": {
        "params_fn": generate_mean_distance_scene_params,
        "class_path": "metta.map.scenes.mean_distance.MeanDistance",
    },
    "multi_left_and_right": {
        "params_fn": generate_multi_left_and_right_scene_params,
        "class_path": "metta.map.scenes.multi_left_and_right.MultiLeftAndRight",
    },
    "radial_maze": {
        "params_fn": generate_radial_maze_scene_params,
        "class_path": "metta.map.scenes.radial_maze.RadialMaze",
    },
    "random_objects": {
        "params_fn": generate_random_objects_scene_params,
        "class_path": "metta.map.scenes.random_objects.RandomObjects",
    },
    "room_grid": {
        "params_fn": generate_room_grid_scene_params,
        "class_path": "metta.map.scenes.room_grid.RoomGrid",
    },
    "wfc": {
        "params_fn": generate_wfc_scene_params,
        "class_path": "metta.map.scenes.wfc.WFC",
    },
}
