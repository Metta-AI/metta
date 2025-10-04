import os
import random
import zipfile
from typing import Optional

import boto3
import numpy as np
from botocore.exceptions import NoCredentialsError
from filelock import FileLock
from pydantic import ConfigDict, Field

from metta.common.util.log_config import getRankAwareLogger
from metta.utils.uri import ParsedURI
from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig

logger = getRankAwareLogger(__name__)

MAPS_ROOT = "s3://softmax-public/maps"


def pick_random_file(path, rng):
    chosen = None
    count = 0
    with os.scandir(path) as it:
        for entry in it:
            count += 1
            # with probability 1/count, pick this entry
            if rng.randrange(count) == 0:
                chosen = entry.name
    return chosen


def download_from_s3(s3_path: str, save_path: str):
    parsed = ParsedURI.parse(s3_path)
    bucket, key = parsed.require_s3()

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        # Download the file directly to disk
        s3_client = boto3.client("s3")
        s3_client.download_file(Bucket=bucket, Key=key, Filename=save_path)
        logger.info(f"Successfully downloaded {parsed.canonical} to {save_path}")

    except NoCredentialsError as e:
        raise e
    except Exception as e:
        raise e


class TerrainFromNumpy(MapBuilder):
    """This class is used to load a terrain environment from numpy arrays on s3.

    It's not a MapGen scene, because we don't know the grid size until we load the file."""

    class Config(MapBuilderConfig["TerrainFromNumpy"]):
        # Allow non-pydantic types like random.Random
        model_config = ConfigDict(arbitrary_types_allowed=True)
        objects: dict[str, int] = Field(default_factory=dict)
        agents: int | dict[str, int] = Field(default=0, ge=0)
        dir: str
        file: Optional[str] = None
        remove_altars: bool = False
        mass_in_center: bool = False
        rng: random.Random = Field(default_factory=random.Random, exclude=True)

    def __init__(self, config: Config):
        self.config = config

    def setup(self):
        root = self.config.dir.split("/")[0]

        map_dir = f"train_dir/{self.config.dir}"
        root_dir = f"train_dir/{root}"

        s3_path = f"{MAPS_ROOT}/{root}.zip"
        local_zipped_dir = root_dir + ".zip"
        # Only one process can hold this lock at a time:
        with FileLock(local_zipped_dir + ".lock"):
            if not os.path.exists(map_dir) and not os.path.exists(local_zipped_dir):
                download_from_s3(s3_path, local_zipped_dir)
            if not os.path.exists(root_dir) and os.path.exists(local_zipped_dir):
                with zipfile.ZipFile(local_zipped_dir, "r") as zip_ref:
                    zip_ref.extractall(os.path.dirname(root_dir))
                os.remove(local_zipped_dir)
                logger.info(f"Extracted {local_zipped_dir} to {root_dir}")
        return map_dir

    def get_valid_positions(self, level, assemblers=False, mass_in_center=False):
        """Get valid positions as a numpy array of shape (N, 2)."""
        empty_mask = level == "empty"

        # Check for empty neighbors - use & for assemblers (all neighbors), | for others (any neighbor)
        neighbor_op = np.logical_and if assemblers else np.logical_or
        has_empty_neighbor = neighbor_op.reduce(
            [
                np.roll(empty_mask, 1, axis=0),  # Up
                np.roll(empty_mask, -1, axis=0),  # Down
                np.roll(empty_mask, 1, axis=1),  # Left
                np.roll(empty_mask, -1, axis=1),  # Right
            ]
        )

        valid_mask = empty_mask & has_empty_neighbor

        # Exclude border cells for non-assemblers
        if not assemblers:
            valid_mask[0, :] = False
            valid_mask[-1, :] = False
            valid_mask[:, 0] = False
            valid_mask[:, -1] = False

        # Get coordinates as numpy array
        valid_positions = np.array(np.where(valid_mask)).T  # Shape: (N, 2)

        if assemblers and mass_in_center:
            center = np.array(level.shape) / 2
            distances = np.sum((valid_positions - center) ** 2, axis=1)
            valid_positions = valid_positions[np.argsort(distances)]

            # Space out positions to ensure neighbors are empty
            occupied = np.zeros(level.shape, dtype=bool)
            spaced_positions = []

            for pos in valid_positions:
                y, x = pos
                y_slice = slice(max(0, y - 1), min(level.shape[0], y + 2))
                x_slice = slice(max(0, x - 1), min(level.shape[1], x + 2))

                if not occupied[y_slice, x_slice].any():
                    spaced_positions.append(pos)
                    occupied[y_slice, x_slice] = True

            return np.array(spaced_positions, dtype=int)

        # Shuffle using indices to avoid numpy array shuffle issues
        indices = np.arange(len(valid_positions))
        indices_list = indices.tolist()
        self.config.rng.shuffle(indices_list)
        return valid_positions[indices_list]

    def clean_grid(self, grid, assemblers=True, mass_in_center=False):
        grid[grid == "agent.agent"] = "empty"
        if self.config.remove_altars:
            grid[grid == "altar"] = "empty"

        # Prepare agent labels
        if isinstance(self.config.agents, int):
            agent_labels = ["agent.agent"] * self.config.agents
        else:
            agent_labels = [f"agent.{name}" for name, count in self.config.agents.items() for _ in range(count)]

        valid_positions = self.get_valid_positions(grid, assemblers, mass_in_center)
        return grid, valid_positions, agent_labels

    def build(self):
        pass


class NavigationFromNumpy(TerrainFromNumpy):
    def __init__(self, config: TerrainFromNumpy.Config):
        super().__init__(config)

    def build(self):
        map_dir = self.setup()
        if self.config.file is None:
            uri = pick_random_file(map_dir, self.config.rng)
        else:
            uri = self.config.file

        grid = np.load(f"{map_dir}/{uri}", allow_pickle=True)

        grid, valid_positions, agent_labels = self.clean_grid(grid, assemblers=False)
        num_agents = len(agent_labels)

        # Place agents
        agent_positions = valid_positions[:num_agents]
        for pos, label in zip(agent_positions, agent_labels, strict=False):
            grid[tuple(pos)] = label

        # Keep remaining positions
        valid_positions = valid_positions[num_agents:]

        for obj_name, count in self.config.objects.items():
            # Adjust count for existing objects
            existing_count = (grid == obj_name).sum()
            count = count - existing_count
            if count <= 0:
                continue

            # Sample positions
            num_to_place = min(count, len(valid_positions))
            indices = np.array(self.config.rng.sample(range(len(valid_positions)), num_to_place))

            # Place objects
            for pos in valid_positions[indices]:
                grid[tuple(pos)] = obj_name

            # Remove used positions efficiently
            mask = np.ones(len(valid_positions), dtype=bool)
            mask[indices] = False
            valid_positions = valid_positions[mask]

        return GameMap(grid=grid)


class CogsVClippiesFromNumpy(TerrainFromNumpy):
    def __init__(self, config: TerrainFromNumpy.Config):
        super().__init__(config)

    def carve_out_patches(self, grid, valid_positions, num_patches):
        """Carve out 9x9 empty patches and return their centers as a numpy array."""
        if num_patches <= 0:
            return grid, np.empty((0, 2), dtype=int)

        patch_size = 9
        half_patch = patch_size // 2
        grid_shape = grid.shape

        # Build set for quick exclusion
        valid_positions_set = set(map(tuple, valid_positions))
        empty_centers = []

        # Pre-compute valid ranges for center positions
        min_coord = half_patch
        max_y = grid_shape[0] - half_patch - 1
        max_x = grid_shape[1] - half_patch - 1

        attempts = 0
        max_attempts = num_patches * 20

        while len(empty_centers) < num_patches and attempts < max_attempts:
            attempts += 1

            center = (self.config.rng.randint(min_coord, max_y), self.config.rng.randint(min_coord, max_x))

            if center in valid_positions_set or center in empty_centers:
                continue

            # Check if patch overlaps any valid position
            y_range = range(center[0] - half_patch, center[0] + half_patch + 1)
            x_range = range(center[1] - half_patch, center[1] + half_patch + 1)

            if any((i, j) in valid_positions_set for i in y_range for j in x_range):
                continue

            # Carve out the patch using slicing (faster than loop)
            y_slice = slice(center[0] - half_patch, center[0] + half_patch + 1)
            x_slice = slice(center[1] - half_patch, center[1] + half_patch + 1)
            grid[y_slice, x_slice] = "empty"
            empty_centers.append(center)

        return grid, np.array(empty_centers, dtype=int)

    def build(self):
        map_dir = self.setup()
        if self.config.file is None:
            uri = pick_random_file(map_dir, self.config.rng)
        else:
            uri = self.config.file

        grid = np.load(f"{map_dir}/{uri}", allow_pickle=True)

        grid, valid_positions, agent_labels = self.clean_grid(
            grid, assemblers=True, mass_in_center=self.config.mass_in_center
        )
        num_agents = len(agent_labels)

        # GUARANTEE: Ensure we have enough valid positions for all agents
        if len(valid_positions) < num_agents:
            patches_needed = num_agents - len(valid_positions)
            grid, empty_centers = self.carve_out_patches(grid, valid_positions, patches_needed)
            if len(empty_centers) > 0:
                valid_positions = np.vstack([valid_positions, empty_centers])

        # Place agents with bias towards center
        # Take first half of positions (already sorted by distance from center if mass_in_center=True)
        half_point = len(valid_positions) // 2
        agent_pool_size = max(half_point, num_agents)  # Ensure we have enough positions
        agent_position_pool = valid_positions[:agent_pool_size]

        # Sample agent positions
        agent_indices = self.config.rng.sample(range(len(agent_position_pool)), num_agents)
        agent_positions = agent_position_pool[agent_indices]

        # Place all agents
        for pos, label in zip(agent_positions, agent_labels, strict=False):
            grid[tuple(pos)] = label

        # Remove agent positions from valid_positions
        agent_positions_set = set(map(tuple, agent_positions))
        mask = np.array([tuple(pos) not in agent_positions_set for pos in valid_positions])
        valid_positions = valid_positions[mask]

        # GUARANTEE: Ensure we have enough valid positions for all objects
        total_objects = sum(self.config.objects.values())
        if len(valid_positions) < total_objects:
            patches_needed = total_objects - len(valid_positions)
            grid, empty_centers = self.carve_out_patches(grid, valid_positions, patches_needed)
            if len(empty_centers) > 0:
                valid_positions = np.vstack([valid_positions, empty_centers])

        # Place objects (ensuring they have empty neighbors since valid_positions came from assemblers=True)
        for obj_name, count in self.config.objects.items():
            if count <= 0:
                continue

            if count <= 2:
                # Place first 'count' positions
                positions = valid_positions[:count]
                for position in positions:
                    grid[tuple(position)] = obj_name
                valid_positions = valid_positions[count:]
            else:
                # Place 2 at start, rest scattered
                center_positions = valid_positions[:2]
                num_surrounding = count - 2

                if num_surrounding > 0 and len(valid_positions) > 2:
                    available = valid_positions[2:]
                    num_to_sample = min(num_surrounding, len(available))
                    sample_indices = self.config.rng.sample(range(len(available)), num_to_sample)
                    other_positions = available[sample_indices]
                    all_positions = np.vstack([center_positions, other_positions])
                else:
                    all_positions = center_positions

                # Place all objects of this type
                for position in all_positions:
                    grid[tuple(position)] = obj_name

                # Remove used positions
                used_set = set(map(tuple, all_positions))
                mask = np.array([tuple(pos) not in used_set for pos in valid_positions])
                valid_positions = valid_positions[mask]

        return GameMap(grid=grid)


# class InContextLearningFromNumpy(TerrainFromNumpy):
#     def __init__(self, config: TerrainFromNumpy.Config):
#         super().__init__(config)

#     def build(self):
#         map_dir = self.setup()

#         if self.config.file is None:
#             uri = pick_random_file(map_dir, self.config.rng)
#         else:
#             uri = self.config.file

#         grid = np.load(f"{map_dir}/{uri}", allow_pickle=True)

#         grid, valid_positions, agent_labels = self.clean_grid(grid)
#         num_agents = len(agent_labels)
#         agent_positions = valid_positions[:num_agents]
#         for pos, label in zip(agent_positions, agent_labels, strict=False):
#             grid[tuple(pos)] = label
#         # placeholder indices for objects
#         mask = ~np.isin(grid, ("agent.agent", "wall", "empty"))
#         converter_indices = np.argwhere(mask)
#         grid[mask] = "empty"

#         object_names = [name for name in self.config.objects for _ in range(self.config.objects[name])]
#         self.config.rng.shuffle(object_names)

#         for idx, object in zip(converter_indices, object_names, strict=False):
#             grid[tuple(idx)] = object

#         return GameMap(grid=grid)
