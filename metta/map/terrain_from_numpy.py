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

    def sort_by_distance_from_center(self, positions, grid_shape):
        center = np.array(grid_shape) / 2  # Use grid dimensions
        distances = np.sum((positions - center) ** 2, axis=1)
        return positions[np.argsort(distances)]

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
        valid_mask[0, :] = False
        valid_mask[-1, :] = False
        valid_mask[:, 0] = False
        valid_mask[:, -1] = False

        # Get coordinates as numpy array
        valid_positions = np.array(np.where(valid_mask)).T  # Shape: (N, 2)

        if assemblers and mass_in_center:
            valid_positions = self.sort_by_distance_from_center(valid_positions, level.shape)

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

    def clean_grid(self, grid, assemblers=True, mass_in_center=False, clear_agents=True):
        if clear_agents:
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
        """
        Carve out square 9x9 'empty' patches to create space.

        Rules:
        - Allowed to carve over walls/objects/empties.
        - NOT allowed to carve over agents (cells equal to "agent.agent").
        - Keeps patches fully inside bounds.
        - Returns centers of patches actually carved.

        Args:
            grid: np.ndarray (H, W) of labels (object/str dtype).
            valid_positions: unused (kept for API compatibility).
            num_patches: number of patches to try to carve.

        Returns:
            grid: modified in-place (also returned for convenience)
            centers: np.ndarray (M, 2) of int (y, x) centers, M <= num_patches
        """
        PATCH = 9
        HALF = PATCH // 2  # = 4
        H, W = grid.shape

        # Precompute where agents are; faster than re-checking strings each loop.
        # TODO update this to be if it starts with "agent."
        agent_mask = grid == "agent.agent"

        centers = []
        centers_set = set()  # O(1) duplicate check
        attempts = 0
        max_attempts = num_patches * 20  # simple cap to avoid long loops

        vp_mask = np.zeros(grid.shape, bool)
        vp_mask[valid_positions[:, 0], valid_positions[:, 1]] = True

        while len(centers) < num_patches and attempts < max_attempts:
            attempts += 1

            # Sample a center that can fit a 9x9 patch fully inside the grid.
            y = self.config.rng.randint(HALF, H - HALF - 1)
            x = self.config.rng.randint(HALF, W - HALF - 1)
            if (y, x) in centers_set:
                continue

            ys = slice(y - HALF, y + HALF + 1)  # inclusive on both sides → width 9
            xs = slice(x - HALF, x + HALF + 1)

            # Skip if the 9x9 patch would touch any agent cell.
            if agent_mask[ys, xs].any() or vp_mask[ys, xs].any():
                continue

            # Carve: set the region to "empty" (allowed over walls/anything except agents).
            grid[ys, xs] = "empty"
            centers.append((y, x))
            centers_set.add((y, x))

        return grid, np.array(centers, dtype=int)

    def build(self):
        map_dir = self.setup()
        if self.config.file is None:
            uri = pick_random_file(map_dir, self.config.rng)
        else:
            uri = self.config.file

        grid = np.load(f"{map_dir}/{uri}", allow_pickle=True)

        grid, valid_agent_positions, agent_labels = self.clean_grid(grid)
        if self.config.mass_in_center:
            valid_agent_positions = self.sort_by_distance_from_center(valid_agent_positions, grid.shape)
        num_agents = len(agent_labels)
        if len(valid_agent_positions) < num_agents:
            raise ValueError(
                f"Not enough valid positions for {num_agents} agents "
                f"(only {len(valid_agent_positions)} available) in map {uri}"
            )

        for pos, label in zip(valid_agent_positions[:num_agents], agent_labels, strict=True):
            grid[tuple(pos)] = label

        grid, valid_assembler_positions, _ = self.clean_grid(
            grid, assemblers=True, mass_in_center=self.config.mass_in_center, clear_agents=False
        )

        # Ensure we have enough valid positions for all objects
        total_objects = sum(self.config.objects.values())
        if len(valid_assembler_positions) < total_objects:
            patches_needed = total_objects - len(valid_assembler_positions)
            print(f"Carving out {patches_needed} patches")
            grid, empty_centers = self.carve_out_patches(grid, valid_assembler_positions, patches_needed)
            if len(empty_centers) > 0:
                valid_assembler_positions = np.vstack([valid_assembler_positions, empty_centers])

        if len(valid_assembler_positions) < total_objects:
            print(
                f"Not enough valid positions for {total_objects} objects "
                f"(only {len(valid_assembler_positions)} available) in map {uri}"
            )

        # Place objects (ensuring they have empty neighbors since valid_positions came from assemblers=True)

        for obj_name, count in self.config.objects.items():
            if count <= 0 or len(valid_assembler_positions) == 0:
                continue
            if count > 1:
                num_to_place_in_center = count // 2
                num_surrounding = count - num_to_place_in_center

                # Place half at center
                center_positions = valid_assembler_positions[:num_to_place_in_center].copy()

                # place the remaining randomly
                if num_surrounding > 0 and len(valid_assembler_positions) > num_to_place_in_center:
                    available = valid_assembler_positions[num_to_place_in_center:].copy()
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
                mask = np.array([tuple(pos) not in used_set for pos in valid_assembler_positions])
                if len(mask) == 0:
                    continue
                valid_assembler_positions = valid_assembler_positions[mask]

            else:
                # Place first 'count' positions
                positions = valid_assembler_positions[:count]
                for position in positions:
                    grid[tuple(position)] = obj_name
                valid_assembler_positions = valid_assembler_positions[count:]

        num_agents_in_grid = (grid == "agent.agent").sum()
        if num_agents_in_grid != len(agent_labels):
            raise ValueError(
                f"Number of agents in grid ({num_agents_in_grid}) does not match ({len(agent_labels)}) for map {uri}"
            )

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
