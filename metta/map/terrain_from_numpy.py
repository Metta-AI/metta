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
        object_names: list[str] = Field(default_factory=list)
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

    def get_valid_positions(self, level):
        # Create a boolean mask for empty cells
        empty_mask = level == "empty"

        # Use numpy's roll to check adjacent cells efficiently
        has_empty_neighbor = (
            np.roll(empty_mask, 1, axis=0)  # Check up
            | np.roll(empty_mask, -1, axis=0)  # Check down
            | np.roll(empty_mask, 1, axis=1)  # Check left
            | np.roll(empty_mask, -1, axis=1)  # Check right
        )

        # Valid positions are empty cells with at least one empty neighbor
        # Exclude border cells (indices 0 and -1)
        valid_mask = empty_mask & has_empty_neighbor
        valid_mask[0, :] = False
        valid_mask[-1, :] = False
        valid_mask[:, 0] = False
        valid_mask[:, -1] = False

        # Get coordinates of valid positions
        valid_positions = list(zip(*np.where(valid_mask), strict=False))
        return valid_positions

    def clean_grid(self, grid):
        grid[grid == "agent.agent"] = "empty"
        if self.config.remove_altars:
            grid[grid == "altar"] = "empty"

        # Prepare agent labels
        if isinstance(self.config.agents, int):
            agent_labels = ["agent.agent"] * self.config.agents
        else:
            agent_labels = [f"agent.{name}" for name, count in self.config.agents.items() for _ in range(count)]

        valid_positions = self.get_valid_positions(grid)
        self.config.rng.shuffle(valid_positions)
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

        grid, valid_positions, agent_labels = self.clean_grid(grid)
        num_agents = len(agent_labels)
        # Place agents in first slice
        agent_positions = valid_positions[:num_agents]
        for pos, label in zip(agent_positions, agent_labels, strict=False):
            grid[pos] = label

        # Convert to set for O(1) removal operations
        valid_positions_set = set(valid_positions[num_agents:])

        for obj_name, count in self.config.objects.items():
            count = count - np.where(grid == obj_name, 1, 0).sum()
            if count < 0:
                continue
            # Sample from remaining valid positions
            positions = self.config.rng.sample(list(valid_positions_set), min(count, len(valid_positions_set)))
            for pos in positions:
                grid[pos] = obj_name
                valid_positions_set.remove(pos)

        return GameMap(grid=grid)


class InContextLearningFromNumpy(TerrainFromNumpy):
    def __init__(self, config: TerrainFromNumpy.Config):
        super().__init__(config)

    def build(self):
        map_dir = self.setup()

        if self.config.file is None:
            uri = pick_random_file(map_dir, self.config.rng)
        else:
            uri = self.config.file

        grid = np.load(f"{map_dir}/{uri}", allow_pickle=True)

        grid, valid_positions, agent_labels = self.clean_grid(grid)
        num_agents = len(agent_labels)
        agent_positions = valid_positions[:num_agents]
        for pos, label in zip(agent_positions, agent_labels, strict=False):
            grid[pos] = label
        # placeholder indices for objects
        mask = ~np.isin(grid, ("agent.agent", "wall", "empty"))
        converter_indices = np.argwhere(mask)
        grid[mask] = "empty"

        object_names = [name for name in self.config.objects for _ in range(self.config.objects[name])]
        self.config.rng.shuffle(object_names)

        for idx, object in zip(converter_indices, object_names, strict=False):
            grid[tuple(idx)] = object

        return GameMap(grid=grid)
