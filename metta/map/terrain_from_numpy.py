import os
import random
import zipfile
from abc import ABC, abstractmethod
from typing import Optional

import boto3
import numpy as np
from botocore.exceptions import NoCredentialsError
from filelock import FileLock
from pydantic import Field

from metta.common.util.log_config import getRankAwareLogger
from mettagrid.map_builder import MapGrid
from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig, WithMaxRetriesConfig
from mettagrid.util.file import parse_uri

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
    parsed = parse_uri(s3_path, allow_none=False)

    if parsed.scheme != "s3":
        raise ValueError(f"Expected S3 URI, got: {parsed.scheme}")
    bucket, key = parsed.bucket, parsed.key

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


class TerrainFromNumpyConfig(MapBuilderConfig["TerrainFromNumpy"], WithMaxRetriesConfig):
    objects: dict[str, int] = Field(default_factory=dict)
    agents: int | dict[str, int] = Field(default=0, ge=0)
    dir: str
    file: Optional[str] = None
    remove_assemblers: bool = False
    seed: Optional[int] = None  # Use seed instead of Random object for picklability


class TerrainFromNumpy(MapBuilder[TerrainFromNumpyConfig], ABC):
    """This class is used to load a terrain environment from numpy arrays on s3.

    It's not a MapGen scene, because we don't know the grid size until we load the file."""

    def __init__(self, config: TerrainFromNumpyConfig):
        super().__init__(config)
        # Create Random instance from seed for use in build methods
        self.rng = random.Random(self.config.seed)

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

    def get_valid_positions(self, level: MapGrid, assemblers=False):
        # Create a boolean mask for empty cells
        empty_mask = level == "empty"

        if assemblers:
            has_empty_neighbor = (
                np.roll(empty_mask, 1, axis=0)  # Check up
                & np.roll(empty_mask, -1, axis=0)  # Check down
                & np.roll(empty_mask, 1, axis=1)  # Check left
                & np.roll(empty_mask, -1, axis=1)  # Check right
            )
            valid_mask = empty_mask & has_empty_neighbor
        else:
            # For non-assemblers, any empty cell is valid?
            # But we should also check for connectivity or at least valid placement rules.
            # For now, assuming same rules or simpler.
            # Actually, let's use the same logic for safety:

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

    def clean_grid(self, grid: MapGrid, assemblers=True):
        grid[grid == "agent.agent"] = "empty"
        if self.config.remove_assemblers:
            grid[grid == "altar"] = "empty"

        # Prepare agent labels
        if isinstance(self.config.agents, int):
            agent_labels = ["agent.agent"] * self.config.agents
        else:
            agent_labels = [f"agent.{name}" for name, count in self.config.agents.items() for _ in range(count)]

        valid_positions = self.get_valid_positions(grid, assemblers)
        self.rng.shuffle(valid_positions)
        return grid, valid_positions, agent_labels

    @abstractmethod
    def build(self) -> GameMap: ...


class NavigationFromNumpyConfig(MapBuilderConfig["NavigationFromNumpy"], WithMaxRetriesConfig):
    """Config for NavigationFromNumpy - explicit module-level class to avoid dynamic CloneConfig.

    This is needed because TerrainFromNumpyConfig is already bound to TerrainFromNumpy.
    When NavigationFromNumpy tries to reuse it, MapBuilder.__init_subclass__ would create
    a local CloneConfig class which can't be pickled for multiprocessing.
    """

    objects: dict[str, int] = Field(default_factory=dict)
    agents: int | dict[str, int] = Field(default=0, ge=0)
    dir: str
    file: Optional[str] = None
    remove_assemblers: bool = False
    seed: Optional[int] = None  # Use seed instead of Random object for picklability


class NavigationFromNumpy(MapBuilder[NavigationFromNumpyConfig]):
    def __init__(self, config: NavigationFromNumpyConfig):
        super().__init__(config)
        # Create Random instance from seed for use in build methods
        self.rng = random.Random(self.config.seed)

    def setup(self):
        """Setup map directory - copied from TerrainFromNumpy."""
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

    def get_valid_positions(self, level: MapGrid, assemblers=False):
        """Get valid positions - copied from TerrainFromNumpy."""
        # Create a boolean mask for empty cells
        empty_mask = level == "empty"

        if assemblers:
            has_empty_neighbor = (
                np.roll(empty_mask, 1, axis=0)  # Check up
                & np.roll(empty_mask, -1, axis=0)  # Check down
                & np.roll(empty_mask, 1, axis=1)  # Check left
                & np.roll(empty_mask, -1, axis=1)  # Check right
            )
            valid_mask = empty_mask & has_empty_neighbor
        else:
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

    def clean_grid(self, grid: MapGrid, assemblers=True):
        """Clean grid - copied from TerrainFromNumpy."""
        grid[grid == "agent.agent"] = "empty"
        if self.config.remove_assemblers:
            grid[grid == "assembler"] = "empty"

        # Prepare agent labels
        if isinstance(self.config.agents, int):
            agent_labels = ["agent.agent"] * self.config.agents
        else:
            agent_labels = [f"agent.{name}" for name, count in self.config.agents.items() for _ in range(count)]

        valid_positions = self.get_valid_positions(grid, assemblers)
        self.rng.shuffle(valid_positions)
        return grid, valid_positions, agent_labels

    def build(self):
        map_dir = self.setup()
        if self.config.file is None:
            uri = pick_random_file(map_dir, self.rng)
        else:
            uri = self.config.file

        grid = np.load(f"{map_dir}/{uri}", allow_pickle=True)

        # Replace 'altar' with 'assembler' for CVC compatibility
        # The terrain numpy files contain 'altar' but C++ only knows 'assembler'
        grid[grid == "altar"] = "assembler"

        grid, valid_positions, agent_labels = self.clean_grid(grid, assemblers=False)
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
            positions = self.rng.sample(list(valid_positions_set), min(count, len(valid_positions_set)))
            for pos in positions:
                grid[pos] = obj_name
                valid_positions_set.remove(pos)

        grid[grid == "altar"] = "assembler"

        return GameMap(grid=grid)
