import logging
import os
import random
import zipfile

import boto3
import numpy as np
from botocore.exceptions import NoCredentialsError
from filelock import FileLock

from metta.common.util.config import Config
from metta.mettagrid.level_builder import Level, LevelBuilder

logger = logging.getLogger("terrain_from_numpy")


def pick_random_file(path):
    chosen = None
    count = 0
    with os.scandir(path) as it:
        for entry in it:
            count += 1
            # with probability 1/count, pick this entry
            if random.randrange(count) == 0:
                chosen = entry.name
    return chosen


def download_from_s3(s3_path: str, save_path: str):
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}. Must start with s3://")

    s3_parts = s3_path[5:].split("/", 1)
    if len(s3_parts) < 2:
        raise ValueError(f"Invalid S3 path: {s3_path}. Must be in format s3://bucket/path")

    bucket = s3_parts[0]
    key = s3_parts[1]

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        # Download the file directly to disk
        s3_client = boto3.client("s3")
        s3_client.download_file(Bucket=bucket, Key=key, Filename=save_path)
        print(f"Successfully downloaded s3://{bucket}/{key} to {save_path}")

    except NoCredentialsError as e:
        raise e
    except Exception as e:
        raise e


class TerrainFromNumpyParams(Config):
    objects: dict[str, int] = {}
    agents: int | dict[str, int] = 0
    dir: str
    file: str | None = None


class TerrainFromNumpy(LevelBuilder):
    """
    This class is used to load a terrain environment from numpy arrays on s3.

    It's not a MapGen scene, because we don't know the grid size until we load the file.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.params = TerrainFromNumpyParams(**kwargs)

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

    def get_labels(self) -> list[str]:
        return [self.params.dir.split("/")[0]]

    def build(self):
        root = self.params.dir.split("/")[0]

        map_dir = f"train_dir/{self.params.dir}"
        root_dir = f"train_dir/{root}"

        s3_path = f"s3://softmax-public/maps/{root}.zip"
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

        if self.params.file is None:
            uri = pick_random_file(map_dir)
        else:
            uri = self.params.file

        grid = np.load(f"{map_dir}/{uri}", allow_pickle=True)

        # remove agents to then repopulate
        grid[grid == "agent.agent"] = "empty"

        # Prepare agent labels
        if isinstance(self.params.agents, int):
            agent_labels = ["agent.agent"] * self.params.agents
        else:
            agent_labels = [f"agent.{name}" for name, count in self.params.agents.items() for _ in range(count)]

        num_agents = len(agent_labels)

        valid_positions = self.get_valid_positions(grid)
        random.shuffle(valid_positions)

        # Place agents in first slice
        agent_positions = valid_positions[:num_agents]
        for pos, label in zip(agent_positions, agent_labels, strict=False):
            grid[pos] = label

        # Convert to set for O(1) removal operations
        valid_positions_set = set(valid_positions[num_agents:])

        for obj_name, count in self.params.objects.items():
            count = count - np.where(grid == obj_name, 1, 0).sum()
            if count < 0:
                continue
            # Sample from remaining valid positions
            positions = random.sample(list(valid_positions_set), min(count, len(valid_positions_set)))
            for pos in positions:
                grid[pos] = obj_name
                valid_positions_set.remove(pos)

        return Level(grid=grid, labels=self.get_labels())
