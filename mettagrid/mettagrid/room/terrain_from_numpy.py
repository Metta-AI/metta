import logging
import os
import random
import time
import zipfile

import boto3
import numpy as np
from botocore.exceptions import NoCredentialsError
from filelock import FileLock
from omegaconf import DictConfig

from mettagrid.room.room import Room

logger = logging.getLogger("terrain_from_numpy")


def safe_load(path, retries=5, delay=1.0):
    for attempt in range(retries):
        try:
            return np.load(path, allow_pickle=True)
        except ValueError:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            raise


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


def download_from_s3(s3_path: str, save_path: str, location: str = "us-east-1"):
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


class TerrainFromNumpy(Room):
    """
    This class is used to load a terrain environment from numpy arrays on s3

    These maps each have 10 agents in them .
    """

    def __init__(
        self,
        objects: DictConfig,
        agents: int | DictConfig = 10,
        dir: str = "terrain_maps_nohearts",
        border_width: int = 0,
        border_object: str = "wall",
        file: str | None = None,
        team: str | None = None,
    ):
        root = dir.split("/")[0]
        zipped_dir = root + ".zip"
        lock_path = zipped_dir + ".lock"
        # Only one process can hold this lock at a time:
        if not os.path.exists(dir) and not os.path.exists(zipped_dir):
            with FileLock(lock_path):
                s3_path = f"s3://softmax-public/maps/{zipped_dir}"
                download_from_s3(s3_path, zipped_dir)
        if not os.path.exists(root) and os.path.exists(zipped_dir):
            with zipfile.ZipFile(zipped_dir, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(root))
            os.remove(zipped_dir)
            logger.info(f"Extracted {zipped_dir} to {root}")
        if file is None:
            self.uri = pick_random_file(dir)
        else:
            self.uri = file
        self.dir = dir
        self._agents = agents
        self._objects = objects
        self.team = team
        super().__init__(border_width=border_width, border_object=border_object, labels=[root])

    def get_valid_positions(self, level, num_needed):
        """
        Fastest approach: Vectorized operations with random sampling
        This is the best balance of speed and true randomness
        """
        # Convert to boolean for faster operations
        empty_mask = level == "empty"
        h, w = empty_mask.shape

        # Use slicing instead of roll (faster, no copies)
        valid_mask = np.zeros((h, w), dtype=bool)

        # Interior cells only
        interior = empty_mask[1:-1, 1:-1]

        # Check all four neighbors at once
        has_neighbor = (
            empty_mask[:-2, 1:-1]  # up
            | empty_mask[2:, 1:-1]  # down
            | empty_mask[1:-1, :-2]  # left
            | empty_mask[1:-1, 2:]  # right
        )

        valid_mask[1:-1, 1:-1] = interior & has_neighbor

        # Get flat indices (faster than coordinate pairs)
        valid_flat = np.flatnonzero(valid_mask)

        if len(valid_flat) <= num_needed:
            # Need all positions
            np.random.shuffle(valid_flat)
            indices = valid_flat
        else:
            # Random sample without replacement
            indices = np.random.choice(valid_flat, size=num_needed, replace=False)

        # Convert flat indices back to coordinates
        return [(idx // w, idx % w) for idx in indices]

    def _build(self):
        level = safe_load(f"{self.dir}/{self.uri}")
        height, width = level.shape
        self.set_size_labels(width, height)

        # remove agents to then repopulate
        level[level == "agent.agent"] = "empty"

        # 3. Prepare agent labels
        if isinstance(self._agents, int):
            agent_labels = ["agent.agent"] * self._agents
        elif isinstance(self._agents, DictConfig):
            agent_labels = [f"agent.{name}" for name, count in self._agents.items() for _ in range(count)]
        else:
            raise TypeError("Unsupported _agents type")
        num_agents = len(agent_labels)
        num_objects = sum(self._objects.values())
        num_needed = num_agents + num_objects

        valid_positions = self.get_valid_positions(level, num_needed)

        # Place agents first
        for i, label in enumerate(agent_labels):
            if i < len(valid_positions):
                level[valid_positions[i]] = label

        # Place objects
        idx = num_agents
        for obj_name, count in self._objects.items():
            for j in range(count):
                if idx < len(valid_positions):
                    level[valid_positions[idx]] = obj_name
                    idx += 1

        self._level = level

        return self._level
