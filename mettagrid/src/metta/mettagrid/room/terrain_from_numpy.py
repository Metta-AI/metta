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

from metta.mettagrid.room.room import Room

logger = logging.getLogger("terrain_from_numpy")


def safe_load(path, retries=5, delay=1.0):
    """
    Safely load numpy array with retries and handling for distributed environments.

    The "array.shape = shape" error in numpy can occur when:
    1. File is being written by another process
    2. File system race conditions in distributed training
    3. Memory mapping issues with concurrent access
    """
    for attempt in range(retries):
        try:
            # Check if file exists and has non-zero size
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            file_size = os.path.getsize(path)
            if file_size == 0:
                raise ValueError(f"File is empty: {path}")

            # Try different loading strategies to handle various failure modes
            try:
                # First attempt: standard load
                return np.load(path, allow_pickle=True)
            except ValueError as e:
                if "array.shape" in str(e):
                    # Shape assignment error - try loading with copy
                    logger.warning(f"Shape assignment error on {path}, attempting workaround")
                    # Load with mmap_mode='c' (copy-on-write) and immediately copy
                    arr = np.load(path, allow_pickle=True, mmap_mode="c")
                    return arr.copy()
                else:
                    raise

        except (ValueError, OSError, FileNotFoundError) as e:
            if attempt < retries - 1:
                logger.warning(f"Failed to load {path} (attempt {attempt + 1}/{retries}): {e}")
                # Exponential backoff with jitter to reduce thundering herd
                sleep_time = delay * (2**attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_time)
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
        file: str | None = None,
        border_width: int = 0,
        border_object: str = "wall",
    ):
        self._dir = dir
        self._file = file
        self._agents = agents
        self._objects = objects
        super().__init__(border_width=border_width, border_object=border_object)

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

    def _build(self):
        root = self._dir.split("/")[0]
        self.labels.append(root)

        map_dir = f"train_dir/{self._dir}"
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

        if self._file is None:
            uri = pick_random_file(map_dir)
            if uri is None:
                raise ValueError(f"No valid files found in {map_dir}")
        else:
            uri = self._file

        file_path = f"{map_dir}/{uri}"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Terrain file not found: {file_path}")

        level = safe_load(file_path)

        # Validate loaded array
        if not isinstance(level, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(level)} from {file_path}")
        if level.ndim != 2:
            raise ValueError(f"Expected 2D array, got {level.ndim}D array from {file_path}")

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

        valid_positions = self.get_valid_positions(level)
        random.shuffle(valid_positions)

        # 5. Place agents in first slice
        agent_positions = valid_positions[:num_agents]
        for pos, label in zip(agent_positions, agent_labels, strict=False):
            level[pos] = label

        # Convert to set for O(1) removal operations
        valid_positions_set = set(valid_positions[num_agents:])

        for obj_name, count in self._objects.items():
            count = count - np.where(level == obj_name, 1, 0).sum()
            if count < 0:
                continue
            # Sample from remaining valid positions
            positions = random.sample(list(valid_positions_set), min(count, len(valid_positions_set)))
            for pos in positions:
                level[pos] = obj_name
                valid_positions_set.remove(pos)

        self._level = level
        return self._level
