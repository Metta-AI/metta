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
from scipy.signal import convolve2d

from metta.mettagrid.room.room import Room

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

    def positions_in_same_area(self, level, num_agents):
        """
        Returns num_agents positions, all with valid_mask=True, all within a 5x5 square
        with the central 3x3 cut out (i.e., only the border of the 5x5 square), chosen randomly.
        Uses convolution to efficiently find valid 5x5 squares.
        If not enough such positions exist, returns [].
        """

        # Compute valid_mask as in get_valid_positions
        empty_mask = level == "empty"
        has_empty_neighbor = np.zeros_like(empty_mask, dtype=bool)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(empty_mask, shift=(dr, dc), axis=(0, 1))
            has_empty_neighbor |= shifted
        valid_mask = empty_mask & has_empty_neighbor
        valid_mask[0, :] = False
        valid_mask[-1, :] = False
        valid_mask[:, 0] = False
        valid_mask[:, -1] = False

        # Build the 5x5 border kernel (1s on border, 0s in center 3x3)
        kernel = np.zeros((5, 5), dtype=int)
        kernel[0, :] = 1
        kernel[4, :] = 1
        kernel[:, 0] = 1
        kernel[:, 4] = 1

        # Convolve valid_mask with the kernel to find all top-left corners of valid 5x5 squares
        conv = convolve2d(valid_mask.astype(int), kernel, mode="valid")
        # Find all top-left corners where all border cells are valid
        possible_squares = np.argwhere(conv >= num_agents)

        if len(possible_squares) == 0:
            return []

        # Choose a random possible square
        square_idx = random.choice(range(len(possible_squares)))
        top_left = possible_squares[square_idx]
        r0, c0 = top_left

        # List all valid positions in the 5x5 border (excluding the central 3x3)
        border_positions = []
        for dr in range(5):
            for dc in range(5):
                # Exclude central 3x3
                if 1 <= dr <= 3 and 1 <= dc <= 3:
                    continue
                rr, cc = r0 + dr, c0 + dc
                if valid_mask[rr, cc]:
                    border_positions.append((rr, cc))

        return border_positions

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
        else:
            uri = self._file

        level = safe_load(f"{map_dir}/{uri}")
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

        valid_positions = self.positions_in_same_area(level, num_agents)
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
