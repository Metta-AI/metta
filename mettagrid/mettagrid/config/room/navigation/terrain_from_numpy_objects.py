import os
import random
import time
import zipfile

import boto3
import numpy as np
from botocore.exceptions import NoCredentialsError
from filelock import FileLock

from mettagrid.config.room.room import Room


def safe_load(path, retries=5, delay=1.0):
    for attempt in range(retries):
        try:
            level: np.ndarray = np.load(path, allow_pickle=True)
        except ValueError:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            raise Exception(f"Failed to load {path}")
        return level


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
    This class is used to load a terrain environment from numpy arrays on s3.

    Features:
    - Support for multiple teams (team_1, team_2, team_3)
    - Colored mines and generators (red, green, blue)
    - Configurable agent distribution
    - Random distribution of altars
    """

    def __init__(
        self,
        dir,
        border_width: int = 0,
        border_object: str = "wall",
        num_agents: int = 10,
        file: str | None = None,
        teams: list | None = None,
        object_colors: list | None = None,
    ):
        zipped_dir = dir + ".zip"
        lock_path = zipped_dir + ".lock"
        # Only one process can hold this lock at a time:
        with FileLock(lock_path):
            if not os.path.exists(dir) and not os.path.exists(zipped_dir):
                s3_path = f"s3://softmax-public/maps/{zipped_dir}"
                download_from_s3(s3_path, zipped_dir)
            if not os.path.exists(dir) and os.path.exists(zipped_dir):
                with zipfile.ZipFile(zipped_dir, "r") as zip_ref:
                    zip_ref.extractall(os.path.dirname(dir))

        self.files = os.listdir(dir)
        self.dir = dir
        self.num_agents = num_agents
        self.uri = file
        self.teams = teams
        if object_colors is None:
            self.object_colors = ["red"]  # default object color is red
        else:
            self.object_colors = object_colors

        super().__init__(border_width=border_width, border_object=border_object, labels=["terrain"])

    def get_valid_positions(self, level):
        valid_positions = []
        for i in range(1, level.shape[0] - 1):
            for j in range(1, level.shape[1] - 1):
                if level[i, j] == "empty":
                    # Check if position is accessible from at least one direction
                    if (
                        level[i - 1, j] == "empty"
                        or level[i + 1, j] == "empty"
                        or level[i, j - 1] == "empty"
                        or level[i, j + 1] == "empty"
                    ):
                        valid_positions.append((i, j))
        return valid_positions

    def _place_objects(self, level, count, object_type, valid_positions):
        """Helper function to place objects on the map."""
        if len(valid_positions) < count:
            count = len(valid_positions)
        positions = random.sample(valid_positions, count)
        for pos in positions:
            level[pos] = object_type
            valid_positions.remove(pos)
        return level, valid_positions

    def _distribute_agents(self, level, valid_positions):
        """Distribute agents across teams."""

        # Distribute agents evenly across teams
        base_agents = self.num_agents // 3
        remainder = self.num_agents % 3

        if self.teams is not None:
            for i, team in enumerate(self.teams):
                num_team_agents = base_agents + (1 if i < remainder else 0)
                level, valid_positions = self._place_objects(level, num_team_agents, f"agent.{team}", valid_positions)
        else:
            level, valid_positions = self._place_objects(level, self.num_agents, "agent", valid_positions)

        return level, valid_positions

    def _place_generators_and_mines(self, level, valid_positions, area):
        """Place colored generators and mines."""

        # Calculate base counts for each color
        base_count = area // random.randint(360, 600)  # Slightly fewer than original to account for colors

        for color in self.object_colors:
            # Add some randomness to counts
            gen_count = max(1, base_count + random.randint(-2, 2))
            mine_count = max(1, base_count + random.randint(-2, 2))

            # Place colored generators
            level, valid_positions = self._place_objects(level, gen_count, f"generator.{color}", valid_positions)

            # Place colored mines
            level, valid_positions = self._place_objects(level, mine_count, f"mine.{color}", valid_positions)

        return level, valid_positions

    def _build(self):
        # Load level
        if self.uri is not None:
            level: np.ndarray = safe_load(f"{self.dir}/{self.uri}")
        else:
            uri = np.random.choice(self.files)
            level: np.ndarray = safe_load(f"{self.dir}/{uri}")
        self.set_size_labels(level.shape[1], level.shape[0])

        # Clear existing agents
        agents_mask = np.char.startswith(level.astype(str), "agent.")
        level[agents_mask] = "empty"

        # Get valid positions and calculate area
        valid_positions = self.get_valid_positions(level)
        area = level.shape[0] * level.shape[1]

        # Place agents
        level, valid_positions = self._distribute_agents(level, valid_positions)

        # Place altars
        num_hearts = area // 180
        level, valid_positions = self._place_objects(level, num_hearts, "altar", valid_positions)

        # Place generators and mines
        level, valid_positions = self._place_generators_and_mines(level, valid_positions, area)

        self._level = level
        return self._level
