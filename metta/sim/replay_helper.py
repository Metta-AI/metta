# Generate a graphical trace of multiple runs.

import json
import logging
import os
import zlib

import boto3
import numpy as np
import wandb

from metta.agent.policy_store import PolicyRecord
from metta.sim.simulation_config import SimulationConfig
from metta.util.file import s3_url, upload_to_s3
from metta.util.wandb.wandb_context import WandbRun
from mettagrid.mettagrid_env import MettaGridEnv


class ReplayHelper:
    """Helper class for generating and uploading replays."""

    def __init__(
        self, config: SimulationConfig, env: MettaGridEnv, policy_record: PolicyRecord, wandb_run: WandbRun | None
    ):
        self.config = config
        self.policy_record = policy_record
        self.wandb_run = wandb_run
        self.env = env
        self.s3_client = boto3.client("s3")

        self.step = 0
        self.grid_objects = []
        self.total_rewards = np.zeros(env.num_agents)
        self.replay = {
            "version": 1,
            "action_names": env.action_names(),
            "inventory_items": env.inventory_item_names(),
            "object_types": env.object_type_names(),
            "map_size": [env.map_width, env.map_height],
            "num_agents": env.num_agents,
            "max_steps": env._max_steps,
            "grid_objects": self.grid_objects,
        }

    def _add_sequence_key(self, grid_object: dict, key: str, step: int, value):
        """Add a key to the replay that is a sequence of values."""
        if key not in grid_object:
            # Add new key.
            grid_object[key] = [[step, value]]
        else:
            # Only add new entry if it has changed:
            if grid_object[key][-1][1] != value:
                grid_object[key].append([step, value])

    def log_pre_step(self, actions: np.ndarray):
        for i, grid_object in enumerate(self.env.grid_objects.values()):
            if len(self.grid_objects) <= i:
                self.grid_objects.append({})
            for key, value in grid_object.items():
                self._add_sequence_key(self.grid_objects[i], key, self.step, value)
            if "agent_id" in grid_object:
                agent_id = grid_object["agent_id"]
                self._add_sequence_key(self.grid_objects[i], "action", self.step, actions[agent_id].tolist())

    def log_post_step(self, rewards: np.ndarray):
        self.total_rewards += rewards
        for i, grid_object in enumerate(self.env.grid_objects.values()):
            if "agent_id" in grid_object:
                agent_id = grid_object["agent_id"]
                self._add_sequence_key(
                    self.grid_objects[i], "action_success", self.step, bool(self.env.action_success[agent_id])
                )
                self._add_sequence_key(self.grid_objects[i], "reward", self.step, rewards[agent_id].item())
                self._add_sequence_key(
                    self.grid_objects[i], "total_reward", self.step, self.total_rewards[agent_id].item()
                )
        self.step += 1

    def write_replay(self, replay_path: str, epoch: int = 0):
        self.replay["max_steps"] = self.step
        # Trim value changes to make them more compact.
        for grid_object in self.grid_objects:
            for key, changes in list(grid_object.items()):
                if isinstance(changes, list) and len(changes) == 1:
                    grid_object[key] = changes[0][1]

        replay_data = json.dumps(self.replay)  # Convert to JSON string
        replay_bytes = replay_data.encode("utf-8")  # Encode to bytes
        compressed_data = zlib.compress(replay_bytes)  # Compress the bytes

        # Make sure the directory exist
        if replay_path.startswith("s3://"):
            self._write_to_s3(compressed_data, replay_path, epoch)
        else:
            self._write_to_file(compressed_data, replay_path)

    def _write_to_file(self, replay_data: bytes, replay_path: str):
        os.makedirs(os.path.dirname(replay_path), exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.info(f"Wrote replay to {replay_path}")
        with open(replay_path, "wb") as f:
            f.write(replay_data)

    def _write_to_s3(self, replay_data: bytes, replay_url: str, epoch: int):
        """Upload the replay to S3 and log the link to WandB."""
        upload_to_s3(replay_data, replay_url, "application/x-compress")
        logger = logging.getLogger(__name__)
        logger.info(f"Uploaded replay to {replay_url}")
        if self.wandb_run is not None:
            player_url = "https://metta-ai.github.io/metta/?replayUrl=" + s3_url(replay_url)
            link_summary = {"replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Replay (Epoch {epoch})</a>')}
            self.wandb_run.log(link_summary)
