from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from metta.mettagrid.mettagrid_env import MettaGridEnv

import json
import zlib

import numpy as np

from metta.mettagrid.util.file import http_url, write_data


class ReplayWriter:
    """Helper class for generating and uploading replays."""

    def __init__(self, replay_dir: str | None = None):
        self.replay_dir = replay_dir
        self.episodes = {}

    def start_episode(self, episode_id: str, env: MettaGridEnv):
        self.episodes[episode_id] = EpisodeReplay(env)

    def log_step(self, episode_id: str, actions: np.ndarray, rewards: np.ndarray):
        self.episodes[episode_id].log_step(actions, rewards)

    def write_replay(self, episode_id: str) -> str | None:
        """Write the replay to the replay directory and return the URL."""
        if self.replay_dir is None:
            return None
        episode_replay = self.episodes[episode_id]
        if episode_replay is None:
            raise ValueError(f"Episode {episode_id} not found")
        replay_path = f"{self.replay_dir}/{episode_id}.json.z"
        episode_replay.write_replay(replay_path)
        return http_url(replay_path)


class EpisodeReplay:
    def __init__(self, env: MettaGridEnv):
        self.env = env
        self.step = 0
        self.grid_objects = []
        self.total_rewards = np.zeros(env.num_agents)
        self.replay_data = {
            "version": 1,
            "action_names": env.action_names,
            "inventory_items": env.inventory_item_names,
            "object_types": env.object_type_names,
            "map_size": [env.map_width, env.map_height],
            "num_agents": env.num_agents,
            "max_steps": env.max_steps,
            "grid_objects": self.grid_objects,
        }

    def log_step(self, actions: np.ndarray, rewards: np.ndarray):
        self.total_rewards += rewards
        for i, grid_object in enumerate(self.env.grid_objects.values()):
            update_object = grid_object.copy()
            if len(self.grid_objects) <= i:
                self.grid_objects.append({})
            if "agent_id" in grid_object:
                agent_id = update_object["agent_id"]
                update_object["action"] = actions[agent_id].tolist()
                update_object["action_success"] = bool(self.env.action_success[agent_id])
                update_object["reward"] = rewards[agent_id].item()
                update_object["total_reward"] = self.total_rewards[agent_id].item()
            self._seq_key_merge(self.grid_objects[i], self.step, update_object)
        self.step += 1

    def _seq_key_merge(self, grid_object: dict, step: int, update_object: dict):
        """Add a sequence keys to replay grid object."""
        for key, value in update_object.items():
            if key not in grid_object:
                # Add new key.
                if step == 0:
                    grid_object[key] = [[step, value]]
                else:
                    grid_object[key] = [[0, 0], [step, value]]
            else:
                # Only add new entry if it has changed:
                if grid_object[key][-1][1] != value:
                    grid_object[key].append([step, value])
        # If key has vanished, add a zero entry.
        for key in grid_object.keys():
            if key not in update_object:
                if grid_object[key][-1][1] != 0:
                    grid_object[key].append([step, 0])

    def get_replay_data(self):
        """Gets full replay as a tree of plain python dictionaries."""
        self.replay_data["max_steps"] = self.step
        # Trim value changes to make them more compact.
        for grid_object in self.grid_objects:
            for key, changes in list(grid_object.items()):
                if isinstance(changes, list) and len(changes) == 1:
                    grid_object[key] = changes[0][1]

        self.replay_data["config"] = OmegaConf.to_container(self.env._task.env_cfg(), resolve=True)

        return self.replay_data

    def write_replay(self, path: str):
        """Writes a replay to a file."""
        replay_data = json.dumps(self.get_replay_data())  # Convert to JSON string
        replay_bytes = replay_data.encode("utf-8")  # Encode to bytes
        compressed_data = zlib.compress(replay_bytes)  # Compress the bytes

        write_data(path, compressed_data, content_type="application/x-compress")
