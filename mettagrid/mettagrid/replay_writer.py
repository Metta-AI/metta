from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mettagrid.mettagrid_env import MettaGridEnv

import json
import zlib

import numpy as np

from mettagrid.util.file import http_url, write_data


class ReplayWriter:
    """Helper class for generating and uploading replays."""

    def __init__(self, replay_dir: str | None = None):
        self.replay_dir = replay_dir
        self.episodes = {}

    def start_episode(self, episode_id: str, env: MettaGridEnv):
        self.episodes[episode_id] = EpisodeReplay(env)

    def log_pre_step(self, episode_id: str, actions: np.ndarray):
        self.episodes[episode_id].log_pre_step(actions)

    def log_post_step(self, episode_id: str, rewards: np.ndarray):
        self.episodes[episode_id].log_post_step(rewards)

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

    def _add_sequence_key(self, grid_object: dict, key: str, step: int, value):
        """Add a key to the replay that is a sequence of values."""
        if key not in grid_object:
            # Add new key.
            grid_object[key] = [[step, value]]
        else:
            # Only add new entry if it has changed:
            if grid_object[key][-1][1] != value:
                grid_object[key].append([step, value])

    def get_replay_data(self):
        """Gets full replay as a tree of plain python dictionaries."""
        self.replay_data["max_steps"] = self.step
        # Trim value changes to make them more compact.
        for grid_object in self.grid_objects:
            for key, changes in list(grid_object.items()):
                if isinstance(changes, list) and len(changes) == 1:
                    grid_object[key] = changes[0][1]
        # Store the env config.
        self.replay_data["config"] = self.env.config
        return self.replay_data

    def write_replay(self, path: str):
        """Writes a replay to a file."""
        replay_data = json.dumps(self.get_replay_data())  # Convert to JSON string
        replay_bytes = replay_data.encode("utf-8")  # Encode to bytes
        compressed_data = zlib.compress(replay_bytes)  # Compress the bytes

        write_data(path, compressed_data, content_type="application/x-compress")
