from __future__ import annotations

from typing import TYPE_CHECKING

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

        if not (replay_path.startswith("s3://") or replay_path.startswith("https://")):
            return None

        return http_url(replay_path)


class EpisodeReplay:
    def __init__(self, env: MettaGridEnv):
        self.env = env
        self.step = 0
        self.objects = []
        self.total_rewards = np.zeros(env.num_agents)
        self.replay_data = {
            "version": 2,
            "action_names": env.action_names,
            "item_names": env.inventory_item_names,
            "type_names": env.object_type_names,
            "map_size": [env.map_width, env.map_height],
            "num_agents": env.num_agents,
            "max_steps": env.max_steps,
            "objects": self.objects,
        }

    def inventory_format(self, inventory: dict) -> list:
        result = []
        for item_id, amount in inventory.items():
            result.append([item_id, amount])
        return result

    def log_step(self, actions: np.ndarray, rewards: np.ndarray):
        self.total_rewards += rewards
        for i, grid_object in enumerate(self.env.grid_objects.values()):
            if len(self.objects) <= i:
                self.objects.append({})

            update_object = {}
            update_object["id"] = grid_object["id"]
            update_object["type_id"] = grid_object["type_id"]
            update_object["location"] = grid_object["location"]
            update_object["orientation"] = grid_object.get("orientation", 0)
            update_object["inventory"] = self.inventory_format(grid_object.get("inventory", {}))
            update_object["inventory_max"] = grid_object.get("inventory_max", 0)
            update_object["color"] = grid_object.get("color", 0)
            update_object["is_swappable"] = grid_object.get("is_swappable", False)

            if "agent_id" in grid_object:
                agent_id = grid_object["agent_id"]
                update_object["agent_id"] = agent_id
                update_object["is_agent"] = True
                update_object["vision_size"] = 11  # TODO: Waiting for env to support this
                update_object["action_id"] = int(actions[agent_id][0])
                update_object["action_param"] = int(actions[agent_id][1])
                update_object["action_success"] = bool(self.env.action_success[agent_id])
                update_object["current_reward"] = rewards[agent_id].item()
                update_object["total_reward"] = self.total_rewards[agent_id].item()
                update_object["freeze_remaining"] = grid_object["freeze_remaining"]
                update_object["is_frozen"] = grid_object["is_frozen"]
                update_object["freeze_duration"] = grid_object["freeze_duration"]
                update_object["group_id"] = grid_object["group_id"]

            elif "input_resources" in grid_object:
                update_object["input_resources"] = self.inventory_format(grid_object["input_resources"])
                update_object["output_resources"] = self.inventory_format(grid_object["output_resources"])
                update_object["output_limit"] = grid_object["output_limit"]
                update_object["conversion_remaining"] = 0  # TODO: Waiting for env to support this
                update_object["is_converting"] = grid_object["is_converting"]
                update_object["conversion_duration"] = grid_object["conversion_duration"]
                update_object["cooldown_remaining"] = 0  # TODO: Waiting for env to support this
                update_object["is_cooling_down"] = grid_object["is_cooling_down"]
                update_object["cooldown_duration"] = grid_object["cooldown_duration"]

            self._seq_key_merge(self.objects[i], self.step, update_object)
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
        for grid_object in self.objects:
            for key, changes in list(grid_object.items()):
                if isinstance(changes, list) and len(changes) == 1:
                    grid_object[key] = changes[0][1]

        return self.replay_data

    def write_replay(self, path: str):
        """Writes a replay to a file."""

        replay_data = json.dumps(self.get_replay_data())  # Convert to JSON string
        replay_bytes = replay_data.encode("utf-8")  # Encode to bytes
        compressed_data = zlib.compress(replay_bytes)  # Compress the bytes

        write_data(path, compressed_data, content_type="application/x-compress")
