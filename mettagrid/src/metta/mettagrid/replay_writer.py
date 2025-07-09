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
        self.objects = []
        self.total_rewards = np.zeros(env.num_agents)
        self.replay_data = {
            "version": 2,
            "num_agents": env.num_agents,
            "max_steps": env.max_steps,
            "map_size": [env.map_width, env.map_height],
            "type_names": env.object_type_names,
            "action_names": env.action_names,
            "item_names": env.inventory_item_names,
            "group_names": [],
            "objects": self.objects,
        }

    def log_step(self, actions: np.ndarray, rewards: np.ndarray):
        self.total_rewards += rewards
        for i, grid_object in enumerate(self.env.grid_objects.values()):
            update_object = {}
            update_object["id"] = grid_object["id"]
            update_object["type_id"] = grid_object["type_id"]
            if "agent_id" in grid_object:
                update_object["agent_id"] = grid_object["agent_id"]
            update_object["position"] = [grid_object["c"], grid_object["r"]]
            if "rotation" in grid_object:
                update_object["rotation"] = grid_object["rotation"]
            update_object["layer"] = grid_object["layer"]
            if "group_id" in grid_object:
                update_object["group_id"] = grid_object["group_id"]
                # If there are not group names, make some up.
                while len(self.replay_data["group_names"]) <= grid_object["group_id"]:
                    self.replay_data["group_names"].append("group_" + str(len(self.replay_data["group_names"])))
            inventory = []
            for key in grid_object:
                if key.startswith("inv:") or key.startswith("agent:inv:"):
                    name = key.split(":")[-1]
                    inventory.append(self.env.inventory_item_names.index(name))
            update_object["inventory"] = inventory
            if "color" in grid_object:
                update_object["color"] = grid_object["color"]

            if len(self.objects) <= i:
                self.objects.append({})

            if "agent_id" in grid_object:
                agent_id = update_object["agent_id"]
                action_tuple = actions[agent_id].tolist()
                update_object["action_id"] = action_tuple[0]
                update_object["action_parameter"] = action_tuple[1]
                update_object["action_success"] = bool(self.env.action_success[agent_id])
                update_object["current_reward"] = rewards[agent_id].item()
                update_object["total_reward"] = self.total_rewards[agent_id].item()
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
        for obj in self.objects:
            for key, changes in list(obj.items()):
                if isinstance(changes, list) and len(changes) == 1:
                    obj[key] = changes[0][1]

        self.replay_data["config"] = OmegaConf.to_container(self.env._task.env_cfg(), resolve=True)

        # # FIX ME: This is a hack as map_width and map_height are not set in the replay_writer.
        # # Go over all objects and find the max x and y and set that as the map size.
        # max_x = 0
        # max_y = 0
        # for obj in self.objects:
        #     for key, changes in list(obj.items()):
        #         if key == "position":
        #             max_x = max(max_x, changes[-1][1][0])
        #             max_y = max(max_y, changes[-1][1][1])
        # self.replay_data["map_size"] = [max_x + 1, max_y + 1]

        return self.replay_data

    def write_replay(self, path: str):
        """Writes a replay to a file."""
        data = self.get_replay_data()
        data["file_name"] = path.split("/")[-1]
        replay_data = json.dumps(data)  # Convert to JSON string
        replay_bytes = replay_data.encode("utf-8")  # Encode to bytes
        compressed_data = zlib.compress(replay_bytes)  # Compress the bytes
        write_data(path, compressed_data, content_type="application/x-compress")
