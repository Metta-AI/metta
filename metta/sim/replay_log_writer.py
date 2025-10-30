from __future__ import annotations

import json
import logging
import uuid
import zlib
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from metta.utils.file import http_url, write_data
from mettagrid.simulator import SimulatorEventHandler
from mettagrid.simulator.simulator import Simulation
from mettagrid.util.grid_object_formatter import format_grid_object

if TYPE_CHECKING:
    pass

logger = logging.getLogger("ReplayLogWriter")


class ReplayLogWriter(SimulatorEventHandler):
    """EventHandler that writes replay logs to storage (S3 or local files)."""

    def __init__(self, replay_dir: str):
        """Initialize ReplayLogWriter.

        Args:
            replay_dir: Local directory where replays will be written.
        """
        self._replay_dir = replay_dir
        self._episode_id = None
        self._episode_replay = None
        self._should_continue = True
        self.episodes: Dict[str, EpisodeReplay] = {}

    def on_episode_start(self) -> None:
        """Start recording a new episode."""
        assert self._sim is not None
        self._episode_id = str(uuid.uuid4())
        self._episode_replay = EpisodeReplay(self._sim)
        self.episodes[self._episode_id] = self._episode_replay
        logger.info("Started recording episode %s", self._episode_id)

    def on_step(
        self,
        current_step: int,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        infos: Dict[str, Any],
    ) -> None:
        """Log a single step in the replay."""
        assert self._episode_replay is not None
        self._episode_replay.log_step(current_step, actions, rewards)

    def should_continue(self) -> bool:
        """Check if rendering should continue."""
        return self._should_continue

    def on_episode_end(self, infos: Dict[str, Any]) -> None:
        """Write the replay to storage and clean up."""
        assert self._episode_replay is not None
        replay_path = f"{self._replay_dir}/{self._episode_id}.json.z"
        self._episode_replay.write_replay(replay_path)
        url = http_url(replay_path)
        infos["replay_url"] = url
        logger.info("Wrote replay for episode %s to %s", self._episode_id, url)


class EpisodeReplay:
    """Helper class for managing replay data for a single episode."""

    def __init__(self, sim: Simulation):
        self.sim = sim
        self.step = 0
        self.objects = []
        self.total_rewards = np.zeros(sim.num_agents)

        self._validate_non_empty_string_list(sim.action_names, "action_names")
        self._validate_non_empty_string_list(sim.resource_names, "item_names")

        self.replay_data = {
            "version": 2,
            "action_names": sim.action_names,
            "item_names": sim.resource_names,
            "type_names": sim.object_type_names,
            "map_size": [sim.map_width, sim.map_height],
            "num_agents": sim.num_agents,
            "max_steps": sim.config.game.max_steps,
            "mg_config": sim.config.model_dump(mode="json"),
            "objects": self.objects,
        }

    def log_step(self, current_step: int, actions: np.ndarray, rewards: np.ndarray):
        """Log a single step of the episode."""
        self.total_rewards += rewards
        for i, grid_object in enumerate(self.sim.grid_objects().values()):
            if len(self.objects) <= i:
                self.objects.append({})

            update_object = format_grid_object(
                grid_object,
                actions,
                self.sim.action_success,
                rewards,
                self.total_rewards,
            )

            self._seq_key_merge(self.objects[i], self.step, update_object)
        self.step += 1
        if current_step != self.step:
            raise ValueError(
                f"Writing multiple steps at once: step {current_step} != Replay step {self.step}."
                "Probably a vecenv issue."
            )

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

    @staticmethod
    def _validate_non_empty_string_list(values: list[str], field_name: str) -> None:
        """Ensure the provided iterable is a list of strings, warn on empty strings with index."""
        if not isinstance(values, list):
            raise ValueError(f"{field_name} must be a list of strings, got {type(values)}")
        for index, value in enumerate(values):
            if not isinstance(value, str):
                raise ValueError(f"{field_name}[{index}] must be a string, got {type(value)}: {repr(value)}")
            if value == "":
                logger.warning(
                    (
                        "%s contains an empty string at index %d; "
                        "frontend tolerates empty names but backend discourages them"
                    ),
                    field_name,
                    index,
                )
