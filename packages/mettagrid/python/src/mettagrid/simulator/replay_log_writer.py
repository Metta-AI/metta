from __future__ import annotations

import gzip
import json
import logging
import uuid
import zlib
from typing import Any, Dict, List

import numpy as np

from mettagrid.renderer.common import METTASCOPE_REPLAY_URL_PREFIX
from mettagrid.simulator.interface import SimulatorEventHandler
from mettagrid.simulator.simulator import Simulation
from mettagrid.util.file import http_url, write_data
from mettagrid.util.grid_object_formatter import format_grid_object

logger = logging.getLogger("ReplayLogWriter")


class InMemoryReplayWriter(SimulatorEventHandler):
    """EventHandler that maintains a list of completed replay results in memory, and does not write them anywhere"""

    def __init__(self):
        super().__init__()
        self._episode_replay: EpisodeReplay
        self._completed_replays: list[EpisodeReplay] = []

    def on_episode_start(self) -> None:
        self._episode_replay = EpisodeReplay(self._sim)

    def get_completed_replays(self) -> list[EpisodeReplay]:
        return self._completed_replays

    def on_step(self) -> None:
        self._episode_replay.log_step(
            self._sim.current_step,
            self._sim._c_sim.actions(),  # type: ignore[attr-defined]
            self._sim._c_sim.rewards(),  # type: ignore[attr-defined]
        )

    def on_episode_end(self) -> None:
        self._completed_replays.append(self._episode_replay)


class ReplayLogWriter(InMemoryReplayWriter):
    """EventHandler that writes replay logs to storage (S3 or local files)."""

    def __init__(self, replay_dir: str):
        """Initialize ReplayLogWriter.

        Args:
            replay_dir: Local directory where replays will be written. Must exist.
        """
        super().__init__()
        self._replay_dir = replay_dir
        self._episode_id: str
        self.episodes: Dict[str, EpisodeReplay] = {}
        self._episode_urls: Dict[str, str] = {}
        self._episode_paths: Dict[str, str] = {}

    def on_episode_start(self) -> None:
        """Start recording a new episode."""
        self._episode_id = str(uuid.uuid4())
        self._episode_replay = EpisodeReplay(self._sim)
        self.episodes[self._episode_id] = self._episode_replay
        logger.debug("Started recording episode %s", self._episode_id)

    def on_episode_end(self) -> None:
        """Write the replay to storage and clean up."""
        replay_path = f"{self._replay_dir}/{self._episode_id}.json.z"
        self._episode_replay.write_replay(replay_path)
        url = http_url(replay_path)
        self._episode_urls[self._episode_id] = url
        self._episode_paths[self._episode_id] = replay_path
        self._sim._context["replay_url"] = url
        logger.info("Wrote replay for episode %s to %s", self._episode_id, url)
        logger.debug("Watch replay at %s", METTASCOPE_REPLAY_URL_PREFIX + url)
        logger.debug(
            "Watch locally: " + f"nim r packages/mettagrid/nim/mettascope/src/mettascope.nim --replay={replay_path}"
        )

    def get_written_replay_urls(self) -> Dict[str, str]:
        """Return URLs for every replay file that has been written to disk."""
        return dict(self._episode_urls)

    def get_written_replay_paths(self) -> List[str]:
        """Return file paths for every replay file that has been written to disk."""
        return list(self._episode_paths.values())


class EpisodeReplay:
    """Helper class for managing replay data for a single episode."""

    # Object types that never change state and only need to be recorded once
    STATIC_OBJECT_TYPES = frozenset({"wall"})

    def __init__(self, sim: Simulation):
        self.sim = sim
        self.step = 0
        self.objects: list[dict[str, Any]] = []
        self.total_rewards = np.zeros(sim.num_agents)
        # Map object IDs to their index in self.objects for consistent ordering
        self._object_id_to_index: dict[int, int] = {}
        self.set_compression("zlib")

        self._validate_non_empty_string_list(sim.action_names, "action_names")
        self._validate_non_empty_string_list(sim.resource_names, "item_names")

        self.replay_data = {
            "version": 3,
            "action_names": sim.action_names,
            "item_names": sim.resource_names,
            "type_names": sim.object_type_names,
            "map_size": [sim.map_width, sim.map_height],
            "num_agents": sim.num_agents,
            "max_steps": sim.config.game.max_steps,
            "mg_config": sim.config.model_dump(mode="json"),
            "objects": self.objects,
        }

    def set_compression(self, compression: str):
        if compression == "zlib":
            self._compression = zlib.compress
            self._content_type = "application/x-compress"
        elif compression == "gzip":
            self._compression = gzip.compress
            self._content_type = "application/gzip"
        else:
            raise ValueError(f"unknown compression {compression!r}, try 'zlib' or 'gzip'")

    def log_step(self, current_step: int, actions: np.ndarray, rewards: np.ndarray):
        """Log a single step of the episode."""
        self.total_rewards += rewards

        # On first step, get ALL objects (including walls) to set up the replay
        # On subsequent steps, use ignore_types to skip static objects at the C++ level
        if self.step == 0:
            grid_objects = self.sim.grid_objects()
        else:
            grid_objects = self.sim.grid_objects(ignore_types=list(self.STATIC_OBJECT_TYPES))

        for obj_id, grid_object in grid_objects.items():
            # Use object ID as index for consistent ordering
            idx = self._object_id_to_index.get(obj_id)
            if idx is None:
                idx = len(self.objects)
                self._object_id_to_index[obj_id] = idx
                self.objects.append({})

            update_object = format_grid_object(
                grid_object,
                actions,
                self.sim.action_success,
                rewards,
                self.total_rewards,
            )

            self._seq_key_merge(self.objects[idx], self.step, update_object)
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
                if (
                    isinstance(changes, list)
                    and len(changes) == 1
                    and isinstance(changes[0], (list, tuple))
                    and len(changes[0]) == 2
                ):
                    grid_object[key] = changes[0][1]

        return self.replay_data

    def write_replay(self, path: str):
        """Writes a replay to a file."""
        replay_data = json.dumps(self.get_replay_data())  # Convert to JSON string
        replay_bytes = replay_data.encode("utf-8")  # Encode to bytes
        compressed_data = self._compression(replay_bytes)  # Compress the bytes

        write_data(path, compressed_data, content_type=self._content_type)

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
