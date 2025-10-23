"""ReplayLogRenderer - A renderer that writes replay logs for game episodes."""

from __future__ import annotations

import json
import logging
import uuid
import zlib
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from metta.utils.file import http_url, write_data
from mettagrid.renderer.renderer import Renderer
from mettagrid.util.action_catalog import build_action_mapping, make_decode_fn
from mettagrid.util.grid_object_formatter import format_grid_object

if TYPE_CHECKING:
    from mettagrid import MettaGridEnv

logger = logging.getLogger("ReplayLogRenderer")


class ReplayLogRenderer(Renderer):
    """Renderer that writes replay logs to storage (S3 or local files)."""

    def __init__(self, replay_dir: str):
        """Initialize ReplayLogRenderer.

        Args:
            replay_dir: S3 path or local directory where replays will be written.
                       If None, replay writing is disabled.
            episode_id: Unique identifier for the episode
        """
        self._replay_dir = replay_dir
        self._episode_id = None
        self._episode_replay = None
        self._should_continue = True
        self.episodes: Dict[str, EpisodeReplay] = {}

    def on_episode_start(self, env: MettaGridEnv) -> None:
        """Start recording a new episode."""
        self._episode_id = str(uuid.uuid4())
        self._episode_replay = EpisodeReplay(env)
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
        self._episode_replay.log_step(actions, rewards)

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

    def render(self) -> None:
        """Render the current state."""
        pass


class EpisodeReplay:
    """Helper class for managing replay data for a single episode."""

    def __init__(self, env: MettaGridEnv):
        self.env = env
        self.step = 0
        self.objects = []
        self.total_rewards = np.zeros(env.num_agents)
        self._flat_action_mapping, self._base_action_names = build_action_mapping(env)
        self._decode_flat_action = make_decode_fn(self._flat_action_mapping)

        self._validate_non_empty_string_list(env.action_names, "action_names")
        self._validate_non_empty_string_list(env.resource_names, "item_names")

        self.replay_data = {
            "version": 2,
            "action_names": self._base_action_names,
            "item_names": env.resource_names,
            "type_names": env.object_type_names,
            "map_size": [env.map_width, env.map_height],
            "num_agents": env.num_agents,
            "max_steps": env.max_steps,
            "mg_config": env.mg_config.model_dump(mode="json"),
            "objects": self.objects,
        }

    def log_step(self, actions: np.ndarray, rewards: np.ndarray):
        """Log a single step of the episode."""
        self.total_rewards += rewards
        for i, grid_object in enumerate(self.env.grid_objects().values()):
            if len(self.objects) <= i:
                self.objects.append({})

            update_object = format_grid_object(
                grid_object,
                actions,
                self.env.action_success,
                rewards,
                self.total_rewards,
                decode_flat_action=self._decode_flat_action,
            )

            self._seq_key_merge(self.objects[i], self.step, update_object)
        self.step += 1

    @staticmethod
    def _is_step_sequence(history: Any) -> bool:
        """Return True when history is a list of [step, value] pairs."""
        if not isinstance(history, list):
            return False
        if not history:
            return False
        tail = history[-1]
        return isinstance(tail, (list, tuple)) and len(tail) >= 2

    def _ensure_history(self, grid_object: dict, key: str, step: int, value: Any) -> None:
        """Normalize history so future appends always operate on [step, value] pairs."""
        if key not in grid_object:
            if step == 0:
                grid_object[key] = [[step, value]]
            else:
                grid_object[key] = [[0, 0], [step, value]]
            return

        history = grid_object[key]
        if self._is_step_sequence(history):
            return

        previous = history
        if isinstance(previous, list) and previous:
            candidate = previous[-1]
            if isinstance(candidate, (list, tuple)) and len(candidate) >= 2:
                previous = candidate[-1]
        grid_object[key] = [[0, previous]]
        if step != 0:
            grid_object[key].append([step, value])

    def _seq_key_merge(self, grid_object: dict, step: int, update_object: dict):
        """Add a sequence keys to replay grid object."""
        for key, value in update_object.items():
            self._ensure_history(grid_object, key, step, value)
            if self._is_step_sequence(grid_object[key]):
                if grid_object[key][-1][1] != value:
                    grid_object[key].append([step, value])

        # If key has vanished, add a zero entry.
        for key in list(grid_object.keys()):
            current_value = (
                grid_object[key][-1][1]
                if self._is_step_sequence(grid_object[key])
                else grid_object[key]
            )
            self._ensure_history(grid_object, key, step, current_value)
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
