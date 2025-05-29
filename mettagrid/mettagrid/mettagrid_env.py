from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, Optional, cast

import gymnasium as gym
import numpy as np
import pufferlib
from omegaconf import OmegaConf
from pufferlib.utils import unroll_nested_dict
from typing_extensions import override

from mettagrid.curriculum import Curriculum
from mettagrid.level_builder import Level
from mettagrid.mettagrid_c import MettaGrid
from mettagrid.replay_writer import ReplayWriter
from mettagrid.stats_writer import StatsWriter
from mettagrid.util.hydra import simple_instantiate


def required(func):
    """Marks methods that PufferEnv requires but does not implement for override."""
    return func


class MettaGridEnv(pufferlib.PufferEnv, gym.Env):
    # Type hints for attributes defined in the C++ extension to help Pylance
    observations: np.ndarray
    terminals: np.ndarray
    truncations: np.ndarray
    rewards: np.ndarray
    actions: np.ndarray

    def __init__(
        self,
        curriculum: Curriculum,
        render_mode: Optional[str],
        level: Optional[Level] = None,
        buf=None,
        stats_writer: Optional[StatsWriter] = None,
        replay_writer: Optional[ReplayWriter] = None,
        **kwargs,
    ):
        if not isinstance(env_cfg, DictConfig):
            raise TypeError(f"env_cfg must be an OmegaConf DictConfig, got {type(env_cfg)}")
        if render_mode is not None and not isinstance(render_mode, str):
            raise TypeError(f"render_mode must be str or None, got {type(render_mode)}")
        if env_map is not None and not isinstance(env_map, np.ndarray):
            raise TypeError(f"env_map must be a numpy.ndarray, got {type(env_map)}")
        if stats_writer is not None and not isinstance(stats_writer, StatsWriter):
            raise TypeError("stats_writer must be a StatsWriter instance or None")
        if replay_writer is not None and not isinstance(replay_writer, ReplayWriter):
            raise TypeError("replay_writer must be a ReplayWriter instance or None")
        self._render_mode = render_mode
        self._curriculum = curriculum
        self._task = self._curriculum.get_task()
        self._level = level
        self._renderer = None
        self._map_labels = []
        self._stats_writer = stats_writer
        self._replay_writer = replay_writer
        self._episode_id = None
        self._reset_at = datetime.datetime.now()
        self._current_seed = 0

        self.labels = self._task.env_cfg().get("labels", None)
        self._should_reset = False

        self._reset_env()
        super().__init__(buf)

    def _make_episode_id(self):
        return str(uuid.uuid4())

    def _reset_env(self):
        # Prepare the level
        self._task = self._curriculum.get_task()
        level = self._level
        if level is None:
            map_builder = simple_instantiate(
                self._task.env_cfg().game.map_builder,
                recursive=self._task.env_cfg().game.get("recursive_map_builder", True),
            )
            level = map_builder.build()

        # Validate the level
        level_agents = np.count_nonzero(np.char.startswith(level.grid, "agent"))
        assert self._task.env_cfg().game.num_agents == level_agents, (
            f"Number of agents {self._task.env_cfg().game.num_agents} does not match number of agents in map {level_agents}"
        )

        # Convert to container for C++ code with explicit casting to Dict[str, Any]
        config_dict = cast(Dict[str, Any], OmegaConf.to_container(self._task.env_cfg()))

        self._map_labels = level.labels

        # Convert string array to list of strings for C++ compatibility
        # TODO: push the not-numpy-array higher up the stack, and consider pushing not-a-sparse-list lower.
        self._c_env = MettaGrid(config_dict, level.grid.tolist())

        self._grid_env = self._c_env

    @override
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        self._reset_env()

        self._c_env.set_buffers(self.observations, self.terminals, self.truncations, self.rewards)

        self._episode_id = self._make_episode_id()
        self._current_seed = seed or 0
        self._reset_at = datetime.datetime.now()
        if self._replay_writer:
            self._replay_writer.start_episode(self._episode_id, self)

        obs, infos = self._c_env.reset()
        self._should_reset = False
        return obs, infos

    @override
    def step(self, actions: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        self.actions[:] = np.array(actions).astype(np.uint32)

        if self._replay_writer:
            self._replay_writer.log_pre_step(self._episode_id, self.actions)

        self._c_env.step(self.actions)

        if self._replay_writer:
            self._replay_writer.log_post_step(self._episode_id, self.rewards)

        infos = {}
        if self.terminals.all() or self.truncations.all():
            self.process_episode_stats(infos)
            self._should_reset = True
            self._task.complete(self.rewards.mean())

        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def process_episode_stats(self, infos: Dict[str, Any]):
        episode_rewards = self._c_env.get_episode_rewards()
        episode_rewards_sum = episode_rewards.sum()
        episode_rewards_mean = episode_rewards_sum / self._c_env.num_agents

        infos.update(
            {
                "episode/reward.sum": episode_rewards_sum,
                "episode/reward.mean": episode_rewards_mean,
                "episode/reward.min": episode_rewards.min(),
                "episode/reward.max": episode_rewards.max(),
                "episode_length": self._c_env.current_step,
            }
        )

        for label in self._map_labels:
            infos.update(
                {
                    f"rewards/map:{label}": episode_rewards_mean,
                }
            )

        if self.labels is not None:
            for label in self.labels:
                infos.update(
                    {
                        f"rewards/env:{label}": episode_rewards_mean,
                    }
                )

        stats = self._c_env.get_episode_stats()

        infos["episode_rewards"] = episode_rewards
        infos["agent_raw"] = stats["agent"]
        infos["game"] = stats["game"]
        infos["agent"] = {}

        for agent_stats in stats["agent"]:
            for n, v in agent_stats.items():
                infos["agent"][n] = infos["agent"].get(n, 0) + v
        for n, v in infos["agent"].items():
            infos["agent"][n] = v / self._c_env.num_agents

        replay_url = None
        if self._replay_writer:
            assert self._episode_id is not None, "Episode ID must be set before writing a replay"
            replay_url = self._replay_writer.write_replay(self._episode_id)
        infos["replay_url"] = replay_url

        if self._stats_writer:
            assert self._episode_id is not None, "Episode ID must be set before writing stats"

            attributes = {
                "seed": self._current_seed,
                "map_w": self.map_width,
                "map_h": self.map_height,
            }

            for k, v in unroll_nested_dict(OmegaConf.to_container(self._task.env_cfg(), resolve=False)):
                attributes[f"config.{k.replace('/', '.')}"] = str(v)

            agent_metrics = {}
            for agent_idx, agent_stats in enumerate(stats["agent"]):
                agent_metrics[agent_idx] = {}
                agent_metrics[agent_idx]["reward"] = float(episode_rewards[agent_idx])
                for k, v in agent_stats.items():
                    agent_metrics[agent_idx][k] = float(v)

            self._stats_writer.record_episode(
                self._episode_id,
                attributes,
                agent_metrics,
                self.max_steps,
                replay_url,
                self._reset_at,
            )
        self._episode_id = None

    @override
    def close(self):
        pass

    @property
    def max_steps(self):
        return self._task.env_cfg().game.max_steps

    @property
    @required
    def single_observation_space(self):
        return self._c_env.observation_space

    @property
    @required
    def single_action_space(self):
        return self._c_env.action_space

    @property
    def action_names(self):
        return self._c_env.action_names()

    @property
    @required
    def num_agents(self):
        return self._c_env.num_agents

    def render(self):
        if self._renderer is None:
            return None

        return self._renderer.render(self._c_env.current_step, self._c_env.grid_objects())

    @property
    def done(self):
        return self._should_reset

    @property
    def grid_features(self):
        return self._c_env.grid_features()

    @property
    def global_features(self):
        return []

    @property
    def render_mode(self):
        return self._render_mode

    @property
    def map_width(self) -> int:
        return self._c_env.map_width

    @property
    def map_height(self) -> int:
        return self._c_env.map_height

    @property
    def grid_objects(self):
        return self._c_env.grid_objects()

    @property
    def max_action_args(self) -> list[int]:
        return self._c_env.max_action_args()

    @property
    def action_success(self):
        return np.asarray(self._c_env.action_success())

    @property
    def object_type_names(self):
        return self._c_env.object_type_names()

    @property
    def inventory_item_names(self):
        return self._c_env.inventory_item_names()
