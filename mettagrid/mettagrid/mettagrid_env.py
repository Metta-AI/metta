# mettagrid/mettagrid_env.py
from __future__ import annotations

import copy
import os
import uuid
from pathlib import Path
from typing import Optional

import gym
import numpy as np
import pufferlib
from omegaconf import DictConfig, OmegaConf

from mettagrid.config.utils import simple_instantiate
from mettagrid.mettagrid_c import MettaGrid  # pylint: disable=E0611
from mettagrid.resolvers import register_resolvers
from mettagrid.stats_writer import StatsWriter


class MettaGridEnv(pufferlib.PufferEnv, gym.Env):
    def __init__(
        self,
        env_cfg: DictConfig,
        render_mode: Optional[str],
        env_map: Optional[np.ndarray] = None,
        buf=None,
        stats_writer_dir: Optional[str] = None,
        **kwargs,
    ):
        self._render_mode = render_mode
        self._cfg_template = env_cfg
        self._env_cfg = self._get_new_env_cfg()
        self._env_map = env_map
        self.should_reset = False
        self._renderer = None
        self._map_builder = None
        self._reset_env()
        self.labels = self._env_cfg.get("labels", None)

        self.stats_writer: Optional[StatsWriter] = None
        if stats_writer_dir:
            fname = f"stats_{os.getpid()}_{uuid.uuid4().hex[:6]}.duckdb"
            stats_writer_path = Path(stats_writer_dir) / fname
            self._writer_path = Path(stats_writer_path).resolve()
            self.stats_writer = StatsWriter(str(self._writer_path))
        else:
            self.stats_writer = None

        super().__init__(buf)

    def _get_new_env_cfg(self):
        env_cfg = OmegaConf.create(copy.deepcopy(self._cfg_template))
        OmegaConf.resolve(env_cfg)
        return env_cfg

    def _reset_env(self):
        if self._env_map is None:
            self._map_builder = simple_instantiate(
                self._env_cfg.game.map_builder,
                recursive=self._env_cfg.game.get("recursive_map_builder", True),
            )
            env_map = self._map_builder.build()
        else:
            env_map = self._env_map

        map_agents = np.count_nonzero(np.char.startswith(env_map, "agent"))
        assert self._env_cfg.game.num_agents == map_agents, (
            f"Number of agents {self._env_cfg.game.num_agents} does not match number of agents in map {map_agents}"
        )

        self._c_env = MettaGrid(self._env_cfg, env_map)
        self._grid_env = self._c_env
        self._num_agents = self._c_env.num_agents()

        env = self._grid_env

        self._env = env
        # self._env = RewardTracker(self._env)
        # self._env = FeatureMasker(self._env, self._cfg.hidden_features)

    def reset(self, seed=None, options=None):
        self._env_cfg = self._get_new_env_cfg()
        self._reset_env()

        self._c_env.set_buffers(self.observations, self.terminals, self.truncations, self.rewards)

        if self.stats_writer:
            self._episode_id = self.stats_writer.start_episode(
                env_name=self._env_cfg.name,
                seed=self._env_cfg.seed,
                map_w=self.map_width,
                map_h=self.map_height,
                meta=OmegaConf.to_container(self._env_cfg, resolve=False),
            )

    # ---------------------------------------------------------------------- #
    # Gym API                                                                #
    # ---------------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        self._reset_env()
        obs, infos = self._c_env.reset()
        return obs, infos

    def step(self, actions):
        self.actions[:] = np.array(actions).astype(np.uint32)
        self._c_env.step(self.actions)

        if self._env_cfg.normalize_rewards:
            self.rewards -= self.rewards.mean()

        infos = {}
        if self.terminals.all() or self.truncations.all():
            self.process_episode_stats(infos)
            self.should_reset = True

        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def process_episode_stats(self, infos: Dict[str, Any]):
        episode_rewards = self._c_env.get_episode_rewards()
        episode_rewards_sum = episode_rewards.sum()
        episode_rewards_mean = episode_rewards_sum / self._num_agents

        infos.update(
            {
                "episode/reward.sum": episode_rewards_sum,
                "episode/reward.mean": episode_rewards_mean,
                "episode/reward.min": episode_rewards.min(),
                "episode/reward.max": episode_rewards.max(),
                "episode_length": self._c_env.current_timestep(),
            }
        )

        if self._map_builder is not None and self._map_builder.labels is not None:
            for label in self._map_builder.labels:
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
            infos["agent"][n] = v / self._num_agents

        if self.stats_writer and self._episode_id is not None:
            for agent_idx, agent_stats in enumerate(stats["agent"]):
                self.stats_writer.log_metric(agent_idx, "reward", float(episode_rewards[agent_idx]))
                for k, v in agent_stats.items():
                    self.stats_writer.log_metric(agent_idx, k, float(v))
            self.stats_writer.end_episode(step_count=self._c_env.current_timestep())
        self._episode_id = None

    @property
    def _max_steps(self):
        return self._env_cfg.game.max_steps

    @property
    def single_observation_space(self):
        return self._env.observation_space

    @property
    def single_action_space(self):
        return self._env.action_space

    def action_names(self):
        return self._env.action_names()

    @property
    def player_count(self):
        return self._num_agents

    @property
    def num_agents(self):
        return self._num_agents

    def render(self):
        if self._renderer is None:
            return None

        return self._renderer.render(self._c_env.current_timestep(), self._c_env.grid_objects())

    @property
    def done(self):
        return self.should_reset

    @property
    def grid_features(self):
        return self._env.grid_features()

    @property
    def global_features(self):
        return []

    @property
    def render_mode(self):
        return self._render_mode

    @property
    def map_width(self):
        return self._c_env.map_width()

    @property
    def map_height(self):
        return self._c_env.map_height()

    @property
    def grid_objects(self):
        return self._c_env.grid_objects()

    @property
    def max_action_args(self):
        return self._c_env.max_action_args()

    @property
    def action_success(self):
        return np.asarray(self._c_env.action_success())

    def object_type_names(self):
        return self._c_env.object_type_names()

    def inventory_item_names(self):
        return self._c_env.inventory_item_names()

    def close(self):
        pass

    def __del__(self):
        if getattr(self, "stats_writer", None):
            self.stats_writer.close()


# Ensure resolvers are registered when this module is imported
register_resolvers()
