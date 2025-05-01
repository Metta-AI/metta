import logging
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import pufferlib

from mettagrid.mettagrid.config.utils import simple_instantiate
from mettagrid.mettagrid.curriculum.curriculum import Curriculum
from mettagrid.mettagrid_c import MettaGrid  # pylint: disable=E0611
from mettagrid.resolvers import register_resolvers

logger = logging.getLogger("mettagrid")


class MettaGridEnv(pufferlib.PufferEnv, gym.Env):
    def __init__(
        self,
        curriculum: Curriculum,
        render_mode: Optional[str],
        buf=None,
    ):
        self._render_mode = render_mode
        self._curriculum = curriculum
        self._renderer = None
        self._reset_env()

        super().__init__(buf)

    def _reset_env(self):
        self._task = self._curriculum.get_task()
        game_cfg = self._task.env_cfg().game
        map_builder = simple_instantiate(
            game_cfg.map_builder,
            recursive=game_cfg.get("recursive_map_builder", True),
        )
        game_map = map_builder.build()
        map_agents = np.count_nonzero(np.char.startswith(game_map, "agent"))
        assert game_cfg.num_agents == map_agents, f"Map has {map_agents} agents, expected {game_cfg.num_agents}"

        logger.info(f"Resetting environment with {game_cfg.agent.freeze_duration} freeze duration")
        self._c_env = MettaGrid(game_cfg, game_map)
        self._grid_env = self._c_env
        self._num_agents = self._c_env.num_agents()
        self._env = self._grid_env
        self._should_reset = False

    def reset(self, seed=None, options=None):
        self._reset_env()
        self._c_env.set_buffers(self.observations, self.terminals, self.truncations, self.rewards)
        return self._c_env.reset()

    def step(self, actions):
        self.actions[:] = np.array(actions).astype(np.uint32)
        self._c_env.step(self.actions)

        infos = {}
        if self.terminals.all() or self.truncations.all():
            self.process_episode_stats(infos)
            self._task.complete(infos["episode_rewards"].sum())
            self._should_reset = True

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

        # xcxc
        # if self._map_builder is not None and self._map_builder.labels is not None:
        #     for label in self._map_builder.labels:
        #         infos.update(
        #             {
        #                 f"rewards/map:{label}": episode_rewards_mean,
        #             }
        #         )

        # if self.labels is not None:
        #     for label in self.labels:
        #         infos.update(
        #             {
        #                 f"rewards/env:{label}": episode_rewards_mean,
        #             }
        #         )

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

    @property
    def _max_steps(self):
        return self._task.game_cfg().max_steps

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
        return self._should_reset

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


# Ensure resolvers are registered when this module is imported
register_resolvers()
