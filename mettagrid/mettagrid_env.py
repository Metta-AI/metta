from typing import Any, Dict

import numpy as np
import pufferlib
from omegaconf import OmegaConf

from mettagrid.config.game_builder import MettaGridGameBuilder
from mettagrid.config.sample_config import sample_config
from mettagrid.mettagrid_c import MettaGrid # pylint: disable=E0611
import gymnasium as gym

class MettaGridEnv(pufferlib.PufferEnv, gym.Env):
    def __init__(self, render_mode: str, buf=None, **cfg):

        self._render_mode = render_mode
        self._cfg = OmegaConf.create(cfg)
        self.make_env()
        self.should_reset = False
        self._renderer = None

        super().__init__(buf)

    def make_env(self):
        scfg = sample_config(self._cfg, self._cfg.sampling)
        assert isinstance(scfg, Dict)
        self._env_cfg = OmegaConf.create(scfg)
        self._game_builder = MettaGridGameBuilder(**self._env_cfg.game) # type: ignore
        level = self._game_builder.level()
        self._c_env = MettaGrid(self._env_cfg, level)
        self._grid_env = self._c_env
        self._num_agents = self._c_env.num_agents()

        env = self._grid_env

        self._env = env
        #self._env = RewardTracker(self._env)
        #self._env = FeatureMasker(self._env, self._cfg.hidden_features)

    def reset(self, seed=None, options=None):
        self.make_env()

        self._c_env.set_buffers(
            self.observations,
            self.terminals,
            self.truncations,
            self.rewards)

        # obs, infos = self._env.reset(**kwargs)
        # self._compute_max_energy()
        # return obs, infos
        obs, infos = self._c_env.reset()
        self.should_reset = False
        return obs, infos

    def step(self, actions):
        self.actions[:] = np.array(actions).astype(np.int32)
        self._c_env.step(self.actions)

        if self._cfg.normalize_rewards:
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
        infos.update({
            "episode/reward.sum": episode_rewards_sum,
            "episode/reward.mean": episode_rewards_mean,
            "episode/reward.min": episode_rewards.min(),
            "episode/reward.max": episode_rewards.max(),
            "episode_length": self._c_env.current_timestep(),
        })
        stats = self._c_env.get_episode_stats()

        infos["episode_rewards"] = 1000 * episode_rewards / self._env_cfg.game.max_steps
        infos["agent_raw"] = stats["agent"]
        infos["game"] = stats["game"]
        infos["agent"] = {}

        for agent_stats in stats["agent"]:
            for n, v in agent_stats.items():
                infos["agent"][n] = infos["agent"].get(n, 0) + v
        for n, v in infos["agent"].items():
            infos["agent"][n] = v / self._num_agents

    def _compute_max_energy(self):
        pass

    @property
    def _max_steps(self):
        return self._game_builder.max_steps

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

        return self._renderer.render(
            self._c_env.current_timestep(),
            self._c_env.grid_objects()
        )

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

    def close(self):
        pass
