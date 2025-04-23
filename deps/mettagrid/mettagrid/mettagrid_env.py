import copy
from typing import Any, Dict, Optional

import gymnasium as gym
import hydra
import numpy as np
import pufferlib
from omegaconf import DictConfig, OmegaConf

from mettagrid.mettagrid_c import MettaGrid  # pylint: disable=E0611
from mettagrid.resolvers import register_resolvers


class MettaGridEnv(pufferlib.PufferEnv, gym.Env):
    def __init__(self, env_cfg: DictConfig, render_mode: Optional[str], buf=None, **kwargs):
        self._render_mode = render_mode
        self._cfg_template = env_cfg
        self._env_cfg = self._get_new_env_cfg()
        self._reset_env()
        self.should_reset = False
        self._renderer = None

        super().__init__(buf)

    def _get_new_env_cfg(self):
        env_cfg = OmegaConf.create(copy.deepcopy(self._cfg_template))
        OmegaConf.resolve(env_cfg)
        return env_cfg

    def _reset_env(self):
        self._map_builder = hydra.utils.instantiate(
            self._env_cfg.game.map_builder,
            _recursive_=self._env_cfg.game.recursive_map_builder,
        )
        env_map = self._map_builder.build()
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

        # obs, infos = self._env.reset(**kwargs)
        # return obs, infos
        obs, infos = self._c_env.reset()
        self.should_reset = False
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

    def close(self):
        pass


class MettaGridEnvSet(MettaGridEnv):
    """
    This is a wrapper around MettaGridEnv that allows for multiple environments to be used for training.
    """

    def __init__(
        self,
        env_cfg: DictConfig,
        render_mode: str,
        buf=None,
        **kwargs,
    ):
        self._envs = list(env_cfg.envs.keys())
        self._probabilities = list(env_cfg.envs.values())
        self._num_agents_global = env_cfg.num_agents
        self._env_cfg = self._get_new_env_cfg()

        super().__init__(env_cfg, render_mode, buf, **kwargs)
        self._cfg_template = None  # we don't use this with multiple envs, so we clear it to emphasize that fact

    def _get_new_env_cfg(self):
        selected_env = np.random.choice(self._envs, p=self._probabilities)
        env_cfg = config_from_path(selected_env)
        if self._num_agents_global != env_cfg.game.num_agents:
            raise ValueError(
                "For MettaGridEnvSet, the number of agents must be the same for all environments. "
                f"Global: {self._num_agents_global}, Env: {env_cfg.game.num_agents}"
            )
        env_cfg = OmegaConf.create(env_cfg)
        OmegaConf.resolve(env_cfg)
        return env_cfg


def make_env_from_cfg(cfg_path: str, *args, **kwargs):
    cfg = OmegaConf.load(cfg_path)
    env = MettaGridEnv(cfg, *args, **kwargs)
    return env


def config_from_path(config_path: str) -> DictConfig:
    env_cfg = hydra.compose(config_name=config_path)

    # when hydra loads a config, it "prefixes" the keys with the path of the config file.
    # We don't want that prefix, so we remove it.
    if config_path.startswith("/"):
        config_path = config_path[1:]
    path = config_path.split("/")
    for p in path[:-1]:
        env_cfg = env_cfg[p]
    return env_cfg


# Ensure resolvers are registered when this module is imported
register_resolvers()
