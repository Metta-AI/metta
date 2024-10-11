from typing import Any, Dict

import numpy as np
import pufferlib
from omegaconf import OmegaConf

from mettagrid.config.game_builder import MettaGridGameBuilder
from mettagrid.config.sample_config import sample_config
from mettagrid.mettagrid_c import MettaGrid # pylint: disable=E0611
from gymnasium import spaces

class MettaGridEnv(pufferlib.PufferEnv):
    def __init__(self, render_mode: str, **cfg):
        super().__init__()

        self._render_mode = render_mode
        self._cfg = OmegaConf.create(cfg)
        self.make_env()

        self._renderer = None
        self._action_space = self._env.action_space
        if self._cfg.flatten_actions:
            self._action_map = [
                (0, 0), # Noop
                (0, 1), # Move Forward
                (0, 2), # Move Backward
                (1, 0), # Rotate Left
                (1, 1), # Rotate Right
                (1, 2), # Rotate Up
                (1, 3), # Rotate Down
                (2, 0), # Use
                (3, 0), # Attack
                (4, 0), # Shield
                (5, 0), # Gift
                (6, 0), # Swap
            ]
            self._action_space = spaces.Discrete(len(self._action_map))

        self.done = False
        self.buf = None

    def make_env(self):
        scfg = sample_config(self._cfg.game, self._cfg.sampling)
        assert isinstance(scfg, Dict)
        game_cfg = OmegaConf.create(scfg)
        self._game_builder = MettaGridGameBuilder(**scfg) # type: ignore
        level = self._game_builder.level()
        self._c_env = MettaGrid(game_cfg, level)
        self._grid_env = self._c_env
        self._num_agents = self._c_env.num_agents()

        # self._grid_env = PufferGridEnv(self._c_env)
        env = self._grid_env

        self._env = env
        #self._env = LastActionTracker(self._grid_env)
        #self._env = Kinship(**sample_config(self._cfg.kinship), env=self._env)
        #self._env = RewardTracker(self._env)
        #self._env = FeatureMasker(self._env, self._cfg.hidden_features)
        self.done = False

    def reset(self, seed=None):
        self.make_env()

        if hasattr(self, "buf") and self.buf is not None:
            self._c_env.set_buffers(
                self.buf.observations,
                self.buf.terminals,
                self.buf.truncations,
                self.buf.rewards)

        # obs, infos = self._env.reset(**kwargs)
        # self._compute_max_energy()
        # return obs, infos
        obs, infos = self._c_env.reset()
        return obs, infos

    def step(self, actions):
        assert not self.done, "Environment already done"

        if self._cfg.flatten_actions:
            new_actions = np.zeros((self._num_agents, 2), dtype=np.int32)
            for idx in range(self._num_agents):
                new_actions[idx] = self._action_map[actions[idx]]
            actions = new_actions

        obs, rewards, terminated, truncated, infos = self._c_env.step(actions.astype(np.int32))

        if self._cfg.normalize_rewards:
            rewards -= rewards.mean()

        infos = {}
        if terminated.all() or truncated.all():
            self.done = True
            self.process_episode_stats(infos)

        return obs, rewards, terminated, truncated, infos

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

        infos["episode_rewards"] = episode_rewards
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
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._action_space

    def action_names(self):
        return self._env.action_names()

    @property
    def player_count(self):
        return self._num_agents

    def render(self):
        if self._renderer is None:
            return None

        return self._renderer.render(
            self._c_env.current_timestep(),
            self._c_env.grid_objects()
        )

    @property
    def grid_features(self):
        return self._env.grid_features()

    @property
    def global_features(self):
        return []

    @property
    def render_mode(self):
        return self._render_mode
