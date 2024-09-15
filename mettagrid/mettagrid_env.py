from typing import Any, Dict

import numpy as np
import pufferlib
from omegaconf import OmegaConf

from mettagrid.config.game_builder import MettaGridGameBuilder
from mettagrid.config.sample_config import sample_config
from mettagrid.mettagrid_c import MettaGrid # pylint: disable=E0611


class MettaGridEnv(pufferlib.PufferEnv):
    def __init__(self, render_mode: str, **cfg):
        super().__init__()

        self._render_mode = render_mode
        self._cfg = OmegaConf.create(cfg)
        self.make_env()

        self._renderer = None

        self.done = False
        self.buf = None


    def make_env(self):
        scfg = sample_config(self._cfg.game)
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
        obs, rewards, terminated, truncated, infos = self._c_env.step(actions.astype(np.int32))

        if self._cfg.normalize_rewards:
            rewards -= rewards.mean()

        infos = {}
        if terminated.all() or truncated.all():
            self.done = True
            self.process_episode_stats(infos)
        return obs, list(rewards), terminated.all(), truncated.all(), infos

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
        # num_generators = self._griddly_yaml["Environment"]["Levels"][0].count("g")
        # num_converters = self._griddly_yaml["Environment"]["Levels"][0].count("c")
        # max_resources = num_generators * min(
        #     self._game_builder.object_configs.generator.initial_resources,
        #     self._max_steps / self._game_builder.object_configs.generator.cooldown)

        # max_conversions = num_converters * (
        #     self._max_steps / self._game_builder.object_configs.converter.cooldown
        # )
        # max_conv_energy = min(max_resources, max_conversions) * \
        #     np.mean(list(self._game_builder.object_configs.converter.energy_output.values()))

        # initial_energy = self._game_builder.object_configs.agent.initial_energy * self._game_builder.num_agents

        # self._max_level_energy = max_conv_energy + initial_energy
        # self._max_level_energy_per_agent = self._max_level_energy / self._game_builder.num_agents

        # self._max_level_reward_per_agent = self._max_level_energy_per_agent


    @property
    def _max_steps(self):
        return self._game_builder.max_steps

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

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
