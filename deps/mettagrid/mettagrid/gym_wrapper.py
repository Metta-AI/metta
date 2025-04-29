from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete


class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleAgentWrapper, self).__init__(env)

    def step(self, action):
        action = np.asarray(action, dtype=np.uint32)
        action = action[None, ...]
        observations, rewards, terminals, truncations, infos = self.env.step(action)

        observations = observations.squeeze(0)
        rewards = rewards.squeeze(0)
        terminals = terminals.squeeze(0)
        truncations = truncations.squeeze(0)

        return observations, rewards, terminals, truncations, infos

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, infos = self.env.reset(seed=seed, options=options)
        obs = obs.squeeze(0)
        return obs, infos

    @property
    def action_space(self):
        return self.env.single_action_space

    @property
    def observation_space(self):
        return self.env.single_observation_space


class MultiToDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(MultiToDiscreteWrapper, self).__init__(env)
        max_action_args = env.unwrapped.max_action_args
        arg_counts = [a + 1 for a in max_action_args]

        self.n_actions = np.sum(arg_counts)
        self.action_map = np.zeros((self.n_actions, 2), dtype=np.int32)

        i = 0
        for action, max_arg in enumerate(arg_counts):
            for arg in range(max_arg):
                self.action_map[i] = (action, arg)
                i += 1

    @property
    def action_space(self):
        return Discrete(self.n_actions)

    @property
    def single_action_space(self):
        return Discrete(self.n_actions)

    def step(self, action):
        mapped_action = self.action_map[action]
        return self.env.step(mapped_action)


class RaylibRendererWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(RaylibRendererWrapper, self).__init__(env)

        import mettagrid.renderer.raylib.raylib_renderer as rl

        self.renderer = rl.MettaGridRaylibRenderer(self.env._c_env, cfg.game)
        self.total_rewards = np.zeros(self.env.num_agents)

    def step(self, actions):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(self._obs).cpu()

        self.renderer.env = self.env._c_env
        self.renderer.update(
            actions,
            obs_tensor,
            self.rewards,
            self.total_rewards,
            self.env._c_env.current_timestep(),
        )
        self.renderer.render_and_wait()
        actions = self.renderer.get_actions()

        self._obs, self.rewards, terminated, truncated, info = self.env.step(actions)
        self.total_rewards += self.rewards

        return self._obs, self.rewards, terminated, truncated, info

    def render(self):
        return self.renderer.render_and_wait()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._obs, infos = self.env.reset(seed=seed, options=options)
        self.total_rewards = np.zeros(self.env.num_agents)
        self.rewards = np.zeros(self.env.num_agents)

        return self._obs, infos
