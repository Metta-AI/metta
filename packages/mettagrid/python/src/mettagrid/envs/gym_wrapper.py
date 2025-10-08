from typing import Any

import gymnasium as gym
import numpy as np

from mettagrid.mettagrid_c import dtype_actions


class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleAgentWrapper, self).__init__(env)

    def step(self, action):
        action_array = np.asarray(action, dtype=dtype_actions)
        if action_array.ndim == 0:
            action_array = action_array.reshape(1)
        elif action_array.ndim == 1 and action_array.shape[0] == 1:
            pass
        else:
            raise ValueError(f"SingleAgentWrapper expects scalar action, got shape {action_array.shape}")
        observations, rewards, terminals, truncations, infos = self.env.step(action_array)

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
