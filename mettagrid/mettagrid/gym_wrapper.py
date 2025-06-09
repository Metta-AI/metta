from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

from mettagrid.mettagrid_env import MettaGridEnv


class SingleAgentWrapper(gym.Wrapper):
    """
    Wraps a multi-agent MettaGridEnv to be used as a single-agent environment.
    Takes only the first agent's observations, rewards, and other returns.
    """

    def __init__(self, env):
        super(SingleAgentWrapper, self).__init__(env)

    def step(self, action) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        metta_grid_env: MettaGridEnv = self.env  # type: ignore
        assert isinstance(metta_grid_env, MettaGridEnv)

        action = np.asarray(action, dtype=np.uint32)
        action = action[None, ...]
        observations, rewards, terminals, truncations, infos = metta_grid_env.step(action)

        observations = observations.squeeze(0)
        rewards = rewards.squeeze(0)
        terminals = terminals.squeeze(0)
        truncations = truncations.squeeze(0)

        return observations, rewards, terminals, truncations, infos

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        metta_grid_env: MettaGridEnv = self.env  # type: ignore
        assert isinstance(metta_grid_env, MettaGridEnv)

        obs, infos = metta_grid_env.reset(seed=seed, options=options)
        obs = obs.squeeze(0)
        return obs, infos

    @property
    def action_space(self):
        metta_grid_env: MettaGridEnv = self.env  # type: ignore
        assert isinstance(metta_grid_env, MettaGridEnv)
        return metta_grid_env.single_action_space

    @property
    def observation_space(self):
        metta_grid_env: MettaGridEnv = self.env  # type: ignore
        assert isinstance(metta_grid_env, MettaGridEnv)
        return metta_grid_env.single_observation_space


class MultiToDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(MultiToDiscreteWrapper, self).__init__(env)

        metta_grid_env = self.env.unwrapped
        assert isinstance(metta_grid_env, MettaGridEnv), "MultiToDiscreteWrapper requires a MettaGridEnv"

        max_action_args = metta_grid_env.max_action_args
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
        metta_grid_env: MettaGridEnv = self.env  # type: ignore
        assert isinstance(metta_grid_env, MettaGridEnv)

        mapped_action = self.action_map[action]
        return metta_grid_env.step(mapped_action)
