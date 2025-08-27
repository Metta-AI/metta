from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete



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
