from functools import lru_cache
import re
import gymnasium as gym
import numpy as np

class LastActionTracker(gym.Wrapper):
    def __init__(self, env):
        super(LastActionTracker, self).__init__(env)
        self._last_actions = None

    def reset(self, **kwargs):
        self._last_actions = np.zeros((self.unwrapped.player_count, 2), dtype=np.int32)
        obs, infos = self.env.reset(**kwargs)
        return self._augment_observations(obs), infos


    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        self._last_actions = actions

        return self._augment_observations(obs), rewards, terms, truncs, infos

    def _augment_observations(self, obs):
        return [{
            "last_action": self._last_actions[agent],
            **agent_obs
        } for agent, agent_obs in enumerate(obs)]


    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "last_action": gym.spaces.Box(
                low=0, high=255, shape=(2,), dtype=np.int32
            ),
            **self.env.observation_space.spaces
        })
