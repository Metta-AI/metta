from functools import lru_cache
import gymnasium as gym
import numpy as np

class RewardTracker(gym.Wrapper):
    def __init__(self, env):
        super(RewardTracker, self).__init__(env)
        self._last_rewards = None

    def reset(self, **kwargs):
        self._last_rewards = np.zeros(self.unwrapped.player_count, dtype=np.float32)
        obs, infos = self.env.reset(**kwargs)
        return self._augment_observations(obs), infos

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        self._last_rewards = rewards

        return self._augment_observations(obs), rewards, terms, truncs, infos

    def _augment_observations(self, obs):
        return [{
            "last_reward": np.array(self._last_rewards[agent]),
            **agent_obs
        } for agent, agent_obs in enumerate(obs)]

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "last_reward": gym.spaces.Box(
                -np.inf, high=np.inf,
                shape=(1,), dtype=np.float32
            ),
            **self.env.observation_space.spaces
        })
