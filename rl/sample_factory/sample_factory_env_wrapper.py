from copy import deepcopy
from typing import Union

import gymnasium as gym


from sample_factory.envs.env_utils import TrainingInfoInterface
from pettingzoo import utils as pettingzoo_utils



class SampleFactoryEnvWrapper(gym.Env, TrainingInfoInterface):

    def __init__(self, env: gym.Env, env_id: int):

        TrainingInfoInterface.__init__(self)

        self.gym_env = env
        self.num_agents = self.gym_env.unwrapped.player_count

        self.curr_episode_steps = 0
        if self.num_agents == 1:
            self.observation_space = self.gym_env.observation_space
            action_space = self.gym_env.action_space
            self.is_multiagent = False
        else:
            self.observation_space = self.gym_env.observation_space
            action_space = self.gym_env.action_space
            self.is_multiagent = True

        if isinstance(action_space, gym.spaces.MultiDiscrete):
            action_space = gym.spaces.Tuple(
                [gym.spaces.Discrete(num_actions) for num_actions in action_space.nvec]
            )

        self.action_space = action_space
        self.episode_rewards = [[] for _ in range(self.num_agents)]
        self.current_episode = 0
        self.env_id = env_id


    def reset(self, **kwargs):
        self.current_episode += 1
        self.curr_episode_steps = 0
        self.episode_rewards = [[] for _ in range(self.num_agents)]
        return self.gym_env.reset()

    def step(self, actions):
        if self.is_multiagent:
            actions = list(actions)

        obs, rewards, terminated, truncated, infos_dict = self.gym_env.step(actions)

        self.curr_episode_steps += 1

        # auto-reset the environment
        if terminated or truncated:
            obs = self.reset()[0]

        if isinstance(self.gym_env, pettingzoo_utils.ParallelEnv):
            rewards = [rewards[agent] for agent in self.gym_env.agents]
            obs = [obs[agent] for agent in self.gym_env.agents]

        # For better readability, make `infos` a list.
        # In case of a single player, get the first element before returning
        infos = [deepcopy(infos_dict) for _ in range(self.num_agents)]
        if "episode_extra_stats" in infos_dict:
            for i in range(self.num_agents):
                infos[i]["episode_extra_stats"] = infos_dict["episode_extra_stats"][i]

        if self.is_multiagent:
            terminated = [terminated] * self.num_agents
            truncated = [truncated] * self.num_agents
        else:
            infos = infos[0]

        return obs, rewards, terminated, truncated, infos

    def render(self, *args, **kwargs):
        return self.gym_env.render(*args, **kwargs)

