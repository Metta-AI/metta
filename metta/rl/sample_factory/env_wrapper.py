from copy import deepcopy

import gymnasium as gym
import numpy as np
from sample_factory.envs.env_utils import TrainingInfoInterface

class SampleFactoryEnvWrapper(gym.Env, TrainingInfoInterface):
    def __init__(self, env: gym.Env, env_id: int):
        TrainingInfoInterface.__init__(self)

        self.env = env
        self.multi_agent = True

        self.observation_space = gym.spaces.Dict(
            {
                "grid_obs": env.observation_space,
                "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
            }
        )

        self.curr_episode_steps = 0
        self.num_agents = env.player_count

        self.action_space = env.action_space

        self.current_episode = 0
        self.env_id = env_id

    def reset(self, **kwargs):
        self.current_episode += 1
        self.curr_episode_steps = 0
        return self.env.reset(**kwargs)

    def step(self, actions):
        actions = np.array(actions).astype(np.int32)
        obs, rewards, terminated, truncated, infos_dict = self.env.step(actions)
        self.curr_episode_steps += 1

        # auto-reset the environment
        if terminated.all() or truncated.all():
            obs = self.reset()[0]

        infos = [{} for _ in range(self.num_agents)]
        if "agent_raw" in infos_dict:
            for i in range(self.num_agents):
                infos[i]["episode_extra_stats"] = deepcopy(infos_dict["agent_raw"][i])
                infos[i]["episode_extra_stats"].update(infos_dict["game"])

        return obs, rewards, terminated, truncated, infos

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
