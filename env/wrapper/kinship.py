import math
import gymnasium as gym
import numpy as np

class Kinship(gym.Wrapper):
    def __init__(self, team_size: int, team_reward: float, env: gym.Env):
        super().__init__(env)
        self._team_size = team_size
        self._team_reward = team_reward
        self._num_agents = self.env.unwrapped.player_count
        self._num_teams = int(math.ceil(self._num_agents / self._team_size))

        self._agent_team = np.array([
            agent // self._team_size for agent in range(self._num_agents)])
        self._team_to_agents = {
            team: np.array([
                agent for agent in range(self._num_agents)
                if self._agent_team[agent] == team
            ]) for team in range(self._num_teams)
        }

        self._agent_id_feature_idx = self.env.unwrapped.grid_features.index("agent:id")

        grid_shape = self.env.unwrapped.observation_space["grid_obs"].shape
        self._kinship_shape = (1, grid_shape[1], grid_shape[2])

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        return self._augment_observations(obs), infos

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        rewards = np.array(rewards)
        agent_idxs = rewards.nonzero()[0]

        if len(agent_idxs) > 0:
            team_rewards = np.zeros(self._num_teams)
            for agent in agent_idxs:
                team = self._agent_team[agent]
                team_rewards[team] += self._team_reward * rewards[agent]
                rewards[agent] -= self._team_reward * rewards[agent]
            team_idxs = team_rewards.nonzero()[0]
            for team in team_idxs:
                team_agents = self._team_to_agents[team]
                team_reward = team_rewards[team] / len(team_agents)
                rewards[team_agents] += team_reward

        return self._augment_observations(obs), rewards, terms, truncs, infos

    def _team_id_obs(self, agent_obs):
        agent_id_obs = agent_obs["grid_obs"][self._agent_id_feature_idx]
        idxs = agent_id_obs.nonzero()
        agent_ids = agent_id_obs[idxs] - 1
        team_ids = self._agent_team[agent_ids]
        team_id_obs = np.zeros_like(agent_id_obs)
        team_id_obs[idxs] = team_ids + 1
        return team_id_obs

    def _augment_observations(self, obs):
        return [{
            "kinship": self._team_id_obs(agent_obs),
            **agent_obs
        } for agent, agent_obs in enumerate(obs)]

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "kinship": gym.spaces.Box(
                -np.inf, high=np.inf,
                shape=self._kinship_shape, dtype=np.float32
            ),
            **self.env.observation_space.spaces
        })
