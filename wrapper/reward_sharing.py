import numpy as np
import gymnasium as gym
import numpy as np


# class RewardAllocator():
#     def __init__(self, num_agents) -> None:
#         self._num_agents = num_agents

#     def compute_shared_rewards(self, rewards):
#         return rewards

#     def obs(self, agent_id, agent_obs):
#         return np.array([agent_obs])

# class MatrixRewardAllocator(RewardAllocator):
#     def __init__(self, num_agents, reward_sharing_matrix) -> None:
#         super().__init__(num_agents)
#         self._reward_sharing_matrix = reward_sharing_matrix
#         self._agent_kinship = np.array(range(num_agents))

#     def compute_shared_rewards(self, rewards):
#         if self._reward_sharing_matrix is None:
#             return rewards

#         return self._reward_sharing_matrix @ rewards.transpose()

# class FamillySparseAllocator(RewardAllocator):
#     def __init__(self, num_agents, families, family_reward_coef) -> None:
#         super().__init__(num_agents)
#         assert family_reward_coef > 0
#         self._families = families
#         self._member_reward_coef = np.zeros(num_agents, dtype=np.float32)
#         self._self_reward_coef = np.zeros(num_agents, dtype=np.float32)
#         self._agent_to_family_id = np.zeros(num_agents, dtype=np.int32)
#         for fid, family in enumerate(families):
#             mc = family_reward_coef / len(family)
#             self._member_reward_coef[family] = mc
#             self._self_reward_coef[family] = 1 - family_reward_coef
#             self._agent_to_family_id[family] = fid

#     def compute_shared_rewards(self, rewards):
#         shared_rewards = np.zeros_like(rewards)
#         for agent, reward in enumerate(rewards):
#             if reward != 0:
#                 family_id = self._agent_to_family_id[agent]
#                 family = self._families[family_id]
#                 shared_rewards[family] += reward * self._member_reward_coef[agent]
#                 shared_rewards[agent] += reward * self._self_reward_coef[agent]
#         return shared_rewards


# class FamillyAllocator(FamillySparseAllocator):
#     def __init__(self, num_agents, num_families, family_reward_coef) -> None:
#         assert num_families > 0 and family_reward_coef > 0

#         agents = np.array(range(num_agents))
#         np.random.shuffle(agents)
#         families = np.array_split(agents, num_families)
#         super().__init__(num_agents, families, family_reward_coef)

# class FamillyMatrixAllocator(MatrixRewardAllocator):
#     def __init__(self, num_agents, num_families, family_reward) -> None:
#         assert num_families > 0 and family_reward > 0

#         agents = np.array(range(num_agents))
#         np.random.shuffle(agents)
#         families = np.array_split(agents, num_families)

#         rsm = np.zeros((len(agents), len(agents)), dtype=np.float32)
#         # share the reward among the families
#         for family_id, family in enumerate(families):
#             fr = family_reward / ( len(family) - 1 )
#             for a in family:
#                 self._agent_kinship[a] = family_id
#                 rsm[a, family] = fr
#                 rsm[a, a] = 1 - family_reward

#         # normalize
#         rsm = rsm / rsm.sum(axis=1, keepdims=True)

#         super().__init__(num_agents, rsm)


# class TeamRewardSharing(gym.wrappers.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self._reward_sharing = None
#         # set up reward sharing
#         self._reward_sharing = RewardAllocator(self._num_agents)
#         num_families = self._level_generator.sample_cfg("rsm_num_families")
#         family_reward = self._level_generator.sample_cfg("rsm_family_reward")
#         if num_families > 0:
#             self._reward_sharing = FamillyAllocator(self._num_agents, num_families, family_reward)

#     def reset(self):
#         self._last_actions = np.zeros((self._num_agents, 2), dtype=np.int32)
#         return self.env.reset()

#     def step(self, actions):
#         obs, rewards, terms, truncs, infos = self.env.step(actions)
#         rewards = self._reward_sharing.compute_shared_rewards(rewards)

#         return self._augment_observations(obs), rewards, terms, truncs, infos

#     def _augment_observations(self, obs):
#         return [{
#             "last_action": np.array(self._last_actions[agent]),
#             **agent_obs
#         } for agent, agent_obs in enumerate(obs)]


#     def observation_space(self):
#         return gym.spaces.Dict({
#             "last_action": gym.spaces.Box(
#                 low=0, high=255, shape=(2,), dtype=np.int32),
#             **self.env.observation_space()
#         })
