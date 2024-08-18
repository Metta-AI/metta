import numpy as np

class PrestigeRewardsEnv():
    def __init__(self) -> None:
        self._prestige_reward_weight = None
        self._prestige_steps = None
        self._episode_prestige_rewards = None

    def _compute_presitge_rewards(self, rewards):
        if self._prestige_reward_weight == 0 or self._step % self._prestige_steps != 0:
            return rewards

        altar_energy = np.array(
            [v for v in self._griddly_env.game.get_global_variable(
                ["stats:energy:used:altar"])["stats:energy:used:altar"].values()
            ])
        if altar_energy.sum() == 0:
            return rewards

        # rank the agents by their rewards
        rank = np.argsort(np.argsort(altar_energy))
        # Scale the ranks to the range -1 to 1
        prestige_rewards = 2.0 * (rank / (rank.size - 1)) - 1.0
        # normalize based on how often we get prestige rewards
        prestige_rewards *= self._prestige_reward_weight * self._prestige_steps / self._max_steps
        self._episode_prestige_rewards += prestige_rewards
        rewards += prestige_rewards
        return rewards

    def on_make_env(self):
        self._episode_rewards = np.array([0] * self._num_agents, dtype=np.float32)
        self._prestige_steps = int(self._level_generator.sample_cfg("reward_rank_steps"))
        self._prestige_reward_weight = self._level_generator.sample_cfg("reward_prestige_weight")

    def on_reset(self):
        self._episode_prestige_rewards = np.array([0] * self._num_agents, dtype=np.float32)

    def on_add_stats(self, infos):
        for agent in range(self._num_agents):
            agent_stats["prestige_reward"] = self._episode_prestige_rewards[agent]

    def update_rewards(self, rewards):
        rewards = self._compute_presitge_rewards(rewards)
        return rewards
