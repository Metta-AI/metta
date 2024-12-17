import time
import logging
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

from agent.policy_store import PolicyStore
from rl.pufferlib.vecenv import make_vecenv

logger = logging.getLogger("evaluator")


class PufferEvaluator():
    def __init__(
        self,
        cfg: OmegaConf,
        policy_store: PolicyStore,
        policy: OmegaConf,
        baselines: List[OmegaConf],
        **kwargs
    ) -> None:

        self._cfg = cfg
        self._device = cfg.device

        self._all_prs = policy_store.policies([policy] + baselines)
        self.l = len(self._all_prs)
        if self.l < 2:
            raise ValueError("Need two or more policies for a tournament.")

        omega = (self.l * (self.l - 1)) / 2  # number of unique matchups: n choose 2
        self._mult = (cfg.evaluator.num_envs + omega - 1) // omega  # multiplicity of each match type
        self._num_envs = self._mult * omega
        self._vecenv = make_vecenv(self._cfg, num_envs=self._num_envs)
        self._min_episodes = (cfg.evaluator.num_episodes + self._num_envs - 1) // self._num_envs * self._num_envs

        self._policy_agent_pct = cfg.evaluator.policy_agents_pct
        self._agents_per_env = cfg.env.game.num_agents
        if self._agents_per_env % 2 != 0:
            # replace 2 w/ policies per env for > 2 policies per env in the future
            raise ValueError("Number of agents per environment must be divisible by policies per env.")

        self._total_agents = self._num_envs * self._agents_per_env

        logger.info(
            'Tournament: Policy Agents: %s, Baseline Agents: %s',
            self._agents_per_env * self._policy_agent_pct,
            self._agents_per_env - (self._agents_per_env * self._policy_agent_pct)
        )

        self.slice_idxs = torch.arange(self._vecenv.num_agents).reshape(self._num_envs, self._agents_per_env).to(device=self._device)

        self.matches = []
        for i in range(self.l):
            s_0 = i * (i + self.l - 1) // 2
            idx_i = (
                self.slice_idxs[s_0 * self._mult : (s_0 + self.l - i - 1) * self._mult, : self._agents_per_env * self._policy_agent_pct]
                .reshape(-1)
            )
            self.matches.append({
                "policy": self._all_prs[i],
                "indices": idx_i,
                "rnn_state": None
            })

            for j in range(i + 1, self.l):
                idx_j = (
                    self.slice_idxs[(s_0 + j - i - 1) * self._mult : (s_0 + j - i) * self._mult, self._agents_per_env * (1 - self._policy_agent_pct) :]
                    .reshape(-1)
                )
                self.matches.append({
                    "policy": self._all_prs[j],
                    "indices": idx_j,
                    "rnn_state": None
                })

        self._max_time_s = cfg.evaluator.max_time_s
        self._completed_episodes = 0
        self._total_rewards = np.zeros(self._total_agents)

    def evaluate(self):
        logger.info(
            "Evaluating policy %s against baselines: %s",
            self._all_prs[0].name,
            ", ".join(pr.name for pr in self._all_prs[1:])
        )
        logger.info(
            "Total agents %s. Agents per env: %s. Agent percent: %s",
            self._total_agents, self._agents_per_env, self._policy_agent_pct
        )
        logger.info(
            "Num envs: %s. Min episodes: %s. Max time: %s",
            self._num_envs, self._min_episodes, self._max_time_s
        )

        obs, _ = self._vecenv.reset()
        game_stats = []
        start = time.time()

        while self._completed_episodes < self._min_episodes and time.time() - start < self._max_time_s:
            left_policy_actions = []
            right_policy_actions = []
            with torch.no_grad():
                obs = torch.as_tensor(obs).to(device=self._device)

                for i in range(self.l):
                    # Hardcoded indexing logic
                    left_policy_actions.append(self.get_policy_actions(i, obs))
                    for j in range(i + 1, self.l):
                        right_policy_actions.append(self.get_policy_actions(i + j, obs))

                dim = 0 if self._cfg.env.flatten_actions else 1
                actions = torch.cat([
                    torch.cat(left_policy_actions, dim=dim).view(self._num_envs, int(self._agents_per_env * self._policy_agent_pct), -1),
                    torch.cat(right_policy_actions, dim=dim).view(self._num_envs, int(self._agents_per_env * (1 - self._policy_agent_pct)), -1),
                ], dim=1)

                if self._cfg.env.flatten_actions:
                    actions = actions.view(-1)
                else:
                    actions = actions.view(self._num_envs * self._agents_per_env, -1)

            obs, rewards, dones, truncated, infos = self._vecenv.step(actions.cpu().numpy())
            self._total_rewards += rewards
            self._completed_episodes += sum([e.done for e in self._vecenv.envs])

            if len(infos) > 0:
                game_stats = self.process_infos(infos, game_stats)

            logger.info("Total completed episodes: %s", self._completed_episodes)
            logger.info("Evaluation time: %s", time.time() - start)

        return game_stats

    def close(self):
        self._vecenv.close()

    def process_infos(self, infos, game_stats):
        # Hardcoded indexing logic
        agent_stats = []
        for i in range(self.l):
            for j in range(self._mult * (self.l - i - 1)):
                for k in range(self._agents_per_env * self._policy_agent_pct):
                    agent_stats.append(infos[j * self._agents_per_env + k]["agent_raw"])
                    agent_stats[j * self._agents_per_env + k]["policy_name"] = self.matches[i]["policy"].name

            for m in range(i + 1, self.l):
                for n in range(self._mult):
                    for o in range(self._agents_per_env * self._policy_agent_pct, self._agents_per_env):
                        idx = (m - i - 1) * self._mult + n
                        idx *= self._agents_per_env
                        idx += o
                        agent_stats.append(infos[idx]["agent_raw"])
                        agent_stats[idx]["policy_name"] = self.matches[m]["policy"].name

        game_stats.append(agent_stats)
        return game_stats

    def get_policy_actions(self, index, obs):
        policy = self.matches[index]["policy"]
        observations = obs[self.matches[index]["indices"]]

        if hasattr(policy, 'lstm'):
            rnn_state = self.matches[index]["rnn_state"]
            actions, _, _, _, self.matches[index]["rnn_state"] = policy(observations, rnn_state)
        else:
            actions, _, _, _ = policy(observations)

        return actions
