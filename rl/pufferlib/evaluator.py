import time
import logging
from typing import List
import numpy as np
import torch
from omegaconf import OmegaConf
from rl.pufferlib.vecenv import make_vecenv
from agent.policy_store import PolicyRecord
logger = logging.getLogger("evaluator")

class PufferEvaluator():
    def __init__(
        self,
        cfg: OmegaConf,
        policy_record: PolicyRecord,
        baseline_records: List[PolicyRecord],
        **kwargs
    ) -> None:

        self._cfg = cfg
        self._device = cfg.device

        self._vecenv = make_vecenv(self._cfg, num_envs=cfg.evaluator.num_envs)
        self._num_envs = cfg.evaluator.num_envs
        self._min_episodes = cfg.evaluator.num_episodes
        self._max_time_s = cfg.evaluator.max_time_s

        self._policy_pr = policy_record
        self._baseline_prs = baseline_records

        self._policy_agent_pct = cfg.evaluator.policy_agents_pct
        if len(self._baseline_prs) == 0:
            self._baseline_prs = [self._policy_pr]
            self._policy_agent_pct = 0.9

        self._agents_per_env = cfg.env.game.num_agents
        self._total_agents = self._num_envs * self._agents_per_env
        self._policy_agents_per_env = max(1, int(self._agents_per_env * self._policy_agent_pct))
        self._baseline_agents_per_env = self._agents_per_env - self._policy_agents_per_env

        logger.info(f'Tournament: Policy Agents: {self._policy_agents_per_env}, ' +
              f'Baseline Agents: {self._baseline_agents_per_env}')

        slice_idxs = torch.arange(self._vecenv.num_agents)\
            .reshape(self._num_envs, self._agents_per_env).to(device=self._device)

        self._policy_idxs = slice_idxs[:, :self._policy_agents_per_env]\
            .reshape(self._policy_agents_per_env * self._num_envs)

        self._baseline_idxs = []
        if len(self._baseline_prs) > 0:
            envs_per_opponent = self._num_envs // len(self._baseline_prs)
            self._baseline_idxs = slice_idxs[:, self._policy_agents_per_env:]\
                .reshape(self._num_envs*self._baseline_agents_per_env)\
                .split(self._baseline_agents_per_env*envs_per_opponent)

        self._completed_episodes = 0
        self._total_rewards = np.zeros(self._total_agents)
        self._agent_stats = [{} for a in range(self._total_agents)]

        # Extract policy names
        self._policy_name = self._policy_pr.name
        self._baseline_names = [b.name for b in self._baseline_prs]

        # Create mapping from agent index to policy name
        self._agent_idx_to_policy_name = {}
        for agent_idx in self._policy_idxs:
            self._agent_idx_to_policy_name[agent_idx.item()] = self._policy_name

        for i, baseline_agent_idxs in enumerate(self._baseline_idxs):
            for agent_idx in baseline_agent_idxs:
                self._agent_idx_to_policy_name[agent_idx.item()] = self._baseline_names[i]

    def evaluate(self):
        logger.info("Evaluating policy:")
        logger.info(self._policy_pr.name)
        logger.info("Against baselines:")
        for baseline in self._baseline_prs:
            logger.info(baseline.name)
        logger.info(f"Total agents: {self._total_agents}")
        logger.info(f"Policy agents per env: {self._policy_agents_per_env}")
        logger.info(f"Baseline agents per env: {self._baseline_agents_per_env}")
        logger.info(f"Num envs: {self._num_envs}")
        logger.info(f"Min episodes: {self._min_episodes}")
        logger.info(f"Max time: {self._max_time_s}")

        obs, _ = self._vecenv.reset()
        policy_rnn_state = None
        baselines_rnn_state = [None for _ in range(len(self._baseline_prs))]

        game_stats = []

        start = time.time()

        while self._completed_episodes < self._min_episodes and time.time() - start < self._max_time_s:
            baseline_actions = []
            with torch.no_grad():
                obs = torch.as_tensor(obs).to(device=self._device)
                my_obs = obs[self._policy_idxs] 

                # Parallelize across opponents
                policy = self._policy_pr.policy() #the given policy to evaluate
                if hasattr(policy, 'lstm'):
                    policy_actions, _, _, _, policy_rnn_state = policy(my_obs, policy_rnn_state)
                else:
                    policy_actions, _, _, _ = policy(my_obs)

                # Iterate opponent policies
                for i in range(len(self._baseline_prs)): #all baseline policies
                    baseline_obs = obs[self._baseline_idxs[i]]
                    baseline_rnn_state = baselines_rnn_state[i]

                    baseline = self._baseline_prs[i].policy()
                    if hasattr(baseline, 'lstm'):
                        baseline_action, _, _, _, baselines_rnn_state[i] = baseline(baseline_obs, baseline_rnn_state)
                    else:
                        baseline_action, _, _, _ = baseline(baseline_obs)

                    baseline_actions.append(baseline_action)


            if len(self._baseline_prs) > 0:
                dim = 0 if self._cfg.env.flatten_actions else 1
                actions = torch.cat([
                    policy_actions.view(self._num_envs, self._policy_agents_per_env, -1),
                    torch.cat(baseline_actions, dim=dim).view(self._num_envs, self._baseline_agents_per_env, -1),
                ], dim=1)
            else:
                actions = policy_actions

            if self._cfg.env.flatten_actions:
                actions = actions.view(-1)
            else:
                actions = actions.view(self._num_envs*self._agents_per_env, -1)

            obs, rewards, dones, truncated, infos = self._vecenv.step(actions.cpu().numpy())
            self._total_rewards += rewards
            self._completed_episodes += sum([e.done for e in self._vecenv.envs])

            if len(infos) > 0:
                for n in range(len(infos)):
                    if "agent_raw" in infos[n]:
                        one_episode = infos[n]["agent_raw"]
                        for m in range(len(one_episode)):
                            agent_idx = m + n * self._agents_per_env
                            if agent_idx in self._agent_idx_to_policy_name:
                                one_episode[m]['policy_name'] = self._agent_idx_to_policy_name[agent_idx]
                            else:
                                one_episode[m]['policy_name'] = "No Name Found"
                        game_stats.append(one_episode)


        logger.info(f"Total episodes: {self._completed_episodes}")
        logger.info(f"Evaluation time: {time.time() - start}")
        return game_stats

    def close(self):
        self._vecenv.close()

