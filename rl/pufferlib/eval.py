import logging
import time

import numpy as np
import torch
from agent.policy_store import PolicyStore, PolicyRecord
from omegaconf import DictConfig, OmegaConf
import hydra

from rl.pufferlib.vecenv import make_vecenv

logger = logging.getLogger("eval")

class Eval():
    def __init__(
        self,
        policy_store: PolicyStore,
        policy_pr: PolicyRecord,
        env_defaults: DictConfig,

        env: str,
        npc_policy_uri: str,
        device: str,

        policy_agents_pct: float = 1.0,
        num_envs: int = 1,
        num_episodes: int = 1,
        max_time_s: int = 60,
        vectorization: str = "serial",

        **kwargs,

    ) -> None:
        env_cfg = hydra.compose(config_name=env)
        self._env_cfg = OmegaConf.merge(env_defaults, env_cfg)

        self._npc_policy_uri = npc_policy_uri
        self._policy_agents_pct = policy_agents_pct
        self._policy_store = policy_store

        self._device = device

        self._num_envs = num_envs
        self._min_episodes = num_episodes
        self._max_time_s = max_time_s

        # load candidate policy
        self._policy_pr = policy_pr

        # load npc policy
        self._npc_pr = None
        if self._npc_policy_uri is None:
            self._policy_agents_pct = 1.0
        else:
            self._npc_pr = self._policy_store.policy(self._npc_policy_uri)

        self._agents_per_env = self._env_cfg.game.num_agents
        self._policy_agents_per_env = max(1, int(self._agents_per_env * self._policy_agents_pct))
        self._npc_agents_per_env = self._agents_per_env - self._policy_agents_per_env
        self._total_agents = self._num_envs * self._agents_per_env

        logger.info(f'Tournament: Policy Agents: {self._policy_agents_per_env}, ' +
              f'Npc Agents: {self._npc_agents_per_env}')

        self._vecenv = make_vecenv(self._env_cfg, vectorization, num_envs=self._num_envs)

        # each index is an agent, and we reshape it into a matrix of num_envs x agents_per_env
        # you can figure out which episode you're in by doing the floor division
        slice_idxs = torch.arange(self._vecenv.num_agents)\
            .reshape(self._num_envs, self._agents_per_env).to(device=self._device)

        self._policy_idxs = slice_idxs[:, :self._policy_agents_per_env]\
            .reshape(self._policy_agents_per_env * self._num_envs)

        self._npc_idxs = []
        self._npc_idxs = slice_idxs[:, self._policy_agents_per_env:]\
            .reshape(self._num_envs*self._npc_agents_per_env)

        self._completed_episodes = 0
        self._total_rewards = np.zeros(self._total_agents)
        self._agent_stats = [{} for a in range(self._total_agents)]

        # Extract policy names
        logger.info(f"Policy name: {self._policy_pr.name}")
        if self._npc_pr is not None:
            logger.info(f"NPC name: {self._npc_pr.name}")

        # Create mapping from agent index to policy name
        self._agent_idx_to_policy_name = {}
        for agent_idx in self._policy_idxs:
            self._agent_idx_to_policy_name[agent_idx.item()] = self._policy_pr.name

        for agent_idx in self._npc_idxs:
            self._agent_idx_to_policy_name[agent_idx.item()] = self._npc_pr.name

    def evaluate(self):
        logger.info(f"Evaluating policy: {self._policy_pr.name} with {self._policy_agents_per_env} agents")
        if self._npc_pr is not None:
            logger.info(f"Against npc policy: {self._npc_pr.name} with {self._npc_agents_per_env} agents")

        logger.info(f"Eval settings: {self._num_envs} envs, {self._min_episodes} episodes, {self._max_time_s} seconds")

        obs, _ = self._vecenv.reset()
        policy_rnn_state = None
        npc_rnn_state = None

        game_stats = []
        start = time.time()

        # set of episodes that parallelize the environments
        while self._completed_episodes < self._min_episodes and time.time() - start < self._max_time_s:
            with torch.no_grad():
                obs = torch.as_tensor(obs).to(device=self._device)
                # observavtions that correspond to policy agent
                my_obs = obs[self._policy_idxs]

                # Parallelize across opponents
                policy = self._policy_pr.policy() # policy to evaluate
                if hasattr(policy, 'lstm'):
                    policy_actions, _, _, _, policy_rnn_state, _, _, _ = policy(my_obs, policy_rnn_state)
                else:
                    policy_actions, _, _, _, _, _, _, _ = policy(my_obs)

                # Iterate opponent policies
                if self._npc_pr is not None:
                    npc_obs = obs[self._npc_idxs]
                    npc_rnn_state = npc_rnn_state

                    npc_policy = self._npc_pr.policy()
                    if hasattr(npc_policy, 'lstm'):
                        npc_action, _, _, _, npc_rnn_state, _, _, _ = npc_policy(npc_obs, npc_rnn_state)
                    else:
                        npc_action, _, _, _, _, _, _, _ = npc_policy(npc_obs)

            actions = policy_actions
            if self._npc_agents_per_env > 0:
                actions = torch.cat([
                    policy_actions.view(self._num_envs, self._policy_agents_per_env, -1),
                    npc_action.view(self._num_envs, self._npc_agents_per_env, -1),
                ], dim=1)

            actions = actions.view(self._num_envs*self._agents_per_env, -1)

            obs, rewards, dones, truncated, infos = self._vecenv.step(actions.cpu().numpy())

            self._total_rewards += rewards
            self._completed_episodes += sum([e.done for e in self._vecenv.envs])

             # infos is a list of dictionaries
            if len(infos) > 0:
                for n in range(len(infos)):
                    if "agent_raw" in infos[n]:
                        one_episode = infos[n]["agent_raw"]
                        for m in range(len(one_episode)):
                            agent_idx = m + n * self._agents_per_env
                            if agent_idx in self._agent_idx_to_policy_name:
                                one_episode[m]['policy_name'] = self._agent_idx_to_policy_name[agent_idx].replace("file://", "")
                            else:
                                one_episode[m]['policy_name'] = "No Name Found"
                        game_stats.append(one_episode)


        logger.info(f"Evaluation time: {time.time() - start}")
        self._vecenv.close()
        return game_stats

class EvalSuite:
    def __init__(
        self,
        policy_store: PolicyStore,
        policy_pr: PolicyRecord,
        env_defaults: DictConfig,
        evals: DictConfig = None,
        **kwargs):

        self._evals_cfgs = evals
        self._evals = []
        for eval_name, eval_cfg in evals.items():
            eval_cfg = OmegaConf.merge(eval_cfg, kwargs)
            eval = Eval(policy_store, policy_pr, env_defaults, **eval_cfg)
            self._evals.append(eval)

    def evaluate(self):
        return {
            eval_name: eval.evaluate()
            for eval_name, eval in zip(self._evals_cfgs.keys(), self._evals)
        }
