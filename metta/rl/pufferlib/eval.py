import logging
import time
from datetime import datetime

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyRecord, PolicyStore
from metta.rl.pufferlib.vecenv import make_vecenv
from metta.util.config import config_from_path
from metta.util.datastruct import flatten_config

logger = logging.getLogger("eval")


class Eval:
    def __init__(
        self,
        policy_store: PolicyStore,
        policy_pr: PolicyRecord,
        run_id: str,
        env: str,
        npc_policy_uri: str,
        device: str,
        env_overrides: DictConfig = None,
        policy_agents_pct: float = 1.0,
        num_envs: int = 1,
        num_episodes: int = 1,
        max_time_s: int = 60,
        vectorization: str = "serial",
        **kwargs,
    ) -> None:
        self._env_cfg = config_from_path(env, env_overrides)
        self._env_name = env

        self._npc_policy_uri = npc_policy_uri
        self._policy_agents_pct = policy_agents_pct
        self._policy_store = policy_store

        self._device = device

        self._num_envs = num_envs
        self._min_episodes = num_episodes
        self._max_time_s = max_time_s

        # load candidate policy
        self._policy_pr = policy_pr
        self._run_id = run_id
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

        self._vecenv = make_vecenv(self._env_cfg, vectorization, num_envs=self._num_envs)

        # each index is an agent, and we reshape it into a matrix of num_envs x agents_per_env
        # you can figure out which episode you're in by doing the floor division
        slice_idxs = (
            torch.arange(self._vecenv.num_agents).reshape(self._num_envs, self._agents_per_env).to(device=self._device)
        )

        self._policy_idxs = slice_idxs[:, : self._policy_agents_per_env].reshape(
            self._policy_agents_per_env * self._num_envs
        )

        self._npc_idxs = []
        self._npc_idxs = slice_idxs[:, self._policy_agents_per_env :].reshape(self._num_envs * self._npc_agents_per_env)

        self._completed_episodes = 0
        self._total_rewards = np.zeros(self._total_agents)
        self._agent_stats = [{} for a in range(self._total_agents)]

        # Extract policy names
        logger.info(f"Policy name: {self._policy_pr.name}")
        if self._npc_pr is not None:
            logger.info(f"NPC name: {self._npc_pr.name}")

        # Create mapping from metta.agent index to policy name
        self._agent_idx_to_policy_name = {}
        for agent_idx in self._policy_idxs:
            self._agent_idx_to_policy_name[agent_idx.item()] = self._policy_pr.name

        for agent_idx in self._npc_idxs:
            self._agent_idx_to_policy_name[agent_idx.item()] = self._npc_pr.name

    def evaluate(self):
        logger.info(
            f"Evaluating policy: {self._policy_pr.name} in {self._env_name} with {self._policy_agents_per_env} agents"
        )
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
                policy = self._policy_pr.policy()  # policy to evaluate
                policy_actions, _, _, _, policy_rnn_state, _, _, _ = policy(my_obs, policy_rnn_state)

                # Iterate opponent policies
                if self._npc_pr is not None:
                    npc_obs = obs[self._npc_idxs]
                    npc_rnn_state = npc_rnn_state

                    npc_policy = self._npc_pr.policy()
                    npc_action, _, _, _, npc_rnn_state, _, _, _ = npc_policy(npc_obs, npc_rnn_state)

            actions = policy_actions
            if self._npc_agents_per_env > 0:
                actions = torch.cat(
                    [
                        policy_actions.view(self._num_envs, self._policy_agents_per_env, -1),
                        npc_action.view(self._num_envs, self._npc_agents_per_env, -1),
                    ],
                    dim=1,
                )

            actions = actions.view(self._num_envs * self._agents_per_env, -1)

            obs, rewards, dones, truncated, infos = self._vecenv.step(actions.cpu().numpy())

            self._total_rewards += rewards
            self._completed_episodes += sum([e.done for e in self._vecenv.envs])

            # Convert the environment configuration to a dictionary and flatten it.
            game_cfg = OmegaConf.to_container(self._env_cfg.game, resolve=False)
            flattened_env = flatten_config(game_cfg, parent_key="game")
            flattened_env["run_id"] = self._run_id
            flattened_env["eval_name"] = self._env_name
            flattened_env["timestamp"] = datetime.now().isoformat()
            flattened_env["npc"] = self._npc_policy_uri

            for n in range(len(infos)):
                if "agent_raw" in infos[n]:
                    agent_episode_data = infos[n]["agent_raw"]
                    episode_reward = infos[n]["episode_rewards"]
                    for agent_i in range(len(agent_episode_data)):
                        agent_idx = agent_i + n * self._agents_per_env

                        if agent_idx in self._agent_idx_to_policy_name:
                            agent_episode_data[agent_i]["policy_name"] = self._agent_idx_to_policy_name[
                                agent_idx
                            ].replace("file://", "")
                        else:
                            agent_episode_data[agent_i]["policy_name"] = "No Name Found"
                        agent_episode_data[agent_i]["episode_reward"] = episode_reward[agent_i].tolist()
                        agent_episode_data[agent_i].update(flattened_env)

                    game_stats.append(agent_episode_data)
        logger.info(f"Evaluation time: {time.time() - start}")
        self._vecenv.close()
        return game_stats


class EvalSuite:
    def __init__(
        self,
        policy_store: PolicyStore,
        policy_pr: PolicyRecord,
        run_id: str,
        env_overrides: DictConfig = None,
        evals: DictConfig = None,
        **kwargs,
    ):
        self._evals_cfgs = evals
        self._evals = []
        for _eval_name, eval_cfg in evals.items():
            eval_cfg = OmegaConf.merge(kwargs, eval_cfg)
            eval = Eval(policy_store, policy_pr, run_id, env_overrides=env_overrides, **eval_cfg)
            self._evals.append(eval)

    def evaluate(self):
        return {
            eval_name: eval.evaluate() for eval_name, eval in zip(self._evals_cfgs.keys(), self._evals, strict=False)
        }
