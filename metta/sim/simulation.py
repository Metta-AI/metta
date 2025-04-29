import logging
import time
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf

from metta.agent.policy_state import PolicyState
from metta.agent.policy_store import PolicyRecord, PolicyStore
from metta.agent.util.distribution_utils import sample_logits
from metta.sim.simulation_config import SimulationConfig, SimulationSuiteConfig
from metta.sim.vecenv import make_vecenv
from metta.util.config import config_from_path
from metta.util.datastruct import flatten_config

logger = logging.getLogger(__name__)


class Simulation:
    """
    A simulation is any process of stepping through a Mettagrid environment.
    Simulations configure things likes how the policies are mapped to the a
    agents, as well as which environments to run in.

    Simulations are used by training, evaluation and (eventually) play+replay.
    """

    def __init__(self, config: SimulationConfig, policy_pr: PolicyRecord, policy_store: PolicyStore, name: str = ""):
        self._config = config
        # TODO: Replace with typed EnvConfig
        self._env_cfg = config_from_path(config.env, config.env_overrides)
        self._env_name = config.env

        self._npc_policy_uri = config.npc_policy_uri
        self._policy_agents_pct = config.policy_agents_pct
        self._policy_store = policy_store

        self._device = config.device

        self._num_envs = config.num_envs
        self._min_episodes = config.num_episodes
        self._max_time_s = config.max_time_s

        # load candidate policy
        self._policy_pr = policy_pr
        self._name = name
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

        self._vecenv = make_vecenv(self._env_cfg, config.vectorization, num_envs=self._num_envs)

        # each index is an agent, and we reshape it into a matrix of num_envs x agents_per_env
        slice_idxs = (
            torch.arange(self._vecenv.num_agents).reshape(self._num_envs, self._agents_per_env).to(device=self._device)
        )

        self._policy_idxs = slice_idxs[:, : self._policy_agents_per_env].reshape(
            self._policy_agents_per_env * self._num_envs
        )

        self._npc_idxs = []
        if self._npc_agents_per_env > 0:
            self._npc_idxs = slice_idxs[:, self._policy_agents_per_env :].reshape(
                self._num_envs * self._npc_agents_per_env
            )

        self._completed_episodes = 0
        self._total_rewards = np.zeros(self._total_agents)
        self._agent_stats = [{} for a in range(self._total_agents)]

        # Create mapping from metta.agent index to policy name
        self._agent_idx_to_policy_name = {}
        for agent_idx in self._policy_idxs:
            self._agent_idx_to_policy_name[agent_idx.item()] = self._policy_pr.name

        for agent_idx in self._npc_idxs:
            self._agent_idx_to_policy_name[agent_idx.item()] = self._npc_pr.name

    def simulate(self):
        logger.info(
            f"Simulating {self._name} policy: {self._policy_pr.name} "
            + f"in {self._env_name} with {self._policy_agents_per_env} agents"
        )
        if self._npc_pr is not None:
            logger.debug(f"Against npc policy: {self._npc_pr.name} with {self._npc_agents_per_env} agents")

        logger.info(f"Simulation settings: {self._config}")

        obs, _ = self._vecenv.reset()
        policy_state = PolicyState()
        npc_state = PolicyState()

        game_stats = []
        start = time.time()

        # set of episodes that parallelize the environments
        while self._completed_episodes < self._min_episodes and time.time() - start < self._max_time_s:
            with torch.no_grad():
                obs = torch.as_tensor(obs).to(device=self._device)
                # observations that correspond to policy agent
                my_obs = obs[self._policy_idxs]

                # Parallelize across opponents
                policy = self._policy_pr.policy()  # policy to evaluate
                logits, _ = policy(my_obs, policy_state)
                policy_actions, _, _, _ = sample_logits(logits)

                # Iterate opponent policies
                if self._npc_pr is not None:
                    npc_obs = obs[self._npc_idxs]
                    npc_policy = self._npc_pr.policy()
                    npc_logits, _ = npc_policy(npc_obs, npc_state)
                    npc_actions, _, _, _ = sample_logits(npc_logits)

            actions = policy_actions
            if self._npc_agents_per_env > 0:
                actions = torch.cat(
                    [
                        policy_actions.view(self._num_envs, self._policy_agents_per_env, -1),
                        npc_actions.view(self._num_envs, self._npc_agents_per_env, -1),
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
            flattened_env["eval_name"] = self._name
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
        logger.debug(f"Simulation time: {time.time() - start}")
        self._vecenv.close()
        return game_stats


class SimulationSuite:
    def __init__(
        self,
        config: SimulationSuiteConfig,
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
    ):
        logger.debug(f"Building Simulation suite from config:{config}")
        self._simulations = dict()

        for name, sim_config in config.simulations.items():
            # Create a Simulation object for each config
            sim = Simulation(config=sim_config, policy_pr=policy_pr, policy_store=policy_store, name=name)
            self._simulations[name] = sim

    def simulate(self):
        # Run all simulations and gather results by name
        return {name: sim.simulate() for name, sim in self._simulations.items()}
