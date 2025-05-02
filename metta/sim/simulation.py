import logging
import os
import time
from datetime import datetime

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_state import PolicyState
from metta.agent.policy_store import PolicyRecord, PolicyStore
from metta.agent.util.distribution_utils import sample_logits
from metta.sim.replay_helper import ReplayHelper
from metta.sim.simulation_config import SimulationConfig, SimulationSuiteConfig
from metta.sim.vecenv import make_vecenv
from metta.util.datastruct import flatten_config
from mettagrid.mettagrid.curriculum.curriculum import Curriculum

logger = logging.getLogger(__name__)


class Simulation:
    """
    A simulation is any process of stepping through a Mettagrid environment.
    Simulations configure things likes how the policies are mapped to the a
    agents, as well as which environments to run in.

    Simulations are used by training, evaluation and (eventually) play+replay.
    """

    def __init__(
        self,
        config: SimulationConfig,
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
        name: str = "",
        wandb_run=None,
    ):
        self._config = config
        self._wandb_run = wandb_run
        # TODO: Replace with typed EnvConfig
        self._curriculum = Curriculum.from_config_path(config.env, DictConfig(config.env_overrides))
        self._task = self._curriculum.get_task()
        self._env_cfg = self._task.env_cfg()
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

        self._vecenv = make_vecenv(self._curriculum, config.vectorization, num_envs=self._num_envs)

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

        self._agent_stats = [{} for a in range(self._total_agents)]

        # Create mapping from metta.agent index to policy name
        self._agent_idx_to_policy_name = {}
        for agent_idx in self._policy_idxs:
            self._agent_idx_to_policy_name[agent_idx.item()] = self._policy_pr.name

        for agent_idx in self._npc_idxs:
            self._agent_idx_to_policy_name[agent_idx.item()] = self._npc_pr.name

        # Initialize replay helpers array and episode counters if replay is enabled
        self._replay_helpers = None
        self._episode_counters = np.zeros(self._num_envs, dtype=int)
        if config.replay_path is not None:
            self._replay_helpers = []
            for env_idx in range(self._num_envs):
                self._replay_helpers.append(self._create_replay_helper(env_idx))

    def _create_replay_helper(self, env_idx):
        """Create a new replay helper for the specified environment index."""
        return ReplayHelper(
            config=self._config,
            env=self._vecenv.envs[env_idx],
            policy_record=self._policy_pr,
            wandb_run=self._wandb_run,
        )

    def _get_replay_path(self, env_idx, episode_count):
        """Generate a unique replay path for the given environment and episode."""
        base_path = self._config.replay_path

        if self._config.num_envs == 1 and self._config.num_episodes == 1:
            return base_path

        if base_path.startswith("s3://"):
            # Parse S3 path into bucket and key parts
            s3_parts = base_path[5:].split("/", 1)
            bucket = s3_parts[0]

            if len(s3_parts) > 1:
                key_parts = s3_parts[1].rsplit("/", 1)
                if len(key_parts) > 1:
                    prefix = key_parts[0]
                    filename = key_parts[1]
                else:
                    prefix = ""
                    filename = key_parts[0]

                # Add environment and episode identifiers to filename
                new_filename = f"ep{episode_count}_env{env_idx}_{filename}"

                if prefix:
                    return f"s3://{bucket}/{prefix}/{new_filename}"
                else:
                    return f"s3://{bucket}/{new_filename}"
            else:
                return f"s3://{bucket}/ep{episode_count}_env{env_idx}_replay.dat"
        else:
            # For local paths
            directory, filename = os.path.split(base_path)
            if not filename:
                filename = "replay.dat"
            new_filename = f"ep{episode_count}_env{env_idx}_{filename}"
            return os.path.join(directory, new_filename)

    def simulate(self, epoch: int = 0, dry_run: bool = False):
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

        # Track environment completion status
        env_dones = [False] * self._num_envs

        # set of episodes that parallelize the environments
        while (self._episode_counters < self._min_episodes).any() and time.time() - start < self._max_time_s:
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
            actions_np = actions.cpu().numpy()

            if self._replay_helpers is not None:
                actions_per_env = actions_np.reshape(self._num_envs, self._agents_per_env, -1)
                for env_idx in range(self._num_envs):
                    if not env_dones[env_idx]:
                        self._replay_helpers[env_idx].log_pre_step(actions_per_env[env_idx].squeeze())
            # Step the environment
            obs, rewards, dones, truncated, infos = self._vecenv.step(actions_np)

            if self._replay_helpers is not None:
                rewards_per_env = rewards.reshape(self._num_envs, self._agents_per_env)
                for env_idx in range(self._num_envs):
                    if not env_dones[env_idx]:
                        self._replay_helpers[env_idx].log_post_step(rewards_per_env[env_idx])

            # ------------------------------------------------------------------
            # Episode bookkeeping (per-environment finite-state machine)
            # ------------------------------------------------------------------
            # Vector-env returns per-agent flags; collapse to one flag per env
            done_now = np.logical_or(
                dones.reshape(self._num_envs, self._agents_per_env).all(axis=1),
                truncated.reshape(self._num_envs, self._agents_per_env).all(axis=1),
            )

            for env_idx in range(self._num_envs):
                # (1) episode *just* finished
                if done_now[env_idx] and not env_dones[env_idx]:
                    env_dones[env_idx] = True

                    if self._replay_helpers is not None:
                        path = self._get_replay_path(env_idx, self._episode_counters[env_idx])
                        self._replay_helpers[env_idx].write_replay(path, epoch=epoch, dry_run=dry_run)
                    self._episode_counters[env_idx] += 1

                # (2) environment has auto-reset → new episode has started -------------
                elif not done_now[env_idx] and env_dones[env_idx]:
                    env_dones[env_idx] = False  # <-- lets us log steps again
                    if self._replay_helpers is not None:
                        self._replay_helpers[env_idx] = self._create_replay_helper(env_idx)

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

        logger.info(f"Simulation time: {time.time() - start}")
        self._vecenv.close()
        return game_stats


class SimulationSuite:
    def __init__(
        self,
        config: SimulationSuiteConfig,
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
        wandb_run=None,
    ):
        logger.debug(f"Building Simulation suite from config:{config}")
        self._simulations = dict()
        self._wandb_run = wandb_run

        for name, sim_config in config.simulations.items():
            # Create a Simulation object for each config and pass wandb_run directly
            sim = Simulation(
                config=sim_config, policy_pr=policy_pr, policy_store=policy_store, name=name, wandb_run=wandb_run
            )
            self._simulations[name] = sim

    # TODO: epoch and dry_run are replay-specific parameters we could probably handle better
    def simulate(self, epoch: int = 0, dry_run: bool = False):
        # Run all simulations and gather results by name
        return {name: sim.simulate(epoch, dry_run) for name, sim in self._simulations.items()}
