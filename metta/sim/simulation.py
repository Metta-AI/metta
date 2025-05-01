"""
<<<<<<< HEAD
Runs a vectorised batch of MettaGrid environments, merges the per-env shard
DBs written by `MettaGridStatsWriter`, and exports a single canonical DB to
`eval_stats_uri` (file / S3 / WandB).

The only stats dependency is `mettagrid.stats_db`.  No mettagrid code imports
from metta.
=======
Simulations are how mettagrid envs are run. Its main reponsibilities include:

* Mapping policies to agents
* Running a vectorised batch of MettaGrid environments
* Merging the per-env shard DBs written by MettaGridStatsWriter
* Exporting a single canonical DB to eval_stats_uri (file / S3 / WandB)
>>>>>>> e6febb77 (cp)
"""

import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from metta.agent.policy_state import PolicyState
from metta.agent.policy_store import PolicyRecord, PolicyStore
from metta.agent.util.distribution_utils import sample_logits
<<<<<<< HEAD
from metta.sim.replay_helper import ReplayHelper
=======
>>>>>>> e6febb77 (cp)
from metta.sim.simulation_config import SimulationConfig, SimulationSuiteConfig
from metta.sim.stats_db import StatsDB
from metta.sim.vecenv import make_vecenv
from metta.util.config import config_from_path
from metta.util.datastruct import flatten_config

logger = logging.getLogger(__name__)


class Simulation:
<<<<<<< HEAD
    """
    A simulation is any process of stepping through a MettaGrid environment.

    *Maps candidate & NPC policies onto agents, steps a vectorised environment,
    optionally records replays, and finally merges / exports StatsDB shards.*
    """

    # ------------------------------------------------------------------ #
    #   construction                                                     #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        config: SimulationConfig,
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
        *,
        name: str = "",
        wandb_run=None,
    ):
        self._config = config
        self._wandb_run = wandb_run
        self._name = name or "default"

        # -------- env config ------------------------------------------- #
        self._env_cfg = config_from_path(config.env, config.env_overrides)
        self._env_name = config.env

        # Each sim writes its own DuckDB file; later merged → canonical DB
        self._stats_dir = Path(config.run_dir) / "stats" / self._name
        self._stats_dir.mkdir(parents=True, exist_ok=True)
        self._env_cfg["stats_writer_path"] = str(self._stats_dir / f"stats_{uuid.uuid4().hex[:8]}.duckdb")

        # -------- policies --------------------------------------------- #
        self._policy_pr = policy_pr
        self._npc_policy_uri = config.npc_policy_uri
        self._policy_store = policy_store

        self._npc_pr = self._policy_store.policy(self._npc_policy_uri) if self._npc_policy_uri else None
        self._policy_agents_pct = config.policy_agents_pct if self._npc_pr else 1.0
=======
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

        # Setup stats writer path
        self._stats_dir = Path(config.run_dir) / "stats" / (name or "default")
        self._stats_dir.mkdir(parents=True, exist_ok=True)
        self._env_cfg["stats_writer_path"] = str(self._stats_dir / f"stats_{uuid.uuid4().hex[:8]}.duckdb")

        # load candidate policy
        self._policy_pr = policy_pr
        self._name = name
        # load npc policy
        self._npc_pr = None
        if self._npc_policy_uri is None:
            self._policy_agents_pct = 1.0
        else:
            self._npc_pr = self._policy_store.policy(self._npc_policy_uri)
>>>>>>> e6febb77 (cp)

        # -------- vec-env & agent bookkeeping -------------------------- #
        self._device = config.device
        self._num_envs = config.num_envs
        self._min_episodes = config.num_episodes
        self._max_time_s = config.max_time_s

        self._vecenv = make_vecenv(self._env_cfg, config.vectorization, num_envs=self._num_envs)
        self._agents_per_env = self._env_cfg.game.num_agents
<<<<<<< HEAD

        # Pre-compute slices (policy vs NPC) into the flattened agent axis
        slice_mat = torch.arange(self._vecenv.num_agents).reshape(self._num_envs, self._agents_per_env).to(self._device)
        self._policy_agents_per_env = max(1, int(self._agents_per_env * self._policy_agents_pct))
        self._npc_agents_per_env = self._agents_per_env - self._policy_agents_per_env
        self._policy_idxs = slice_mat[:, : self._policy_agents_per_env].reshape(-1)
        self._npc_idxs = (
            slice_mat[:, self._policy_agents_per_env :].reshape(-1)
            if self._npc_agents_per_env
            else torch.tensor([], device=self._device)
        )

        # -------- replay helpers --------------------------------------- #
        self._replay_helpers: list[ReplayHelper] | None = None
        self._episode_counters = np.zeros(self._num_envs, dtype=int)
        if config.replay_path:
            self._replay_helpers = [self._create_replay_helper(env_i) for env_i in range(self._num_envs)]

        # -------- misc -------------------------------------------------- #
        # Map agent-idx → policy-name once for whole run (used in stats)
        self._agent_idx_to_policy_name = {
            **{idx.item(): self._policy_pr.name for idx in self._policy_idxs},
            **(
                {idx.item(): self._npc_pr.name for idx in self._npc_idxs}
                if self._npc_pr is not None and len(self._npc_idxs)
                else {}
            ),
        }

    # ------------------------------------------------------------------ #
    #   helpers                                                          #
    # ------------------------------------------------------------------ #
    def _create_replay_helper(self, env_idx: int) -> ReplayHelper:
        """One helper per environment, re-created on every reset."""
        return ReplayHelper(
            config=self._config,
            env=self._vecenv.envs[env_idx],
            policy_record=self._policy_pr,
            wandb_run=self._wandb_run,
        )

    # -- replay path ---------------------------------------------------- #
    def _get_replay_path(self, env_idx: int, episode_count: int) -> str:
        """Generate a unique (local or S3) path per env-episode."""
        base = self._config.replay_path
        if self._num_envs == 1 and self._min_episodes == 1:
            return base

        if base.startswith("s3://"):
            bucket, key = base[5:].split("/", 1)
            prefix, fname = os.path.split(key)
            new_fname = f"ep{episode_count}_env{env_idx}_{fname or 'replay.dat'}"
            return f"s3://{bucket}/{prefix}/{new_fname}" if prefix else f"s3://{bucket}/{new_fname}"

        # local
        directory, fname = os.path.split(base)
        fname = fname or "replay.dat"
        return os.path.join(directory, f"ep{episode_count}_env{env_idx}_{fname}")

    # -- stats DB merge/export ----------------------------------------- #
    def _finalise_stats(self) -> None:
        """Merge worker DuckDB shards → single DB, optionally export."""
=======
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

    def _finalise_stats(self) -> None:
        # Build agent-policy mapping
>>>>>>> e6febb77 (cp)
        agent_map: Dict[int, Tuple[str, int]] = {
            idx.item(): (self._policy_pr.uri, self._policy_pr.version) for idx in self._policy_idxs
        }
        if self._npc_pr is not None:
            agent_map.update({idx.item(): (self._npc_pr.uri, self._npc_pr.version) for idx in self._npc_idxs})

        merged_db = StatsDB.merge_worker_dbs(self._stats_dir, agent_map)
        logger.info(
            "Merged %d shards for '%s' → %s", len(list(self._stats_dir.glob("*.duckdb"))), self._name, merged_db
        )

        if self._config.eval_stats_uri:
            StatsDB.export_db(merged_db, self._config.eval_stats_uri)
            logger.info("Exported stats DB → %s", self._config.eval_stats_uri)
<<<<<<< HEAD

    # ------------------------------------------------------------------ #
    #   main loop                                                        #
    # ------------------------------------------------------------------ #
    def simulate(self, *, epoch: int = 0, dry_run: bool = False):
        logger.info(
            "Simulating '%s' – policy=%s  env=%s  policy_agents/env=%d",
            self._name,
            self._policy_pr.name,
            self._env_name,
            self._policy_agents_per_env,
        )
        if self._npc_pr:
            logger.debug(
                "           against NPC policy=%s (%d npc agents/env)", self._npc_pr.name, self._npc_agents_per_env
            )

=======

    def simulate(self):
        logger.info(
            f"Simulating {self._name} policy: {self._policy_pr.name} "
            + f"in {self._env_name} with {self._policy_agents_per_env} agents"
        )
        if self._npc_pr is not None:
            logger.debug(f"Against npc policy: {self._npc_pr.name} with {self._npc_agents_per_env} agents")

        logger.info(f"Simulation settings: {self._config}")

>>>>>>> e6febb77 (cp)
        obs, _ = self._vecenv.reset()
        policy_state = PolicyState()
        npc_state = PolicyState()

<<<<<<< HEAD
        game_stats: list[dict] = []
        env_dones = [False] * self._num_envs
        t0 = time.time()

        while (self._episode_counters < self._min_episodes).any() and (time.time() - t0) < self._max_time_s:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self._device)

                # candidate policy
                pol_logits, _ = self._policy_pr.policy()(obs_t[self._policy_idxs], policy_state)
                pol_actions, _, _, _ = sample_logits(pol_logits)

                # npc policy
                if self._npc_pr and len(self._npc_idxs):
                    npc_logits, _ = self._npc_pr.policy()(obs_t[self._npc_idxs], npc_state)
                    npc_actions, _, _, _ = sample_logits(npc_logits)

            # stitch [policy | npc] actions back to vec-env layout
            actions_t = pol_actions
            if self._npc_agents_per_env:
                actions_t = torch.cat(
                    [
                        pol_actions.view(self._num_envs, self._policy_agents_per_env, -1),
                        npc_actions.view(self._num_envs, self._npc_agents_per_env, -1),
                    ],
                    dim=1,
                ).view(-1, pol_actions.shape[-1])

            actions_np = actions_t.cpu().numpy()

            # log pre-step actions to replay
            if self._replay_helpers:
                act_env = actions_np.reshape(self._num_envs, self._agents_per_env, -1)
                for e in range(self._num_envs):
                    if not env_dones[e]:
                        self._replay_helpers[e].log_pre_step(act_env[e].squeeze())

            # STEP
            obs, rewards, dones, truncated, infos = self._vecenv.step(actions_np)

            # log rewards post-step
            if self._replay_helpers:
                rew_env = rewards.reshape(self._num_envs, self._agents_per_env)
                for e in range(self._num_envs):
                    if not env_dones[e]:
                        self._replay_helpers[e].log_post_step(rew_env[e])

            # ------------------------------------------------------ #
            #   episode / env-reset FSM                              #
            # ------------------------------------------------------ #
            done_now = np.logical_or(
                dones.reshape(self._num_envs, self._agents_per_env).all(axis=1),
                truncated.reshape(self._num_envs, self._agents_per_env).all(axis=1),
            )

            for e in range(self._num_envs):
                # (1) episode just finished
                if done_now[e] and not env_dones[e]:
                    env_dones[e] = True
                    if self._replay_helpers:
                        path = self._get_replay_path(e, self._episode_counters[e])
                        self._replay_helpers[e].write_replay(path, epoch=epoch, dry_run=dry_run)
                    self._episode_counters[e] += 1

                # (2) auto-reset → new episode started
                elif not done_now[e] and env_dones[e]:
                    env_dones[e] = False
                    if self._replay_helpers:
                        self._replay_helpers[e] = self._create_replay_helper(e)

            # ------------------------------------------------------ #
            #   per-agent episode stats (flatten & stash)            #
            # ------------------------------------------------------ #
            game_cfg = OmegaConf.to_container(self._env_cfg.game, resolve=False)
            flat_env = flatten_config(game_cfg, parent_key="game")
            flat_env.update(
                {
                    "eval_name": self._name,
                    "timestamp": datetime.now().isoformat(),
                    "npc": self._npc_policy_uri,
                }
            )

            for n in range(len(infos)):
                if "agent_raw" in infos[n]:
                    ep_data = infos[n]["agent_raw"]
                    ep_rew = infos[n]["episode_rewards"]
                    for a_i, data in enumerate(ep_data):
                        agent_idx = a_i + n * self._agents_per_env
                        data["policy_name"] = self._agent_idx_to_policy_name.get(agent_idx, "unknown").replace(
                            "file://", ""
                        )
                        data["episode_reward"] = ep_rew[a_i].tolist()
                        data.update(flat_env)
                    game_stats.append(ep_data)

        # done
        self._vecenv.close()
        self._finalise_stats()
        logger.info(
            "Simulation '%s' finished (%d episodes, %.1fs).",
            self._name,
            int(self._episode_counters.sum()),
            time.time() - t0,
        )
        return game_stats


# --------------------------------------------------------------------------- #
#   Suite of simulations                                                      #
# --------------------------------------------------------------------------- #
class SimulationSuite:
    """
    Thin wrapper around a dict {name → Simulation}.  Delegates `.simulate()`
    and aggregates the results keyed by simulation-name.
    """

=======
        completed_episodes = 0
        start = time.time()

        # set of episodes that parallelize the environments
        while completed_episodes < self._min_episodes and time.time() - start < self._max_time_s:
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

            obs, _, dones, _, _ = self._vecenv.step(actions.cpu().numpy())
            completed_episodes += sum([e.done for e in self._vecenv.envs])

        logger.debug(f"Simulation time: {time.time() - start}")
        self._vecenv.close()
        self._finalise_stats()
        logger.info("Simulation %s finished after %d episodes.", self._name, completed_episodes)


class SimulationSuite:
>>>>>>> e6febb77 (cp)
    def __init__(
        self,
        config: SimulationSuiteConfig,
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
<<<<<<< HEAD
        *,
        wandb_run=None,
    ):
        self._wandb_run = wandb_run
        self._sims: Dict[str, Simulation] = {
            name: Simulation(sim_cfg, policy_pr, policy_store, name=name, wandb_run=wandb_run)
            for name, sim_cfg in config.simulations.items()
        }

    # Replay-specific `epoch` / `dry_run` bubbled through for caller convenience
    def simulate(self, *, epoch: int = 0, dry_run: bool = False) -> Dict[str, list]:
        return {n: sim.simulate(epoch=epoch, dry_run=dry_run) for n, sim in self._sims.items()}
=======
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
>>>>>>> e6febb77 (cp)
