# metta/sim/simulation.py
"""
Vectorised simulation runner.

• Launches a MettaGrid vec-env batch
• Each worker writes its own *.duckdb* shard
• At shutdown the shards are merged into **one** StatsDB object that the
  caller can further merge / export.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from metta.agent.policy_state import PolicyState
from metta.agent.policy_store import PolicyRecord, PolicyStore
from metta.sim.replay_helper import ReplayHelper
from metta.sim.simulation_config import SimulationConfig, SimulationSuiteConfig
from metta.sim.stats_db import StatsDB
from metta.sim.vecenv import make_vecenv
from metta.util.config import config_from_path

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#   Single simulation                                                         #
# --------------------------------------------------------------------------- #
class Simulation:
    """
    A vectorised batch of MettaGrid environments sharing the same parameters.
    """

    # ------------------------------------------------------------------ #
    #   construction                                                     #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        name: str,
        config: SimulationConfig,
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
        *,
        wandb_run=None,
        stats_dir: str | None = None,
        replay_dir: str | None = None,
    ):
        self._name = name
        self._config = config
        self._wandb_run = wandb_run

        # ---------------- env config ----------------------------------- #
        self._env_cfg = config_from_path(config.env, config.env_overrides)
        self._env_name = config.env

        self._stats_dir = Path(stats_dir).expanduser() if stats_dir else Path("tmp/stats") / self._name
        self._stats_dir.mkdir(parents=True, exist_ok=True)
        self._replay_dir = replay_dir

        # ---------------- device / vec-env ----------------------------- #
        self._device = config.device
        self._vecenv = make_vecenv(
            self._env_cfg,
            config.vectorization,
            num_envs=config.num_envs,
            stats_writer_dir=str(self._stats_dir),
        )

        self._num_envs = config.num_envs
        self._min_episodes = config.num_episodes
        self._max_time_s = config.max_time_s
        self._agents_per_env = self._env_cfg.game.num_agents

        # ---------------- policies ------------------------------------- #
        self._policy_pr = policy_pr
        self._policy_store = policy_store
        self._npc_pr = policy_store.policy(config.npc_policy_uri) if config.npc_policy_uri else None
        self._policy_agents_pct = config.policy_agents_pct if self._npc_pr is not None else 1.0

        # Let every policy know the active action-set of this env.
        action_names = self._vecenv.driver_env.action_names()
        max_args = self._vecenv.driver_env._c_env.max_action_args()
        self._policy_pr.policy().activate_actions(action_names, max_args, self._device)
        if self._npc_pr is not None:
            self._npc_pr.policy().activate_actions(action_names, max_args, self._device)

        # ---------------- agent-index bookkeeping ---------------------- #
        idx_matrix = torch.arange(self._vecenv.num_agents, device=self._device).reshape(
            self._num_envs, self._agents_per_env
        )
        self._policy_agents_per_env = max(1, int(self._agents_per_env * self._policy_agents_pct))
        self._npc_agents_per_env = self._agents_per_env - self._policy_agents_per_env

        self._policy_idxs = idx_matrix[:, : self._policy_agents_per_env].reshape(-1)
        self._npc_idxs = (
            idx_matrix[:, self._policy_agents_per_env :].reshape(-1)
            if self._npc_agents_per_env
            else torch.tensor([], device=self._device)
        )

        # ---------------- optional replay helpers ---------------------- #
        self._replay_helpers: List[ReplayHelper] | None = None
        self._episode_counters = np.zeros(self._num_envs, dtype=int)
        if replay_dir:
            self._replay_helpers = [self._make_replay_helper(e) for e in range(self._num_envs)]

    # ------------------------------------------------------------------ #
    #   helpers                                                          #
    # ------------------------------------------------------------------ #
    def _make_replay_helper(self, env_idx: int) -> ReplayHelper:
        return ReplayHelper(
            config=self._config,
            env=self._vecenv.envs[env_idx],
            policy_record=self._policy_pr,
            wandb_run=self._wandb_run,
        )

    def _replay_path(self, env_idx: int, ep: int) -> str:
        if env_idx == 0 and ep == 0:
            return f"{self._replay_dir}/replay.json.z"
        return f"{self._replay_dir}/replay_ep{ep}_env{env_idx}.json.z"

    # ------------------------- stats helpers -------------------------- #
    def _merge_worker_dbs(self) -> StatsDB:
        """Merge all *.duckdb* shards for this simulation → one `StatsDB`."""
        agent_map: Dict[int, Tuple[str, str | None]] = {
            idx.item(): self._policy_pr.key_and_version() for idx in self._policy_idxs
        }
        if self._npc_pr is not None:
            agent_map.update({idx.item(): self._npc_pr.key_and_version() for idx in self._npc_idxs})

        db = StatsDB.merge_worker_dbs(self._stats_dir, agent_map)
        logger.info("Merged %s → %s", self._stats_dir, db.path)
        return db

    # ------------------------------------------------------------------ #
    #   public API                                                       #
    # ------------------------------------------------------------------ #
    def simulate(self) -> StatsDB:
        """
        Run the simulation; returns the merged `StatsDB`.
        """
        logger.info(
            "Sim '%s': %d env × %d agents (%.0f%% candidate)",
            self._name,
            self._num_envs,
            self._agents_per_env,
            100 * self._policy_agents_per_env / self._agents_per_env,
        )
        logger.info("Stats dir: %s", self._stats_dir)
        # ---------------- reset ------------------------------- #
        obs, _ = self._vecenv.reset()
        policy_state = PolicyState()
        npc_state = PolicyState()
        env_done_flags = [False] * self._num_envs

        t0 = time.time()
        while (self._episode_counters < self._min_episodes).any() and (time.time() - t0) < self._max_time_s:
            # ---------------- forward passes ------------------------- #
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self._device)

                # Candidate-policy agents
                my_obs = obs_t[self._policy_idxs]
                policy = self._policy_pr.policy()
                policy_actions, _, _, _, _ = policy(my_obs, policy_state)

                # NPC agents (if any)
                if self._npc_pr is not None and len(self._npc_idxs):
                    npc_obs = obs_t[self._npc_idxs]
                    npc_policy = self._npc_pr.policy()
                    npc_actions, _, _, _, _ = npc_policy(npc_obs, npc_state)

            # ---------------- action stitching ----------------------- #
            acts_t = policy_actions
            if self._npc_agents_per_env:
                acts_t = torch.cat(
                    [
                        policy_actions.view(self._num_envs, self._policy_agents_per_env, -1),
                        npc_actions.view(self._num_envs, self._npc_agents_per_env, -1),
                    ],
                    dim=1,
                ).view(-1, policy_actions.shape[-1])
            acts_np = acts_t.cpu().numpy()

            # ---------------- replay (pre-step) ----------------------- #
            if self._replay_helpers:
                per_env = acts_np.reshape(self._num_envs, self._agents_per_env, -1)
                for e in range(self._num_envs):
                    if not env_done_flags[e]:
                        self._replay_helpers[e].log_pre_step(per_env[e].squeeze())

            # ---------------- env.step ------------------------------- #
            obs, rewards, dones, trunc, _ = self._vecenv.step(acts_np)

            # ---------------- replay (post-step) ---------------------- #
            if self._replay_helpers:
                per_env_r = rewards.reshape(self._num_envs, self._agents_per_env)
                for e in range(self._num_envs):
                    if not env_done_flags[e]:
                        self._replay_helpers[e].log_post_step(per_env_r[e])

            # ---------------- episode FSM ---------------------------- #
            done_now = np.logical_or(
                dones.reshape(self._num_envs, self._agents_per_env).all(1),
                trunc.reshape(self._num_envs, self._agents_per_env).all(1),
            )
            for e in range(self._num_envs):
                if done_now[e] and not env_done_flags[e]:
                    env_done_flags[e] = True
                    if self._replay_helpers:
                        path = self._replay_path(e, self._episode_counters[e])
                        self._replay_helpers[e].write_replay(path)
                    self._episode_counters[e] += 1
                elif not done_now[e] and env_done_flags[e]:
                    env_done_flags[e] = False
                    if self._replay_helpers:
                        self._replay_helpers[e] = self._make_replay_helper(e)

        # ---------------- teardown & DB merge ------------------------ #
        self._vecenv.close()
        db = self._merge_worker_dbs()

        logger.info(
            "Sim '%s' finished: %d episodes in %.1fs",
            self._name,
            int(self._episode_counters.sum()),
            time.time() - t0,
        )
        return db


# --------------------------------------------------------------------------- #
#   Suite of simulations                                                      #
# --------------------------------------------------------------------------- #
class SimulationSuite:
    """
    Runs a collection of named simulations and returns **one merged StatsDB**
    containing the union of their statistics.
    """

    def __init__(
        self,
        config: SimulationSuiteConfig,
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
        *,
        wandb_run=None,
        stats_dir: str | None = None,
        replay_dir: str | None = None,
    ):
        self._sims: Dict[str, Simulation] = {
            n: Simulation(
                n,
                cfg,
                policy_pr,
                policy_store,
                wandb_run=wandb_run,
                stats_dir=f"{stats_dir}/{n}" if stats_dir else None,
                replay_dir=f"{replay_dir}/{n}" if replay_dir else None,
            )
            for n, cfg in config.simulations.items()
        }

    # ------------------------------------------------------------------ #
    #   public API                                                       #
    # ------------------------------------------------------------------ #
    def simulate(self) -> StatsDB:
        """Run every simulation, merge their DBs, return a single `StatsDB`."""
        merged_db: StatsDB | None = None

        for name, sim in self._sims.items():
            logger.info("=== Simulation '%s' ===", name)
            db = sim.simulate()

            if merged_db is None:
                merged_db = db
            else:
                merged_db.merge_in(db)
                db.close()  # release file handle of merged shard

        assert merged_db is not None, "SimulationSuite contained no simulations"
        return merged_db
