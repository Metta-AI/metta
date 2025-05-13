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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from metta.agent.policy_state import PolicyState
from metta.agent.policy_store import PolicyRecord, PolicyStore
from metta.sim.simulation_config import SimulationConfig
from metta.sim.simulation_stats_db import SimulationStatsDB
from metta.sim.vecenv import make_vecenv
from metta.util.config import config_from_path
from metta.util.tracing import trace
from mettagrid.replay_writer import ReplayWriter
from mettagrid.stats_writer import StatsWriter

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
    @trace
    def __init__(
        self,
        name: str,
        config: SimulationConfig,
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
        suite=None,
        stats_dir: str = "/tmp/stats",
        replay_dir: str | None = None,
    ):
        self._name = name
        self._suite = suite
        self._config = config
        self._id = uuid.uuid4().hex[:12]

        # ---------------- env config ----------------------------------- #
        self._env_cfg = config_from_path(config.env, config.env_overrides)
        self._env_name = config.env

        replay_dir = f"{replay_dir}/{self._id}" if replay_dir else None

        sim_stats_dir = (Path(stats_dir) / self._id).resolve()
        sim_stats_dir.mkdir(parents=True, exist_ok=True)
        self._stats_dir = sim_stats_dir
        self._stats_writer = StatsWriter(sim_stats_dir)
        self._replay_writer = ReplayWriter(replay_dir)
        self._device = config.device
        logger.debug(f"Creating vecenv with {config.num_envs} environments")
        self._vecenv = make_vecenv(
            self._env_cfg,
            config.vectorization,
            num_envs=config.num_envs,
            stats_writer=self._stats_writer,
            replay_writer=self._replay_writer,
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
        self._episode_counters = np.zeros(self._num_envs, dtype=int)

    # ------------------------------------------------------------------ #
    #   public API                                                       #
    # ------------------------------------------------------------------ #
    @trace
    def start_simulation(self):
        """
        Start the simulation.
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
        self._obs, _ = self._vecenv.reset()
        self._policy_state = PolicyState()
        self._npc_state = PolicyState()
        self._env_done_flags = [False] * self._num_envs

        self._t0 = time.time()

    @trace
    def generate_actions(self):
        """
        Generate actions for the simulation.
        """
        # ---------------- forward passes ------------------------- #
        with torch.no_grad():
            obs_t = torch.as_tensor(self._obs, device=self._device)

            # Candidate-policy agents
            my_obs = obs_t[self._policy_idxs]
            policy = self._policy_pr.policy()
            policy_actions, _, _, _, _ = policy(my_obs, self._policy_state)

            # NPC agents (if any)
            if self._npc_pr is not None and len(self._npc_idxs):
                npc_obs = obs_t[self._npc_idxs]
                npc_policy = self._npc_pr.policy()
                npc_actions, _, _, _, _ = npc_policy(npc_obs, self._npc_state)

        # ---------------- action stitching ----------------------- #
        actions = policy_actions
        if self._npc_agents_per_env:
            actions = torch.cat(
                [
                    policy_actions.view(self._num_envs, self._policy_agents_per_env, -1),
                    npc_actions.view(self._num_envs, self._npc_agents_per_env, -1),
                ],
                dim=1,
            )

        actions = actions.view(self._num_envs * self._agents_per_env, -1)
        actions_np = actions.cpu().numpy()

        return actions_np

    @trace
    def step_simulation(self, actions_np: np.ndarray):
        # ---------------- env.step ------------------------------- #
        obs, _, dones, trunc, _ = self._vecenv.step(actions_np)

        # ---------------- episode FSM ---------------------------- #
        done_now = np.logical_or(
            dones.reshape(self._num_envs, self._agents_per_env).all(1),
            trunc.reshape(self._num_envs, self._agents_per_env).all(1),
        )
        for e in range(self._num_envs):
            if done_now[e] and not self._env_done_flags[e]:
                self._env_done_flags[e] = True
                self._episode_counters[e] += 1
            elif not done_now[e] and self._env_done_flags[e]:
                self._env_done_flags[e] = False

    @trace
    def end_simulation(self) -> SimulationResults:
        # ---------------- teardown & DB merge ------------------------ #
        self._vecenv.close()
        db = self._from_shards_and_context()

        logger.info(
            "Sim '%s' finished: %d episodes in %.1fs",
            self._name,
            int(self._episode_counters.sum()),
            time.time() - self._t0,
        )
        return SimulationResults(db)

    @trace
    def simulate(self) -> SimulationResults:
        """
        Run the simulation; returns the merged `StatsDB`.
        """
        self.start_simulation()

        while (self._episode_counters < self._min_episodes).any() and (time.time() - self._t0) < self._max_time_s:
            action = self.generate_actions()
            self.step_simulation(action)

        return self.end_simulation()

    # ------------------------- stats helpers -------------------------- #
    def _from_shards_and_context(self) -> SimulationStatsDB:
        """Merge all *.duckdb* shards for this simulation → one `StatsDB`."""
        agent_map: Dict[int, Tuple[str, int]] = {idx.item(): self._policy_pr for idx in self._policy_idxs}
        if self._npc_pr is not None:
            agent_map.update({idx.item(): self._npc_pr for idx in self._npc_idxs})

        suite_name = "" if self._suite is None else self._suite.name
        db = SimulationStatsDB.from_shards_and_context(
            self._id, self._stats_dir, agent_map, self._name, suite_name, self._config.env, self._policy_pr
        )
        return db


@dataclass
class SimulationResults:
    """
    Results of a simulation.
    For now just a stats db. Replay plays can be retrieved from the stats db.
    """

    stats_db: SimulationStatsDB
