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
    def simulate(self) -> SimulationResults:
        """
        Run the simulation; returns the merged `StatsDB`.
        """
        # logger.info(
        #     "Sim '%s': %d env × %d agents (%.0f%% candidate)",
        #     self._name,
        #     self._num_envs,
        #     self._agents_per_env,
        #     100 * self._policy_agents_per_env / self._agents_per_env,
        # )
        # logger.info("Stats dir: %s", self._stats_dir)
        # ---------------- reset ------------------------------- #

        start = time.time()

        assert len(self._vecenv.envs) == 1
        env = self._vecenv.envs[0]

        obs, _ = env.reset()
        policy_state = PolicyState()
        npc_state = PolicyState()
        env_done_flags = [False] * self._num_envs
        end = time.time()
        print("startup time", end - start)

        start = time.time()
        t0 = time.time()

        cum_nn_time = 0
        cum_sim_time = 0

        policy = self._policy_pr.policy()
        print("policy", policy)

        # while (self._episode_counters < self._min_episodes).any() and (time.time() - t0) < self._max_time_s:
        for x in range(1000):
            # ---------------- forward passes ------------------------- #
            start_nn = time.time()
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self._device)

                # Candidate-policy agents
                my_obs = obs_t[self._policy_idxs]
                policy_actions, _, _, _, _ = policy(my_obs, policy_state)
                # print("policy_actions", policy_actions)

                # NPC agents (if any)
                # if self._npc_pr is not None and len(self._npc_idxs):
                #     npc_obs = obs_t[self._npc_idxs]
                #     npc_policy = self._npc_pr.policy()
                #     npc_actions, _, _, _, _ = npc_policy(npc_obs, npc_state)

            # ---------------- action stitching ----------------------- #
            actions = policy_actions
            # if self._npc_agents_per_env:
            #     actions = torch.cat(
            #         [
            #             policy_actions.view(self._num_envs, self._policy_agents_per_env, -1),
            #             npc_actions.view(self._num_envs, self._npc_agents_per_env, -1),
            #         ],
            #         dim=1,
            #     )

            actions = actions.view(self._num_envs * self._agents_per_env, -1)
            actions_np = actions.cpu().numpy()

            end_nn = time.time()
            cum_nn_time += end_nn - start_nn

            # ---------------- env.step ------------------------------- #
            start_step = time.time()
            obs, _, dones, trunc, _ = env.step(actions_np)
            end_step = time.time()
            # print("sim time: ", end_step - start_step)
            cum_sim_time += end_step - start_step

            # ---------------- episode FSM ---------------------------- #
            # done_now = np.logical_or(
            #     dones.reshape(self._num_envs, self._agents_per_env).all(1),
            #     trunc.reshape(self._num_envs, self._agents_per_env).all(1),
            # )
            # for e in range(self._num_envs):
            #     if done_now[e] and not env_done_flags[e]:
            #         env_done_flags[e] = True
            #         self._episode_counters[e] += 1
            #     elif not done_now[e] and env_done_flags[e]:
            #         env_done_flags[e] = False

        end = time.time()
        print("total", end - start)
        print("  cumulative nn time", cum_nn_time)
        print("  cumulative sim time", cum_sim_time)
        print("    SPS", 1 / (cum_sim_time / 1000 / self._agents_per_env))

        # ---------------- teardown & DB merge ------------------------ #
        env.close()
        # db = self._from_shards_and_context()

        # logger.info(
        #     "Sim '%s' finished: %d episodes in %.1fs",
        #     self._name,
        #     int(self._episode_counters.sum()),
        #     time.time() - t0,
        # )
        # return SimulationResults(db)

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
