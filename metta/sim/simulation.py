# metta/sim/simulation.py
"""
Runs a vectorised batch of MettaGrid environments, merges the per-env shard
DBs written by MettaGridStatsWriter, and exports a single canonical DB to
`eval_stats_uri` (file / S3 / WandB).

The only stats dependency is mettagrid.stats_db.  No mettagrid code imports
from metta.
"""

from __future__ import annotations
import logging, time, uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import numpy as np, torch
from omegaconf import OmegaConf

from mettagrid.stats_db import StatsDB  # ← run-level DB helper
from metta.agent.policy_store import PolicyRecord, PolicyStore
from metta.sim.simulation_config import SimulationConfig, SimulationSuiteConfig
from metta.sim.vecenv import make_vecenv
from metta.util.config import config_from_path
from metta.util.datastruct import flatten_config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#   Single simulation                                                         #
# --------------------------------------------------------------------------- #
class Simulation:
    def __init__(
        self,
        cfg: SimulationConfig,
        policy_pr: PolicyRecord,
        store: PolicyStore,
        name: str = "",
    ) -> None:
        self._cfg, self._name = cfg, name or "default"

        # ---------------- env config ----------------------------------- #
        self._env_cfg = config_from_path(cfg.env, cfg.env_overrides)
        self._stats_dir = Path(cfg.run_dir) / "stats" / self._name
        self._stats_dir.mkdir(parents=True, exist_ok=True)
        self._env_cfg["stats_writer_path"] = str(self._stats_dir / f"stats_{uuid.uuid4().hex[:8]}.duckdb")

        # ---------------- policies ------------------------------------- #
        self._policy_pr = policy_pr
        self._npc_pr = store.policy(cfg.npc_policy_uri) if cfg.npc_policy_uri else None
        self._policy_agents_pct = cfg.policy_agents_pct if self._npc_pr else 1.0

        # ---------------- vecenv --------------------------------------- #
        self._device = cfg.device
        self._vecenv = make_vecenv(self._env_cfg, cfg.vectorization, num_envs=cfg.num_envs)

        self._agents_per_env = self._env_cfg.game.num_agents
        self._policy_idxs, self._npc_idxs = self._agent_slices(cfg.num_envs)
        self._min_episodes, self._max_time_s = cfg.num_episodes, cfg.max_time_s

    # ------------------------------------------------------------------ #
    #   helper – agent index slices                                       #
    # ------------------------------------------------------------------ #
    def _agent_slices(self, n_envs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p_cnt = max(1, int(self._agents_per_env * self._policy_agents_pct))
        npc_cnt = self._agents_per_env - p_cnt
        m = torch.arange(self._vecenv.num_agents).reshape(n_envs, self._agents_per_env).to(self._device)
        pol = m[:, :p_cnt].reshape(-1)
        npc = m[:, p_cnt:].reshape(-1) if npc_cnt else torch.tensor([], device=self._device)
        return pol, npc

    # ------------------------------------------------------------------ #
    #   merge shards + export                                             #
    # ------------------------------------------------------------------ #
    def _finalise_stats(self) -> None:
        # Build agent-policy mapping once for the run
        agent_map: Dict[int, Tuple[str, int]] = {
            idx.item(): (self._policy_pr.uri, self._policy_pr.version) for idx in self._policy_idxs
        }
        if self._npc_pr is not None:
            agent_map.update({idx.item(): (self._npc_pr.uri, self._npc_pr.version) for idx in self._npc_idxs})

        merged_db = StatsDB.merge_worker_dbs(self._stats_dir, agent_map)
        logger.info("Merged %s shards → %s", self._name, merged_db)

        if self._cfg.eval_stats_uri:
            StatsDB.export_db(merged_db, self._cfg.eval_stats_uri)
            logger.info("Exported stats DB → %s", self._cfg.eval_stats_uri)

    # ------------------------------------------------------------------ #
    #   main loop                                                         #
    # ------------------------------------------------------------------ #
    def simulate(self) -> None:
        obs, _ = self._vecenv.reset()
        pol_state = npc_state = None
        episodes, t0 = 0, time.time()

        while episodes < self._min_episodes and time.time() - t0 < self._max_time_s:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self._device)
                act_pol, *_, pol_state, _ = self._policy_pr.policy()(obs_t[self._policy_idxs], pol_state)
                if self._npc_pr and len(self._npc_idxs):
                    act_npc, *_, npc_state, _ = self._npc_pr.policy()(obs_t[self._npc_idxs], npc_state)

            # stitch candidate + npc actions
            actions = act_pol
            if len(self._npc_idxs):
                actions = torch.cat(
                    [
                        act_pol.view(self._vecenv.num_envs, -1, act_pol.shape[-1]),
                        act_npc.view(self._vecenv.num_envs, -1, act_pol.shape[-1]),
                    ],
                    dim=1,
                ).view(-1, act_pol.shape[-1])

            obs, _, dones, _, _ = self._vecenv.step(actions.cpu().numpy())
            episodes += sum(e.done for e in self._vecenv.envs)

        self._vecenv.close()
        self._finalise_stats()
        logger.info("Simulation %s finished after %d episodes.", self._name, episodes)


# --------------------------------------------------------------------------- #
#   Suite: dictionary of named simulations                                    #
# --------------------------------------------------------------------------- #
class SimulationSuite:
    def __init__(self, cfg: SimulationSuiteConfig, policy_pr: PolicyRecord, store: PolicyStore):
        self._sims = {
            name: Simulation(sub_cfg, policy_pr, store, name=name) for name, sub_cfg in cfg.simulations.items()
        }

    def simulate(self) -> Dict[str, None]:
        return {n: sim.simulate() for n, sim in self._sims.items()}
