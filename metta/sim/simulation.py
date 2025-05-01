"""
Runs a vectorised batch of MettaGrid environments, maps candidate & NPC
policies onto agents, records optional replays, merges the per-env DuckDB
shards produced by `MettaGridStatsWriter`, and finally exports a single
canonical DB to `eval_stats_uri` (local file, S3, or WandB).
"""

from __future__ import annotations

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
from metta.sim.replay_helper import ReplayHelper
from metta.sim.simulation_config import SimulationConfig, SimulationSuiteConfig
from metta.sim.stats_db import StatsDB
from metta.sim.vecenv import make_vecenv
from metta.util.config import config_from_path
from metta.util.datastruct import flatten_config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#   Single simulation                                                         #
# --------------------------------------------------------------------------- #
class Simulation:
    """
    Wraps one vectorised batch of MettaGrid environments.

    Responsibilities
    ----------------
    1. Map candidate & NPC policies to agents.
    2. Step the vec-env until a target episode/time budget is met.
    3. Optionally emit per-step replays (`ReplayHelper`).
    4. Merge the worker DuckDB shards → one DB and export it.
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

        # ---------------- env config ----------------------------------- #
        self._env_cfg = config_from_path(config.env, config.env_overrides)
        self._env_name = config.env

        self._stats_dir = Path(config.run_dir) / "stats" / self._name
        self._stats_dir.mkdir(parents=True, exist_ok=True)
        self._env_cfg["stats_writer_path"] = str(self._stats_dir / f"stats_{uuid.uuid4().hex[:8]}.duckdb")

        # ---------------- policies ------------------------------------- #
        self._policy_pr = policy_pr
        self._policy_store = policy_store
        self._npc_pr = self._policy_store.policy(config.npc_policy_uri) if config.npc_policy_uri else None
        self._policy_agents_pct = config.policy_agents_pct if self._npc_pr else 1.0

        # ---------------- vec-env -------------------------------------- #
        self._device = config.device
        self._num_envs = config.num_envs
        self._min_episodes = config.num_episodes
        self._max_time_s = config.max_time_s

        self._vecenv = make_vecenv(self._env_cfg, config.vectorization, num_envs=self._num_envs)
        self._agents_per_env = self._env_cfg.game.num_agents

        # Agent-index slices
        slice_mat = torch.arange(self._vecenv.num_agents).reshape(self._num_envs, self._agents_per_env).to(self._device)
        self._policy_agents_per_env = max(1, int(self._agents_per_env * self._policy_agents_pct))
        self._npc_agents_per_env = self._agents_per_env - self._policy_agents_per_env
        self._policy_idxs = slice_mat[:, : self._policy_agents_per_env].reshape(-1)
        self._npc_idxs = (
            slice_mat[:, self._policy_agents_per_env :].reshape(-1)
            if self._npc_agents_per_env
            else torch.tensor([], device=self._device)
        )

        # ---------------- replay helpers ------------------------------- #
        self._replay_helpers: list[ReplayHelper] | None = None
        self._episode_counters = np.zeros(self._num_envs, dtype=int)
        if config.replay_path:
            self._replay_helpers = [self._create_replay_helper(e) for e in range(self._num_envs)]

        # Mapping agent-idx → policy-name
        self._agent_idx_to_policy_name = {
            **{i.item(): self._policy_pr.name for i in self._policy_idxs},
            **(
                {i.item(): self._npc_pr.name for i in self._npc_idxs}
                if self._npc_pr is not None and len(self._npc_idxs)
                else {}
            ),
        }

    # ------------------------------------------------------------------ #
    #   helpers                                                          #
    # ------------------------------------------------------------------ #
    def _create_replay_helper(self, env_idx: int) -> ReplayHelper:
        return ReplayHelper(
            config=self._config,
            env=self._vecenv.envs[env_idx],
            policy_record=self._policy_pr,
            wandb_run=self._wandb_run,
        )

    def _get_replay_path(self, env_idx: int, episode_count: int) -> str:
        base = self._config.replay_path
        if self._num_envs == 1 and self._min_episodes == 1:
            return base

        if base.startswith("s3://"):
            bucket, key = base[5:].split("/", 1)
            prefix, fname = os.path.split(key)
            fname = fname or "replay.dat"
            return (
                f"s3://{bucket}/{prefix}/ep{episode_count}_env{env_idx}_{fname}"
                if prefix
                else f"s3://{bucket}/ep{episode_count}_env{env_idx}_{fname}"
            )

        directory, fname = os.path.split(base)
        fname = fname or "replay.dat"
        return os.path.join(directory, f"ep{episode_count}_env{env_idx}_{fname}")

    def _finalise_stats(self) -> None:
        agent_map: Dict[int, Tuple[str, int]] = {
            idx.item(): (self._policy_pr.uri, self._policy_pr.version) for idx in self._policy_idxs
        }
        if self._npc_pr is not None:
            agent_map.update({idx.item(): (self._npc_pr.uri, self._npc_pr.version) for idx in self._npc_idxs})

        merged_db = StatsDB.merge_worker_dbs(self._stats_dir, agent_map)
        logger.info(
            "Merged %d DuckDB shards for '%s' → %s", len(list(self._stats_dir.glob("*.duckdb"))), self._name, merged_db
        )
        if self._config.eval_stats_uri:
            StatsDB.export_db(merged_db, self._config.eval_stats_uri)
            logger.info("Exported stats DB → %s", self._config.eval_stats_uri)

    # ------------------------------------------------------------------ #
    #   main loop                                                        #
    # ------------------------------------------------------------------ #
    def simulate(self, *, epoch: int = 0, dry_run: bool = False) -> list[dict]:
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

        obs, _ = self._vecenv.reset()
        policy_state = PolicyState()
        npc_state = PolicyState()

        game_stats: list[dict] = []
        env_dones = [False] * self._num_envs
        t0 = time.time()

        while (self._episode_counters < self._min_episodes).any() and (time.time() - t0) < self._max_time_s:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self._device)
                pol_logits, _ = self._policy_pr.policy()(obs_t[self._policy_idxs], policy_state)
                pol_actions, *_ = sample_logits(pol_logits)

                if self._npc_pr and len(self._npc_idxs):
                    npc_logits, _ = self._npc_pr.policy()(obs_t[self._npc_idxs], npc_state)
                    npc_actions, *_ = sample_logits(npc_logits)

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

            if self._replay_helpers:
                act_env = actions_np.reshape(self._num_envs, self._agents_per_env, -1)
                for e in range(self._num_envs):
                    if not env_dones[e]:
                        self._replay_helpers[e].log_pre_step(act_env[e].squeeze())

            obs, rewards, dones, truncated, infos = self._vecenv.step(actions_np)

            if self._replay_helpers:
                rew_env = rewards.reshape(self._num_envs, self._agents_per_env)
                for e in range(self._num_envs):
                    if not env_dones[e]:
                        self._replay_helpers[e].log_post_step(rew_env[e])

            done_now = np.logical_or(
                dones.reshape(self._num_envs, self._agents_per_env).all(axis=1),
                truncated.reshape(self._num_envs, self._agents_per_env).all(axis=1),
            )

            for e in range(self._num_envs):
                if done_now[e] and not env_dones[e]:
                    env_dones[e] = True
                    if self._replay_helpers:
                        path = self._get_replay_path(e, self._episode_counters[e])
                        self._replay_helpers[e].write_replay(path, epoch=epoch, dry_run=dry_run)
                    self._episode_counters[e] += 1
                elif not done_now[e] and env_dones[e]:
                    env_dones[e] = False
                    if self._replay_helpers:
                        self._replay_helpers[e] = self._create_replay_helper(e)

            game_cfg = OmegaConf.to_container(self._env_cfg.game, resolve=False)
            flat_env = flatten_config(game_cfg, parent_key="game")
            flat_env.update(
                {
                    "eval_name": self._name,
                    "timestamp": datetime.now().isoformat(),
                    "npc": self._config.npc_policy_uri,
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
    """Thin wrapper around a dict {name → Simulation}."""

    def __init__(
        self,
        config: SimulationSuiteConfig,
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
        *,
        wandb_run=None,
    ):
        self._sims: Dict[str, Simulation] = {
            n: Simulation(cfg, policy_pr, policy_store, name=n, wandb_run=wandb_run)
            for n, cfg in config.simulations.items()
        }

    def simulate(self, *, epoch: int = 0, dry_run: bool = False) -> Dict[str, list]:
        return {n: sim.simulate(epoch=epoch, dry_run=dry_run) for n, sim in self._sims.items()}
