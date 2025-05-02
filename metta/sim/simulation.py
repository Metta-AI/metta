"""
Runs a vectorised batch of MettaGrid environments, maps candidate & NPC
policies onto agents, optionally records replays, merges the per-env DuckDB
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
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from metta.agent.policy_state import PolicyState
from metta.agent.policy_store import PolicyRecord, PolicyStore
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
    One vectorised batch of MettaGrid envs.

    Responsibilities
    ----------------
    1. Map candidate & NPC policies to agents.
    2. Run until an episode or time budget is met.
    3. Optionally emit replays (`ReplayHelper`).
    4. Merge worker DuckDB shards → one DB and export it.
    """

    # ------------------------------------------------------------------ #
    #   construction                                                     #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        config: SimulationConfig,
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
        replay_path: str | None = None,
        name: str = "",
        wandb_run=None,
    ):
        self._config = config
        self._wandb_run = wandb_run
        self._replay_path = replay_path
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
<<<<<<< HEAD
        self._total_agents = self._num_envs * self._agents_per_env

        self._vecenv = make_vecenv(self._env_cfg, config.vectorization, num_envs=self._num_envs)

        # tell the policy which actions are available for this environment
        actions_names = self._vecenv.driver_env.action_names()
        actions_max_params = self._vecenv.driver_env._c_env.max_action_args()
        self._policy_pr.policy().activate_actions(actions_names, actions_max_params, self._device)
        if self._npc_pr is not None:
            # tell the npc policy which actions are available for this environment
            self._npc_pr.policy().activate_actions(actions_names, actions_max_params, self._device)

        # each index is an agent, and we reshape it into a matrix of num_envs x agents_per_env
        slice_idxs = (
            torch.arange(self._vecenv.num_agents).reshape(self._num_envs, self._agents_per_env).to(device=self._device)
=======
        self._policy_idxs = slice_mat[:, : self._policy_agents_per_env].reshape(-1)
        self._npc_idxs = (
            slice_mat[:, self._policy_agents_per_env :].reshape(-1)
            if self._npc_agents_per_env
            else torch.tensor([], device=self._device)
>>>>>>> 93e036b2 (cp)
        )

        # ---------------- replay helpers ------------------------------- #
        self._replay_helpers: List[ReplayHelper] | None = None
        self._episode_counters = np.zeros(self._num_envs, dtype=int)
<<<<<<< HEAD
        if self._replay_path is not None:
            self._replay_helpers = []
            for env_idx in range(self._num_envs):
                self._replay_helpers.append(self._create_replay_helper(env_idx))
=======
        if config.replay_path:
            self._replay_helpers = [self._create_replay_helper(e) for e in range(self._num_envs)]
>>>>>>> 93e036b2 (cp)

        # Map agent-idx → policy-name (used in per-agent stats)
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

<<<<<<< HEAD
    def _get_replay_path(self, env_idx: int, episode_count: int) -> str | None:
        """Generate a unique replay path for the given environment and episode."""
        base_path = self._replay_path

        if base_path is None:
            return None

        if env_idx == 0 and episode_count == 0:
            return base_path
=======
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
>>>>>>> 93e036b2 (cp)

        directory, fname = os.path.split(base)
        fname = fname or "replay.dat"
        return os.path.join(directory, f"ep{episode_count}_env{env_idx}_{fname}")

<<<<<<< HEAD
            assert len(s3_parts) > 1
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
=======
    # ------------------------------------------------------------------ #
    #   merge shards + export                                            #
    # ------------------------------------------------------------------ #
    def _finalise_stats(self) -> None:
        agent_map: Dict[int, Tuple[str, int]] = {
            idx.item(): (self._policy_pr.uri, self._policy_pr.version) for idx in self._policy_idxs
        }
>>>>>>> 93e036b2 (cp)
        if self._npc_pr is not None:
            agent_map.update({idx.item(): (self._npc_pr.uri, self._npc_pr.version) for idx in self._npc_idxs})

<<<<<<< HEAD
        logger.info(f"Simulation settings: {self._config}")
        logger.info(f"Replay path: {self._replay_path}")
=======
        merged_db = StatsDB.merge_worker_dbs(self._stats_dir, agent_map)
        logger.info(
            "Merged %d DuckDB shards for '%s' → %s",
            len(list(self._stats_dir.glob("*.duckdb"))),
            self._name,
            merged_db,
        )

        if self._config.eval_stats_uri:
            StatsDB.export_db(merged_db, self._config.eval_stats_uri)
            logger.info("Exported stats DB → %s", self._config.eval_stats_uri)

    # ------------------------------------------------------------------ #
    #   main loop                                                        #
    # ------------------------------------------------------------------ #
    def simulate(self, *, epoch: int = 0, dry_run: bool = False) -> List[dict]:
        logger.info(
            "Simulating '%s' – policy=%s  env=%s  policy_agents/env=%d",
            self._name,
            self._policy_pr.name,
            self._env_name,
            self._policy_agents_per_env,
        )
        if self._npc_pr:
            logger.debug(
                "           against NPC policy=%s (%d npc agents/env)",
                self._npc_pr.name,
                self._npc_agents_per_env,
            )
>>>>>>> 93e036b2 (cp)

        obs, _ = self._vecenv.reset()
        policy_state = PolicyState()
        npc_state = PolicyState()

        game_stats: List[dict] = []
        env_dones = [False] * self._num_envs
        t0 = time.time()

        while (self._episode_counters < self._min_episodes).any() and (time.time() - t0) < self._max_time_s:
            # ---------- forward pass / action sampling ---------------- #
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self._device)
                pol_logits, _ = self._policy_pr.policy()(obs_t[self._policy_idxs], policy_state)
                pol_actions, *_ = sample_logits(pol_logits)

<<<<<<< HEAD
                # Parallelize across opponents
                policy = self._policy_pr.policy()  # policy to evaluate
                policy_actions, _, _, _, _ = policy(my_obs, policy_state)

                # Iterate opponent policies
                if self._npc_pr is not None:
                    npc_obs = obs[self._npc_idxs]
                    npc_policy = self._npc_pr.policy()
                    npc_actions, _, _, _, _ = npc_policy(npc_obs, npc_state)

            actions = policy_actions
            if self._npc_agents_per_env > 0:
                actions = torch.cat(
=======
                if self._npc_pr and len(self._npc_idxs):
                    npc_logits, _ = self._npc_pr.policy()(obs_t[self._npc_idxs], npc_state)
                    npc_actions, *_ = sample_logits(npc_logits)

            # stitch [policy | npc] actions → vec-env layout
            actions_t = pol_actions
            if self._npc_agents_per_env:
                actions_t = torch.cat(
>>>>>>> 93e036b2 (cp)
                    [
                        pol_actions.view(self._num_envs, self._policy_agents_per_env, -1),
                        npc_actions.view(self._num_envs, self._npc_agents_per_env, -1),
                    ],
                    dim=1,
                ).view(-1, pol_actions.shape[-1])

            actions_np = actions_t.cpu().numpy()

            # ---------- pre-step replay logging ----------------------- #
            if self._replay_helpers:
                act_env = actions_np.reshape(self._num_envs, self._agents_per_env, -1)
                for e in range(self._num_envs):
                    if not env_dones[e]:
                        self._replay_helpers[e].log_pre_step(act_env[e].squeeze())

            # ---------- env.step -------------------------------------- #
            obs, rewards, dones, truncated, infos = self._vecenv.step(actions_np)

            # ---------- post-step replay logging ---------------------- #
            if self._replay_helpers:
                rew_env = rewards.reshape(self._num_envs, self._agents_per_env)
                for e in range(self._num_envs):
                    if not env_dones[e]:
                        self._replay_helpers[e].log_post_step(rew_env[e])

            # ---------- episode / env-reset FSM ----------------------- #
            done_now = np.logical_or(
                dones.reshape(self._num_envs, self._agents_per_env).all(axis=1),
                truncated.reshape(self._num_envs, self._agents_per_env).all(axis=1),
            )

            for e in range(self._num_envs):
                # episode finished
                if done_now[e] and not env_dones[e]:
                    env_dones[e] = True
                    if self._replay_helpers:
                        path = self._get_replay_path(e, self._episode_counters[e])
                        self._replay_helpers[e].write_replay(path, epoch=epoch, dry_run=dry_run)
                    self._episode_counters[e] += 1
                # env auto-reset → start new episode
                elif not done_now[e] and env_dones[e]:
                    env_dones[e] = False
                    if self._replay_helpers:
                        self._replay_helpers[e] = self._create_replay_helper(e)

<<<<<<< HEAD
                    if self._replay_helpers is not None:
                        path = self._get_replay_path(env_idx, self._episode_counters[env_idx])
                        self._replay_helpers[env_idx].write_replay(path, epoch=epoch)
                    self._episode_counters[env_idx] += 1

                # (2) environment has auto-reset → new episode has started -------------
                elif not done_now[env_idx] and env_dones[env_idx]:
                    env_dones[env_idx] = False  # <-- lets us log steps again
                    if self._replay_helpers is not None:
                        self._replay_helpers[env_idx] = self._create_replay_helper(env_idx)

            # Convert the environment configuration to a dictionary and flatten it.
=======
            # ---------- per-agent stats bundling ---------------------- #
>>>>>>> 93e036b2 (cp)
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

        # ---------- teardown ----------------------------------------- #
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
        replay_dir: str | None = None,
    ):
<<<<<<< HEAD
        logger.debug(f"Building Simulation suite from config:{config}")
        self._simulations = dict()
        self._wandb_run = wandb_run
        self._replay_dir = replay_dir

        for name, sim_config in config.simulations.items():
            replay_path = self._replay_path_for_sim(name)
            # Create a Simulation object for each config and pass wandb_run directly
            sim = Simulation(
                config=sim_config,
                policy_pr=policy_pr,
                policy_store=policy_store,
                name=name,
                wandb_run=wandb_run,
                replay_path=replay_path,
            )
            self._simulations[name] = sim

    def _replay_path_for_sim(self, name: str) -> str:
        if self._replay_dir is None:
            return None
        elif self._replay_dir.startswith("s3://"):
            return f"{self._replay_dir.rstrip('/')}/{name}/replay.json.z"
        else:
            return os.path.join(self._replay_dir, name, "replay.json.z")

    # TODO: epoch is a replay-specific parameter we could probably handle better
    def simulate(self, epoch: int = 0):
        # Run all simulations and gather results by name
        return {name: sim.simulate(epoch) for name, sim in self._simulations.items()}
=======
        self._sims: Dict[str, Simulation] = {
            n: Simulation(cfg, policy_pr, policy_store, name=n, wandb_run=wandb_run)
            for n, cfg in config.simulations.items()
        }

    def simulate(self, *, epoch: int = 0, dry_run: bool = False) -> Dict[str, List[dict]]:
        return {n: sim.simulate(epoch=epoch, dry_run=dry_run) for n, sim in self._sims.items()}
>>>>>>> 93e036b2 (cp)
