# metta/sim/simulation.py
from __future__ import annotations

"""
Vectorised simulation runner.

• Launches a MettaGrid vec-env batch
• Every worker writes its own *.duckdb* shard
• On shutdown the shards are merged into **one** StatsDB that the caller
  can further merge / export.
"""

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List

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
# Single simulation                                                           #
# --------------------------------------------------------------------------- #
class Simulation:
    def __init__(
        self,
        name: str,
        config: SimulationConfig,
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
<<<<<<< HEAD
        replay_path: str | None = None,
        name: str = "",
=======
>>>>>>> 137a628d (cp)
        wandb_run=None,
        stats_dir: str | None = None,
    ):
<<<<<<< HEAD
        self._config = config
        self._wandb_run = wandb_run
        self._replay_path = replay_path
=======
        self._cfg = config
        self._wandb = wandb_run
>>>>>>> b7f94640 (cp)
        self._name = name or "default"

        # ------------------------------------------------------------------ #
        # Env configuration                                                  #
        # ------------------------------------------------------------------ #
        self._env_cfg = config_from_path(config.env, config.env_overrides)
        self._env_name = config.env

        stats_root: Path = Path(stats_dir).expanduser() if stats_dir is not None else Path("tmp/stats") / self._name
        stats_root.mkdir(parents=True, exist_ok=True)

        # unique suffix so concurrent workers never step on each other
        unique = f"{os.getpid():05d}_{uuid.uuid4().hex[:8]}"
        self._env_cfg["stats_writer_path"] = str(stats_root / f"stats_{unique}.duckdb")
        self._stats_dir = stats_root

        # ------------------------------------------------------------------ #
        # Policies                                                           #
        # ------------------------------------------------------------------ #
        self._policy_pr = policy_pr
        self._policy_store = policy_store
        self._npc_pr = self._policy_store.policy(config.npc_policy_uri) if config.npc_policy_uri else None
        self._policy_agents_pct = config.policy_agents_pct if self._npc_pr else 1.0

        # ------------------------------------------------------------------ #
        # Vectorised env                                                     #
        # ------------------------------------------------------------------ #
        self._device = config.device
        self._num_envs = config.num_envs
        self._min_episodes = config.num_episodes
        self._max_time_s = config.max_time_s

        self._vecenv = make_vecenv(self._env_cfg, config.vectorization, num_envs=self._num_envs)
        self._agents_per_env = self._env_cfg.game.num_agents

        # agent-index slices (CPU tensors are fine)
        slice_mat = torch.arange(self._vecenv.num_agents).reshape(self._num_envs, self._agents_per_env)
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
<<<<<<< HEAD
            slice_mat[:, self._policy_agents_per_env :].reshape(-1)
            if self._npc_agents_per_env
            else torch.tensor([], device=self._device)
>>>>>>> 93e036b2 (cp)
=======
            slice_mat[:, self._policy_agents_per_env :].reshape(-1) if self._npc_agents_per_env else torch.tensor([])
>>>>>>> b7f94640 (cp)
        )

        # Replay helpers (optional)
        self._replay_helpers: List[ReplayHelper] | None = None
        self._episode_counters = np.zeros(self._num_envs, dtype=int)
<<<<<<< HEAD
        if self._replay_path is not None:
            self._replay_helpers = []
            for env_idx in range(self._num_envs):
                self._replay_helpers.append(self._create_replay_helper(env_idx))
=======
        if config.replay_path:
<<<<<<< HEAD
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
=======
            self._replay_helpers = [self._make_replay_helper(i) for i in range(self._num_envs)]
>>>>>>> b7f94640 (cp)

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _make_replay_helper(self, env_idx: int) -> ReplayHelper:
        return ReplayHelper(
            config=self._cfg,
            env=self._vecenv.envs[env_idx],
            policy_record=self._policy_pr,
            wandb_run=self._wandb,
        )

<<<<<<< HEAD
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

=======
    @staticmethod
    def _replay_path(base: str, env_idx: int, ep: int) -> str:
>>>>>>> b7f94640 (cp)
        if base.startswith("s3://"):
            bucket, key = base[5:].split("/", 1)
            prefix, fname = os.path.split(key)
            fname = fname or "replay.dat"
<<<<<<< HEAD
            return (
                f"s3://{bucket}/{prefix}/ep{episode_count}_env{env_idx}_{fname}"
                if prefix
                else f"s3://{bucket}/ep{episode_count}_env{env_idx}_{fname}"
            )
>>>>>>> 93e036b2 (cp)

=======
            dst = f"ep{ep}_env{env_idx}_{fname}"
            return f"s3://{bucket}/{prefix}/{dst}" if prefix else f"s3://{bucket}/{dst}"
>>>>>>> b7f94640 (cp)
        directory, fname = os.path.split(base)
        fname = fname or "replay.dat"
        return os.path.join(directory, f"ep{ep}_env{env_idx}_{fname}")

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
    # Stats helpers                                                      #
    # ------------------------------------------------------------------ #

    def _merge_worker_dbs(self) -> StatsDB:
        """Merge all *.duckdb* shards for this simulation → one DB object."""
        # Map {agent-index → (policy_key, policy_version)}
        agent_map: Dict[int, tuple[str, str | None]] = {
            idx.item(): self._policy_pr.key_and_version() for idx in self._policy_idxs
        }
>>>>>>> 93e036b2 (cp)
        if self._npc_pr is not None:
            agent_map.update({idx.item(): self._npc_pr.key_and_version() for idx in self._npc_idxs})

<<<<<<< HEAD
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
=======
        db = StatsDB.merge_worker_dbs(self._stats_dir, agent_map)
        logger.info("Merged %s → %s", self._stats_dir, db.path.name)
        return db
>>>>>>> b7f94640 (cp)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def simulate(self, *, epoch: int = 0, dry_run: bool = False) -> StatsDB:
        """
        Run the simulation.  Returns the (already-merged) *StatsDB* which the
        caller may merge / export further.
        """
        logger.info(
            "Sim '%s': %d env × %d agents (%.0f%% candidate)",
            self._name,
            self._num_envs,
            self._agents_per_env,
            100 * self._policy_agents_per_env / self._agents_per_env,
        )
<<<<<<< HEAD
        if self._npc_pr:
            logger.debug(
                "           against NPC policy=%s (%d npc agents/env)",
                self._npc_pr.name,
                self._npc_agents_per_env,
            )
>>>>>>> 93e036b2 (cp)
=======
>>>>>>> b7f94640 (cp)

        obs, _ = self._vecenv.reset()
        p_state = PolicyState()
        npc_state = PolicyState()
        env_done_flags = [False] * self._num_envs

        t0 = time.time()
        while (self._episode_counters < self._min_episodes).any() and time.time() - t0 < self._max_time_s:
            # ------------------------------------------------------------------ #
            # policy forward passes                                              #
            # ------------------------------------------------------------------ #
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=self._device)
                p_logits, _ = self._policy_pr.policy()(obs_t[self._policy_idxs], p_state)
                p_actions, *_ = sample_logits(p_logits)

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

            # ------------------------------------------------------------------ #
            # stitch actions                                                     #
            # ------------------------------------------------------------------ #
            acts_t = p_actions
            if self._npc_agents_per_env:
<<<<<<< HEAD
                actions_t = torch.cat(
>>>>>>> 93e036b2 (cp)
=======
                acts_t = torch.cat(
>>>>>>> b7f94640 (cp)
                    [
                        p_actions.view(self._num_envs, self._policy_agents_per_env, -1),
                        npc_actions.view(self._num_envs, self._npc_agents_per_env, -1),
                    ],
                    dim=1,
                ).view(-1, p_actions.shape[-1])
            acts_np = acts_t.cpu().numpy()

            # Pre-step replay
            if self._replay_helpers:
                per_env = acts_np.reshape(self._num_envs, self._agents_per_env, -1)
                for e in range(self._num_envs):
                    if not env_done_flags[e]:
                        self._replay_helpers[e].log_pre_step(per_env[e].squeeze())

            # ------------------------------------------------------------------ #
            # env step                                                           #
            # ------------------------------------------------------------------ #
            obs, rewards, dones, trunc, infos = self._vecenv.step(acts_np)

            # Post-step replay
            if self._replay_helpers:
                per_env_r = rewards.reshape(self._num_envs, self._agents_per_env)
                for e in range(self._num_envs):
                    if not env_done_flags[e]:
                        self._replay_helpers[e].log_post_step(per_env_r[e])

            # ------------------------------------------------------------------ #
            # episode FSM                                                        #
            # ------------------------------------------------------------------ #
            done_now = np.logical_or(
                dones.reshape(self._num_envs, self._agents_per_env).all(1),
                trunc.reshape(self._num_envs, self._agents_per_env).all(1),
            )

            for e in range(self._num_envs):
                if done_now[e] and not env_done_flags[e]:
                    env_done_flags[e] = True
                    if self._replay_helpers:
                        path = self._replay_path(self._cfg.replay_path, e, self._episode_counters[e])
                        self._replay_helpers[e].write_replay(path, epoch=epoch, dry_run=dry_run)
                    self._episode_counters[e] += 1
                elif not done_now[e] and env_done_flags[e]:
                    env_done_flags[e] = False
                    if self._replay_helpers:
                        self._replay_helpers[e] = self._make_replay_helper(e)

<<<<<<< HEAD
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
=======
        # ------------------------------------------------------------------ #
>>>>>>> b7f94640 (cp)
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
# Simulation suite                                                            #
# --------------------------------------------------------------------------- #
class SimulationSuite:
    """
    Run a set of named simulations and *return one merged StatsDB* containing
    the union of all their statistics.
    """

    def __init__(
        self,
        config: SimulationSuiteConfig,
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
        *,
        wandb_run=None,
<<<<<<< HEAD
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
=======
        stats_dir: str | None = None,
    ):
        def sim_stats_dir(name: str) -> str | None:
            return f"{stats_dir}/{name}" if stats_dir and name else None

>>>>>>> 137a628d (cp)
        self._sims: Dict[str, Simulation] = {
            n: Simulation(n, cfg, policy_pr, policy_store, wandb_run=wandb_run, stats_dir=sim_stats_dir(n))
            for n, cfg in config.simulations.items()
        }

<<<<<<< HEAD
    def simulate(self, *, epoch: int = 0, dry_run: bool = False) -> Dict[str, List[dict]]:
        return {n: sim.simulate(epoch=epoch, dry_run=dry_run) for n, sim in self._sims.items()}
>>>>>>> 93e036b2 (cp)
=======
    #

    def simulate(self, *, epoch: int = 0, dry_run: bool = False) -> StatsDB:
        """
        Run every constituent Simulation, merge their DBs and return the final
        *StatsDB*.  Caller is free to `StatsDB.export_db(...)` afterwards.
        """
        merged_db: StatsDB | None = None

        for name, sim in self._sims.items():
            logger.info("=== Simulation '%s' ===", name)
            db = sim.simulate(epoch=epoch, dry_run=dry_run)

            if merged_db is None:
                merged_db = db  # first db becomes the accumulator
            else:
                merged_db.merge_in(db)
                db.close()  # close shard once merged to release file handle

        assert merged_db is not None, "SimulationSuite contained no simulations"
        return merged_db
>>>>>>> b7f94640 (cp)
