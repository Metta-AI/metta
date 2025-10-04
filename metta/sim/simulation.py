from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from einops import rearrange

from metta.agent.mocks import MockAgent
from metta.agent.policy import Policy
from metta.agent.utils import obs_to_td
from metta.app_backend.clients.stats_client import HttpStatsClient, StatsClient
from metta.cogworks.curriculum.curriculum import Curriculum, CurriculumConfig
from metta.common.util.heartbeat import record_heartbeat
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.policy_artifact import PolicyArtifact
from metta.rl.training.training_environment import EnvironmentMetaData
from metta.rl.vecenv import make_vecenv
from metta.sim.replay_writer import S3ReplayWriter
from metta.sim.simulation_config import SimulationConfig
from metta.sim.simulation_stats_db import SimulationStatsDB
from metta.sim.stats import DuckDBStatsWriter
from metta.sim.thumbnail_automation import maybe_generate_and_upload_thumbnail
from metta.sim.utils import get_or_create_policy_ids
from mettagrid import MettaGridEnv, dtype_actions

SYNTHETIC_EVAL_SUITE = "training"

logger = logging.getLogger(__name__)


class SimulationCompatibilityError(Exception):
    """Raised when there's a compatibility issue that prevents simulation from running."""

    pass


class Simulation:
    """A vectorized batch of MettaGrid environments sharing the same parameters."""

    def __init__(
        self,
        cfg: SimulationConfig,
        policy_uri: str | None,
        device: torch.device,
        vectorization: str,
        stats_dir: str = "/tmp/stats",
        replay_dir: str | None = None,
        stats_client: StatsClient | None = None,
        stats_epoch_id: uuid.UUID | None = None,
        eval_task_id: uuid.UUID | None = None,
    ):
        self._config = cfg
        self._id = uuid.uuid4().hex[:12]
        self._eval_task_id = eval_task_id

        replay_dir = f"{replay_dir}/{self._id}" if replay_dir else None

        sim_stats_dir = (Path(stats_dir) / self._id).resolve()
        sim_stats_dir.mkdir(parents=True, exist_ok=True)
        self._stats_dir = sim_stats_dir
        self._stats_writer = DuckDBStatsWriter(sim_stats_dir)
        self._replay_writer = S3ReplayWriter(replay_dir)
        self._device = device

        self._full_name = f"{cfg.suite}/{cfg.name}"

        if policy_uri:
            policy_artifact = CheckpointManager.load_artifact_from_uri(policy_uri)
            resolved_policy_uri = CheckpointManager.normalize_uri(policy_uri)
        else:
            policy_artifact = PolicyArtifact(policy=MockAgent())
            resolved_policy_uri = "mock://"

        # Calculate number of parallel environments and episodes per environment
        # to achieve the target total number of episodes
        max_envs = os.cpu_count() or 1
        if cfg.num_episodes <= max_envs:
            # If we want fewer episodes than CPUs, create one env per episode
            num_envs = cfg.num_episodes
            episodes_per_env = 1
        else:
            # Otherwise, use all CPUs and distribute episodes
            num_envs = max_envs
            episodes_per_env = (cfg.num_episodes + num_envs - 1) // num_envs  # Ceiling division

        logger.info(
            f"Creating vecenv with {num_envs} environments, {episodes_per_env} "
            f"episodes per env (total target: {cfg.num_episodes})"
        )

        self._vecenv = make_vecenv(
            Curriculum(CurriculumConfig.from_mg(cfg.env)),
            vectorization,
            num_envs=num_envs,
            stats_writer=self._stats_writer,
            replay_writer=self._replay_writer,
        )

        self._num_envs = num_envs
        self._min_episodes = episodes_per_env
        self._max_time_s = cfg.max_time_s
        self._agents_per_env = cfg.env.game.num_agents

        self._policy_artifact = policy_artifact
        self._policy: Policy | None = None
        self._policy_uri = resolved_policy_uri
        # Load NPC policy if specified
        if cfg.npc_policy_uri:
            self._npc_artifact = CheckpointManager.load_artifact_from_uri(cfg.npc_policy_uri)
            self._npc_policy: Policy | None = None
        else:
            self._npc_artifact = None
            self._npc_policy = None
        self._npc_policy_uri = cfg.npc_policy_uri
        self._policy_agents_pct = cfg.policy_agents_pct if cfg.npc_policy_uri else 1.0

        self._stats_client: StatsClient | None = stats_client
        self._stats_epoch_id: uuid.UUID | None = stats_epoch_id

        driver_env = self._vecenv.driver_env  # type: ignore
        metta_grid_env: MettaGridEnv = getattr(driver_env, "_env", driver_env)
        assert isinstance(metta_grid_env, MettaGridEnv), f"Expected MettaGridEnv, got {type(metta_grid_env)}"

        env_metadata = EnvironmentMetaData(
            obs_width=metta_grid_env.obs_width,
            obs_height=metta_grid_env.obs_height,
            obs_features=metta_grid_env.observation_features,
            action_names=metta_grid_env.action_names,
            max_action_args=metta_grid_env.max_action_args,
            num_agents=metta_grid_env.num_agents,
            observation_space=metta_grid_env.observation_space,
            action_space=metta_grid_env.action_space,
            feature_normalizations=metta_grid_env.feature_normalizations,
        )

        self._policy = self._materialize_policy(self._policy_artifact, self._policy, env_metadata)

        if self._npc_artifact is not None:
            self._npc_policy = self._materialize_policy(self._npc_artifact, self._npc_policy, env_metadata)

        # agent-index bookkeeping
        idx_matrix = torch.arange(metta_grid_env.num_agents * self._num_envs, device=self._device).reshape(
            self._num_envs, self._agents_per_env
        )
        self._policy_agents_per_env = max(1, int(self._agents_per_env * self._policy_agents_pct))
        self._npc_agents_per_env = self._agents_per_env - self._policy_agents_per_env

        self._policy_idxs = idx_matrix[:, : self._policy_agents_per_env].reshape(-1)
        self._npc_idxs = (
            idx_matrix[:, self._policy_agents_per_env :].reshape(-1)
            if self._npc_agents_per_env
            else torch.tensor([], device=self._device, dtype=torch.long)
        )
        self._episode_counters = np.zeros(self._num_envs, dtype=int)

    def _materialize_policy(
        self,
        artifact: PolicyArtifact,
        existing_policy: Policy | None,
        env_metadata: EnvironmentMetaData,
    ) -> Policy:
        using_existing = existing_policy is not None
        if using_existing:
            policy = existing_policy
        else:
            policy = artifact.instantiate(env_metadata, device=self._device)

        policy = policy.to(self._device)
        policy.eval()

        if using_existing and hasattr(policy, "initialize_to_environment"):
            policy.initialize_to_environment(env_metadata, self._device)
        return policy

    @classmethod
    def create(
        cls,
        sim_config: SimulationConfig,
        device: str,
        vectorization: str,
        stats_dir: str = "./train_dir/stats",
        replay_dir: str = "./train_dir/replays",
        policy_uri: str | None = None,
    ) -> "Simulation":
        """Create a Simulation with sensible defaults."""
        # Create replay directory path with simulation name
        full_replay_dir = f"{replay_dir}/{sim_config.name}"

        # Create and return simulation
        return cls(
            sim_config,
            policy_uri=policy_uri,
            device=torch.device(device),
            vectorization=vectorization,
            stats_dir=stats_dir,
            replay_dir=full_replay_dir,
        )

    def start_simulation(self) -> None:
        """Start the simulation."""
        logger.info(
            "Sim '%s': %d env × %d agents (%.0f%% candidate)",
            self._full_name,
            self._num_envs,
            self._agents_per_env,
            100 * self._policy_agents_per_env / self._agents_per_env,
        )
        logger.info("Stats dir: %s", self._stats_dir)
        self._obs, _ = self._vecenv.reset()
        self._env_done_flags = [False] * self._num_envs

        self._t0 = time.time()

    def _get_actions_for_agents(self, agent_indices: torch.Tensor, policy) -> torch.Tensor:
        """Get actions for a group of agents, preserving agent dimension for single-agent cases."""
        agent_obs = self._obs[agent_indices]
        # Ensure agent dimension is preserved for single-agent environments
        if agent_obs.ndim == 2 and len(agent_indices) == 1:
            agent_obs = agent_obs[None, ...]  # Add back the agent dimension
        td = obs_to_td(agent_obs, self._device)
        policy(td)
        return td["actions"]

    def generate_actions(self) -> np.ndarray:
        """Generate actions for the simulation."""
        if __debug__:
            num_policy = len(self._policy_idxs)
            num_npc = len(self._npc_idxs)

            if num_policy > 0:
                assert self._policy_idxs[0] == 0, f"Policy indices should start at 0, got {self._policy_idxs[0]}"
                assert self._policy_idxs[-1] == num_policy - 1, (
                    f"Policy indices should be continuous 0 to {num_policy - 1}, last index is {self._policy_idxs[-1]}"
                )
                assert list(self._policy_idxs) == list(range(num_policy)), (
                    "Policy indices should be continuous sequence starting from 0"
                )

            if self._npc_policy is not None and num_npc > 0:
                expected_npc_start = num_policy
                assert self._npc_idxs[0] == expected_npc_start, (
                    f"NPC indices should start at {expected_npc_start}, got {self._npc_idxs[0]}"
                )
                assert self._npc_idxs[-1] == expected_npc_start + num_npc - 1, (
                    f"NPC indices should end at {expected_npc_start + num_npc - 1}, got {self._npc_idxs[-1]}"
                )
                assert list(self._npc_idxs) == list(range(expected_npc_start, expected_npc_start + num_npc)), (
                    f"NPC indices should be continuous sequence from {expected_npc_start}"
                )

            if num_policy > 0 and num_npc > 0:
                policy_set = set(self._policy_idxs)
                npc_set = set(self._npc_idxs)
                assert policy_set.isdisjoint(npc_set), (
                    f"Policy and NPC indices should not overlap. Overlap: {policy_set.intersection(npc_set)}"
                )

        with torch.no_grad():
            policy_actions = self._get_actions_for_agents(self._policy_idxs.cpu(), self._policy)

            npc_actions = None
            if self._npc_policy is not None and len(self._npc_idxs):
                npc_actions = self._get_actions_for_agents(self._npc_idxs, self._npc_policy)

        actions = policy_actions
        if self._npc_agents_per_env:
            policy_actions = rearrange(
                policy_actions,
                "(envs policy_agents) act -> envs policy_agents act",
                envs=self._num_envs,
                policy_agents=self._policy_agents_per_env,
            )
            npc_actions = rearrange(
                npc_actions,
                "(envs npc_agents) act -> envs npc_agents act",
                envs=self._num_envs,
                npc_agents=self._npc_agents_per_env,
            )
            # Concatenate along agents dimension
            actions = torch.cat([policy_actions, npc_actions], dim=1)
            # Flatten back to (total_agents, action_dim)
            actions = rearrange(actions, "envs agents act -> (envs agents) act")

        actions_np = actions.cpu().numpy().astype(dtype_actions)
        return actions_np

    def step_simulation(self, actions_np: np.ndarray) -> None:
        obs, rewards, dones, trunc, infos = self._vecenv.step(actions_np)

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

    def _maybe_generate_thumbnail(self) -> str | None:
        """Generate thumbnail if this is the first run for this eval_name."""
        try:
            # Skip synthetic evaluation framework simulations
            if self._config.suite == SYNTHETIC_EVAL_SUITE:
                logger.debug(f"Skipping thumbnail generation for synthetic simulation: {self._full_name}")
                return None

            # Get any replay data from this simulation
            if not self._replay_writer.episodes:
                logger.warning(f"No replay data available for thumbnail generation: {self._full_name}")
                return None

            # Use first available episode replay and get its ID
            episode_id = next(iter(self._replay_writer.episodes.keys()))
            episode_replay = self._replay_writer.episodes[episode_id]
            replay_data = episode_replay.get_replay_data()

            # Attempt to generate and upload thumbnail using episode ID (like replay files)
            success, thumbnail_url = maybe_generate_and_upload_thumbnail(replay_data, episode_id)
            if success:
                logger.info(f"Generated thumbnail for episode_id: {episode_id}")
                return thumbnail_url
            else:
                logger.debug(f"Thumbnail generation failed for episode_id: {episode_id}")
                return None

        except Exception as e:
            logger.error(f"Thumbnail generation failed for {self._full_name}: {e}")
            return None

    def end_simulation(self) -> SimulationResults:
        self._vecenv.close()
        db = self._from_shards_and_context()

        # Generate thumbnail before writing to database so we can include the URL
        thumbnail_url = self._maybe_generate_thumbnail()
        self._write_remote_stats(db, thumbnail_url=thumbnail_url)

        logger.info(
            "Sim '%s' finished: %d episodes in %.1fs",
            self._full_name,
            int(self._episode_counters.sum()),
            time.time() - self._t0,
        )
        return SimulationResults(db)

    def simulate(self) -> SimulationResults:
        """Run the simulation; returns the merged `StatsDB`."""
        self.start_simulation()

        self._policy.reset_memory()
        if self._npc_policy is not None:
            self._npc_policy.reset_memory()

        # Track iterations for heartbeat
        iteration_count = 0
        heartbeat_interval = 100  # Record heartbeat every 100 iterations

        while (self._episode_counters < self._min_episodes).any() and (time.time() - self._t0) < self._max_time_s:
            actions_np = self.generate_actions()
            self.step_simulation(actions_np)

            # Record heartbeat periodically
            iteration_count += 1
            if iteration_count % heartbeat_interval == 0:
                record_heartbeat()

        return self.end_simulation()

    def _from_shards_and_context(self) -> SimulationStatsDB:
        """Merge all *.duckdb* shards for this simulation → one `StatsDB`."""
        # Create agent map using URIs for database integration
        agent_map: Dict[int, str] = {}

        # Add policy agents to the map if they have a URI
        if self._policy_uri:
            for idx in self._policy_idxs:
                agent_map[int(idx.item())] = self._policy_uri

        # Add NPC agents to the map if they exist
        if self._npc_policy is not None and self._npc_policy_uri:
            for idx in self._npc_idxs:
                agent_map[int(idx.item())] = self._npc_policy_uri

        # Pass the policy URI directly
        db = SimulationStatsDB.from_shards_and_context(
            sim_id=self._id,
            dir_with_shards=self._stats_dir,
            agent_map=agent_map,
            sim_name=self._config.suite,
            sim_env=self._config.name,
            policy_uri=self._policy_uri or "",
        )
        return db

    def _write_remote_stats(self, stats_db: SimulationStatsDB, thumbnail_url: str | None = None) -> None:
        """Write stats to the remote stats database."""
        if self._stats_client is not None and isinstance(self._stats_client, HttpStatsClient):
            # Use policy_uri directly
            policy_details: list[tuple[str, str | None]] = []

            if self._policy_uri:  # Only add if we have a URI
                policy_details.append((self._policy_uri, None))

            # Add NPC policy if it exists
            if self._npc_policy_uri:
                policy_details.append((self._npc_policy_uri, "NPC policy"))

            policy_ids = get_or_create_policy_ids(self._stats_client, policy_details, self._stats_epoch_id)

            agent_map: Dict[int, uuid.UUID] = {}

            if self._policy_uri:
                for idx in self._policy_idxs:
                    agent_map[int(idx.item())] = policy_ids[self._policy_uri]

            if self._npc_policy_uri:
                for idx in self._npc_idxs:
                    agent_map[int(idx.item())] = policy_ids[self._npc_policy_uri]

            # Get all episodes from the database
            episodes_df = stats_db.query("SELECT * FROM episodes")

            for _, episode_row in episodes_df.iterrows():
                episode_id = episode_row["id"]

                # Get agent metrics for this episode
                agent_metrics_df = stats_db.query(f"SELECT * FROM agent_metrics WHERE episode_id = '{episode_id}'")
                # agent_id -> metric_name -> metric_value
                agent_metrics: Dict[int, Dict[str, float]] = {}

                for _, metric_row in agent_metrics_df.iterrows():
                    agent_id = int(metric_row["agent_id"])
                    metric_name = metric_row["metric"]
                    metric_value = float(metric_row["value"])

                    if agent_id not in agent_metrics:
                        agent_metrics[agent_id] = {}
                    agent_metrics[agent_id][metric_name] = metric_value

                # Get episode attributes
                attributes_df = stats_db.query(f"SELECT * FROM episode_attributes WHERE episode_id = '{episode_id}'")
                attributes: Dict[str, Any] = {}

                for _, attr_row in attributes_df.iterrows():
                    attr_name = attr_row["attribute"]
                    attr_value = attr_row["value"]
                    attributes[attr_name] = attr_value

                # Record the episode remotely
                try:
                    self._stats_client.record_episode(
                        agent_policies=agent_map,
                        agent_metrics=agent_metrics,
                        primary_policy_id=policy_ids[self._policy_uri],
                        stats_epoch=self._stats_epoch_id,
                        sim_suite=self._config.suite,
                        env_name=self._config.name,
                        replay_url=episode_row.get("replay_url"),
                        attributes=attributes,
                        eval_task_id=self._eval_task_id,
                        thumbnail_url=thumbnail_url,
                    )
                except Exception as e:
                    logger.error(f"Failed to record episode {episode_id} remotely: {e}")
                    # Continue with other episodes even if one fails

    def get_policy_state(self):
        """Get the policy state for memory manipulation."""
        # Return the policy state if it has one
        if hasattr(self._policy, "state"):
            return self._policy.state
        return None

    @property
    def full_name(self) -> str:
        return self._full_name

    def get_envs(self):
        """Returns a list of all envs in the simulation."""
        return self._vecenv.envs

    def get_env(self):
        """Make sure this sim has a single env, and return it."""
        if len(self._vecenv.envs) != 1:
            raise ValueError("Attempting to get single env, but simulation has multiple envs")
        return self._vecenv.envs[0]

    def get_replays(self) -> dict:
        """Get all replays for this simulation."""
        return self._replay_writer.episodes.values()

    def get_replay(self) -> dict:
        """Makes sure this sim has a single replay, and return it."""
        # If no episodes yet, create initial replay data from the environment
        if len(self._replay_writer.episodes) == 0:
            env = self.get_env()
            # Return initial replay structure with action names
            return {
                "version": 2,
                "action_names": env.action_names,
                "item_names": env.resource_names if hasattr(env, "resource_names") else [],
                "type_names": env.object_type_names if hasattr(env, "object_type_names") else [],
                "num_agents": env.num_agents,
                "max_steps": env.max_steps,
                "map_size": [env.height, env.width],
                "file_name": "live_play",
                "steps": [],
            }
        if len(self._replay_writer.episodes) != 1:
            raise ValueError("Attempting to get single replay, but simulation has multiple episodes")
        # Get the single episode directly
        episode_id = next(iter(self._replay_writer.episodes))
        return self._replay_writer.episodes[episode_id].get_replay_data()


@dataclass
class SimulationResults:
    """Results of a simulation.
    For now just a stats db. Replay plays can be retrieved from the stats db."""

    stats_db: SimulationStatsDB
    replay_urls: dict[str, list[str]] | None = None  # Maps simulation names to lists of replay URLs
