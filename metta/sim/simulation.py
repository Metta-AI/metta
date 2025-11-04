from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from metta.agent.mocks import MockAgent
from metta.agent.policy import Policy
from metta.app_backend.clients.stats_client import HttpStatsClient, StatsClient
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.policy_artifact import PolicyArtifact
from metta.sim.replay_log_writer import ReplayLogWriter
from metta.sim.simulation_config import SimulationConfig
from metta.sim.simulation_stats_db import SimulationStatsDB
from metta.sim.stats import DuckDBStatsWriter
from metta.sim.thumbnail_automation import maybe_generate_and_upload_thumbnail
from metta.sim.utils import get_or_create_policy_ids
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.policy.policy import AgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator.rollout import Rollout

SYNTHETIC_EVAL_SUITE = "training"

logger = logging.getLogger(__name__)


class SimulationCompatibilityError(Exception):
    """Raised when there's a compatibility issue that prevents simulation from running."""

    pass


class Simulation:
    """A simulation that runs episodes sequentially using Rollout (non-parallel)."""

    def __init__(
        self,
        cfg: SimulationConfig,
        policy_uri: str | None,
        replay_dir: str | None,
        stats_dir: str = "/tmp/stats",
        stats_client: StatsClient | None = None,
        stats_epoch_id: uuid.UUID | None = None,
        eval_task_id: uuid.UUID | None = None,
    ):
        self._config = cfg
        self._id = uuid.uuid4().hex[:12]
        self._eval_task_id = eval_task_id

        sim_stats_dir = (Path(stats_dir) / self._id).resolve()
        sim_stats_dir.mkdir(parents=True, exist_ok=True)
        self._stats_dir = sim_stats_dir
        self._stats_writer = DuckDBStatsWriter(sim_stats_dir)
        self._replay_writer: ReplayLogWriter | None = None
        if replay_dir is not None:
            self._replay_writer = ReplayLogWriter(f"{replay_dir}/{self._id}")
        self._device = torch.device("cpu")

        self._full_name = f"{cfg.suite}/{cfg.name}"

        if policy_uri:
            policy_artifact = CheckpointManager.load_artifact_from_uri(policy_uri)
            resolved_policy_uri = CheckpointManager.normalize_uri(policy_uri)
        else:
            policy_artifact = PolicyArtifact(policy=MockAgent())
            resolved_policy_uri = "mock://"

        self._num_episodes = cfg.num_episodes
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

        # We need to create a temporary environment to get PolicyEnvInterface
        # This is needed to instantiate the policies
        self._env_cfg: MettaGridConfig = cfg.env

        # Calculate agent assignments
        self._policy_agents_per_env = max(1, int(self._agents_per_env * self._policy_agents_pct))
        self._npc_agents_per_env = self._agents_per_env - self._policy_agents_per_env

        self._episode_counter = 0
        self._rollouts_data = []  # Store rollout data for stats

    def _materialize_policy(
        self,
        artifact: PolicyArtifact,
        existing_policy: Policy | None,
        policy_env_info: PolicyEnvInterface,
    ) -> Policy:
        using_existing = existing_policy is not None
        if using_existing:
            policy = existing_policy
        else:
            policy = artifact.instantiate(policy_env_info, device=self._device)

        policy = policy.to(self._device)
        policy.eval()

        if using_existing and hasattr(policy, "initialize_to_environment"):
            policy.initialize_to_environment(policy_env_info, self._device)
        return policy

    @classmethod
    def create(
        cls,
        sim_config: SimulationConfig,
        stats_dir: str = "./train_dir/stats",
        replay_dir: str | None = "./train_dir/replays",
        policy_uri: str | None = None,
    ) -> "Simulation":
        """Create a Simulation with sensible defaults."""
        # Create replay directory path with simulation name
        full_replay_dir = f"{replay_dir}/{sim_config.name}" if replay_dir is not None else None

        # Create and return simulation
        return cls(
            sim_config,
            policy_uri,
            stats_dir=stats_dir,
            replay_dir=full_replay_dir,
        )

    def start_simulation(self) -> None:
        """Start the simulation."""
        logger.info(
            "Sim '%s': Sequential execution, %d episodes, %d agents (%.0f%% candidate)",
            self._full_name,
            self._num_episodes,
            self._agents_per_env,
            100 * self._policy_agents_per_env / self._agents_per_env,
        )
        logger.info("Stats dir: %s", self._stats_dir)

        # Create policy environment interface and instantiate policies
        policy_env_info = PolicyEnvInterface.from_mg_cfg(self._env_cfg)
        self._policy = self._materialize_policy(self._policy_artifact, self._policy, policy_env_info)

        if self._npc_artifact is not None:
            self._npc_policy = self._materialize_policy(self._npc_artifact, self._npc_policy, policy_env_info)

        self._t0 = time.time()

    def _create_agent_policies(self, seed: int) -> list[AgentPolicy]:
        """Create agent policies for a single episode."""
        agent_policies = []

        for agent_id in range(self._agents_per_env):
            # Assign main policy to first N agents, NPC policy to remaining
            if agent_id < self._policy_agents_per_env:
                agent_policy = self._policy.agent_policy(agent_id)
            else:
                if self._npc_policy is not None:
                    agent_policy = self._npc_policy.agent_policy(agent_id)
                else:
                    # Fallback to main policy if no NPC policy
                    agent_policy = self._policy.agent_policy(agent_id)

            agent_policies.append(agent_policy)

        return agent_policies

    def simulate(self) -> SimulationResults:
        """Run the simulation; returns the merged `StatsDB`."""
        self.start_simulation()

        self._policy.reset_memory()
        if self._npc_policy is not None:
            self._npc_policy.reset_memory()

        # Run episodes sequentially
        for episode_idx in range(self._num_episodes):
            # Check timeout
            if (time.time() - self._t0) >= self._max_time_s:
                logger.warning("Simulation timeout reached after %d/%d episodes", episode_idx, self._num_episodes)
                break

            # Create agent policies for this episode
            agent_policies = self._create_agent_policies(seed=episode_idx)

            # Create and run rollout
            rollout = Rollout(
                self._env_cfg,
                agent_policies,
                max_action_time_ms=10000,
                render_mode=None,
                seed=episode_idx,
            )

            rollout.run_until_done()

            self._episode_counter += 1

            # Log progress
            if (episode_idx + 1) % 10 == 0:
                elapsed = time.time() - self._t0
                logger.info(
                    "Sim '%s': %d/%d episodes complete (%.1fs)",
                    self._full_name,
                    episode_idx + 1,
                    self._num_episodes,
                    elapsed,
                )

        return self.end_simulation()

    def end_simulation(self) -> SimulationResults:
        db = self._from_shards_and_context()

        # Generate thumbnail before writing to database so we can include the URL
        thumbnail_url = self._maybe_generate_thumbnail()
        self._write_remote_stats(db, thumbnail_url=thumbnail_url)

        logger.info(
            "Sim '%s' finished: %d episodes in %.1fs",
            self._full_name,
            self._episode_counter,
            time.time() - self._t0,
        )
        return SimulationResults(db)

    def _maybe_generate_thumbnail(self) -> str | None:
        """Generate thumbnail if this is the first run for this eval_name."""
        try:
            if self._replay_writer is None:
                logger.debug("Replay logging disabled; skipping thumbnail generation")
                return None

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
            logger.error(f"Thumbnail generation failed for {self._full_name}: {e}", exc_info=True)
            return None

    def _from_shards_and_context(self) -> SimulationStatsDB:
        """Merge all *.duckdb* shards for this simulation → one `StatsDB`."""
        # Create agent map using URIs for database integration
        agent_map: Dict[int, str] = {}

        # Add policy agents to the map if they have a URI
        if self._policy_uri:
            for agent_id in range(self._policy_agents_per_env):
                agent_map[agent_id] = self._policy_uri

        # Add NPC agents to the map if they exist
        if self._npc_policy is not None and self._npc_policy_uri:
            for agent_id in range(self._policy_agents_per_env, self._agents_per_env):
                agent_map[agent_id] = self._npc_policy_uri

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
                for agent_id in range(self._policy_agents_per_env):
                    agent_map[agent_id] = policy_ids[self._policy_uri]

            if self._npc_policy_uri:
                for agent_id in range(self._policy_agents_per_env, self._agents_per_env):
                    agent_map[agent_id] = policy_ids[self._npc_policy_uri]

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
                    logger.error(f"Failed to record episode {episode_id} remotely: {e}", exc_info=True)
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

    def get_replays(self) -> dict:
        """Get all replays for this simulation."""
        if self._replay_writer is None:
            return {}
        return self._replay_writer.episodes.values()

    def get_replay(self) -> dict:
        """Makes sure this sim has a single replay, and return it."""
        if self._replay_writer is None:
            raise ValueError("Attempting to get single replay, but simulation has no replay writer.")

        # If no episodes yet, return empty replay structure
        if len(self._replay_writer.episodes) == 0:
            # Return initial replay structure with action names
            actions_dict = self._env_cfg.game.actions.model_dump()
            action_names = [name for name, cfg in actions_dict.items() if cfg.get("enabled", True)]

            return {
                "version": 2,
                "action_names": action_names,
                "item_names": [],
                "type_names": [],
                "num_agents": self._agents_per_env,
                "max_steps": self._env_cfg.game.max_steps,
                "map_size": [self._env_cfg.game.map.height, self._env_cfg.game.map.width]
                if hasattr(self._env_cfg.game, "map")
                else [0, 0],
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
