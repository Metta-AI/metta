"""
Vectorized simulation runner.

• Launches a MettaGrid vec-env batch
• Each worker writes its own *.duckdb* shard
• At shutdown the shards are merged into **one** StatsDB object that the
  caller can further merge / export.
"""

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

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.agent.utils import obs_to_td
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.util.heartbeat import record_heartbeat
from metta.mettagrid import MettaGridEnv, dtype_actions
from metta.mettagrid.replay_writer import ReplayWriter
from metta.mettagrid.stats_writer import StatsWriter
from metta.rl.policy_management import initialize_policy_for_environment
from metta.rl.vecenv import make_vecenv
from metta.sim.simulation_config import SimulationConfig
from metta.sim.simulation_stats_db import SimulationStatsDB
from metta.sim.thumbnail_automation import maybe_generate_and_upload_thumbnail
from metta.sim.utils import get_or_create_policy_ids, wandb_policy_name_to_uri

# Prefix for synthetic evaluation framework simulations (not real environment evaluations)
SYNTHETIC_EVAL_PREFIX = "eval/"

logger = logging.getLogger(__name__)


class SimulationCompatibilityError(Exception):
    """Raised when there's a compatibility issue that prevents simulation from running."""

    pass


class Simulation:
    """
    A vectorized batch of MettaGrid environments sharing the same parameters.
    """

    def __init__(
        self,
        name: str,
        cfg: SimulationConfig,
        policy_pr: PolicyRecord,
        policy_store: PolicyStore,
        device: torch.device,
        vectorization: str,
        sim_suite_name: str | None = None,
        stats_dir: str = "/tmp/stats",
        replay_dir: str | None = None,
        stats_client: StatsClient | None = None,
        stats_epoch_id: uuid.UUID | None = None,
        wandb_policy_name: str | None = None,
        eval_task_id: uuid.UUID | None = None,
        episode_tags: list[str] | None = None,
    ):
        self._name = name
        self._sim_suite_name = sim_suite_name
        self._config = cfg
        self._id = uuid.uuid4().hex[:12]
        self._eval_task_id = eval_task_id
        self._episode_tags = episode_tags
        self._wandb_policy_name: str | None = None
        self._wandb_uri: str | None = None
        if wandb_policy_name is not None:
            self._wandb_policy_name, self._wandb_uri = wandb_policy_name_to_uri(wandb_policy_name)

        # ---------------- env config ----------------------------------- #
        replay_dir = f"{replay_dir}/{self._id}" if replay_dir else None

        sim_stats_dir = (Path(stats_dir) / self._id).resolve()
        sim_stats_dir.mkdir(parents=True, exist_ok=True)
        self._stats_dir = sim_stats_dir
        self._stats_writer = StatsWriter(sim_stats_dir)
        self._replay_writer = ReplayWriter(replay_dir)
        self._device = device

        # ----------------
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
            cfg.env.to_curriculum(),
            vectorization,
            num_envs=num_envs,
            stats_writer=self._stats_writer,
            replay_writer=self._replay_writer,
        )

        self._num_envs = num_envs
        self._min_episodes = episodes_per_env
        self._max_time_s = cfg.max_time_s
        self._agents_per_env = cfg.env.game.num_agents

        # ---------------- policies ------------------------------------- #
        self._policy_pr = policy_pr
        self._policy_store = policy_store
        self._npc_pr = policy_store.policy_record(cfg.npc_policy_uri) if cfg.npc_policy_uri else None
        self._policy_agents_pct = cfg.policy_agents_pct if self._npc_pr is not None else 1.0

        self._stats_client: StatsClient | None = stats_client
        self._stats_epoch_id: uuid.UUID | None = stats_epoch_id

        # Get the MettaGridEnv (may be wrapped in CurriculumEnv)
        driver_env = self._vecenv.driver_env  # type: ignore
        metta_grid_env: MettaGridEnv = getattr(driver_env, "_env", driver_env)
        assert isinstance(metta_grid_env, MettaGridEnv), f"Expected MettaGridEnv, got {type(metta_grid_env)}"

        # Initialize policy to environment
        initialize_policy_for_environment(
            policy_record=self._policy_pr,
            metta_grid_env=metta_grid_env,
            device=self._device,
            restore_feature_mapping=True,
        )

        if self._npc_pr is not None:
            # Initialize NPC policy to environment
            initialize_policy_for_environment(
                policy_record=self._npc_pr,
                metta_grid_env=metta_grid_env,
                device=self._device,
                restore_feature_mapping=True,
            )

        # ---------------- agent-index bookkeeping ---------------------- #
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

    @classmethod
    def create(
        cls,
        sim_config: SimulationConfig,
        policy_store: PolicyStore,
        device: str,
        vectorization: str,
        stats_dir: str = "./train_dir/stats",
        replay_dir: str = "./train_dir/replays",
        policy_uri: str | None = None,
        run_name: str = "simulation_run",
    ) -> "Simulation":
        """Create a Simulation with sensible defaults.

        Args:
            sim_config: Simulation configuration with environment settings
            policy_store: PolicyStore instance for managing policies
            device: Device to run on (e.g., "cpu", "cuda")
            vectorization: Vectorization backend (e.g., "serial", "multiprocessing")
            stats_dir: Directory for simulation statistics
            replay_dir: Directory for replay files
            policy_uri: Optional policy URI to load (None for mock policy)
            run_name: Name for the mock run if no policy URI provided

        Returns:
            Configured Simulation instance
        """
        # Get policy record or create a mock
        policy_record = policy_store.policy_record_or_mock(policy_uri, run_name)

        # Create replay directory path with simulation name
        full_replay_dir = f"{replay_dir}/{sim_config.name}"

        # Create and return simulation
        return cls(
            sim_config.name,
            sim_config,
            policy_record,
            policy_store,
            device=torch.device(device),
            vectorization=vectorization,
            stats_dir=stats_dir,
            replay_dir=full_replay_dir,
        )

    def start_simulation(self) -> None:
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
        self._env_done_flags = [False] * self._num_envs

        self._t0 = time.time()

    def generate_actions(self) -> np.ndarray:
        """
        Generate actions for the simulation.
        """
        if __debug__:
            # Debug assertion: verify indices are correctly ordered
            # Policy indices should be 0 to N-1
            # NPC indices should be N to M-1
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

            if self._npc_pr is not None and num_npc > 0:
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

            # Verify no overlap between policy and NPC indices
            if num_policy > 0 and num_npc > 0:
                policy_set = set(self._policy_idxs)
                npc_set = set(self._npc_idxs)
                assert policy_set.isdisjoint(npc_set), (
                    f"Policy and NPC indices should not overlap. Overlap: {policy_set.intersection(npc_set)}"
                )

        # ---------------- forward passes ------------------------- #
        with torch.no_grad():
            # Candidate-policy agents
            my_obs = self._obs[self._policy_idxs]
            td = obs_to_td(my_obs, self._device)  # One-liner conversion
            policy = self._policy_pr.policy
            policy(td)
            policy_actions = td["actions"]

            # NPC agents (if any)
            if self._npc_pr is not None and len(self._npc_idxs):
                npc_obs = self._obs[self._npc_idxs]
                td = obs_to_td(npc_obs, self._device)  # One-liner conversion
                npc_policy = self._npc_pr.policy
                try:
                    npc_policy(td)
                    npc_actions = td["actions"]
                except Exception as e:
                    logger.error(f"Error generating NPC actions: {e}")
                    raise SimulationCompatibilityError(
                        f"[{self._name}] Error generating NPC actions for {self._npc_pr.run_name}: {e}"
                    ) from e

        # ---------------- action stitching ----------------------- #
        actions = policy_actions
        if self._npc_agents_per_env:
            # Reshape policy and npc actions to (num_envs, agents_per_env, action_dim)
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
        # ---------------- env.step ------------------------------- #
        obs, rewards, dones, trunc, infos = self._vecenv.step(actions_np)

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

    def _maybe_generate_thumbnail(self) -> str | None:
        """Generate thumbnail if this is the first run for this eval_name.

        Returns:
            Thumbnail URL if generated successfully, None otherwise
        """
        try:
            # Skip synthetic evaluation framework simulations
            if self._name.startswith(SYNTHETIC_EVAL_PREFIX):
                logger.debug(f"Skipping thumbnail generation for synthetic simulation: {self._name}")
                return None

            # Get any replay data from this simulation
            if not self._replay_writer.episodes:
                logger.warning(f"No replay data available for thumbnail generation: {self._name}")
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
            logger.error(f"Thumbnail generation failed for {self._name}: {e}")
            return None

    def end_simulation(self) -> SimulationResults:
        # ---------------- teardown & DB merge ------------------------ #
        self._vecenv.close()
        db = self._from_shards_and_context()

        # Generate thumbnail before writing to database so we can include the URL
        thumbnail_url = self._maybe_generate_thumbnail()
        self._write_remote_stats(db, thumbnail_url=thumbnail_url)

        logger.info(
            "Sim '%s' finished: %d episodes in %.1fs",
            self._name,
            int(self._episode_counters.sum()),
            time.time() - self._t0,
        )
        return SimulationResults(db)

    def simulate(self) -> SimulationResults:
        """
        Run the simulation; returns the merged `StatsDB`.
        """
        self.start_simulation()

        self._policy_pr.policy.reset_memory()
        if self._npc_pr is not None:
            self._npc_pr.policy.reset_memory()

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
        # Make sure we're creating a dictionary of the right type
        agent_map: Dict[int, PolicyRecord] = {}

        # Add policy agents to the map
        for idx in self._policy_idxs:
            agent_map[int(idx.item())] = self._policy_pr

        # Add NPC agents to the map if they exist
        if self._npc_pr is not None:
            for idx in self._npc_idxs:
                agent_map[int(idx.item())] = self._npc_pr

        suite_name = "" if self._sim_suite_name is None else self._sim_suite_name
        db = SimulationStatsDB.from_shards_and_context(
            self._id, self._stats_dir, agent_map, self._name, suite_name, self._policy_pr
        )
        return db

    def _get_policy_name(self) -> str:
        return self._wandb_policy_name if self._wandb_policy_name is not None else self._policy_pr.run_name

    def _get_policy_uri(self) -> str:
        return self._wandb_uri if self._wandb_uri is not None else self._policy_pr.uri

    def _write_remote_stats(self, stats_db: SimulationStatsDB, thumbnail_url: str | None = None) -> None:
        """Write stats to the remote stats database."""
        if self._stats_client is not None:
            policy_name = self._get_policy_name()
            policy_uri = self._get_policy_uri()
            policy_details: list[tuple[str, str, str | None]] = [(policy_name, policy_uri, None)]
            if self._npc_pr is not None:
                policy_details.append((self._npc_pr.run_name, self._npc_pr.uri, None))

            policy_ids = get_or_create_policy_ids(self._stats_client, policy_details, self._stats_epoch_id)

            agent_map: Dict[int, uuid.UUID] = {}
            for idx in self._policy_idxs:
                agent_map[int(idx.item())] = policy_ids[policy_name]

            if self._npc_pr is not None:
                for idx in self._npc_idxs:
                    agent_map[int(idx.item())] = policy_ids[self._npc_pr.run_name]

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
                episode_tags = self._episode_tags or None
                try:
                    self._stats_client.record_episode(
                        agent_policies=agent_map,
                        agent_metrics=agent_metrics,
                        primary_policy_id=policy_ids[policy_name],
                        stats_epoch=self._stats_epoch_id,
                        eval_name=self._name,
                        simulation_suite="" if self._sim_suite_name is None else self._sim_suite_name,
                        replay_url=episode_row.get("replay_url"),
                        attributes=attributes,
                        eval_task_id=self._eval_task_id,
                        tags=episode_tags,
                        thumbnail_url=thumbnail_url,
                    )
                except Exception as e:
                    logger.error(f"Failed to record episode {episode_id} remotely: {e}")
                    # Continue with other episodes even if one fails

    def get_replays(self) -> dict:
        """Get all replays for this simulation."""
        return self._replay_writer.episodes.values()

    def get_replay(self) -> dict:
        """Makes sure this sim has a single replay, and return it."""
        if len(self._replay_writer.episodes) != 1:
            raise ValueError("Attempting to get single replay, but simulation has multiple episodes")
        for _, episode_replay in self._replay_writer.episodes.items():
            return episode_replay.get_replay_data()

    def get_envs(self):
        """Returns a list of all envs in the simulation."""
        return self._vecenv.envs

    def get_env(self):
        """Make sure this sim has a single env, and return it."""
        if len(self._vecenv.envs) != 1:
            raise ValueError("Attempting to get single env, but simulation has multiple envs")
        return self._vecenv.envs[0]

    @property
    def policy_record(self) -> PolicyRecord:
        """Get the policy record used in this simulation."""
        return self._policy_pr

    @property
    def name(self) -> str:
        return self._name

    def get_policy_state(self):
        """Get the policy state for memory manipulation.

        Returns a PolicyState object with lstm_h and lstm_c attributes if available.
        Note: The actual state management depends on the specific policy implementation.
        """
        # The policy is the LSTM wrapper
        policy = self._policy_pr.policy

        # Try to get LSTM state from the policy
        # This depends on the specific policy implementation
        lstm_h = None
        lstm_c = None

        # Check if it's a pufferlib LSTMWrapper
        if hasattr(policy, "lstm") and hasattr(policy.lstm, "weight_hh_l0"):
            # For pufferlib LSTMWrapper, the state is managed internally during forward pass
            # We would need to track it differently or access it through the forward pass
            # For now, return None as this requires deeper integration
            return None

        # Check if policy has direct lstm_h and lstm_c attributes (custom implementations)
        if hasattr(policy, "lstm_h") and hasattr(policy, "lstm_c"):
            lstm_h = policy.lstm_h
            lstm_c = policy.lstm_c

        # Check if policy has a component that manages LSTM state (MettaAgent style)
        elif hasattr(policy, "component") and hasattr(policy.component, "lstm"):
            lstm_component = policy.component.lstm
            if hasattr(lstm_component, "lstm_h") and hasattr(lstm_component, "lstm_c"):
                # These are dictionaries mapping env_id to tensors
                # Get the first one for single-env simulations
                if 0 in lstm_component.lstm_h and 0 in lstm_component.lstm_c:
                    lstm_h = lstm_component.lstm_h[0]
                    lstm_c = lstm_component.lstm_c[0]

        if lstm_h is not None and lstm_c is not None:
            # Return an object-like dict that allows attribute access
            class PolicyState:
                def __init__(self, lstm_h, lstm_c):
                    self.lstm_h = lstm_h
                    self.lstm_c = lstm_c

            return PolicyState(lstm_h, lstm_c)

        return None


@dataclass
class SimulationResults:
    """
    Results of a simulation.
    For now just a stats db. Replay plays can be retrieved from the stats db.
    """

    stats_db: SimulationStatsDB
    replay_urls: dict[str, list[str]] | None = None  # Maps simulation names to lists of replay URLs
