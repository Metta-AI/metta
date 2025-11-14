"""Policy evaluation management."""

from __future__ import annotations

import logging
import uuid
from functools import partial
from typing import Any, Optional

import torch
from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.cogworks.curriculum import Curriculum
from metta.common.util.git_helpers import GitError, get_task_commit_hash
from metta.common.util.git_repo import REPO_SLUG
from metta.common.wandb.context import WandbRun
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training import TrainerComponent
from metta.rl.training.optimizer import is_schedulefree_optimizer
from metta.sim.handle_results import render_eval_summary, send_eval_results_to_wandb
from metta.sim.runner import SimulationRunResult, run_simulations
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_replay_dir
from mettagrid.base_config import Config
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import PolicySpec

logger = logging.getLogger(__name__)


class EvaluatorConfig(Config):
    """Configuration for evaluation."""

    epoch_interval: int = 100  # 0 to disable
    evaluate_local: bool = True
    evaluate_remote: bool = False
    num_training_tasks: int = 2
    simulations: list[SimulationConfig] = Field(default_factory=list)
    training_replay_envs: list[SimulationConfig] = Field(
        default_factory=list,
        description=(
            "Optional explicit simulation configs to use when recording training replays. "
            "When provided, these override the default behaviour of sampling num_training_tasks "
            "from the active curriculum."
        ),
    )
    replay_dir: Optional[str] = None
    skip_git_check: bool = Field(default=False)
    git_hash: str | None = Field(default=None)
    allow_eval_without_stats: bool = Field(
        default=False,
        description="Allow evaluations to run without stats infrastructure (useful for local development/testing)",
    )


class NoOpEvaluator(TrainerComponent):
    """No-op evaluator for when evaluation is disabled."""

    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        pass


class Evaluator(TrainerComponent):
    """Manages policy evaluation."""

    def __init__(
        self,
        config: EvaluatorConfig,
        device: torch.device,
        seed: int,
        run_name: str,
        stats_client: Optional[StatsClient] = None,
        wandb_run: Optional[WandbRun] = None,
    ):
        super().__init__()
        self._master_only = True
        self._config = config
        self._device = device
        self._seed = seed
        self._run_name = run_name
        self._stats_client = stats_client
        self._wandb_run = wandb_run

        self._configure_evaluation_settings(
            eval_cfg=self._config,
            stats_client=self._stats_client,
        )

    @staticmethod
    def _configure_evaluation_settings(
        *,
        eval_cfg: EvaluatorConfig,
        stats_client: Optional[StatsClient],
    ) -> None:
        # Set default replay directory
        if eval_cfg.replay_dir is None:
            eval_cfg.replay_dir = auto_replay_dir()
            logger.info(f"Setting replay_dir to {eval_cfg.replay_dir}")

        # Configure remote evaluations
        if not eval_cfg.evaluate_remote:
            return

        # Check prerequisites for remote evaluations
        if not stats_client:
            eval_cfg.evaluate_remote = False
            logger.info("Not connected to stats server, disabling remote evaluations")
            return

        if not eval_cfg.epoch_interval:
            eval_cfg.evaluate_remote = False
            logger.info("Epoch interval set to 0, disabling remote evaluations")
            return

        # Get git hash if not already set
        if not eval_cfg.git_hash:
            try:
                eval_cfg.git_hash = get_task_commit_hash(
                    target_repo=REPO_SLUG,
                    skip_git_check=eval_cfg.skip_git_check,
                )
            except GitError as e:
                raise GitError(f"{e}\n\nYou can skip this check with evaluator.skip_git_check=true") from e

            if eval_cfg.git_hash:
                logger.info(f"Git hash for remote evaluations: {eval_cfg.git_hash}")
            else:
                logger.info("No git hash available for remote evaluations")

    def should_evaluate(self, epoch: int) -> bool:
        interval = self._config.epoch_interval
        if interval <= 0:
            return False
        return epoch % interval == 0

    def write_eval_results_to_observatory(
        self,
        *,
        stats_client: StatsClient,
        rollout_results: list[SimulationRunResult],
        policy_spec: PolicySpec,
        epoch: int,
        agent_step: int,
    ) -> None:
        """Write evaluation results to the observatory by creating a DuckDB and uploading it."""
        from metta.app_backend.episode_stats_db import (
            create_episode_stats_db,
            insert_agent_metric,
            insert_agent_policy,
            insert_episode,
            insert_episode_tag,
        )

        # Create or get policy
        policy_id = stats_client.create_policy(
            name=self._run_name,
            attributes={},
            is_system_policy=False,
        )

        # Create policy version
        policy_version_id = stats_client.create_policy_version(
            policy_id=policy_id.id,
            git_hash=self._config.git_hash,
            policy_spec=policy_spec.model_dump(mode="json"),
            attributes={"epoch": epoch, "agent_step": agent_step},
        )

        # Create DuckDB with episode stats
        try:
            conn, duckdb_path = create_episode_stats_db()

            # Process each simulation result
            for sim_result in rollout_results:
                sim_config = sim_result.run
                results = sim_result.results

                # Process each episode
                for episode_idx in range(len(results.rewards)):
                    episode_id = str(uuid.uuid4())
                    assignments = results.assignments[episode_idx]
                    rewards = results.rewards[episode_idx]
                    replay_url = list(sim_result.replay_urls.values())[0]
                    # Insert episode record
                    insert_episode(
                        conn,
                        episode_id=episode_id,
                        primary_pv_id=str(policy_version_id.id),
                        replay_url=replay_url,
                        thumbnail_url=None,
                        eval_task_id=None,
                    )

                    # Insert episode tags
                    for key, value in sim_config.episode_tags.items():
                        insert_episode_tag(conn, episode_id, key, value)

                    # Insert agent policies and metrics
                    for agent_id in range(len(assignments)):
                        # For now, assume single policy (index 0)
                        pv_id = str(policy_version_id.id)

                        insert_agent_policy(conn, episode_id, pv_id, agent_id)

                        # Insert reward metric
                        insert_agent_metric(conn, episode_id, agent_id, "reward", float(rewards[agent_id]))
                        agent_metrics = results.stats[episode_idx]["agent"][agent_id]
                        for metric_name, metric_value in agent_metrics.items():
                            insert_agent_metric(conn, episode_id, agent_id, metric_name, metric_value)

            conn.close()

            # Upload DuckDB file
            logger.info(f"Uploading evaluation results to observatory (DuckDB size: {duckdb_path})")
            response = stats_client.bulk_upload_episodes(str(duckdb_path))
            logger.info(
                f"Successfully uploaded {response.episodes_created} episodes to observatory at {response.duckdb_s3_uri}"
            )

        except Exception as e:
            logger.error(f"Failed to write evaluation results to observatory: {e}", exc_info=True)
            raise

    def evaluate(
        self,
        policy_uri: Optional[str],
        curriculum: Any,
        epoch: int,
        agent_step: int,
    ) -> None:
        if not policy_uri:
            logger.warning("No policy URI available for evaluation")
            return

        # Build simulation configurations
        sims = self._build_simulations(curriculum)

        # Try remote evaluation first if configured
        # Pasha: FIX THIS

        # Local evaluation
        if self._config.evaluate_local:
            rollout_results, policy_spec = self._evaluate_local(policy_uri=policy_uri, simulations=sims)
            render_eval_summary(rollout_results, policy_names=[self._spec_display_name(policy_spec)])

            if self._wandb_run:
                send_eval_results_to_wandb(
                    rollout_results=rollout_results,
                    epoch=epoch,
                    agent_step=agent_step,
                    wandb_run=self._wandb_run,
                    should_finish_run=False,
                )

            if self._stats_client:
                self.write_eval_results_to_observatory(
                    stats_client=self._stats_client,
                    rollout_results=rollout_results,
                    policy_spec=policy_spec,
                    epoch=epoch,
                    agent_step=agent_step,
                )

    def _build_policy_spec(self, policy_uri: str) -> PolicySpec:
        return CheckpointManager.policy_spec_from_uri(
            policy_uri,
            device=self._device,
        )

    @staticmethod
    def _spec_display_name(policy_spec: PolicySpec) -> str:
        init_kwargs = policy_spec.init_kwargs or {}
        return init_kwargs.get("display_name") or policy_spec.name

    def _build_simulations(self, curriculum: Curriculum) -> list[SimulationConfig]:
        sims = []

        # Add training task evaluations
        if self._config.training_replay_envs:
            for idx, sim_cfg in enumerate(self._config.training_replay_envs):
                # Clone to avoid mutating caller-provided instances
                sim_copy = sim_cfg.model_copy(deep=True)
                # Ensure required identifiers are populated
                if not getattr(sim_copy, "suite", None):
                    sim_copy.suite = "training"
                if not getattr(sim_copy, "name", None):
                    sim_copy.name = f"train_task_{idx}"
                sims.append(sim_copy)
        else:
            for i in range(self._config.num_training_tasks):
                sims.append(
                    SimulationConfig(
                        suite="training",
                        name=f"train_task_{i}",
                        env=curriculum.get_task().get_env_cfg().model_copy(deep=True),
                    )
                )

        # Add configured simulations
        sims.extend(self._config.simulations)

        return sims

    def _evaluate_local(
        self,
        policy_uri: str,
        simulations: list[SimulationConfig],
    ) -> tuple[list[SimulationRunResult], PolicySpec]:
        logger.info(f"Evaluating policy locally from {policy_uri}")

        policy_spec = self._build_policy_spec(policy_uri)
        policy_initializers = [partial(initialize_or_load_policy, policy_spec=policy_spec)]
        rollout_results = run_simulations(
            policy_initializers=policy_initializers,
            simulations=[sim.to_simulation_run_config() for sim in simulations],
            replay_dir=self._config.replay_dir,
            seed=self._seed,
            enable_replays=True,
        )
        return rollout_results, policy_spec

    def on_epoch_end(self, epoch: int) -> None:
        if not self.should_evaluate(epoch):
            return

        policy_uri = self.context.latest_policy_uri()
        if not policy_uri:
            logger.warning("Evaluator: skipping epoch %s because no policy checkpoint is available", epoch)
            return

        curriculum: Curriculum | None = getattr(self.context.env, "_curriculum", None)
        if curriculum is None:
            logger.warning("Evaluator: curriculum unavailable; skipping evaluation")
            return

        optimizer = getattr(self.context, "optimizer", None)
        is_schedulefree = optimizer is not None and is_schedulefree_optimizer(optimizer)
        if is_schedulefree and optimizer is not None:
            optimizer.eval()

        self.evaluate(
            policy_uri=policy_uri,
            curriculum=curriculum,
            epoch=epoch,
            agent_step=self.context.agent_step,
        )

        # Restore train mode after evaluation for ScheduleFree optimizers
        if is_schedulefree and optimizer is not None:
            optimizer.train()
