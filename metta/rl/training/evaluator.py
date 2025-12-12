"""Policy evaluation management."""

from __future__ import annotations

import io
import logging
import os
import uuid
import zipfile
from typing import Any, Optional

import torch
from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.cogworks.curriculum import Curriculum
from metta.common.util.git_helpers import GitError, get_current_git_branch, get_task_commit_hash
from metta.common.util.git_repo import REPO_SLUG
from metta.common.util.heartbeat import record_heartbeat
from metta.common.wandb.context import WandbRun
from metta.rl.training import TrainerComponent
from metta.rl.training.optimizer import is_schedulefree_optimizer
from metta.sim.handle_results import render_eval_summary
from metta.sim.remote import evaluate_remotely
from metta.sim.simulate_and_record import ObservatoryWriter, WandbWriter, simulate_and_record
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_replay_dir
from mettagrid.base_config import Config
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.submission import POLICY_SPEC_FILENAME
from mettagrid.util.file import write_data
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri

logger = logging.getLogger(__name__)


class EvaluatorConfig(Config):
    """Configuration for evaluation."""

    epoch_interval: int = 100  # 0 to disable
    evaluate_local: bool = Field(
        default_factory=lambda: os.getenv("SKYPILOT_TASK_ID") is None,
        description="Run evals locally. Defaults to True locally, False on SkyPilot.",
    )
    evaluate_remote: bool = Field(
        default_factory=lambda: os.getenv("SKYPILOT_TASK_ID") is not None,
        description="Run evals remotely via Observatory. Defaults to False locally, True on SkyPilot.",
    )
    num_training_tasks: int = 2
    parallel_evals: int = Field(
        default=9,
        description="Max number of simulations to run in parallel during eval; set to 1 to keep sequential",
        ge=1,
    )
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
    use_branch_checkout: bool = Field(
        default=False,
        description="Use git branch name instead of commit SHA for remote eval checkout (useful for rewritten history)",
    )
    verbose: bool = Field(default=False)
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

        self._replay_dir = config.replay_dir or auto_replay_dir()
        self._evaluate_remote = config.evaluate_remote and stats_client is not None

        self._git_hash = config.git_hash
        if self._evaluate_remote and not self._git_hash:
            try:
                if config.use_branch_checkout:
                    # Use branch name instead of commit SHA
                    branch_name = get_current_git_branch(
                        target_repo=REPO_SLUG,
                        skip_git_check=config.skip_git_check,
                    )
                    if branch_name:
                        logger.info(f"Using branch-based checkout: {branch_name}")
                        self._git_hash = branch_name
                    else:
                        logger.warning("Could not determine branch, falling back to commit hash")
                        self._git_hash = get_task_commit_hash(
                            target_repo=REPO_SLUG,
                            skip_git_check=config.skip_git_check,
                        )
                else:
                    # Use commit SHA (default behavior)
                    self._git_hash = get_task_commit_hash(
                        target_repo=REPO_SLUG,
                        skip_git_check=config.skip_git_check,
                    )
            except GitError as e:
                raise GitError(f"{e}\n\nYou can skip this check with evaluator.skip_git_check=true") from e

    def should_evaluate(self, epoch: int) -> bool:
        interval = self._config.epoch_interval
        if interval <= 0:
            return False
        return epoch % interval == 0

    def _create_submission_zip(self, policy_spec: PolicySpec) -> bytes:
        """Create a submission zip containing policy_spec.json."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr(POLICY_SPEC_FILENAME, policy_spec.model_dump_json())
        return buffer.getvalue()

    def _upload_submission_zip(self, policy_spec: PolicySpec) -> str | None:
        """Upload a submission zip to S3 and return the s3_path."""
        checkpoint_uri = policy_spec.init_kwargs.get("checkpoint_uri")
        if not checkpoint_uri or not checkpoint_uri.startswith("s3://"):
            return None

        submission_path = checkpoint_uri.replace(".mpt", "-submission.zip")
        zip_data = self._create_submission_zip(policy_spec)
        write_data(submission_path, zip_data, content_type="application/zip")
        logger.info("Uploaded submission zip to %s", submission_path)
        return submission_path

    def _create_policy_version(
        self,
        *,
        stats_client: StatsClient,
        policy_spec: PolicySpec,
        epoch: int,
        agent_step: int,
    ) -> uuid.UUID:
        """Create a policy version in Observatory with a submission zip."""

        # Create or get policy
        policy_id = stats_client.create_policy(
            name=self._run_name,
            attributes={},
            is_system_policy=False,
        )

        # Upload submission zip to S3
        s3_path = self._upload_submission_zip(policy_spec)

        # Create policy version
        policy_version_id = stats_client.create_policy_version(
            policy_id=policy_id.id,
            git_hash=self._git_hash,
            policy_spec=policy_spec.model_dump(mode="json"),
            attributes={"epoch": epoch, "agent_step": agent_step},
            s3_path=s3_path,
        )

        return policy_version_id.id

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
        sim_run_configs = [sim.to_simulation_run_config() for sim in sims]
        policy_spec = policy_spec_from_uri(policy_uri, device=str(self._device))
        policy_version_id: uuid.UUID | None = None
        if self._stats_client:
            policy_version_id = self._create_policy_version(
                stats_client=self._stats_client,
                policy_spec=policy_spec,
                epoch=epoch,
                agent_step=agent_step,
            )

        # Remote evaluation
        if self._evaluate_remote and self._stats_client and policy_version_id:
            response = evaluate_remotely(
                policy_version_id=policy_version_id,
                simulations=sim_run_configs,
                stats_client=self._stats_client,
                git_hash=self._git_hash,
                push_metrics_to_wandb=(self._wandb_run is not None),
            )
            logger.info(f"Created remote evaluation task {response}")

        # Local evaluation
        if self._config.evaluate_local:
            logger.info(f"Evaluating policy locally from {policy_uri}")

            observatory_writer = None
            if self._stats_client and policy_version_id:
                policy_version_id_str = str(policy_version_id)
                observatory_writer = ObservatoryWriter(
                    stats_client=self._stats_client,
                    policy_version_ids=[policy_version_id_str],
                    primary_policy_version_id=policy_version_id_str,
                )

            wandb_writer = None
            if self._wandb_run:
                wandb_writer = WandbWriter(
                    wandb_run=self._wandb_run,
                    epoch=epoch,
                    agent_step=agent_step,
                )

            def on_progress(msg: str) -> None:
                logger.info(msg)
                record_heartbeat()

            rollout_results = simulate_and_record(
                policy_specs=[policy_spec],
                simulations=sim_run_configs,
                replay_dir=self._replay_dir,
                seed=self._seed,
                max_workers=self._config.parallel_evals,
                observatory_writer=observatory_writer,
                wandb_writer=wandb_writer,
                on_progress=on_progress,
            )
            render_eval_summary(
                rollout_results, policy_names=[self._spec_display_name(policy_spec)], verbose=self._config.verbose
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
