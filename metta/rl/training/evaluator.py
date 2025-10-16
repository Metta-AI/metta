"""Policy evaluation management."""

import logging
from typing import Any, Optional
from uuid import UUID

import torch
from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.cogworks.curriculum import Curriculum
from metta.common.util.git_remote import get_git_hash_for_remote_task
from metta.common.util.git_repo import REPO_SLUG
from metta.eval.eval_request_config import EvalResults, EvalRewardSummary
from metta.eval.eval_service import evaluate_policy
from metta.rl.evaluate import (
    evaluate_policy_remote_with_checkpoint_manager,
    upload_replay_html,
)
from metta.rl.training import TrainerComponent
from metta.rl.training.optimizer import is_schedulefree_optimizer
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_replay_dir
from mettagrid.base_config import Config

logger = logging.getLogger(__name__)


class EvaluatorConfig(Config):
    """Configuration for evaluation."""

    epoch_interval: int = 100  # 0 to disable
    evaluate_local: bool = True
    evaluate_remote: bool = False
    num_training_tasks: int = 2
    simulations: list[SimulationConfig] = Field(default_factory=list)
    replay_dir: Optional[str] = None
    skip_git_check: bool = Field(default=False)
    git_hash: str | None = Field(default=None)


class NoOpEvaluator(TrainerComponent):
    """No-op evaluator for when evaluation is disabled."""

    def __init__(self) -> None:
        super().__init__()
        self._latest_scores = EvalRewardSummary()

    def get_latest_scores(self) -> EvalRewardSummary:
        return self._latest_scores

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        self.context.latest_eval_scores = self._latest_scores

    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        pass


class Evaluator(TrainerComponent):
    """Manages policy evaluation."""

    def __init__(
        self,
        config: EvaluatorConfig,
        device: torch.device,
        system_cfg: Any,
        stats_client: Optional[StatsClient] = None,
    ):
        super().__init__()
        self._master_only = True
        self._config = config
        self._device = device
        self._system_cfg = system_cfg
        self._stats_client = stats_client
        self._latest_scores = EvalRewardSummary()

        self._configure_evaluation_settings(
            eval_cfg=self._config,
            stats_client=self._stats_client,
        )

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        self.context.latest_eval_scores = self._latest_scores

    @staticmethod
    def _configure_evaluation_settings(
        *,
        eval_cfg: EvaluatorConfig,
        stats_client: Optional[StatsClient],
    ) -> None:
        if eval_cfg.replay_dir is None:
            eval_cfg.replay_dir = auto_replay_dir()
            logger.info(f"Setting replay_dir to {eval_cfg.replay_dir}")

        # Determine git hash for remote simulations
        if eval_cfg.evaluate_remote:
            if not stats_client:
                eval_cfg.evaluate_remote = False
                logger.info("Not connected to stats server, disabling remote evaluations")
            elif not eval_cfg.epoch_interval:
                eval_cfg.evaluate_remote = False
                logger.info("Epoch interval set to 0, disabling remote evaluations")
            elif not eval_cfg.git_hash:
                eval_cfg.git_hash = get_git_hash_for_remote_task(
                    target_repo=REPO_SLUG,
                    skip_git_check=eval_cfg.skip_git_check,
                    skip_cmd="evaluator.skip_git_check=true",
                )
                if eval_cfg.git_hash:
                    logger.info(f"Git hash for remote evaluations: {eval_cfg.git_hash}")
                else:
                    logger.info("No git hash available for remote evaluations")

    def should_evaluate(self, epoch: int) -> bool:
        interval = self._config.epoch_interval
        if interval <= 0:
            return False
        return epoch % interval == 0

    def evaluate(
        self,
        policy_uri: Optional[str],
        curriculum: Any,
        epoch: int,
        agent_step: int,
        stats_epoch_id: Optional[UUID] = None,
    ) -> EvalRewardSummary:
        if not policy_uri:
            logger.warning("No policy URI available for evaluation")
            return EvalRewardSummary()

        # Build simulation configurations
        sims = self._build_simulations(curriculum)

        # Try remote evaluation first if configured
        if self._config.evaluate_remote:
            try:
                self._evaluate_remote(
                    policy_uri=policy_uri,
                    simulations=sims,
                    stats_epoch_id=stats_epoch_id,
                )
                # Remote evaluation doesn't return scores directly
                # They would be reported through other channels
                self.context.latest_eval_scores = self._latest_scores
                return self._latest_scores
            except Exception as e:
                logger.error(f"Failed to evaluate policy remotely: {e}", exc_info=True)
                if not self._config.evaluate_local:
                    return EvalRewardSummary()
                logger.info("Falling back to local evaluation")

        # Local evaluation
        if self._config.evaluate_local:
            evaluation_results = self._evaluate_local(
                policy_uri=policy_uri,
                simulations=sims,
                stats_epoch_id=stats_epoch_id,
            )

            # Upload replays if available
            stats_reporter = getattr(self.context, "stats_reporter", None)
            if stats_reporter and evaluation_results.replay_urls:
                wandb_run = getattr(stats_reporter, "wandb_run", None)
                if wandb_run:
                    upload_replay_html(
                        replay_urls=evaluation_results.replay_urls,
                        agent_step=agent_step,
                        epoch=epoch,
                        wandb_run=wandb_run,
                        step_metric_key="metric/epoch",
                        epoch_metric_key="metric/epoch",
                    )

            self._latest_scores = evaluation_results.scores
            self.context.latest_eval_scores = self._latest_scores
            return evaluation_results.scores

        return EvalRewardSummary()

    def _build_simulations(self, curriculum: Curriculum) -> list[SimulationConfig]:
        sims = []

        # Add training task evaluations
        for i in range(self._config.num_training_tasks):
            sims.append(
                SimulationConfig(
                    suite="training",
                    name=f"train_task_{i}",
                    env=curriculum.get_task().get_env_cfg(),
                )
            )

        # Add configured simulations
        sims.extend(self._config.simulations)

        return sims

    def _evaluate_remote(
        self,
        policy_uri: str,
        simulations: list[SimulationConfig],
        stats_epoch_id: Optional[UUID] = None,
    ) -> None:
        logger.info(f"Evaluating policy remotely from {policy_uri}")
        stats_reporter = getattr(self.context, "stats_reporter", None)
        wandb_run = getattr(stats_reporter, "wandb_run", None) if stats_reporter else None
        evaluate_policy_remote_with_checkpoint_manager(
            policy_uri=policy_uri,
            simulations=simulations,
            stats_epoch_id=stats_epoch_id,
            stats_client=self._stats_client,
            wandb_run=wandb_run,
            evaluation_cfg=self._config,
        )

    def _evaluate_local(
        self,
        policy_uri: str,
        simulations: list[SimulationConfig],
        stats_epoch_id: Optional[UUID] = None,
    ) -> EvalResults:
        logger.info(f"Evaluating policy locally from {policy_uri}")
        return evaluate_policy(
            checkpoint_uri=policy_uri,
            simulations=simulations,
            device=self._device,
            vectorization=self._system_cfg.vectorization,
            replay_dir=self._config.replay_dir,
            stats_epoch_id=stats_epoch_id,
            stats_client=self._stats_client,
        )

    def get_latest_scores(self) -> EvalRewardSummary:
        return self._latest_scores

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

        stats_reporter = self.context.stats_reporter
        if not stats_reporter:
            logger.warning("Evaluator: skipping epoch %s because stats_reporter is not available", epoch)
            return

        if not hasattr(stats_reporter, "state") or stats_reporter.state is None:
            logger.warning("Evaluator: skipping epoch %s because stats_reporter.state is not available", epoch)
            return

        stats_run_id = getattr(stats_reporter.state, "stats_run_id", None)
        if not stats_run_id:
            logger.warning("Evaluator: skipping epoch %s because stats_run_id is not available", epoch)
            return

        stats_epoch_id = stats_reporter.create_epoch(
            stats_run_id,  # Now the type checker knows this is not None
            epoch,  # Technically this is wrong, but we're not actually using this field
            epoch,
            attributes={"source": "evaluation", "agent_step": self.context.agent_step},
        )

        optimizer = getattr(self.context, "optimizer", None)
        is_schedulefree = optimizer is not None and is_schedulefree_optimizer(optimizer)
        if is_schedulefree:
            optimizer.eval()

        scores = self.evaluate(
            policy_uri=policy_uri,
            curriculum=curriculum,
            epoch=epoch,
            agent_step=self.context.agent_step,
            stats_epoch_id=stats_epoch_id,
        )

        # Restore train mode after evaluation for ScheduleFree optimizers
        if is_schedulefree:
            optimizer.train()

        stats_reporter = getattr(self.context, "stats_reporter", None)
        if stats_reporter:
            stats_reporter.update_eval_scores(scores)
