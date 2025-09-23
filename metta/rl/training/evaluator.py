"""Policy evaluation management."""

import logging
from typing import Any, List, Optional
from uuid import UUID

import torch
from pydantic import Field

import gitta as git
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.util.git_repo import REPO_SLUG
from metta.eval.eval_request_config import EvalResults, EvalRewardSummary
from metta.eval.eval_service import evaluate_policy
from metta.rl.evaluate import (
    evaluate_policy_remote_with_checkpoint_manager,
    upload_replay_html,
)
from metta.rl.training import TrainerComponent
from metta.rl.training.checkpointer import CheckpointConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_replay_dir
from mettagrid.config import Config

logger = logging.getLogger(__name__)


class EvaluatorConfig(Config):
    """Configuration for evaluation."""

    epoch_interval: int = 100  # 0 to disable
    evaluate_local: bool = True
    evaluate_remote: bool = False
    num_training_tasks: int = 2
    simulations: List[SimulationConfig] = Field(default_factory=list)
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
        checkpoint_cfg: CheckpointConfig,
        stats_client: Optional[StatsClient] = None,
    ):
        """Initialize evaluator.

        Args:
            config: Evaluation configuration
            device: Device to evaluate on
            system_cfg: System configuration
            stats_client: Optional stats client
        """
        super().__init__()
        self._master_only = True
        self._config = config
        self._device = device
        self._system_cfg = system_cfg
        self._stats_client = stats_client
        self._latest_scores = EvalRewardSummary()

        self._configure_evaluation_settings(
            eval_cfg=self._config,
            checkpoint_cfg=checkpoint_cfg,
            stats_client=self._stats_client,
        )

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        self.context.latest_eval_scores = self._latest_scores

    @staticmethod
    def _configure_evaluation_settings(
        *,
        eval_cfg: EvaluatorConfig,
        checkpoint_cfg: CheckpointConfig,
        stats_client: Optional[StatsClient],
    ) -> None:
        """Configure evaluation settings.

        Args:
            eval_cfg: Evaluation configuration from TrainTool
            checkpoint_cfg: Trainer checkpoint configuration
            stats_client: Optional stats client
        """
        if eval_cfg.epoch_interval and eval_cfg.epoch_interval < checkpoint_cfg.checkpoint_interval:
            raise ValueError(
                "evaluator.epoch_interval must be >= trainer.checkpoint.checkpoint_interval "
                "to ensure policies are saved before evaluation"
            )

        if eval_cfg.evaluate_remote and not checkpoint_cfg.remote_prefix:
            eval_cfg.evaluate_remote = False
            logger.info("Remote prefix unset; disabling remote evaluations")

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
                eval_cfg.git_hash = git.get_git_hash_for_remote_task(
                    target_repo=REPO_SLUG,
                    skip_git_check=eval_cfg.skip_git_check,
                    skip_cmd="evaluator.skip_git_check=true",
                )
                if eval_cfg.git_hash:
                    logger.info(f"Git hash for remote evaluations: {eval_cfg.git_hash}")
                else:
                    logger.info("No git hash available for remote evaluations")

    def should_evaluate(self, epoch: int) -> bool:
        """Check if evaluation should run at this epoch.

        Args:
            epoch: Current epoch

        Returns:
            True if evaluation should run
        """
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
        """Run evaluation on the policy.

        Args:
            policy_uri: URI of the policy checkpoint to evaluate
            curriculum: Training curriculum for getting tasks
            epoch: Current epoch
            agent_step: Current agent step
            stats_epoch_id: Optional stats epoch ID

        Returns:
            Evaluation scores
        """
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
                        metric_prefix="training_eval",
                        step_metric_key="metric/epoch",
                        epoch_metric_key="metric/epoch",
                    )

            self._latest_scores = evaluation_results.scores
            self.context.latest_eval_scores = self._latest_scores
            return evaluation_results.scores

        return EvalRewardSummary()

    def _build_simulations(self, curriculum: Any) -> List[SimulationConfig]:
        """Build simulation configurations for evaluation.

        Args:
            curriculum: Training curriculum

        Returns:
            List of simulation configurations
        """
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
        simulations: List[SimulationConfig],
        stats_epoch_id: Optional[UUID] = None,
    ) -> None:
        """Run remote evaluation.

        Args:
            policy_uri: URI of policy to evaluate
            simulations: Simulations to run
            stats_epoch_id: Optional stats epoch ID
        """
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
        simulations: List[SimulationConfig],
        stats_epoch_id: Optional[UUID] = None,
    ) -> EvalResults:
        """Run local evaluation.

        Args:
            policy_uri: URI of policy to evaluate
            simulations: Simulations to run
            stats_epoch_id: Optional stats epoch ID

        Returns:
            Evaluation results
        """
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
        """Get the latest evaluation scores.

        Returns:
            Latest evaluation scores
        """
        return self._latest_scores

    def on_epoch_end(self, epoch: int) -> None:  # type: ignore[override]
        """Run evaluation at epoch end if due."""
        if not self.should_evaluate(epoch):
            return

        policy_uri = self.context.latest_policy_uri()

        if not policy_uri:
            logger.debug("Evaluator: skipping epoch %s because no policy checkpoint is available", epoch)
            return

        curriculum = getattr(self.context.env, "_curriculum", None)
        if curriculum is None:
            logger.debug("Evaluator: curriculum unavailable; skipping evaluation")
            return

        stats_reporter = self.context.stats_reporter
        stats_epoch_id = None
        if stats_reporter and getattr(stats_reporter.state, "stats_run_id", None):
            stats_epoch_id = stats_reporter.create_epoch(
                stats_reporter.state.stats_run_id,
                epoch,
                epoch,
                attributes={"source": "evaluation"},
            )

        scores = self.evaluate(
            policy_uri=policy_uri,
            curriculum=curriculum,
            epoch=epoch,
            agent_step=self.context.agent_step,
            stats_epoch_id=stats_epoch_id,
        )

        stats_reporter = getattr(self.context, "stats_reporter", None)
        if stats_reporter:
            stats_reporter.update_eval_scores(scores)
