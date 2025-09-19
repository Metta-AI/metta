"""Policy evaluation management."""

import logging
from typing import TYPE_CHECKING, Any, List, Optional
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
from metta.rl.training.component import TrainerComponent
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_replay_dir
from mettagrid.config import Config

if TYPE_CHECKING:
    from metta.rl.training.stats_reporter import StatsReporter

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
        trainer_cfg: Any,
        stats_client: Optional[StatsClient] = None,
        stats_reporter: Optional["StatsReporter"] = None,
    ):
        """Initialize evaluator.

        Args:
            config: Evaluation configuration
            device: Device to evaluate on
            system_cfg: System configuration
            trainer_cfg: Trainer configuration
            stats_client: Optional stats client
            stats_reporter: Optional stats reporter for wandb/metrics
        """
        super().__init__()
        self._master_only = True
        self._config = config
        self._device = device
        self._system_cfg = system_cfg
        self._trainer_cfg = trainer_cfg
        self._stats_client = stats_client
        self._stats_reporter = stats_reporter
        self._latest_scores = EvalRewardSummary()
        self._stats_reporter = stats_reporter

    def register(self, context) -> None:  # type: ignore[override]
        super().register(context)
        self.context.latest_eval_scores = self._latest_scores

    @classmethod
    def from_config(
        cls,
        config: Optional[EvaluatorConfig],
        device: Optional[torch.device] = None,
        system_cfg: Optional[Any] = None,
        trainer_cfg: Optional[Any] = None,
        stats_client: Optional[StatsClient] = None,
        stats_reporter: Optional["StatsReporter"] = None,
    ):
        """Create an Evaluator from optional config, returning no-op if None.

        Args:
            config: Optional evaluation configuration
            device: Optional torch device
            system_cfg: Optional system configuration
            trainer_cfg: Optional trainer configuration
            stats_client: Optional stats client
            stats_reporter: Optional stats reporter

        Returns:
            Evaluator instance (no-op if config is None)
        """
        if config is None:
            return NoOpEvaluator()

        # Configure evaluation settings
        if trainer_cfg and trainer_cfg.evaluation:
            cls._configure_evaluation_settings(trainer_cfg.evaluation, stats_client)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(
            config=config,
            device=device,
            system_cfg=system_cfg,
            trainer_cfg=trainer_cfg,
            stats_client=stats_client,
            stats_reporter=stats_reporter,
        )

    @staticmethod
    def _configure_evaluation_settings(eval_cfg: Any, stats_client: Optional[StatsClient]) -> None:
        """Configure evaluation settings.

        Args:
            eval_cfg: Evaluation configuration from trainer config
            stats_client: Optional stats client
        """
        if eval_cfg.replay_dir is None:
            eval_cfg.replay_dir = auto_replay_dir()
            logger.info(f"Setting replay_dir to {eval_cfg.replay_dir}")

        # Determine git hash for remote simulations
        if eval_cfg.evaluate_remote:
            if not stats_client:
                eval_cfg.evaluate_remote = False
                logger.info("Not connected to stats server, disabling remote evaluations")
            elif not eval_cfg.evaluate_interval:
                eval_cfg.evaluate_remote = False
                logger.info("Evaluate interval set to 0, disabling remote evaluations")
            elif not eval_cfg.git_hash:
                eval_cfg.git_hash = git.get_git_hash_for_remote_task(
                    target_repo=REPO_SLUG,
                    skip_git_check=eval_cfg.skip_git_check,
                    skip_cmd="trainer.evaluation.skip_git_check=true",
                )
                if eval_cfg.git_hash:
                    logger.info(f"Git hash for remote evaluations: {eval_cfg.git_hash}")
                else:
                    logger.info("No git hash available for remote evaluations")

    @property
    def stats_reporter(self):
        """Get the stats reporter."""
        return self._stats_reporter

    @stats_reporter.setter
    def stats_reporter(self, value):
        """Set the stats reporter."""
        self._stats_reporter = value

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
            if self._stats_reporter and evaluation_results.replay_urls:
                wandb_run = getattr(self._stats_reporter, "wandb_run", None)
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
        wandb_run = getattr(self._stats_reporter, "wandb_run", None) if self._stats_reporter else None
        evaluate_policy_remote_with_checkpoint_manager(
            policy_uri=policy_uri,
            simulations=simulations,
            stats_epoch_id=stats_epoch_id,
            stats_client=self._stats_client,
            wandb_run=wandb_run,
            trainer_cfg=self._trainer_cfg,
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
                stats_reporter.state.stats_epoch_start,
                epoch,
            )
            stats_reporter.update_epoch_tracking(epoch + 1)

        scores = self.evaluate(
            policy_uri=policy_uri,
            curriculum=curriculum,
            epoch=epoch,
            agent_step=self.context.agent_step,
            stats_epoch_id=stats_epoch_id,
        )

        if self._stats_reporter:
            self._stats_reporter.update_eval_scores(scores)
