"""Policy evaluation management."""

import logging
from typing import TYPE_CHECKING, Any, List, Optional
from uuid import UUID

import torch
from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.eval.eval_request_config import EvalResults, EvalRewardSummary
from metta.eval.eval_service import evaluate_policy
from metta.rl.evaluate import (
    evaluate_policy_remote_with_checkpoint_manager,
    upload_replay_html,
)
from metta.rl.training.components import ComponentConfig, MasterComponent
from metta.sim.simulation_config import SimulationConfig

if TYPE_CHECKING:
    from metta.rl.trainer_v2 import Trainer
    from metta.rl.training.stats_reporter import StatsReporter

logger = logging.getLogger(__name__)


class EvaluationConfig(ComponentConfig):
    """Configuration for evaluation."""

    interval: int = 100  # Use standard component interval
    evaluate_local: bool = True
    evaluate_remote: bool = False
    num_training_tasks: int = 2
    simulations: List[SimulationConfig] = Field(default_factory=list)
    replay_dir: Optional[str] = None


class NoOpEvaluator(MasterComponent):
    """No-op evaluator for when evaluation is disabled."""

    def __init__(self):
        """Initialize no-op evaluator."""
        # Create a minimal config for the no-op evaluator
        config = EvaluationConfig(
            evaluate_local=False, evaluate_remote=False, interval=999999, num_training_tasks=0, simulations=[]
        )
        super().__init__(config)
        self._latest_scores = EvalRewardSummary()

    def get_latest_scores(self) -> EvalRewardSummary:
        return self._latest_scores

    def on_epoch_end(self, trainer: "Trainer") -> None:
        pass


class Evaluator(MasterComponent):
    """Manages policy evaluation."""

    @classmethod
    def from_config(
        cls,
        config: Optional[EvaluationConfig],
        device: Optional[torch.device] = None,
        system_cfg: Optional[Any] = None,
        trainer_cfg: Optional[Any] = None,
        stats_client: Optional[StatsClient] = None,
        stats_reporter: Optional["StatsReporter"] = None,
    ) -> "Evaluator":
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

    def __init__(
        self,
        config: EvaluationConfig,
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
        super().__init__(config)
        self._config = config
        self._device = device
        self._system_cfg = system_cfg
        self._trainer_cfg = trainer_cfg
        self._stats_client = stats_client
        self._stats_reporter = stats_reporter
        self._latest_scores = EvalRewardSummary()

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
        return epoch % self._config.evaluate_interval == 0

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

    def on_epoch_end(self, trainer: "Trainer") -> None:
        """Run evaluation at epoch end if due.

        Args:
            trainer: The trainer instance
        """
        # Run evaluation
        policy = trainer.policy_checkpointer.save_policy_to_buffer(trainer.policy)

        epoch = trainer.trainer_state.epoch
        scores = self.evaluate(
            policy=policy,
            epoch=epoch,
            curriculum_tasks=trainer.curriculum.available_tasks,
        )

        # Update stats reporter if available
        if trainer.stats_reporter:
            trainer.stats_reporter.update_eval_scores(scores)
