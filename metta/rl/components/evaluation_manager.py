"""Manages policy evaluation and replay generation."""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.eval.eval_request_config import EvalRewardSummary
from metta.eval.eval_stats_db import EvalStatsDB
from metta.interface.evaluation import create_evaluation_config_suite
from metta.rl.trainer_config import TrainerConfig
from metta.rl.util.evaluation import generate_replay
from metta.rl.wandb import upload_replay_html
from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig
from metta.sim.simulation_suite import SimulationSuite

logger = logging.getLogger(__name__)


class EvaluationManager:
    """Manages policy evaluation and replay generation during training."""

    def __init__(
        self,
        trainer_config: TrainerConfig,
        policy_store: PolicyStore,
        device: torch.device,
        stats_dir: str,
        is_master: bool = True,
    ):
        """Initialize evaluation manager.

        Args:
            trainer_config: Training configuration
            policy_store: Policy store for saving/loading policies
            device: Device to run computations on
            stats_dir: Directory for storing evaluation statistics
            is_master: Whether this is the master process
        """
        self.trainer_config = trainer_config
        self.policy_store = policy_store
        self.device = device
        self.stats_dir = stats_dir
        self.is_master = is_master

        # Create evaluation configuration
        self.evaluation_config = create_evaluation_config_suite()

        # Track last evaluation epoch
        self.last_evaluation_epoch = -1

    def should_evaluate(self, epoch: int) -> bool:
        """Check if evaluation should run at this epoch.

        Args:
            epoch: Current training epoch

        Returns:
            True if evaluation should run
        """
        if not self.is_master:
            return False

        if self.trainer_config.simulation.evaluate_interval <= 0:
            return False

        return epoch % self.trainer_config.simulation.evaluate_interval == 0

    def evaluate_policy(
        self,
        policy_record: PolicyRecord,
        epoch: int,
        curriculum: Optional[Any] = None,
        wandb_run: Optional[Any] = None,
    ) -> EvalRewardSummary:
        """Evaluate a policy and return scores.

        Args:
            policy_record: Policy to evaluate
            epoch: Current training epoch
            curriculum: Optional curriculum object for adding training task
            wandb_run: Optional wandb run for logging

        Returns:
            EvalRewardSummary with evaluation scores
        """
        if not self.is_master:
            return EvalRewardSummary()

        logger.info(f"Evaluating policy at epoch {epoch}")

        # Create extended evaluation config with training task
        extended_eval_config = SimulationSuiteConfig(
            name=self.evaluation_config.name,
            simulations=dict(self.evaluation_config.simulations),
            env_overrides=self.evaluation_config.env_overrides,
            num_episodes=self.evaluation_config.num_episodes,
        )

        # Add training task to the suite if curriculum is provided
        if curriculum:
            training_task_config = SingleEnvSimulationConfig(
                env="/env/mettagrid/mettagrid",
                num_episodes=1,
                env_overrides=curriculum.get_task().env_cfg(),
            )
            extended_eval_config.simulations["eval/training_task"] = training_task_config

        # Run evaluation suite
        sim_suite = SimulationSuite(
            config=extended_eval_config,
            policy_pr=policy_record,
            policy_store=self.policy_store,
            device=self.device,
            vectorization="serial",
            stats_dir=self.stats_dir,
            stats_client=None,
            stats_epoch_id=None,
            wandb_policy_name=None,
        )

        results = sim_suite.simulate()
        stats_db = EvalStatsDB.from_sim_stats_db(results.stats_db)
        logger.info("Evaluation complete")

        # Build evaluation metrics
        category_scores: Dict[str, float] = {}
        categories = set()
        for sim_name in extended_eval_config.simulations.keys():
            categories.add(sim_name.split("/")[0])

        for category in categories:
            score = stats_db.get_average_metric_by_filter("reward", policy_record, f"sim_name LIKE '%{category}%'")
            logger.info(f"{category} score: {score}")
            if score is not None:
                category_scores[category] = score

        # Get detailed per-simulation scores
        per_sim_scores: Dict[Tuple[str, str], float] = {}
        all_scores = stats_db.simulation_scores(policy_record, "reward")
        for (_, sim_name, _), score in all_scores.items():
            category = sim_name.split("/")[0]
            sim_short_name = sim_name.split("/")[-1]
            per_sim_scores[(category, sim_short_name)] = score

        eval_scores = EvalRewardSummary(
            category_scores=category_scores,
            simulation_scores=per_sim_scores,
        )

        # Set policy metadata score
        category_score_values = list(category_scores.values())
        if category_score_values:
            policy_record.metadata["score"] = float(np.mean(category_score_values))
            logger.info(f"Set policy metadata score to {policy_record.metadata['score']}")

        stats_db.close()

        # Track that we evaluated at this epoch
        self.last_evaluation_epoch = epoch

        # Upload replay HTML if available
        if wandb_run and hasattr(results, "replay_urls") and results.replay_urls:
            upload_replay_html(
                replay_urls=results.replay_urls,
                agent_step=0,  # Will be provided by caller
                epoch=epoch,
                wandb_run=wandb_run,
            )

        return eval_scores

    def generate_replay(
        self,
        policy_record: PolicyRecord,
        epoch: int,
        curriculum: Any,
        wandb_run: Optional[Any] = None,
    ) -> None:
        """Generate replay for a policy.

        Args:
            policy_record: Policy to generate replay for
            epoch: Current training epoch
            curriculum: Curriculum object
            wandb_run: Optional wandb run for logging
        """
        if not self.is_master:
            return

        logger.info(f"Generating replay at epoch {epoch}")

        generate_replay(
            policy_record=policy_record,
            policy_store=self.policy_store,
            curriculum=curriculum,
            epoch=epoch,
            device=self.device,
            vectorization="serial",
            replay_dir=self.trainer_config.simulation.replay_dir,
            wandb_run=wandb_run,
        )

    def final_evaluation_needed(self, epoch: int) -> bool:
        """Check if final evaluation is needed.

        Args:
            epoch: Current/final training epoch

        Returns:
            True if final evaluation should be performed
        """
        return self.is_master and self.last_evaluation_epoch < epoch
