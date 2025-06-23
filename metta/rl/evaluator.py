"""Policy evaluation components."""

import logging
from typing import TYPE_CHECKING, Dict, Optional
from uuid import UUID

import torch

from metta.eval.eval_stats_db import EvalStatsDB
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_suite import SimulationSuite

if TYPE_CHECKING:
    from app_backend.stats_client import StatsClient
    from metta.agent.policy_store import PolicyRecord, PolicyStore

logger = logging.getLogger(__name__)


class PolicyEvaluator:
    """Evaluates policies on a suite of simulation tasks.

    This class handles running simulations to evaluate policy performance
    and collecting statistics.
    """

    def __init__(
        self,
        sim_suite_config: SimulationSuiteConfig,
        policy_store: "PolicyStore",
        device: torch.device,
        vectorization: str = "serial",
        stats_client: Optional["StatsClient"] = None,
    ):
        """Initialize the evaluator.

        Args:
            sim_suite_config: Configuration for simulation suite
            policy_store: Store for loading policies
            device: Device to run evaluations on
            vectorization: Vectorization strategy for simulations
            stats_client: Optional client for reporting stats
        """
        self.sim_suite_config = sim_suite_config
        self.policy_store = policy_store
        self.device = device
        self.vectorization = vectorization
        self.stats_client = stats_client

    def evaluate(
        self,
        policy_record: "PolicyRecord",
        stats_run_id: Optional[UUID] = None,
        stats_epoch_start: int = 0,
        stats_epoch_end: int = 0,
    ) -> Dict[str, float]:
        """Evaluate a policy on the simulation suite.

        Args:
            policy_record: Policy to evaluate
            stats_run_id: Optional run ID for stats tracking
            stats_epoch_start: Starting epoch for stats
            stats_epoch_end: Ending epoch for stats

        Returns:
            Dictionary of evaluation metrics
        """
        # Create stats epoch if needed
        stats_epoch_id = None
        if stats_run_id is not None and self.stats_client is not None:
            stats_epoch_id = self.stats_client.create_epoch(
                run_id=stats_run_id,
                start_training_epoch=stats_epoch_start,
                end_training_epoch=stats_epoch_end,
                attributes={},
            ).id

        # Run simulation suite
        logger.info(f"Evaluating policy: {policy_record.uri} with config: {self.sim_suite_config}")
        sim = SimulationSuite(
            config=self.sim_suite_config,
            policy_pr=policy_record,
            policy_store=self.policy_store,
            device=self.device,
            vectorization=self.vectorization,
            stats_dir="/tmp/stats",
            stats_client=self.stats_client,
            stats_epoch_id=stats_epoch_id,
        )
        result = sim.simulate()
        stats_db = EvalStatsDB.from_sim_stats_db(result.stats_db)
        logger.info("Evaluation complete")

        # Build evaluation metrics
        evals = {}

        # Get categories from simulation names
        categories = set()
        for sim_name in self.sim_suite_config.simulations.keys():
            categories.add(sim_name.split("/")[0])

        # Compute category scores
        for category in categories:
            score = stats_db.get_average_metric_by_filter("reward", policy_record, f"sim_name LIKE '%{category}%'")
            if score is not None:
                evals[f"{category}/score"] = score
                logger.info(f"{category} score: {score}")

        # Get detailed per-simulation scores
        all_scores = stats_db.simulation_scores(policy_record, "reward")
        for (_, sim_name, _), score in all_scores.items():
            category = sim_name.split("/")[0]
            sim_short_name = sim_name.split("/")[-1]
            evals[f"{category}/{sim_short_name}"] = score

        return evals


class AsyncPolicyEvaluator(PolicyEvaluator):
    """Asynchronous policy evaluator for non-blocking evaluation.

    This extends PolicyEvaluator to support running evaluations
    in the background while training continues.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._evaluation_thread = None
        self._latest_results = None

    def evaluate_async(
        self,
        policy_record: "PolicyRecord",
        callback=None,
        **kwargs,
    ):
        """Start asynchronous evaluation.

        Args:
            policy_record: Policy to evaluate
            callback: Optional callback function for results
            **kwargs: Additional arguments for evaluate()
        """
        import threading

        def _run_evaluation():
            results = self.evaluate(policy_record, **kwargs)
            self._latest_results = results
            if callback:
                callback(results)

        self._evaluation_thread = threading.Thread(target=_run_evaluation)
        self._evaluation_thread.start()

    def get_latest_results(self) -> Optional[Dict[str, float]]:
        """Get the latest evaluation results if available."""
        return self._latest_results

    def wait_for_completion(self) -> Optional[Dict[str, float]]:
        """Wait for ongoing evaluation to complete and return results."""
        if self._evaluation_thread is not None:
            self._evaluation_thread.join()
        return self._latest_results
