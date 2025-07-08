from __future__ import annotations

import logging
from pathlib import Path

import torch
from pydantic import BaseModel, Field, field_serializer

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.app_backend.stats_client import StatsClient
from metta.eval.eval_stats_db import EvalStatsDB
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_stats_db import SimulationStatsDB
from metta.sim.simulation_suite import SimulationSuite


class EvaluationScores(BaseModel):
    overall_score: float | None = Field(default=None, description="Overall average score across all suites")
    suite_scores: dict[str, float] = Field(default_factory=dict, description="Scores by simulation suite")
    simulation_scores: dict[str, float] = Field(
        default_factory=dict, description="Individual simulation scores (key format: suite_name/sim_name)"
    )

    class Config:
        extra = "forbid"


class EvaluationResults(BaseModel):
    policy_record: PolicyRecord = Field(..., description="Policy record")
    scores: EvaluationScores = Field(..., description="Evaluation scores")
    stats_db_path: str = Field(..., description="Path to the stats database")
    replay_dir: str = Field(..., description="Directory containing replay files")
    stats_db: SimulationStatsDB = Field(..., description="The stats database object", exclude=True)

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    @field_serializer("policy_record")
    def serialize_policy_record(self, value: PolicyRecord):
        return {
            "run_name": value.run_name,
            "uri": value.uri,
        }


class SavedEvaluationResults(BaseModel):
    """Model for serializable evaluation results (without stats_db object)."""

    policy_name: str
    policy_uri: str
    scores: EvaluationScores
    stats_db_path: str
    replay_dir: str

    class Config:
        extra = "forbid"


class EvaluationService:
    def __init__(
        self,
        policy_store: PolicyStore,
        device: str = "cpu",
        vectorization: str = "multiprocessing",
        stats_client: StatsClient | None = None,
        logger: logging.Logger | None = None,
    ):
        self.policy_store = policy_store
        self.device = device
        self.vectorization = vectorization
        self.stats_client = stats_client
        self.logger = logger or logging.getLogger(__name__)

    def run_evaluation(
        self,
        policy_pr: PolicyRecord,
        sim_config: SimulationSuiteConfig,
        stats_dir: str | None = None,
        replay_dir: str | None = None,
        wandb_policy_name: str | None = None,
    ) -> EvaluationResults:
        """
        Run evaluation for a single policy.

        Args:
            policy_pr: Policy record to evaluate
            sim_config: Simulation suite configuration
            stats_dir: Directory to store stats database
            replay_dir: Directory to store replay files
            wandb_policy_name: Wandb policy name (e.g. 'entity/project/artifact:version')

        Returns:
            Dictionary containing evaluation results and scores
        """
        # Set up default paths if not provided
        if not stats_dir:
            stats_dir = f"./eval_results/stats/{policy_pr.run_name}"
        if not replay_dir:
            replay_dir = f"./eval_results/replays/{policy_pr.run_name}"

        # Ensure directories exist
        Path(stats_dir).mkdir(parents=True, exist_ok=True)
        Path(replay_dir).mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Evaluating policy: {policy_pr.run_name}")

        # Create and run simulation suite
        # Note: stats_epoch_id is None for standalone evaluation
        # Policies will be created with null epoch_id and episodes with null stats_epoch
        sim_suite = SimulationSuite(
            config=sim_config,
            policy_pr=policy_pr,
            policy_store=self.policy_store,
            device=torch.device(self.device),
            vectorization=self.vectorization,
            stats_dir=stats_dir,
            stats_client=self.stats_client,
            replay_dir=replay_dir,
            stats_epoch_id=None,
            wandb_policy_name=wandb_policy_name,
        )

        # Run simulations. This will:
        # 1. Create local DuckDB with episode data
        # 2. If stats_client is configured, send episode data to remote stats server
        #    via Simulation._write_remote_stats()
        sim_results = sim_suite.simulate()

        # Extract scores
        eval_stats_db = EvalStatsDB.from_sim_stats_db(sim_results.stats_db)
        scores = self.extract_scores(eval_stats_db, policy_pr)

        # Export stats database
        stats_db_path = f"{stats_dir}/stats.db"
        sim_results.stats_db.export(stats_db_path)

        return EvaluationResults(
            policy_record=policy_pr,
            scores=scores,
            stats_db_path=stats_db_path,
            replay_dir=replay_dir,
            stats_db=sim_results.stats_db,
        )

    def extract_scores(self, stats_db: EvalStatsDB, policy_pr: PolicyRecord) -> EvaluationScores:
        suite_scores = {}
        simulation_scores = {}
        overall_score = None

        # Check policy URI
        if policy_pr.uri is None:
            self.logger.warning("Policy URI is None, cannot extract scores")
            return EvaluationScores()

        # Get all simulations for this policy
        sims_df = stats_db.query(f"""
            SELECT DISTINCT sim_suite, sim_name
            FROM policy_simulation_agent_samples
            WHERE policy_key = '{policy_pr.uri}' AND policy_version = {policy_pr.metadata.epoch}
        """)

        if len(sims_df) == 0:
            self.logger.warning("No simulations found in database")
            return EvaluationScores()

        # Group by simulation suite (category) - matching trainer's approach
        suites = sims_df["sim_suite"].unique()

        for suite in suites:
            # Use get_average_metric_by_filter like the trainer does
            suite_score = stats_db.get_average_metric_by_filter("reward", policy_pr, f"sim_suite = '{suite}'")

            if suite_score is not None:
                suite_scores[suite] = suite_score

                # Get individual simulation scores within this suite
                suite_sims = sims_df[sims_df["sim_suite"] == suite]["sim_name"].unique()

                for sim_name in suite_sims:
                    # Use get_average_metric_by_filter for each simulation
                    sim_score = stats_db.get_average_metric_by_filter(
                        "reward", policy_pr, f"sim_suite = '{suite}' AND sim_name = '{sim_name}'"
                    )

                    if sim_score is not None:
                        simulation_scores[f"{suite}/{sim_name}"] = sim_score

        # Calculate overall score as average of suite scores (matching trainer)
        suite_score_values = list(suite_scores.values())
        if suite_score_values:
            overall_score = sum(suite_score_values) / len(suite_score_values)

        return EvaluationScores(
            overall_score=overall_score, suite_scores=suite_scores, simulation_scores=simulation_scores
        )
