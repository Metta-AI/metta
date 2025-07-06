import logging
import uuid
from typing import Callable, Optional, Set

import torch

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.app_backend.stats_client import StatsClient
from metta.eval.eval_stats_db import EvalStatsDB
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_suite import SimulationSuite

logger = logging.getLogger(__name__)


class PolicyEvaluator:
    def __init__(
        self,
        device: str,
        vectorization: str,
        stats_dir: str,
        sim_suite_config: SimulationSuiteConfig,
        policy_store: PolicyStore,
        stats_client: Optional[StatsClient] = None,
    ):
        self.sim_suite_config = sim_suite_config
        self.policy_store = policy_store
        self.device = torch.device(device)
        self.vectorization = vectorization
        self.evals: dict[str, float] = {}  # used for wandb
        self.stats_dir = stats_dir
        self.stats_client = stats_client
        self.stats_epoch_id: uuid.UUID | None = None

    def evaluate_policy(
        self,
        policy_record: PolicyRecord,
        stats_epoch_start: int,
        stats_epoch_end: int,
        stats_run_id: uuid.UUID,
        wandb_policy_name: Optional[str] = None,
        record_heartbeat: Callable[[], None] = lambda: None,
    ) -> None:
        if stats_run_id is not None and self.stats_client is not None:
            self.stats_epoch_id = self.stats_client.create_epoch(
                run_id=stats_run_id,
                start_training_epoch=stats_epoch_start,
                end_training_epoch=stats_epoch_end,
                attributes={},
            ).id

        logger.info(f"Evaluating policy: {policy_record.uri} with config: {self.sim_suite_config}")

        sim = SimulationSuite(
            config=self.sim_suite_config,
            policy_pr=policy_record,
            policy_store=self.policy_store,
            device=self.device,
            vectorization=self.vectorization,
            stats_dir=self.stats_dir,
            stats_client=self.stats_client,
            stats_epoch_id=self.stats_epoch_id,
            wandb_policy_name=wandb_policy_name,
        )

        result = sim.simulate()
        stats_db = EvalStatsDB.from_sim_stats_db(result.stats_db)
        logger.info("Simulation complete")

        self.evals: dict[str, float] = {}
        categories: Set[str] = set()

        for sim_name in self.sim_suite_config.simulations.keys():
            categories.add(sim_name.split("/")[0])

        for category in categories:
            score = stats_db.get_average_metric_by_filter("reward", policy_record, f"sim_name LIKE '%{category}%'")
            logger.info(f"{category} score: {score}")
            record_heartbeat()

            if score is not None:
                self.evals[f"{category}/score"] = score

        all_scores = stats_db.simulation_scores(policy_record, "reward")
        for (_, sim_name, _), score in all_scores.items():
            category = sim_name.split("/")[0]
            sim_short_name = sim_name.split("/")[-1]
            self.evals[f"{category}/{sim_short_name}"] = score

        result.stats_db.close()

    def get_category_scores(self) -> dict[str, float]:
        return {key.split("/")[0]: value for key, value in self.evals.items() if key.endswith("/score")}

    def calculate_overall_score(self, category_scores: dict[str, float]) -> float:
        if not category_scores:
            return 0.0
        return sum(category_scores.values()) / len(category_scores)
