import logging
from typing import Sequence

from metta.app_backend.clients.stats_client import HttpStatsClient
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest
from metta.common.tool.tool import Tool
from metta.common.util.git_helpers import get_task_commit_hash
from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.simulation_config import SimulationConfig
from metta.sim.utils import get_or_create_policy_ids
from metta.tools.utils.auto_config import auto_stats_server_uri

logger = logging.getLogger(__name__)


# Used to evaluate a policy on a remote simulation suite
class RequestRemoteEvalTool(Tool):
    stats_server_uri: str | None = auto_stats_server_uri()

    policy_uri: str
    simulations: Sequence[SimulationConfig]

    def invoke(self, args: dict[str, str]) -> int | None:
        # Errors if stats_server_uri is not set or authentication fails
        stats_client = HttpStatsClient.create(self.stats_server_uri)
        normalized_uri = CheckpointManager.normalize_uri(self.policy_uri)
        policy_id = get_or_create_policy_ids(stats_client, [(normalized_uri, "")])[normalized_uri]

        git_hash = get_task_commit_hash(skip_git_check=True)

        task = stats_client.create_task(
            TaskCreateRequest(
                policy_id=policy_id,
                sim_suite=self.simulations[0].suite,
                attributes={
                    "git_hash": git_hash,
                    "simulations": [sim.model_dump() for sim in self.simulations],
                },
            )
        )

        logger.info(f"Policy evaluator: created task {task.id} for {normalized_uri} on {self.simulations[0].name}")

        return 0


# Used to evaluate a multi-policy on a remote simulation suite
class RequestMultiPolicyRemoteEvalTool(Tool):
    stats_server_uri: str | None = auto_stats_server_uri()

    policy_uri_proportions: dict[str, float]
    simulations: Sequence[SimulationConfig]

    def invoke(self, args: dict[str, str]) -> int | None:
        # Errors if stats_server_uri is not set or authentication fails
        stats_client = HttpStatsClient.create(self.stats_server_uri)
        normalized_uri_proportions = {
            CheckpointManager.normalize_uri(uri): proportion for uri, proportion in self.policy_uri_proportions.items()
        }
        policy_ids = get_or_create_policy_ids(stats_client, [(n, None) for n in normalized_uri_proportions.keys()])
        if not policy_ids:
            logger.error("Failed to create or find policy IDs")
            return 1
        first_policy_id = next(iter(policy_ids.values()))

        git_hash = get_task_commit_hash(skip_git_check=True)

        policy_proportions = {policy_uuid: normalized_uri_proportions[uri] for uri, policy_uuid in policy_ids.items()}

        task = stats_client.create_task(
            TaskCreateRequest(
                policy_id=first_policy_id,  # This is not important
                sim_suite=self.simulations[0].suite,
                attributes={
                    "git_hash": git_hash,
                    "simulations": [sim.model_dump() for sim in self.simulations],
                    "policy_id_proportions": {str(k): v for k, v in policy_proportions.items()},
                },
            )
        )

        logger.info(f"Policy evaluator: created task {task.id} for {policy_proportions} on {self.simulations[0].name}")

        return 0
