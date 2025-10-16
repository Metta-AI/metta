import logging
from typing import Sequence

from metta.app_backend.clients.stats_client import HttpStatsClient
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest
from metta.common.tool.tool import Tool
from metta.common.util.constants import PROD_STATS_SERVER_URI
from metta.common.util.git_remote import get_git_hash_for_remote_task
from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.simulation_config import SimulationConfig
from metta.sim.utils import get_or_create_policy_ids

logger = logging.getLogger(__name__)


class EvalRemoteTool(Tool):
    stats_server_uri: str = PROD_STATS_SERVER_URI

    policy_uri: str
    simulations: Sequence[SimulationConfig]

    def invoke(self, args: dict[str, str]) -> int | None:
        stats_client = HttpStatsClient.create(self.stats_server_uri)
        normalized_uri = CheckpointManager.normalize_uri(self.policy_uri)
        policy_id = get_or_create_policy_ids(stats_client, [(normalized_uri, "")])[normalized_uri]

        git_hash = get_git_hash_for_remote_task(skip_git_check=True)

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
