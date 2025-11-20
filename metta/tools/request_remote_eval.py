import logging
import uuid
from typing import Sequence

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.tool.tool import Tool
from metta.common.util.git_helpers import get_task_commit_hash
from metta.sim.remote import evaluate_remotely
from metta.sim.runner import SimulationRunConfig
from metta.tools.utils.auto_config import auto_stats_server_uri

logger = logging.getLogger(__name__)


# Used to evaluate a policy on a remote simulation suite
class RequestRemoteEvalTool(Tool):
    stats_server_uri: str | None = auto_stats_server_uri()

    policy_version_id: str
    simulations: Sequence[SimulationRunConfig]

    def invoke(self, args: dict[str, str]) -> int | None:
        if self.stats_server_uri is None:
            raise ValueError("stats_server_uri is required")

        stats_client = StatsClient.create(self.stats_server_uri)
        git_hash = get_task_commit_hash(skip_git_check=True)

        task = evaluate_remotely(uuid.UUID(self.policy_version_id), self.simulations, stats_client, git_hash)

        logger.info(f"Policy evaluator: created task {task}")

        return 0
