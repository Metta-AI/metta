import logging
from typing import Sequence

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.tool.tool import Tool
from metta.common.util.git_helpers import get_task_commit_hash
from metta.sim.remote import evaluate_remotely
from metta.sim.runner import SimulationRunConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_stats_server_uri

logger = logging.getLogger(__name__)


# Used to evaluate a policy on a remote simulation suite
class RequestRemoteEvalTool(Tool):
    policy_uri: str
    simulations: Sequence[SimulationRunConfig] | Sequence[SimulationConfig]
    stats_server_uri: str | None = auto_stats_server_uri()
    push_metrics_to_wandb: bool = False

    def _to_simulation_run_configs(self) -> list[SimulationRunConfig]:
        result = []
        for sim in self.simulations:
            if isinstance(sim, SimulationConfig):
                result.append(sim.to_simulation_run_config())
            else:
                result.append(sim)
        return result

    def invoke(self, args: dict[str, str]) -> int | None:
        if self.stats_server_uri is None:
            raise ValueError("stats_server_uri is required")

        stats_client = StatsClient.create(self.stats_server_uri)
        git_hash = get_task_commit_hash(skip_git_check=True)

        task = evaluate_remotely(
            self._to_simulation_run_configs(),
            stats_client,
            policy_uri=self.policy_uri,
            git_hash=git_hash,
            push_metrics_to_wandb=self.push_metrics_to_wandb,
        )

        logger.info(f"Policy evaluator: created task {task}")

        return 0
