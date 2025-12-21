import logging
from typing import Sequence

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.tool.tool import Tool
from metta.sim.remote import evaluate_remotely
from metta.sim.runner import SimulationRunConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.utils.auto_config import auto_stats_server_uri

logger = logging.getLogger(__name__)


# Used to evaluate a policy on a remote simulation suite
class RequestRemoteEvalTool(Tool):
    policy_version_id: str | None = None
    policy_uri: str | None = None
    simulations: Sequence[SimulationRunConfig] | Sequence[SimulationConfig]
    stats_server_uri: str | None = auto_stats_server_uri()
    push_metrics_to_wandb: bool = False
    git_hash: str | None = None

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

        task = evaluate_remotely(
            simulations=self._to_simulation_run_configs(),
            stats_client=stats_client,
            policy_version_id=self.policy_version_id,
            policy_uri=self.policy_uri,
            git_hash=self.git_hash,
            push_metrics_to_wandb=self.push_metrics_to_wandb,
        )

        logger.info(f"Policy evaluator: created task {task}")

        return 0
