import logging
import typing

import metta.app_backend.clients.stats_client
import metta.app_backend.routes.eval_task_routes
import metta.common.tool.tool
import metta.common.util.constants
import metta.common.util.git_helpers
import metta.rl.checkpoint_manager
import metta.sim.simulation_config
import metta.sim.utils

logger = logging.getLogger(__name__)


class EvalRemoteTool(metta.common.tool.tool.Tool):
    stats_server_uri: str = metta.common.util.constants.PROD_STATS_SERVER_URI

    policy_uri: str
    simulations: typing.Sequence[metta.sim.simulation_config.SimulationConfig]

    def invoke(self, args: dict[str, str]) -> int | None:
        stats_client = metta.app_backend.clients.stats_client.HttpStatsClient.create(self.stats_server_uri)
        normalized_uri = metta.rl.checkpoint_manager.CheckpointManager.normalize_uri(self.policy_uri)
        policy_id = metta.sim.utils.get_or_create_policy_ids(stats_client, [(normalized_uri, "")])[normalized_uri]

        git_hash = metta.common.util.git_helpers.get_task_commit_hash(skip_git_check=True)

        task = stats_client.create_task(
            metta.app_backend.routes.eval_task_routes.TaskCreateRequest(
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
