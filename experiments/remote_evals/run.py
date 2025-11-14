import base64
import json
from typing import Sequence

from pydantic import Field

from metta.app_backend.clients.stats_client import HttpStatsClient
from metta.common.tool.tool import ToolResult, ToolWithResult
from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.utils.auto_config import auto_replay_dir


class ExecuteRemoteEvalTool(ToolWithResult):
    simulations: Sequence[SimulationConfig]  # list of simulations to run
    policy_uri: str  # policy uri to evaluate
    eval_task_id: str
    replay_dir: str = Field(default_factory=auto_replay_dir)
    enable_replays: bool = True
    result_file_path: str  # path to the file where the results will be written

    group: str | None = None  # Separate group parameter like in train.py

    stats_server_uri: str | None = None  # If set, send stats to this http server
    push_metrics_to_wandb: bool = True

    def run_job(self) -> ToolResult:
        # Will error if stats_server_uri does not exist or we are not not authenticated with it
        stats_client = HttpStatsClient.create(self.stats_server_uri)

        eval_tool = EvaluateTool(
            simulations=self.simulations,
            policy_uris=[self.policy_uri],
            replay_dir=self.replay_dir,
            enable_replays=self.enable_replays,
            group=self.group,
            stats_server_uri=self.stats_server_uri,
            eval_task_id=self.eval_task_id,
            push_metrics_to_wandb=self.push_metrics_to_wandb,
        )
        normalized_uri = CheckpointManager.normalize_uri(self.policy_uri)

        eval_results = eval_tool.eval_policy(normalized_uri=normalized_uri, stats_client=stats_client)
        if len(eval_results.scores.simulation_scores) == 0:
            return ToolResult(result="failure", error="No simulations were run")
        elif len(eval_results.scores.simulation_scores) != len(self.simulations):
            # Find missing simulations
            missing_simulations = [
                sim for sim in self.simulations if sim.full_name not in eval_results.scores.simulation_scores
            ]
            return ToolResult(
                result="success",
                warnings=[f"Failed to run simulations: {missing_simulations}"],
            )

        return ToolResult(result="success")


# Used by eval_task_worker.py
def eval(
    policy_uri: str,
    simulations_json_base64_path: str,
    result_file_path: str,
    eval_task_id: str,
) -> ExecuteRemoteEvalTool:
    # Decode from base64 to avoid OmegaConf auto-parsing issues
    with open(simulations_json_base64_path, "rb") as f:
        simulations_json_base64 = f.read()
    simulations_json = base64.b64decode(simulations_json_base64).decode()
    simulations = [SimulationConfig.model_validate(sim) for sim in json.loads(simulations_json)]

    return ExecuteRemoteEvalTool(
        simulations=simulations,
        policy_uri=policy_uri,
        result_file_path=result_file_path,
        eval_task_id=eval_task_id,
    )
