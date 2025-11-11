import base64
from typing import Any, Sequence

from pydantic import Field

from metta.app_backend.clients.stats_client import HttpStatsClient
from metta.common.tool.tool import ToolResult, ToolWithResult
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.utils.auto_config import auto_replay_dir


class ExecuteRemoteEvalTool(ToolWithResult):
    simulations: Sequence[SimulationConfig]  # list of simulations to run
    policies: dict[str, float]  # policy uri to evaluate
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
            replay_dir=self.replay_dir,
            enable_replays=self.enable_replays,
            group=self.group,
            stats_server_uri=self.stats_server_uri,
            eval_task_id=self.eval_task_id,
            push_metrics_to_wandb=self.push_metrics_to_wandb,
        )

        eval_results = eval_tool.eval_policy(
            normalized_uri=normalized_uri, stats_client=stats_client
        )
        if len(eval_results.scores.simulation_scores) == 0:
            return ToolResult(result="failure", error="No simulations were run")
        elif len(eval_results.scores.simulation_scores) != len(self.simulations):
            # Find missing simulations
            missing_simulations = [
                sim
                for sim in self.simulations
                if sim.full_name not in eval_results.scores.simulation_scores
            ]
            return ToolResult(
                result="success",
                warnings=[f"Failed to run simulations: {missing_simulations}"],
            )

        return ToolResult(result="success")


def _read_b64_encoded_json_from_file(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        simulations_json_base64 = f.read()
    return base64.b64decode(simulations_json_base64).decode()


# Used by eval_task_worker.py
def eval_single(
    policy_uri: str,
    simulations_json_base64_path: str,
    result_file_path: str,
    eval_task_id: str,
) -> ExecuteRemoteEvalTool:
    # Decode from base64 to avoid OmegaConf auto-parsing issues
    simulations = [
        SimulationConfig.model_validate(sim)
        for sim in _read_b64_encoded_json_from_file(simulations_json_base64_path)
    ]

    return ExecuteRemoteEvalTool(
        simulations=simulations,
        policies={policy_uri: 1.0},
        result_file_path=result_file_path,
        eval_task_id=eval_task_id,
    )


def eval_multi(
    policies_json_base64_path: str,
    simulations_json_base64_path: str,
    result_file_path: str,
    eval_task_id: str,
) -> ExecuteRemoteEvalTool:
    # Decode from base64 to avoid OmegaConf auto-parsing issues
    simulations = [
        SimulationConfig.model_validate(sim)
        for sim in _read_b64_encoded_json_from_file(simulations_json_base64_path)
    ]
    policies = _read_b64_encoded_json_from_file(policies_json_base64_path)

    return ExecuteRemoteEvalTool(
        simulations=simulations,
        policies=policies,
        result_file_path=result_file_path,
        eval_task_id=eval_task_id,
    )
